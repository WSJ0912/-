"""Training loop for the ``cxr_first`` competition dataset.

Only the organizer-provided training split is used for fitting and the
organizer-provided validation split for early stopping, checkpoint selection, and
F1 threshold tuning. The test split is never read here.
"""

from __future__ import annotations

import json
import random
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .data import CompetitionDataset, load_split, records_to_arrays
from .labels import LABEL_IDS
from .metrics import macro_auc, macro_f1, per_class_auc, tune_thresholds
from .model import (
    build_model,
    load_model_from_checkpoint,
    pretrained_weight_metadata,
    save_checkpoint,
)


@dataclass(frozen=True)
class TrainConfig:
    data_root: Path
    output_dir: Path
    image_size: int = 384
    epochs: int = 8
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    seed: int = 17
    device: str = "cuda"
    pretrained: bool = True
    num_workers: int = 4
    patience: int = 3
    max_train_images: int | None = None
    max_valid_images: int | None = None


def _set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(name: str) -> Any:
    import torch

    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(
            "CUDA was requested but is not available; use --device cpu for a smoke test "
            "and rent a GPU for formal training"
        )
    return device


def _git_commit() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return completed.stdout.strip()
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - environment dependent
        return "unknown"


def _class_pos_weight(targets: np.ndarray) -> np.ndarray:
    positives = targets.sum(axis=0)
    negatives = targets.shape[0] - positives
    weight = negatives / np.maximum(positives, 1.0)
    return np.clip(weight, 1.0, 10.0).astype(np.float32)


def _loader(dataset: Any, *, batch_size: int, shuffle: bool, num_workers: int, device: Any) -> Any:
    from torch.utils.data import DataLoader

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )


def _evaluate(model: Any, loader: Any, device: Any) -> np.ndarray:
    import torch

    model.eval()
    scores: list[np.ndarray] = []
    with torch.no_grad():
        for images, _targets in loader:
            logits = model(images.to(device, non_blocking=True))
            scores.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(scores, axis=0)


def train(config: TrainConfig) -> dict[str, Any]:
    """Run supervised fine-tuning and write checkpoints plus a result manifest."""

    import torch
    from torch import nn

    _set_seed(config.seed)
    device = _resolve_device(config.device)
    root = Path(config.data_root)
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    train_records = load_split(root / "train.csv", root / "train")
    valid_records = load_split(root / "valid.csv", root / "valid")
    if config.max_train_images is not None:
        train_records = train_records[: config.max_train_images]
    if config.max_valid_images is not None:
        valid_records = valid_records[: config.max_valid_images]
    if not train_records or not valid_records:
        raise ValueError("training and validation splits must not be empty")

    train_targets, _ = records_to_arrays(train_records)
    valid_targets, _ = records_to_arrays(valid_records)
    pos_weight = torch.tensor(_class_pos_weight(train_targets), device=device)

    model = build_model(len(LABEL_IDS), pretrained=config.pretrained).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(config.epochs, 1))

    train_loader = _loader(
        CompetitionDataset(train_records, image_size=config.image_size, training=True),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        device=device,
    )
    valid_loader = _loader(
        CompetitionDataset(valid_records, image_size=config.image_size, training=False),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        device=device,
    )

    checkpoint_config: dict[str, Any] = {
        "imageSize": config.image_size,
        "labelIds": list(LABEL_IDS),
        "trainingCommit": _git_commit(),
        "pretrained": pretrained_weight_metadata() if config.pretrained else None,
        "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "architecture": "densenet121",
        "validationUse": "early stopping, checkpoint selection, F1 threshold tuning",
    }

    history: list[dict[str, Any]] = []
    best_auc = -1.0
    best_epoch = -1
    stale_epochs = 0
    best_checkpoint: dict[str, Any] | None = None
    for epoch in range(1, config.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * images.size(0)
            seen += images.size(0)
        scheduler.step()
        valid_scores = _evaluate(model, valid_loader, device)
        valid_auc = macro_auc(valid_targets, valid_scores)
        history.append(
            {
                "epoch": epoch,
                "trainLoss": running_loss / max(seen, 1),
                "validMacroAuc": valid_auc,
                "learningRate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        print(
            f"epoch {epoch}/{config.epochs} loss={history[-1]['trainLoss']:.4f} "
            f"validMacroAuc={valid_auc}"
        )
        if valid_auc is not None and valid_auc > best_auc + 1e-6:
            best_auc = valid_auc
            best_epoch = epoch
            stale_epochs = 0
            best_checkpoint = save_checkpoint(
                output / "best.pt",
                model=model,
                config=checkpoint_config,
                label_ids=LABEL_IDS,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                print(f"early stopping after {epoch} epochs without improvement")
                break

    if best_checkpoint is None:
        raise RuntimeError("training finished without any computable validation AUROC")

    best_model, saved_config = load_model_from_checkpoint(output / "best.pt")
    best_model.to(device)
    best_loader = _loader(
        CompetitionDataset(valid_records, image_size=config.image_size, training=False),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        device=device,
    )
    _evaluate(best_model, best_loader, device)  # warm-up to keep CUDA kernels ready
    valid_scores = _evaluate(best_model, best_loader, device)
    thresholds = tune_thresholds(valid_targets, valid_scores)
    threshold_f1 = macro_f1(valid_targets, valid_scores, thresholds)

    result: dict[str, Any] = {
        "schemaVersion": "cxr-first-result-v1",
        "dataset": "cxr_first",
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
        "labelIds": list(LABEL_IDS),
        "bestEpoch": best_epoch,
        "validMacroAuc": best_auc,
        "validMacroF1": threshold_f1,
        "thresholds": {str(label_id): value for label_id, value in thresholds.items()},
        "perClassAuc": {str(key): value for key, value in per_class_auc(valid_targets, valid_scores).items()},
        "history": history,
        "checkpoint": {
            "path": str(output / "best.pt"),
            "sha256": best_checkpoint["checkpoint_sha256"],
            "bytes": best_checkpoint["checkpoint_bytes"],
        },
        "inferenceConfig": saved_config,
    }
    (output / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result
