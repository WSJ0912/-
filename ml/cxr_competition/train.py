"""Training loop for the ``cxr_first`` competition dataset.

Only the organizer-provided training split is used for fitting and the
organizer-provided validation split for early stopping, checkpoint selection, and
F1 threshold tuning. The test split is never read here.

Mixed precision (``amp=True``) and resumable training (``resume=last.pt``) keep
rented-GPU sessions short; the cache builder keeps JPEG decoding off the GPU
timeline.
"""

from __future__ import annotations

import json
import random
import subprocess
import time
from contextlib import nullcontext
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

TRAINING_STATE_FORMAT = "cxr-first-train-state-v1"


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
    amp: bool = False
    resume: Path | None = None
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


def _loader(
    dataset: Any,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: Any,
) -> Any:
    from torch.utils.data import DataLoader

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )


def _make_autocast(use_amp: bool) -> Any:
    """Return a CUDA autocast context; empty on CPU or when AMP is off."""

    if not use_amp:
        return nullcontext()
    import torch

    try:
        return torch.amp.autocast("cuda", enabled=True)
    except (AttributeError, TypeError):  # pragma: no cover - older torch
        return torch.cuda.amp.autocast(enabled=True)


def _make_scaler(use_amp: bool) -> Any:
    if not use_amp:
        return None
    import torch

    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except (AttributeError, TypeError):  # pragma: no cover - older torch
        return torch.cuda.amp.GradScaler(enabled=True)


def _evaluate(model: Any, loader: Any, device: Any, use_amp: bool = False) -> np.ndarray:
    import torch

    model.eval()
    scores: list[np.ndarray] = []
    with torch.no_grad(), _make_autocast(use_amp):
        for images, _targets in loader:
            logits = model(images.to(device, non_blocking=True))
            scores.append(torch.sigmoid(logits).float().cpu().numpy())
    return np.concatenate(scores, axis=0)


def _save_training_state(
    path: Path,
    *,
    epoch: int,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    history: list[dict[str, Any]],
    best_auc: float,
    best_epoch: int,
    stale_epochs: int,
) -> None:
    import torch

    payload = {
        "format": TRAINING_STATE_FORMAT,
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "history": history,
        "bestAuc": best_auc,
        "bestEpoch": best_epoch,
        "staleEpochs": stale_epochs,
    }
    torch.save(payload, path)


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
        "amp": bool(config.amp),
    }

    use_amp = bool(config.amp and device.type == "cuda")
    scaler = _make_scaler(use_amp)

    history: list[dict[str, Any]] = []
    best_auc = -1.0
    best_epoch = -1
    stale_epochs = 0
    start_epoch = 1
    best_checkpoint: dict[str, Any] | None = None
    if config.resume is not None:
        payload = torch.load(Path(config.resume), map_location="cpu", weights_only=False)
        if payload.get("format") != TRAINING_STATE_FORMAT:
            raise ValueError(f"unsupported training state in {config.resume}")
        model.load_state_dict(payload["model_state"])
        model.to(device)
        optimizer.load_state_dict(payload["optimizer_state"])
        scheduler.load_state_dict(payload["scheduler_state"])
        history = list(payload.get("history", []))
        best_auc = float(payload.get("bestAuc", -1.0))
        best_epoch = int(payload.get("bestEpoch", -1))
        stale_epochs = int(payload.get("staleEpochs", 0))
        start_epoch = int(payload["epoch"]) + 1
        print(f"resuming from {config.resume} at epoch {start_epoch} (best AUC {best_auc:.4f})")
        if (output / "best.pt").is_file():
            best_checkpoint = {"checkpoint_sha256": "", "checkpoint_bytes": 0}

    training_started = time.perf_counter()
    for epoch in range(start_epoch, config.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        epoch_started = time.perf_counter()
        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with _make_autocast(use_amp):
                logits = model(images)
                loss = criterion(logits, targets)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            running_loss += float(loss.item()) * images.size(0)
            seen += images.size(0)
        scheduler.step()
        valid_scores = _evaluate(model, valid_loader, device, use_amp)
        valid_auc = macro_auc(valid_targets, valid_scores)
        history.append(
            {
                "epoch": epoch,
                "trainLoss": running_loss / max(seen, 1),
                "validMacroAuc": valid_auc,
                "learningRate": float(optimizer.param_groups[0]["lr"]),
                "epochSeconds": round(time.perf_counter() - epoch_started, 1),
            }
        )
        print(
            f"epoch {epoch}/{config.epochs} loss={history[-1]['trainLoss']:.4f} "
            f"validMacroAuc={valid_auc} ({history[-1]['epochSeconds']}s)"
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
        _save_training_state(
            output / "last.pt",
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            history=history,
            best_auc=best_auc,
            best_epoch=best_epoch,
            stale_epochs=stale_epochs,
        )
        if stale_epochs >= config.patience:
            print(f"early stopping after {epoch} epochs without improvement")
            break

    if best_checkpoint is None or not (output / "best.pt").is_file():
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
    _evaluate(best_model, best_loader, device, use_amp)  # warm-up to keep CUDA kernels ready
    valid_scores = _evaluate(best_model, best_loader, device, use_amp)
    thresholds = tune_thresholds(valid_targets, valid_scores)
    threshold_f1 = macro_f1(valid_targets, valid_scores, thresholds)

    if best_checkpoint.get("checkpoint_sha256"):
        checkpoint_sha = best_checkpoint["checkpoint_sha256"]
        checkpoint_bytes = best_checkpoint["checkpoint_bytes"]
    else:  # resumed run kept the existing best.pt
        checkpoint_sha = ""
        checkpoint_bytes = (output / "best.pt").stat().st_size

    result: dict[str, Any] = {
        "schemaVersion": "cxr-first-result-v1",
        "dataset": "cxr_first",
        "config": {
            key: str(value) if isinstance(value, (Path,)) else value
            for key, value in asdict(config).items()
        },
        "labelIds": list(LABEL_IDS),
        "bestEpoch": best_epoch,
        "validMacroAuc": best_auc,
        "validMacroF1": threshold_f1,
        "thresholds": {str(label_id): value for label_id, value in thresholds.items()},
        "perClassAuc": {str(key): value for key, value in per_class_auc(valid_targets, valid_scores).items()},
        "history": history,
        "amp": use_amp,
        "trainingSeconds": round(time.perf_counter() - training_started, 1),
        "checkpoint": {
            "path": str(output / "best.pt"),
            "sha256": checkpoint_sha,
            "bytes": checkpoint_bytes,
        },
        "inferenceConfig": saved_config,
    }
    (output / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result
