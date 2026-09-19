from __future__ import annotations

import json
import math
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .artifacts import build_medexperiment
from .data import CheXpertDataset, CheXpertRecord, records_to_arrays
from .labels import CHEXPERT_LABELS, PRIMARY_LABELS
from .losses import capped_inverse_prevalence
from .manifests import read_chexpert_split_manifest
from .metrics import add_patient_bootstrap_intervals, evaluate_multilabel
from .model import build_model
from .thresholds import select_thresholds
from .training import TrainingConfig, predict_loader, seed_everything, train_epoch


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _primary_macro(metrics: Mapping[str, Any], key: str) -> float | None:
    values = [metrics["per_class"][label][key] for label in PRIMARY_LABELS if metrics["per_class"][label][key] is not None]
    return float(np.mean(values)) if values else None


def _failure_cases(
    records: Sequence[CheXpertRecord],
    targets: np.ndarray,
    mask: np.ndarray,
    probabilities: np.ndarray,
    thresholds: Mapping[str, float],
    limit: int = 10,
) -> dict[str, Any]:
    failures: dict[str, Any] = {}
    for index, label in enumerate(CHEXPERT_LABELS):
        valid = mask[:, index].astype(bool)
        predicted = probabilities[:, index] >= thresholds.get(label, 0.5)
        false_positive = np.flatnonzero(valid & (targets[:, index] == 0) & predicted)
        false_negative = np.flatnonzero(valid & (targets[:, index] == 1) & ~predicted)
        false_positive = false_positive[np.argsort(-probabilities[false_positive, index])][:limit]
        false_negative = false_negative[np.argsort(probabilities[false_negative, index])][:limit]
        failures[label] = {
            "falsePositive": [{"studyId": records[item].study_id, "score": float(probabilities[item, index])} for item in false_positive],
            "falseNegative": [{"studyId": records[item].study_id, "score": float(probabilities[item, index])} for item in false_negative],
        }
    return failures


def run_training(
    split_manifest: str | Path,
    output_directory: str | Path,
    *,
    method: str,
    seed: int,
    pretrained: bool = True,
    config: TrainingConfig | None = None,
    device_name: str | None = None,
    bootstrap_resamples: int = 2000,
    formal: bool = True,
) -> dict[str, Any]:
    """Run one of the six formal experiments from a locked patient split."""

    if method not in {"baseline", "mixstyle"}:
        raise ValueError("method must be baseline or mixstyle")
    cfg = config or TrainingConfig()
    cfg.validate()
    if seed not in cfg.seeds:
        raise ValueError(f"seed must be one of the fixed formal seeds: {cfg.seeds}")
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("training requires PyTorch") from exc

    seed_everything(seed)
    records = read_chexpert_split_manifest(split_manifest)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    generator = torch.Generator().manual_seed(seed)
    loaders = {
        name: DataLoader(
            CheXpertDataset(values, cfg.image_size),
            batch_size=cfg.batch_size,
            shuffle=name == "train",
            num_workers=min(4, os.cpu_count() or 1),
            pin_memory=device.type == "cuda",
            generator=generator if name == "train" else None,
        )
        for name, values in records.items()
    }
    train_targets, train_mask, _ = records_to_arrays(records["train"])
    class_weights = capped_inverse_prevalence(
        torch.from_numpy(train_targets), torch.from_numpy(train_mask), cfg.max_class_weight
    ).to(device)
    model = build_model(mixstyle=method == "mixstyle", pretrained=pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=1.0e-4)
    best_score = -math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []
    checkpoint = output / "best.pt"
    for epoch in range(1, cfg.max_epochs + 1):
        loss = train_epoch(model, loaders["train"], optimizer, class_weights, device)
        validation_logits, validation_targets, validation_mask = predict_loader(model, loaders["validation"], device)
        validation_probabilities = 1.0 / (1.0 + np.exp(-validation_logits))
        validation_metrics = evaluate_multilabel(validation_targets, validation_probabilities, CHEXPERT_LABELS, validation_mask)
        score = _primary_macro(validation_metrics, "auroc")
        history.append({"epoch": epoch, "trainingLoss": loss, "primaryMacroAuroc": score})
        comparable = score if score is not None else -math.inf
        if comparable > best_score:
            best_score = comparable
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "method": method,
                    "seed": seed,
                    "formal": formal,
                    "labels": list(CHEXPERT_LABELS),
                    "config": asdict(cfg),
                },
                checkpoint,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= cfg.patience:
                break
    if not checkpoint.is_file():
        raise RuntimeError("validation set did not contain enough observed outcomes for model selection")
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["state_dict"])
    validation_logits, validation_targets, validation_mask = predict_loader(model, loaders["validation"], device)
    validation_probabilities = 1.0 / (1.0 + np.exp(-validation_logits))
    thresholds = select_thresholds(validation_targets, validation_probabilities, CHEXPERT_LABELS, 0.90, validation_mask)
    test_logits, test_targets, test_mask = predict_loader(model, loaders["test"], device)
    test_probabilities = 1.0 / (1.0 + np.exp(-test_logits))
    test_metrics = evaluate_multilabel(test_targets, test_probabilities, CHEXPERT_LABELS, test_mask, thresholds)
    _, _, patient_ids = records_to_arrays(records["test"])
    add_patient_bootstrap_intervals(
        test_metrics,
        test_targets,
        test_probabilities,
        test_mask,
        patient_ids,
        CHEXPERT_LABELS,
        thresholds,
        bootstrap_resamples,
        seed,
    )
    summary = {
        "method": method,
        "seed": seed,
        "formal": formal,
        "pretrained": pretrained,
        "device": str(device),
        "bestEpoch": best_epoch,
        "validationPrimaryMacroAuroc": None if best_score == -math.inf else best_score,
        "testPrimaryMacroAuroc": _primary_macro(test_metrics, "auroc"),
        "testPrimaryMacroAuprc": _primary_macro(test_metrics, "auprc"),
        "thresholds": thresholds,
        "history": history,
        "testMetrics": test_metrics,
    }
    _json(output / "result.json", summary)
    _json(output / "thresholds.json", thresholds)
    _json(output / "failure_cases.local.json", _failure_cases(records["test"], test_targets, test_mask, test_probabilities, thresholds))
    return summary


def aggregate_experiment(
    run_results: Sequence[Mapping[str, Any]],
    experiment_id: str,
    dataset_manifest_sha256: str,
    output_path: str | Path,
    *,
    require_formal: bool = True,
) -> Path:
    if not run_results:
        raise ValueError("at least one completed run is required")
    if require_formal and any(result.get("formal") is not True for result in run_results):
        raise ValueError("formal experiment bundles cannot include smoke-test results")
    methods = {str(result["method"]) for result in run_results}
    if not methods <= {"baseline", "mixstyle"}:
        raise ValueError("unknown experiment method")
    def values(key: str, method: str) -> list[float]:
        return [float(result[key]) for result in run_results if result["method"] == method and result.get(key) is not None]
    aggregate: dict[str, float | None] = {}
    for method in sorted(methods):
        for key in ("testPrimaryMacroAuroc", "testPrimaryMacroAuprc"):
            observations = values(key, method)
            aggregate[f"{method}_{key}_mean"] = float(np.mean(observations)) if observations else None
            aggregate[f"{method}_{key}_std"] = float(np.std(observations, ddof=1)) if len(observations) > 1 else None
    per_class: dict[str, dict[str, float | None]] = {}
    for label in CHEXPERT_LABELS:
        per_class[label] = {}
        for method in sorted(methods):
            for metric in ("auroc", "auprc", "sensitivity", "specificity", "f1", "brier"):
                observations = [result["testMetrics"]["per_class"][label][metric] for result in run_results if result["method"] == method]
                numeric = [float(value) for value in observations if value is not None]
                per_class[label][f"{method}_{metric}_mean"] = float(np.mean(numeric)) if numeric else None
    bundle = {
        "schemaVersion": "medexperiment-1",
        "experimentId": experiment_id,
        "config": {"methods": sorted(methods), "formal": require_formal, "formalRunLimit": 6, "uncertainPolicy": "masked", "thresholdTargetSensitivity": 0.90},
        "seeds": sorted({int(result["seed"]) for result in run_results}),
        "datasetManifestSha256": dataset_manifest_sha256,
        "aggregateMetrics": aggregate,
        "perClassMetrics": per_class,
        "curves": {},
        "modelCard": {"intendedUse": "adult AP/PA chest radiograph research prototype", "limitations": ["not a medical device", "not an independent diagnosis", "external MIMIC evaluation must remain sealed until final evaluation"]},
    }
    return build_medexperiment(bundle, output_path)
