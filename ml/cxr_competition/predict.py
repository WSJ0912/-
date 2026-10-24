"""Inference, validation metrics, and submission writing for ``cxr_first``.

Competition rule 6 requires deterministic per-image preprocessing and a fixed,
frozen model at test time. TTA and ensembles are allowed but kept off by default
so ``predict.py`` and ``cost.py`` share one identical code path.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .data import TestImageDataset, load_split, records_to_arrays
from .labels import LABEL_IDS
from .metrics import aggregate_by_study, macro_auc, macro_f1, per_class_auc, tune_thresholds
from .model import load_model_from_checkpoint

SUPPORTED_AGGREGATIONS = ("none", "max", "mean")
SUPPORTED_NORMALIZATIONS = ("none", "l1")


def normalize_probabilities(scores: np.ndarray, method: str = "l1") -> np.ndarray:
    """Normalize per-row probabilities.

    ``l1`` divides every row by its sum so each submitted study row is a
    normalized probability vector (the organizer asks for normalized
    probabilities); ``none`` keeps the aggregated values untouched.
    """

    score_array = np.asarray(scores, dtype=np.float64)
    if method == "none":
        return score_array
    if method != "l1":
        raise ValueError(f"normalize must be one of {SUPPORTED_NORMALIZATIONS}")
    sums = score_array.sum(axis=1, keepdims=True)
    safe_sums = np.where(sums > 0.0, sums, 1.0)
    return score_array / safe_sums


def _resolve_device(name: str) -> Any:
    import torch

    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return device


def collect_scores(
    *,
    checkpoint_path: str | Path,
    records: Sequence[Any],
    batch_size: int,
    device_name: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run the frozen model over unlabelled image records."""

    import torch
    from torch.utils.data import DataLoader

    model, config = load_model_from_checkpoint(checkpoint_path)
    device = _resolve_device(device_name)
    model.to(device)
    model.eval()
    image_size = int(config["imageSize"])
    loader = DataLoader(
        TestImageDataset(records, image_size=image_size),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    batches: list[np.ndarray] = []
    with torch.no_grad():
        for images, _dicom_ids in loader:
            logits = model(images.to(device, non_blocking=True))
            batches.append(torch.sigmoid(logits).cpu().numpy())
    if not batches:  # pragma: no cover - defensive
        raise ValueError("no images to score")
    return np.concatenate(batches, axis=0), config


def write_submission(
    path: str | Path,
    study_ids: Sequence[str],
    scores: np.ndarray,
    *,
    label_ids: Sequence[int] = LABEL_IDS,
    gate: float | None = None,
) -> Path:
    """Write the organizer's long-format submission CSV.

    Columns are ``Study_id, Predict_class, Probability, Gate`` with ten rows per
    study (class ids 1-10). ``Gate`` is left blank to use the default 0.5
    threshold; pass ``gate`` to write a custom threshold on every row.
    """

    destinations = Path(path)
    destinations.parent.mkdir(parents=True, exist_ok=True)
    score_array = np.asarray(scores, dtype=np.float64)
    if score_array.shape != (len(study_ids), len(label_ids)):
        raise ValueError("scores must be [N, 10] and align with study ids")
    if gate is not None and not 0.0 < float(gate) < 1.0:
        raise ValueError("gate threshold must be inside (0, 1)")
    gate_text = "" if gate is None else f"{float(gate):.6f}"
    with destinations.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Study_id", "Predict_class", "Probability", "Gate"])
        for study_id, row in zip(study_ids, score_array, strict=True):
            for label_id, value in zip(label_ids, row, strict=True):
                writer.writerow([study_id, label_id, f"{float(value):.6f}", gate_text])
    return destinations


def predict(
    *,
    checkpoint_path: str | Path,
    data_root: str | Path,
    split: str,
    output_csv: str | Path,
    batch_size: int = 8,
    device: str = "cuda",
    aggregate: str = "mean",
    normalize: str = "l1",
    gate: float | None = None,
) -> dict[str, Any]:
    """Score one split and write a study-level submission CSV.

    Default behaviour matches the organizer request: images of one study are
    aggregated with the mean of their sigmoid probabilities, every study row is
    L1-normalized, and the CSV is written in the official long format
    (``Study_id, Predict_class, Probability, Gate``). ``aggregate="none"``
    writes one row per image and requires ``normalize="none"``.
    """

    if aggregate not in SUPPORTED_AGGREGATIONS:
        raise ValueError(f"aggregate must be one of {SUPPORTED_AGGREGATIONS}")
    if normalize not in SUPPORTED_NORMALIZATIONS:
        raise ValueError(f"normalize must be one of {SUPPORTED_NORMALIZATIONS}")
    if aggregate == "none" and normalize != "none":
        raise ValueError("per-image submissions must not be row-normalized")
    root = Path(data_root)
    labelled = split != "test"
    records = load_split(root / f"{split}.csv", root / split, labelled=labelled)
    scores, _config = collect_scores(
        checkpoint_path=checkpoint_path,
        records=records,
        batch_size=batch_size,
        device_name=device,
    )
    summary: dict[str, Any] = {
        "split": split,
        "images": len(records),
        "studies": len({record.study_id for record in records}),
        "aggregate": aggregate,
        "normalize": normalize,
        "gate": gate,
    }
    if aggregate == "none":
        ids = [record.dicom_id for record in records]
    else:
        ids, scores = aggregate_by_study(
            [record.study_id for record in records], scores, method=aggregate
        )
        scores = normalize_probabilities(scores, normalize)
    write_submission(output_csv, ids, scores, gate=gate)
    summary.update({"rows": len(ids) * len(LABEL_IDS), "output": str(output_csv)})
    return summary


def evaluate(
    *,
    checkpoint_path: str | Path,
    data_root: str | Path,
    output_json: str | Path | None = None,
    batch_size: int = 8,
    device: str = "cpu",
) -> dict[str, Any]:
    """Validation-split metrics: macro AUROC, tuned macro F1, and per-class AUC."""

    root = Path(data_root)
    records = load_split(root / "valid.csv", root / "valid")
    targets, _mask = records_to_arrays(records)
    scores, _config = collect_scores(
        checkpoint_path=checkpoint_path,
        records=records,
        batch_size=batch_size,
        device_name=device,
    )
    thresholds = tune_thresholds(targets, scores)
    result: dict[str, Any] = {
        "schemaVersion": "cxr-first-evaluation-v1",
        "split": "valid",
        "images": len(records),
        "macroAuc": macro_auc(targets, scores),
        "macroF1": macro_f1(targets, scores, thresholds),
        "thresholds": {str(label_id): value for label_id, value in thresholds.items()},
        "perClassAuc": {str(key): value for key, value in per_class_auc(targets, scores).items()},
    }
    if output_json is not None:
        destination = Path(output_json)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return result


def submission_summary(path: str | Path) -> Mapping[str, Any]:
    """Sanity-check a written submission file (official long format)."""

    source = Path(path)
    with source.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    if len(rows) < 2:
        raise ValueError(f"{source} has no data rows")
    header = rows[0]
    if len(header) not in (3, 4) or header[:3] != ["Study_id", "Predict_class", "Probability"]:
        raise ValueError(f"{source} header must be Study_id,Predict_class,Probability[,Gate]")
    data = rows[1:]
    if len(data) % len(LABEL_IDS) != 0:
        raise ValueError(f"{source} must contain ten rows (classes) per Study_id")
    studies = len({row[0] for row in data})
    if len(data) != studies * len(LABEL_IDS):
        raise ValueError(f"{source} has an uneven number of class rows per Study_id")
    for row in data:
        value = float(row[2])
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{source} probability outside [0, 1]: {value}")
    return {"path": str(source), "rows": len(data), "studies": studies, "columns": header}
