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
from .labels import LABEL_IDS, label_column
from .metrics import aggregate_by_study, macro_auc, macro_f1, per_class_auc, tune_thresholds
from .model import load_model_from_checkpoint

SUPPORTED_AGGREGATIONS = ("none", "max", "mean")


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
    ids: Sequence[str],
    scores: np.ndarray,
    *,
    id_column: str = "Dicom_id",
    label_ids: Sequence[int] = LABEL_IDS,
) -> Path:
    """Write ``<id column>,class_01,...,class_10`` probability rows."""

    destinations = Path(path)
    destinations.parent.mkdir(parents=True, exist_ok=True)
    score_array = np.asarray(scores, dtype=np.float64)
    if score_array.shape != (len(ids), len(label_ids)):
        raise ValueError("scores must be [N, 10] and align with ids")
    with destinations.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([id_column, *[label_column(label_id) for label_id in label_ids]])
        for row_id, row in zip(ids, score_array, strict=True):
            writer.writerow([row_id, *[f"{value:.6f}" for value in row]])
    return destinations


def predict(
    *,
    checkpoint_path: str | Path,
    data_root: str | Path,
    split: str,
    output_csv: str | Path,
    batch_size: int = 8,
    device: str = "cuda",
    aggregate: str = "none",
) -> dict[str, Any]:
    """Score one split and write a submission CSV.

    ``aggregate="none"`` writes one row per image (``Dicom_id``); ``"max"`` or
    ``"mean"`` aggregates images to one row per study (``Study_id``).
    """

    if aggregate not in SUPPORTED_AGGREGATIONS:
        raise ValueError(f"aggregate must be one of {SUPPORTED_AGGREGATIONS}")
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
    }
    if aggregate == "none":
        ids = [record.dicom_id for record in records]
        id_column = "Dicom_id"
    else:
        ids, scores = aggregate_by_study(
            [record.study_id for record in records], scores, method=aggregate
        )
        id_column = "Study_id"
    write_submission(output_csv, ids, scores, id_column=id_column)
    summary.update({"rows": len(ids), "idColumn": id_column, "output": str(output_csv)})
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
    """Sanity-check a written submission file."""

    source = Path(path)
    with source.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    if len(rows) < 2:
        raise ValueError(f"{source} has no data rows")
    header = rows[0]
    if len(header) != len(LABEL_IDS) + 1:
        raise ValueError(f"{source} header must contain one id column and 10 class columns")
    return {"path": str(source), "rows": len(rows) - 1, "columns": header}
