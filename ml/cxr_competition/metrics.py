"""Evaluation metrics for the competition: macro AUROC and macro F1.

The organizer ranks teams by macro-averaged AUC over the ten class ids and uses
macro-averaged F1 only as a tie breaker. Thresholds for F1 are therefore fixed
on the validation split and never tuned on the test split.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


def _resolve_label_ids(
    width: int, label_ids: Sequence[int] | None
) -> tuple[int, ...]:
    """Return explicit label ids or derive ``1..width`` for compact unit tests."""

    if label_ids is None:
        return tuple(range(1, width + 1))
    resolved = tuple(int(item) for item in label_ids)
    if len(resolved) != width:
        raise ValueError(f"label ids {list(resolved)} do not match score width {width}")
    return resolved


def per_class_auc(
    targets: np.ndarray,
    scores: np.ndarray,
    label_ids: Sequence[int] | None = None,
) -> dict[int, float | None]:
    """Return one AUROC per class id, ``None`` when the class is single-valued."""

    target_array = np.asarray(targets, dtype=np.float64)
    score_array = np.asarray(scores, dtype=np.float64)
    if target_array.shape != score_array.shape:
        raise ValueError("targets and scores must share the same shape")
    resolved = _resolve_label_ids(target_array.shape[1], label_ids)
    result: dict[int, float | None] = {}
    for index, label_id in enumerate(resolved):
        truth = target_array[:, index]
        if truth.min() == truth.max():
            result[label_id] = None
            continue
        result[label_id] = float(roc_auc_score(truth, score_array[:, index]))
    return result


def macro_auc(
    targets: np.ndarray,
    scores: np.ndarray,
    label_ids: Sequence[int] | None = None,
) -> float | None:
    """Mean of the per-class AUROCs; ``None`` when no class is computable."""

    values = [
        value for value in per_class_auc(targets, scores, label_ids).values() if value is not None
    ]
    return float(np.mean(values)) if values else None


def per_class_f1(
    targets: np.ndarray,
    scores: np.ndarray,
    thresholds: Mapping[int, float],
    label_ids: Sequence[int] | None = None,
) -> dict[int, float | None]:
    """Return one F1 score per class id at the supplied fixed thresholds."""

    target_array = np.asarray(targets, dtype=np.int64)
    score_array = np.asarray(scores, dtype=np.float64)
    if target_array.shape != score_array.shape:
        raise ValueError("targets and scores must share the same shape")
    resolved = _resolve_label_ids(target_array.shape[1], label_ids)
    result: dict[int, float | None] = {}
    for index, label_id in enumerate(resolved):
        truth = target_array[:, index]
        if truth.min() == truth.max():
            result[label_id] = None
            continue
        predicted = (score_array[:, index] >= float(thresholds[label_id])).astype(np.int64)
        result[label_id] = float(f1_score(truth, predicted, zero_division=0))
    return result


def macro_f1(
    targets: np.ndarray,
    scores: np.ndarray,
    thresholds: Mapping[int, float],
    label_ids: Sequence[int] | None = None,
) -> float | None:
    values = [
        value
        for value in per_class_f1(targets, scores, thresholds, label_ids).values()
        if value is not None
    ]
    return float(np.mean(values)) if values else None


def tune_thresholds(
    targets: np.ndarray,
    scores: np.ndarray,
    label_ids: Sequence[int] | None = None,
    grid: Sequence[float] | None = None,
) -> dict[int, float]:
    """Choose one F1-maximising threshold per class on the validation split."""

    target_array = np.asarray(targets, dtype=np.int64)
    score_array = np.asarray(scores, dtype=np.float64)
    if target_array.shape != score_array.shape:
        raise ValueError("targets and scores must share the same shape")
    resolved = _resolve_label_ids(target_array.shape[1], label_ids)
    candidates = list(grid) if grid is not None else [round(0.02 * step, 2) for step in range(1, 50)]
    thresholds: dict[int, float] = {}
    for index, label_id in enumerate(resolved):
        truth = target_array[:, index]
        best_threshold = 0.5
        best_score = -1.0
        for candidate in candidates:
            predicted = (score_array[:, index] >= float(candidate)).astype(np.int64)
            value = float(f1_score(truth, predicted, zero_division=0))
            if value > best_score:
                best_score = value
                best_threshold = float(candidate)
        thresholds[label_id] = best_threshold
    return thresholds


def aggregate_by_study(
    study_ids: Sequence[str],
    scores: np.ndarray,
    *,
    method: str = "max",
) -> tuple[list[str], np.ndarray]:
    """Aggregate image scores to study level.

    ``method`` is ``"max"`` (any positive image wins), ``"mean"`` or ``"none"``.
    Study order follows first appearance in ``study_ids``.
    """

    score_array = np.asarray(scores, dtype=np.float64)
    if len(study_ids) != len(score_array):
        raise ValueError("study_ids and scores must have the same length")
    if method == "none":
        return list(study_ids), score_array
    if method not in {"max", "mean"}:
        raise ValueError("method must be one of 'none', 'max', 'mean'")
    order: list[str] = []
    grouped: dict[str, list[np.ndarray]] = {}
    for study_id, row in zip(study_ids, score_array, strict=True):
        if study_id not in grouped:
            order.append(study_id)
            grouped[study_id] = []
        grouped[study_id].append(row)
    reducer = np.max if method == "max" else np.mean
    aggregated = np.stack([reducer(np.stack(grouped[study_id]), axis=0) for study_id in order])
    return order, aggregated


def study_level_targets(
    study_ids: Sequence[str],
    image_targets: np.ndarray,
    *,
    method: str = "max",
) -> np.ndarray:
    """Aggregate image targets to study level (``max`` keeps any-positive)."""

    _order, aggregated = aggregate_by_study(study_ids, image_targets, method=method)
    return aggregated
