from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np


def _valid_arrays(y_true: Any, scores: Any, mask: Any | None = None) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=float).reshape(-1)
    s = np.asarray(scores, dtype=float).reshape(-1)
    valid = np.isfinite(y) & np.isfinite(s)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool).reshape(-1)
    return y[valid], s[valid]


def binary_auroc(y_true: Any, scores: Any, mask: Any | None = None) -> float | None:
    y, s = _valid_arrays(y_true, scores, mask)
    positives = y == 1
    negatives = y == 0
    n_pos, n_neg = int(positives.sum()), int(negatives.sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    sorted_scores = s[order]
    ranks = np.empty(len(s), dtype=float)
    start = 0
    while start < len(s):
        end = start + 1
        while end < len(s) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    positive_rank_sum = ranks[positives].sum()
    return float((positive_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def binary_auprc(y_true: Any, scores: Any, mask: Any | None = None) -> float | None:
    y, s = _valid_arrays(y_true, scores, mask)
    n_pos = int((y == 1).sum())
    if n_pos == 0:
        return None
    order = np.argsort(-s, kind="mergesort")
    sorted_y = (y[order] == 1).astype(float)
    sorted_scores = s[order]
    last_for_score = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(sorted_scores) - 1]
    true_positives = np.cumsum(sorted_y)[last_for_score]
    predicted_positives = last_for_score + 1
    precision = true_positives / predicted_positives
    recall = true_positives / n_pos
    recall_delta = np.diff(np.r_[0.0, recall])
    return float(np.sum(recall_delta * precision))


def binary_curve_points(
    y_true: Any,
    scores: Any,
    mask: Any | None = None,
) -> dict[str, list[dict[str, float | None]]]:
    """Return aggregate ROC and precision-recall points without sample records."""

    y, s = _valid_arrays(y_true, scores, mask)
    positives = y == 1
    negatives = y == 0
    n_pos, n_neg = int(positives.sum()), int(negatives.sum())
    if n_pos == 0 or n_neg == 0:
        return {"roc": [], "pr": []}

    order = np.argsort(-s, kind="mergesort")
    sorted_scores = s[order]
    sorted_positive = positives[order].astype(int)
    last_for_score = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(sorted_scores) - 1]
    true_positives = np.cumsum(sorted_positive)[last_for_score]
    false_positives = (last_for_score + 1) - true_positives
    thresholds = sorted_scores[last_for_score]

    roc: list[dict[str, float | None]] = [{"x": 0.0, "y": 0.0, "threshold": None}]
    precision_recall: list[dict[str, float | None]] = [
        {"x": 0.0, "y": 1.0, "threshold": None}
    ]
    for true_positive, false_positive, threshold in zip(
        true_positives, false_positives, thresholds, strict=True
    ):
        recall = float(true_positive / n_pos)
        roc.append(
            {
                "x": float(false_positive / n_neg),
                "y": recall,
                "threshold": float(threshold),
            }
        )
        precision_recall.append(
            {
                "x": recall,
                "y": float(true_positive / (true_positive + false_positive)),
                "threshold": float(threshold),
            }
        )
    return {"roc": roc, "pr": precision_recall}


def multilabel_curve_points(
    y_true: Any,
    scores: Any,
    labels: Sequence[str],
    mask: Any | None = None,
) -> dict[str, dict[str, list[dict[str, float | None]]]]:
    """Build per-label aggregate curves; no prediction or patient rows are returned."""

    y = np.asarray(y_true)
    p = np.asarray(scores)
    m = np.asarray(mask) if mask is not None else np.ones_like(y, dtype=bool)
    if y.shape != p.shape or y.shape != m.shape or y.ndim != 2 or y.shape[1] != len(labels):
        raise ValueError("y_true, scores, mask must be [N, number_of_labels]")
    return {
        label: binary_curve_points(y[:, index], p[:, index], m[:, index])
        for index, label in enumerate(labels)
    }


def confusion_at_threshold(
    y_true: Any, scores: Any, threshold: float, mask: Any | None = None
) -> dict[str, float | int | None]:
    y, s = _valid_arrays(y_true, scores, mask)
    if len(y) == 0:
        return {"sensitivity": None, "specificity": None, "f1": None, "tp": 0, "tn": 0, "fp": 0, "fn": 0}
    predicted = s >= threshold
    actual = y == 1
    tp = int(np.sum(predicted & actual))
    tn = int(np.sum(~predicted & ~actual))
    fp = int(np.sum(predicted & ~actual))
    fn = int(np.sum(~predicted & actual))
    sensitivity = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if sensitivity is not None and precision + sensitivity else 0.0
    return {"sensitivity": sensitivity, "specificity": specificity, "f1": f1, "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def brier_score(y_true: Any, scores: Any, mask: Any | None = None) -> float | None:
    y, s = _valid_arrays(y_true, scores, mask)
    return float(np.mean((s - y) ** 2)) if len(y) else None


def bootstrap_ci(
    metric: Callable[[np.ndarray, np.ndarray, np.ndarray | None], float | None],
    y_true: Any,
    scores: Any,
    mask: Any | None = None,
    patient_ids: Sequence[str] | None = None,
    n_resamples: int = 2000,
    seed: int = 20240913,
) -> dict[str, float | None]:
    """Patient-level percentile bootstrap; never resamples individual views."""

    y = np.asarray(y_true)
    s = np.asarray(scores)
    m = np.asarray(mask) if mask is not None else None
    if y.shape[0] != s.shape[0]:
        raise ValueError("y_true and scores must have equal first dimension")
    ids = np.asarray(patient_ids if patient_ids is not None else np.arange(y.shape[0]), dtype=str)
    if ids.shape[0] != y.shape[0]:
        raise ValueError("patient_ids length must match first dimension")
    unique = np.unique(ids)
    point = metric(y, s, m)
    if len(unique) == 0 or point is None:
        return {"estimate": point, "lower": None, "upper": None, "n": 0}
    rng = np.random.default_rng(seed)
    values: list[float] = []
    groups = {patient: np.flatnonzero(ids == patient) for patient in unique}
    for _ in range(n_resamples):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indexes = np.concatenate([groups[patient] for patient in sampled])
        value = metric(y[indexes], s[indexes], m[indexes] if m is not None else None)
        if value is not None and np.isfinite(value):
            values.append(float(value))
    if not values:
        return {"estimate": point, "lower": None, "upper": None, "n": 0}
    lower, upper = np.percentile(values, [2.5, 97.5])
    return {"estimate": point, "lower": float(lower), "upper": float(upper), "n": len(values)}


def add_patient_bootstrap_intervals(
    result: dict[str, Any],
    y_true: Any,
    scores: Any,
    mask: Any,
    patient_ids: Sequence[str],
    labels: Sequence[str],
    thresholds: dict[str, float],
    n_resamples: int = 2000,
    seed: int = 20240913,
) -> dict[str, Any]:
    """Attach patient-level 95% CIs for every required per-class metric."""

    y = np.asarray(y_true)
    p = np.asarray(scores)
    m = np.asarray(mask)
    if y.shape != p.shape or y.shape != m.shape or y.ndim != 2:
        raise ValueError("y_true, scores, and mask must have equal [N, labels] shapes")
    if y.shape[1] != len(labels) or len(patient_ids) != y.shape[0]:
        raise ValueError("labels and patient_ids must match the metric arrays")

    def confusion_metric(name: str, threshold: float) -> Callable[[np.ndarray, np.ndarray, np.ndarray | None], float | None]:
        def calculate(targets: np.ndarray, probabilities: np.ndarray, observed: np.ndarray | None) -> float | None:
            value = confusion_at_threshold(targets, probabilities, threshold, observed)[name]
            return None if value is None else float(value)

        return calculate

    for label_index, label in enumerate(labels):
        threshold = float(thresholds.get(label, 0.5))
        metric_functions: tuple[tuple[str, Callable[[np.ndarray, np.ndarray, np.ndarray | None], float | None]], ...] = (
            ("auroc", binary_auroc),
            ("auprc", binary_auprc),
            ("sensitivity", confusion_metric("sensitivity", threshold)),
            ("specificity", confusion_metric("specificity", threshold)),
            ("f1", confusion_metric("f1", threshold)),
            ("brier", brier_score),
        )
        for metric_offset, (metric_name, metric) in enumerate(metric_functions):
            result["per_class"][label][f"{metric_name}_ci"] = bootstrap_ci(
                metric,
                y[:, label_index],
                p[:, label_index],
                m[:, label_index],
                patient_ids,
                n_resamples,
                seed + label_index * len(metric_functions) + metric_offset,
            )
    return result


def evaluate_multilabel(
    y_true: Any,
    scores: Any,
    labels: Sequence[str],
    mask: Any | None = None,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    y = np.asarray(y_true)
    p = np.asarray(scores)
    m = np.asarray(mask) if mask is not None else np.ones_like(y, dtype=bool)
    if y.shape != p.shape or y.shape != m.shape or y.ndim != 2 or y.shape[1] != len(labels):
        raise ValueError("y_true, scores, mask must be [N, number_of_labels]")
    per_class: dict[str, dict[str, Any]] = {}
    for index, label in enumerate(labels):
        threshold = (thresholds or {}).get(label, 0.5)
        per_class[label] = {
            "auroc": binary_auroc(y[:, index], p[:, index], m[:, index]),
            "auprc": binary_auprc(y[:, index], p[:, index], m[:, index]),
            "brier": brier_score(y[:, index], p[:, index], m[:, index]),
            **confusion_at_threshold(y[:, index], p[:, index], threshold, m[:, index]),
            "threshold": threshold,
        }
    def macro(key: str) -> float | None:
        values = [item[key] for item in per_class.values() if item[key] is not None]
        return float(np.mean(values)) if values else None
    return {"per_class": per_class, "macro": {key: macro(key) for key in ("auroc", "auprc", "brier", "sensitivity", "specificity", "f1")}}
