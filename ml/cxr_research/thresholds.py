from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .metrics import confusion_at_threshold


def select_threshold_for_sensitivity(
    y_true: Any,
    scores: Any,
    target_sensitivity: float = 0.90,
    mask: Any | None = None,
) -> float:
    if not 0 < target_sensitivity <= 1:
        raise ValueError("target_sensitivity must be in (0, 1]")
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(scores, dtype=float)
    valid = np.isfinite(y) & np.isfinite(p)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    y, p = y[valid], p[valid]
    if not len(y) or not np.any(y == 1):
        return 0.5
    candidates = np.unique(np.concatenate(([0.0, 1.0], p)))
    eligible: list[tuple[float, float]] = []
    for threshold in candidates:
        stats = confusion_at_threshold(y, p, float(threshold))
        sensitivity = stats["sensitivity"]
        if sensitivity is not None and sensitivity >= target_sensitivity:
            specificity = stats["specificity"] if stats["specificity"] is not None else -1.0
            eligible.append((float(threshold), float(specificity)))
    if not eligible:
        return 0.0
    # Highest threshold gives the most specific operating point among those
    # meeting the screening sensitivity target; specificity breaks ties.
    eligible.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return eligible[0][0]


def select_thresholds(
    y_true: Any,
    scores: Any,
    labels: Sequence[str],
    target_sensitivity: float = 0.90,
    mask: Any | None = None,
) -> dict[str, float]:
    y = np.asarray(y_true)
    p = np.asarray(scores)
    m = np.asarray(mask) if mask is not None else None
    if y.shape != p.shape or y.ndim != 2 or y.shape[1] != len(labels):
        raise ValueError("y_true and scores must be [N, number_of_labels]")
    return {
        label: select_threshold_for_sensitivity(
            y[:, index], p[:, index], target_sensitivity, m[:, index] if m is not None else None
        )
        for index, label in enumerate(labels)
    }
