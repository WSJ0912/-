from __future__ import annotations

import unittest
from itertools import pairwise

import numpy as np
from cxr_research.labels import CHEXPERT_LABELS
from cxr_research.metrics import (
    binary_auprc,
    binary_auroc,
    binary_curve_points,
    bootstrap_ci,
    evaluate_multilabel,
)
from cxr_research.splits import assert_no_patient_overlap, manifest_sha256, split_patient_records
from cxr_research.thresholds import select_threshold_for_sensitivity


class SplitMetricTests(unittest.TestCase):
    def test_patient_level_split_is_deterministic_and_disjoint(self) -> None:
        records = [{"patient_id": f"P{i}", "study_id": f"S{i}-{j}"} for i in range(30) for j in range(2)]
        first = split_patient_records(records, seed=7)
        second = split_patient_records(records, seed=7)
        self.assertEqual(first, second)
        assert_no_patient_overlap(first)
        sets = first.patient_sets()
        self.assertTrue(all(not (sets[left] & sets[right]) for left in sets for right in sets if left != right))

    def test_manifest_hash_is_order_independent(self) -> None:
        left = manifest_sha256([{"study": "b"}, {"study": "a"}])
        right = manifest_sha256([{"study": "a"}, {"study": "b"}])
        self.assertEqual(left, right)

    def test_auc_and_average_precision(self) -> None:
        y = np.array([0, 0, 1, 1])
        p = np.array([0.1, 0.2, 0.8, 0.9])
        self.assertAlmostEqual(binary_auroc(y, p), 1.0)
        self.assertAlmostEqual(binary_auprc(y, p), 1.0)

    def test_average_precision_handles_tied_scores_as_one_threshold(self) -> None:
        y = np.array([1, 0, 1, 0])
        p = np.array([0.5, 0.5, 0.5, 0.5])
        self.assertAlmostEqual(binary_auprc(y, p), 0.5)

    def test_aggregate_curve_points_are_monotonic_and_contain_no_rows(self) -> None:
        curves = binary_curve_points(
            np.array([0, 1, 0, 1]),
            np.array([0.1, 0.9, 0.4, 0.7]),
        )
        self.assertGreaterEqual(len(curves["roc"]), 2)
        self.assertTrue(
            all(
                left["x"] <= right["x"]
                for left, right in pairwise(curves["roc"])
            )
        )
        self.assertTrue(
            all(
                left["y"] <= right["y"]
                for left, right in pairwise(curves["roc"])
            )
        )
        self.assertEqual(set(curves["roc"][0]), {"x", "y", "threshold"})

    def test_threshold_meets_screening_sensitivity(self) -> None:
        y = np.array([0, 0, 1, 1, 1])
        p = np.array([0.05, 0.3, 0.4, 0.7, 0.95])
        threshold = select_threshold_for_sensitivity(y, p, target_sensitivity=0.9)
        self.assertGreaterEqual(float(np.sum((p >= threshold) & (y == 1))) / 3.0, 0.9)

    def test_patient_bootstrap_uses_patient_groups(self) -> None:
        y = np.array([[0], [1], [0], [1]])
        p = np.array([[0.1], [0.9], [0.2], [0.8]])
        ids = ["a", "a", "b", "b"]
        result = bootstrap_ci(lambda a, b, m: binary_auroc(a, b, m), y[:, 0], p[:, 0], patient_ids=ids, n_resamples=40)
        self.assertEqual(result["n"], 40)
        self.assertEqual(result["estimate"], 1.0)

    def test_multilabel_unknown_metric_is_none(self) -> None:
        y = np.zeros((2, 14))
        p = np.full((2, 14), 0.5)
        result = evaluate_multilabel(y, p, CHEXPERT_LABELS)
        self.assertIsNone(result["per_class"]["Atelectasis"]["auroc"])
        self.assertIsNone(result["macro"]["auroc"])


if __name__ == "__main__":
    unittest.main()
