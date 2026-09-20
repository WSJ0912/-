from __future__ import annotations

import json
import unittest

from cxr_research.artifacts import read_medexperiment
from cxr_research.cli import _select_deployment, _validate_deployment_checkpoint
from cxr_research.labels import CHEXPERT_LABELS
from cxr_research.runner import aggregate_experiment
from test_utils import case_directory


class DeploymentProtocolTests(unittest.TestCase):
    def test_selects_best_of_three_formal_mixstyle_validation_scores(self) -> None:
        root = case_directory("deployment-selection")
        paths = []
        for seed, score in ((17, 0.71), (29, 0.76), (43, 0.74)):
            run = root / f"seed-{seed}"
            run.mkdir()
            (run / "best.pt").write_bytes(b"test checkpoint marker")
            result_path = run / "result.json"
            result_path.write_text(
                json.dumps(
                    {
                        "method": "mixstyle",
                        "seed": seed,
                        "formal": True,
                        "validationPrimaryMacroAuroc": score,
                    }
                ),
                encoding="utf-8",
            )
            paths.append(result_path)

        record = _select_deployment(paths, root / "selection.json")
        self.assertEqual(record["selected"]["seed"], 29)
        self.assertEqual(len(record["candidates"]), 3)
        self.assertEqual(len(record["selected"]["checkpointSha256"]), 64)
        selected_checkpoint = root / "seed-29" / "best.pt"
        validated = _validate_deployment_checkpoint(
            root / "selection.json",
            selected_checkpoint,
            {"method": "mixstyle", "seed": 29},
        )
        self.assertEqual(validated["selected"]["seed"], 29)

        selected_checkpoint.write_bytes(b"tampered checkpoint")
        with self.assertRaises(ValueError):
            _validate_deployment_checkpoint(
                root / "selection.json",
                selected_checkpoint,
                {"method": "mixstyle", "seed": 29},
            )

    def test_smoke_results_cannot_enter_formal_experiment_bundle(self) -> None:
        root = case_directory("deployment-smoke-rejection")
        with self.assertRaises(ValueError):
            aggregate_experiment(
                [{"method": "mixstyle", "seed": 17, "formal": False}],
                "formal-test",
                "a" * 64,
                root / "formal.medexperiment",
            )

    def test_formal_bundle_preserves_real_aggregate_curve_points(self) -> None:
        root = case_directory("deployment-aggregate-curves")
        per_class = {
            label: {
                "auroc": 0.7,
                "auprc": 0.6,
                "sensitivity": 0.9,
                "specificity": 0.5,
                "f1": 0.65,
                "brier": 0.2,
            }
            for label in CHEXPERT_LABELS
        }
        curve = {
            "roc": [
                {"x": 0.0, "y": 0.0, "threshold": None},
                {"x": 1.0, "y": 1.0, "threshold": 0.2},
            ],
            "pr": [
                {"x": 0.0, "y": 1.0, "threshold": None},
                {"x": 1.0, "y": 0.5, "threshold": 0.2},
            ],
        }
        package = aggregate_experiment(
            [
                {
                    "method": "mixstyle",
                    "seed": 17,
                    "formal": True,
                    "testPrimaryMacroAuroc": 0.7,
                    "testPrimaryMacroAuprc": 0.6,
                    "testMetrics": {"per_class": per_class},
                    "curves": {label: curve for label in CHEXPERT_LABELS},
                }
            ],
            "formal-curves",
            "a" * 64,
            root / "formal.medexperiment",
        )
        bundle = read_medexperiment(package)
        self.assertEqual(len(bundle["curves"]), len(CHEXPERT_LABELS))
        series = bundle["curves"]["mixstyle-seed-17-label-0"]
        self.assertEqual(series["label"], CHEXPERT_LABELS[0])
        self.assertEqual(series["roc"][-1]["y"], 1.0)


if __name__ == "__main__":
    unittest.main()
