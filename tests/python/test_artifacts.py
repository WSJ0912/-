from __future__ import annotations

import hashlib
import json
import unittest
import zipfile

from cxr_research.artifacts import (
    build_medexperiment,
    build_medmodel,
    read_medexperiment,
    read_medmodel_manifest,
    validate_experiment_bundle,
    validate_model_manifest,
)
from cxr_research.export import create_manifest
from cxr_research.labels import CHEXPERT_LABELS
from inference_service.model_registry import ModelRegistry
from test_utils import build_synthetic_medmodel, case_directory


class ArtifactTests(unittest.TestCase):
    def test_medmodel_hashes_and_exact_labels(self) -> None:
        root = case_directory("artifacts-model")
        try:
            model = root / "model.onnx"
            model.write_bytes(b"synthetic test model, not a clinical weight")
            labels = root / "labels.json"
            labels.write_text(json.dumps({"labels": CHEXPERT_LABELS}), encoding="utf-8")
            thresholds = {label: 0.5 for label in CHEXPERT_LABELS}
            manifest = create_manifest(
                model,
                "test-model",
                "0.1.0-test",
                "mixstyle",
                "test",
                [17, 29, 43],
                ["synthetic"],
                "test-only",
                thresholds,
                {"labels.json": labels},
            )
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            package = root / "model.medmodel"
            build_medmodel(root, package)
            loaded = read_medmodel_manifest(package)
            self.assertEqual(loaded["labels"], list(CHEXPERT_LABELS))
            self.assertEqual(loaded["modelSha256"], hashlib.sha256(model.read_bytes()).hexdigest())
        finally:
            for path in root.iterdir():
                if path.is_file():
                    path.unlink()

    def test_invalid_model_labels_fail(self) -> None:
        manifest = {"schemaVersion": "medmodel-1", "modelId": "x", "version": "1", "architecture": "densenet121", "labels": ["wrong"], "input": {}, "thresholds": {}, "preprocessing": {}, "training": {}, "dataSources": [], "license": "x", "files": {"model.onnx": "0" * 64}}
        with self.assertRaises(ValueError):
            validate_model_manifest(manifest)

    def test_model_and_experiment_identifiers_cannot_escape_storage(self) -> None:
        root = case_directory("artifacts-identifiers")
        package = build_synthetic_medmodel(root)
        with zipfile.ZipFile(package, "r") as archive:
            manifest = json.loads(archive.read("manifest.json"))
        manifest["modelId"] = "../outside"
        with self.assertRaises(ValueError):
            validate_model_manifest(manifest)

        bundle = {
            "schemaVersion": "medexperiment-1",
            "experimentId": "../outside",
            "config": {},
            "seeds": [1],
            "datasetManifestSha256": "a" * 64,
            "aggregateMetrics": {},
            "perClassMetrics": {},
            "curves": {},
            "modelCard": {},
        }
        with self.assertRaises(ValueError):
            validate_experiment_bundle(bundle)

    def test_active_model_is_restored_after_service_restart(self) -> None:
        root = case_directory("artifacts-model-restore")
        package = build_synthetic_medmodel(root)
        model_directory = root / "installed"
        first = ModelRegistry(model_directory)
        installed = first.install(package)
        self.assertTrue(installed["ready"])

        restored = ModelRegistry(model_directory)
        self.assertTrue(restored.status()["ready"])
        self.assertEqual(restored.status()["modelSha256"], installed["modelSha256"])

    def test_experiment_cannot_contain_patient_level_fields(self) -> None:
        bundle = {"schemaVersion": "medexperiment-1", "experimentId": "e", "config": {"patient_ids": ["P1"]}, "seeds": [1], "datasetManifestSha256": "a" * 64, "aggregateMetrics": {}, "perClassMetrics": {}, "curves": {}, "modelCard": {}}
        with self.assertRaises(ValueError):
            validate_experiment_bundle(bundle)

    def test_experiment_round_trip(self) -> None:
        bundle = {"schemaVersion": "medexperiment-1", "experimentId": "e", "config": {}, "seeds": [1], "datasetManifestSha256": "a" * 64, "aggregateMetrics": {"macro_auroc": None}, "perClassMetrics": {}, "curves": {}, "modelCard": {}}
        root = case_directory("artifacts-experiment")
        package = root / "e.medexperiment"
        build_medexperiment(bundle, package)
        self.assertEqual(read_medexperiment(package), bundle)

    def test_experiment_rejects_zip_path_traversal(self) -> None:
        root = case_directory("artifacts-unsafe-experiment")
        package = root / "unsafe.medexperiment"
        bundle = {"schemaVersion": "medexperiment-1", "experimentId": "e", "config": {}, "seeds": [1], "datasetManifestSha256": "a" * 64, "aggregateMetrics": {}, "perClassMetrics": {}, "curves": {}, "modelCard": {}}
        with zipfile.ZipFile(package, "w") as archive:
            archive.writestr("experiment.json", json.dumps(bundle))
            archive.writestr("../patient.csv", "patient_id")
        with self.assertRaises(ValueError):
            read_medexperiment(package)


if __name__ == "__main__":
    unittest.main()
