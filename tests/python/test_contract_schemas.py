from __future__ import annotations

import base64
import copy
import json
import subprocess
import sys
import unittest
from pathlib import Path
from typing import Any

from cxr_research.artifacts import (
    model_manifest_sha256,
    validate_experiment_bundle,
    validate_model_manifest,
)
from cxr_research.labels import CHEXPERT_LABELS, CONTRACT_SCHEMAS
from jsonschema import Draft202012Validator, FormatChecker

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_ROOT = ROOT / "contracts" / "schemas"
GENERATOR = ROOT / "scripts" / "generate-contracts.py"
GENERATED_OUTPUTS = (
    ROOT / "ml" / "cxr_research" / "labels.py",
    ROOT / "contracts" / "types.ts",
    ROOT / "apps" / "desktop" / "src" / "generated" / "contracts.ts",
)
TIMESTAMP = "2026-09-20T12:00:00+00:00"


def valid_samples() -> dict[str, dict[str, Any]]:
    probabilities = {label: 0.5 for label in CHEXPERT_LABELS}
    logits = {label: 0.0 for label in CHEXPERT_LABELS}
    thresholds = {label: 0.5 for label in CHEXPERT_LABELS}
    return {
        "study": {
            "studyId": "ST-ABC123",
            "createdAt": TIMESTAMP,
            "modality": "DX",
            "bodyPart": "CHEST",
            "viewPosition": "PA",
            "adultConfirmed": True,
            "status": "ready",
        },
        "prediction": {
            "predictionId": "PRD-ABC123",
            "studyId": "ST-ABC123",
            "modelId": "mixstyle-cxr",
            "modelVersion": "0.1.0-test",
            "modelSha256": "a" * 64,
            "manifestSha256": "b" * 64,
            "createdAt": TIMESTAMP,
            "probabilities": probabilities,
            "logits": logits,
            "thresholds": thresholds,
            "camAvailable": True,
            "source": "onnx",
        },
        "prediction-cam": {
            "label": "Atelectasis",
            "width": 2,
            "height": 2,
            "pixelsBase64": base64.b64encode(bytes((0, 64, 128, 255))).decode("ascii"),
            "sha256": "c" * 64,
        },
        "review": {
            "reviewId": "REV-ABC123",
            "studyId": "ST-ABC123",
            "predictionId": "PRD-ABC123",
            "doctorId": "USR-DOCTOR",
            "createdAt": TIMESTAMP,
            "decisions": {"Atelectasis": "confirmed"},
            "notes": "reviewed",
        },
        "report-revision": {
            "reportId": "RPT-ABC123",
            "studyId": "ST-ABC123",
            "revision": 1,
            "authorId": "USR-DOCTOR",
            "createdAt": TIMESTAMP,
            "body": "findings",
            "status": "draft",
            "confirmedAt": None,
            "reviewId": None,
        },
        "model-manifest": {
            "schemaVersion": "medmodel-1",
            "modelId": "mixstyle-cxr",
            "version": "0.1.0-test",
            "architecture": "densenet121",
            "labels": list(CHEXPERT_LABELS),
            "input": {
                "width": 320,
                "height": 320,
                "channels": 1,
                "normalization": "imagenet-gray",
            },
            "thresholds": thresholds,
            "preprocessing": {"viewPositions": ["AP", "PA"], "adultOnly": True},
            "training": {"commit": "test-commit", "seeds": [17, 29, 43], "method": "mixstyle"},
            "dataSources": ["authorized-source"],
            "license": "test-only",
            "files": {"model.onnx": "a" * 64, "labels.json": "d" * 64},
            "modelSha256": "a" * 64,
            "manifestSha256": "b" * 64,
        },
        "experiment-bundle": {
            "schemaVersion": "medexperiment-1",
            "experimentId": "formal-test",
            "config": {
                "methods": ["baseline", "mixstyle"],
                "formal": True,
                "formalRunLimit": 6,
            },
            "seeds": [17, 29, 43],
            "datasetManifestSha256": "e" * 64,
            "aggregateMetrics": {"mixstyle_testPrimaryMacroAuroc_mean": 0.8},
            "perClassMetrics": {
                "Atelectasis": {
                    "mixstyle_auroc_mean": 0.81,
                    "mixstyle_auroc_ci": {
                        "estimate": 0.81,
                        "lower": 0.74,
                        "upper": 0.87,
                        "n": 2000,
                    },
                }
            },
            "curves": {
                "mixstyle.17.Atelectasis": {
                    "label": "Atelectasis",
                    "method": "mixstyle",
                    "seed": 17,
                    "roc": [
                        {"x": 0.0, "y": 0.0, "threshold": 1.0},
                        {"x": 1.0, "y": 1.0, "threshold": None},
                    ],
                    "pr": [
                        {"x": 0.0, "y": 1.0, "threshold": 1.0},
                        {"x": 1.0, "y": 0.5, "threshold": None},
                    ],
                }
            },
            "modelCard": {
                "intendedUse": "adult AP/PA chest radiograph research prototype",
                "limitations": ["not a medical device"],
                "selection": "best validation macro AUROC",
                "warnings": ["requires external validation"],
            },
        },
    }


class ContractSchemaTests(unittest.TestCase):
    def test_schema_files_are_valid_draft_2020_12(self) -> None:
        paths = sorted(SCHEMA_ROOT.glob("*.schema.json"))
        self.assertEqual(
            {path.name for path in paths},
            {f"{name}.schema.json" for name in CONTRACT_SCHEMAS},
        )
        for path in paths:
            with self.subTest(path=path.name):
                schema = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(schema["$schema"], "https://json-schema.org/draft/2020-12/schema")
                Draft202012Validator.check_schema(schema)

    def test_all_contract_samples_are_accepted(self) -> None:
        samples = valid_samples()
        self.assertEqual(set(samples), set(CONTRACT_SCHEMAS))
        for name, sample in samples.items():
            with self.subTest(schema=name):
                validator = Draft202012Validator(
                    CONTRACT_SCHEMAS[name], format_checker=FormatChecker()
                )
                self.assertEqual(list(validator.iter_errors(sample)), [])

    def test_extra_fields_and_wrong_types_are_rejected(self) -> None:
        samples = valid_samples()
        extra = copy.deepcopy(samples["prediction"])
        extra["localPath"] = "C:/private/image.dcm"
        wrong_type = copy.deepcopy(samples["experiment-bundle"])
        wrong_type["aggregateMetrics"]["macro_auroc"] = "0.8"
        patient_object = copy.deepcopy(samples["experiment-bundle"])
        patient_object["modelCard"]["details"] = {"patient_id": "P1"}
        cases = (
            ("prediction", extra),
            ("experiment-bundle", wrong_type),
            ("experiment-bundle", patient_object),
        )
        for name, value in cases:
            with self.subTest(schema=name, value=value):
                validator = Draft202012Validator(CONTRACT_SCHEMAS[name])
                self.assertTrue(list(validator.iter_errors(value)))

    def test_runtime_experiment_validation_runs_schema_before_semantics(self) -> None:
        sample = valid_samples()["experiment-bundle"]
        validate_experiment_bundle(sample)
        malformed = copy.deepcopy(sample)
        malformed["unexpected"] = True
        with self.assertRaisesRegex(ValueError, "schema validation failed"):
            validate_experiment_bundle(malformed)

    def test_runtime_model_validation_runs_schema_before_semantics(self) -> None:
        manifest = valid_samples()["model-manifest"]
        manifest["manifestSha256"] = model_manifest_sha256(manifest)
        validate_model_manifest(manifest)
        malformed = copy.deepcopy(manifest)
        malformed["training"]["seedPolicy"] = "untracked"
        with self.assertRaisesRegex(ValueError, "schema validation failed"):
            validate_model_manifest(malformed)

    def test_labels_json_matches_every_generated_label_contract(self) -> None:
        dictionary = json.loads((ROOT / "contracts" / "labels.json").read_text(encoding="utf-8"))
        expected = [item["key"] for item in dictionary["labels"]]
        self.assertEqual(expected, list(CHEXPERT_LABELS))
        manifest = CONTRACT_SCHEMAS["model-manifest"]
        self.assertEqual(manifest["properties"]["labels"]["const"], expected)

    def test_generated_contracts_exist_and_are_current(self) -> None:
        for path in GENERATED_OUTPUTS:
            with self.subTest(path=path):
                self.assertTrue(path.is_file())
                self.assertIn(
                    "Generated by scripts/generate-contracts.py",
                    path.read_text(encoding="utf-8"),
                )
        result = subprocess.run(
            [sys.executable, str(GENERATOR), "--check"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr or result.stdout)
        desktop = GENERATED_OUTPUTS[-1].read_text(encoding="utf-8")
        self.assertIn("export interface CurvePoint", desktop)
        self.assertIn("export interface ExperimentBundle", desktop)
        self.assertIn("export interface PredictionCam", desktop)


if __name__ == "__main__":
    unittest.main()
