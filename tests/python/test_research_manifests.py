from __future__ import annotations

import csv
import json
import unittest

from cxr_research.data import load_chexpert_csv
from cxr_research.external import (
    evaluate_mimic_external,
    sample_mimic_external,
    verify_external_manifest,
)
from cxr_research.labels import CHEXPERT_LABELS
from cxr_research.manifests import (
    build_chexpert_split_manifest,
    read_chexpert_split_manifest,
    write_manifest,
)
from PIL import Image
from test_utils import build_synthetic_medmodel, case_directory


class ResearchManifestTests(unittest.TestCase):
    def test_real_chexpert_path_shape_and_patient_split(self) -> None:
        root = case_directory("chexpert-manifest")
        csv_path = root / "train.csv"
        fieldnames = ["Path", "Frontal/Lateral", "AP/PA", "Age", *CHEXPERT_LABELS]
        rows = []
        for index in range(10):
            relative = f"CheXpert-v1.0-small/train/patient{index:05d}/study1/view1_frontal.jpg"
            image = root / relative
            image.parent.mkdir(parents=True, exist_ok=True)
            Image.new("L", (128, 128), 100 + index).save(image)
            rows.append(
                {
                    "Path": relative,
                    "Frontal/Lateral": "Frontal",
                    "AP/PA": "AP" if index % 2 else "PA",
                    "Age": "40",
                    "Atelectasis": "1" if index % 2 else "0",
                }
            )
        rows.extend(
            [
                {
                    "Path": "CheXpert-v1.0-small/train/patient99990/study1/lateral.jpg",
                    "Frontal/Lateral": "Lateral",
                    "AP/PA": "",
                    "Age": "50",
                },
                {
                    "Path": "CheXpert-v1.0-small/train/patient99991/study1/frontal.jpg",
                    "Frontal/Lateral": "Frontal",
                    "AP/PA": "PA",
                    "Age": "17",
                },
            ]
        )
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        records = load_chexpert_csv(csv_path, root)
        self.assertEqual(len(records), 10)
        self.assertEqual(records[0].patient_id, "00000")
        self.assertEqual(records[0].study_id, "1")
        self.assertEqual(records[0].mask[0], 1.0)

        manifest = build_chexpert_split_manifest(records, csv_path, seed=19)
        self.assertGreater(manifest["imageBytes"], 0)
        manifest_path = write_manifest(root / "split.json", manifest)
        split = read_chexpert_split_manifest(manifest_path)
        patient_sets = {
            name: {record.patient_id for record in values}
            for name, values in split.items()
        }
        self.assertTrue(patient_sets["train"])
        self.assertFalse(patient_sets["train"] & patient_sets["validation"])
        self.assertFalse(patient_sets["train"] & patient_sets["test"])

        changed = json.loads(manifest_path.read_text(encoding="utf-8"))
        changed["records"]["train"][0]["study_id"] = "tampered"
        manifest_path.write_text(json.dumps(changed), encoding="utf-8")
        with self.assertRaises(ValueError):
            read_chexpert_split_manifest(manifest_path)

    def test_mimic_sampling_is_deterministic_budgeted_and_locked(self) -> None:
        root = case_directory("mimic-manifest")
        image_root = root / "images"
        image_root.mkdir()
        rows = []
        for patient in range(1, 7):
            for study in range(1, 3):
                relative = f"p{patient}/s{study}.jpg"
                target = image_root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(bytes([patient, study]) * 10)
                rows.append(
                    {
                        "subject_id": str(patient),
                        "study_id": str(study),
                        "path": relative,
                        "ViewPosition": "PA" if study == 1 else "AP",
                        "age": "55",
                        "outcome": "must-not-be-read",
                    }
                )
        rows.append(
            {
                "subject_id": "99",
                "study_id": "1",
                "path": "missing.jpg",
                "ViewPosition": "LATERAL",
                "age": "16",
                "outcome": "must-not-be-read",
            }
        )
        csv_path = root / "metadata.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        first_path = root / "mimic-a.json"
        second_path = root / "mimic-b.json"
        first = sample_mimic_external(
            csv_path,
            first_path,
            image_root=image_root,
            seed=7,
            target_patients=6,
            max_bytes=80,
        )
        second = sample_mimic_external(
            csv_path,
            second_path,
            image_root=image_root,
            seed=7,
            target_patients=6,
            max_bytes=80,
        )
        self.assertEqual(first["records"], second["records"])
        self.assertLessEqual(first["totalBytes"], 80)
        self.assertEqual(first["maxBytes"], 80)
        self.assertEqual(len({item["patientId"] for item in first["records"]}), len(first["records"]))
        self.assertEqual(verify_external_manifest(first_path)["manifestSha256"], first["manifestSha256"])

        tampered = json.loads(first_path.read_text(encoding="utf-8"))
        tampered["selectedPatients"] += 1
        first_path.write_text(json.dumps(tampered), encoding="utf-8")
        with self.assertRaises(ValueError):
            verify_external_manifest(first_path)

    def test_mimic_sampling_refuses_to_seal_an_empty_selection(self) -> None:
        """A wrong --image-root must fail instead of sealing an empty test set."""

        root = case_directory("mimic-empty-selection")
        image_root = root / "images"
        image_root.mkdir()
        rows = []
        for patient in range(1, 5):
            relative = f"p{patient}/s1.jpg"
            rows.append(
                {
                    "subject_id": str(patient),
                    "study_id": "1",
                    "path": relative,
                    "ViewPosition": "PA",
                    "age": "60",
                }
            )
        csv_path = root / "metadata.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        output_path = root / "mimic-empty.json"
        with self.assertRaises(ValueError):
            sample_mimic_external(
                csv_path,
                output_path,
                image_root=image_root,
                target_patients=4,
            )
        # Nothing may be sealed when no image was found.
        self.assertFalse(output_path.exists())
        self.assertFalse(output_path.with_suffix(".json.sha256").exists())

    def test_mimic_sampling_makes_partial_downloads_visible(self) -> None:
        """Eligible patients whose files are absent are counted, not silently dropped."""

        root = case_directory("mimic-partial-download")
        image_root = root / "images"
        image_root.mkdir()
        rows = []
        for patient in range(1, 7):
            relative = f"p{patient}/s1.jpg"
            if patient <= 4:
                target = image_root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(bytes([patient]) * 10)
            rows.append(
                {
                    "subject_id": str(patient),
                    "study_id": "1",
                    "path": relative,
                    "ViewPosition": "PA",
                    "age": "60",
                }
            )
        csv_path = root / "metadata.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        manifest = sample_mimic_external(
            csv_path,
            root / "mimic-partial.json",
            image_root=image_root,
            target_patients=6,
        )
        self.assertEqual(manifest["consideredPatients"], 6)
        self.assertEqual(manifest["missingImages"], 2)
        self.assertEqual(manifest["selectedPatients"], 4)

    def test_final_external_evaluation_emits_aggregate_result_only(self) -> None:
        root = case_directory("mimic-final-evaluation")
        image_root = root / "images"
        image_root.mkdir()
        metadata_rows = []
        label_rows = []
        for patient in range(1, 5):
            relative = f"p{patient}/s{patient}.jpg"
            image = image_root / relative
            image.parent.mkdir(parents=True, exist_ok=True)
            Image.new("L", (160, 160), 80 + patient).save(image)
            metadata_rows.append(
                {
                    "subject_id": str(patient),
                    "study_id": str(patient),
                    "path": relative,
                    "ViewPosition": "PA",
                    "age": "50",
                }
            )
            labels = {label: "1" if patient % 2 else "0" for label in CHEXPERT_LABELS}
            label_rows.append(
                {"subject_id": str(patient), "study_id": str(patient), **labels}
            )
        metadata_csv = root / "metadata.csv"
        with metadata_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=metadata_rows[0].keys())
            writer.writeheader()
            writer.writerows(metadata_rows)
        labels_csv = root / "labels.csv"
        with labels_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=label_rows[0].keys())
            writer.writeheader()
            writer.writerows(label_rows)
        sealed_path = root / "sealed.json"
        sample_mimic_external(
            metadata_csv,
            sealed_path,
            image_root=image_root,
            seed=11,
            target_patients=4,
            max_bytes=10 * 1024**2,
        )
        result = evaluate_mimic_external(
            sealed_path,
            labels_csv,
            build_synthetic_medmodel(root),
            root / "external-output",
            image_root=image_root,
            batch_size=1,
            bootstrap_resamples=10,
        )
        serialized = json.dumps(result).lower()
        self.assertEqual(result["sampleCount"], 4)
        self.assertNotIn("patientid", serialized)
        self.assertNotIn("subject_id", serialized)
        self.assertNotIn("predictions", serialized)
        for metric in ("auroc", "auprc", "sensitivity", "specificity", "f1", "brier"):
            self.assertIn(f"{metric}_ci", result["metrics"]["per_class"]["Atelectasis"])
        self.assertTrue((root / "external-output" / "external-result.json").is_file())
        self.assertTrue((root / "external-output" / "failure_cases.local.json").is_file())


if __name__ == "__main__":
    unittest.main()
