from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
from cxr_competition.data import (
    group_by_study,
    load_split,
    parse_predict_class,
    records_to_arrays,
)
from cxr_competition.labels import LABEL_IDS
from cxr_competition.metrics import aggregate_by_study, macro_auc, per_class_auc, tune_thresholds
from cxr_competition.model import build_model, load_checkpoint, save_checkpoint
from cxr_competition.predict import (
    normalize_probabilities,
    submission_summary,
    write_submission,
)


def _write_image(path: Path) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (8, 8), color=128).save(path)


def _write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


class CompetitionDataTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / "cxr_first"
        train_header = [
            "Dicom_id",
            "Subject_id",
            "Study_id",
            "ViewPosition",
            "PatientOrientation",
            "Rows",
            "Columns",
            "Predict_class",
        ]
        _write_csv(
            self.root / "train.csv",
            train_header,
            [
                ["pic_1", "subject_1", "study_1", "AP", "Erect", 1024, 1024, "7;9"],
                ["pic_2", "subject_1", "study_1", "LATERAL", "Erect", 1024, 1024, "10"],
                ["pic_3", "subject_2", "study_2", "PA", "Erect", 1024, 1024, "6"],
            ],
        )
        for subject, study, dicom in (
            ("subject_1", "study_1", "pic_1"),
            ("subject_1", "study_1", "pic_2"),
            ("subject_2", "study_2", "pic_3"),
        ):
            _write_image(self.root / "train" / subject / study / f"{dicom}.jpg")
        test_header = train_header[:-1]
        _write_csv(
            self.root / "test.csv",
            test_header,
            [["pic_9", "subject_9", "study_9", "PA", "Erect", 1024, 1024]],
        )
        _write_image(self.root / "test" / "subject_9" / "study_9" / "pic_9.jpg")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_parse_predict_class_sorts_and_deduplicates(self) -> None:
        self.assertEqual(parse_predict_class("9;7;9"), (7, 9))
        self.assertEqual(parse_predict_class(" 10 "), (10,))

    def test_parse_predict_class_rejects_invalid_ids(self) -> None:
        with self.assertRaises(ValueError):
            parse_predict_class("0")
        with self.assertRaises(ValueError):
            parse_predict_class("")
        with self.assertRaises(ValueError):
            parse_predict_class("7;x")

    def test_load_split_builds_multi_hot_targets_and_paths(self) -> None:
        records = load_split(self.root / "train.csv", self.root / "train")
        self.assertEqual(len(records), 3)
        self.assertEqual(records[0].targets, tuple(1.0 if i in (7, 9) else 0.0 for i in LABEL_IDS))
        self.assertEqual(records[1].targets, tuple(1.0 if i == 10 else 0.0 for i in LABEL_IDS))
        self.assertTrue(Path(records[0].image_path).is_file())
        targets, mask = records_to_arrays(records)
        self.assertEqual(targets.shape, (3, 10))
        self.assertEqual(mask.shape, (3, 10))
        self.assertEqual(int(targets[:, 6].sum()), 1)
        self.assertEqual(int(targets[:, 9].sum()), 1)

    def test_load_split_handles_unlabelled_test_rows(self) -> None:
        records = load_split(self.root / "test.csv", self.root / "test", labelled=False)
        self.assertIsNone(records[0].targets)
        with self.assertRaises(ValueError):
            records_to_arrays(records)

    def test_missing_image_is_reported(self) -> None:
        _write_csv(
            self.root / "valid.csv",
            [
                "Dicom_id",
                "Subject_id",
                "Study_id",
                "ViewPosition",
                "PatientOrientation",
                "Rows",
                "Columns",
                "Predict_class",
            ],
            [["pic_x", "subject_x", "study_x", "PA", "Erect", 1, 1, "1"]],
        )
        with self.assertRaises(FileNotFoundError):
            load_split(self.root / "valid.csv", self.root / "valid")

    def test_group_by_study_keeps_csv_order(self) -> None:
        records = load_split(self.root / "train.csv", self.root / "train")
        groups = group_by_study(records)
        self.assertEqual(list(groups), ["study_1", "study_2"])
        self.assertEqual(len(groups["study_1"]), 2)


class CompetitionMetricTests(unittest.TestCase):
    def test_macro_auc_and_per_class(self) -> None:
        targets = np.asarray([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=np.float32)
        scores = np.asarray([[0.9, 0.1], [0.2, 0.8], [0.7, 0.6], [0.1, 0.4]], dtype=np.float32)
        self.assertAlmostEqual(macro_auc(targets, scores), 1.0)
        values = per_class_auc(targets, scores)
        self.assertAlmostEqual(values[1], 1.0)
        self.assertAlmostEqual(values[2], 1.0)

    def test_single_class_auc_is_none(self) -> None:
        targets = np.asarray([[1, 0], [1, 1]], dtype=np.float32)
        scores = np.asarray([[0.9, 0.2], [0.8, 0.7]], dtype=np.float32)
        values = per_class_auc(targets, scores)
        self.assertIsNone(values[1])

    def test_tune_thresholds_returns_every_label(self) -> None:
        targets = np.asarray([[1, 0], [1, 1], [0, 0], [0, 1]], dtype=np.float32)
        scores = np.asarray([[0.9, 0.2], [0.8, 0.7], [0.1, 0.3], [0.2, 0.6]], dtype=np.float32)
        thresholds = tune_thresholds(targets, scores)
        self.assertEqual(set(thresholds), {1, 2})
        self.assertTrue(all(0.0 < value < 1.0 for value in thresholds.values()))

    def test_aggregate_study_max(self) -> None:
        study_ids = ["s1", "s1", "s2"]
        scores = np.asarray([[0.1, 0.2], [0.5, 0.3], [0.4, 0.7]], dtype=np.float32)
        ids, aggregated = aggregate_by_study(study_ids, scores, method="max")
        self.assertEqual(ids, ["s1", "s2"])
        np.testing.assert_allclose(aggregated, np.asarray([[0.5, 0.3], [0.4, 0.7]]))


class CompetitionSubmissionTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "submission.csv"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_write_submission_long_format(self) -> None:
        scores = np.arange(20, dtype=np.float64).reshape(2, 10) / 20.0
        write_submission(self.path, ["study_1", "study_2"], scores)
        with self.path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.reader(handle))
        self.assertEqual(rows[0], ["Study_id", "Predict_class", "Probability", "Gate"])
        self.assertEqual(len(rows), 21)  # header + 2 studies * 10 classes
        self.assertEqual(rows[1], ["study_1", "1", "0.000000", ""])
        self.assertEqual(rows[11], ["study_2", "1", "0.500000", ""])
        self.assertEqual([row[1] for row in rows[1:11]], [str(label_id) for label_id in LABEL_IDS])

    def test_write_submission_custom_gate(self) -> None:
        write_submission(self.path, ["study_1"], np.full((1, 10), 0.5), gate=0.3)
        with self.path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.reader(handle))
        self.assertEqual(rows[1][3], "0.300000")

    def test_write_submission_rejects_invalid_gate(self) -> None:
        with self.assertRaises(ValueError):
            write_submission(self.path, ["study_1"], np.full((1, 10), 0.5), gate=1.5)

    def test_normalize_probabilities_l1(self) -> None:
        scores = np.asarray([[0.2, 0.3], [1.0, 1.0]], dtype=np.float64)
        normalized = normalize_probabilities(scores, "l1")
        np.testing.assert_allclose(normalized.sum(axis=1), np.ones(2))
        np.testing.assert_allclose(normalized[0], np.asarray([0.4, 0.6]))
        np.testing.assert_allclose(normalize_probabilities(scores, "none"), scores)

    def test_normalize_probabilities_rejects_unknown_method(self) -> None:
        with self.assertRaises(ValueError):
            normalize_probabilities(np.ones((1, 2)), "softmax")

    def test_submission_summary_rejects_wrong_header(self) -> None:
        self.path.write_text("Dicom_id,class_01\npic_1,0.1\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            submission_summary(self.path)

    def test_submission_summary_accepts_valid_file(self) -> None:
        scores = np.full((1, 10), 0.5)
        write_submission(self.path, ["study_1"], scores)
        summary = submission_summary(self.path)
        self.assertEqual(summary["rows"], 10)
        self.assertEqual(summary["studies"], 1)
        self.assertEqual(summary["columns"], ["Study_id", "Predict_class", "Probability", "Gate"])


class CompetitionModelTests(unittest.TestCase):
    def test_forward_shape_and_checkpoint_roundtrip(self) -> None:
        import torch

        model = build_model(len(LABEL_IDS), pretrained=False)
        model.eval()
        with torch.no_grad():
            logits = model(torch.zeros(1, 3, 64, 64))
        self.assertEqual(tuple(logits.shape), (1, len(LABEL_IDS)))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best.pt"
            saved = save_checkpoint(
                path,
                model=model,
                config={"imageSize": 64, "labelIds": list(LABEL_IDS)},
            )
            self.assertEqual(len(saved["checkpoint_sha256"]), 64)
            payload = load_checkpoint(path)
            self.assertEqual(payload["config"]["imageSize"], 64)


if __name__ == "__main__":
    unittest.main()
