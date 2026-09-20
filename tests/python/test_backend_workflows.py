from __future__ import annotations

import base64
import hashlib
import shutil
import sqlite3
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from cxr_research.labels import CHEXPERT_LABELS
from fastapi.testclient import TestClient
from inference_service.core import PlatformCore
from inference_service.main import create_app
from inference_service.security import Role, UserSession
from inference_service.storage import Database, utc_now
from test_utils import case_directory


def study(study_id: str) -> dict[str, object]:
    return {
        "studyId": study_id,
        "createdAt": utc_now(),
        "modality": "DX",
        "bodyPart": "CHEST",
        "viewPosition": "PA",
        "adultConfirmed": True,
        "deidentifiedPath": "unused.png",
        "importHash": hashlib.sha256(study_id.encode()).hexdigest(),
        "status": "ready",
    }


def manifest(model_sha: str = "a" * 64, manifest_sha: str = "b" * 64) -> dict[str, object]:
    return {
        "modelId": "test-model",
        "version": "1.0.0",
        "modelSha256": model_sha,
        "manifestSha256": manifest_sha,
        "labels": list(CHEXPERT_LABELS),
        "thresholds": {label: 0.5 for label in CHEXPERT_LABELS},
    }


def prediction(
    prediction_id: str,
    study_id: str,
    model_manifest: dict[str, object] | None = None,
    *,
    cam_available: bool = False,
) -> dict[str, object]:
    model_manifest = model_manifest or manifest()
    return {
        "predictionId": prediction_id,
        "studyId": study_id,
        "modelId": model_manifest["modelId"],
        "modelVersion": model_manifest["version"],
        "modelSha256": model_manifest["modelSha256"],
        "manifestSha256": model_manifest["manifestSha256"],
        "createdAt": utc_now(),
        "probabilities": {label: 0.25 for label in CHEXPERT_LABELS},
        "logits": {label: 0.0 for label in CHEXPERT_LABELS},
        "thresholds": {label: 0.5 for label in CHEXPERT_LABELS},
        "camAvailable": cam_available,
        "source": "onnx",
    }


def cam_rows() -> list[dict[str, object]]:
    rows = []
    for index, label in enumerate(CHEXPERT_LABELS):
        pixels = bytes((index, index + 1, index + 2, index + 3, index + 4, index + 5))
        rows.append(
            {
                "label": label,
                "width": 3,
                "height": 2,
                "pixels": pixels,
                "sha256": hashlib.sha256(pixels).hexdigest(),
            }
        )
    return rows


def nested_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return {
            str(key).lower()
            for key in value
        } | set().union(*(nested_keys(child) for child in value.values()))
    if isinstance(value, list):
        return set().union(*(nested_keys(child) for child in value))
    return set()


class StubModels:
    def __init__(self, model_manifest: dict[str, object]) -> None:
        self.manifest = model_manifest
        self.active = SimpleNamespace(manifest=model_manifest)
        self.calls = 0

    def status(self) -> dict[str, object]:
        return dict(self.manifest)

    def predict(self, _: object) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
        self.calls += 1
        logits = np.zeros((1, len(CHEXPERT_LABELS)), dtype=np.float32)
        cams = np.arange(
            len(CHEXPERT_LABELS) * 6, dtype=np.float32
        ).reshape(1, len(CHEXPERT_LABELS), 2, 3)
        return logits, cams, self.manifest


class CaptureAssistant:
    def __init__(self) -> None:
        self.payload: dict[str, object] | None = None

    def report_draft(self, payload: dict[str, object]) -> dict[str, object]:
        self.payload = payload
        return {"draft": "captured", "cautions": []}


class BackendWorkflowTests(unittest.TestCase):
    def test_foreign_keys_and_initial_admin_are_atomic(self) -> None:
        root = case_directory("backend-foreign-keys")
        db = Database(root / "platform.sqlite3")
        for _ in range(2):
            with db.connection() as connection:
                self.assertEqual(
                    connection.execute("PRAGMA foreign_keys").fetchone()[0], 1
                )

        with self.assertRaises(sqlite3.IntegrityError):
            db.save_review(
                {
                    "reviewId": "REV-ORPHAN",
                    "studyId": "ST-MISSING",
                    "predictionId": "PRD-MISSING",
                    "doctorId": "USR-MISSING",
                    "createdAt": utc_now(),
                    "decisions": {},
                    "notes": "",
                }
            )

        def create_admin(index: int) -> str:
            try:
                return db.create_initial_admin(f"admin-{index}", "hash")
            except PermissionError:
                return "denied"

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(create_admin, (1, 2)))
        self.assertEqual(results.count("denied"), 1)
        self.assertEqual(db.user_count(), 1)

    def test_legacy_report_revisions_are_migrated_with_owner_and_review_fk(self) -> None:
        root = case_directory("backend-report-migration")
        database_path = root / "legacy.sqlite3"
        with sqlite3.connect(database_path) as connection:
            connection.executescript(
                """
                CREATE TABLE users (
                    user_id TEXT PRIMARY KEY,
                    username TEXT NOT NULL UNIQUE,
                    role TEXT NOT NULL,
                    password_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE studies (
                    study_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    modality TEXT NOT NULL,
                    body_part TEXT NOT NULL,
                    view_position TEXT NOT NULL,
                    adult_confirmed INTEGER NOT NULL,
                    deidentified_path TEXT NOT NULL,
                    import_hash TEXT NOT NULL,
                    status TEXT NOT NULL
                );
                CREATE TABLE report_revisions (
                    report_id TEXT NOT NULL,
                    study_id TEXT NOT NULL REFERENCES studies(study_id),
                    revision INTEGER NOT NULL,
                    author_id TEXT NOT NULL REFERENCES users(user_id),
                    created_at TEXT NOT NULL,
                    body TEXT NOT NULL,
                    status TEXT NOT NULL,
                    confirmed_at TEXT,
                    review_id TEXT,
                    PRIMARY KEY (report_id, revision)
                );
                """
            )
            connection.execute(
                "INSERT INTO users VALUES (?, ?, ?, ?, ?)",
                ("USR-1", "doctor", "doctor", "hash", utc_now()),
            )
            item = study("ST-LEGACY")
            connection.execute(
                "INSERT INTO studies VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    item["studyId"],
                    item["createdAt"],
                    item["modality"],
                    item["bodyPart"],
                    item["viewPosition"],
                    1,
                    item["deidentifiedPath"],
                    item["importHash"],
                    item["status"],
                ),
            )
            connection.execute(
                "INSERT INTO report_revisions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    "RPT-LEGACY",
                    "ST-LEGACY",
                    1,
                    "USR-1",
                    utc_now(),
                    "legacy body",
                    "draft",
                    None,
                    "REV-DANGLING",
                ),
            )

        unsafe_path = root / "legacy-unsafe-owner.sqlite3"
        shutil.copy2(database_path, unsafe_path)
        with sqlite3.connect(unsafe_path) as connection:
            connection.execute(
                "INSERT INTO users VALUES (?, ?, ?, ?, ?)",
                ("USR-2", "other", "doctor", "hash", utc_now()),
            )
            connection.execute(
                "INSERT INTO report_revisions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    "RPT-LEGACY",
                    "ST-LEGACY",
                    2,
                    "USR-2",
                    utc_now(),
                    "unsafe takeover",
                    "draft",
                    None,
                    None,
                ),
            )
        with self.assertRaisesRegex(RuntimeError, "ownership migration conflict"):
            Database(unsafe_path)

        unsafe_review_path = root / "legacy-unsafe-review.sqlite3"
        shutil.copy2(database_path, unsafe_review_path)
        with sqlite3.connect(unsafe_review_path) as connection:
            connection.executescript(
                """
                CREATE TABLE predictions (
                    prediction_id TEXT PRIMARY KEY,
                    study_id TEXT NOT NULL
                );
                CREATE TABLE reviews (
                    review_id TEXT PRIMARY KEY,
                    study_id TEXT NOT NULL,
                    prediction_id TEXT NOT NULL,
                    doctor_id TEXT NOT NULL
                );
                """
            )
            connection.execute(
                "INSERT INTO users VALUES (?, ?, ?, ?, ?)",
                ("USR-2", "other", "doctor", "hash", utc_now()),
            )
            connection.execute(
                "INSERT INTO predictions VALUES (?, ?)",
                ("PRD-LEGACY", "ST-LEGACY"),
            )
            connection.execute(
                "INSERT INTO reviews VALUES (?, ?, ?, ?)",
                ("REV-WRONG-DOCTOR", "ST-LEGACY", "PRD-LEGACY", "USR-2"),
            )
            connection.execute(
                "UPDATE report_revisions SET review_id = ?",
                ("REV-WRONG-DOCTOR",),
            )
        with self.assertRaisesRegex(RuntimeError, "review migration conflict"):
            Database(unsafe_review_path)

        db = Database(database_path)
        self.assertEqual(db.get_report_record("RPT-LEGACY")["ownerId"], "USR-1")
        history = db.list_report_revisions("RPT-LEGACY", "USR-1")
        self.assertEqual(history[0]["body"], "legacy body")
        self.assertIsNone(history[0]["reviewId"])
        with db.connection() as connection:
            targets = {
                row["table"]
                for row in connection.execute(
                    "PRAGMA foreign_key_list(report_revisions)"
                ).fetchall()
            }
        self.assertTrue({"reports", "reviews"}.issubset(targets))
        with self.assertRaises(sqlite3.IntegrityError):
            db.save_report_revision(
                {
                    "reportId": "RPT-BAD-REVIEW",
                    "studyId": "ST-LEGACY",
                    "revision": 1,
                    "authorId": "USR-1",
                    "createdAt": utc_now(),
                    "body": "draft",
                    "status": "draft",
                    "reviewId": "REV-MISSING",
                }
            )

    def test_inference_is_idempotent_and_cams_survive_restart(self) -> None:
        root = case_directory("backend-inference-cams")
        core = PlatformCore(root)
        doctor_id = core.database.create_user("doctor", "doctor", "hash")
        actor = UserSession(doctor_id, Role.DOCTOR)
        core.database.save_study(study("ST-INFER"))
        models = StubModels(manifest())
        core.models = models

        with patch(
            "inference_service.core.preprocess_for_model",
            return_value=np.zeros((1, 1, 320, 320), dtype=np.float32),
        ):
            first = core.infer(actor, "ST-INFER")
            second = core.infer(actor, "ST-INFER")
        self.assertEqual(first["predictionId"], second["predictionId"])
        self.assertEqual(models.calls, 1)
        self.assertTrue(first["camAvailable"])
        self.assertEqual(len(core.database.list_prediction_cams(first["predictionId"])), 14)

        cam = core.prediction_cam(actor, first["predictionId"], 0)
        self.assertEqual((cam["width"], cam["height"]), (3, 2))
        self.assertEqual(len(base64.b64decode(cam["pixelsBase64"])), 6)
        restarted = PlatformCore(root)
        self.assertEqual(
            restarted.prediction_cam(actor, first["predictionId"], 0), cam
        )
        with restarted.database.connection() as connection:
            connection.execute(
                "DELETE FROM prediction_cams WHERE prediction_id = ? AND label = ?",
                (first["predictionId"], CHEXPERT_LABELS[-1]),
            )
        restarted_after_partial_cam = PlatformCore(root)
        stored = restarted_after_partial_cam.database.get_prediction(
            first["predictionId"]
        )
        self.assertIsNotNone(stored)
        self.assertFalse(stored["camAvailable"])

        core.database.save_study(study("ST-BAD-CAMS"))
        invalid_cams = cam_rows()
        invalid_cams[-1]["label"] = "Unexpected Label"
        with self.assertRaisesRegex(ValueError, "CAM labels"):
            core.database.save_prediction(
                prediction(
                    "PRD-BAD-CAMS",
                    "ST-BAD-CAMS",
                    cam_available=True,
                ),
                invalid_cams,
            )

        conflicting = StubModels(manifest("c" * 64, "d" * 64))
        core.models = conflicting
        with self.assertRaisesRegex(ValueError, "model identity conflict"):
            core.infer(actor, "ST-INFER")
        self.assertEqual(conflicting.calls, 0)

        core.database.save_study(study("ST-CONCURRENT"))
        candidates = [
            prediction("PRD-CONCURRENT-A", "ST-CONCURRENT"),
            prediction("PRD-CONCURRENT-B", "ST-CONCURRENT"),
        ]
        with ThreadPoolExecutor(max_workers=2) as executor:
            saved = list(executor.map(core.database.save_prediction, candidates))
        self.assertEqual(saved[0]["predictionId"], saved[1]["predictionId"])

        core.database.save_study(study("ST-IDENTITY"))
        v1 = manifest("1" * 64, "2" * 64)
        v2 = manifest("3" * 64, "4" * 64)
        v2["version"] = "2.0.0"
        core.database.save_prediction(prediction("PRD-V1", "ST-IDENTITY", v1))
        core.database.save_prediction(prediction("PRD-V2", "ST-IDENTITY", v2))
        with self.assertRaisesRegex(ValueError, "model identity conflict"):
            core.database.reusable_prediction(
                "ST-IDENTITY",
                str(v1["modelId"]),
                str(v1["version"]),
                str(v2["modelSha256"]),
                str(v2["manifestSha256"]),
            )

    def test_report_history_and_owner_checks(self) -> None:
        root = case_directory("backend-report-owner")
        core = PlatformCore(root)
        first_id = core.database.create_user("first", "doctor", "hash")
        second_id = core.database.create_user("second", "doctor", "hash")
        first = UserSession(first_id, Role.DOCTOR)
        second = UserSession(second_id, Role.DOCTOR)
        core.database.save_study(study("ST-REPORT"))
        core.database.save_prediction(prediction("PRD-REPORT", "ST-REPORT"))
        linked_review = core.save_review(
            first,
            "ST-REPORT",
            "PRD-REPORT",
            {"Atelectasis": "denied"},
            "linked review",
        )

        revision_one = core.save_report_draft(first, "ST-REPORT", "one")
        self.assertEqual(revision_one["reviewId"], linked_review["reviewId"])
        revision_two = core.save_report_draft(
            first,
            "ST-REPORT",
            "two",
            report_id=revision_one["reportId"],
        )
        with self.assertRaises(PermissionError):
            core.save_report_draft(
                second,
                "ST-REPORT",
                "takeover",
                report_id=revision_one["reportId"],
            )
        self.assertEqual(len(core.list_reports(first)), 1)
        self.assertEqual(core.list_reports(second), [])
        self.assertEqual(
            [item["revision"] for item in core.report_history(first, revision_one["reportId"])],
            [2, 1],
        )
        with self.assertRaises(PermissionError):
            core.report_history(second, revision_one["reportId"])
        with self.assertRaises(ValueError):
            core.confirm_report(first, revision_one["reportId"], 1)
        with self.assertRaises(PermissionError):
            core.confirm_report(second, revision_one["reportId"], 2)
        with self.assertRaises(PermissionError):
            core.export_report(
                second,
                revision_one["reportId"],
                revision_two["revision"],
                str(root / "stolen.pdf"),
            )
        confirmed = core.confirm_report(first, revision_one["reportId"], 2)
        self.assertEqual(confirmed["status"], "confirmed")
        later_manifest = manifest("5" * 64, "6" * 64)
        later_manifest["version"] = "2.0.0"
        core.database.save_prediction(
            prediction("PRD-REPORT-LATER", "ST-REPORT", later_manifest)
        )
        with patch(
            "inference_service.core.export_report_pdf",
            return_value=root / "report.pdf",
        ) as export_pdf:
            core.export_report(
                first,
                revision_one["reportId"],
                revision_two["revision"],
                str(root / "report.pdf"),
            )
        self.assertEqual(export_pdf.call_args.args[2]["predictionId"], "PRD-REPORT")
        with self.assertRaises(ValueError):
            core.save_report_draft(
                first,
                "ST-REPORT",
                "three",
                report_id=revision_one["reportId"],
            )

        core.database.save_study(study("ST-UNREVIEWED"))
        unreviewed = core.save_report_draft(first, "ST-UNREVIEWED", "draft")
        with self.assertRaisesRegex(ValueError, "doctor review"):
            core.confirm_report(first, unreviewed["reportId"], 1)

    def test_api_routes_use_database_assistant_context_not_client_values(self) -> None:
        root = case_directory("backend-api-context")
        app = create_app(root)
        process_headers = {"Authorization": f"Bearer {app.state.process_token}"}
        with TestClient(app) as client:
            setup = client.post(
                "/api/setup",
                headers=process_headers,
                json={"username": "admin", "password": "admin-password-123"},
            )
            admin_headers = {
                **process_headers,
                "X-Session-Token": setup.json()["sessionToken"],
            }
            created = client.post(
                "/api/users/doctors",
                headers=admin_headers,
                json={
                    "username": "doctor",
                    "password": "doctor-password-123",
                    "role": "doctor",
                },
            )
            doctor_id = created.json()["userId"]
            login = client.post(
                "/api/login",
                headers=process_headers,
                json={"username": "doctor", "password": "doctor-password-123"},
            )
            doctor_headers = {
                **process_headers,
                "X-Session-Token": login.json()["sessionToken"],
            }

            app.state.core.database.save_study(study("ST-CONTEXT"))
            stored_prediction = prediction(
                "PRD-CONTEXT", "ST-CONTEXT", cam_available=True
            )
            app.state.core.database.save_prediction(stored_prediction, cam_rows())
            review = app.state.core.save_review(
                UserSession(doctor_id, Role.DOCTOR),
                "ST-CONTEXT",
                "PRD-CONTEXT",
                {"Atelectasis": "denied"},
                "database note",
            )

            draft = client.post(
                "/api/reports/draft",
                headers=doctor_headers,
                json={"studyId": "ST-CONTEXT", "body": "first"},
            )
            self.assertEqual(draft.status_code, 200, draft.text)
            reports = client.get("/api/reports", headers=doctor_headers)
            history = client.get(
                f"/api/reports/{draft.json()['reportId']}/revisions",
                headers=doctor_headers,
            )
            self.assertEqual(reports.status_code, 200, reports.text)
            self.assertEqual(history.status_code, 200, history.text)
            self.assertEqual(len(reports.json()), 1)
            self.assertEqual(len(history.json()), 1)

            captured: list[dict[str, object]] = []

            def capture(_: object, payload: dict[str, object]) -> dict[str, object]:
                captured.append(payload)
                return {"draft": "database context", "cautions": []}

            with patch(
                "inference_service.assistant.OpenAIAssistant.report_draft",
                new=capture,
            ):
                rejected = client.post(
                    "/api/assistant/report",
                    headers={**doctor_headers, "X-Assistant-Key": "test-key"},
                    json={
                        "studyId": "ST-CONTEXT",
                        "reviewId": review["reviewId"],
                        "clinicianText": "doctor text",
                        "observations": [
                            {"label": "Atelectasis", "probability": 0.99}
                        ],
                        "review": {"decisions": {"Atelectasis": "confirmed"}},
                    },
                )
                response = client.post(
                    "/api/assistant/report",
                    headers={**doctor_headers, "X-Assistant-Key": "test-key"},
                    json={
                        "studyId": "ST-CONTEXT",
                        "reviewId": review["reviewId"],
                        "clinicianText": "doctor text",
                    },
                )
            self.assertEqual(rejected.status_code, 422, rejected.text)
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(len(captured), 1)
            self.assertEqual(captured[0]["observations"][0]["probability"], 0.25)
            self.assertEqual(
                captured[0]["review"],
                {
                    "decisions": {"Atelectasis": "denied"},
                    "notes": "database note",
                },
            )
            self.assertTrue(
                nested_keys(captured[0]).isdisjoint(
                    {
                        "studyid",
                        "predictionid",
                        "reviewid",
                        "path",
                        "image",
                        "pixels",
                    }
                )
            )
            with self.assertRaises(PermissionError):
                app.state.core.build_report_assistant_context(
                    UserSession("USR-OTHER", Role.DOCTOR),
                    "ST-CONTEXT",
                    "spoof",
                    review["reviewId"],
                )

            captured_experiments: list[dict[str, object]] = []

            def capture_experiment(
                _: object, payload: dict[str, object]
            ) -> dict[str, object]:
                captured_experiments.append(payload)
                return {"summary": "aggregate only", "limitations": []}

            with patch(
                "inference_service.assistant.OpenAIAssistant.experiment_summary",
                new=capture_experiment,
            ):
                experiment = client.post(
                    "/api/assistant/experiment",
                    headers={**doctor_headers, "X-Assistant-Key": "test-key"},
                    json={
                        "aggregateMetrics": {"macroAuroc": 0.81},
                        "perClassMetrics": {
                            "Atelectasis": {
                                "auroc": {
                                    "estimate": 0.82,
                                    "lower": 0.78,
                                    "upper": 0.86,
                                    "n": 500,
                                }
                            }
                        },
                        "experimentNotes": "locked aggregate result",
                    },
                )
            self.assertEqual(experiment.status_code, 200, experiment.text)
            self.assertEqual(
                captured_experiments[0]["perClassMetrics"]["Atelectasis"]["auroc"],
                {
                    "estimate": 0.82,
                    "lower": 0.78,
                    "upper": 0.86,
                    "n": 500,
                },
            )

            cam = client.get(
                "/api/predictions/PRD-CONTEXT/cams/0", headers=doctor_headers
            )
            self.assertEqual(cam.status_code, 200, cam.text)
            self.assertEqual(cam.json()["label"], CHEXPERT_LABELS[0])


if __name__ == "__main__":
    unittest.main()
