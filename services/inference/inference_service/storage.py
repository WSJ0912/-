from __future__ import annotations

import json
import secrets
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}-{secrets.token_hex(10).upper()}"


class Database:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self.connection() as connection:
            connection.executescript(
                """
                PRAGMA foreign_keys = ON;
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    username TEXT NOT NULL UNIQUE,
                    role TEXT NOT NULL CHECK (role IN ('admin', 'doctor')),
                    password_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS studies (
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
                CREATE TABLE IF NOT EXISTS predictions (
                    prediction_id TEXT PRIMARY KEY,
                    study_id TEXT NOT NULL REFERENCES studies(study_id),
                    model_id TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    model_sha256 TEXT NOT NULL,
                    manifest_sha256 TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    probabilities_json TEXT NOT NULL,
                    logits_json TEXT NOT NULL,
                    thresholds_json TEXT NOT NULL,
                    cam_available INTEGER NOT NULL,
                    source TEXT NOT NULL,
                    UNIQUE(study_id, model_id, model_version)
                );
                CREATE TABLE IF NOT EXISTS reviews (
                    review_id TEXT PRIMARY KEY,
                    study_id TEXT NOT NULL REFERENCES studies(study_id),
                    prediction_id TEXT NOT NULL REFERENCES predictions(prediction_id),
                    doctor_id TEXT NOT NULL REFERENCES users(user_id),
                    created_at TEXT NOT NULL,
                    decisions_json TEXT NOT NULL,
                    notes TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS report_revisions (
                    report_id TEXT NOT NULL,
                    study_id TEXT NOT NULL REFERENCES studies(study_id),
                    revision INTEGER NOT NULL,
                    author_id TEXT NOT NULL REFERENCES users(user_id),
                    created_at TEXT NOT NULL,
                    body TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('draft', 'confirmed')),
                    confirmed_at TEXT,
                    review_id TEXT,
                    PRIMARY KEY (report_id, revision)
                );
                """
            )
            prediction_columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(predictions)").fetchall()
            }
            if "model_sha256" not in prediction_columns:
                connection.execute(
                    "ALTER TABLE predictions ADD COLUMN model_sha256 TEXT NOT NULL DEFAULT ''"
                )
            if "manifest_sha256" not in prediction_columns:
                connection.execute(
                    "ALTER TABLE predictions ADD COLUMN manifest_sha256 TEXT NOT NULL DEFAULT ''"
                )

    def create_user(self, username: str, role: str, password_hash: str) -> str:
        if role not in {"admin", "doctor"}:
            raise ValueError("invalid user role")
        user_id = new_id("USR")
        with self.connection() as connection:
            connection.execute(
                "INSERT INTO users(user_id, username, role, password_hash, created_at) VALUES (?, ?, ?, ?, ?)",
                (user_id, username, role, password_hash, utc_now()),
            )
        return user_id

    def user_count(self) -> int:
        with self.connection() as connection:
            row = connection.execute("SELECT COUNT(*) AS count FROM users").fetchone()
            return int(row["count"])

    def find_user(self, username: str) -> sqlite3.Row | None:
        with self.connection() as connection:
            return connection.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()

    def list_users(self) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute("SELECT user_id, username, role, created_at FROM users ORDER BY created_at").fetchall()
            return [dict(row) for row in rows]

    def save_study(self, study: dict[str, Any]) -> None:
        with self.connection() as connection:
            connection.execute(
                """INSERT INTO studies(study_id, created_at, modality, body_part, view_position,
                   adult_confirmed, deidentified_path, import_hash, status)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (study["studyId"], study["createdAt"], study["modality"], study["bodyPart"], study["viewPosition"], int(study["adultConfirmed"]), study["deidentifiedPath"], study["importHash"], study["status"]),
            )

    def import_hash_exists(self, import_hash: str) -> bool:
        with self.connection() as connection:
            return connection.execute("SELECT 1 FROM studies WHERE import_hash = ? LIMIT 1", (import_hash,)).fetchone() is not None

    def get_study(self, study_id: str) -> dict[str, Any] | None:
        with self.connection() as connection:
            row = connection.execute("SELECT * FROM studies WHERE study_id = ?", (study_id,)).fetchone()
            if row is None:
                return None
            item = dict(row)
            return {
                "studyId": item["study_id"],
                "createdAt": item["created_at"],
                "modality": item["modality"],
                "bodyPart": item["body_part"],
                "viewPosition": item["view_position"],
                "adultConfirmed": bool(item["adult_confirmed"]),
                "deidentifiedPath": item["deidentified_path"],
                "importHash": item["import_hash"],
                "status": item["status"],
            }

    def list_studies(self) -> list[dict[str, Any]]:
        with self.connection() as connection:
            ids = [row["study_id"] for row in connection.execute("SELECT study_id FROM studies ORDER BY created_at DESC").fetchall()]
        return [study for study_id in ids if (study := self.get_study(study_id)) is not None]

    def save_prediction(self, prediction: dict[str, Any]) -> None:
        with self.connection() as connection:
            connection.execute(
                """INSERT INTO predictions(prediction_id, study_id, model_id, model_version, created_at,
                   model_sha256, manifest_sha256, probabilities_json, logits_json, thresholds_json,
                   cam_available, source)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    prediction["predictionId"],
                    prediction["studyId"],
                    prediction["modelId"],
                    prediction["modelVersion"],
                    prediction["createdAt"],
                    prediction["modelSha256"],
                    prediction["manifestSha256"],
                    json.dumps(prediction["probabilities"], ensure_ascii=False),
                    json.dumps(prediction["logits"], ensure_ascii=False),
                    json.dumps(prediction["thresholds"], ensure_ascii=False),
                    int(prediction["camAvailable"]),
                    prediction["source"],
                ),
            )

    def save_review(self, review: dict[str, Any]) -> None:
        with self.connection() as connection:
            connection.execute(
                "INSERT INTO reviews(review_id, study_id, prediction_id, doctor_id, created_at, decisions_json, notes) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (review["reviewId"], review["studyId"], review["predictionId"], review["doctorId"], review["createdAt"], json.dumps(review["decisions"], ensure_ascii=False), review["notes"]),
            )

    def get_review(self, review_id: str) -> dict[str, Any] | None:
        with self.connection() as connection:
            row = connection.execute("SELECT * FROM reviews WHERE review_id = ?", (review_id,)).fetchone()
            if row is None:
                return None
            item = dict(row)
            return {
                "reviewId": item["review_id"],
                "studyId": item["study_id"],
                "predictionId": item["prediction_id"],
                "doctorId": item["doctor_id"],
                "createdAt": item["created_at"],
                "decisions": json.loads(item["decisions_json"]),
                "notes": item["notes"],
            }

    def latest_review(self, study_id: str, doctor_id: str | None = None) -> dict[str, Any] | None:
        query = "SELECT review_id FROM reviews WHERE study_id = ?"
        parameters: tuple[Any, ...] = (study_id,)
        if doctor_id is not None:
            query += " AND doctor_id = ?"
            parameters = (study_id, doctor_id)
        query += " ORDER BY created_at DESC LIMIT 1"
        with self.connection() as connection:
            row = connection.execute(query, parameters).fetchone()
        return self.get_review(str(row["review_id"])) if row is not None else None

    def save_report_revision(self, report: dict[str, Any]) -> None:
        with self.connection() as connection:
            if report["status"] == "confirmed":
                raise ValueError("use confirm_report to lock a report revision")
            existing = connection.execute(
                "SELECT status FROM report_revisions WHERE report_id = ? AND revision = ?",
                (report["reportId"], report["revision"]),
            ).fetchone()
            if existing is not None:
                raise ValueError("report revisions are immutable and cannot be overwritten")
            connection.execute(
                "INSERT INTO report_revisions(report_id, study_id, revision, author_id, created_at, body, status, confirmed_at, review_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (report["reportId"], report["studyId"], report["revision"], report["authorId"], report["createdAt"], report["body"], "draft", None, report.get("reviewId")),
            )

    def confirm_report(self, report_id: str, revision: int, confirmed_at: str | None = None) -> None:
        with self.connection() as connection:
            cursor = connection.execute(
                "UPDATE report_revisions SET status = 'confirmed', confirmed_at = ? WHERE report_id = ? AND revision = ? AND status = 'draft'",
                (confirmed_at or utc_now(), report_id, revision),
            )
            if cursor.rowcount != 1:
                raise ValueError("report revision does not exist or is already locked")

    def latest_prediction(self, study_id: str) -> sqlite3.Row | None:
        with self.connection() as connection:
            return connection.execute("SELECT * FROM predictions WHERE study_id = ? ORDER BY created_at DESC LIMIT 1", (study_id,)).fetchone()

    def get_prediction(self, prediction_id: str) -> dict[str, Any] | None:
        with self.connection() as connection:
            row = connection.execute("SELECT * FROM predictions WHERE prediction_id = ?", (prediction_id,)).fetchone()
            if row is None:
                return None
            item = dict(row)
            return {
                "predictionId": item["prediction_id"],
                "studyId": item["study_id"],
                "modelId": item["model_id"],
                "modelVersion": item["model_version"],
                "modelSha256": item["model_sha256"],
                "manifestSha256": item["manifest_sha256"],
                "createdAt": item["created_at"],
                "probabilities": json.loads(item["probabilities_json"]),
                "logits": json.loads(item["logits_json"]),
                "thresholds": json.loads(item["thresholds_json"]),
                "camAvailable": bool(item["cam_available"]),
                "source": item["source"],
            }

    def latest_report(self, report_id: str) -> dict[str, Any] | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM report_revisions WHERE report_id = ? ORDER BY revision DESC LIMIT 1", (report_id,)
            ).fetchone()
            if row is None:
                return None
            item = dict(row)
            return {
                "reportId": item["report_id"],
                "studyId": item["study_id"],
                "revision": item["revision"],
                "authorId": item["author_id"],
                "createdAt": item["created_at"],
                "body": item["body"],
                "status": item["status"],
                "confirmedAt": item["confirmed_at"],
                "reviewId": item["review_id"],
            }

    def get_report(self, report_id: str, revision: int) -> dict[str, Any] | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM report_revisions WHERE report_id = ? AND revision = ?", (report_id, revision)
            ).fetchone()
            if row is None:
                return None
            item = dict(row)
            return {
                "reportId": item["report_id"],
                "studyId": item["study_id"],
                "revision": item["revision"],
                "authorId": item["author_id"],
                "createdAt": item["created_at"],
                "body": item["body"],
                "status": item["status"],
                "confirmedAt": item["confirmed_at"],
                "reviewId": item["review_id"],
            }
