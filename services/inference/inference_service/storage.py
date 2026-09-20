from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from cxr_research.labels import CHEXPERT_LABELS


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
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            enabled = connection.execute("PRAGMA foreign_keys").fetchone()
            if enabled is None or int(enabled[0]) != 1:
                raise RuntimeError("SQLite foreign key enforcement could not be enabled")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self.connection() as connection:
            connection.executescript(
                """
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
                CREATE TABLE IF NOT EXISTS reports (
                    report_id TEXT PRIMARY KEY,
                    study_id TEXT NOT NULL REFERENCES studies(study_id),
                    owner_id TEXT NOT NULL REFERENCES users(user_id),
                    created_at TEXT NOT NULL
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

            self._initialize_report_tables(connection)
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS prediction_cams (
                    prediction_id TEXT NOT NULL REFERENCES predictions(prediction_id) ON DELETE CASCADE,
                    label TEXT NOT NULL,
                    width INTEGER NOT NULL CHECK (width > 0),
                    height INTEGER NOT NULL CHECK (height > 0),
                    pixels BLOB NOT NULL,
                    sha256 TEXT NOT NULL,
                    PRIMARY KEY (prediction_id, label)
                );
                CREATE INDEX IF NOT EXISTS ix_predictions_model_identity
                    ON predictions(study_id, model_sha256, manifest_sha256);
                CREATE INDEX IF NOT EXISTS ix_reviews_study_doctor
                    ON reviews(study_id, doctor_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS ix_reports_owner
                    ON reports(owner_id, created_at DESC);
                """
            )
            # Historical builds marked feature tensors or partial rows as CAMs.
            placeholders = ", ".join("?" for _ in CHEXPERT_LABELS)
            connection.execute(
                f"""UPDATE predictions SET cam_available = 0
                    WHERE cam_available <> 0
                      AND (
                          (SELECT COUNT(*) FROM prediction_cams
                           WHERE prediction_cams.prediction_id = predictions.prediction_id) <> ?
                          OR
                          (SELECT COUNT(*) FROM prediction_cams
                           WHERE prediction_cams.prediction_id = predictions.prediction_id
                             AND prediction_cams.label IN ({placeholders})) <> ?
                      )""",
                (len(CHEXPERT_LABELS), *CHEXPERT_LABELS, len(CHEXPERT_LABELS)),
            )
            violations = connection.execute("PRAGMA foreign_key_check").fetchall()
            if violations:
                table = str(violations[0]["table"])
                raise RuntimeError(f"SQLite foreign key check failed for table {table}")

    @staticmethod
    def _table_exists(connection: sqlite3.Connection, table: str) -> bool:
        return connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table,),
        ).fetchone() is not None

    def _initialize_report_tables(self, connection: sqlite3.Connection) -> None:
        if not self._table_exists(connection, "report_revisions"):
            self._create_report_revisions_table(connection, "report_revisions")
            return

        orphan = connection.execute(
            """SELECT rr.report_id
               FROM report_revisions rr
               LEFT JOIN studies s ON s.study_id = rr.study_id
               LEFT JOIN users u ON u.user_id = rr.author_id
               WHERE s.study_id IS NULL OR u.user_id IS NULL
               LIMIT 1"""
        ).fetchone()
        if orphan is not None:
            raise RuntimeError(
                f"cannot migrate orphaned report revision {orphan['report_id']}"
            )

        connection.execute(
            """INSERT OR IGNORE INTO reports(report_id, study_id, owner_id, created_at)
               SELECT rr.report_id, rr.study_id, rr.author_id, rr.created_at
               FROM report_revisions rr
               JOIN (
                   SELECT report_id, MIN(revision) AS first_revision
                   FROM report_revisions
                   GROUP BY report_id
               ) first
                 ON first.report_id = rr.report_id
                AND first.first_revision = rr.revision"""
        )
        mismatch = connection.execute(
            """SELECT reports.report_id
               FROM reports
               JOIN report_revisions rr ON rr.report_id = reports.report_id
               WHERE reports.study_id <> rr.study_id
                  OR reports.owner_id <> rr.author_id
               LIMIT 1"""
        ).fetchone()
        if mismatch is not None:
            raise RuntimeError(
                f"report ownership migration conflict for {mismatch['report_id']}"
            )

        review_mismatch = connection.execute(
            """SELECT rr.report_id, rr.revision
               FROM report_revisions rr
               JOIN reviews rv ON rv.review_id = rr.review_id
               LEFT JOIN predictions p ON p.prediction_id = rv.prediction_id
               WHERE rr.review_id IS NOT NULL
                 AND (
                     rv.study_id <> rr.study_id
                     OR rv.doctor_id <> rr.author_id
                     OR p.prediction_id IS NULL
                     OR p.study_id <> rr.study_id
                 )
               LIMIT 1"""
        ).fetchone()
        if review_mismatch is not None:
            raise RuntimeError(
                "report review migration conflict for "
                f"{review_mismatch['report_id']} revision "
                f"{review_mismatch['revision']}"
            )

        foreign_tables = {
            str(row["table"])
            for row in connection.execute(
                "PRAGMA foreign_key_list(report_revisions)"
            ).fetchall()
        }
        if {"reports", "reviews"}.issubset(foreign_tables):
            return

        old_count = int(
            connection.execute(
                "SELECT COUNT(*) AS count FROM report_revisions"
            ).fetchone()["count"]
        )
        connection.execute("DROP TABLE IF EXISTS report_revisions_migrated")
        self._create_report_revisions_table(
            connection, "report_revisions_migrated"
        )
        connection.execute(
            """INSERT INTO report_revisions_migrated(
                   report_id, study_id, revision, author_id, created_at,
                   body, status, confirmed_at, review_id
               )
               SELECT rr.report_id, rr.study_id, rr.revision, rr.author_id,
                      rr.created_at, rr.body, rr.status, rr.confirmed_at,
                      CASE WHEN reviews.review_id IS NULL THEN NULL ELSE rr.review_id END
               FROM report_revisions rr
               LEFT JOIN reviews ON reviews.review_id = rr.review_id"""
        )
        new_count = int(
            connection.execute(
                "SELECT COUNT(*) AS count FROM report_revisions_migrated"
            ).fetchone()["count"]
        )
        if new_count != old_count:
            raise RuntimeError("report revision migration row count mismatch")
        connection.execute("DROP TABLE report_revisions")
        connection.execute(
            "ALTER TABLE report_revisions_migrated RENAME TO report_revisions"
        )

    @staticmethod
    def _create_report_revisions_table(
        connection: sqlite3.Connection, table: str
    ) -> None:
        if table not in {"report_revisions", "report_revisions_migrated"}:
            raise ValueError("invalid report revision table name")
        connection.execute(
            f"""CREATE TABLE {table} (
                report_id TEXT NOT NULL REFERENCES reports(report_id),
                study_id TEXT NOT NULL REFERENCES studies(study_id),
                revision INTEGER NOT NULL,
                author_id TEXT NOT NULL REFERENCES users(user_id),
                created_at TEXT NOT NULL,
                body TEXT NOT NULL,
                status TEXT NOT NULL CHECK (status IN ('draft', 'confirmed')),
                confirmed_at TEXT,
                review_id TEXT REFERENCES reviews(review_id),
                PRIMARY KEY (report_id, revision)
            )"""
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

    def create_initial_admin(self, username: str, password_hash: str) -> str:
        user_id = new_id("USR")
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            count = int(
                connection.execute(
                    "SELECT COUNT(*) AS count FROM users"
                ).fetchone()["count"]
            )
            if count != 0:
                raise PermissionError("initial administrator already exists")
            connection.execute(
                "INSERT INTO users(user_id, username, role, password_hash, created_at) VALUES (?, ?, 'admin', ?, ?)",
                (user_id, username, password_hash, utc_now()),
            )
        return user_id

    def user_count(self) -> int:
        with self.connection() as connection:
            row = connection.execute("SELECT COUNT(*) AS count FROM users").fetchone()
            return int(row["count"])

    def find_user(self, username: str) -> sqlite3.Row | None:
        with self.connection() as connection:
            return connection.execute(
                "SELECT * FROM users WHERE username = ?", (username,)
            ).fetchone()

    def list_users(self) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT user_id, username, role, created_at FROM users ORDER BY created_at"
            ).fetchall()
            return [dict(row) for row in rows]

    def save_study(self, study: dict[str, Any]) -> None:
        with self.connection() as connection:
            connection.execute(
                """INSERT INTO studies(study_id, created_at, modality, body_part, view_position,
                   adult_confirmed, deidentified_path, import_hash, status)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    study["studyId"],
                    study["createdAt"],
                    study["modality"],
                    study["bodyPart"],
                    study["viewPosition"],
                    int(study["adultConfirmed"]),
                    study["deidentifiedPath"],
                    study["importHash"],
                    study["status"],
                ),
            )

    def import_hash_exists(self, import_hash: str) -> bool:
        with self.connection() as connection:
            return connection.execute(
                "SELECT 1 FROM studies WHERE import_hash = ? LIMIT 1",
                (import_hash,),
            ).fetchone() is not None

    def get_study(self, study_id: str) -> dict[str, Any] | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM studies WHERE study_id = ?", (study_id,)
            ).fetchone()
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
            ids = [
                row["study_id"]
                for row in connection.execute(
                    "SELECT study_id FROM studies ORDER BY created_at DESC"
                ).fetchall()
            ]
        return [
            study
            for study_id in ids
            if (study := self.get_study(study_id)) is not None
        ]

    @staticmethod
    def _prediction_from_row(row: sqlite3.Row | Mapping[str, Any]) -> dict[str, Any]:
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

    @staticmethod
    def _matching_prediction(
        connection: sqlite3.Connection,
        study_id: str,
        model_id: str,
        model_version: str,
        model_sha256: str,
        manifest_sha256: str,
    ) -> sqlite3.Row | None:
        version_match = connection.execute(
            """SELECT * FROM predictions
               WHERE study_id = ? AND model_id = ? AND model_version = ?
               LIMIT 1""",
            (study_id, model_id, model_version),
        ).fetchone()
        if version_match is not None:
            if (
                str(version_match["model_sha256"]) == model_sha256
                and str(version_match["manifest_sha256"]) == manifest_sha256
            ):
                return version_match
            raise ValueError(
                "model identity conflict: the same model id/version has different hashes"
            )
        return connection.execute(
            """SELECT * FROM predictions
               WHERE study_id = ? AND model_sha256 = ? AND manifest_sha256 = ?
               ORDER BY created_at LIMIT 1""",
            (study_id, model_sha256, manifest_sha256),
        ).fetchone()

    def reusable_prediction(
        self,
        study_id: str,
        model_id: str,
        model_version: str,
        model_sha256: str,
        manifest_sha256: str,
    ) -> dict[str, Any] | None:
        with self.connection() as connection:
            row = self._matching_prediction(
                connection,
                study_id,
                model_id,
                model_version,
                model_sha256,
                manifest_sha256,
            )
            return self._prediction_from_row(row) if row is not None else None

    def save_prediction(
        self,
        prediction: dict[str, Any],
        cams: Iterable[Mapping[str, Any]] = (),
    ) -> dict[str, Any]:
        cam_rows = list(cams)
        if len(cam_rows) not in {0, 14}:
            raise ValueError("a prediction must contain either zero or fourteen CAMs")
        if cam_rows and {str(cam["label"]) for cam in cam_rows} != set(CHEXPERT_LABELS):
            raise ValueError("prediction CAM labels must match the CheXpert dictionary")
        if bool(prediction["camAvailable"]) != (len(cam_rows) == 14):
            raise ValueError("prediction CAM availability does not match stored CAMs")
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = self._matching_prediction(
                connection,
                prediction["studyId"],
                prediction["modelId"],
                prediction["modelVersion"],
                prediction["modelSha256"],
                prediction["manifestSha256"],
            )
            if existing is not None:
                return self._prediction_from_row(existing)
            try:
                connection.execute(
                    """INSERT INTO predictions(
                           prediction_id, study_id, model_id, model_version, created_at,
                           model_sha256, manifest_sha256, probabilities_json, logits_json,
                           thresholds_json, cam_available, source
                       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
            except sqlite3.IntegrityError as exc:
                existing = self._matching_prediction(
                    connection,
                    prediction["studyId"],
                    prediction["modelId"],
                    prediction["modelVersion"],
                    prediction["modelSha256"],
                    prediction["manifestSha256"],
                )
                if existing is not None:
                    return self._prediction_from_row(existing)
                raise ValueError("prediction identifier conflict") from exc

            for cam in cam_rows:
                pixels = bytes(cam["pixels"])
                width = int(cam["width"])
                height = int(cam["height"])
                if width <= 0 or height <= 0 or len(pixels) != width * height:
                    raise ValueError("invalid CAM dimensions")
                digest = hashlib.sha256(pixels).hexdigest()
                if digest != str(cam["sha256"]):
                    raise ValueError("CAM SHA-256 mismatch")
                connection.execute(
                    """INSERT INTO prediction_cams(
                           prediction_id, label, width, height, pixels, sha256
                       ) VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        prediction["predictionId"],
                        str(cam["label"]),
                        width,
                        height,
                        sqlite3.Binary(pixels),
                        digest,
                    ),
                )
            return dict(prediction)

    def latest_prediction(self, study_id: str) -> sqlite3.Row | None:
        with self.connection() as connection:
            return connection.execute(
                """SELECT * FROM predictions WHERE study_id = ?
                   ORDER BY created_at DESC, prediction_id DESC LIMIT 1""",
                (study_id,),
            ).fetchone()

    def get_prediction(self, prediction_id: str) -> dict[str, Any] | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM predictions WHERE prediction_id = ?",
                (prediction_id,),
            ).fetchone()
            return self._prediction_from_row(row) if row is not None else None

    def get_prediction_cam(
        self, prediction_id: str, label: str
    ) -> dict[str, Any] | None:
        with self.connection() as connection:
            row = connection.execute(
                """SELECT prediction_id, label, width, height, pixels, sha256
                   FROM prediction_cams
                   WHERE prediction_id = ? AND label = ?""",
                (prediction_id, label),
            ).fetchone()
            if row is None:
                return None
            return {
                "predictionId": str(row["prediction_id"]),
                "label": str(row["label"]),
                "width": int(row["width"]),
                "height": int(row["height"]),
                "pixels": bytes(row["pixels"]),
                "sha256": str(row["sha256"]),
            }

    def list_prediction_cams(self, prediction_id: str) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                """SELECT prediction_id, label, width, height, pixels, sha256
                   FROM prediction_cams WHERE prediction_id = ? ORDER BY label""",
                (prediction_id,),
            ).fetchall()
            return [
                {
                    "predictionId": str(row["prediction_id"]),
                    "label": str(row["label"]),
                    "width": int(row["width"]),
                    "height": int(row["height"]),
                    "pixels": bytes(row["pixels"]),
                    "sha256": str(row["sha256"]),
                }
                for row in rows
            ]

    def save_review(self, review: dict[str, Any]) -> None:
        with self.connection() as connection:
            connection.execute(
                """INSERT INTO reviews(
                       review_id, study_id, prediction_id, doctor_id,
                       created_at, decisions_json, notes
                   ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    review["reviewId"],
                    review["studyId"],
                    review["predictionId"],
                    review["doctorId"],
                    review["createdAt"],
                    json.dumps(review["decisions"], ensure_ascii=False),
                    review["notes"],
                ),
            )

    def get_review(self, review_id: str) -> dict[str, Any] | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM reviews WHERE review_id = ?", (review_id,)
            ).fetchone()
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

    def latest_review(
        self, study_id: str, doctor_id: str | None = None
    ) -> dict[str, Any] | None:
        query = "SELECT review_id FROM reviews WHERE study_id = ?"
        parameters: tuple[Any, ...] = (study_id,)
        if doctor_id is not None:
            query += " AND doctor_id = ?"
            parameters = (study_id, doctor_id)
        query += " ORDER BY created_at DESC, review_id DESC LIMIT 1"
        with self.connection() as connection:
            row = connection.execute(query, parameters).fetchone()
        return self.get_review(str(row["review_id"])) if row is not None else None

    @staticmethod
    def _report_from_row(row: sqlite3.Row | Mapping[str, Any]) -> dict[str, Any]:
        item = dict(row)
        return {
            "reportId": item["report_id"],
            "studyId": item["study_id"],
            "revision": int(item["revision"]),
            "authorId": item["author_id"],
            "createdAt": item["created_at"],
            "body": item["body"],
            "status": item["status"],
            "confirmedAt": item["confirmed_at"],
            "reviewId": item["review_id"],
        }

    def get_report_record(self, report_id: str) -> dict[str, Any] | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM reports WHERE report_id = ?", (report_id,)
            ).fetchone()
            if row is None:
                return None
            return {
                "reportId": str(row["report_id"]),
                "studyId": str(row["study_id"]),
                "ownerId": str(row["owner_id"]),
                "createdAt": str(row["created_at"]),
            }

    def append_report_revision(
        self,
        report_id: str,
        study_id: str,
        owner_id: str,
        body: str,
        review_id: str | None = None,
        created_at: str | None = None,
    ) -> dict[str, Any]:
        report = {
            "reportId": report_id,
            "studyId": study_id,
            "authorId": owner_id,
            "createdAt": created_at or utc_now(),
            "body": body,
            "status": "draft",
            "reviewId": review_id,
        }
        return self._insert_report_revision(report, requested_revision=None)

    def save_report_revision(self, report: dict[str, Any]) -> dict[str, Any]:
        if report["status"] == "confirmed":
            raise ValueError("use confirm_report to lock a report revision")
        return self._insert_report_revision(
            report, requested_revision=int(report["revision"])
        )

    def _insert_report_revision(
        self, report: Mapping[str, Any], requested_revision: int | None
    ) -> dict[str, Any]:
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            header = connection.execute(
                "SELECT * FROM reports WHERE report_id = ?",
                (report["reportId"],),
            ).fetchone()
            if header is None:
                if requested_revision not in {None, 1}:
                    raise ValueError("the first report revision must be revision 1")
                connection.execute(
                    """INSERT INTO reports(report_id, study_id, owner_id, created_at)
                       VALUES (?, ?, ?, ?)""",
                    (
                        report["reportId"],
                        report["studyId"],
                        report["authorId"],
                        report["createdAt"],
                    ),
                )
                revision = 1
            else:
                if str(header["owner_id"]) != str(report["authorId"]):
                    raise PermissionError("report belongs to another doctor")
                if str(header["study_id"]) != str(report["studyId"]):
                    raise ValueError("report revisions cannot change study")
                latest = connection.execute(
                    """SELECT revision, status FROM report_revisions
                       WHERE report_id = ? ORDER BY revision DESC LIMIT 1""",
                    (report["reportId"],),
                ).fetchone()
                if latest is None:
                    raise RuntimeError("report header has no revisions")
                if str(latest["status"]) == "confirmed":
                    raise ValueError(
                        "confirmed report is locked; create a new report instead"
                    )
                revision = int(latest["revision"]) + 1
                if requested_revision is not None and requested_revision != revision:
                    raise ValueError(
                        "report revisions are immutable and cannot be overwritten"
                    )
            connection.execute(
                """INSERT INTO report_revisions(
                       report_id, study_id, revision, author_id, created_at,
                       body, status, confirmed_at, review_id
                   ) VALUES (?, ?, ?, ?, ?, ?, 'draft', NULL, ?)""",
                (
                    report["reportId"],
                    report["studyId"],
                    revision,
                    report["authorId"],
                    report["createdAt"],
                    report["body"],
                    report.get("reviewId"),
                ),
            )
            row = connection.execute(
                """SELECT * FROM report_revisions
                   WHERE report_id = ? AND revision = ?""",
                (report["reportId"], revision),
            ).fetchone()
            assert row is not None
            return self._report_from_row(row)

    def confirm_report(
        self,
        report_id: str,
        revision: int,
        confirmed_at: str | None = None,
        owner_id: str | None = None,
    ) -> None:
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            header = connection.execute(
                "SELECT owner_id FROM reports WHERE report_id = ?", (report_id,)
            ).fetchone()
            if header is None:
                raise ValueError("report does not exist")
            if owner_id is not None and str(header["owner_id"]) != owner_id:
                raise PermissionError("report belongs to another doctor")
            latest = connection.execute(
                """SELECT revision, status FROM report_revisions
                   WHERE report_id = ? ORDER BY revision DESC LIMIT 1""",
                (report_id,),
            ).fetchone()
            if latest is None or int(latest["revision"]) != revision:
                raise ValueError("only the latest report revision may be confirmed")
            if str(latest["status"]) != "draft":
                raise ValueError("report revision does not exist or is already locked")
            cursor = connection.execute(
                """UPDATE report_revisions
                   SET status = 'confirmed', confirmed_at = ?
                   WHERE report_id = ? AND revision = ? AND status = 'draft'""",
                (confirmed_at or utc_now(), report_id, revision),
            )
            if cursor.rowcount != 1:
                raise ValueError("report revision does not exist or is already locked")

    def list_reports(self, owner_id: str) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                """SELECT rr.*
                   FROM reports r
                   JOIN report_revisions rr ON rr.report_id = r.report_id
                   WHERE r.owner_id = ?
                     AND rr.revision = (
                         SELECT MAX(latest.revision)
                         FROM report_revisions latest
                         WHERE latest.report_id = r.report_id
                     )
                   ORDER BY rr.created_at DESC, rr.report_id""",
                (owner_id,),
            ).fetchall()
            return [self._report_from_row(row) for row in rows]

    def list_report_revisions(
        self, report_id: str, owner_id: str
    ) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                """SELECT rr.*
                   FROM report_revisions rr
                   JOIN reports r ON r.report_id = rr.report_id
                   WHERE rr.report_id = ? AND r.owner_id = ?
                   ORDER BY rr.revision DESC""",
                (report_id, owner_id),
            ).fetchall()
            return [self._report_from_row(row) for row in rows]

    def latest_report(
        self, report_id: str, owner_id: str | None = None
    ) -> dict[str, Any] | None:
        query = """SELECT rr.* FROM report_revisions rr
                   JOIN reports r ON r.report_id = rr.report_id
                   WHERE rr.report_id = ?"""
        parameters: tuple[Any, ...] = (report_id,)
        if owner_id is not None:
            query += " AND r.owner_id = ?"
            parameters = (report_id, owner_id)
        query += " ORDER BY rr.revision DESC LIMIT 1"
        with self.connection() as connection:
            row = connection.execute(query, parameters).fetchone()
            return self._report_from_row(row) if row is not None else None

    def get_report(
        self,
        report_id: str,
        revision: int,
        owner_id: str | None = None,
    ) -> dict[str, Any] | None:
        query = """SELECT rr.* FROM report_revisions rr
                   JOIN reports r ON r.report_id = rr.report_id
                   WHERE rr.report_id = ? AND rr.revision = ?"""
        parameters: tuple[Any, ...] = (report_id, revision)
        if owner_id is not None:
            query += " AND r.owner_id = ?"
            parameters = (report_id, revision, owner_id)
        with self.connection() as connection:
            row = connection.execute(query, parameters).fetchone()
            return self._report_from_row(row) if row is not None else None
