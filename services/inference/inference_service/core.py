from __future__ import annotations

import base64
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from cxr_research.artifacts import read_medexperiment
from cxr_research.labels import CHEXPERT_LABELS, validate_label_order

from .assistant import OpenAIAssistant
from .deid import deidentify_dicom, reencode_raster, sha256_file, temporary_study_id
from .model_registry import ModelRegistry
from .preprocess import inspect_file, load_grayscale, preprocess_for_model
from .reports import export_report_pdf
from .security import Role, UserSession, hash_password, require_permission, verify_password
from .storage import Database, new_id, utc_now


@dataclass(frozen=True)
class ServicePaths:
    root: Path
    staging: Path
    studies: Path
    models: Path
    experiments: Path
    database: Path

    @classmethod
    def under(cls, root: str | Path) -> "ServicePaths":
        base = Path(root).resolve()
        return cls(base, base / "staging", base / "studies", base / "models", base / "experiments", base / "platform.sqlite3")

    def create(self) -> None:
        for directory in (self.root, self.staging, self.studies, self.models, self.experiments):
            directory.mkdir(parents=True, exist_ok=True)


class PlatformCore:
    def __init__(self, root: str | Path, assistant: OpenAIAssistant | None = None) -> None:
        self.paths = ServicePaths.under(root)
        self.paths.create()
        for leftover in self.paths.staging.iterdir():
            if leftover.is_file():
                leftover.unlink(missing_ok=True)
        self._staged_hashes: dict[str, str] = {}
        self.database = Database(self.paths.database)
        self.models = ModelRegistry(self.paths.models)
        self.assistant = assistant or OpenAIAssistant()

    def setup_status(self) -> dict[str, Any]:
        return {"requiresSetup": self.database.user_count() == 0}

    @staticmethod
    def _public_study(study: Mapping[str, Any]) -> dict[str, Any]:
        private_fields = {"deidentifiedPath", "importHash", "derivedHash"}
        return {key: value for key, value in study.items() if key not in private_fields}

    def create_initial_admin(self, username: str, password: str) -> str:
        if self.database.user_count() != 0:
            raise PermissionError("initial administrator already exists")
        return self.database.create_user(username, Role.ADMIN.value, hash_password(password))

    def authenticate(self, username: str, password: str) -> UserSession:
        user = self.database.find_user(username)
        if user is None or not verify_password(str(user["password_hash"]), password):
            raise PermissionError("invalid username or password")
        return UserSession(str(user["user_id"]), Role(str(user["role"])))

    def create_doctor(self, actor: UserSession, username: str, password: str) -> str:
        require_permission(actor.role, "manage_users")
        return self.database.create_user(username, Role.DOCTOR.value, hash_password(password))

    def stage_imports(self, actor: UserSession, paths: Iterable[str]) -> list[dict[str, Any]]:
        require_permission(actor.role, "import")
        staged: list[dict[str, Any]] = []
        for value in paths:
            source = Path(value).resolve()
            if not source.is_file():
                staged.append({"sourceName": source.name, "accepted": False, "reasons": ["file_not_found"]})
                continue
            import_hash = sha256_file(source)
            if self.database.import_hash_exists(import_hash) or import_hash in self._staged_hashes.values():
                staged.append({"sourceName": source.name, "accepted": False, "reasons": ["duplicate_file"]})
                continue
            staging_id = new_id("TMP")
            target = self.paths.staging / f"{staging_id}{source.suffix.lower()}"
            shutil.copy2(source, target)
            validation = inspect_file(target)
            resolvable_reasons = {
                "adult_confirmation_required",
                "chest_radiograph_confirmation_required",
                "ap_pa_confirmation_required",
                "burned_in_text_review_required",
                "burned_in_text_requires_redaction_review",
            }
            hard_reasons = set(validation.reasons) - resolvable_reasons
            if hard_reasons:
                target.unlink(missing_ok=True)
                staged.append(
                    {
                        "sourceName": source.name,
                        "accepted": False,
                        "reasons": sorted(hard_reasons),
                    }
                )
                continue
            self._staged_hashes[staging_id] = import_hash
            staged.append(
                {
                    "stagingId": staging_id,
                    "sourceName": source.name,
                    "validation": asdict(validation),
                }
            )
        return staged

    def _staged_file(self, staging_id: str) -> Path:
        if not staging_id.startswith("TMP-") or any(character not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-" for character in staging_id.upper()):
            raise ValueError("invalid staging id")
        matches = list(self.paths.staging.glob(f"{staging_id}.*"))
        if len(matches) != 1:
            raise FileNotFoundError("staged import not found")
        path = matches[0].resolve()
        if self.paths.staging.resolve() not in path.parents:
            raise PermissionError("staged path escaped the staging directory")
        return path

    def cancel_import(self, actor: UserSession, staging_id: str) -> None:
        require_permission(actor.role, "import")
        path = self._staged_file(staging_id)
        path.unlink(missing_ok=True)
        self._staged_hashes.pop(staging_id, None)

    def commit_import(
        self,
        actor: UserSession,
        staging_id: str,
        *,
        adult_confirmed: bool,
        chest_confirmed: bool,
        view_confirmed: bool,
        view_position: str | None,
        burned_in_reviewed: bool,
        rectangles: Iterable[Mapping[str, int]],
    ) -> dict[str, Any]:
        require_permission(actor.role, "import")
        staged = self._staged_file(staging_id)
        rectangle_list = [dict(item) for item in rectangles]
        try:
            validation = inspect_file(
                staged,
                adult_confirmed=adult_confirmed,
                chest_confirmed=chest_confirmed,
                view_confirmed=view_confirmed,
                burned_in_reviewed=burned_in_reviewed,
            )
            if not validation.accepted:
                raise ValueError("文件未通过准入检查: " + ", ".join(validation.reasons))
            if validation.file_type == "raster" and view_position not in {"AP", "PA"}:
                raise ValueError("PNG/JPEG 必须由人工确认具体 AP 或 PA 体位")
            if validation.metadata.get("burnedInAnnotation") == "YES" and not rectangle_list:
                raise ValueError("标记有烧录文字的 DICOM 必须提供人工遮挡矩形")
            source_hash = sha256_file(staged)
            if self.database.import_hash_exists(source_hash):
                raise ValueError("duplicate_file")
            study_id = temporary_study_id()
            if validation.file_type == "dicom":
                destination = self.paths.studies / f"{study_id}.dcm"
                derived_hash = deidentify_dicom(staged, destination, rectangle_list)
                resolved_view = validation.view_position
            else:
                destination = self.paths.studies / f"{study_id}.png"
                derived_hash = reencode_raster(staged, destination, "PNG", rectangle_list)
                resolved_view = str(view_position)
            study = {
                "studyId": study_id,
                "createdAt": utc_now(),
                "modality": validation.modality,
                "bodyPart": "CHEST",
                "viewPosition": resolved_view,
                "adultConfirmed": True,
                "deidentifiedPath": str(destination),
                "importHash": source_hash,
                "derivedHash": derived_hash,
                "status": "ready",
            }
            self.database.save_study(study)
            return self._public_study(study)
        finally:
            staged.unlink(missing_ok=True)
            self._staged_hashes.pop(staging_id, None)

    def list_studies(self, actor: UserSession) -> list[dict[str, Any]]:
        require_permission(actor.role, "import")
        return [self._public_study(study) for study in self.database.list_studies()]

    @staticmethod
    def _pixel_payload(path: str | Path) -> dict[str, Any]:
        pixels = load_grayscale(path)
        height, width = pixels.shape
        max_side = max(height, width)
        if max_side > 1024:
            try:
                from PIL import Image

                scale = 1024.0 / max_side
                image = Image.fromarray(np.round(pixels * 255).astype(np.uint8), mode="L")
                image = image.resize((max(1, round(width * scale)), max(1, round(height * scale))), Image.Resampling.BILINEAR)
                pixels = np.asarray(image, dtype=np.float32) / 255.0
                height, width = pixels.shape
            except ImportError as exc:
                raise RuntimeError("Pillow is required for safe local previews") from exc
        encoded = base64.b64encode(np.round(pixels * 255).astype(np.uint8).tobytes()).decode("ascii")
        return {"width": int(width), "height": int(height), "pixelsBase64": encoded, "min": 0, "max": 255}

    def preview_staged(self, actor: UserSession, staging_id: str) -> dict[str, Any]:
        require_permission(actor.role, "import")
        return self._pixel_payload(self._staged_file(staging_id))

    def preview_study(self, actor: UserSession, study_id: str) -> dict[str, Any]:
        require_permission(actor.role, "review")
        study = self.database.get_study(study_id)
        if study is None:
            raise FileNotFoundError("study not found")
        return self._pixel_payload(study["deidentifiedPath"])

    def install_model(self, actor: UserSession, package_path: str) -> dict[str, Any]:
        require_permission(actor.role, "manage_models")
        return self.models.install(package_path)

    def infer(self, actor: UserSession, study_id: str) -> dict[str, Any]:
        require_permission(actor.role, "infer")
        study = self.database.get_study(study_id)
        if study is None:
            raise FileNotFoundError("study not found")
        if study["status"] not in {"ready", "reviewed"} or study["viewPosition"] not in {"AP", "PA"} or not study["adultConfirmed"]:
            raise ValueError("study is not eligible for inference")
        image = preprocess_for_model(study["deidentifiedPath"])
        logits_batch, features, manifest = self.models.predict(image)
        validate_label_order(tuple(manifest["labels"]))
        logits = np.asarray(logits_batch, dtype=float)[0]
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        prediction = {
            "predictionId": new_id("PRD"),
            "studyId": study_id,
            "modelId": manifest["modelId"],
            "modelVersion": manifest["version"],
            "modelSha256": manifest["modelSha256"],
            "manifestSha256": manifest["manifestSha256"],
            "createdAt": utc_now(),
            "probabilities": {label: float(probabilities[index]) for index, label in enumerate(CHEXPERT_LABELS)},
            "logits": {label: float(logits[index]) for index, label in enumerate(CHEXPERT_LABELS)},
            "thresholds": dict(manifest.get("thresholds", {})),
            "camAvailable": bool(np.asarray(features).size),
            "source": "onnx",
        }
        self.database.save_prediction(prediction)
        return prediction

    def save_review(
        self,
        actor: UserSession,
        study_id: str,
        prediction_id: str,
        decisions: Mapping[str, str],
        notes: str,
    ) -> dict[str, Any]:
        require_permission(actor.role, "review")
        if any(label not in CHEXPERT_LABELS for label in decisions):
            raise ValueError("review contains an unknown label")
        if any(decision not in {"confirmed", "denied", "uncertain"} for decision in decisions.values()):
            raise ValueError("invalid review decision")
        prediction = self.database.get_prediction(prediction_id)
        if prediction is None or prediction["studyId"] != study_id:
            raise ValueError("prediction does not belong to the study")
        review = {
            "reviewId": new_id("REV"),
            "studyId": study_id,
            "predictionId": prediction_id,
            "doctorId": actor.user_id,
            "createdAt": utc_now(),
            "decisions": dict(decisions),
            "notes": notes,
        }
        self.database.save_review(review)
        return review

    def save_report_draft(
        self,
        actor: UserSession,
        study_id: str,
        body: str,
        report_id: str | None = None,
        review_id: str | None = None,
    ) -> dict[str, Any]:
        require_permission(actor.role, "report")
        if self.database.get_study(study_id) is None:
            raise FileNotFoundError("study not found")
        if review_id is None:
            latest_review = self.database.latest_review(study_id, actor.user_id)
            review_id = latest_review["reviewId"] if latest_review else None
        elif (review := self.database.get_review(review_id)) is None:
            raise FileNotFoundError("review not found")
        elif review["studyId"] != study_id or review["doctorId"] != actor.user_id:
            raise PermissionError("report review must belong to this study and doctor")
        report_id = report_id or new_id("RPT")
        latest = self.database.latest_report(report_id)
        if latest and latest["studyId"] != study_id:
            raise ValueError("report revisions cannot change study")
        if latest and latest["status"] == "confirmed":
            raise ValueError("confirmed report is locked; create a new report instead")
        revision = int(latest["revision"]) + 1 if latest else 1
        report = {
            "reportId": report_id,
            "studyId": study_id,
            "revision": revision,
            "authorId": actor.user_id,
            "createdAt": utc_now(),
            "body": body,
            "status": "draft",
            "reviewId": review_id,
        }
        self.database.save_report_revision(report)
        return report

    def confirm_report(self, actor: UserSession, report_id: str, revision: int) -> dict[str, Any]:
        require_permission(actor.role, "confirm_report")
        report = self.database.get_report(report_id, revision)
        if report is None:
            raise FileNotFoundError("report revision not found")
        if report["authorId"] != actor.user_id:
            raise PermissionError("only the doctor who authored this revision may confirm it")
        self.database.confirm_report(report_id, revision)
        confirmed = self.database.get_report(report_id, revision)
        assert confirmed is not None
        return confirmed

    def export_report(self, actor: UserSession, report_id: str, revision: int, output_path: str) -> str:
        require_permission(actor.role, "report")
        report = self.database.get_report(report_id, revision)
        if report is None or report["status"] != "confirmed":
            raise ValueError("only a confirmed report revision can be exported")
        review = self.database.get_review(report["reviewId"]) if report.get("reviewId") else None
        if review is not None:
            prediction = self.database.get_prediction(review["predictionId"])
        else:
            prediction_row = self.database.latest_prediction(report["studyId"])
            prediction = (
                self.database.get_prediction(str(prediction_row["prediction_id"]))
                if prediction_row is not None
                else None
            )
        if prediction is None:
            raise ValueError("report has no AI prediction appendix")
        return str(export_report_pdf(output_path, report, prediction, review))

    def import_experiment(self, actor: UserSession, package_path: str) -> dict[str, Any]:
        require_permission(actor.role, "experiment")
        bundle = read_medexperiment(package_path)
        target = self.paths.experiments / f"{bundle['experimentId']}.medexperiment"
        shutil.copy2(package_path, target)
        return bundle
