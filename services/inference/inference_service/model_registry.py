from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cxr_research.artifacts import read_medmodel_manifest
from cxr_research.labels import CHEXPERT_LABELS


class ModelUnavailableError(RuntimeError):
    pass


@dataclass
class LoadedModel:
    package_path: Path
    manifest: dict[str, Any]
    session: Any

    def predict(self, image: Any) -> tuple[Any, Any | None]:
        if self.session is None:
            raise ModelUnavailableError("尚未安装模型，无法生成预测")
        input_name = self.session.get_inputs()[0].name
        output_names = [output.name for output in self.session.get_outputs()]
        outputs = self.session.run(None, {input_name: image})
        if "logits" not in output_names:
            raise RuntimeError("ONNX 模型必须包含 logits 输出")
        logits = outputs[output_names.index("logits")]
        cams = outputs[output_names.index("cams")] if "cams" in output_names else None
        if getattr(logits, "shape", (0, 0))[-1] != len(CHEXPERT_LABELS):
            raise RuntimeError("ONNX 模型输出不是完整 CheXpert 14 项")
        if cams is not None and (
            len(getattr(cams, "shape", ())) != 4
            or getattr(cams, "shape", (0, 0))[1] != len(CHEXPERT_LABELS)
        ):
            raise RuntimeError("ONNX cams 输出必须是 [N, 14, H, W]")
        return logits, cams


class ModelRegistry:
    def __init__(self, model_directory: str | Path) -> None:
        self.model_directory = Path(model_directory)
        self.model_directory.mkdir(parents=True, exist_ok=True)
        self._active: LoadedModel | None = None
        self._load_error: str | None = None
        self._restore_active()

    @property
    def pointer_path(self) -> Path:
        return self.model_directory / "active-model.json"

    @property
    def active(self) -> LoadedModel | None:
        return self._active

    def status(self) -> dict[str, Any]:
        if self._active is None:
            if self._load_error:
                return {
                    "installed": True,
                    "ready": False,
                    "message": "已安装模型无法通过完整性校验或加载",
                }
            return {"installed": False, "ready": False, "message": "尚未安装模型"}
        return {
            "installed": True,
            "ready": self._active.session is not None,
            "modelId": self._active.manifest["modelId"],
            "version": self._active.manifest["version"],
            "modelSha256": self._active.manifest["modelSha256"],
            "manifestSha256": self._active.manifest["manifestSha256"],
            "labels": self._active.manifest["labels"],
        }

    def install(self, package_path: str | Path, activate: bool = True) -> dict[str, Any]:
        package = Path(package_path)
        manifest = read_medmodel_manifest(package)
        destination = self.model_directory / f"{manifest['modelId']}-{manifest['version']}.medmodel"
        payload = package.read_bytes()
        temporary = destination.with_suffix(".medmodel.tmp")
        temporary.write_bytes(payload)
        try:
            loaded = self._load(temporary, manifest) if activate else None
            temporary.replace(destination)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        if loaded is not None:
            loaded.package_path = destination
            self._active = loaded
            self._load_error = None
            self._write_pointer(destination, manifest)
        return self.status()

    def load_existing(self, package_path: str | Path) -> dict[str, Any]:
        package = Path(package_path)
        manifest = read_medmodel_manifest(package)
        self._active = self._load(package, manifest)
        self._load_error = None
        if package.parent.resolve() == self.model_directory.resolve():
            self._write_pointer(package, manifest)
        return self.status()

    def _write_pointer(self, package: Path, manifest: dict[str, Any]) -> None:
        pointer = {
            "package": package.name,
            "modelSha256": manifest["modelSha256"],
            "manifestSha256": manifest["manifestSha256"],
        }
        temporary = self.pointer_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(pointer, ensure_ascii=True), encoding="utf-8")
        temporary.replace(self.pointer_path)

    def _restore_active(self) -> None:
        if not self.pointer_path.is_file():
            return
        try:
            pointer = json.loads(self.pointer_path.read_text(encoding="utf-8"))
            package_name = str(pointer["package"])
            package = (self.model_directory / package_name).resolve()
            if (
                package.parent != self.model_directory.resolve()
                or package.suffix.lower() != ".medmodel"
            ):
                raise ValueError("active model pointer contains an unsafe package path")
            manifest = read_medmodel_manifest(package)
            if (
                manifest["modelSha256"] != pointer.get("modelSha256")
                or manifest["manifestSha256"] != pointer.get("manifestSha256")
            ):
                raise ValueError("active model pointer hash mismatch")
            self._active = self._load(package, manifest)
        except (KeyError, OSError, TypeError, ValueError, RuntimeError, zipfile.BadZipFile):
            self._active = None
            self._load_error = "active model restore failed"

    def _load(self, package: Path, manifest: dict[str, Any]) -> LoadedModel:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ModelUnavailableError("onnxruntime 未安装，不能加载模型") from exc
        with zipfile.ZipFile(package, "r") as archive:
            model_bytes = archive.read("model.onnx")
        session = ort.InferenceSession(model_bytes, providers=["CPUExecutionProvider"])
        return LoadedModel(package, manifest, session)

    def predict(self, image: Any) -> tuple[Any, Any | None, dict[str, Any]]:
        if self._active is None:
            raise ModelUnavailableError("尚未安装模型，无法生成预测")
        logits, cams = self._active.predict(image)
        return logits, cams, self._active.manifest
