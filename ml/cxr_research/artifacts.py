from __future__ import annotations

import hashlib
import json
import re
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from jsonschema import Draft202012Validator, FormatChecker

from .labels import CHEXPERT_LABELS, CONTRACT_SCHEMAS, validate_label_order

SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
_FORMAT_CHECKER = FormatChecker()
_CONTRACT_VALIDATORS: dict[str, Draft202012Validator] = {}
for _schema_name, _schema in CONTRACT_SCHEMAS.items():
    Draft202012Validator.check_schema(_schema)
    _CONTRACT_VALIDATORS[_schema_name] = Draft202012Validator(
        _schema, format_checker=_FORMAT_CHECKER
    )


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def model_manifest_sha256(manifest: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in manifest.items() if key != "manifestSha256"}
    return sha256_bytes(canonical_json(payload))


def _safe_archive_name(name: str) -> str:
    normalized = name.replace("\\", "/")
    candidate = PurePosixPath(normalized)
    if (
        not name
        or "\x00" in name
        or name.endswith(("/", "\\"))
        or candidate.is_absolute()
        or not candidate.parts
        or ".." in candidate.parts
        or ":" in candidate.parts[0]
    ):
        raise ValueError(f"unsafe archive path: {name!r}")
    return normalized


def _validate_sha256(value: Any, field: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return digest


def _validate_identifier(value: Any, field: str) -> str:
    identifier = str(value)
    if not SAFE_IDENTIFIER.fullmatch(identifier):
        raise ValueError(f"{field} must be a safe 1-80 character identifier")
    return identifier


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"non-finite JSON number is not allowed: {value}")


def _load_json(value: str | bytes) -> Any:
    return json.loads(value, parse_constant=_reject_nonfinite_json)


def _validation_path(parts: Any) -> str:
    path = "$"
    for part in parts:
        path += f"[{part}]" if isinstance(part, int) else f".{part}"
    return path


def _validate_contract(schema_name: str, value: Any) -> None:
    validator = _CONTRACT_VALIDATORS[schema_name]
    errors = sorted(
        validator.iter_errors(value),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if errors:
        first = errors[0]
        raise ValueError(
            f"{schema_name} schema validation failed at "
            f"{_validation_path(first.absolute_path)}: {first.message}"
        )


def validate_model_manifest(manifest: Mapping[str, Any]) -> None:
    _validate_contract("model-manifest", manifest)
    required = {
        "schemaVersion",
        "modelId",
        "version",
        "architecture",
        "labels",
        "input",
        "thresholds",
        "preprocessing",
        "training",
        "dataSources",
        "license",
        "files",
        "modelSha256",
        "manifestSha256",
    }
    missing = required - set(manifest)
    if missing:
        raise ValueError(f"model manifest missing fields: {sorted(missing)}")
    if manifest["schemaVersion"] != "medmodel-1":
        raise ValueError("unsupported model manifest schema")
    _validate_identifier(manifest["modelId"], "modelId")
    _validate_identifier(manifest["version"], "version")
    validate_label_order(tuple(manifest["labels"]))
    if manifest["architecture"] != "densenet121":
        raise ValueError("only DenseNet-121 is supported in v0.1")
    if manifest["input"] != {"width": 320, "height": 320, "channels": 1, "normalization": "imagenet-gray"}:
        raise ValueError("model input definition must be 320x320 single-channel imagenet-gray")
    if manifest["preprocessing"] != {"viewPositions": ["AP", "PA"], "adultOnly": True}:
        raise ValueError("model preprocessing must be adult AP/PA only")
    training = manifest["training"]
    if not isinstance(training, Mapping):
        raise ValueError("model training metadata must be an object")
    if training.get("method") not in {"baseline", "mixstyle"}:
        raise ValueError("model training method must be baseline or mixstyle")
    if not str(training.get("commit", "")).strip():
        raise ValueError("model training commit is required")
    if not isinstance(training.get("seeds"), list) or not training["seeds"]:
        raise ValueError("model training seeds are required")
    if not isinstance(manifest["dataSources"], list) or not manifest["dataSources"]:
        raise ValueError("at least one model data source is required")
    if not str(manifest["license"]).strip():
        raise ValueError("model license is required")
    thresholds = manifest["thresholds"]
    if set(thresholds) != set(CHEXPERT_LABELS):
        raise ValueError("model thresholds must contain exactly the CheXpert 14 labels")
    if any(not 0 <= float(value) <= 1 for value in thresholds.values()):
        raise ValueError("model thresholds must be in [0, 1]")
    if not isinstance(manifest["files"], Mapping) or not manifest["files"]:
        raise ValueError("model manifest must include file SHA-256 values")
    if not {"model.onnx", "labels.json"} <= set(manifest["files"]):
        raise ValueError("model manifest must include model.onnx and labels.json")
    for name, digest in manifest["files"].items():
        if not isinstance(name, str):
            raise ValueError("model file names must be strings")
        normalized = _safe_archive_name(name)
        if normalized != name or name == "manifest.json":
            raise ValueError(f"invalid model file path: {name!r}")
        _validate_sha256(digest, f"files[{name!r}]")
    model_digest = _validate_sha256(manifest["modelSha256"], "modelSha256")
    if model_digest != manifest["files"]["model.onnx"]:
        raise ValueError("modelSha256 must match files['model.onnx']")
    manifest_digest = _validate_sha256(manifest["manifestSha256"], "manifestSha256")
    if manifest_digest != model_manifest_sha256(manifest):
        raise ValueError("model manifest SHA-256 mismatch")


def validate_experiment_bundle(bundle: Mapping[str, Any]) -> None:
    _validate_contract("experiment-bundle", bundle)
    required = {"schemaVersion", "experimentId", "config", "seeds", "datasetManifestSha256", "aggregateMetrics", "perClassMetrics", "curves", "modelCard"}
    missing = required - set(bundle)
    if missing:
        raise ValueError(f"experiment bundle missing fields: {sorted(missing)}")
    if bundle["schemaVersion"] != "medexperiment-1":
        raise ValueError("unsupported experiment bundle schema")
    _validate_identifier(bundle["experimentId"], "experimentId")
    _validate_sha256(bundle["datasetManifestSha256"], "datasetManifestSha256")
    forbidden = {"patient_id", "subject_id", "mimic_id", "patient_ids", "predictions", "image_path", "dicom_path"}

    def walk(value: Any, key: str = "") -> None:
        if key.lower() in forbidden:
            raise ValueError(f"experiment bundle contains prohibited patient-level field: {key}")
        if isinstance(value, Mapping):
            for child_key, child_value in value.items():
                walk(child_value, str(child_key))
        elif isinstance(value, list):
            for child in value:
                walk(child, key)

    walk(bundle)


def build_medmodel(source_dir: str | Path, output_path: str | Path) -> Path:
    """Package a validated model directory without downloading or inventing weights."""

    source = Path(source_dir)
    manifest_path = source / "manifest.json"
    if not manifest_path.is_file() or not (source / "model.onnx").is_file():
        raise FileNotFoundError("a .medmodel source must contain manifest.json and model.onnx")
    manifest = _load_json(manifest_path.read_text(encoding="utf-8"))
    validate_model_manifest(manifest)
    for relative, expected in manifest["files"].items():
        file_path = source / relative
        if not file_path.is_file() or sha256_file(file_path) != expected:
            raise ValueError(f"file hash mismatch: {relative}")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for relative in ["manifest.json", *manifest["files"].keys()]:
            archive.write(source / relative, relative)
    return destination


def read_medmodel_manifest(package_path: str | Path) -> dict[str, Any]:
    with zipfile.ZipFile(package_path, "r") as archive:
        try:
            manifest = _load_json(archive.read("manifest.json"))
        except (KeyError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError("invalid .medmodel: manifest.json is missing or malformed") from exc
        validate_model_manifest(manifest)
        expected_names = {"manifest.json", *manifest["files"].keys()}
        archive_names = archive.namelist()
        if len(archive_names) != len(set(archive_names)) or set(archive_names) != expected_names:
            raise ValueError("invalid .medmodel: unexpected, duplicate, or missing archive entries")
        for name in archive_names:
            try:
                normalized = _safe_archive_name(name)
            except ValueError as exc:
                raise ValueError("invalid .medmodel: unsafe archive path") from exc
            if normalized != name:
                raise ValueError("invalid .medmodel: unsafe archive path")
        for relative, expected in manifest["files"].items():
            try:
                actual = sha256_bytes(archive.read(relative))
            except KeyError as exc:
                raise ValueError(f"invalid .medmodel: missing {relative}") from exc
            if actual != expected:
                raise ValueError(f"invalid .medmodel: hash mismatch for {relative}")
        try:
            label_dictionary = _load_json(archive.read("labels.json"))
        except (KeyError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError("invalid .medmodel: labels.json is missing or malformed") from exc
        labels = label_dictionary.get("labels") if isinstance(label_dictionary, Mapping) else None
        if tuple(labels or ()) != CHEXPERT_LABELS:
            raise ValueError("invalid .medmodel: labels.json does not match CheXpert 14")
        return manifest


def build_medexperiment(
    bundle: Mapping[str, Any],
    output_path: str | Path,
    extra_files: Mapping[str, str | Path] | None = None,
) -> Path:
    """Create a portable aggregate-only experiment package."""

    validate_experiment_bundle(bundle)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "experiment.json",
            json.dumps(bundle, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        )
        for relative, path in (extra_files or {}).items():
            try:
                normalized = _safe_archive_name(relative)
            except ValueError as exc:
                raise ValueError(f"invalid experiment artifact path: {relative}") from exc
            if normalized != relative or relative == "experiment.json":
                raise ValueError(f"invalid experiment artifact path: {relative}")
            if Path(path).suffix.lower() in {".dcm", ".dicom", ".png", ".jpg", ".jpeg", ".csv", ".parquet"}:
                raise ValueError("patient-level or image artifacts cannot be included in .medexperiment")
            archive.write(path, normalized)
    return destination


def read_medexperiment(package_path: str | Path) -> dict[str, Any]:
    path = Path(package_path)
    try:
        with zipfile.ZipFile(path, "r") as archive:
            names = archive.namelist()
            if len(names) != len(set(names)) or "experiment.json" not in names:
                raise ValueError("invalid .medexperiment archive entries")
            forbidden_suffixes = {".dcm", ".dicom", ".png", ".jpg", ".jpeg", ".csv", ".parquet"}
            total_size = 0
            for info in archive.infolist():
                try:
                    normalized = _safe_archive_name(info.filename)
                except ValueError as exc:
                    raise ValueError("invalid .medexperiment archive path") from exc
                if normalized != info.filename or Path(normalized).suffix.lower() in forbidden_suffixes:
                    raise ValueError("invalid .medexperiment archive content")
                total_size += info.file_size
            if total_size > 100 * 1024**2:
                raise ValueError("invalid .medexperiment: uncompressed content exceeds 100 MB")
            raw = archive.read("experiment.json")
    except (zipfile.BadZipFile, KeyError) as exc:
        raise ValueError("invalid .medexperiment package") from exc
    try:
        bundle = _load_json(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError("invalid .medexperiment experiment.json") from exc
    validate_experiment_bundle(bundle)
    return bundle
