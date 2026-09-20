from __future__ import annotations

import csv
import hashlib
import json
import random
import zipfile
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import numpy as np

from .artifacts import read_medmodel_manifest
from .data import preprocess_raster
from .labels import CHEXPERT_LABELS, PRIMARY_LABELS, encode_row
from .metrics import add_patient_bootstrap_intervals, evaluate_multilabel

TOTAL_DATA_BUDGET_BYTES = 20 * 1024**3
DEFAULT_MIMIC_BUDGET_BYTES = 8 * 1024**3


def _sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode()).hexdigest()


def sample_mimic_external(
    metadata_csv: str | Path,
    output_manifest: str | Path,
    *,
    image_root: str | Path | None = None,
    patient_column: str = "subject_id",
    study_column: str = "study_id",
    image_column: str = "path",
    view_column: str = "ViewPosition",
    age_column: str = "age",
    seed: int = 20240913,
    target_patients: int = 8000,
    max_bytes: int = DEFAULT_MIMIC_BUDGET_BYTES,
) -> dict[str, Any]:
    """Select one adult AP/PA image per patient without consulting outcomes."""

    if max_bytes <= 0 or max_bytes > TOTAL_DATA_BUDGET_BYTES:
        raise ValueError("MIMIC byte budget must be positive and no greater than 20 GiB")
    root = Path(image_root) if image_root else Path(metadata_csv).parent
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with Path(metadata_csv).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {patient_column, study_column, image_column, view_column, age_column}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"MIMIC metadata missing columns: {sorted(missing)}")
        for row in reader:
            try:
                age = int(float(row[age_column]))
            except (TypeError, ValueError):
                continue
            view = str(row[view_column]).strip().upper()
            if age < 18 or view not in {"AP", "PA"}:
                continue
            groups[str(row[patient_column])].append(
                {
                    "patientId": str(row[patient_column]),
                    "studyId": str(row[study_column]),
                    "relativePath": str(row[image_column]).replace("\\", "/"),
                    "viewPosition": view,
                    "age": age,
                }
            )
    rng = random.Random(seed)
    patients = sorted(groups)
    rng.shuffle(patients)
    considered = 0
    missing_images = 0
    chosen: list[dict[str, Any]] = []
    for patient in patients[:target_patients]:
        candidates = sorted(groups[patient], key=lambda item: (item["studyId"], item["relativePath"]))
        item = candidates[rng.randrange(len(candidates))]
        considered += 1
        image_path = root / item["relativePath"]
        if not image_path.is_file():
            missing_images += 1
            continue
        item = {**item, "sizeBytes": image_path.stat().st_size}
        chosen.append(item)
    # A wrong --image-root silently produced an empty but sealed manifest before;
    # refuse instead, because an empty selection cannot be a valid external test set.
    if considered and not chosen:
        raise ValueError(
            f"none of the {considered} sampled MIMIC images exist under {root}; "
            "check --image-root, and do not seal a manifest from an empty selection"
        )
    # Budget reduction follows the frozen random order and file sizes only.
    within_budget: list[dict[str, Any]] = []
    total = 0
    for item in chosen:
        size = int(item["sizeBytes"])
        if total + size > max_bytes:
            continue
        within_budget.append(item)
        total += size
    payload: dict[str, Any] = {
        "schemaVersion": "mimic-external-v1",
        "seed": seed,
        "selectionRule": "adult AP/PA; one image per patient; fixed random order; file-size-only budget reduction",
        "requestedPatients": target_patients,
        "selectedPatients": len(within_budget),
        "consideredPatients": considered,
        "missingImages": missing_images,
        "maxBytes": max_bytes,
        "totalBytes": total,
        "records": within_budget,
    }
    payload["manifestSha256"] = _sha256(payload)
    destination = Path(output_manifest)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    destination.with_suffix(destination.suffix + ".sha256").write_text(payload["manifestSha256"] + "\n", encoding="ascii")
    return payload


def verify_external_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = payload.pop("manifestSha256", None)
    actual = _sha256(payload)
    lock_path = manifest_path.with_suffix(manifest_path.suffix + ".sha256")
    locked = lock_path.read_text(encoding="ascii").strip() if lock_path.is_file() else None
    if not expected or expected != actual or locked != actual:
        raise ValueError("external manifest or SHA-256 lock has changed")
    payload["manifestSha256"] = actual
    return payload


def _external_image_path(root: Path, relative_path: str, expected_size: int) -> Path:
    normalized = relative_path.replace("\\", "/")
    candidate = PurePosixPath(normalized)
    if candidate.is_absolute() or ".." in candidate.parts or not candidate.parts:
        raise ValueError("external manifest contains an unsafe image path")
    image_path = root.joinpath(*candidate.parts).resolve()
    if root.resolve() not in image_path.parents:
        raise ValueError("external image path escaped the configured root")
    if not image_path.is_file() or image_path.stat().st_size != expected_size:
        raise ValueError(f"external image missing or size changed: {relative_path}")
    return image_path


def _primary_macro(metrics: Mapping[str, Any], key: str) -> float | None:
    values = [
        metrics["per_class"][label][key]
        for label in PRIMARY_LABELS
        if metrics["per_class"][label][key] is not None
    ]
    return float(np.mean(values)) if values else None


def _external_failure_cases(
    records: list[Mapping[str, Any]],
    targets: np.ndarray,
    mask: np.ndarray,
    probabilities: np.ndarray,
    thresholds: Mapping[str, float],
    limit: int = 10,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for index, label in enumerate(CHEXPERT_LABELS):
        valid = mask[:, index].astype(bool)
        predicted = probabilities[:, index] >= float(thresholds[label])
        false_positive = np.flatnonzero(valid & (targets[:, index] == 0) & predicted)
        false_negative = np.flatnonzero(valid & (targets[:, index] == 1) & ~predicted)
        false_positive = false_positive[np.argsort(-probabilities[false_positive, index])][:limit]
        false_negative = false_negative[np.argsort(probabilities[false_negative, index])][:limit]
        result[label] = {
            "falsePositive": [
                {
                    "studyId": str(records[item]["studyId"]),
                    "score": float(probabilities[item, index]),
                }
                for item in false_positive
            ],
            "falseNegative": [
                {
                    "studyId": str(records[item]["studyId"]),
                    "score": float(probabilities[item, index]),
                }
                for item in false_negative
            ],
        }
    return result


def evaluate_mimic_external(
    manifest_path: str | Path,
    labels_csv: str | Path,
    medmodel_path: str | Path,
    output_directory: str | Path,
    *,
    image_root: str | Path,
    patient_column: str = "subject_id",
    study_column: str = "study_id",
    batch_size: int = 16,
    bootstrap_resamples: int = 2000,
) -> dict[str, Any]:
    """Run final external evaluation only after the sealed manifest verifies.

    The public result contains aggregate values only. Study identifiers are
    written solely to ``failure_cases.local.json`` and must never be packaged.
    """

    if batch_size <= 0 or bootstrap_resamples <= 0:
        raise ValueError("batch_size and bootstrap_resamples must be positive")
    sealed = verify_external_manifest(manifest_path)
    model_manifest = read_medmodel_manifest(medmodel_path)
    root = Path(image_root)
    records = list(sealed["records"])
    if not records:
        raise ValueError("external manifest contains no records")
    image_paths = [
        _external_image_path(root, str(record["relativePath"]), int(record["sizeBytes"]))
        for record in records
    ]

    # Outcome labels are intentionally opened only after the manifest and all
    # selected files pass their precommitted integrity checks.
    label_rows: dict[tuple[str, str], dict[str, str]] = {}
    with Path(labels_csv).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {patient_column, study_column, *CHEXPERT_LABELS}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"MIMIC labels missing columns: {sorted(missing)}")
        for row in reader:
            key = (str(row[patient_column]), str(row[study_column]))
            if key in label_rows:
                raise ValueError(f"duplicate MIMIC label row for study: {key[1]}")
            label_rows[key] = row

    targets: list[list[float]] = []
    masks: list[list[float]] = []
    for record in records:
        key = (str(record["patientId"]), str(record["studyId"]))
        row = label_rows.get(key)
        if row is None:
            raise ValueError(f"MIMIC labels missing sealed study: {record['studyId']}")
        target, mask = encode_row(row)
        targets.append(target)
        masks.append(mask)

    try:
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("external evaluation requires onnxruntime") from exc
    logits_batches: list[np.ndarray] = []
    with zipfile.ZipFile(medmodel_path, "r") as archive:
        model_bytes = archive.read("model.onnx")
    session = ort.InferenceSession(model_bytes, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    for start in range(0, len(image_paths), batch_size):
        batch = np.stack(
            [preprocess_raster(path) for path in image_paths[start : start + batch_size]],
            axis=0,
        )
        outputs = session.run(None, {input_name: batch})
        logits = np.asarray(outputs[0], dtype=np.float32)
        if logits.shape != (len(batch), len(CHEXPERT_LABELS)):
            raise RuntimeError("external model output is not [batch, 14]")
        logits_batches.append(logits)

    logits = np.concatenate(logits_batches, axis=0)
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -80, 80)))
    target_array = np.asarray(targets, dtype=np.float32)
    mask_array = np.asarray(masks, dtype=np.float32)
    thresholds = {
        label: float(model_manifest["thresholds"][label]) for label in CHEXPERT_LABELS
    }
    metrics = evaluate_multilabel(
        target_array,
        probabilities,
        CHEXPERT_LABELS,
        mask_array,
        thresholds,
    )
    patient_ids = [str(record["patientId"]) for record in records]
    add_patient_bootstrap_intervals(
        metrics,
        target_array,
        probabilities,
        mask_array,
        patient_ids,
        CHEXPERT_LABELS,
        thresholds,
        bootstrap_resamples,
        int(sealed["seed"]),
    )
    result = {
        "schemaVersion": "mimic-external-result-v1",
        "datasetManifestSha256": sealed["manifestSha256"],
        "modelId": model_manifest["modelId"],
        "modelVersion": model_manifest["version"],
        "modelSha256": model_manifest["modelSha256"],
        "sampleCount": len(records),
        "primaryMacroAuroc": _primary_macro(metrics, "auroc"),
        "primaryMacroAuprc": _primary_macro(metrics, "auprc"),
        "metrics": metrics,
    }
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    (output / "external-result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output / "failure_cases.local.json").write_text(
        json.dumps(
            _external_failure_cases(
                records,
                target_array,
                mask_array,
                probabilities,
                thresholds,
            ),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return result
