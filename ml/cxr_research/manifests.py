from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .data import CheXpertRecord, record_dict, record_from_dict
from .splits import manifest_sha256, split_patient_records


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_chexpert_split_manifest(
    records: Sequence[CheXpertRecord],
    source_csv: str | Path,
    seed: int = 20240913,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> dict[str, Any]:
    image_paths = {Path(record.image_path) for record in records}
    missing_images = [str(path) for path in image_paths if not path.is_file()]
    if missing_images:
        raise FileNotFoundError(f"CheXpert images are missing: {missing_images[0]}")
    rows = [record_dict(record) for record in records]
    split = split_patient_records(rows, seed=seed, ratios=ratios)
    payload: dict[str, Any] = {
        "schemaVersion": "chexpert-split-v1",
        "seed": seed,
        "ratios": list(ratios),
        "sourceCsvSha256": file_sha256(source_csv),
        "imageBytes": sum(path.stat().st_size for path in image_paths),
        "records": {
            "train": list(split.train),
            "validation": list(split.validation),
            "test": list(split.test),
        },
    }
    payload["manifestSha256"] = manifest_sha256(
        {"split": name, **record}
        for name, values in payload["records"].items()
        for record in values
    )
    return payload


def write_manifest(path: str | Path, manifest: Mapping[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return destination


def read_chexpert_split_manifest(path: str | Path) -> dict[str, list[CheXpertRecord]]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("schemaVersion") != "chexpert-split-v1":
        raise ValueError("unsupported CheXpert split manifest")
    actual = manifest_sha256(
        {"split": name, **record}
        for name, values in value["records"].items()
        for record in values
    )
    if actual != value.get("manifestSha256"):
        raise ValueError("CheXpert split manifest hash mismatch")
    result = {name: [record_from_dict(record) for record in values] for name, values in value["records"].items()}
    patient_sets = {name: {record.patient_id for record in records} for name, records in result.items()}
    names = tuple(patient_sets)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            if patient_sets[left] & patient_sets[right]:
                raise ValueError(f"patient leakage between {left} and {right}")
    return result
