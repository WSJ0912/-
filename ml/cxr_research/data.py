from __future__ import annotations

import csv
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .labels import CHEXPERT_LABELS, encode_row

PATIENT_PATTERN = re.compile(r"(?:^|[/\\])patient(\d+)(?:[/\\]|$)", re.IGNORECASE)
STUDY_PATTERN = re.compile(r"(?:^|[/\\])study(\d+)(?:[/\\]|$)", re.IGNORECASE)


@dataclass(frozen=True)
class CheXpertRecord:
    patient_id: str
    study_id: str
    image_path: str
    view_position: str
    age: int
    targets: tuple[float, ...]
    mask: tuple[float, ...]


def _identifier(row: Mapping[str, Any], explicit: str, pattern: re.Pattern[str], path: str) -> str:
    value = str(row.get(explicit, "")).strip()
    if value:
        return value
    match = pattern.search(path)
    if not match:
        raise ValueError(f"cannot derive {explicit} from CheXpert path: {path}")
    return match.group(1)


def _adult_age(value: Any) -> int | None:
    try:
        age = int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None
    return age if age >= 18 else None


def load_chexpert_csv(
    csv_path: str | Path,
    image_root: str | Path | None = None,
    *,
    adult_frontal_only: bool = True,
) -> list[CheXpertRecord]:
    """Load the real CheXpert CSV shape without inventing Patient/Study columns.

    Official CheXpert files encode patient and study identifiers inside ``Path``.
    Explicit ``Patient``/``Study`` columns are accepted for derived manifests, but
    are not required. The production research protocol excludes pediatric,
    lateral, and unknown-position rows before splitting.
    """

    path = Path(csv_path)
    root = Path(image_root) if image_root is not None else path.parent
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if "Path" not in set(reader.fieldnames or ()):
            raise ValueError("CheXpert CSV missing Path column")
        records: list[CheXpertRecord] = []
        for row in reader:
            relative = str(row["Path"]).strip()
            frontal_lateral = str(row.get("Frontal/Lateral", "")).strip().upper()
            view = str(row.get("AP/PA", row.get("ViewPosition", ""))).strip().upper()
            age = _adult_age(row.get("Age"))
            if adult_frontal_only and (frontal_lateral not in {"FRONTAL", ""} or view not in {"AP", "PA"} or age is None):
                continue
            if age is None:
                raise ValueError(f"missing or pediatric age for included row: {relative}")
            if view not in {"AP", "PA"}:
                raise ValueError(f"unknown view position for included row: {relative}")
            targets, mask = encode_row(row)
            normalized_relative = Path(relative.replace("\\", "/"))
            records.append(
                CheXpertRecord(
                    patient_id=_identifier(row, "Patient", PATIENT_PATTERN, relative),
                    study_id=_identifier(row, "Study", STUDY_PATTERN, relative),
                    image_path=str((root / normalized_relative).resolve()),
                    view_position=view,
                    age=age,
                    targets=tuple(targets),
                    mask=tuple(mask),
                )
            )
    if not records:
        raise ValueError("no adult AP/PA CheXpert records were found")
    return records


def records_to_arrays(records: Iterable[CheXpertRecord]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    values = list(records)
    if not values:
        return np.empty((0, len(CHEXPERT_LABELS))), np.empty((0, len(CHEXPERT_LABELS))), []
    return (
        np.asarray([item.targets for item in values], dtype=np.float32),
        np.asarray([item.mask for item in values], dtype=np.float32),
        [item.patient_id for item in values],
    )


def record_dict(record: CheXpertRecord) -> dict[str, Any]:
    return asdict(record)


def record_from_dict(value: Mapping[str, Any]) -> CheXpertRecord:
    return CheXpertRecord(
        patient_id=str(value["patient_id"]),
        study_id=str(value["study_id"]),
        image_path=str(value["image_path"]),
        view_position=str(value["view_position"]),
        age=int(value["age"]),
        targets=tuple(float(item) for item in value["targets"]),
        mask=tuple(float(item) for item in value["mask"]),
    )


def preprocess_raster(path: str | Path, image_size: int = 320) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Pillow is required for CheXpert image loading") from exc
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("L"), dtype=np.float32)
    low, high = np.percentile(pixels, [1, 99])
    if high <= low:
        low, high = float(pixels.min()), float(pixels.max())
    normalized = np.clip((pixels - low) / max(high - low, 1e-6), 0, 1)
    resized = Image.fromarray(np.round(normalized * 255).astype(np.uint8), mode="L").resize(
        (image_size, image_size), Image.Resampling.BILINEAR
    )
    array = np.asarray(resized, dtype=np.float32) / 255.0
    return ((array - 0.485) / 0.229)[None, :, :].astype(np.float32)


try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover
    torch = None
    Dataset = object  # type: ignore[assignment,misc]


class CheXpertDataset(Dataset):
    def __init__(self, records: Sequence[CheXpertRecord], image_size: int = 320) -> None:
        if torch is None:
            raise RuntimeError("CheXpertDataset requires PyTorch")
        self.records = tuple(records)
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Any, Any, Any]:
        record = self.records[index]
        image = torch.from_numpy(preprocess_raster(record.image_path, self.image_size))
        target = torch.tensor(record.targets, dtype=torch.float32)
        mask = torch.tensor(record.mask, dtype=torch.float32)
        return image, target, mask
