"""Dataset indexing and PyTorch datasets for the ``cxr_first`` task.

The organizer ships one CSV per split. Every row is a single chest X-ray image
and ``Predict_class`` is a semicolon separated list of class ids. Images are
stored as ``<split>/<Subject_id>/<Study_id>/<Dicom_id>.jpg``.

Only the organizer-provided train/valid/test splits are used; the three splits
are already patient-disjoint and the test CSV carries no labels.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np

from .labels import LABEL_IDS

IMAGE_SUFFIXES: tuple[str, ...] = (".jpg", ".jpeg", ".png")
REQUIRED_COLUMNS = ("Dicom_id", "Subject_id", "Study_id")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class CompetitionRecord:
    """One labelled or unlabelled image row."""

    dicom_id: str
    subject_id: str
    study_id: str
    view_position: str
    patient_orientation: str
    rows: int
    columns: int
    image_path: str
    targets: tuple[float, ...] | None


def parse_predict_class(value: Any, label_ids: Sequence[int] = LABEL_IDS) -> tuple[int, ...]:
    """Parse ``Predict_class`` like ``"7;9"`` into a sorted tuple of class ids."""

    text = "" if value is None else str(value).strip()
    if not text:
        raise ValueError("Predict_class is empty; a labelled row needs at least one class id")
    parsed: list[int] = []
    for part in text.split(";"):
        token = part.strip()
        if not token:
            continue
        try:
            label_id = int(token)
        except ValueError as exc:
            raise ValueError(f"invalid Predict_class token: {token!r}") from exc
        if label_id not in label_ids:
            raise ValueError(f"Predict_class id {label_id} is outside {list(label_ids)}")
        if label_id not in parsed:
            parsed.append(label_id)
    if not parsed:
        raise ValueError(f"Predict_class has no usable id: {value!r}")
    return tuple(sorted(parsed))


def targets_from_ids(
    active_ids: Iterable[int], label_ids: Sequence[int] = LABEL_IDS
) -> tuple[float, ...]:
    """Build the multi-hot target vector in ``label_ids`` order."""

    active = set(active_ids)
    return tuple(1.0 if label_id in active else 0.0 for label_id in label_ids)


def resolve_image_path(
    split_root: str | Path, subject_id: str, study_id: str, dicom_id: str
) -> Path:
    """Resolve ``<root>/<subject>/<study>/<dicom>.{jpg,jpeg,png}``."""

    root = Path(split_root)
    for suffix in IMAGE_SUFFIXES:
        candidate = root / subject_id / study_id / f"{dicom_id}{suffix}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"missing image {subject_id}/{study_id}/{dicom_id} under {root}"
    )


def load_split(
    csv_path: str | Path,
    split_root: str | Path,
    *,
    label_ids: Sequence[int] = LABEL_IDS,
    labelled: bool = True,
    require_images: bool = True,
) -> list[CompetitionRecord]:
    """Load one split CSV into validated records.

    ``labelled=False`` is used for the test CSV, which has no ``Predict_class``
    column by design.
    """

    path = Path(csv_path)
    root = Path(split_root)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        missing = set(REQUIRED_COLUMNS) - fields
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        if labelled and "Predict_class" not in fields:
            raise ValueError(f"{path} has no Predict_class column for a labelled split")
        records: list[CompetitionRecord] = []
        seen: set[str] = set()
        for line_number, row in enumerate(reader, start=2):
            dicom_id = str(row["Dicom_id"]).strip()
            if not dicom_id:
                raise ValueError(f"{path}:{line_number} has an empty Dicom_id")
            if dicom_id in seen:
                raise ValueError(f"{path}:{line_number} duplicates Dicom_id {dicom_id}")
            seen.add(dicom_id)
            subject_id = str(row["Subject_id"]).strip()
            study_id = str(row["Study_id"]).strip()
            if not subject_id or not study_id:
                raise ValueError(f"{path}:{line_number} needs Subject_id and Study_id")
            if require_images:
                image_path = str(resolve_image_path(root, subject_id, study_id, dicom_id))
            else:
                image_path = str(root / subject_id / study_id / f"{dicom_id}.jpg")
            targets: tuple[float, ...] | None = None
            if labelled:
                active = parse_predict_class(row.get("Predict_class"), label_ids)
                targets = targets_from_ids(active, label_ids)
            records.append(
                CompetitionRecord(
                    dicom_id=dicom_id,
                    subject_id=subject_id,
                    study_id=study_id,
                    view_position=str(row.get("ViewPosition") or "").strip(),
                    patient_orientation=str(row.get("PatientOrientation") or "").strip(),
                    rows=_safe_int(row.get("Rows")),
                    columns=_safe_int(row.get("Columns")),
                    image_path=image_path,
                    targets=targets,
                )
            )
    if not records:
        raise ValueError(f"{path} contains no data rows")
    return records


def _safe_int(value: Any) -> int:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return 0


def records_to_arrays(
    records: Sequence[CompetitionRecord],
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(targets[N, 10], mask[N, 10])``; unlabelled rows are rejected."""

    if any(record.targets is None for record in records):
        raise ValueError("records_to_arrays requires a labelled split")
    targets = np.asarray([record.targets for record in records], dtype=np.float32)
    if targets.ndim != 2:  # pragma: no cover - defensive
        raise ValueError("targets must be a 2D array")
    return targets, np.ones_like(targets, dtype=np.float32)


def group_by_study(records: Iterable[CompetitionRecord]) -> dict[str, list[CompetitionRecord]]:
    """Group image records by ``Study_id`` preserving CSV order."""

    groups: dict[str, list[CompetitionRecord]] = {}
    for record in records:
        groups.setdefault(record.study_id, []).append(record)
    return groups


def study_targets(records: Sequence[CompetitionRecord]) -> tuple[str, ...]:
    """Aggregate image-level targets to study level: any positive image wins."""

    groups = group_by_study(records)
    study_ids: list[str] = []
    for study_id, group in groups.items():
        if any(record.targets is None for record in group):
            raise ValueError("study_targets requires a labelled split")
        study_ids.append(study_id)
    return tuple(study_ids)


def study_target_matrix(records: Sequence[CompetitionRecord]) -> dict[str, np.ndarray]:
    """Map each study id to its multi-hot study-level target vector."""

    result: dict[str, np.ndarray] = {}
    for study_id, group in group_by_study(records).items():
        if any(record.targets is None for record in group):
            raise ValueError("study_target_matrix requires a labelled split")
        stacked = np.asarray([record.targets for record in group], dtype=np.float32)
        result[study_id] = stacked.max(axis=0)
    return result


def build_transforms(image_size: int, *, training: bool) -> Any:
    """Torchvision transforms shared by training and inference."""

    from torchvision import transforms

    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    if training:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
                transforms.RandomRotation(degrees=7, fill=0),
                transforms.ColorJitter(brightness=0.15, contrast=0.15),
                transforms.ToTensor(),
                normalize,
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ]
    )


class CompetitionDataset:
    """Labelled image dataset used for training and validation."""

    def __init__(
        self,
        records: Sequence[CompetitionRecord],
        *,
        image_size: int,
        training: bool,
    ) -> None:
        if any(record.targets is None for record in records):
            raise ValueError("CompetitionDataset requires labelled records")
        self.records = tuple(records)
        self.transform = build_transforms(image_size, training=training)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        import torch
        from PIL import Image

        record = self.records[index]
        with Image.open(record.image_path) as image:
            pixels = self.transform(image.convert("RGB"))
        target = torch.tensor(record.targets, dtype=torch.float32)
        return pixels, target


class TestImageDataset:
    """Unlabelled test images returning ``(image, dicom_id)`` pairs."""

    def __init__(self, records: Sequence[CompetitionRecord], *, image_size: int) -> None:
        self.records = tuple(records)
        self.transform = build_transforms(image_size, training=False)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Any, str]:
        from PIL import Image

        record = self.records[index]
        with Image.open(record.image_path) as image:
            pixels = self.transform(image.convert("RGB"))
        return pixels, record.dicom_id


def iter_batches(items: Sequence[Any], batch_size: int) -> Iterator[Sequence[Any]]:
    """Small helper used by inference loops that avoid torch DataLoader."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]
