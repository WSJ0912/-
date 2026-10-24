"""Numeric label ids for the ``cxr_first`` competition dataset.

The organizer publishes ten class ids (1-10) and keeps the id-to-name mapping in
the competition material. This repository stores the mapping as an optional JSON
file so training, evaluation, and submission stay correct even before the
official names are transcribed: numeric ids are the source of truth.

Expected mapping file shape::

    {"labels": [
        {"id": 1, "key": "atelectasis", "zh": "肺不张", "en": "Atelectasis"},
        {"id": 2, "key": "cardiomegaly", "zh": "心脏增大", "en": "Cardiomegaly"}
    ]}
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

LABEL_IDS: tuple[int, ...] = tuple(range(1, 11))


@dataclass(frozen=True)
class CompetitionLabel:
    id: int
    key: str
    zh: str
    en: str


OFFICIAL_LABELS: tuple[CompetitionLabel, ...] = (
    CompetitionLabel(1, "enlarged_cardiomediastinum", "纵隔心影增大", "Enlarged Cardiomediastinum"),
    CompetitionLabel(2, "pneumothorax", "气胸", "Pneumothorax"),
    CompetitionLabel(3, "consolidation", "肺部实变", "Consolidation"),
    CompetitionLabel(4, "pneumonia", "肺炎", "Pneumonia"),
    CompetitionLabel(5, "edema", "肺水肿", "Edema"),
    CompetitionLabel(6, "cardiomegaly", "心影扩大", "Cardiomegaly"),
    CompetitionLabel(7, "atelectasis", "肺不张", "Atelectasis"),
    CompetitionLabel(8, "lung_opacity", "肺部阴影", "Lung Opacity"),
    CompetitionLabel(9, "pleural_effusion", "胸腔积液", "Pleural Effusion"),
    CompetitionLabel(10, "no_finding", "无异常发现", "No Finding"),
)

# The official id-to-name table supplied by the organizer. Class 10 is the
# single-label "No Finding" class; classes 1-9 are multi-label findings.
DEFAULT_LABELS: tuple[CompetitionLabel, ...] = OFFICIAL_LABELS


def label_column(label_id: int) -> str:
    """Return the stable submission/config column name for one class id."""

    return f"class_{label_id:02d}"


def load_label_mapping(path: str | Path) -> tuple[CompetitionLabel, ...]:
    """Load and validate an official id-to-name mapping file."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = payload.get("labels") if isinstance(payload, dict) else payload
    if not isinstance(entries, list):
        raise ValueError("label mapping must be a list or an object with a 'labels' list")
    labels: list[CompetitionLabel] = []
    seen_keys: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("each label mapping entry must be an object")
        label_id = int(entry["id"])
        key = str(entry.get("key") or f"class_{label_id:02d}")
        if key in seen_keys:
            raise ValueError(f"duplicate label key: {key}")
        seen_keys.add(key)
        labels.append(
            CompetitionLabel(
                id=label_id,
                key=key,
                zh=str(entry.get("zh") or f"类别 {label_id}"),
                en=str(entry.get("en") or entry.get("name") or f"Class {label_id}"),
            )
        )
    ids = tuple(label.id for label in labels)
    if ids != LABEL_IDS:
        raise ValueError(f"label mapping ids must be exactly {list(LABEL_IDS)}, got {list(ids)}")
    return tuple(labels)
