from __future__ import annotations

from dataclasses import dataclass
from math import isnan
from typing import Any, Mapping


@dataclass(frozen=True)
class LabelDefinition:
    key: str
    zh: str
    en: str
    kind: str
    primary: bool


LABEL_DEFINITIONS: tuple[LabelDefinition, ...] = (
    LabelDefinition("Atelectasis", "肺不张", "Atelectasis", "abnormality", True),
    LabelDefinition("Cardiomegaly", "心脏增大", "Cardiomegaly", "abnormality", True),
    LabelDefinition("Consolidation", "实变", "Consolidation", "abnormality", True),
    LabelDefinition("Edema", "水肿", "Edema", "abnormality", True),
    LabelDefinition("Enlarged Cardiomediastinum", "纵隔增宽", "Enlarged Cardiomediastinum", "abnormality", False),
    LabelDefinition("Fracture", "骨折", "Fracture", "abnormality", False),
    LabelDefinition("Lung Lesion", "肺部病灶", "Lung Lesion", "abnormality", False),
    LabelDefinition("Lung Opacity", "肺部阴影", "Lung Opacity", "abnormality", False),
    LabelDefinition("No Finding", "未见异常", "No Finding", "status", False),
    LabelDefinition("Pleural Effusion", "胸腔积液", "Pleural Effusion", "abnormality", True),
    LabelDefinition("Pleural Other", "其他胸膜病变", "Pleural Other", "abnormality", False),
    LabelDefinition("Pneumonia", "肺炎", "Pneumonia", "abnormality", False),
    LabelDefinition("Pneumothorax", "气胸", "Pneumothorax", "abnormality", False),
    LabelDefinition("Support Devices", "支持性器械", "Support Devices", "status", False),
)

CHEXPERT_LABELS: tuple[str, ...] = tuple(item.key for item in LABEL_DEFINITIONS)
PRIMARY_LABELS: tuple[str, ...] = tuple(item.key for item in LABEL_DEFINITIONS if item.primary)
LABEL_INDEX = {name: index for index, name in enumerate(CHEXPERT_LABELS)}


def validate_label_order(labels: list[str] | tuple[str, ...]) -> None:
    """Reject partial, reordered, or foreign label dictionaries."""

    if tuple(labels) != CHEXPERT_LABELS:
        raise ValueError(
            "模型必须严格使用 CheXpert 14 项标签，不能重命名或补齐其他模型的输出"
        )


def label_metadata(label: str) -> LabelDefinition:
    try:
        return LABEL_DEFINITIONS[LABEL_INDEX[label]]
    except KeyError as exc:
        raise KeyError(f"unknown CheXpert label: {label}") from exc


def parse_chexpert_value(value: Any) -> tuple[float, float]:
    """Return (target, mask) without converting uncertain labels to negatives.

    CheXpert uses 1/0/-1/blank. Positive and negative observations are valid
    training targets. Uncertain and missing observations have mask 0 and are
    excluded from the masked loss.
    """

    if value is None:
        return 0.0, 0.0
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in {"nan", "none", "null"}:
            return 0.0, 0.0
        try:
            value = float(text)
        except ValueError as exc:
            raise ValueError(f"invalid CheXpert label value: {value!r}") from exc
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid CheXpert label value: {value!r}") from exc
    if isnan(number):
        return 0.0, 0.0
    if number == 1.0:
        return 1.0, 1.0
    if number == 0.0:
        return 0.0, 1.0
    if number == -1.0:
        return 0.0, 0.0
    raise ValueError(f"CheXpert labels must be -1, 0, 1, or blank; got {value!r}")


def encode_row(row: Mapping[str, Any]) -> tuple[list[float], list[float]]:
    targets: list[float] = []
    mask: list[float] = []
    for label in CHEXPERT_LABELS:
        target, valid = parse_chexpert_value(row.get(label))
        targets.append(target)
        mask.append(valid)
    return targets, mask
