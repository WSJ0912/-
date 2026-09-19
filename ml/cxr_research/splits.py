from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class PatientSplit:
    train: tuple[dict[str, Any], ...]
    validation: tuple[dict[str, Any], ...]
    test: tuple[dict[str, Any], ...]
    seed: int

    def patient_sets(self, patient_key: str = "patient_id") -> dict[str, set[str]]:
        return {
            name: {str(row[patient_key]) for row in rows}
            for name, rows in (("train", self.train), ("validation", self.validation), ("test", self.test))
        }


def split_patient_records(
    records: Sequence[Mapping[str, Any]],
    seed: int = 20240913,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    patient_key: str = "patient_id",
) -> PatientSplit:
    """Deterministically split complete patients, never individual images."""

    if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-8 or any(r <= 0 for r in ratios):
        raise ValueError("ratios must be three positive values summing to 1")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if patient_key not in record or record[patient_key] in (None, ""):
            raise ValueError(f"missing patient key: {patient_key}")
        groups[str(record[patient_key])].append(dict(record))
    patients = list(groups)
    random.Random(seed).shuffle(patients)
    total = len(patients)
    raw_counts = [ratio * total for ratio in ratios]
    counts = [int(value) for value in raw_counts]
    for index in sorted(range(3), key=lambda i: raw_counts[i] - counts[i], reverse=True)[: total - sum(counts)]:
        counts[index] += 1
    # Keep a non-empty split when enough patients are available.
    if total >= 3:
        counts = [max(1, count) for count in counts]
        while sum(counts) > total:
            largest = max(range(3), key=lambda i: counts[i])
            if counts[largest] > 1:
                counts[largest] -= 1
            else:
                break
    train_end = counts[0]
    validation_end = train_end + counts[1]
    assignments = (
        patients[:train_end],
        patients[train_end:validation_end],
        patients[validation_end:],
    )
    result = PatientSplit(
        tuple(row for patient in assignments[0] for row in groups[patient]),
        tuple(row for patient in assignments[1] for row in groups[patient]),
        tuple(row for patient in assignments[2] for row in groups[patient]),
        seed,
    )
    assert_no_patient_overlap(result, patient_key=patient_key)
    return result


def assert_no_patient_overlap(split: PatientSplit, patient_key: str = "patient_id") -> None:
    sets = split.patient_sets(patient_key)
    names = tuple(sets)
    for index, left_name in enumerate(names):
        for right_name in names[index + 1 :]:
            overlap = sets[left_name] & sets[right_name]
            if overlap:
                raise AssertionError(f"patient leakage between {left_name} and {right_name}: {sorted(overlap)}")


def manifest_sha256(records: Iterable[Mapping[str, Any]]) -> str:
    """Hash a canonical record list; images and model outputs are not included."""

    normalized = [dict(record) for record in records]
    normalized.sort(key=lambda item: json.dumps(item, ensure_ascii=True, sort_keys=True))
    payload = json.dumps(normalized, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()
