from __future__ import annotations

import hashlib
import secrets
from pathlib import Path
from typing import Iterable

import numpy as np


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _new_uid() -> str:
    try:
        from pydicom.uid import generate_uid

        return str(generate_uid())
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pydicom is required for DICOM deidentification") from exc


def _apply_rectangles(array: np.ndarray, rectangles: Iterable[dict[str, int]]) -> np.ndarray:
    result = array.copy()
    height, width = result.shape[:2]
    for rectangle in rectangles:
        try:
            left = max(0, min(width, int(rectangle["left"])))
            top = max(0, min(height, int(rectangle["top"])))
            right = max(left, min(width, int(rectangle["right"])))
            bottom = max(top, min(height, int(rectangle["bottom"])))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("redaction rectangles require integer left/top/right/bottom") from exc
        if right <= left or bottom <= top:
            raise ValueError("redaction rectangles must overlap the image with positive area")
        result[top:bottom, left:right] = 0
    return result


def deidentify_dicom(
    source: str | Path,
    destination: str | Path,
    rectangles: Iterable[dict[str, int]] = (),
) -> str:
    """Write a derived DICOM and return its SHA-256; source is never modified."""

    try:
        import pydicom
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pydicom is required for DICOM deidentification") from exc
    source_path, destination_path = Path(source), Path(destination)
    dataset = pydicom.dcmread(source_path, force=False)
    # Keep only fields needed for admission, display and pixel decoding.
    allowed_names = {
        "SOPClassUID", "SOPInstanceUID", "StudyInstanceUID", "SeriesInstanceUID",
        "Modality", "BodyPartExamined", "ViewPosition", "Rows", "Columns",
        "PhotometricInterpretation", "SamplesPerPixel", "BitsAllocated", "BitsStored",
        "HighBit", "PixelRepresentation", "PixelData", "SeriesNumber", "InstanceNumber",
        "WindowCenter", "WindowWidth", "RescaleIntercept", "RescaleSlope", "BurnedInAnnotation",
    }
    for element in list(dataset):
        if element.keyword not in allowed_names:
            del dataset[element.tag]
    for key in ("PatientName", "PatientID", "AccessionNumber", "StudyID"):
        if hasattr(dataset, key):
            delattr(dataset, key)
    dataset.remove_private_tags()
    dataset.SOPInstanceUID = _new_uid()
    dataset.StudyInstanceUID = _new_uid()
    dataset.SeriesInstanceUID = _new_uid()
    dataset.BurnedInAnnotation = "NO"
    from pydicom.dataset import FileMetaDataset
    from pydicom.uid import PYDICOM_IMPLEMENTATION_UID, ExplicitVRLittleEndian

    source_transfer_syntax = getattr(
        getattr(dataset, "file_meta", None),
        "TransferSyntaxUID",
        ExplicitVRLittleEndian,
    )
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = dataset.SOPClassUID
    file_meta.MediaStorageSOPInstanceUID = dataset.SOPInstanceUID
    file_meta.TransferSyntaxUID = source_transfer_syntax
    file_meta.ImplementationClassUID = PYDICOM_IMPLEMENTATION_UID
    dataset.file_meta = file_meta
    if rectangles:
        pixels = dataset.pixel_array
        redacted = _apply_rectangles(np.asarray(pixels), rectangles)
        little_endian_dtype = redacted.dtype.newbyteorder("<")
        dataset.PixelData = redacted.astype(little_endian_dtype, copy=False).tobytes()
        dataset.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        dataset["PixelData"].VR = "OB" if int(dataset.BitsAllocated) <= 8 else "OW"
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_as(destination_path, enforce_file_format=True)
    return sha256_file(destination_path)


def reencode_raster(
    source: str | Path,
    destination: str | Path,
    format_name: str | None = None,
    rectangles: Iterable[dict[str, int]] = (),
) -> str:
    """Decode and re-encode PNG/JPEG so EXIF and source metadata are discarded."""

    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Pillow is required for raster deidentification") from exc
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        clean_array = np.asarray(image.convert("L"))
        clean = Image.fromarray(_apply_rectangles(clean_array, rectangles), mode="L")
        clean.save(destination_path, format=format_name or destination_path.suffix.lstrip(".").upper() or "PNG", exif=b"")
    return sha256_file(destination_path)


def temporary_study_id() -> str:
    return "ST-" + secrets.token_hex(10).upper()
