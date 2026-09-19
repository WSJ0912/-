from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

SUPPORTED_EXTENSIONS = {".dcm", ".dicom", ".png", ".jpg", ".jpeg"}


@dataclass(frozen=True)
class ImportValidation:
    accepted: bool
    file_type: str
    modality: str
    body_part: str
    view_position: str
    adult_confirmed: bool
    burned_in_reviewed: bool
    reasons: tuple[str, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)
    metadata: dict[str, str] = field(default_factory=dict)


def _parse_age(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip().upper()
    if not text:
        return None
    try:
        if text.endswith("Y"):
            return int(text[:-1])
        return int(float(text))
    except ValueError:
        return None


def _quality_reasons(width: int, height: int) -> list[str]:
    if width < 128 or height < 128:
        return ["image_resolution_too_small"]
    ratio = width / height if height else 0
    if ratio < 0.35 or ratio > 3.0:
        return ["severe_crop_or_invalid_aspect_ratio"]
    return []


def inspect_file(
    path: str | Path,
    *,
    adult_confirmed: bool = False,
    chest_confirmed: bool = False,
    view_confirmed: bool = False,
    burned_in_reviewed: bool = False,
) -> ImportValidation:
    """Validate admission before a file can reach the model.

    PNG/JPEG have no reliable view or age metadata, so the caller must provide
    explicit human confirmations. Unknown DICOM position, pediatric age,
    lateral views, non-chest body parts, and unreadable files are rejected.
    """

    file_path = Path(path)
    extension = file_path.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        return ImportValidation(False, "unknown", "UNKNOWN", "UNKNOWN", "UNKNOWN", False, False, ("unsupported_format",))
    if not file_path.is_file():
        return ImportValidation(False, "unknown", "UNKNOWN", "UNKNOWN", "UNKNOWN", False, False, ("file_not_found",))
    if extension in {".dcm", ".dicom"}:
        return _inspect_dicom(file_path, adult_confirmed, view_confirmed, burned_in_reviewed)
    return _inspect_raster(
        file_path,
        adult_confirmed,
        chest_confirmed,
        view_confirmed,
        burned_in_reviewed,
    )


def _inspect_raster(
    path: Path,
    adult_confirmed: bool,
    chest_confirmed: bool,
    view_confirmed: bool,
    burned_in_reviewed: bool,
) -> ImportValidation:
    try:
        from PIL import Image

        with Image.open(path) as image:
            width, height = image.size
            image.verify()
    except Exception:
        return ImportValidation(False, "raster", "UNKNOWN", "UNKNOWN", "UNKNOWN", adult_confirmed, burned_in_reviewed, ("cannot_decode_image",))
    reasons = _quality_reasons(width, height)
    if not adult_confirmed:
        reasons.append("adult_confirmation_required")
    if not chest_confirmed:
        reasons.append("chest_radiograph_confirmation_required")
    if not view_confirmed:
        reasons.append("ap_pa_confirmation_required")
    if not burned_in_reviewed:
        reasons.append("burned_in_text_review_required")
    return ImportValidation(
        not reasons,
        "raster",
        "UNKNOWN",
        "CHEST" if chest_confirmed else "UNKNOWN",
        "AP_OR_PA_PENDING" if view_confirmed else "UNKNOWN",
        adult_confirmed,
        burned_in_reviewed,
        tuple(reasons),
        ("raster_metadata_reencoded_on_commit",),
        {"width": str(width), "height": str(height)},
    )


def _inspect_dicom(path: Path, adult_confirmed: bool, view_confirmed: bool, burned_in_reviewed: bool) -> ImportValidation:
    try:
        import pydicom

        dataset = pydicom.dcmread(path, stop_before_pixels=True, force=False)
    except ImportError:
        return ImportValidation(False, "dicom", "UNKNOWN", "UNKNOWN", "UNKNOWN", adult_confirmed, burned_in_reviewed, ("pydicom_not_installed",))
    except Exception:
        return ImportValidation(False, "dicom", "UNKNOWN", "UNKNOWN", "UNKNOWN", adult_confirmed, burned_in_reviewed, ("cannot_decode_dicom",))
    modality = str(getattr(dataset, "Modality", "UNKNOWN")).upper()
    body_part = str(getattr(dataset, "BodyPartExamined", "UNKNOWN")).upper()
    view = str(getattr(dataset, "ViewPosition", "UNKNOWN")).upper()
    age = _parse_age(getattr(dataset, "PatientAge", None))
    burned = str(getattr(dataset, "BurnedInAnnotation", "UNKNOWN")).upper()
    reasons: list[str] = []
    if modality not in {"DX", "CR"}:
        reasons.append("unsupported_modality")
    if body_part not in {"CHEST", "THORAX"}:
        reasons.append("not_a_chest_study")
    if view not in {"AP", "PA"}:
        reasons.append("unsupported_or_unknown_view")
    if view == "LATERAL":
        reasons.append("lateral_view_not_supported")
    if age is not None and age < 18:
        reasons.append("pediatric_case_not_supported")
    if age is None and not adult_confirmed:
        reasons.append("adult_confirmation_required")
    if age is None and adult_confirmed:
        pass
    if burned == "YES" and not burned_in_reviewed:
        reasons.append("burned_in_text_requires_redaction_review")
    if burned == "UNKNOWN" and not burned_in_reviewed:
        reasons.append("burned_in_text_review_required")
    rows = int(getattr(dataset, "Rows", 0) or 0)
    columns = int(getattr(dataset, "Columns", 0) or 0)
    frames = int(getattr(dataset, "NumberOfFrames", 1) or 1)
    samples_per_pixel = int(getattr(dataset, "SamplesPerPixel", 1) or 1)
    if frames != 1:
        reasons.append("multiframe_dicom_not_supported")
    if samples_per_pixel != 1:
        reasons.append("color_dicom_not_supported")
    reasons.extend(_quality_reasons(columns, rows))
    return ImportValidation(
        not reasons,
        "dicom",
        modality,
        "CHEST" if body_part in {"CHEST", "THORAX"} else body_part,
        view,
        bool(age is not None and age >= 18) or adult_confirmed,
        burned_in_reviewed or burned == "NO",
        tuple(dict.fromkeys(reasons)),
        ("DICOM metadata will be replaced with a derived deidentified copy",),
        {
            "age": str(age) if age is not None else "UNKNOWN",
            "burnedInAnnotation": burned,
            "numberOfFrames": str(frames),
            "samplesPerPixel": str(samples_per_pixel),
        },
    )


def load_grayscale(path: str | Path) -> np.ndarray:
    file_path = Path(path)
    if file_path.suffix.lower() in {".dcm", ".dicom"}:
        try:
            import pydicom

            dataset = pydicom.dcmread(file_path, force=False)
            pixels = dataset.pixel_array.astype(np.float32)
            if str(getattr(dataset, "PhotometricInterpretation", "MONOCHROME2")).upper() == "MONOCHROME1":
                pixels = pixels.max() - pixels
        except ImportError as exc:
            raise RuntimeError("pydicom is required for DICOM inference") from exc
    else:
        try:
            from PIL import Image

            with Image.open(file_path) as image:
                pixels = np.asarray(image.convert("L"), dtype=np.float32)
        except Exception as exc:
            raise ValueError("cannot decode raster image") from exc
    if pixels.ndim != 2 or pixels.size == 0:
        raise ValueError("image must decode to a non-empty 2D grayscale array")
    low, high = np.percentile(pixels, [1, 99])
    if high <= low:
        low, high = float(pixels.min()), float(pixels.max())
    normalized = np.clip((pixels - low) / max(high - low, 1e-6), 0, 1)
    return normalized.astype(np.float32)


def preprocess_for_model(path: str | Path, image_size: int = 320) -> np.ndarray:
    pixels = load_grayscale(path)
    try:
        from PIL import Image

        image = Image.fromarray(np.round(pixels * 255).astype(np.uint8), mode="L")
        image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
    except ImportError as exc:
        raise RuntimeError("Pillow is required for model preprocessing") from exc
    # ImageNet grayscale adaptation: repeat the intended luminance statistics,
    # while keeping a single-channel ONNX contract.
    array = (array - 0.485) / 0.229
    return array[None, None, :, :].astype(np.float32)
