from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pydicom
from inference_service.core import PlatformCore
from inference_service.deid import deidentify_dicom, reencode_raster
from inference_service.preprocess import inspect_file
from inference_service.security import Role, UserSession
from PIL import Image
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import (
    DigitalXRayImageStorageForPresentation,
    ExplicitVRLittleEndian,
    generate_uid,
)
from test_utils import case_directory


def write_synthetic_dicom(
    path: Path,
    *,
    body_part: str = "CHEST",
    view_position: str = "PA",
    age: str = "045Y",
    burned_in: str = "YES",
    number_of_frames: int = 1,
) -> tuple[str, str, str]:
    sop_uid, study_uid, series_uid = generate_uid(), generate_uid(), generate_uid()
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = DigitalXRayImageStorageForPresentation
    file_meta.MediaStorageSOPInstanceUID = sop_uid
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.SourceApplicationEntityTitle = "SYNTHETIC_AE"
    dataset = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = DigitalXRayImageStorageForPresentation
    dataset.SOPInstanceUID = sop_uid
    dataset.StudyInstanceUID = study_uid
    dataset.SeriesInstanceUID = series_uid
    dataset.Modality = "DX"
    dataset.BodyPartExamined = body_part
    dataset.ViewPosition = view_position
    dataset.PatientAge = age
    dataset.PatientName = "Synthetic^Person"
    dataset.PatientID = "TEST-IDENTIFIER"
    dataset.AccessionNumber = "TEST-ACCESSION"
    dataset.BurnedInAnnotation = burned_in
    dataset.Rows = 256
    dataset.Columns = 256
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 0
    if number_of_frames != 1:
        dataset.NumberOfFrames = number_of_frames
    dataset.PixelData = np.full((256, 256), 1000, dtype=np.uint16).tobytes()
    dataset.add_new((0x0011, 0x1010), "LO", "PRIVATE TEST VALUE")
    dataset.save_as(path, enforce_file_format=True)
    return sop_uid, study_uid, series_uid


class DataSafetyTests(unittest.TestCase):
    def test_raster_requires_explicit_human_confirmations(self) -> None:
        directory = case_directory("safety-admission")
        source = directory / "source.jpg"
        Image.new("L", (512, 512), 120).save(source, exif=b"metadata")
        rejected = inspect_file(source)
        self.assertFalse(rejected.accepted)
        self.assertIn("adult_confirmation_required", rejected.reasons)
        self.assertIn("chest_radiograph_confirmation_required", rejected.reasons)
        accepted = inspect_file(
            source,
            adult_confirmed=True,
            chest_confirmed=True,
            view_confirmed=True,
            burned_in_reviewed=True,
        )
        self.assertTrue(accepted.accepted)

    def test_raster_reencode_discards_metadata(self) -> None:
        directory = case_directory("safety-reencode")
        source = directory / "source.jpg"
        target = directory / "clean.png"
        Image.new("L", (256, 256), 120).save(source, exif=b"metadata")
        reencode_raster(source, target)
        with Image.open(target) as image:
            self.assertFalse(bool(image.getexif()))

    def test_core_refuses_prediction_without_model(self) -> None:
        root = case_directory("safety-core")
        source = root / "source.png"
        Image.new("L", (256, 256), 120).save(source)
        runtime_root = root / "runtime"
        database_path = runtime_root / "platform.sqlite3"
        if database_path.exists():
            database_path.unlink()
        core = PlatformCore(runtime_root)
        admin = UserSession(core.create_initial_admin("admin", "a-strong-test-password"), Role.ADMIN)
        doctor_id = core.create_doctor(admin, "doctor", "another-strong-password")
        doctor = UserSession(doctor_id, Role.DOCTOR)
        staged = core.stage_imports(doctor, [str(source)])[0]
        study = core.commit_import(
            doctor,
            staged["stagingId"],
            adult_confirmed=True,
            chest_confirmed=True,
            view_confirmed=True,
            view_position="PA",
            burned_in_reviewed=True,
            rectangles=[],
        )
        with self.assertRaises(RuntimeError):
            core.infer(doctor, study["studyId"])

    def test_dicom_deidentification_replaces_uids_and_applies_redaction(self) -> None:
        directory = case_directory("safety-dicom")
        source = directory / "source.dcm"
        target = directory / "derived.dcm"
        original_uids = write_synthetic_dicom(source)
        inspection = inspect_file(source, burned_in_reviewed=True)
        self.assertTrue(inspection.accepted)
        deidentify_dicom(
            source,
            target,
            [{"left": 10, "top": 20, "right": 30, "bottom": 40}],
        )
        derived = pydicom.dcmread(target)
        self.assertFalse(hasattr(derived, "PatientName"))
        self.assertFalse(hasattr(derived, "PatientID"))
        self.assertFalse(any(element.tag.is_private for element in derived.iterall()))
        self.assertNotEqual(str(derived.SOPInstanceUID), original_uids[0])
        self.assertNotEqual(str(derived.StudyInstanceUID), original_uids[1])
        self.assertNotEqual(str(derived.SeriesInstanceUID), original_uids[2])
        self.assertEqual(str(derived.BurnedInAnnotation), "NO")
        self.assertEqual(derived.file_meta.TransferSyntaxUID, ExplicitVRLittleEndian)
        self.assertFalse(hasattr(derived.file_meta, "SourceApplicationEntityTitle"))
        self.assertTrue(np.all(derived.pixel_array[20:40, 10:30] == 0))

        with self.assertRaises(ValueError):
            deidentify_dicom(
                source,
                target,
                [{"left": 300, "top": 300, "right": 320, "bottom": 320}],
            )

    def test_multiframe_dicom_is_rejected(self) -> None:
        directory = case_directory("safety-multiframe")
        source = directory / "multiframe.dcm"
        write_synthetic_dicom(source, burned_in="NO", number_of_frames=2)
        inspection = inspect_file(source)
        self.assertFalse(inspection.accepted)
        self.assertIn("multiframe_dicom_not_supported", inspection.reasons)

    def test_unknown_body_part_and_corrupt_file_are_removed_from_staging(self) -> None:
        directory = case_directory("safety-hard-rejection")
        unknown = directory / "unknown.dcm"
        write_synthetic_dicom(unknown, body_part="UNKNOWN", burned_in="NO")
        self.assertIn(
            "not_a_chest_study",
            inspect_file(unknown, burned_in_reviewed=True).reasons,
        )

        corrupt = directory / "corrupt.jpg"
        corrupt.write_bytes(b"not an image")
        runtime_root = directory / "runtime"
        core = PlatformCore(runtime_root)
        admin = UserSession(core.create_initial_admin("admin", "a-strong-test-password"), Role.ADMIN)
        doctor = UserSession(
            core.create_doctor(admin, "doctor", "another-strong-password"),
            Role.DOCTOR,
        )
        staged = core.stage_imports(doctor, [str(corrupt)])[0]
        self.assertNotIn("stagingId", staged)
        self.assertEqual(list(core.paths.staging.iterdir()), [])

    def test_duplicate_source_is_rejected_before_second_staging_copy(self) -> None:
        directory = case_directory("safety-duplicate")
        source = directory / "source.png"
        Image.new("L", (256, 256), 120).save(source)
        core = PlatformCore(directory / "runtime")
        admin = UserSession(core.create_initial_admin("admin", "a-strong-test-password"), Role.ADMIN)
        doctor = UserSession(
            core.create_doctor(admin, "doctor", "another-strong-password"),
            Role.DOCTOR,
        )
        first = core.stage_imports(doctor, [str(source)])[0]
        pending_duplicate = core.stage_imports(doctor, [str(source)])[0]
        self.assertEqual(pending_duplicate["reasons"], ["duplicate_file"])
        self.assertNotIn("stagingId", pending_duplicate)
        core.commit_import(
            doctor,
            first["stagingId"],
            adult_confirmed=True,
            chest_confirmed=True,
            view_confirmed=True,
            view_position="PA",
            burned_in_reviewed=True,
            rectangles=[],
        )
        second = core.stage_imports(doctor, [str(source)])[0]
        self.assertEqual(second["reasons"], ["duplicate_file"])
        self.assertNotIn("stagingId", second)


if __name__ == "__main__":
    unittest.main()
