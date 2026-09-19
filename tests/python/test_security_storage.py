from __future__ import annotations

import unittest

from inference_service.security import (
    LocalTokenStore,
    Role,
    hash_password,
    require_permission,
    verify_password,
)
from inference_service.storage import Database, utc_now
from test_utils import case_directory


class SecurityStorageTests(unittest.TestCase):
    def test_argon2id_hash_and_roles(self) -> None:
        password_hash = hash_password("a-strong-test-password")
        self.assertTrue(password_hash.startswith("$argon2id$"))
        self.assertTrue(verify_password(password_hash, "a-strong-test-password"))
        self.assertFalse(verify_password(password_hash, "wrong-password"))
        require_permission(Role.DOCTOR, "confirm_report")
        with self.assertRaises(PermissionError):
            require_permission(Role.ADMIN, "confirm_report")

    def test_process_token_is_rotated(self) -> None:
        store = LocalTokenStore()
        old = store.token
        self.assertTrue(store.validate(old))
        new = store.rotate()
        self.assertNotEqual(old, new)
        self.assertFalse(store.validate(old))

    def test_report_revision_cannot_be_overwritten_after_confirmation(self) -> None:
        root = case_directory("storage")
        database_path = root / "test.sqlite3"
        if database_path.exists():
            database_path.unlink()
        db = Database(database_path)
        user_id = db.create_user("doctor", "doctor", "hash")
        study = {"studyId": "ST-1", "createdAt": utc_now(), "modality": "DX", "bodyPart": "CHEST", "viewPosition": "AP", "adultConfirmed": True, "deidentifiedPath": "derived.png", "importHash": "a" * 64, "status": "ready"}
        db.save_study(study)
        db.save_report_revision({"reportId": "RPT-1", "studyId": "ST-1", "revision": 1, "authorId": user_id, "createdAt": utc_now(), "body": "draft", "status": "draft"})
        db.confirm_report("RPT-1", 1)
        with self.assertRaises(ValueError):
            db.save_report_revision({"reportId": "RPT-1", "studyId": "ST-1", "revision": 1, "authorId": user_id, "createdAt": utc_now(), "body": "overwrite", "status": "draft"})


if __name__ == "__main__":
    unittest.main()
