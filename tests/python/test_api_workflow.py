from __future__ import annotations

import unittest

from fastapi.testclient import TestClient
from inference_service.main import create_app
from PIL import Image
from test_utils import build_synthetic_medmodel, case_directory


class ApiWorkflowTests(unittest.TestCase):
    def test_local_api_end_to_end_with_real_onnx_runtime(self) -> None:
        root = case_directory("api-workflow")
        app = create_app(root / "runtime")
        process_headers = {"Authorization": f"Bearer {app.state.process_token}"}
        with TestClient(app) as client:
            self.assertEqual(client.get("/api/status", headers=process_headers).status_code, 200)
            setup = client.post(
                "/api/setup",
                headers=process_headers,
                json={"username": "admin", "password": "admin-password-123"},
            )
            self.assertEqual(setup.status_code, 200)
            admin_headers = {
                **process_headers,
                "X-Session-Token": setup.json()["sessionToken"],
            }
            created = client.post(
                "/api/users/doctors",
                headers=admin_headers,
                json={"username": "doctor", "password": "doctor-password-123", "role": "doctor"},
            )
            self.assertEqual(created.status_code, 200)

            package = build_synthetic_medmodel(root)
            installed = client.post(
                "/api/models/install",
                headers=admin_headers,
                json={"packagePath": str(package)},
            )
            self.assertEqual(installed.status_code, 200, installed.text)
            self.assertTrue(installed.json()["ready"])
            self.assertEqual(len(installed.json()["modelSha256"]), 64)

            login = client.post(
                "/api/login",
                headers=process_headers,
                json={"username": "doctor", "password": "doctor-password-123"},
            )
            self.assertEqual(login.status_code, 200)
            doctor_headers = {
                **process_headers,
                "X-Session-Token": login.json()["sessionToken"],
            }
            image_path = root / "adult-pa-test.png"
            Image.new("L", (512, 512), 128).save(image_path)
            staged = client.post(
                "/api/import/stage",
                headers=doctor_headers,
                json={"paths": [str(image_path)]},
            ).json()[0]
            self.assertNotIn("stagedPath", staged)
            self.assertNotIn("importHash", staged)
            committed = client.post(
                f"/api/import/{staged['stagingId']}/commit",
                headers=doctor_headers,
                json={
                    "adultConfirmed": True,
                    "chestConfirmed": True,
                    "viewConfirmed": True,
                    "viewPosition": "PA",
                    "burnedInReviewed": True,
                    "rectangles": [],
                },
            )
            self.assertEqual(committed.status_code, 200, committed.text)
            study_id = committed.json()["studyId"]
            self.assertNotIn("deidentifiedPath", committed.json())
            self.assertNotIn("importHash", committed.json())
            self.assertNotIn("derivedHash", committed.json())

            prediction_response = client.post(
                f"/api/studies/{study_id}/predict",
                headers=doctor_headers,
            )
            self.assertEqual(prediction_response.status_code, 200, prediction_response.text)
            prediction = prediction_response.json()
            self.assertEqual(len(prediction["probabilities"]), 14)
            self.assertTrue(all(value == 0.5 for value in prediction["probabilities"].values()))
            self.assertEqual(len(prediction["modelSha256"]), 64)

            review = client.post(
                "/api/reviews",
                headers=doctor_headers,
                json={
                    "studyId": study_id,
                    "predictionId": prediction["predictionId"],
                    "decisions": {"Atelectasis": "denied"},
                    "notes": "合成工作流测试",
                },
            )
            self.assertEqual(review.status_code, 200, review.text)

            draft = client.post(
                "/api/reports/draft",
                headers=doctor_headers,
                json={"studyId": study_id, "body": "检查所见：合成测试。"},
            )
            self.assertEqual(draft.status_code, 200, draft.text)
            self.assertEqual(draft.json()["reviewId"], review.json()["reviewId"])
            admin_confirm = client.post(
                "/api/reports/confirm",
                headers=admin_headers,
                json={"reportId": draft.json()["reportId"], "revision": 1},
            )
            self.assertEqual(admin_confirm.status_code, 403)
            confirmed = client.post(
                "/api/reports/confirm",
                headers=doctor_headers,
                json={"reportId": draft.json()["reportId"], "revision": 1},
            )
            self.assertEqual(confirmed.status_code, 200, confirmed.text)

            pdf_path = root / "report.pdf"
            exported = client.post(
                "/api/reports/export",
                headers=doctor_headers,
                json={
                    "reportId": draft.json()["reportId"],
                    "revision": 1,
                    "outputPath": str(pdf_path),
                },
            )
            self.assertEqual(exported.status_code, 200, exported.text)
            self.assertTrue(pdf_path.read_bytes().startswith(b"%PDF"))

            offline = client.post(
                "/api/assistant/report",
                headers=doctor_headers,
                json={
                    "observations": [],
                    "review": {"Atelectasis": "denied"},
                    "clinicianText": "离线模板正文",
                },
            )
            self.assertEqual(offline.status_code, 200)
            self.assertEqual(offline.json()["draft"], "离线模板正文")
            self.assertTrue(offline.json()["cautions"])

            logged_out = client.post("/api/logout", headers=doctor_headers)
            self.assertEqual(logged_out.status_code, 200)
            self.assertEqual(logged_out.json(), {"loggedOut": True})
            self.assertEqual(
                client.get("/api/studies", headers=doctor_headers).status_code,
                401,
            )


if __name__ == "__main__":
    unittest.main()
