from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ml"))
sys.path.insert(0, str(ROOT / "services" / "inference"))
sys.path.insert(0, str(ROOT / "tests" / "python"))

from PIL import Image  # noqa: E402
from test_utils import build_synthetic_medmodel  # noqa: E402


def call(
    root_url: str,
    process_token: str,
    endpoint: str,
    *,
    method: str = "GET",
    body: dict[str, Any] | None = None,
    session_token: str | None = None,
) -> Any:
    headers = {"Authorization": f"Bearer {process_token}"}
    if session_token:
        headers["X-Session-Token"] = session_token
    payload = None
    if body is not None:
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        f"{root_url}{endpoint}",
        data=payload,
        headers=headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {endpoint} failed ({error.code}): {detail}") from error


def verify(executable: Path) -> dict[str, Any]:
    if not executable.is_file():
        raise FileNotFoundError(executable)
    root = ROOT / ".test-runtime" / f"packaged-service-verification-{os.getpid()}"
    root.mkdir(parents=True, exist_ok=False)
    try:
        runtime = root / "runtime"
        extraction = root / "pyinstaller-temp"
        extraction.mkdir()
        environment = os.environ.copy()
        environment.update({"TEMP": str(extraction), "TMP": str(extraction)})
        process = subprocess.Popen(
            [str(executable), "--root", str(runtime), "--port", "0"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=environment,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        try:
            assert process.stdout is not None
            handshake_line = process.stdout.readline().strip()
            if not handshake_line:
                assert process.stderr is not None
                raise RuntimeError(
                    "packaged service did not emit a handshake: "
                    + process.stderr.read(4000)
                )
            handshake = json.loads(handshake_line)
            if handshake.get("host") != "127.0.0.1":
                raise RuntimeError("packaged service did not bind to loopback")
            port = int(handshake["port"])
            process_token = str(handshake["processToken"])
            if not 0 < port < 65536 or len(process_token) < 32:
                raise RuntimeError("packaged service emitted an invalid handshake")
            root_url = f"http://127.0.0.1:{port}"
            health = call(root_url, process_token, "/health")

            setup = call(
                root_url,
                process_token,
                "/api/setup",
                method="POST",
                body={"username": "admin", "password": "packaged-admin-123"},
            )
            admin_token = str(setup["sessionToken"])
            call(
                root_url,
                process_token,
                "/api/users/doctors",
                method="POST",
                body={
                    "username": "doctor",
                    "password": "packaged-doctor-123",
                    "role": "doctor",
                },
                session_token=admin_token,
            )
            package = build_synthetic_medmodel(root)
            model = call(
                root_url,
                process_token,
                "/api/models/install",
                method="POST",
                body={"packagePath": str(package)},
                session_token=admin_token,
            )

            login = call(
                root_url,
                process_token,
                "/api/login",
                method="POST",
                body={"username": "doctor", "password": "packaged-doctor-123"},
            )
            doctor_token = str(login["sessionToken"])
            image_path = root / "synthetic-adult-pa.png"
            Image.new("L", (512, 512), 128).save(image_path)
            staged = call(
                root_url,
                process_token,
                "/api/import/stage",
                method="POST",
                body={"paths": [str(image_path)]},
                session_token=doctor_token,
            )[0]
            study = call(
                root_url,
                process_token,
                f"/api/import/{staged['stagingId']}/commit",
                method="POST",
                body={
                    "adultConfirmed": True,
                    "chestConfirmed": True,
                    "viewConfirmed": True,
                    "viewPosition": "PA",
                    "burnedInReviewed": True,
                    "rectangles": [],
                },
                session_token=doctor_token,
            )
            prediction = call(
                root_url,
                process_token,
                f"/api/studies/{study['studyId']}/predict",
                method="POST",
                session_token=doctor_token,
            )
            review = call(
                root_url,
                process_token,
                "/api/reviews",
                method="POST",
                body={
                    "studyId": study["studyId"],
                    "predictionId": prediction["predictionId"],
                    "decisions": {"Atelectasis": "uncertain"},
                    "notes": "合成打包服务验证",
                },
                session_token=doctor_token,
            )
            draft = call(
                root_url,
                process_token,
                "/api/reports/draft",
                method="POST",
                body={
                    "studyId": study["studyId"],
                    "body": "检查所见：合成打包服务验证。",
                    "reviewId": review["reviewId"],
                },
                session_token=doctor_token,
            )
            confirmed = call(
                root_url,
                process_token,
                "/api/reports/confirm",
                method="POST",
                body={"reportId": draft["reportId"], "revision": draft["revision"]},
                session_token=doctor_token,
            )
            pdf_path = root / "报告-合成验证.pdf"
            call(
                root_url,
                process_token,
                "/api/reports/export",
                method="POST",
                body={
                    "reportId": draft["reportId"],
                    "revision": draft["revision"],
                    "outputPath": str(pdf_path),
                },
                session_token=doctor_token,
            )
            if not pdf_path.read_bytes().startswith(b"%PDF"):
                raise RuntimeError("packaged service did not produce a valid PDF")
            call(
                root_url,
                process_token,
                "/api/logout",
                method="POST",
                session_token=doctor_token,
            )
            result = {
                "ok": True,
                "host": handshake["host"],
                "health": health["ok"],
                "modelReady": model["ready"],
                "predictionLabels": len(prediction["probabilities"]),
                "reportStatus": confirmed["status"],
                "pdfBytes": pdf_path.stat().st_size,
            }
            return result
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
    finally:
        # Keep the ignored test directory for post-failure inspection. A later
        # run recreates it deterministically via case_directory().
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the packaged inference service")
    parser.add_argument("executable", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.executable.resolve()), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
