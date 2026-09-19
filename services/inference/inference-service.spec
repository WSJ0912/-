# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path
import sys

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs


project_root = Path(SPECPATH).resolve().parents[1]
datas = collect_data_files("reportlab")
python_prefix = Path(sys.prefix).resolve()
openssl_binaries = []
for dll_name in ("libcrypto-3-x64.dll", "libssl-3-x64.dll"):
    candidates = (
        python_prefix / "Library" / "bin" / dll_name,
        python_prefix / "DLLs" / dll_name,
        python_prefix / dll_name,
    )
    source = next((candidate for candidate in candidates if candidate.is_file()), None)
    if source is not None:
        openssl_binaries.append((str(source), "."))
binaries = collect_dynamic_libs("onnxruntime") + openssl_binaries
hiddenimports = [
    "uvicorn.logging",
    "uvicorn.loops.auto",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan.on",
]
a = Analysis(
    [str(project_root / "services" / "inference" / "launcher.py")],
    pathex=[
        str(project_root / "ml"),
        str(project_root / "services" / "inference"),
    ],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=[
        "torch",
        "torchvision",
        "tkinter",
        "matplotlib",
        "pandas",
        "scipy",
        "sklearn",
        "onnx",
        "onnxruntime.quantization",
        "onnxruntime.tools",
        "onnxruntime.transformers",
    ],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    name="inference-service",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    exclude_binaries=True,
)

collection = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    name="inference-service",
)
