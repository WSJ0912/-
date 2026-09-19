from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .artifacts import model_manifest_sha256, sha256_file, validate_model_manifest
from .labels import CHEXPERT_LABELS


def export_onnx(
    model: Any,
    output_path: str | Path,
    opset_version: int = 17,
    verify_input_shape: tuple[int, int, int, int] = (1, 1, 320, 320),
) -> Path:
    """Export logits and feature maps; never substitutes a missing model."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("ONNX export requires PyTorch") from exc
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    example = torch.zeros(verify_input_shape, dtype=torch.float32)
    with torch.no_grad():
        torch.onnx.export(
            model,
            example,
            destination,
            input_names=["image"],
            output_names=["logits", "cam_features"],
            dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}, "cam_features": {0: "batch"}},
            opset_version=opset_version,
            do_constant_folding=True,
            dynamo=False,
        )
    return destination


def verify_pytorch_onnx(
    model: Any,
    onnx_path: str | Path,
    sample: Any,
    tolerance: float = 1e-4,
) -> dict[str, float | bool]:
    """Compare probabilities, not logits, with the deployment runtime."""

    try:
        import numpy as np
        import onnxruntime as ort
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("ONNX verification requires numpy, PyTorch and onnxruntime") from exc
    model.eval()
    sample_tensor = sample if isinstance(sample, torch.Tensor) else torch.as_tensor(sample, dtype=torch.float32)
    with torch.no_grad():
        torch_logits, _ = model(sample_tensor)
        torch_probabilities = torch.sigmoid(torch_logits).cpu().numpy()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_logits, _ = session.run(None, {session.get_inputs()[0].name: sample_tensor.cpu().numpy()})
    onnx_probabilities = 1.0 / (1.0 + np.exp(-np.asarray(onnx_logits)))
    error = float(np.max(np.abs(torch_probabilities - onnx_probabilities)))
    return {"max_probability_error": error, "within_tolerance": bool(error <= tolerance)}


def create_manifest(
    model_path: str | Path,
    model_id: str,
    version: str,
    method: str,
    training_commit: str,
    seeds: list[int],
    data_sources: list[str],
    license_name: str,
    thresholds: Mapping[str, float],
    extra_files: Mapping[str, str | Path] | None = None,
) -> dict[str, Any]:
    model_file = Path(model_path)
    files: dict[str, str] = {"model.onnx": sha256_file(model_file)}
    for name, path in (extra_files or {}).items():
        files[name] = sha256_file(path)
    manifest = {
        "schemaVersion": "medmodel-1",
        "modelId": model_id,
        "version": version,
        "architecture": "densenet121",
        "labels": list(CHEXPERT_LABELS),
        "input": {"width": 320, "height": 320, "channels": 1, "normalization": "imagenet-gray"},
        "thresholds": dict(thresholds),
        "preprocessing": {"viewPositions": ["AP", "PA"], "adultOnly": True},
        "training": {"commit": training_commit, "seeds": list(seeds), "method": method},
        "dataSources": list(data_sources),
        "license": license_name,
        "files": files,
        "modelSha256": files["model.onnx"],
    }
    manifest["manifestSha256"] = model_manifest_sha256(manifest)
    validate_model_manifest(manifest)
    return manifest


def write_manifest(path: str | Path, manifest: Mapping[str, Any]) -> None:
    validate_model_manifest(manifest)
    Path(path).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
