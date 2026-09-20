from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import onnx
from cxr_research.artifacts import build_medmodel
from cxr_research.export import create_manifest
from cxr_research.labels import CHEXPERT_LABELS
from onnx import TensorProto, helper, numpy_helper

RUNTIME_ROOT = Path(__file__).resolve().parents[2] / ".test-runtime"
RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)


def case_directory(name: str) -> Path:
    path = RUNTIME_ROOT / name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_synthetic_medmodel(root: Path) -> Path:
    source = root / "test-model-source"
    source.mkdir()
    model_path = source / "model.onnx"
    labels_path = source / "labels.json"
    labels_path.write_text(
        json.dumps({"labels": CHEXPERT_LABELS}),
        encoding="utf-8",
    )
    input_info = helper.make_tensor_value_info(
        "image",
        TensorProto.FLOAT,
        [1, 1, 320, 320],
    )
    logits_info = helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 14])
    features_info = helper.make_tensor_value_info(
        "cam_features",
        TensorProto.FLOAT,
        [1, 1, 10, 10],
    )
    cams_info = helper.make_tensor_value_info(
        "cams",
        TensorProto.FLOAT,
        [1, 14, 10, 10],
    )
    logits = numpy_helper.from_array(np.zeros((1, 14), dtype=np.float32), "logits_value")
    features = numpy_helper.from_array(
        np.ones((1, 1, 10, 10), dtype=np.float32),
        "features_value",
    )
    cam_values = np.stack(
        [np.arange(100, dtype=np.float32).reshape(10, 10) + index for index in range(14)]
    )[None, ...]
    cams = numpy_helper.from_array(cam_values, "cams_value")
    graph = helper.make_graph(
        [
            helper.make_node("Constant", [], ["logits"], value=logits),
            helper.make_node("Constant", [], ["cam_features"], value=features),
            helper.make_node("Constant", [], ["cams"], value=cams),
        ],
        "synthetic-test-only-cxr",
        [input_info],
        [logits_info, features_info, cams_info],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save(model, model_path)
    manifest = create_manifest(
        model_path,
        "synthetic-test-only",
        "0.0.1-test",
        "mixstyle",
        "test-commit",
        [17],
        ["synthetic test fixture"],
        "test-only-not-for-distribution",
        {label: 0.5 for label in CHEXPERT_LABELS},
        {"labels.json": labels_path},
    )
    (source / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return build_medmodel(source, root / "synthetic-test-only.medmodel")
