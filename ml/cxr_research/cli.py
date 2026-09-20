from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import (
    build_medmodel,
    read_medmodel_manifest,
    sha256_file,
    validate_experiment_bundle,
)
from .data import load_chexpert_csv
from .export import create_manifest, export_onnx, verify_pytorch_onnx, write_manifest
from .external import (
    TOTAL_DATA_BUDGET_BYTES,
    evaluate_mimic_external,
    sample_mimic_external,
    verify_external_manifest,
)
from .labels import CHEXPERT_LABELS
from .manifests import (
    build_chexpert_split_manifest,
    read_chexpert_split_manifest,
)
from .manifests import (
    write_manifest as write_split_manifest,
)
from .metrics import evaluate_multilabel
from .model import build_model
from .runner import aggregate_experiment, run_training
from .thresholds import select_thresholds
from .training import TrainingConfig, formal_run_plan


def _json_file(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CheXpert 14 项可复现胸片研究工具")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate-medmodel", help="校验 .medmodel ZIP 包")
    validate.add_argument("path", type=Path)

    experiment = sub.add_parser("validate-experiment", help="校验 .medexperiment 包或 JSON")
    experiment.add_argument("path", type=Path)

    metrics = sub.add_parser("evaluate-npz", help="从 NPZ 计算聚合指标")
    metrics.add_argument("path", type=Path, help="包含 y_true、scores、mask 的 NPZ")
    metrics.add_argument("--thresholds", type=Path)

    prepare = sub.add_parser("prepare-chexpert", help="读取官方 CSV 并锁定患者级划分")
    prepare.add_argument("csv", type=Path)
    prepare.add_argument("output", type=Path)
    prepare.add_argument("--image-root", type=Path)
    prepare.add_argument("--seed", type=int, default=20240913)

    plan = sub.add_parser("formal-plan", help="输出固定的六次正式训练计划")
    plan.add_argument("--output", type=Path)

    train = sub.add_parser("train", help="运行一次基线或 MixStyle 训练")
    train.add_argument("split_manifest", type=Path)
    train.add_argument("output_directory", type=Path)
    train.add_argument("--method", choices=("baseline", "mixstyle"), required=True)
    train.add_argument("--seed", type=int, choices=TrainingConfig().seeds, required=True)
    train.add_argument("--device")
    train.add_argument("--batch-size", type=_positive_int, default=TrainingConfig().batch_size)
    train.add_argument("--max-epochs", type=_positive_int, default=TrainingConfig().max_epochs)
    train.add_argument("--patience", type=_positive_int, default=TrainingConfig().patience)
    train.add_argument("--learning-rate", type=float, default=TrainingConfig().learning_rate)
    train.add_argument("--bootstrap-resamples", type=_positive_int, default=2000)
    train.add_argument(
        "--smoke-test",
        action="store_true",
        help="仅验证环境；关闭 ImageNet 权重且结果不得作为正式实验",
    )

    sample = sub.add_parser("sample-mimic", help="不读取结果标签地生成 MIMIC 外部测试清单")
    sample.add_argument("metadata_csv", type=Path)
    sample.add_argument("output_manifest", type=Path)
    sample.add_argument(
        "--chexpert-manifest",
        type=Path,
        required=True,
        help="prepare-chexpert 生成的清单，用于执行 20 GiB 总空间预算",
    )
    sample.add_argument("--image-root", type=Path)
    sample.add_argument("--seed", type=int, default=20240913)
    sample.add_argument("--target-patients", type=_positive_int, default=8000)
    sample.add_argument(
        "--max-bytes",
        type=_positive_int,
        help="可选的 MIMIC 额外上限；实际值不会超过总预算剩余额度",
    )
    sample.add_argument(
        "--total-budget-bytes",
        type=_positive_int,
        default=TOTAL_DATA_BUDGET_BYTES,
    )
    sample.add_argument("--patient-column", default="subject_id")
    sample.add_argument("--study-column", default="study_id")
    sample.add_argument("--image-column", default="path")
    sample.add_argument("--view-column", default="ViewPosition")
    sample.add_argument("--age-column", default="age")

    verify = sub.add_parser("verify-mimic", help="核验 MIMIC 清单和独立 SHA-256 锁")
    verify.add_argument("manifest", type=Path)

    external = sub.add_parser("evaluate-mimic", help="对封存 MIMIC 清单执行一次最终外部评价")
    external.add_argument("manifest", type=Path)
    external.add_argument("labels_csv", type=Path)
    external.add_argument("medmodel", type=Path)
    external.add_argument("output_directory", type=Path)
    external.add_argument("--image-root", type=Path, required=True)
    external.add_argument("--patient-column", default="subject_id")
    external.add_argument("--study-column", default="study_id")
    external.add_argument("--batch-size", type=_positive_int, default=16)
    external.add_argument("--bootstrap-resamples", type=_positive_int, default=2000)

    export = sub.add_parser("export-onnx", help="导出并校验 ONNX，再生成模型清单")
    export.add_argument("checkpoint", type=Path)
    export.add_argument("output_directory", type=Path)
    export.add_argument("--thresholds", type=Path, required=True)
    export.add_argument(
        "--selection",
        type=Path,
        required=True,
        help="select-deployment 生成的部署选择记录",
    )
    export.add_argument("--model-id", required=True)
    export.add_argument("--version", required=True)
    export.add_argument("--training-commit", required=True)
    export.add_argument("--data-source", action="append", required=True)
    export.add_argument("--license", dest="license_name", required=True)
    export.add_argument("--package", type=Path)

    select = sub.add_parser(
        "select-deployment",
        help="仅按三个正式 MixStyle 运行的 CheXpert 验证指标选择部署 checkpoint",
    )
    select.add_argument("results", nargs=3, type=Path)
    select.add_argument("--output", type=Path, required=True)

    package = sub.add_parser("package-medmodel", help="从已准备目录构建 .medmodel")
    package.add_argument("source_directory", type=Path)
    package.add_argument("output", type=Path)

    aggregate = sub.add_parser("aggregate-experiment", help="聚合固定种子结果并生成实验包")
    aggregate.add_argument("results", nargs="+", type=Path)
    aggregate.add_argument("--experiment-id", required=True)
    aggregate.add_argument("--dataset-manifest", type=Path, required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    aggregate.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="仅用于开发检查；正式实验必须包含 2 种方法各 3 个固定种子",
    )
    return parser


def _run_export(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("export-onnx requires PyTorch") from exc

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    method = str(state.get("method", ""))
    if method != "mixstyle" or state.get("formal") is not True:
        raise ValueError("deployment export requires a formal MixStyle checkpoint")
    if tuple(state.get("labels", ())) != CHEXPERT_LABELS:
        raise ValueError("checkpoint labels are not the fixed CheXpert 14 protocol")
    selection = _validate_deployment_checkpoint(args.selection, args.checkpoint, state)
    model = build_model(mixstyle=method == "mixstyle", pretrained=False)
    model.load_state_dict(state["state_dict"])

    output = args.output_directory
    output.mkdir(parents=True, exist_ok=True)
    onnx_path = export_onnx(model, output / "model.onnx")
    verification = verify_pytorch_onnx(
        model,
        onnx_path,
        torch.zeros((1, 1, 320, 320), dtype=torch.float32),
    )
    if not verification["within_tolerance"]:
        raise RuntimeError(
            "PyTorch/ONNX probability error exceeds 1e-4: "
            f"{verification['max_probability_error']}"
        )
    verification_path = output / "onnx-verification.json"
    verification_path.write_text(
        json.dumps(verification, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    labels_path = output / "labels.json"
    labels_path.write_text(
        json.dumps({"labels": CHEXPERT_LABELS}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    thresholds = _json_file(args.thresholds)
    manifest = create_manifest(
        onnx_path,
        args.model_id,
        args.version,
        method,
        args.training_commit,
        [int(state["seed"])],
        list(args.data_source),
        args.license_name,
        thresholds,
        {"labels.json": labels_path, "onnx-verification.json": verification_path},
    )
    write_manifest(output / "manifest.json", manifest)
    package_path = build_medmodel(output, args.package) if args.package else None
    return {
        "onnx": str(onnx_path),
        "manifest": str(output / "manifest.json"),
        "package": str(package_path) if package_path else None,
        "verification": verification,
        "deploymentSelection": selection,
    }


def _validate_formal_results(results: list[dict[str, Any]]) -> None:
    expected = {(entry["method"], int(entry["seed"])) for entry in formal_run_plan()}
    actual = {(str(entry.get("method")), int(entry.get("seed", -1))) for entry in results}
    if actual != expected or len(results) != len(expected):
        raise ValueError(
            "formal aggregation requires exactly baseline and mixstyle for seeds 17, 29, and 43"
        )


def _select_deployment(results: list[Path], output: Path) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for result_path in results:
        result = _json_file(result_path)
        if result.get("method") != "mixstyle" or result.get("formal") is not True:
            raise ValueError("deployment selection accepts formal MixStyle results only")
        seed = int(result.get("seed", -1))
        score = result.get("validationPrimaryMacroAuroc")
        if seed not in TrainingConfig().seeds or not isinstance(score, (int, float)):
            raise ValueError("deployment candidates require fixed seeds and validation AUROC")
        score = float(score)
        if not math.isfinite(score):
            raise ValueError("deployment validation AUROC must be finite")
        checkpoint = result_path.parent / "best.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"checkpoint missing for seed {seed}: {checkpoint}")
        candidates.append(
            {
                "method": "mixstyle",
                "seed": seed,
                "validationPrimaryMacroAuroc": score,
                "result": str(result_path.resolve()),
                "resultSha256": sha256_file(result_path),
                "checkpoint": str(checkpoint.resolve()),
                "checkpointSha256": sha256_file(checkpoint),
            }
        )
    if {item["seed"] for item in candidates} != set(TrainingConfig().seeds):
        raise ValueError("deployment selection requires MixStyle seeds 17, 29, and 43")
    selected = max(candidates, key=lambda item: (item["validationPrimaryMacroAuroc"], -item["seed"]))
    record = {
        "schemaVersion": "deployment-selection-v1",
        "selectionMetric": "CheXpert validation primary five-label macro AUROC",
        "candidates": candidates,
        "selected": selected,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return record


def _validate_deployment_checkpoint(
    selection_path: Path,
    checkpoint_path: Path,
    checkpoint_state: dict[str, Any],
) -> dict[str, Any]:
    selection = _json_file(selection_path)
    if selection.get("schemaVersion") != "deployment-selection-v1":
        raise ValueError("invalid deployment selection record")
    candidates = selection.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 3:
        raise ValueError("deployment selection must contain three candidates")
    if {candidate.get("seed") for candidate in candidates if isinstance(candidate, dict)} != set(TrainingConfig().seeds):
        raise ValueError("deployment selection must contain the three fixed seeds")
    for candidate in candidates:
        if not isinstance(candidate, dict) or candidate.get("method") != "mixstyle":
            raise ValueError("deployment selection contains an invalid candidate")
        result_path = Path(str(candidate.get("result", "")))
        candidate_checkpoint = Path(str(candidate.get("checkpoint", "")))
        if (
            not result_path.is_file()
            or candidate.get("resultSha256") != sha256_file(result_path)
            or not candidate_checkpoint.is_file()
            or candidate.get("checkpointSha256") != sha256_file(candidate_checkpoint)
        ):
            raise ValueError("deployment selection candidate files changed after selection")
    selected = selection.get("selected")
    if not isinstance(selected, dict):
        raise ValueError("deployment selection does not contain a selected candidate")
    expected = max(
        candidates,
        key=lambda item: (float(item["validationPrimaryMacroAuroc"]), -int(item["seed"])),
    )
    if selected != expected:
        raise ValueError("deployment selection is not the best validation candidate")
    if Path(str(selected.get("checkpoint", ""))).resolve() != checkpoint_path.resolve():
        raise ValueError("checkpoint is not the validation-selected deployment candidate")
    if selected.get("checkpointSha256") != sha256_file(checkpoint_path):
        raise ValueError("selected deployment checkpoint hash mismatch")
    if selected.get("method") != "mixstyle" or selected.get("seed") != checkpoint_state.get("seed"):
        raise ValueError("deployment selection does not match checkpoint metadata")
    return selection


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "validate-medmodel":
        manifest = read_medmodel_manifest(args.path)
        result = {
            "valid": True,
            "modelId": manifest["modelId"],
            "version": manifest["version"],
            "modelSha256": manifest["modelSha256"],
        }
    elif args.command == "validate-experiment":
        if args.path.suffix.lower() == ".medexperiment":
            from .artifacts import read_medexperiment

            bundle = read_medexperiment(args.path)
        else:
            bundle = _json_file(args.path)
            validate_experiment_bundle(bundle)
        result = {"valid": True, "experimentId": bundle["experimentId"]}
    elif args.command == "evaluate-npz":
        arrays = np.load(args.path)
        y_true, scores = arrays["y_true"], arrays["scores"]
        mask = arrays["mask"] if "mask" in arrays else np.ones_like(y_true)
        thresholds = (
            _json_file(args.thresholds)
            if args.thresholds
            else select_thresholds(y_true, scores, CHEXPERT_LABELS, mask=mask)
        )
        result = evaluate_multilabel(y_true, scores, CHEXPERT_LABELS, mask, thresholds)
    elif args.command == "prepare-chexpert":
        records = load_chexpert_csv(args.csv, args.image_root)
        manifest = build_chexpert_split_manifest(records, args.csv, seed=args.seed)
        write_split_manifest(args.output, manifest)
        result = {
            "output": str(args.output),
            "manifestSha256": manifest["manifestSha256"],
            "counts": {name: len(values) for name, values in manifest["records"].items()},
        }
    elif args.command == "formal-plan":
        result = {"formal": True, "runs": list(formal_run_plan())}
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
    elif args.command == "train":
        config = replace(
            TrainingConfig(),
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            patience=args.patience,
            learning_rate=args.learning_rate,
        )
        result = run_training(
            args.split_manifest,
            args.output_directory,
            method=args.method,
            seed=args.seed,
            pretrained=not args.smoke_test,
            config=config,
            device_name=args.device,
            bootstrap_resamples=args.bootstrap_resamples,
            formal=not args.smoke_test,
        )
    elif args.command == "sample-mimic":
        read_chexpert_split_manifest(args.chexpert_manifest)
        chexpert_manifest = _json_file(args.chexpert_manifest)
        chexpert_bytes = int(chexpert_manifest.get("imageBytes", -1))
        if chexpert_bytes < 0:
            raise ValueError("CheXpert manifest does not contain imageBytes")
        if args.total_budget_bytes > TOTAL_DATA_BUDGET_BYTES:
            raise ValueError("v0.1 total data budget cannot exceed 20 GiB")
        remaining_bytes = args.total_budget_bytes - chexpert_bytes
        if remaining_bytes <= 0:
            raise ValueError("CheXpert data already exhausts the configured total budget")
        mimic_budget = min(args.max_bytes or remaining_bytes, remaining_bytes)
        sampled = sample_mimic_external(
            args.metadata_csv,
            args.output_manifest,
            image_root=args.image_root,
            patient_column=args.patient_column,
            study_column=args.study_column,
            image_column=args.image_column,
            view_column=args.view_column,
            age_column=args.age_column,
            seed=args.seed,
            target_patients=args.target_patients,
            max_bytes=mimic_budget,
        )
        result = {
            "output": str(args.output_manifest),
            "selectedPatients": sampled["selectedPatients"],
            "consideredPatients": sampled["consideredPatients"],
            "missingImages": sampled["missingImages"],
            "totalBytes": sampled["totalBytes"],
            "chexpertBytes": chexpert_bytes,
            "mimicBudgetBytes": mimic_budget,
            "totalBudgetBytes": args.total_budget_bytes,
            "manifestSha256": sampled["manifestSha256"],
        }
    elif args.command == "verify-mimic":
        manifest = verify_external_manifest(args.manifest)
        result = {
            "valid": True,
            "selectedPatients": manifest["selectedPatients"],
            "manifestSha256": manifest["manifestSha256"],
        }
    elif args.command == "evaluate-mimic":
        result = evaluate_mimic_external(
            args.manifest,
            args.labels_csv,
            args.medmodel,
            args.output_directory,
            image_root=args.image_root,
            patient_column=args.patient_column,
            study_column=args.study_column,
            batch_size=args.batch_size,
            bootstrap_resamples=args.bootstrap_resamples,
        )
    elif args.command == "export-onnx":
        result = _run_export(args)
    elif args.command == "select-deployment":
        result = _select_deployment(args.results, args.output)
    elif args.command == "package-medmodel":
        result = {"package": str(build_medmodel(args.source_directory, args.output))}
    elif args.command == "aggregate-experiment":
        read_chexpert_split_manifest(args.dataset_manifest)
        dataset_manifest = _json_file(args.dataset_manifest)
        results = [_json_file(path) for path in args.results]
        if not args.allow_incomplete:
            _validate_formal_results(results)
        package = aggregate_experiment(
            results,
            args.experiment_id,
            str(dataset_manifest["manifestSha256"]),
            args.output,
            require_formal=not args.allow_incomplete,
        )
        result = {"package": str(package), "formal": not args.allow_incomplete}
    else:  # pragma: no cover
        return 2
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0
