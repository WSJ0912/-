"""Command line interface for the ``cxr_first`` competition pipeline."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from .data import load_split, records_to_arrays
from .labels import LABEL_IDS
from .predict import evaluate, predict, submission_summary
from .train import TrainConfig, train

DEFAULT_DATA_ROOT = Path("cxr_first") / "cxr_first"


def _add_data_root(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"folder holding train.csv/valid.csv/test.csv and the image folders (default: {DEFAULT_DATA_ROOT})",
    )


def dataset_stats(data_root: str | Path) -> dict[str, Any]:
    """Summarize splits, label prevalence, view positions, and leakage checks."""

    root = Path(data_root)
    splits: dict[str, Any] = {}
    records_by_split: dict[str, list[Any]] = {}
    for split, labelled in (("train", True), ("valid", True), ("test", False)):
        records = load_split(root / f"{split}.csv", root / split, labelled=labelled)
        records_by_split[split] = records
        stats: dict[str, Any] = {
            "images": len(records),
            "studies": len({record.study_id for record in records}),
            "subjects": len({record.subject_id for record in records}),
            "viewPositions": dict(sorted(Counter(r.view_position or "unknown" for r in records).items())),
            "patientOrientations": dict(
                sorted(Counter(r.patient_orientation or "unknown" for r in records).items())
            ),
        }
        if labelled:
            targets, _mask = records_to_arrays(records)
            stats["perClassPositiveImages"] = {
                str(label_id): int(targets[:, index].sum())
                for index, label_id in enumerate(LABEL_IDS)
            }
            stats["perClassPrevalence"] = {
                str(label_id): round(float(targets[:, index].mean()), 4)
                for index, label_id in enumerate(LABEL_IDS)
            }
        splits[split] = stats
    subject_sets = {split: {r.subject_id for r in records} for split, records in records_by_split.items()}
    patient_overlap = {
        "trainValid": len(subject_sets["train"] & subject_sets["valid"]),
        "trainTest": len(subject_sets["train"] & subject_sets["test"]),
        "validTest": len(subject_sets["valid"] & subject_sets["test"]),
    }
    if any(patient_overlap.values()):
        raise ValueError(f"patient leakage across splits: {patient_overlap}")
    return {"dataRoot": str(root), "labelIds": list(LABEL_IDS), "splits": splits, "patientOverlap": patient_overlap}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cxr-competition",
        description="Training, evaluation, and submission pipeline for the cxr_first task",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    stats_parser = subparsers.add_parser("stats", help="print dataset statistics as JSON")
    _add_data_root(stats_parser)
    stats_parser.add_argument("--output", type=Path, default=None)

    train_parser = subparsers.add_parser("train", help="fine-tune ImageNet DenseNet-121 on the train split")
    _add_data_root(train_parser)
    train_parser.add_argument("--output-dir", type=Path, default=Path("runs/cxr-first"))
    train_parser.add_argument("--image-size", type=int, default=384)
    train_parser.add_argument("--epochs", type=int, default=8)
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--lr", type=float, default=3e-4)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument("--seed", type=int, default=17)
    train_parser.add_argument("--device", default="cuda")
    train_parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    train_parser.set_defaults(pretrained=True)
    train_parser.add_argument("--num-workers", type=int, default=4)
    train_parser.add_argument("--patience", type=int, default=3)
    train_parser.add_argument("--amp", action="store_true", help="mixed-precision training on CUDA")
    train_parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="resume from a last.pt training state written by a previous run",
    )
    train_parser.add_argument("--max-train-images", type=int, default=None, help="smoke-test only")
    train_parser.add_argument("--max-valid-images", type=int, default=None, help="smoke-test only")

    predict_parser = subparsers.add_parser("predict", help="score valid/test and write submission CSV")
    _add_data_root(predict_parser)
    predict_parser.add_argument("--checkpoint", type=Path, required=True)
    predict_parser.add_argument("--split", choices=("valid", "test"), default="test")
    predict_parser.add_argument("--output", type=Path, required=True)
    predict_parser.add_argument("--batch-size", type=int, default=8)
    predict_parser.add_argument("--device", default="cuda")
    predict_parser.add_argument("--aggregate", choices=("none", "max", "mean"), default="mean")
    predict_parser.add_argument("--normalize", choices=("none", "l1"), default="l1")
    predict_parser.add_argument(
        "--gate",
        type=float,
        default=None,
        help="custom threshold written to the Gate column (default: blank, i.e. 0.5)",
    )

    evaluate_parser = subparsers.add_parser("evaluate", help="validation macro AUROC / macro F1")
    _add_data_root(evaluate_parser)
    evaluate_parser.add_argument("--checkpoint", type=Path, required=True)
    evaluate_parser.add_argument("--output", type=Path, default=None)
    evaluate_parser.add_argument("--batch-size", type=int, default=8)
    evaluate_parser.add_argument("--device", default="cpu")

    verify_parser = subparsers.add_parser("verify-submission", help="sanity-check a submission CSV")
    verify_parser.add_argument("--submission", type=Path, required=True)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "stats":
        payload = dataset_stats(args.data_root)
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text + "\n", encoding="utf-8")
        print(text)
        return 0
    if args.command == "train":
        config = TrainConfig(
            data_root=args.data_root,
            output_dir=args.output_dir,
            image_size=args.image_size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            seed=args.seed,
            device=args.device,
            pretrained=args.pretrained,
            num_workers=args.num_workers,
            patience=args.patience,
            amp=args.amp,
            resume=args.resume,
            max_train_images=args.max_train_images,
            max_valid_images=args.max_valid_images,
        )
        result = train(config)
        print(json.dumps({"bestEpoch": result["bestEpoch"], "validMacroAuc": result["validMacroAuc"]}, ensure_ascii=False))
        return 0
    if args.command == "predict":
        summary = predict(
            checkpoint_path=args.checkpoint,
            data_root=args.data_root,
            split=args.split,
            output_csv=args.output,
            batch_size=args.batch_size,
            device=args.device,
            aggregate=args.aggregate,
            normalize=args.normalize,
            gate=args.gate,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    if args.command == "evaluate":
        result = evaluate(
            checkpoint_path=args.checkpoint,
            data_root=args.data_root,
            output_json=args.output,
            batch_size=args.batch_size,
            device=args.device,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "verify-submission":
        print(json.dumps(submission_summary(args.submission), ensure_ascii=False, indent=2))
        return 0
    parser.error(f"unhandled command: {args.command}")  # pragma: no cover
    return 1  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
