"""DenseNet-121 model, checkpoint format, and pretrained-weight provenance.

Competition rule 1 only allows non-medical open-source pretrained weights. The
torchvision ImageNet DenseNet-121 checkpoint is used exclusively; no medical
pretraining, continued pretraining, or self-supervised training is performed in
this package.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

from .labels import LABEL_IDS

CHECKPOINT_FORMAT = "cxr-first-checkpoint-v1"


def pretrained_weight_metadata() -> dict[str, str]:
    """Provenance of the torchvision ImageNet DenseNet-121 weights.

    The URL and licence are recorded for competition appendix A (A1-A5). The
    SHA-256 is computed after downloading and storing the file under
    ``pretrain_model/`` in the submission package.
    """

    from torchvision import models

    weights = models.DenseNet121_Weights.IMAGENET1K_V1
    return {
        "name": "DenseNet-121 (torchvision official ImageNet weights)",
        "url": str(weights.url),
        "upstream_dataset": "ImageNet-1K (natural images, non-medical)",
        "license": "BSD-3-Clause (torchvision model weights)",
        "enum": str(weights),
    }


def build_model(num_labels: int = len(LABEL_IDS), *, pretrained: bool = False) -> Any:
    """Build DenseNet-121 with a fresh ``num_labels`` classifier head."""

    from torch import nn
    from torchvision import models

    weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
    network = models.densenet121(weights=weights)
    network.classifier = nn.Linear(network.classifier.in_features, num_labels)
    return network


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_checkpoint(
    path: str | Path,
    *,
    model: Any,
    config: Mapping[str, Any],
    label_ids: tuple[int, ...] = LABEL_IDS,
) -> dict[str, Any]:
    """Persist model weights plus the inference contract and return metadata."""

    import torch

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": CHECKPOINT_FORMAT,
        "model_state": model.state_dict(),
        "config": dict(config),
        "label_ids": list(label_ids),
        "pretrained": pretrained_weight_metadata(),
    }
    torch.save(payload, destination)
    payload["checkpoint_sha256"] = file_sha256(destination)
    payload["checkpoint_bytes"] = destination.stat().st_size
    return payload


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load a checkpoint saved by :func:`save_checkpoint`.

    The checkpoint is a local, trusted training artifact; ``weights_only=False``
    is explicit because the payload also carries plain JSON-like metadata.
    """

    import torch

    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"unsupported checkpoint format in {path}")
    label_ids = tuple(int(item) for item in payload.get("label_ids", ()))
    if label_ids != LABEL_IDS:
        raise ValueError(f"checkpoint labels must be {list(LABEL_IDS)}, got {list(label_ids)}")
    return payload


def load_model_from_checkpoint(path: str | Path) -> tuple[Any, dict[str, Any]]:
    """Restore the network and its saved inference configuration."""

    payload = load_checkpoint(path)
    config = dict(payload["config"])
    model = build_model(len(payload["label_ids"]), pretrained=False)
    model.load_state_dict(payload["model_state"])
    return model, config
