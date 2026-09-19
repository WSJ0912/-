from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from .labels import CHEXPERT_LABELS
from .losses import masked_bce_with_logits


@dataclass(frozen=True)
class TrainingConfig:
    image_size: int = 320
    max_epochs: int = 15
    patience: int = 3
    batch_size: int = 32
    learning_rate: float = 1e-4
    max_class_weight: float = 10.0
    seeds: tuple[int, int, int] = (17, 29, 43)
    max_full_runs: int = 6

    def validate(self) -> None:
        if self.image_size != 320:
            raise ValueError("v0.1 requires 320x320 inputs")
        if self.max_epochs > 15:
            raise ValueError("the formal experiment limit is 15 epochs")
        if len(self.seeds) != 3 or self.max_full_runs < 6:
            raise ValueError("baseline and MixStyle require three fixed seeds each")
        if self.max_class_weight <= 0:
            raise ValueError("max_class_weight must be positive")


def formal_run_plan(config: TrainingConfig | None = None) -> tuple[dict[str, Any], ...]:
    cfg = config or TrainingConfig()
    cfg.validate()
    plan = tuple(
        {"method": method, "mixstyle": method == "mixstyle", "seed": seed}
        for method in ("baseline", "mixstyle")
        for seed in cfg.seeds
    )
    if len(plan) > 6:
        raise ValueError("formal GPU budget permits at most six complete training runs")
    return plan


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover
        pass


def train_epoch(model: Any, loader: Iterable[Any], optimizer: Any, class_weights: Any, device: Any = "cpu") -> float:
    """Train one epoch over batches yielding (images, targets, observed_mask)."""

    model.train()
    total_loss = 0.0
    batches = 0
    for images, targets, mask in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(images)
        loss = masked_bce_with_logits(logits, targets, mask, class_weights)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().cpu())
        batches += 1
    return total_loss / batches if batches else 0.0


def predict_loader(model: Any, loader: Iterable[Any], device: Any = "cpu") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import torch

    model.eval()
    logits: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    with torch.no_grad():
        for images, batch_targets, batch_mask in loader:
            images = images.to(device, non_blocking=True)
            batch_logits, _ = model(images)
            logits.append(batch_logits.detach().cpu().numpy())
            targets.append(batch_targets.detach().cpu().numpy())
            masks.append(batch_mask.detach().cpu().numpy())
    if not logits:
        empty = np.empty((0, len(CHEXPERT_LABELS)), dtype=np.float32)
        return empty, empty, empty
    return np.concatenate(logits), np.concatenate(targets), np.concatenate(masks)
