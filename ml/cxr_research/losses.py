from __future__ import annotations

from typing import Any

try:  # Torch is an optional dependency for metadata/metric-only installs.
    import torch
    import torch.nn.functional as F
    from torch import Tensor
except ImportError:  # pragma: no cover - exercised only in minimal installs
    torch = None
    Tensor = Any  # type: ignore[misc,assignment]
    F = None


def masked_bce_with_logits(
    logits: Tensor,
    targets: Tensor,
    mask: Tensor,
    class_weights: Tensor | None = None,
    max_class_weight: float = 10.0,
) -> Tensor:
    """Calculate BCE only on observed CheXpert labels.

    Class weights are clipped to a configurable upper bound to prevent a rare
    label from dominating a minibatch. An all-masked minibatch returns a zero
    scalar connected to ``logits`` so the training loop remains differentiable.
    """

    if torch is None:
        raise RuntimeError("masked_bce_with_logits requires PyTorch")
    if logits.shape != targets.shape or logits.shape != mask.shape:
        raise ValueError("logits, targets, and mask must have identical shapes")
    if logits.ndim != 2:
        raise ValueError("expected [batch, labels] tensors")
    valid = mask.to(dtype=logits.dtype).clamp(0, 1)
    if class_weights is None:
        weights = torch.ones(logits.shape[-1], device=logits.device, dtype=logits.dtype)
    else:
        if class_weights.numel() != logits.shape[-1]:
            raise ValueError("class_weights length must equal the number of labels")
        weights = class_weights.to(device=logits.device, dtype=logits.dtype).clamp(
            min=0.0, max=max_class_weight
        )
    elementwise = F.binary_cross_entropy_with_logits(
        logits,
        targets.to(logits.dtype),
        reduction="none",
        pos_weight=weights,
    )
    denominator = valid.sum()
    return (elementwise * valid).sum() / denominator.clamp_min(1.0)


def capped_inverse_prevalence(
    targets: Tensor,
    mask: Tensor,
    max_weight: float = 10.0,
) -> Tensor:
    """Compute positive-class inverse prevalence using only observed labels."""

    if torch is None:
        raise RuntimeError("capped_inverse_prevalence requires PyTorch")
    valid = mask.to(dtype=targets.dtype)
    positives = (targets * valid).sum(dim=0)
    observed = valid.sum(dim=0)
    negatives = (observed - positives).clamp_min(0.0)
    weights = negatives / positives.clamp_min(1.0)
    weights = torch.where(observed > 0, weights, torch.ones_like(weights))
    return weights.clamp(min=1.0, max=max_weight)
