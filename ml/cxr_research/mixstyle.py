from __future__ import annotations

try:
    import torch
    from torch import Tensor, nn
except ImportError:  # pragma: no cover
    torch = None
    Tensor = object  # type: ignore[assignment,misc]
    nn = object  # type: ignore[assignment,misc]


if torch is not None:

    class MixStyle(nn.Module):
        """Mix instance-level feature statistics for domain generalization."""

        def __init__(self, p: float = 0.5, alpha: float = 0.1, eps: float = 1e-6) -> None:
            super().__init__()
            if not 0 <= p <= 1:
                raise ValueError("p must be between 0 and 1")
            if alpha <= 0:
                raise ValueError("alpha must be positive")
            self.p = float(p)
            self.alpha = float(alpha)
            self.eps = float(eps)

        def forward(self, x: Tensor) -> Tensor:
            if not self.training or torch.rand((), device=x.device).item() > self.p:
                return x
            if x.ndim != 4:
                raise ValueError("MixStyle expects [batch, channels, height, width]")
            batch = x.shape[0]
            if batch < 2:
                return x
            mean = x.mean(dim=(2, 3), keepdim=True)
            variance = x.var(dim=(2, 3), keepdim=True, unbiased=False)
            std = (variance + self.eps).sqrt()
            normalized = (x - mean) / std
            permutation = torch.randperm(batch, device=x.device)
            lam = torch.distributions.Beta(self.alpha, self.alpha).sample((batch, 1, 1, 1)).to(x.device)
            mixed_mean = lam * mean + (1.0 - lam) * mean[permutation]
            mixed_std = lam * std + (1.0 - lam) * std[permutation]
            return normalized * mixed_std + mixed_mean

else:

    class MixStyle:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("MixStyle requires PyTorch")
