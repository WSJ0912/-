from __future__ import annotations

from typing import Any

from .labels import CHEXPERT_LABELS
from .mixstyle import MixStyle

try:
    import torch
    import torch.nn.functional as F
    from torch import Tensor, nn
    from torchvision.models import DenseNet121_Weights, densenet121
except ImportError:  # pragma: no cover
    torch = None
    Tensor = Any  # type: ignore[misc,assignment]
    nn = None
    F = None
    DenseNet121_Weights = None
    densenet121 = None


if torch is not None:

    class DenseNet121MultiLabel(nn.Module):
        """ImageNet-compatible DenseNet-121 with exactly 14 CheXpert outputs."""

        def __init__(
            self,
            mixstyle: bool = False,
            mixstyle_p: float = 0.5,
            mixstyle_alpha: float = 0.1,
            pretrained: bool = False,
        ) -> None:
            super().__init__()
            if pretrained:
                weights = DenseNet121_Weights.DEFAULT
            else:
                weights = None
            backbone = densenet121(weights=weights)
            original = backbone.features.conv0
            replacement = nn.Conv2d(1, original.out_channels, kernel_size=7, stride=2, padding=3, bias=False)
            if pretrained:
                with torch.no_grad():
                    replacement.weight.copy_(original.weight.mean(dim=1, keepdim=True))
            backbone.features.conv0 = replacement
            self.features = backbone.features
            self.classifier = nn.Linear(backbone.classifier.in_features, len(CHEXPERT_LABELS))
            self.use_mixstyle = bool(mixstyle)
            self.mixstyle_1 = MixStyle(mixstyle_p, mixstyle_alpha) if mixstyle else nn.Identity()
            self.mixstyle_2 = MixStyle(mixstyle_p, mixstyle_alpha) if mixstyle else nn.Identity()

        def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
            for name, layer in self.features.named_children():
                x = layer(x)
                if self.use_mixstyle and name == "denseblock1":
                    x = self.mixstyle_1(x)
                elif self.use_mixstyle and name == "denseblock2":
                    x = self.mixstyle_2(x)
            feature_map = F.relu(x, inplace=False)
            pooled = F.adaptive_avg_pool2d(feature_map, (1, 1)).flatten(1)
            logits = self.classifier(pooled)
            return logits, feature_map

        def forward_logits(self, x: Tensor) -> Tensor:
            return self(x)[0]

        def cam(self, feature_map: Tensor) -> Tensor:
            """Return one non-negative class activation map per label."""

            weights = self.classifier.weight.view(1, len(CHEXPERT_LABELS), -1, 1, 1)
            maps = (feature_map.unsqueeze(1) * weights).sum(dim=2)
            return maps.clamp_min(0)


else:

    class DenseNet121MultiLabel:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("DenseNet121MultiLabel requires torch and torchvision")


def build_model(mixstyle: bool = False, pretrained: bool = False) -> DenseNet121MultiLabel:
    return DenseNet121MultiLabel(mixstyle=mixstyle, pretrained=pretrained)
