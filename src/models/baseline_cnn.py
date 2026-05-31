"""Baseline CNN trained from scratch.

A deliberately simple 4-block convolutional network — the "from scratch"
reference point against which transfer-learning models are compared. Kept small
on purpose (project rule: the baseline must be simple; no 5-6 block custom nets).

Architecture: 4 conv blocks (Conv-BN-ReLU x2 + MaxPool), widths 32/64/128/256,
then global average pooling + dropout + linear head. GAP (instead of large FC
layers) keeps the parameter count modest and is robust to the input size.

Used for both the flat 38-class baseline and as a per-species disease head in the
hierarchical baseline — just vary ``num_classes``.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _ConvBlock(nn.Module):
    """Two Conv-BN-ReLU layers followed by 2x2 max pooling."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BaselineCNN(nn.Module):
    """4-block CNN with global average pooling and a dropout-regularized head.

    Parameters
    ----------
    num_classes:
        Number of output logits.
    in_channels:
        Input image channels (3 for RGB).
    widths:
        Channel widths of the 4 conv blocks.
    dropout:
        Dropout probability before the final linear layer.
    """

    def __init__(
        self,
        num_classes: int,
        *,
        in_channels: int = 3,
        widths: tuple[int, int, int, int] = (32, 64, 128, 256),
        dropout: float = 0.5,
    ):
        super().__init__()
        if num_classes < 1:
            raise ValueError("num_classes must be >= 1")

        blocks: list[nn.Module] = []
        prev = in_channels
        for w in widths:
            blocks.append(_ConvBlock(prev, w))
            prev = w
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(prev, num_classes),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)
