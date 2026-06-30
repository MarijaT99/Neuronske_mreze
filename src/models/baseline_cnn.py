"""Baseline CNN treniran od nule (from scratch).

Namerno jednostavna konvoluciona mreža sa 4 bloka - "from scratch" referentna
tačka u odnosu na koju se porede modeli zasnovani na transfer learning-u. Namerno
je držana malom (pravilo projekta: baseline mora biti jednostavan; bez custom
mreža sa 5-6 blokova).

Arhitektura: 4 konvoluciona bloka (Conv-BN-ReLU x2 + MaxPool), širine 32/64/128/256,
zatim global average pooling + dropout + linearni head. GAP (umesto velikih FC
slojeva) drži broj parametara skromnim i otporan je na veličinu ulaza.

Koristi se i za flat baseline sa 38 klasa i kao disease head po species-u u
hijerarhijskom baseline-u - dovoljno je samo promeniti ``num_classes``.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _ConvBlock(nn.Module):
    """Dva Conv-BN-ReLU sloja praćena 2x2 max pooling-om."""

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
    """CNN sa 4 bloka, global average pooling-om i dropout-regularizovanim head-om.

    Parameters
    ----------
    num_classes:
        Broj izlaznih logits-a.
    in_channels:
        Broj kanala ulazne slike (3 za RGB).
    widths:
        Širine kanala 4 konvoluciona bloka.
    dropout:
        Verovatnoća dropout-a pre poslednjeg linearnog sloja.
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
