"""Model factory — gradi bilo koji model iz config-spec-a.

Konstrukcija ostaje config-driven (pravilo projekta: bez hardkodovanih izbora
modela). Model spec je dict poput::

    {"type": "baseline_cnn", "dropout": 0.5}
    {"type": "resnet50", "mode": "finetune", "unfreeze_blocks": 2, "pretrained": true}
    {"type": "efficientnet_b0", "mode": "feature_extraction"}

``num_classes`` se prosleđuje zasebno (zavisi od zadatka: 38 za flat baseline,
num_species za species head, k za disease head po species-u).
"""

from __future__ import annotations

import torch.nn as nn

from src.models.baseline_cnn import BaselineCNN
from src.models.transfer import SUPPORTED as TRANSFER_ARCHS
from src.models.transfer import TransferModel

BASELINE_TYPES = {"baseline_cnn", "cnn", "baseline"}


def build_model(spec: dict, num_classes: int) -> nn.Module:
    """Instancira model na osnovu spec dict-a i broja klasa."""
    spec = dict(spec)  # plitka kopija; izvlačimo (pop) 'type'
    model_type = spec.pop("type", None)
    if model_type is None:
        raise ValueError("Model spec must include a 'type' key.")

    if model_type in BASELINE_TYPES:
        return BaselineCNN(num_classes=num_classes, **spec)

    if model_type in TRANSFER_ARCHS:
        return TransferModel(arch=model_type, num_classes=num_classes, **spec)

    raise ValueError(
        f"Unknown model type {model_type!r}. "
        f"Expected one of {sorted(BASELINE_TYPES)} or {sorted(TRANSFER_ARCHS)}."
    )
