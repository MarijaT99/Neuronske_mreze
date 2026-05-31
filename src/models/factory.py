"""Model factory — build any model from a config-style spec.

Keeps construction config-driven (project rule: no hardcoded model choices).
A model spec is a dict like::

    {"type": "baseline_cnn", "dropout": 0.5}
    {"type": "resnet50", "mode": "finetune", "unfreeze_blocks": 2, "pretrained": true}
    {"type": "efficientnet_b0", "mode": "feature_extraction"}

``num_classes`` is supplied separately (it depends on the task: 38 for the flat
baseline, num_species for the species head, k for a per-species disease head).
"""

from __future__ import annotations

import torch.nn as nn

from src.models.baseline_cnn import BaselineCNN
from src.models.transfer import SUPPORTED as TRANSFER_ARCHS
from src.models.transfer import TransferModel

BASELINE_TYPES = {"baseline_cnn", "cnn", "baseline"}


def build_model(spec: dict, num_classes: int) -> nn.Module:
    """Instantiate a model from a spec dict and a class count."""
    spec = dict(spec)  # shallow copy; we pop 'type'
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
