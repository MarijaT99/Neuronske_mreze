"""Imbalance-handling samplers and class weights.

The PlantVillage dataset is imbalanced at both levels (38-class ratio ~36x,
species ratio ~49x). We support three independent strategies, comparable in
experiments:

1. ``class_weight`` in CrossEntropyLoss   -> :func:`compute_class_weights`
2. ``WeightedRandomSampler`` in DataLoader -> :func:`make_weighted_sampler`
3. Focal loss                              -> see :mod:`src.training.losses`

All are computed from **train-split counts only** (never val/test) to avoid leakage.
"""

from __future__ import annotations

import numpy as np


def compute_class_weights(
    labels: np.ndarray,
    num_classes: int | None = None,
    *,
    scheme: str = "inverse",
    beta: float = 0.9999,
) -> np.ndarray:
    """Per-class loss weights from integer labels.

    schemes:
    * ``"inverse"``           — w_c ∝ 1 / count_c
    * ``"inverse_sqrt"``      — w_c ∝ 1 / sqrt(count_c) (gentler)
    * ``"effective"``         — class-balanced (Cui et al. 2019): (1-beta)/(1-beta^n_c)

    Weights are normalized to mean 1 so the loss scale stays comparable.
    Returns a float array of length ``num_classes``.
    """
    labels = np.asarray(labels)
    if num_classes is None:
        num_classes = int(labels.max()) + 1
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)  # guard against empty classes

    if scheme == "inverse":
        w = 1.0 / counts
    elif scheme == "inverse_sqrt":
        w = 1.0 / np.sqrt(counts)
    elif scheme == "effective":
        eff_num = 1.0 - np.power(beta, counts)
        w = (1.0 - beta) / eff_num
    else:
        raise ValueError(f"Unknown scheme {scheme!r} (inverse|inverse_sqrt|effective).")

    w = w / w.mean()  # normalize to mean 1
    return w.astype(np.float32)


def make_weighted_sampler(labels: np.ndarray, num_classes: int | None = None):
    """Build a ``WeightedRandomSampler`` that draws classes ~uniformly.

    Each sample's weight is 1/count_of_its_class, so on average each class is
    equally represented per epoch. ``num_samples`` is set to ``len(labels)`` so
    an "epoch" keeps its usual size.
    """
    import torch
    from torch.utils.data import WeightedRandomSampler

    labels = np.asarray(labels)
    if num_classes is None:
        num_classes = int(labels.max()) + 1
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)

    per_class_w = 1.0 / counts
    sample_w = per_class_w[labels]
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_w, dtype=torch.double),
        num_samples=len(labels),
        replacement=True,
    )
