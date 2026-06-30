"""Sampleri i težine klasa za rad sa neuravnoteženim podacima.

PlantVillage dataset je neuravnotežen na oba nivoa (odnos za 38 klasa ~36x,
odnos za species ~49x). Podržavamo tri nezavisne strategije, uporedive u
eksperimentima:

1. ``class_weight`` u CrossEntropyLoss     -> :func:`compute_class_weights`
2. ``WeightedRandomSampler`` u DataLoader-u -> :func:`make_weighted_sampler`
3. Focal loss                              -> vidi :mod:`src.training.losses`

Sve se računa **samo iz brojeva na train splitu** (nikada val/test) da bi se
izbeglo curenje podataka.
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
    """Loss težine po klasi iz celobrojnih labela.

    sheme:
    * ``"inverse"``           — w_c ∝ 1 / count_c
    * ``"inverse_sqrt"``      — w_c ∝ 1 / sqrt(count_c) (blaže)
    * ``"effective"``         — class-balanced (Cui et al. 2019): (1-beta)/(1-beta^n_c)

    Težine se normalizuju na srednju vrednost 1 kako bi skala loss-a ostala uporediva.
    Vraća float niz dužine ``num_classes``.
    """
    labels = np.asarray(labels)
    if num_classes is None:
        num_classes = int(labels.max()) + 1
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)  # zaštita od praznih klasa

    if scheme == "inverse":
        w = 1.0 / counts
    elif scheme == "inverse_sqrt":
        w = 1.0 / np.sqrt(counts)
    elif scheme == "effective":
        eff_num = 1.0 - np.power(beta, counts)
        w = (1.0 - beta) / eff_num
    else:
        raise ValueError(f"Unknown scheme {scheme!r} (inverse|inverse_sqrt|effective).")

    w = w / w.mean()  # normalizuj na srednju vrednost 1
    return w.astype(np.float32)


def make_weighted_sampler(labels: np.ndarray, num_classes: int | None = None):
    """Napravi ``WeightedRandomSampler`` koji bira klase približno uniformno.

    Težina svakog uzorka je 1/count_njegove_klase, pa je u proseku svaka klasa
    podjednako zastupljena po epoch-u. ``num_samples`` se postavlja na
    ``len(labels)`` tako da "epoch" zadrži uobičajenu veličinu.
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
