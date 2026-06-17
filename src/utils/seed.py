"""Global seeding for reproducibility.

Sets seeds for ``random``, ``numpy`` and ``torch`` (CPU + CUDA) and enables
deterministic cuDNN. Import-safe: ``torch`` is imported lazily so this module
can be used in environments where torch is not installed (e.g. for the local
data-split tooling).
"""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int = 42, *, deterministic: bool = True) -> int:
    """Seed all RNGs. Returns the seed for convenience/logging."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Deterministic cuDNN guarantees bit-reproducible convolutions but forces
        # slow algorithms (ResNet-50 is ~5-10x slower on a T4). Set the env var
        # PDH_CUDNN_BENCHMARK=1 to trade that exact determinism for cuDNN's fast
        # autotuned kernels — seeds still fix weights, data order and augmentation,
        # so runs stay reproducible up to minor conv-algorithm numerical noise.
        if deterministic and not os.environ.get("PDH_CUDNN_BENCHMARK"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
    except ImportError:
        # torch not present (e.g. local split tooling) — numpy/random still seeded.
        pass

    return seed


def seed_worker(worker_id: int) -> None:
    """DataLoader ``worker_init_fn`` — gives each worker a distinct, seeded RNG.

    Use together with a seeded ``generator`` passed to ``DataLoader`` for fully
    reproducible shuffling/augmentation across workers.
    """
    import torch

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int = 42):
    """Return a seeded ``torch.Generator`` for DataLoader shuffling."""
    import torch

    g = torch.Generator()
    g.manual_seed(seed)
    return g
