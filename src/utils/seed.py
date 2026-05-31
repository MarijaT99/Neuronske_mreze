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
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
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
