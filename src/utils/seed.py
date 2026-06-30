"""Globalno postavljanje seed-a radi reproducibilnosti.

Postavlja seed za ``random``, ``numpy`` i ``torch`` (CPU + CUDA) i uključuje
deterministički cuDNN. Bezbedno za import: ``torch`` se učitava lenjo, pa se ovaj
modul može koristiti i u okruženjima gde torch nije instaliran (npr. za lokalne
alate za split podataka).
"""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int = 42, *, deterministic: bool = True) -> int:
    """Postavi seed za sve RNG-ove. Vraća seed radi praktičnosti/logovanja."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Deterministički cuDNN garantuje bit-reproducibilne konvolucije, ali nameće
        # spore algoritme (ResNet-50 je ~5-10x sporiji na T4). Postavi env promenljivu
        # PDH_CUDNN_BENCHMARK=1 da tu strogu determinističnost zameniš za brze,
        # autotuned cuDNN kernele - seed i dalje fiksira težine, redosled podataka i
        # augmentation, pa runovi ostaju reproducibilni do minornog numeričkog šuma
        # konvolucionog algoritma.
        if deterministic and not os.environ.get("PDH_CUDNN_BENCHMARK"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
    except ImportError:
        # torch nije prisutan (npr. lokalni alati za split) - numpy/random su i dalje seed-ovani.
        pass

    return seed


def seed_worker(worker_id: int) -> None:
    """DataLoader ``worker_init_fn`` - daje svakom worker-u zaseban, seed-ovan RNG.

    Koristi zajedno sa seed-ovanim ``generator``-om prosleđenim ``DataLoader``-u za
    potpuno reproducibilno mešanje/augmentation kroz sve worker-e.
    """
    import torch

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int = 42):
    """Vrati seed-ovan ``torch.Generator`` za mešanje u DataLoader-u."""
    import torch

    g = torch.Generator()
    g.manual_seed(seed)
    return g
