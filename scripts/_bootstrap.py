"""Čini lokalna pokretanja otpornim bez obzira na CWD i kodiranje putanje.

Uvezi ovo kao *prvu* liniju svakog skripta u ``scripts/``::

    import _bootstrap  # noqa: F401  -- repo root on sys.path + CWD pinned

Zašto ovo postoji (lokalni razvoj na Windows-u + OneDrive):

* Putanja repozitorijuma sadrži ne-ASCII karakter ("Republički") i nalazi se u
  folderu sinhronizovanom preko OneDrive-a. PowerShell-ov radni direktorijum se
  povremeno resetuje na *roditeljski* direktorijum repozitorijuma između poziva,
  što kvari pozive sa relativnim putanjama i relativne ``--out-dir``/``--report``
  argumente.
* Razrešavanje repo root-a iz ``__file__`` i ``chdir`` na njega čini svaku
  relativnu putanju determinističkom bez obzira na to odakle je interpreter
  pokrenut.

Ovaj modul dira samo stanje lokalno za proces (``sys.path``, CWD, jednu env
promenljivu). Ne uvozi nijedan projektni kod i bezbedan je za dvostruki uvoz
(idempotentan).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# ``scripts/_bootstrap.py`` -> repo root je jedan direktorijum iznad.
ROOT = Path(__file__).resolve().parent.parent


def setup() -> Path:
    """Fiksira sys.path + CWD na repo root. Vraća razrešeni root."""
    root_str = str(ROOT)

    # 1. Omogući da ``import src...`` radi iz bilo kog CWD.
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    # 2. Fiksira CWD na repo root da bi se relativne putanje razrešavale
    #    deterministički čak i ako je PowerShell odlutao u roditeljski direktorijum.
    if Path.cwd().resolve() != ROOT:
        os.chdir(ROOT)

    # 3. Ne zatrpavaj OneDrive-sinhronizovano stablo sa __pycache__.
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

    return ROOT


setup()
