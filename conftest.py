"""pytest bootstrap — čini repozitorijum uvozivim i fiksira radni direktorijum.

pytest ga automatski učitava. Garantuje da ``import src...`` radi i da je radni
direktorijum koren repozitorijuma tokom prikupljanja i izvršavanja testova, bez
obzira na to odakle je ``pytest`` pokrenut (CWD u PowerShell-u može odlutati do
roditeljskog direktorijuma pod OneDrive-om). Bez zavisnosti za testove — čista
zaštita putanje/CWD-a.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if Path.cwd().resolve() != ROOT:
    os.chdir(ROOT)

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
