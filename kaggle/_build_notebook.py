"""Regeneriše train_kaggle.ipynb sa čistim UTF-8 tekstom i isporukom putem git-clone.

Pokretanje:  python kaggle/_build_notebook.py
Upisuje kaggle/train_kaggle.ipynb. Drži se u repozitorijumu kako bi notebook bio
reproducibilan iz izvornog koda umesto ručno editovanog JSON-a (izbegava se kvarenje
enkodovanja).
"""
from __future__ import annotations

import json
from pathlib import Path


def md(*lines: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": list(_join(lines))}


def code(*lines: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": list(_join(lines))}


def _join(lines):
    """Pretvara tuple logičkih linija u ipynb izvor (svaka linija završena novim redom osim poslednje)."""
    text = "\n".join(lines)
    parts = text.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]


cells = [
    md(
        "# Treniranje na Kaggle-u - hijerarhijska klasifikacija bolesti biljaka",
        "",
        "Šablon za pokretanje eksperimenata na Kaggle Notebook-u (T4 x2 / P100). Koraci:",
        "",
        "1. Dodaj dataset `abdallahalidev/plantvillage-dataset` (koristimo `color` folder).",
        "2. Kloniraj ovaj repozitorijum u `/kaggle/working` (sa pravom upisa) da bi `src/` import-i i",
        "   skripte mogle da upisuju checkpoint-e.",
        "3. Sačuvani split je već u `data/splits/` - ponovo ga iskoristi (**nemoj** ga regenerisati).",
        "4. Istreniraj flat baseline-e / glavu za vrstu / glave za bolest, pa evaluiraj pipeline.",
        "",
        "> **Politika metrika:** macro F1 je **primarna** metrika (dataset je neuravnotežen).",
        "> Accuracy se navodi samo kao sekundarna vrednost - nikada kao glavni rezultat.",
    ),
    md(
        "## 0. Dovlačenje repozitorijuma u radni dir sa pravom upisa i podešavanje putanja",
        "",
        "Dva načina isporuke, isprobavaju se redom (bez ikakvih izmena):",
        "",
        "1. **Dataset mod (bez Interneta):** otpremi ovaj repozitorijum kao Kaggle Dataset i dodaj ga kao",
        "   Input. Ćelija ga pronalazi pod `/kaggle/input/...` (bilo koji folder koji sadrži `src/`",
        "   + `scripts/`) i kopira u `/kaggle/working` (da skripte mogu da upisuju checkpoint-e).",
        "2. **Clone mod (zahteva uključen Internet):** ako nema dataset-a sa repozitorijumom, `git clone` sa",
        "   GitHub-a. Za privatni repozitorijum dodaj Kaggle Secret `GITHUB_TOKEN` (Add-ons -> Secrets).",
        "",
        "Kaggle-ova zaključana slika već ima torch / torchvision / timm / albumentations / sklearn,",
        "pa **treniranje radi potpuno offline**. (`grad-cam` za analizu grešaka u Fazi 6 može zahtevati",
        "Internet - nije potreban za treniranje.)",
    ),
    code(
        "import sys, os, shutil, subprocess",
        "from pathlib import Path",
        "",
        "REPO_URL = 'https://github.com/MarijaT99/Neuronske_mreze.git'",
        "REPO_DIR = Path('/kaggle/working/Neuronske_mreze')",
        "",
        "def find_repo_in_input():",
        "    \"\"\"Prolazi kroz /kaggle/input (depth<=4) tražeći folder koji ima i src/ i scripts/.",
        "    Preskače 'color' stablo slika da bi ostalo brzo.\"\"\"",
        "    base = Path('/kaggle/input')",
        "    if not base.exists():",
        "        return None",
        "    stack = [(base, 0)]",
        "    while stack:",
        "        d, depth = stack.pop()",
        "        if (d / 'src').is_dir() and (d / 'scripts').is_dir():",
        "            return d",
        "        if depth < 4:",
        "            for c in d.iterdir():",
        "                if c.is_dir() and c.name != 'color':",
        "                    stack.append((c, depth + 1))",
        "    return None",
        "",
        "if not REPO_DIR.exists():",
        "    src = find_repo_in_input()",
        "    if src is not None:",
        "        print('Dataset mode: copying repo from', src)",
        "        shutil.copytree(src, REPO_DIR)",
        "    else:",
        "        print('Clone mode: git clone from GitHub (needs Internet On)')",
        "        clone_url = REPO_URL",
        "        try:",
        "            from kaggle_secrets import UserSecretsClient",
        "            _tok = UserSecretsClient().get_secret('GITHUB_TOKEN')",
        "            if _tok and '@' not in REPO_URL:",
        "                clone_url = REPO_URL.replace('https://', f'https://{_tok}@')",
        "        except Exception:",
        "            pass  # javni repozitorijum, ili secret nije postavljen",
        "        subprocess.run(['git', 'clone', '--depth', '1', clone_url, str(REPO_DIR)], check=True)",
        "",
        "sys.path.insert(0, str(REPO_DIR))",
        "os.chdir(REPO_DIR)",
        "",
        "import torch",
        "print('repo:', REPO_DIR, '| CUDA:', torch.cuda.is_available(), '|',",
        "      torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')",
    ),
    md(
        "## 1. Pronalaženje PlantVillage `color` slika i provera split-a",
        "",
        "Tačna putanja montiranja varira (Kaggle je može ugnezditi pod",
        "`/kaggle/input/datasets/<user>/<slug>/...`), pa **tražimo** `color` folder",
        "umesto da ga hard-kodiramo. Train skripte čitaju `data.data_root` kao putanju relativnu u odnosu na",
        "koren repozitorijuma (`data/raw/plantvillage dataset/color`); naredna ćelija tu pravi symlink ka",
        "pronađenom montiranom direktorijumu, tako da nisu potrebne izmene config-a.",
    ),
    code(
        "def find_color_dir():",
        "    \"\"\"Pronalazi PlantVillage 'color' direktorijum pod /kaggle/input (depth<=5).\"\"\"",
        "    stack = [(Path('/kaggle/input'), 0)]",
        "    while stack:",
        "        d, depth = stack.pop()",
        "        if d.name == 'color' and (d / 'Tomato___healthy').is_dir():",
        "            return d",
        "        if depth < 5:",
        "            for c in d.iterdir():",
        "                if c.is_dir():",
        "                    stack.append((c, depth + 1))",
        "    return None",
        "",
        "DATA_ROOT = find_color_dir()",
        "assert DATA_ROOT is not None, 'PlantVillage color dir not found in /kaggle/input'",
        "print('DATA_ROOT:', DATA_ROOT)",
        "",
        "# Sačuvani split je verzionisan u repozitorijumu; proveri da je prisutan.",
        "from src.data.splits import load_split, LabelMaps",
        "maps = LabelMaps.from_json(REPO_DIR / 'data/splits/label_maps.json')",
        "print('species:', maps.num_species, '| classes:', maps.num_classes)",
        "print('train rows:', len(load_split(REPO_DIR / 'data/splits', 'train')))",
    ),
    md(
        "## 2. Pravljenje symlink-a ka dataset-u na putanji koju config-ovi očekuju",
    ),
    code(
        "target = REPO_DIR / 'data/raw/plantvillage dataset/color'",
        "target.parent.mkdir(parents=True, exist_ok=True)",
        "if not target.exists():",
        "    os.symlink(DATA_ROOT, target)",
        "print('linked:', target, '->', os.readlink(target) if target.is_symlink() else 'N/A')",
    ),
    md(
        "## 2b. Offline ImageNet težine za transfer learning modele",
        "",
        "Baseline treniran from scratch ne zahteva težine, ali ResNet-50 / EfficientNet-B0 obično",
        "preuzimaju ImageNet težine sa HuggingFace-a - što ne uspeva bez Interneta. Otpremi",
        "dva `.safetensors` fajla kao Kaggle Dataset i dodaj ga kao Input; ova ćelija pronalazi",
        "folder i usmerava `transfer.py` ka njemu preko `PDH_PRETRAINED_DIR` (da ih timm učita sa",
        "diska). `HF_HUB_OFFLINE=1` preskače kašnjenja zbog ponovnih mrežnih pokušaja. Preskoči ovu ćeliju ako pokrećeš samo",
        "baseline.",
    ),
    code(
        "def find_weights_dir():",
        "    \"\"\"Pronalazi folder koji sadrži timm *.safetensors pod /kaggle/input.\"\"\"",
        "    stack = [(Path('/kaggle/input'), 0)]",
        "    while stack:",
        "        d, depth = stack.pop()",
        "        if any(d.glob('*.safetensors')):",
        "            return d",
        "        if depth < 5:",
        "            for c in d.iterdir():",
        "                if c.is_dir() and c.name != 'color':",
        "                    stack.append((c, depth + 1))",
        "    return None",
        "",
        "wdir = find_weights_dir()",
        "if wdir is not None:",
        "    os.environ['PDH_PRETRAINED_DIR'] = str(wdir)",
        "    os.environ['HF_HUB_OFFLINE'] = '1'",
        "    print('PDH_PRETRAINED_DIR =', wdir)",
        "    print('weights:', [p.name for p in wdir.glob('*.safetensors')])",
        "else:",
        "    print('WARNING: no *.safetensors found - transfer models will try to download (needs Internet).')",
    ),
    md(
        "## 3. Flat baseline-i",
        "",
        "Jednostavna CNN (from scratch) i ResNet-50 flat referenca. Svaki upisuje",
        "`experiments/<name>/best.pth` (najbolji po val macro F1) + `history.json` + `config.yaml`.",
    ),
    code("!python scripts/train_flat.py --config configs/baseline_cnn_flat.yaml --device cuda"),
    code("!python scripts/train_flat.py --config configs/resnet50_flat.yaml --device cuda"),
    md(
        "## 4. Hijerarhijski: glava za vrstu + glave za bolest po vrsti",
    ),
    code("!python scripts/train_species.py --config configs/resnet50_hierarchical.yaml --device cuda"),
    code(
        "# Trenira netrivijalne glave za vrstu u petlji (vrste sa jednom klasom se preskaču).",
        "!python scripts/train_disease.py --config configs/resnet50_hierarchical.yaml --device cuda",
    ),
    md(
        "## 5. End-to-end evaluacija + poređenje flat naspram hijerarhijskog",
        "",
        "Prikazuje end-to-end macro / weighted F1, balanced accuracy, **propagaciju grešaka**",
        "(koliki deo end-to-end grešaka potiče od pogrešne predikcije vrste), i",
        "razliku u macro F1 između flat i hijerarhijskog pristupa - ključni rezultat teze.",
    ),
    code(
        "!python scripts/evaluate_pipeline.py \\",
        "    --config configs/resnet50_hierarchical.yaml \\",
        "    --flat-config configs/resnet50_flat.yaml \\",
        "    --device cuda",
    ),
    md(
        "## 6. Preuzimanje rezultata",
        "",
        "`experiments/<name>/` sadrži `best.pth`, `history.json`, `config.yaml`,",
        "`test_metrics.json`. Spakuj JSON metrike (male) u zip radi preuzimanja; checkpoint-i su",
        "podrazumevano isključeni (ukloni `-x` flag da bi bili uključeni).",
    ),
    code(
        "!cd {REPO_DIR} && zip -r /kaggle/working/experiments.zip experiments -x '*.pth' \\",
        "    && echo 'zipped (checkpoints excluded; remove -x to include them)'",
    ),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path(__file__).resolve().parent / "train_kaggle.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print("wrote", out)
