"""Regenerate train_kaggle.ipynb with clean UTF-8 text and git-clone delivery.

Run:  python kaggle/_build_notebook.py
Writes kaggle/train_kaggle.ipynb. Kept in-repo so the notebook is reproducible
from source rather than hand-edited JSON (avoids encoding corruption).
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
    """Turn a tuple of logical lines into ipynb source (newline-terminated except last)."""
    text = "\n".join(lines)
    parts = text.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]


cells = [
    md(
        "# Train on Kaggle — Hierarchical Plant Disease Classification",
        "",
        "Template for running the experiments on a Kaggle Notebook (T4 x2 / P100). Steps:",
        "",
        "1. Add the dataset `abdallahalidev/plantvillage-dataset` (we use the `color` folder).",
        "2. Clone this repo into `/kaggle/working` (writable) so `src/` imports and the",
        "   scripts can write checkpoints.",
        "3. The persisted split is already in `data/splits/` — reuse it (do **not** regenerate).",
        "4. Train the flat baselines / species head / disease heads, then evaluate the pipeline.",
        "",
        "> **Metrics policy:** macro F1 is the **primary** metric (the dataset is imbalanced).",
        "> Accuracy is reported only as a secondary figure — never as the headline result.",
    ),
    md(
        "## 0. Get the repo into the writable working dir & set up paths",
        "",
        "Two delivery modes, tried in order (no edits needed):",
        "",
        "1. **Dataset mode (no Internet):** upload this repo as a Kaggle Dataset and add it as",
        "   an Input. The cell finds it under `/kaggle/input/...` (any folder containing `src/`",
        "   + `scripts/`) and copies it to `/kaggle/working` (so scripts can write checkpoints).",
        "2. **Clone mode (needs Internet On):** if no repo dataset is found, `git clone` from",
        "   GitHub. For a private repo add a Kaggle Secret `GITHUB_TOKEN` (Add-ons -> Secrets).",
        "",
        "Kaggle's pinned image already has torch / torchvision / timm / albumentations / sklearn,",
        "so **training works fully offline**. (`grad-cam` for Phase-6 error analysis may need",
        "Internet — not required for training.)",
    ),
    code(
        "import sys, os, shutil, subprocess",
        "from pathlib import Path",
        "",
        "REPO_URL = 'https://github.com/MarijaT99/Neuronske_mreze.git'",
        "REPO_DIR = Path('/kaggle/working/Neuronske_mreze')",
        "",
        "def find_repo_in_input():",
        "    \"\"\"Look 2 levels deep under /kaggle/input for a folder that looks like the repo.\"\"\"",
        "    base = Path('/kaggle/input')",
        "    if not base.exists():",
        "        return None",
        "    cands = []",
        "    for p in base.iterdir():",
        "        if p.is_dir():",
        "            cands.append(p)",
        "            cands += [c for c in p.iterdir() if c.is_dir()]",
        "    for cand in cands:",
        "        if (cand / 'src').is_dir() and (cand / 'scripts').is_dir():",
        "            return cand",
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
        "            pass  # public repo, or no secret set",
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
        "## 1. Point the data pipeline at the Kaggle dataset mount",
        "",
        "The PlantVillage color images mount at:",
        "`/kaggle/input/plantvillage-dataset/plantvillage dataset/color`.",
        "",
        "The train scripts read `data.data_root` from the YAML config as a path **relative to",
        "the repo root** (`data/raw/plantvillage dataset/color`). The cleanest way to make that",
        "resolve on Kaggle without editing every config is to **symlink** the mount into",
        "`data/raw/` (next cell).",
    ),
    code(
        "DATA_ROOT = '/kaggle/input/plantvillage-dataset/plantvillage dataset/color'",
        "assert Path(DATA_ROOT).exists(), f'Fix DATA_ROOT: {DATA_ROOT}'",
        "",
        "# The persisted split is versioned in the repo; sanity-check it is present.",
        "from src.data.splits import load_split, LabelMaps",
        "maps = LabelMaps.from_json(REPO_DIR / 'data/splits/label_maps.json')",
        "print('species:', maps.num_species, '| classes:', maps.num_classes)",
        "print('train rows:', len(load_split(REPO_DIR / 'data/splits', 'train')))",
    ),
    md(
        "## 2. Symlink the dataset into the path the configs expect",
    ),
    code(
        "target = REPO_DIR / 'data/raw/plantvillage dataset/color'",
        "target.parent.mkdir(parents=True, exist_ok=True)",
        "if not target.exists():",
        "    os.symlink(DATA_ROOT, target)",
        "print('linked:', target, '->', os.readlink(target) if target.is_symlink() else 'N/A')",
    ),
    md(
        "## 3. Flat baselines",
        "",
        "The simple CNN (from scratch) and the ResNet-50 flat reference. Each writes",
        "`experiments/<name>/best.pth` (best by val macro F1) + `history.json` + `config.yaml`.",
    ),
    code("!python scripts/train_flat.py --config configs/baseline_cnn_flat.yaml --device cuda"),
    code("!python scripts/train_flat.py --config configs/resnet50_flat.yaml --device cuda"),
    md(
        "## 4. Hierarchical: species head + per-species disease heads",
    ),
    code("!python scripts/train_species.py --config configs/resnet50_hierarchical.yaml --device cuda"),
    code(
        "# Trains the non-trivial species heads in a loop (single-class species are skipped).",
        "!python scripts/train_disease.py --config configs/resnet50_hierarchical.yaml --device cuda",
    ),
    md(
        "## 5. End-to-end evaluation + flat-vs-hierarchical comparison",
        "",
        "Reports end-to-end macro / weighted F1, balanced accuracy, **error propagation**",
        "(what share of end-to-end errors come from a wrong species prediction), and the",
        "flat-vs-hierarchical macro-F1 delta — the key result of the thesis.",
    ),
    code(
        "!python scripts/evaluate_pipeline.py \\",
        "    --config configs/resnet50_hierarchical.yaml \\",
        "    --flat-config configs/resnet50_flat.yaml \\",
        "    --device cuda",
    ),
    md(
        "## 6. Download results",
        "",
        "`experiments/<name>/` holds `best.pth`, `history.json`, `config.yaml`,",
        "`test_metrics.json`. Zip the JSON metrics (small) to download; checkpoints are",
        "excluded by default (remove the `-x` flag to include them).",
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
