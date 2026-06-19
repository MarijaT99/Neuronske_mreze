# Hierarchical Plant Disease Classification

Master's thesis project — *Neuronske mreže*, FTN Novi Sad (ML & AI).
**Author:** Marija Tadić, E9 12/2024.

Hierarchical classification of plant leaf diseases from photographs on the
[PlantVillage](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
dataset (`color` version, 54,306 RGB images, 14 species, 38 (species, disease) classes).

- **Level 1 (species):** 14 plant-species classes.
- **Level 2 (disease):** disease type *within* a recognized species (one disease classifier per species).
- A **flat 38-class baseline** is trained for comparison — the hierarchy may or may not add value, which is itself a valid result.

## Evaluation philosophy

> "Nema smisla računati Accuracy ako skup podataka nije balansiran."

The dataset is **imbalanced at both hierarchy levels**, so accuracy is *never* a primary
metric. Primary metrics:

- **Macro F1** (treats all classes equally) — primary.
- **Weighted F1** (weighted by support) — realistic view.
- Per-class precision/recall + confusion matrices.
- **Balanced accuracy** as the accuracy-like alternative.
- **End-to-end hierarchical metrics** with explicit **error-propagation** analysis.

Accuracy may appear only as a secondary metric, always with a note on why it is not representative.

## Project layout

```
configs/        YAML config per experiment (config-driven, no hardcoded values)
data/splits/    Persisted stratified train/val/test indices (versioned)
data/raw/       Raw images (gitignored)
notebooks/      EDA, sanity checks, error analysis
src/            data / models / training / evaluation / utils packages
scripts/        prepare_splits, train_*, evaluate_pipeline, tune_hyperparams
experiments/    Checkpoints & logs (gitignored except metadata)
paper/          IEEE LaTeX
kaggle/         Kaggle notebook templates (training runs on Kaggle T4/P100)
```

## Quickstart

```bash
# 1. Environment
python -m venv .venv
# Windows:  .venv\Scripts\activate    |    Unix:  source .venv/bin/activate
pip install -r requirements.txt

# 2. Get the data (color version only — NOT segmented/grayscale)
#    Put your Kaggle API token at ~/.kaggle/kaggle.json, then:
kaggle datasets download -d abdallahalidev/plantvillage-dataset -p data/raw --unzip
#    The color images live under:
#    data/raw/plantvillage dataset/color/<Species___Disease>/*.jpg

# 3. Create the persisted stratified split (70/15/15, fixed seed)
python scripts/prepare_splits.py \
    --data-root "data/raw/plantvillage dataset/color" \
    --out-dir data/splits \
    --seed 42

# 4. Explore
jupyter lab notebooks/01_eda.ipynb
```

## Local development on Windows / OneDrive

This repo lives under a **OneDrive-synced folder** whose absolute path contains a
**non-ASCII character** (`Republički`). Two environment quirks follow from that and
have bitten development before:

1. **`cv2.imread` returns `None`** on the non-ASCII path. Worked around in
   [`src/data/dataset.py`](src/data/dataset.py) by reading bytes with
   `np.fromfile` + `cv2.imdecode`. Use that path when loading images.
2. **Runtime read-backs of in-repo files can come back garbled** while OneDrive
   syncs. The sanity scripts therefore write their reports to `$TEMP` by default
   (override with `--report PATH` or `REPORT_PATH`); keep generated reports out of
   the repo.
3. **PowerShell's working directory intermittently resets to the repo's parent**
   between commands, which breaks relative-path script invocations and relative
   `--out-dir`/`--report` arguments.

Recommended invocation — use the wrapper, which pins the CWD to the repo root and
disables bytecode files no matter where you launched it:

```powershell
.\run.ps1 scripts\sanity_check.py
.\run.ps1 scripts\prepare_splits.py --data-root "data\raw\plantvillage dataset\color" --out-dir data\splits --seed 42
```

If you run Python directly instead, do both of these first:

```powershell
Set-Location "C:\...\Master\plant-disease-hierarchical"   # always start at the repo root
$env:PYTHONDONTWRITEBYTECODE = 1                           # no __pycache__ in the synced tree
```

New scripts under `scripts/` can stay CWD-independent by importing the bootstrap
helper as their first line (it puts the repo root on `sys.path`, pins the CWD,
and sets the env var above):

```python
import _bootstrap  # noqa: F401
```

`pytest` is covered automatically by the repo-root [`conftest.py`](conftest.py),
which applies the same path/CWD guard during collection.

## Reproducibility

- Global seeds for `torch`, `numpy`, `random`; `torch.backends.cudnn.deterministic = True`.
- Every experiment has its own YAML config; all hyperparameters are logged.
- The train/val/test split is **persisted as CSV** (with class labels + indices) and never regenerated.
- Best model is checkpointed by **macro F1**, not by loss.

## Workflow note

Code is developed locally; **training runs on Kaggle Notebooks** (free T4/P100; the
dataset already exists on the platform). Do not write training code before the data
pipeline passes its sanity check.

## Known limitation — background bias

PlantVillage images have near-uniform per-class backgrounds; models can learn the
background instead of the disease. Addressed via aggressive augmentation (color jitter,
random erasing, perspective), Grad-CAM in error analysis, and explicit discussion in the paper.

## Paper

The final paper is `Report.pdf` (repo root). Its LaTeX source is in `paper/main.tex`
(IEEE conference template, in Serbian) with the figures under `paper/figures/`.
To rebuild it, compile `paper/main.tex` with pdfLaTeX (two passes).
