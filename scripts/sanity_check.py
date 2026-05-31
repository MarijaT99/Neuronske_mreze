"""Data-pipeline sanity check (must pass before any model code is written).

Exercises the real pipeline on real images and asserts invariants:
* datasets build in both modes and load actual image tensors of the right shape;
* flat labels are in [0, 38) and hierarchical labels are valid per species;
* heavy train aug changes pixels while eval transform is deterministic;
* class weights and the weighted sampler are computed from train counts and the
  sampler actually rebalances class frequencies;
* a hierarchical per-species subset loads.

Writes a human-readable report; exits non-zero if any check fails. The report
defaults to a path OUTSIDE the OneDrive-synced repo (override with --report PATH
or REPORT_PATH env var) because OneDrive corrupts in-repo runtime read-backs.
"""

from __future__ import annotations

import os
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import PlantVillageDataset  # noqa: E402
from src.data.loaders import build_dataset, build_loader  # noqa: E402
from src.data.splits import LabelMaps, load_split  # noqa: E402
from src.data.transforms import build_eval_transforms, build_train_transforms  # noqa: E402
from src.training.samplers import compute_class_weights, make_weighted_sampler  # noqa: E402
from src.utils.seed import seed_everything  # noqa: E402

DATA_ROOT = Path(
    os.environ.get(
        "PDH_DATA_ROOT",
        r"C:\Users\Marija\OneDrive - Republički zavod za statistiku\Desktop\Master"
        r"\archive (1)\plantvillage dataset\color",
    )
)
SPLITS_DIR = ROOT / "data" / "splits"
IMG_SIZE = 224


def _report_path() -> Path:
    if "--report" in sys.argv:
        return Path(sys.argv[sys.argv.index("--report") + 1])
    if os.environ.get("REPORT_PATH"):
        return Path(os.environ["REPORT_PATH"])
    return Path(tempfile.gettempdir()) / "pdh_sanity_report.txt"


REPORT_PATH = _report_path()
report: list[str] = []
failures: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    status = "PASS" if cond else "FAIL"
    report.append(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        failures.append(name)


def main() -> int:
    seed_everything(42)
    maps = LabelMaps.from_json(SPLITS_DIR / "label_maps.json")

    # --- 1. flat dataset loads a real tensor -------------------------------
    flat_tf = build_eval_transforms(IMG_SIZE)
    flat_ds = PlantVillageDataset(
        data_root=DATA_ROOT, splits_dir=SPLITS_DIR, split="val",
        mode="flat", transform=flat_tf,
    )
    img, label = flat_ds[0]
    check("flat: image is CHW float tensor 3x224x224",
          isinstance(img, torch.Tensor) and tuple(img.shape) == (3, IMG_SIZE, IMG_SIZE),
          f"shape={tuple(img.shape)} dtype={img.dtype}")
    check("flat: label in [0,38)", 0 <= label < maps.num_classes, f"label={label}")

    sample_labels = [flat_ds[i][1] for i in range(0, len(flat_ds), max(1, len(flat_ds) // 200))]
    check("flat: all sampled labels in [0,38)",
          all(0 <= l < maps.num_classes for l in sample_labels),
          f"min={min(sample_labels)} max={max(sample_labels)}")

    # --- 2. hierarchical dataset ------------------------------------------
    hier_ds = PlantVillageDataset(
        data_root=DATA_ROOT, splits_dir=SPLITS_DIR, split="val",
        mode="hierarchical", transform=flat_tf,
    )
    img2, sp, dis = hier_ds[0]
    check("hier: returns (img, species_id, disease_id)",
          isinstance(img2, torch.Tensor) and isinstance(sp, int) and isinstance(dis, int),
          f"species={sp} disease={dis}")
    ok_dis = True
    for i in range(0, len(hier_ds), max(1, len(hier_ds) // 300)):
        _, s, d = hier_ds[i]
        if not (0 <= d < maps.num_diseases_for(s)):
            ok_dis = False
            break
    check("hier: disease_id within species range for all sampled", ok_dis)

    # --- 3. augmentation behaviour ----------------------------------------
    train_tf = build_train_transforms(IMG_SIZE, aug_strength="heavy")
    raw_ds = PlantVillageDataset(
        data_root=DATA_ROOT, splits_dir=SPLITS_DIR, split="train",
        mode="flat", transform=None,
    )
    raw_img = raw_ds[0][0]  # numpy HWC uint8
    a1 = train_tf(image=raw_img)["image"]
    a2 = train_tf(image=raw_img)["image"]
    check("train aug: two draws differ (stochastic)",
          not torch.allclose(a1, a2), f"max|diff|={(a1 - a2).abs().max():.4f}")
    e1 = flat_tf(image=raw_img)["image"]
    e2 = flat_tf(image=raw_img)["image"]
    check("eval tf: deterministic (identical draws)", torch.allclose(e1, e2))

    # --- 4. class weights + weighted sampler ------------------------------
    train_df = load_split(SPLITS_DIR, "train")
    labels = train_df["class_id"].to_numpy()
    w = compute_class_weights(labels, maps.num_classes, scheme="inverse")
    check("class weights: length 38 and mean≈1",
          len(w) == maps.num_classes and abs(float(w.mean()) - 1.0) < 1e-3,
          f"mean={w.mean():.4f} min={w.min():.3f} max={w.max():.3f}")
    counts = np.bincount(labels, minlength=maps.num_classes)
    check("class weights: rarest class has the largest weight",
          int(w.argmax()) == int(counts.argmin()),
          f"argmax_w={w.argmax()} argmin_count={counts.argmin()}")

    sampler = make_weighted_sampler(labels, maps.num_classes)
    drawn = np.array([labels[i] for i in list(sampler)[:20000]])
    drawn_counts = np.bincount(drawn, minlength=maps.num_classes)
    orig_ratio = counts.max() / max(counts.min(), 1)
    drawn_ratio = drawn_counts.max() / max(drawn_counts.min(), 1)
    check("weighted sampler: flattens class frequency (ratio drops a lot)",
          drawn_ratio < orig_ratio / 5,
          f"orig_ratio={orig_ratio:.1f}x -> sampled_ratio={drawn_ratio:.1f}x")

    # --- 5. DataLoader batch (num_workers=0 for Windows safety) ------------
    loader = build_loader(flat_ds, batch_size=8, is_train=False, num_workers=0)
    bx, by = next(iter(loader))
    check("loader: batch shapes",
          tuple(bx.shape) == (8, 3, IMG_SIZE, IMG_SIZE) and tuple(by.shape) == (8,),
          f"x={tuple(bx.shape)} y={tuple(by.shape)}")

    # --- 6. per-species subset --------------------------------------------
    tomato_id = maps.species_to_id["Tomato"]
    tomato_ds = build_dataset(
        data_root=DATA_ROOT, splits_dir=SPLITS_DIR, split="val",
        mode="hierarchical", img_size=IMG_SIZE, species_filter=tomato_id,
    )
    species_in_subset = {tomato_ds[i][1] for i in range(0, len(tomato_ds), 50)}
    check("species_filter: subset contains only the requested species",
          species_in_subset == {tomato_id},
          f"n={len(tomato_ds)} species_ids={species_in_subset}")

    header = [
        "=== DATA PIPELINE SANITY CHECK ===",
        f"data_root exists: {DATA_ROOT.exists()}",
        f"val size: {len(flat_ds)}  train size: {len(raw_ds)}",
        "",
    ]
    REPORT_PATH.write_text(
        "\n".join(
            header + report
            + ["", f"RESULT: {'ALL PASS' if not failures else 'FAILURES: ' + ', '.join(failures)}"]
        ),
        encoding="utf-8",
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    try:
        code = main()
    except Exception:
        REPORT_PATH.write_text("SANITY CHECK CRASHED:\n" + traceback.format_exc(), encoding="utf-8")
        code = 2
    sys.exit(code)
