"""Inference-based figures for the paper: 38-class confusion matrix + Grad-CAM.

Runs the trained FLAT ResNet-50 on the test split (locally, CPU is fine) and:
* writes results/confusion_flat.png (row-normalized = per-class recall),
* writes results/confused_pairs.csv (top off-diagonal confusions),
* writes results/per_class_f1.csv (per-class precision/recall/F1/support),
* writes results/gradcam.png (a grid of Grad-CAM overlays — background-bias evidence).

Run:  python scripts/make_error_analysis.py
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix

from src.data.loaders import build_dataset, build_loader
from src.data.splits import LabelMaps
from src.evaluation.error_analysis import (
    gradcam_overlay,
    most_confused_pairs,
    plot_confusion_matrix,
)
from src.evaluation.metrics import compute_metrics, per_class_table
from src.models.factory import build_model

ROOT = Path(_bootstrap.ROOT)
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)


def find_data_root() -> Path:
    candidates = [
        ROOT.parent / "archive (1)/plantvillage dataset/color",
        ROOT.parent / "Neuronske mreže/dataset/plantvillage dataset/color",
        ROOT / "data/raw/plantvillage dataset/color",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise SystemExit("PlantVillage 'color' dir not found locally.")


def main() -> int:
    data_root = find_data_root()
    splits_dir = ROOT / "data/splits"
    maps = LabelMaps.from_json(splits_dir / "label_maps.json")
    names = [maps.id_to_class.get(i, maps.id_to_class.get(str(i))) for i in range(maps.num_classes)]

    device = torch.device("cpu")
    model = build_model({"type": "resnet50", "pretrained": False, "mode": "finetune",
                         "unfreeze_blocks": 2, "drop_rate": 0.2}, maps.num_classes)
    ckpt = torch.load(ROOT / "experiments/resnet50_flat/best.pth", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    print(f"loaded flat model (epoch {ckpt.get('epoch')}, score {ckpt.get('score'):.4f})")

    test_ds = build_dataset(data_root=data_root, splits_dir=splits_dir, split="test",
                            mode="flat", img_size=224)
    loader = build_loader(test_ds, batch_size=64, is_train=False, num_workers=0)

    # --- inference over the test set ---
    y_true, y_pred = [], []
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            logits = model(x)
            y_true.append(y.numpy())
            y_pred.append(logits.argmax(1).numpy())
            if (i + 1) % 20 == 0:
                print(f"  batch {i + 1}/{len(loader)}")
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    print(f"test images: {len(y_true)}")

    # --- confusion matrix + tables ---
    cm = confusion_matrix(y_true, y_pred, labels=list(range(maps.num_classes)))
    plot_confusion_matrix(cm, names, normalize=True,
                          title="Flat ResNet-50 — 38-class confusion (test, row=recall)",
                          save_path=OUT / "confusion_flat.png")
    most_confused_pairs(cm, names, top_k=15).to_csv(OUT / "confused_pairs.csv", index=False)

    res = compute_metrics(y_true, y_pred, num_classes=maps.num_classes)
    per_class_table(res, names).to_csv(OUT / "per_class_f1.csv", index=False)
    print("wrote confusion_flat.png, confused_pairs.csv, per_class_f1.csv")

    # --- Grad-CAM grid (background-bias evidence) ---
    import matplotlib.pyplot as plt

    correct = np.where(y_true == y_pred)[0]
    wrong = np.where(y_true != y_pred)[0]
    # 3 correct from distinct classes + up to 3 misclassified
    picks, seen = [], set()
    for idx in correct:
        c = int(y_true[idx])
        if c not in seen:
            picks.append((int(idx), False)); seen.add(c)
        if len(picks) == 3:
            break
    for idx in wrong[:3]:
        picks.append((int(idx), True))

    n = len(picks)
    fig, axes = plt.subplots(1, n, figsize=(3.0 * n, 3.4))
    if n == 1:
        axes = [axes]
    for ax, (idx, is_wrong) in zip(axes, picks):
        img, _ = test_ds[idx]
        overlay, _ = gradcam_overlay(model, img, target_class=int(y_pred[idx]))
        ax.imshow(overlay)
        ax.axis("off")
        t, p = names[int(y_true[idx])], names[int(y_pred[idx])]
        color = "red" if is_wrong else "black"
        title = f"true: {t}\npred: {p}" if is_wrong else f"{t}\n(correct)"
        ax.set_title(title, fontsize=7, color=color)
    fig.suptitle("Grad-CAM (flat ResNet-50) — where the model looks", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "gradcam.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote gradcam.png")
    print(f"\nAll error-analysis figures in {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
