"""Analiza grešaka: confusion matrice, izdvajanje pogrešnih klasifikacija i Grad-CAM.

Dve svrhe:

1. **Confusion matrice** (po nivou + finalna 38-class) da bi se videlo *koje* se
   klase mešaju — daleko informativnije od jednog broja za accuracy na
   neuravnoteženim podacima.
2. **Grad-CAM** na pogrešno klasifikovanim primerima da bi se izložio poznati
   PlantVillage **background bias**: ako heatmap osvetli uniformnu pozadinu umesto
   lezije, model je naučio prečicu. Ovo je ključni kvalitativni dokaz rada i
   navedeno ograničenje.

Teške zavisnosti (matplotlib, seaborn, pytorch_grad_cam) uvoze se lenjo unutar
funkcija kako bi import ovog modula bio jeftin i radio na mašinama koje imaju samo
data/eval stack (ovo se izvršava na Kaggle-u).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------- #
# Confusion matrice
# --------------------------------------------------------------------------- #
def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    *,
    normalize: bool = True,
    title: str = "Confusion matrix",
    figsize: tuple[int, int] = (12, 10),
    save_path: str | Path | None = None,
    annotate: bool | None = None,
):
    """Iscrtaj heatmap confusion matrice. Vraća matplotlib Figure.

    ``normalize`` deli svaki red njegovim support-om (normalizacija po tačnoj klasi)
    tako da se dijagonala čita kao recall po klasi — pravi prikaz kod neuravnoteženosti.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm = np.asarray(cm, dtype=float)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)

    if annotate is None:
        annotate = len(class_names) <= 20  # izbegni gužvu za 38-class matricu

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm, ax=ax, cmap="viridis", square=True,
        xticklabels=class_names, yticklabels=class_names,
        annot=annotate, fmt=".2f" if annotate else "",
        vmin=0, vmax=1 if normalize else None,
        cbar_kws={"label": "recall" if normalize else "count"},
    )
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=90, fontsize=7)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=7)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def most_confused_pairs(
    cm: np.ndarray,
    class_names: list[str],
    *,
    top_k: int = 15,
):
    """Vrati najčešće vandijagonalne (true, pred, count) zabune kao DataFrame."""
    import pandas as pd

    cm = np.asarray(cm)
    rows = []
    n = cm.shape[0]
    for i in range(n):
        for j in range(n):
            if i != j and cm[i, j] > 0:
                rows.append((class_names[i], class_names[j], int(cm[i, j])))
    df = pd.DataFrame(rows, columns=["true", "predicted", "count"])
    return df.sort_values("count", ascending=False).head(top_k).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Izdvajanje pogrešnih klasifikacija
# --------------------------------------------------------------------------- #
def find_misclassified(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    max_per_pair: int | None = None,
) -> np.ndarray:
    """Indeksi gde je predikcija != tačna vrednost (opciono ograničeno po (true,pred) paru)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    wrong = np.where(y_true != y_pred)[0]
    if max_per_pair is None:
        return wrong
    seen: dict[tuple[int, int], int] = {}
    keep = []
    for idx in wrong:
        key = (int(y_true[idx]), int(y_pred[idx]))
        if seen.get(key, 0) < max_per_pair:
            keep.append(idx)
            seen[key] = seen.get(key, 0) + 1
    return np.array(keep, dtype=int)


# --------------------------------------------------------------------------- #
# Grad-CAM
# --------------------------------------------------------------------------- #
def _default_target_layer(model):
    """Poslednji konvolucioni sloj po najboljoj proceni za podržane arhitekture."""
    # TransferModel obavija timm backbone.
    backbone = getattr(model, "backbone", model)
    # ResNet: layer4. EfficientNet (timm): conv_head / blocks[-1].
    if hasattr(backbone, "layer4"):
        return backbone.layer4[-1]
    if hasattr(backbone, "conv_head"):
        return backbone.conv_head
    if hasattr(backbone, "blocks"):
        return backbone.blocks[-1]
    # BaselineCNN: poslednji konvolucioni blok.
    if hasattr(model, "features"):
        return model.features[-1]
    raise ValueError("Could not infer a Grad-CAM target layer; pass one explicitly.")


def gradcam_overlay(
    model,
    image_tensor,
    *,
    target_class: int | None = None,
    target_layer=None,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
):
    """Izračunaj Grad-CAM overlay za jedan normalizovan (C,H,W) image tensor.

    Vraća ``(overlay_rgb, cam)`` gde je overlay_rgb HWC float slika u [0,1] sa
    heatmap-om stopljenim preko de-normalizovanog ulaza.
    """
    import torch
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    if target_layer is None:
        target_layer = _default_target_layer(model)

    device = next(model.parameters()).device
    input_tensor = image_tensor.unsqueeze(0).to(device)

    targets = None
    if target_class is not None:
        targets = [ClassifierOutputTarget(int(target_class))]

    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

    # De-normalizuj ulaz radi prikaza.
    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t = torch.tensor(std).view(3, 1, 1)
    rgb = (image_tensor.cpu() * std_t + mean_t).clamp(0, 1).permute(1, 2, 0).numpy()

    overlay = show_cam_on_image(rgb.astype(np.float32), grayscale_cam, use_rgb=True)
    return overlay / 255.0, grayscale_cam
