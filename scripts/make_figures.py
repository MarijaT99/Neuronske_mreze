"""Generiše figure za rad kojima je potreban samo zabeleženi JSON (bez inference modela).

Čita experiments/**/history.json i experiments/resnet50_hierarchical/test_metrics.json
i upisuje PNG slike u results/. Confusion matrix / Grad-CAM (za koje je potrebno pokretanje
modela na test setu) generišu se zasebno.

Run:  python scripts/make_figures.py
"""
from __future__ import annotations

import _bootstrap  # noqa: F401  (koren repozitorijuma na sys.path + fiksiran CWD)
import json
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", context="paper")
ROOT = Path(_bootstrap.ROOT)
EXP = ROOT / "experiments"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)


def load_history(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fig_training_curves() -> None:
    """Val macro F1 po epoch-u za glavne modele + loss za resnet50_flat."""
    models = {
        "Baseline CNN (flat)": EXP / "baseline_cnn_flat/history.json",
        "ResNet-50 (flat)": EXP / "resnet50_flat/history.json",
        "ResNet-50 species (L1)": EXP / "resnet50_hierarchical/species/history.json",
    }
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    for label, p in models.items():
        if not p.exists():
            continue
        h = load_history(p)["history"]
        ep = [e["epoch"] for e in h]
        ax1.plot(ep, [e["macro_f1"] for e in h], marker="o", ms=3, label=label)
    ax1.set(xlabel="Epoch", ylabel="Validation macro F1",
            title="Validation macro F1 per epoch")
    ax1.legend(fontsize=8)
    ax1.set_ylim(0, 1.02)

    # train/val loss za glavnu flat referencu
    h = load_history(EXP / "resnet50_flat/history.json")["history"]
    ep = [e["epoch"] for e in h]
    ax2.plot(ep, [e["train_loss"] for e in h], marker="o", ms=3, label="train loss")
    ax2.plot(ep, [e["val_loss"] for e in h], marker="o", ms=3, label="val loss")
    ax2.set(xlabel="Epoch", ylabel="Loss", title="ResNet-50 (flat) loss")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT / "training_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote results/training_curves.png")


def fig_flat_vs_hier() -> None:
    """Grupisani stubići: flat naspram hijerarhijskog na test setu (4 metrike)."""
    tm = json.loads((EXP / "resnet50_hierarchical/test_metrics.json").read_text(encoding="utf-8"))
    flat = tm["flat"]
    hier = tm["hierarchical"]
    metrics = [
        ("Macro F1", "macro_f1", "end_to_end_macro_f1"),
        ("Weighted F1", "weighted_f1", "end_to_end_weighted_f1"),
        ("Balanced acc", "balanced_accuracy", "end_to_end_balanced_acc"),
        ("Accuracy", "accuracy", "end_to_end_accuracy"),
    ]
    labels = [m[0] for m in metrics]
    flat_v = [flat[m[1]] for m in metrics]
    hier_v = [hier[m[2]] for m in metrics]

    x = range(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7, 4.2))
    b1 = ax.bar([i - w / 2 for i in x], flat_v, w, label="Flat (38-class)")
    b2 = ax.bar([i + w / 2 for i in x], hier_v, w, label="Hierarchical")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylim(0.9, 1.0)
    ax.set_ylabel("Score (test set)")
    ax.set_title("Flat vs Hierarchical — test set (macro F1 is primary)")
    ax.legend()
    for bars in (b1, b2):
        for b in bars:
            ax.annotate(f"{b.get_height():.3f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                        ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "flat_vs_hierarchical.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote results/flat_vs_hierarchical.png")


def fig_disease_heads() -> None:
    """Najbolji val macro F1 disease head-a po species (sortirano)."""
    ddir = EXP / "resnet50_hierarchical/disease"
    rows = []
    for sub in sorted(ddir.iterdir()):
        hp = sub / "history.json"
        if hp.exists():
            rows.append((sub.name.replace("_(including_sour)", "").replace("_(maize)", ""),
                         load_history(hp)["best_score"]))
    rows.sort(key=lambda r: r[1])
    names = [r[0] for r in rows]
    vals = [r[1] for r in rows]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    colors = ["#d62728" if v < 0.95 else "#1f77b4" for v in vals]
    bars = ax.barh(names, vals, color=colors)
    ax.set_xlim(0.85, 1.005)
    ax.set_xlabel("Best validation macro F1")
    ax.set_title("Per-species disease classifier (L2)")
    for b, v in zip(bars, vals):
        ax.annotate(f"{v:.3f}", (b.get_width(), b.get_y() + b.get_height() / 2),
                    ha="left", va="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "disease_heads_macro_f1.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote results/disease_heads_macro_f1.png")


def fig_error_propagation() -> None:
    """Odakle potiču end-to-end greške hijerarhije?"""
    tm = json.loads((EXP / "resnet50_hierarchical/test_metrics.json").read_text(encoding="utf-8"))
    h = tm["hierarchical"]
    e_species = h["errors_from_species"]
    e_disease = h["errors_from_disease_only"]

    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    wedges, _, _ = ax.pie(
        [e_species, e_disease],
        labels=[f"Wrong species (L1)\n{e_species} errors",
                f"Wrong disease only (L2)\n{e_disease} errors"],
        autopct=lambda p: f"{p:.1f}%",
        colors=["#d62728", "#1f77b4"], startangle=90,
        textprops={"fontsize": 9},
    )
    ax.set_title("Hierarchy end-to-end error sources\n"
                 f"(total {e_species + e_disease} errors)")
    fig.tight_layout()
    fig.savefig(OUT / "error_propagation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote results/error_propagation.png")


def main() -> int:
    fig_training_curves()
    fig_flat_vs_hier()
    fig_disease_heads()
    fig_error_propagation()
    print(f"\nAll figures in {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
