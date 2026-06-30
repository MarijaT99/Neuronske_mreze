"""Sanity check training infrastrukture: metrike, loss-evi i stvarni mini training run.

Offline, CPU, sintetički podaci. Proverava:
* compute_metrics se poklapa sa ručno izračunatim vrednostima na malenom neuravnoteženom
  primeru, i tu je macro F1 != accuracy (cela poenta izbora metrike);
* build_loss dispečuje ce / weighted_ce / focal; focal(gamma=0) == CE;
* FocalLoss daje manju težinu lakim primerima u odnosu na CE;
* Trainer se izvršava end-to-end na sintetičkom dataset-u, upisuje best.pth + history.json,
  pravi checkpoint po macro F1 i radi early stopping.

Upisuje izveštaj u $TEMP; izlazi sa kodom različitim od nule pri padu.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  -- koren repozitorijuma na sys.path + fiksiran CWD
import os
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.metrics import compute_metrics, confusion
from src.training.losses import FocalLoss, build_loss
from src.training.trainer import Trainer, TrainConfig
from src.utils.seed import seed_everything


def _report_path() -> Path:
    if "--report" in sys.argv:
        return Path(sys.argv[sys.argv.index("--report") + 1])
    if os.environ.get("REPORT_PATH"):
        return Path(os.environ["REPORT_PATH"])
    return Path(tempfile.gettempdir()) / "pdh_training_sanity.txt"


REPORT = _report_path()
report: list[str] = []
failures: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    report.append(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        failures.append(name)


def main() -> int:
    seed_everything(42)

    # --- metrike: neuravnoteženi primer gde je accuracy >> macro F1 ------
    # 18 iz klase 0, 2 iz klase 1. Predvidi sve 0 -> acc 0.9 ali nizak macro F1.
    y_true = np.array([0] * 18 + [1] * 2)
    y_pred = np.array([0] * 20)
    m = compute_metrics(y_true, y_pred, num_classes=2)
    check("metrics: accuracy 0.90 on all-majority prediction", abs(m.accuracy - 0.9) < 1e-9,
          f"acc={m.accuracy:.3f}")
    # za klasu 1 F1 = 0 -> macro F1 = (F1_0 + 0)/2; F1_0 = 2*0.9*1/(1.9)=0.947 -> macro≈0.474
    check("metrics: macro F1 << accuracy (imbalance exposed)", m.macro_f1 < 0.5,
          f"macro_f1={m.macro_f1:.3f}")
    check("metrics: balanced accuracy = 0.5 (recall 1 and 0)",
          abs(m.balanced_accuracy - 0.5) < 1e-9, f"balAcc={m.balanced_accuracy:.3f}")
    # savršeno predviđanje -> sve 1.0
    mp = compute_metrics(y_true, y_true, num_classes=2)
    check("metrics: perfect -> macro F1 = 1.0", abs(mp.macro_f1 - 1.0) < 1e-9)
    cm = confusion(y_true, y_pred, num_classes=2)
    check("metrics: confusion matrix shape (2,2) and sums to N",
          cm.shape == (2, 2) and cm.sum() == 20)

    # --- losses ----------------------------------------------------------
    logits = torch.randn(16, 5)
    target = torch.randint(0, 5, (16,))
    cw = torch.rand(5) + 0.5

    ce = build_loss({"type": "ce"})
    wce = build_loss({"type": "weighted_ce"}, class_weights=cw)
    focal0 = FocalLoss(gamma=0.0)
    check("loss: build ce/weighted_ce", isinstance(ce, nn.CrossEntropyLoss)
          and isinstance(wce, nn.CrossEntropyLoss))
    # focal(gamma=0) == obični CE
    diff = (focal0(logits, target) - ce(logits, target)).abs().item()
    check("loss: focal(gamma=0) == CE", diff < 1e-5, f"|diff|={diff:.2e}")
    # focal(gamma=2) < CE (laki primeri dobijaju manju težinu -> ovde manji srednji loss)
    focal2 = FocalLoss(gamma=2.0)
    check("loss: focal(gamma=2) reduces loss vs CE on same logits",
          focal2(logits, target).item() < ce(logits, target).item(),
          f"focal={focal2(logits, target).item():.3f} ce={ce(logits, target).item():.3f}")
    fb = build_loss({"type": "focal", "gamma": 2.0, "use_weights": True}, class_weights=cw)
    check("loss: build focal with weights", isinstance(fb, FocalLoss) and fb.alpha is not None)

    # --- Trainer end-to-end na sintetičkim razdvojivim podacima ----------
    # Maleni, približno linearno razdvojiv problem da bi macro F1 zaista rastao.
    torch.manual_seed(0)
    n_per, n_cls, dim = 60, 3, 16
    xs, ys = [], []
    centers = torch.randn(n_cls, dim) * 3
    for c in range(n_cls):
        xs.append(centers[c] + torch.randn(n_per, dim))
        ys.append(torch.full((n_per,), c))
    X = torch.cat(xs); Y = torch.cat(ys)
    # preoblikovati u oblik nalik slici (B,3,H,W)? Ne — umesto toga koristimo maleni MLP model.
    perm = torch.randperm(len(X))
    X, Y = X[perm], Y[perm]
    split = int(0.7 * len(X))
    tr = TensorDataset(X[:split], Y[:split])
    va = TensorDataset(X[split:], Y[split:])
    tl = DataLoader(tr, batch_size=16, shuffle=True)
    vl = DataLoader(va, batch_size=16)

    model = nn.Sequential(nn.Linear(dim, 32), nn.ReLU(), nn.Linear(32, n_cls))
    cfg = TrainConfig(epochs=8, lr=1e-2, optimizer="adam", scheduler="cosine",
                      early_stopping_patience=5, amp=False, monitor="macro_f1")
    out_dir = Path(tempfile.gettempdir()) / "pdh_trainer_out"
    trainer = Trainer(model, tl, vl, build_loss({"type": "ce"}), cfg,
                      num_classes=n_cls, device="cpu", out_dir=out_dir)
    history = trainer.fit()

    check("trainer: produced per-epoch history", len(history) >= 1)
    check("trainer: best.pth written", (out_dir / "best.pth").exists())
    check("trainer: history.json written", (out_dir / "history.json").exists())
    check("trainer: learned something (final macro F1 > 0.6)",
          history[-1].metrics["macro_f1"] > 0.6,
          f"final macroF1={history[-1].metrics['macro_f1']:.3f}")
    check("trainer: best_score tracked and monitor is macro_f1",
          trainer.cfg.monitor == "macro_f1" and trainer.best_score > 0)
    # best.pth odgovara zabeleženom najboljem score-u
    ckpt = torch.load(out_dir / "best.pth", map_location="cpu")
    check("trainer: checkpoint stores monitor + score",
          ckpt["monitor"] == "macro_f1" and abs(ckpt["score"] - trainer.best_score) < 1e-9)

    REPORT.write_text(
        "\n".join(["=== TRAINING SANITY ==="] + report
                  + ["", f"RESULT: {'ALL PASS' if not failures else 'FAILURES: ' + ', '.join(failures)}"]),
        encoding="utf-8",
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    try:
        code = main()
    except Exception:
        REPORT.write_text("TRAINING SANITY CRASHED:\n" + traceback.format_exc(), encoding="utf-8")
        code = 2
    sys.exit(code)
