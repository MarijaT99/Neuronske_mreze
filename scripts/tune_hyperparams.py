"""Optuna hyperparameter tuning for a flat experiment (optimizes val macro F1).

Searches LR, weight decay, dropout, augmentation strength, and imbalance loss.
Each trial trains a short run (few epochs) and reports the best validation macro
F1 — the project's primary metric (NOT accuracy). Designed to run on Kaggle.

Usage::

    python scripts/tune_hyperparams.py --config configs/resnet50_flat.yaml \
        --trials 20 --epochs 6
"""

from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse
import json
from pathlib import Path

import torch

from _train_common import (
    build_loss_for,
    class_weights_for,
    load_config,
    make_loaders_flat,
    make_train_config,
    resolve_path,
)
from src.models.factory import build_model
from src.training.trainer import Trainer
from src.utils.seed import seed_everything


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=6, help="Short epochs per trial.")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import optuna

    base_cfg = load_config(resolve_path(args.config))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    num_classes = base_cfg.task.num_classes
    seed = base_cfg.get("seed", 42)

    def objective(trial: "optuna.Trial") -> float:
        seed_everything(seed)
        cfg = load_config(resolve_path(args.config))  # fresh copy per trial
        d = cfg.to_dict()

        # --- search space -------------------------------------------------
        d["training"]["lr"] = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
        d["training"]["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        d["training"]["epochs"] = args.epochs
        d["data"]["aug_strength"] = trial.suggest_categorical(
            "aug_strength", ["medium", "heavy"]
        )
        d["imbalance"]["loss"]["type"] = trial.suggest_categorical(
            "loss", ["weighted_ce", "focal"]
        )
        if d["imbalance"]["loss"]["type"] == "focal":
            d["imbalance"]["loss"]["use_weights"] = True
            d["imbalance"]["loss"]["gamma"] = trial.suggest_float("gamma", 1.0, 3.0)
        if "dropout" in d.get("model", {}):
            d["model"]["dropout"] = trial.suggest_float("dropout", 0.2, 0.6)

        from src.utils.config import Config
        cfg = Config(d)

        train_loader, val_loader, train_ds, _ = make_loaders_flat(cfg, mode="flat")
        weights = class_weights_for(cfg, train_ds.labels(), num_classes, device)
        criterion = build_loss_for(cfg, weights)
        model = build_model(dict(cfg.model.to_dict()), num_classes)

        trainer = Trainer(
            model, train_loader, val_loader, criterion, make_train_config(cfg),
            num_classes=num_classes, device=device, out_dir=None, label_index=1,
        )
        trainer.fit()
        return trainer.best_score  # best val macro F1

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.trials)

    print("\n=== BEST TRIAL (val macro F1) ===")
    print(f"  value = {study.best_value:.4f}")
    for k, v in study.best_params.items():
        print(f"  {k} = {v}")

    out = resolve_path(base_cfg.experiment.get("out_dir", "experiments")) / base_cfg.experiment.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "optuna_best.json").write_text(
        json.dumps({"best_value": study.best_value, "best_params": study.best_params}, indent=2),
        encoding="utf-8",
    )
    print(f"\nWrote {out / 'optuna_best.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
