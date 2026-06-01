"""Train the flat 38-class baseline (or any flat classifier).

Usage (from repo root, or via run.ps1 which pins the CWD)::

    python scripts/train_flat.py --config configs/resnet50_flat.yaml
    .\\run.ps1 scripts\\train_flat.py --config configs\\baseline_cnn_flat.yaml

Checkpoints best.pth (by val macro F1) + history.json + config.yaml under
experiments/<name>/.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse

import torch

from _train_common import (
    build_loss_for,
    class_weights_for,
    experiment_out_dir,
    load_config,
    make_loaders_flat,
    make_train_config,
    resolve_path,
    snapshot_config,
)
from src.models.factory import build_model
from src.training.trainer import Trainer
from src.utils.seed import seed_everything


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = load_config(resolve_path(args.config))
    seed_everything(cfg.get("seed", 42))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    num_classes = cfg.task.num_classes
    train_loader, val_loader, train_ds, _ = make_loaders_flat(cfg, mode="flat")

    weights = class_weights_for(cfg, train_ds.labels(), num_classes, device)
    criterion = build_loss_for(cfg, weights)
    model = build_model(dict(cfg.model.to_dict() if hasattr(cfg.model, "to_dict")
                             else cfg.model), num_classes)

    out_dir = experiment_out_dir(cfg)
    snapshot_config(cfg, out_dir)

    trainer = Trainer(
        model, train_loader, val_loader, criterion, make_train_config(cfg),
        num_classes=num_classes, device=device, out_dir=out_dir, label_index=1,
    )
    print(f"[train_flat] {cfg.experiment.name} | classes={num_classes} | device={device}")
    trainer.fit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
