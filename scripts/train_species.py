"""Trenira species (L1) klasifikator za hijerarhiju — 14 klasa.

Koristi hijerarhijski mod dataset-a i trenira na species_id (label_index=1,
pozicija species u (image, species_id, disease_id) tuple-u).

Upotreba::

    python scripts/train_species.py --config configs/resnet50_hierarchical.yaml
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
    load_maps,
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

    maps = load_maps(cfg)
    num_species = maps.num_species

    # Hijerarhijski batch-evi su (image, species_id, disease_id); species je indeks 1.
    train_loader, val_loader, train_ds, _ = make_loaders_flat(cfg, mode="hierarchical")

    species_labels = train_ds.df["species_id"].to_numpy()
    weights = class_weights_for(cfg, species_labels, num_species, device)
    criterion = build_loss_for(cfg, weights)

    model_spec = dict(cfg.species_model.to_dict() if hasattr(cfg.species_model, "to_dict")
                      else cfg.species_model)
    model = build_model(model_spec, num_species)

    out_dir = experiment_out_dir(cfg, subname="species")
    snapshot_config(cfg, out_dir)

    trainer = Trainer(
        model, train_loader, val_loader, criterion, make_train_config(cfg),
        num_classes=num_species, device=device, out_dir=out_dir, label_index=1,
    )
    print(f"[train_species] {cfg.experiment.name} | species={num_species} | device={device}")
    trainer.fit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
