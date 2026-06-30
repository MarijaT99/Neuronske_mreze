"""Trenira disease (L2) klasifikatore po species — jedan po netrivijalnom species.

Prolazi kroz species koji imaju više od jedne disease klase (9 od 14; preostalih
5 su species sa samo jednom klasom "healthy" kojima ne treba head). Svaki head se
trenira samo na podskupu tog species, predviđajući disease_id_in_species
(label_index=2 u hijerarhijskom tuple-u).

Upotreba::

    python scripts/train_disease.py --config configs/resnet50_hierarchical.yaml
    # treniraj samo jedan species:
    python scripts/train_disease.py --config ... --only Tomato
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
    ap.add_argument("--only", default=None, help="Train only this species (name).")
    args = ap.parse_args()

    cfg = load_config(resolve_path(args.config))
    seed_everything(cfg.get("seed", 42))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    maps = load_maps(cfg)
    # Netrivijalni species = više od jedne disease klase.
    non_trivial = [
        (name, sid) for name, sid in maps.species_to_id.items()
        if maps.num_diseases_for(sid) > 1
    ]
    if args.only is not None:
        non_trivial = [(n, s) for n, s in non_trivial if n == args.only]
        if not non_trivial:
            raise SystemExit(f"Species {args.only!r} is not a non-trivial species.")

    print(f"[train_disease] {cfg.experiment.name} | "
          f"{len(non_trivial)} species to train | device={device}")

    model_spec = dict(cfg.disease_model.to_dict() if hasattr(cfg.disease_model, "to_dict")
                      else cfg.disease_model)

    for name, sid in non_trivial:
        k = maps.num_diseases_for(sid)
        # Ponovo postavi seed po head-u da bi pokretanje svakog head-a bilo reproducibilno bez obzira na redosled.
        seed_everything(cfg.get("seed", 42))

        train_loader, val_loader, train_ds, _ = make_loaders_flat(
            cfg, mode="hierarchical", species_filter=sid
        )
        disease_labels = train_ds.df["disease_id_in_species"].to_numpy()
        weights = class_weights_for(cfg, disease_labels, k, device)
        criterion = build_loss_for(cfg, weights)
        model = build_model(model_spec, k)

        out_dir = experiment_out_dir(cfg, subname=f"disease/{name}")
        snapshot_config(cfg, out_dir)

        trainer = Trainer(
            model, train_loader, val_loader, criterion, make_train_config(cfg),
            num_classes=k, device=device, out_dir=out_dir, label_index=2,
        )
        print(f"  -> {name} (species_id={sid}, diseases={k}, n_train={len(train_ds)})")
        trainer.fit()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
