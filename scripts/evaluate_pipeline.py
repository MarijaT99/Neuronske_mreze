"""End-to-end evaluation of a trained hierarchy on the test split.

Loads the trained species head + per-species disease heads from an experiment
directory, assembles a HierarchicalClassifier, runs it on the test set, and
reports the imbalance-aware end-to-end metrics plus error-propagation accounting.
Optionally also evaluates a flat model checkpoint for the flat-vs-hierarchical
comparison that is central to the thesis.

Usage::

    python scripts/evaluate_pipeline.py --config configs/resnet50_hierarchical.yaml
    python scripts/evaluate_pipeline.py --config ... --flat-config configs/resnet50_flat.yaml

Writes metrics.json (+ prints a summary). Designed to run on Kaggle after training.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from _train_common import (
    experiment_out_dir,
    load_config,
    load_maps,
    make_loaders_flat,
    resolve_path,
)
from src.data.loaders import build_dataset, build_loader
from src.evaluation.hierarchical_eval import evaluate_hierarchical
from src.evaluation.metrics import compute_metrics
from src.models.factory import build_model
from src.models.hierarchical import HierarchicalClassifier
from src.utils.seed import seed_everything


def _load_state(model: torch.nn.Module, ckpt_path: Path, device) -> torch.nn.Module:
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model


@torch.no_grad()
def _collect_hierarchical(hc: HierarchicalClassifier, loader, device):
    true_s, true_d, pred_s, pred_d = [], [], [], []
    for images, species_id, disease_id in loader:
        images = images.to(device)
        pred = hc.predict(images)
        true_s.append(species_id.numpy())
        true_d.append(disease_id.numpy())
        pred_s.append(pred.species_id.cpu().numpy())
        pred_d.append(pred.disease_id_in_species.cpu().numpy())
    return (np.concatenate(true_s), np.concatenate(true_d),
            np.concatenate(pred_s), np.concatenate(pred_d))


@torch.no_grad()
def _collect_flat(model, loader, device):
    true, pred = [], []
    for images, label in loader:
        images = images.to(device)
        logits = model(images)
        true.append(label.numpy())
        pred.append(logits.argmax(1).cpu().numpy())
    return np.concatenate(true), np.concatenate(pred)


def build_hierarchy(cfg, maps, exp_dir: Path, device) -> HierarchicalClassifier:
    """Assemble the trained hierarchy from checkpoints under exp_dir."""
    species_spec = dict(cfg.species_model.to_dict() if hasattr(cfg.species_model, "to_dict")
                        else cfg.species_model)
    species_model = build_model(species_spec, maps.num_species)
    _load_state(species_model, exp_dir / "species" / "best.pth", device)

    disease_spec = dict(cfg.disease_model.to_dict() if hasattr(cfg.disease_model, "to_dict")
                        else cfg.disease_model)
    disease_models: dict[int, torch.nn.Module] = {}
    for name, sid in maps.species_to_id.items():
        k = maps.num_diseases_for(sid)
        if k <= 1:
            continue  # trivial species -> no head
        ckpt = exp_dir / "disease" / name / "best.pth"
        if not ckpt.exists():
            raise FileNotFoundError(f"Missing disease checkpoint for {name}: {ckpt}")
        m = build_model(disease_spec, k)
        _load_state(m, ckpt, device)
        disease_models[sid] = m

    return HierarchicalClassifier(species_model, disease_models,
                                  num_species=maps.num_species).to(device)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Hierarchical experiment config.")
    ap.add_argument("--flat-config", default=None, help="Optional flat config to compare.")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = load_config(resolve_path(args.config))
    seed_everything(cfg.get("seed", 42))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    maps = load_maps(cfg)

    data = cfg.data
    data_root = resolve_path(data.data_root)
    splits_dir = resolve_path(data.splits_dir)

    out: dict = {}

    # --- hierarchical end-to-end on test --------------------------------
    exp_dir = experiment_out_dir(cfg)
    hc = build_hierarchy(cfg, maps, exp_dir, device)
    test_ds = build_dataset(data_root=data_root, splits_dir=splits_dir, split="test",
                            mode="hierarchical", img_size=data.img_size)
    test_loader = build_loader(test_ds, batch_size=data.batch_size, is_train=False,
                               num_workers=data.get("num_workers", 4))
    ts, td, ps, pd_ = _collect_hierarchical(hc, test_loader, device)
    hres = evaluate_hierarchical(true_species=ts, true_disease_in_species=td,
                                 pred_species=ps, pred_disease_in_species=pd_, maps=maps)
    out["hierarchical"] = hres.summary()

    print("\n=== HIERARCHICAL (test) ===")
    for k, v in hres.summary().items():
        print(f"  {k:42s} {v:.4f}" if isinstance(v, float) else f"  {k:42s} {v}")

    # --- optional flat comparison ---------------------------------------
    if args.flat_config:
        fcfg = load_config(resolve_path(args.flat_config))
        fdir = experiment_out_dir(fcfg)
        fmodel = build_model(dict(fcfg.model.to_dict() if hasattr(fcfg.model, "to_dict")
                                  else fcfg.model), fcfg.task.num_classes)
        _load_state(fmodel, fdir / "best.pth", device)
        ftest = build_dataset(data_root=data_root, splits_dir=splits_dir, split="test",
                              mode="flat", img_size=fcfg.data.img_size)
        floader = build_loader(ftest, batch_size=fcfg.data.batch_size, is_train=False,
                               num_workers=fcfg.data.get("num_workers", 4))
        fy, fp = _collect_flat(fmodel, floader, device)
        fres = compute_metrics(fy, fp, num_classes=maps.num_classes)
        out["flat"] = fres.scalars()
        print("\n=== FLAT (test) ===")
        for k, v in fres.scalars().items():
            print(f"  {k:42s} {v:.4f}")
        print("\n=== COMPARISON (macro F1 is primary) ===")
        print(f"  flat         macro F1 = {fres.macro_f1:.4f}")
        print(f"  hierarchical macro F1 = {hres.end_to_end.macro_f1:.4f}")
        delta = hres.end_to_end.macro_f1 - fres.macro_f1
        print(f"  delta (hier - flat)   = {delta:+.4f}  "
              f"({'hierarchy helps' if delta > 0 else 'flat as good or better'})")

    metrics_path = exp_dir / "test_metrics.json"
    metrics_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
