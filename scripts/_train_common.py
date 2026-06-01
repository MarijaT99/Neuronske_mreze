"""Shared helpers for the train_* scripts (config, weights, runtime wiring).

Importing this also runs _bootstrap (repo root on sys.path, CWD pinned).
"""

from __future__ import annotations

import _bootstrap  # noqa: F401
from pathlib import Path

import numpy as np
import torch

from src.data.loaders import build_dataset, build_loader
from src.data.splits import LabelMaps, load_split
from src.training.losses import build_loss
from src.training.samplers import compute_class_weights
from src.training.trainer import TrainConfig
from src.utils.config import Config, load_config, save_config
from src.utils.seed import seed_everything

ROOT = Path(_bootstrap.ROOT)


def resolve_path(p: str) -> Path:
    """Resolve a config path relative to repo root unless already absolute."""
    path = Path(p)
    return path if path.is_absolute() else (ROOT / path)


def make_train_config(cfg: Config) -> TrainConfig:
    t = cfg.training
    return TrainConfig(
        epochs=t.epochs,
        lr=float(t.lr),
        weight_decay=float(t.get("weight_decay", 1e-4)),
        optimizer=t.get("optimizer", "adamw"),
        scheduler=t.get("scheduler", "cosine"),
        early_stopping_patience=t.get("early_stopping_patience", 7),
        grad_clip=t.get("grad_clip", None),
        amp=t.get("amp", True),
        monitor=t.get("monitor", "macro_f1"),
    )


def class_weights_for(
    cfg: Config,
    labels: np.ndarray,
    num_classes: int,
    device: torch.device,
) -> torch.Tensor | None:
    """Build class weights tensor if the loss/scheme needs them, else None."""
    loss_type = cfg.imbalance.loss.get("type", "ce")
    use_weights = loss_type == "weighted_ce" or (
        loss_type == "focal" and cfg.imbalance.loss.get("use_weights", False)
    )
    if not use_weights:
        return None
    scheme = cfg.imbalance.get("class_weight_scheme", "inverse")
    w = compute_class_weights(labels, num_classes, scheme=scheme)
    return torch.as_tensor(w, dtype=torch.float32, device=device)


def build_loss_for(cfg: Config, class_weights: torch.Tensor | None):
    return build_loss(cfg.imbalance.loss.to_dict()
                      if hasattr(cfg.imbalance.loss, "to_dict")
                      else dict(cfg.imbalance.loss), class_weights)


def make_loaders_flat(cfg: Config, *, species_filter: int | None = None,
                      mode: str = "flat"):
    """Build train/val loaders for a flat or single-head task."""
    data = cfg.data
    data_root = resolve_path(data.data_root)
    splits_dir = resolve_path(data.splits_dir)

    common = dict(
        data_root=data_root, splits_dir=splits_dir, mode=mode,
        img_size=data.img_size,
    )
    train_ds = build_dataset(split="train", aug_strength=data.get("aug_strength", "heavy"),
                             species_filter=species_filter, **common)
    val_ds = build_dataset(split="val", species_filter=species_filter, **common)

    train_loader = build_loader(
        train_ds, batch_size=data.batch_size, is_train=True,
        use_weighted_sampler=cfg.imbalance.get("use_weighted_sampler", False),
        num_workers=data.get("num_workers", 4), seed=cfg.get("seed", 42),
    )
    val_loader = build_loader(
        val_ds, batch_size=data.batch_size, is_train=False,
        num_workers=data.get("num_workers", 4), seed=cfg.get("seed", 42),
    )
    return train_loader, val_loader, train_ds, val_ds


def experiment_out_dir(cfg: Config, *, subname: str | None = None) -> Path:
    base = resolve_path(cfg.experiment.get("out_dir", "experiments"))
    name = cfg.experiment.name
    out = base / name if subname is None else base / name / subname
    out.mkdir(parents=True, exist_ok=True)
    return out


def snapshot_config(cfg: Config, out_dir: Path) -> None:
    """Persist the fully-resolved config next to the checkpoints (reproducibility)."""
    save_config(cfg, out_dir / "config.yaml")


def load_maps(cfg: Config) -> LabelMaps:
    return LabelMaps.from_json(resolve_path(cfg.data.splits_dir) / "label_maps.json")
