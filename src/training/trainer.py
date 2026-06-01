"""Generic training loop with macro-F1 checkpointing and early stopping.

Model-agnostic: works with BaselineCNN and the timm TransferModel (both take a
batch of images and return class logits). Designed for the **flat** task and for
training an individual head in the hierarchy (species head, or a per-species
disease head) — the hierarchy is assembled at eval time from separately trained
heads.

Key project rules baked in here:

* **Checkpoint / early-stop on validation macro F1, not loss.** (Imbalanced data:
  loss and accuracy both over-credit majority classes.)
* Every epoch logs the full imbalance-aware metric suite (macro/weighted F1,
  balanced accuracy, accuracy-as-secondary) plus train/val loss.
* Reproducible: seeding is the caller's responsibility (``seed_everything``); the
  trainer is deterministic given seeded loaders.

The trainer is intentionally framework-light (no Lightning) so it runs cleanly in
a Kaggle notebook.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.evaluation.metrics import compute_metrics


@dataclass
class TrainConfig:
    """Training hyperparameters (usually populated from a YAML config)."""

    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adamw"           # adamw | adam | sgd
    scheduler: str = "cosine"          # cosine | plateau | none
    early_stopping_patience: int = 7   # epochs without macro-F1 improvement
    grad_clip: float | None = None
    amp: bool = True                   # mixed precision (CUDA only)
    monitor: str = "macro_f1"          # checkpoint/early-stop metric (higher=better)


@dataclass
class EpochLog:
    epoch: int
    train_loss: float
    val_loss: float
    metrics: dict[str, float] = field(default_factory=dict)


def _build_optimizer(params, cfg: TrainConfig):
    params = list(params)
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "adam":
        return torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(params, lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay,
                               nesterov=True)
    raise ValueError(f"Unknown optimizer {cfg.optimizer!r}")


def _build_scheduler(optimizer, cfg: TrainConfig):
    if cfg.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    if cfg.scheduler == "plateau":
        # maximize macro F1
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3)
    if cfg.scheduler == "none":
        return None
    raise ValueError(f"Unknown scheduler {cfg.scheduler!r}")


class Trainer:
    """Train one classifier, checkpointing the best epoch by validation macro F1.

    Parameters
    ----------
    model:
        A module mapping (B,3,H,W) -> (B,num_classes) logits.
    train_loader, val_loader:
        DataLoaders yielding (image, label). For hierarchical heads, pass loaders
        whose label is the relevant target (species_id or disease_id_in_species).
    criterion:
        Loss module (from :func:`src.training.losses.build_loss`).
    cfg:
        :class:`TrainConfig`.
    num_classes:
        Number of classes (for stable per-class metrics).
    device:
        torch device; defaults to cuda if available.
    out_dir:
        Where to write ``best.pth`` and ``history.json`` (optional).
    label_index:
        Index of the label in each batch tuple. 0 means (img, label); for a
        hierarchical species head use the species position, etc. Default: the
        batch is (image, label) so the label is element 1.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        cfg: TrainConfig,
        *,
        num_classes: int,
        device: torch.device | str | None = None,
        out_dir: str | Path | None = None,
        label_index: int = 1,
    ):
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion.to(self.device)
        self.cfg = cfg
        self.num_classes = num_classes
        self.out_dir = Path(out_dir) if out_dir is not None else None
        if self.out_dir is not None:
            self.out_dir.mkdir(parents=True, exist_ok=True)
        self.label_index = label_index

        self.optimizer = _build_optimizer(
            (p for p in self.model.parameters() if p.requires_grad), cfg
        )
        self.scheduler = _build_scheduler(self.optimizer, cfg)
        self.use_amp = cfg.amp and self.device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        self.history: list[EpochLog] = []
        self.best_score: float = -float("inf")
        self.best_epoch: int = -1

    # ------------------------------------------------------------------ #
    def _unpack(self, batch):
        """Return (images, labels) from a batch tuple using label_index."""
        images = batch[0].to(self.device, non_blocking=True)
        labels = batch[self.label_index].to(self.device, non_blocking=True)
        return images, labels

    def _train_one_epoch(self) -> float:
        self.model.train()
        running, n = 0.0, 0
        for batch in self.train_loader:
            images, labels = self._unpack(batch)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                logits = self.model(images)
                loss = self.criterion(logits, labels)
            self.scaler.scale(loss).backward()
            if self.cfg.grad_clip is not None:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            bs = labels.size(0)
            running += loss.item() * bs
            n += bs
        return running / max(n, 1)

    @torch.no_grad()
    def _evaluate(self, loader: DataLoader) -> tuple[float, np.ndarray, np.ndarray]:
        self.model.eval()
        running, n = 0.0, 0
        all_true, all_pred = [], []
        for batch in loader:
            images, labels = self._unpack(batch)
            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                logits = self.model(images)
                loss = self.criterion(logits, labels)
            bs = labels.size(0)
            running += loss.item() * bs
            n += bs
            all_true.append(labels.cpu().numpy())
            all_pred.append(logits.argmax(dim=1).cpu().numpy())
        y_true = np.concatenate(all_true) if all_true else np.array([])
        y_pred = np.concatenate(all_pred) if all_pred else np.array([])
        return running / max(n, 1), y_true, y_pred

    # ------------------------------------------------------------------ #
    def fit(self) -> list[EpochLog]:
        epochs_no_improve = 0
        for epoch in range(1, self.cfg.epochs + 1):
            t0 = time.time()
            train_loss = self._train_one_epoch()
            val_loss, y_true, y_pred = self._evaluate(self.val_loader)
            result = compute_metrics(y_true, y_pred, num_classes=self.num_classes)
            score = result.scalars()[self.cfg.monitor]

            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(score)
                else:
                    self.scheduler.step()

            log = EpochLog(epoch=epoch, train_loss=train_loss, val_loss=val_loss,
                           metrics=result.scalars())
            self.history.append(log)
            self._log_epoch(log, time.time() - t0)

            # Checkpoint best by macro F1 (or configured monitor).
            if score > self.best_score:
                self.best_score = score
                self.best_epoch = epoch
                epochs_no_improve = 0
                self._save_checkpoint(epoch, score)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.cfg.early_stopping_patience:
                    print(f"Early stopping at epoch {epoch} "
                          f"(no {self.cfg.monitor} improvement for "
                          f"{self.cfg.early_stopping_patience} epochs).")
                    break

        self._save_history()
        print(f"Best {self.cfg.monitor}={self.best_score:.4f} at epoch {self.best_epoch}.")
        return self.history

    # ------------------------------------------------------------------ #
    def _log_epoch(self, log: EpochLog, secs: float) -> None:
        m = log.metrics
        print(
            f"epoch {log.epoch:3d} | {secs:5.1f}s | "
            f"train_loss {log.train_loss:.4f} | val_loss {log.val_loss:.4f} | "
            f"macroF1 {m['macro_f1']:.4f} | wF1 {m['weighted_f1']:.4f} | "
            f"balAcc {m['balanced_accuracy']:.4f} | acc {m['accuracy']:.4f}*"
        )

    def _save_checkpoint(self, epoch: int, score: float) -> None:
        if self.out_dir is None:
            return
        torch.save(
            {
                "epoch": epoch,
                "model_state": self.model.state_dict(),
                "score": score,
                "monitor": self.cfg.monitor,
            },
            self.out_dir / "best.pth",
        )

    def _save_history(self) -> None:
        if self.out_dir is None:
            return
        payload = {
            "best_epoch": self.best_epoch,
            "best_score": self.best_score,
            "monitor": self.cfg.monitor,
            "history": [
                {"epoch": e.epoch, "train_loss": e.train_loss,
                 "val_loss": e.val_loss, **e.metrics}
                for e in self.history
            ],
        }
        (self.out_dir / "history.json").write_text(json.dumps(payload, indent=2),
                                                   encoding="utf-8")
