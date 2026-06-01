"""Loss functions for imbalanced classification.

Three comparable strategies (the project compares them):

1. **Weighted cross-entropy** — ``nn.CrossEntropyLoss(weight=...)`` with weights
   from :func:`src.training.samplers.compute_class_weights`. Provided here via a
   thin builder for config-driven construction.
2. **Focal loss** (Lin et al. 2017) — down-weights easy examples, focusing
   training on hard/rare ones. Supports an optional per-class ``alpha`` weight.

``build_loss`` dispatches from a config spec so experiments stay config-driven.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Multi-class focal loss.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Parameters
    ----------
    gamma:
        Focusing parameter (0 reduces to (weighted) cross-entropy).
    alpha:
        Optional per-class weight tensor of shape (num_classes,). Acts like the
        class weights in weighted CE.
    reduction:
        'mean' | 'sum' | 'none'.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: torch.Tensor | None = None,
        reduction: str = "mean",
    ):
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be 'mean'|'sum'|'none'")
        self.gamma = gamma
        self.reduction = reduction
        # Register alpha as a buffer so .to(device)/state_dict handle it.
        if alpha is not None:
            self.register_buffer("alpha", torch.as_tensor(alpha, dtype=torch.float32))
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Per-sample CE without reduction; weight applies alpha_t automatically.
        ce = F.cross_entropy(logits, target, weight=self.alpha, reduction="none")
        # p_t = exp(-ce_unweighted); recover via softmax prob of the true class.
        log_pt = F.log_softmax(logits, dim=1).gather(1, target.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()
        focal_factor = (1.0 - pt).pow(self.gamma)
        loss = focal_factor * ce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def build_loss(spec: dict, class_weights: torch.Tensor | None = None) -> nn.Module:
    """Build a loss module from a config spec.

    Specs::

        {"type": "ce"}                              # plain cross-entropy
        {"type": "weighted_ce"}                     # CE with class_weights
        {"type": "focal", "gamma": 2.0}             # focal, no alpha
        {"type": "focal", "gamma": 2.0, "use_weights": true}  # focal w/ alpha

    ``class_weights`` (from train counts) is required for 'weighted_ce' and for
    focal with ``use_weights: true``.
    """
    spec = dict(spec)
    loss_type = spec.pop("type", "ce")

    if loss_type == "ce":
        return nn.CrossEntropyLoss()

    if loss_type == "weighted_ce":
        if class_weights is None:
            raise ValueError("weighted_ce requires class_weights.")
        return nn.CrossEntropyLoss(weight=class_weights)

    if loss_type == "focal":
        use_weights = spec.pop("use_weights", False)
        gamma = spec.pop("gamma", 2.0)
        alpha = class_weights if use_weights else None
        if use_weights and class_weights is None:
            raise ValueError("focal with use_weights=True requires class_weights.")
        return FocalLoss(gamma=gamma, alpha=alpha)

    raise ValueError(f"Unknown loss type {loss_type!r} (ce|weighted_ce|focal).")
