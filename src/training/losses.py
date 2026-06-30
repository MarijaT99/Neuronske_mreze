"""Loss funkcije za klasifikaciju na neuravnoteženim podacima.

Tri uporedive strategije (projekat ih poredi):

1. **Weighted cross-entropy** - ``nn.CrossEntropyLoss(weight=...)`` sa težinama
   iz :func:`src.training.samplers.compute_class_weights`. Ovde se obezbeđuje
   kroz jednostavan builder radi konstrukcije vođene config-om.
2. **Focal loss** (Lin et al. 2017) - smanjuje težinu lakih primera i fokusira
   treniranje na teške/retke. Podržava opcioni ``alpha`` po klasi.

``build_loss`` bira loss iz config specifikacije kako bi eksperimenti ostali
vođeni config-om.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Multi-class focal loss.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Parametri
    ---------
    gamma:
        Parametar fokusiranja (0 se svodi na (weighted) cross-entropy).
    alpha:
        Opcioni tensor težina po klasi oblika (num_classes,). Ponaša se kao
        težine klasa u weighted CE.
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
        # Registruj alpha kao buffer kako bi .to(device)/state_dict njime upravljali.
        if alpha is not None:
            self.register_buffer("alpha", torch.as_tensor(alpha, dtype=torch.float32))
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # CE po uzorku bez reduction; weight automatski primenjuje alpha_t.
        ce = F.cross_entropy(logits, target, weight=self.alpha, reduction="none")
        # p_t = exp(-ce_unweighted); dobija se preko softmax verovatnoće tačne klase.
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
    """Napravi loss modul iz config specifikacije.

    Specifikacije::

        {"type": "ce"}                              # obična cross-entropy
        {"type": "weighted_ce"}                     # CE sa class_weights
        {"type": "focal", "gamma": 2.0}             # focal, bez alpha
        {"type": "focal", "gamma": 2.0, "use_weights": true}  # focal sa alpha

    ``class_weights`` (iz brojeva na train splitu) je obavezno za 'weighted_ce' i
    za focal sa ``use_weights: true``.
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
