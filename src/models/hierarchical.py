"""Hierarchical classifier container.

Holds one **species** classifier (L1, 14 classes) plus a dict of per-species
**disease** classifiers (L2). At inference the species model picks a species, and
the corresponding disease head predicts the disease *within* that species.

This mirrors the project's two-level scheme and makes **error propagation**
explicit: if the species head is wrong, the disease prediction is taken from the
wrong head and is (almost always) wrong too. The end-to-end (species, disease)
prediction is what the hierarchical evaluation scores.

Design notes
------------
* Species with a single disease class (Blueberry, Orange, Raspberry, Soybean,
  Squash — all "healthy") need no real disease head; the container returns their
  only disease id (0) directly. ``trivial_species`` records these.
* This is a container, not a single ``nn.Module`` forward graph: the heads are
  trained independently (simpler, matches the plan) and combined only at eval.
  Each head is still an ``nn.Module`` and is registered so ``.to(device)``,
  ``state_dict()`` etc. work on the whole thing.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class HierarchicalPrediction:
    """Batched hierarchical prediction results."""

    species_id: torch.Tensor          # (B,) predicted species
    disease_id_in_species: torch.Tensor  # (B,) predicted disease, local to species
    species_logits: torch.Tensor      # (B, num_species)


class HierarchicalClassifier(nn.Module):
    """Container: a species model + per-species disease heads.

    Parameters
    ----------
    species_model:
        L1 classifier producing ``num_species`` logits.
    disease_models:
        Mapping ``species_id -> nn.Module`` (each producing that species'
        disease logits). Species absent from the dict are treated as trivial
        (single disease class, always predicted as id 0).
    num_species:
        Total number of species (for validation / bookkeeping).
    """

    def __init__(
        self,
        species_model: nn.Module,
        disease_models: dict[int, nn.Module],
        *,
        num_species: int,
    ):
        super().__init__()
        self.species_model = species_model
        # ModuleDict keys must be strings.
        self.disease_models = nn.ModuleDict({str(k): m for k, m in disease_models.items()})
        self.num_species = num_species
        self.trivial_species = sorted(
            set(range(num_species)) - {int(k) for k in self.disease_models.keys()}
        )

    # ------------------------------------------------------------------ #
    def has_disease_head(self, species_id: int) -> bool:
        return str(species_id) in self.disease_models

    def disease_head(self, species_id: int) -> nn.Module:
        return self.disease_models[str(species_id)]

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> HierarchicalPrediction:
        """End-to-end prediction with error propagation built in.

        Species is predicted first; each sample's disease is then predicted by
        the head of its *predicted* species (so species errors propagate).
        Samples whose predicted species is trivial get disease id 0.
        """
        self.eval()
        species_logits = self.species_model(x)
        species_pred = species_logits.argmax(dim=1)

        disease_pred = torch.zeros_like(species_pred)
        for sid in species_pred.unique().tolist():
            mask = species_pred == sid
            if not self.has_disease_head(sid):
                continue  # trivial species -> disease id 0 (already set)
            head = self.disease_head(sid)
            logits = head(x[mask])
            disease_pred[mask] = logits.argmax(dim=1)

        return HierarchicalPrediction(
            species_id=species_pred,
            disease_id_in_species=disease_pred,
            species_logits=species_logits,
        )

    def forward(self, x: torch.Tensor) -> HierarchicalPrediction:
        return self.predict(x)
