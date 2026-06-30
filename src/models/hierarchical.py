"""Kontejner hijerarhijskog klasifikatora.

Sadrži jedan **species** klasifikator (L1, 14 klasa) i dict **disease**
klasifikatora po species-u (L2). Tokom inferencije species model bira species, a
odgovarajući disease head predviđa bolest *unutar* tog species-a.

Ovo preslikava dvonivovsku šemu projekta i čini **error propagation**
eksplicitnim: ako species head pogreši, disease predikcija se uzima iz pogrešnog
head-a i (gotovo uvek) je takođe pogrešna. End-to-end (species, disease)
predikcija je ono što hijerarhijska evaluacija ocenjuje.

Napomene o dizajnu
------------------
* Species sa samo jednom disease klasom (Blueberry, Orange, Raspberry, Soybean,
  Squash — svi "healthy") ne zahtevaju pravi disease head; kontejner direktno
  vraća njihov jedini disease id (0). ``trivial_species`` ih beleži.
* Ovo je kontejner, a ne jedinstveni ``nn.Module`` forward graf: head-ovi se
  treniraju nezavisno (jednostavnije, u skladu sa planom) i kombinuju tek pri
  evaluaciji. Svaki head je i dalje ``nn.Module`` i registrovan je, tako da
  ``.to(device)``, ``state_dict()`` itd. rade nad celinom.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class HierarchicalPrediction:
    """Rezultati hijerarhijske predikcije za batch."""

    species_id: torch.Tensor          # (B,) predviđeni species
    disease_id_in_species: torch.Tensor  # (B,) predviđena bolest, lokalna za species
    species_logits: torch.Tensor      # (B, num_species)


class HierarchicalClassifier(nn.Module):
    """Kontejner: species model + disease head-ovi po species-u.

    Parameters
    ----------
    species_model:
        L1 klasifikator koji proizvodi ``num_species`` logits-a.
    disease_models:
        Mapiranje ``species_id -> nn.Module`` (svaki proizvodi disease logits-e
        tog species-a). Species-i kojih nema u dict-u tretiraju se kao trivijalni
        (jedna disease klasa, uvek predviđena kao id 0).
    num_species:
        Ukupan broj species-a (za validaciju / vođenje evidencije).
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
        # Ključevi ModuleDict-a moraju biti string-ovi.
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
        """End-to-end predikcija sa ugrađenim error propagation-om.

        Prvo se predviđa species; bolest svakog uzorka zatim predviđa head
        njegovog *predviđenog* species-a (tako da se greške u species-u
        propagiraju). Uzorci čiji je predviđeni species trivijalan dobijaju
        disease id 0.
        """
        self.eval()
        species_logits = self.species_model(x)
        species_pred = species_logits.argmax(dim=1)

        disease_pred = torch.zeros_like(species_pred)
        for sid in species_pred.unique().tolist():
            mask = species_pred == sid
            if not self.has_disease_head(sid):
                continue  # trivijalan species -> disease id 0 (već postavljen)
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
