"""End-to-end hijerarhijska evaluacija + analiza error propagation.

Ovo je srž poređenja u radu. Hijerarhijska predikcija je tačna samo ako su **oba**
nivoa tačna: species head mora izabrati pravi species *i* disease head tog
species-a mora izabrati pravu bolest. Pošto je disease head koji se koristi pri
inferenci onaj za *predviđeni* species, greška u species-u skoro uvek izaziva i
grešku u bolesti - to je **error propagation**, i mi to eksplicitno kvantifikujemo.

Uključena su dva prostora labela:

* disease id po species-u (ono što svaki disease head daje), i
* globalni 38-class id (ono što flat baseline daje).

Hijerarhijske (species, disease-in-species) predikcije preslikavamo nazad na
globalni 38-class id preko label mapa, tako da se flat i hijerarhijski modeli
ocenjuju na **istom** skupu metrika za 38 klasa (macro F1 itd.) radi poštenog
poređenja.

Ulazi su nizovi već prikupljeni iz modela (ovde nema torch-a) pa je ovaj modul
lako unit-testirati i ponovo koristiti.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.data.splits import LabelMaps
from src.evaluation.metrics import MetricResult, compute_metrics


@dataclass
class HierarchicalEvalResult:
    """Potpun pregled jednog prolaza hijerarhijske evaluacije."""

    # End-to-end metrike za 38 klasa (direktno uporedive sa flat baseline-om).
    end_to_end: MetricResult
    # Metrike nivoa 1 (species).
    species: MetricResult
    # Metrike bolesti na nivou 2 računate SAMO na uzorcima čiji je species tačan
    # (izoluje sopstveni kvalitet disease head-ova od prenetih grešaka u species-u).
    disease_given_correct_species: MetricResult

    # Obračun error propagation.
    n: int
    species_correct: int
    end_to_end_correct: int
    # Od svih end-to-end grešaka, koliko ih je imalo POGREŠAN species (preneto) u
    # odnosu na tačan species ali pogrešnu bolest (prava greška na nivou 2)?
    errors_from_species: int
    errors_from_disease_only: int

    def summary(self) -> dict[str, float]:
        prop = (self.errors_from_species / max(self.errors_from_species + self.errors_from_disease_only, 1))
        return {
            "end_to_end_macro_f1": self.end_to_end.macro_f1,
            "end_to_end_weighted_f1": self.end_to_end.weighted_f1,
            "end_to_end_balanced_acc": self.end_to_end.balanced_accuracy,
            "end_to_end_accuracy": self.end_to_end.accuracy,  # sekundarno
            "species_macro_f1": self.species.macro_f1,
            "disease_given_correct_species_macro_f1": self.disease_given_correct_species.macro_f1,
            "species_correct_frac": self.species_correct / max(self.n, 1),
            "end_to_end_correct_frac": self.end_to_end_correct / max(self.n, 1),
            "errors_from_species": self.errors_from_species,
            "errors_from_disease_only": self.errors_from_disease_only,
            "share_of_errors_due_to_species": prop,
        }


def _to_global_class_id(
    species_id: np.ndarray,
    disease_in_species: np.ndarray,
    maps: LabelMaps,
) -> np.ndarray:
    """Preslikaj (species_id, disease-in-species) parove na globalni 38-class id."""
    # Napravi lookup: (species_id, local_disease) -> global class_id.
    lookup: dict[tuple[int, int], int] = {}
    for class_name, gid in maps.class_to_id.items():
        species, disease = class_name.split("___", 1)
        sid = maps.species_to_id[species]
        local = maps.disease_in_species_to_id[sid][disease]
        lookup[(sid, local)] = gid

    out = np.empty(len(species_id), dtype=np.int64)
    for i, (s, d) in enumerate(zip(species_id, disease_in_species)):
        # Ograniči disease id na važeći opseg za taj species (disease head može
        # dati samo id-jeve koje ima; defanzivno za trivijalne species -> 0).
        key = (int(s), int(d))
        if key not in lookup:
            key = (int(s), 0)
        out[i] = lookup[key]
    return out


def evaluate_hierarchical(
    *,
    true_species: np.ndarray,
    true_disease_in_species: np.ndarray,
    pred_species: np.ndarray,
    pred_disease_in_species: np.ndarray,
    maps: LabelMaps,
) -> HierarchicalEvalResult:
    """Izračunaj end-to-end + metrike po nivou i obračun error propagation.

    Sva četiri niza su poravnata po uzorku. ``pred_disease_in_species`` mora biti
    bolest koju predviđa head *predviđenog* species-a (da bi se obuhvatila
    propagacija) - tačno ono što ``HierarchicalClassifier.predict`` vraća.
    """
    true_species = np.asarray(true_species)
    true_disease_in_species = np.asarray(true_disease_in_species)
    pred_species = np.asarray(pred_species)
    pred_disease_in_species = np.asarray(pred_disease_in_species)
    n = len(true_species)

    # --- preslikaj obe strane na globalne 38-class id-jeve za end-to-end metrike ----
    true_global = _to_global_class_id(true_species, true_disease_in_species, maps)
    pred_global = _to_global_class_id(pred_species, pred_disease_in_species, maps)

    end_to_end = compute_metrics(true_global, pred_global, num_classes=maps.num_classes)
    species = compute_metrics(true_species, pred_species, num_classes=maps.num_species)

    # --- kvalitet na nivou 2 izolovan od propagacije ----------------------
    species_ok = pred_species == true_species
    if species_ok.any():
        dgcs = compute_metrics(
            true_disease_in_species[species_ok],
            pred_disease_in_species[species_ok],
        )
    else:
        dgcs = compute_metrics(np.array([0]), np.array([0]))

    # --- obračun error propagation ----------------------------------------
    end_to_end_ok = pred_global == true_global
    end_to_end_correct = int(end_to_end_ok.sum())
    errors = ~end_to_end_ok
    errors_from_species = int((errors & ~species_ok).sum())
    errors_from_disease_only = int((errors & species_ok).sum())

    return HierarchicalEvalResult(
        end_to_end=end_to_end,
        species=species,
        disease_given_correct_species=dgcs,
        n=n,
        species_correct=int(species_ok.sum()),
        end_to_end_correct=end_to_end_correct,
        errors_from_species=errors_from_species,
        errors_from_disease_only=errors_from_disease_only,
    )
