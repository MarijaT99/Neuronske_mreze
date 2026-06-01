"""End-to-end hierarchical evaluation + error-propagation analysis.

This is the heart of the thesis comparison. A hierarchical prediction is correct
only if **both** levels are right: the species head must pick the right species
*and* that species' disease head must pick the right disease. Because the disease
head used at inference is the one for the *predicted* species, a species error
almost always forces a disease error — this is **error propagation**, and we
quantify it explicitly.

Two label spaces are involved:

* per-species disease id (what each disease head outputs), and
* the global 38-class id (what the flat baseline outputs).

We map hierarchical (species, disease-in-species) predictions back to the global
38-class id via the label maps, so flat and hierarchical models are scored on the
**same** 38-class metric suite (macro F1 etc.) for a fair comparison.

Inputs are arrays already collected from a model (no torch here) so this module
is easy to unit-test and reuse.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.data.splits import LabelMaps
from src.evaluation.metrics import MetricResult, compute_metrics


@dataclass
class HierarchicalEvalResult:
    """Full breakdown of a hierarchical evaluation pass."""

    # End-to-end 38-class metrics (directly comparable to the flat baseline).
    end_to_end: MetricResult
    # Level-1 (species) metrics.
    species: MetricResult
    # Level-2 disease metrics computed ONLY on samples whose species was correct
    # (isolates the disease heads' own quality from propagated species errors).
    disease_given_correct_species: MetricResult

    # Error-propagation accounting.
    n: int
    species_correct: int
    end_to_end_correct: int
    # Of all end-to-end errors, how many had a WRONG species (propagated) vs a
    # right species but wrong disease (genuine L2 error)?
    errors_from_species: int
    errors_from_disease_only: int

    def summary(self) -> dict[str, float]:
        prop = (self.errors_from_species / max(self.errors_from_species + self.errors_from_disease_only, 1))
        return {
            "end_to_end_macro_f1": self.end_to_end.macro_f1,
            "end_to_end_weighted_f1": self.end_to_end.weighted_f1,
            "end_to_end_balanced_acc": self.end_to_end.balanced_accuracy,
            "end_to_end_accuracy": self.end_to_end.accuracy,  # secondary
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
    """Map (species_id, disease-in-species) pairs to the global 38-class id."""
    # Build a lookup: (species_id, local_disease) -> global class_id.
    lookup: dict[tuple[int, int], int] = {}
    for class_name, gid in maps.class_to_id.items():
        species, disease = class_name.split("___", 1)
        sid = maps.species_to_id[species]
        local = maps.disease_in_species_to_id[sid][disease]
        lookup[(sid, local)] = gid

    out = np.empty(len(species_id), dtype=np.int64)
    for i, (s, d) in enumerate(zip(species_id, disease_in_species)):
        # Clamp disease id into the valid range for that species (a disease head
        # can only output ids it has; defensive for trivial species -> 0).
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
    """Compute end-to-end + per-level metrics and error-propagation accounting.

    All four arrays are aligned per-sample. ``pred_disease_in_species`` must be
    the disease predicted by the head of the *predicted* species (so propagation
    is captured) — exactly what ``HierarchicalClassifier.predict`` returns.
    """
    true_species = np.asarray(true_species)
    true_disease_in_species = np.asarray(true_disease_in_species)
    pred_species = np.asarray(pred_species)
    pred_disease_in_species = np.asarray(pred_disease_in_species)
    n = len(true_species)

    # --- map both sides to global 38-class ids for end-to-end metrics ----
    true_global = _to_global_class_id(true_species, true_disease_in_species, maps)
    pred_global = _to_global_class_id(pred_species, pred_disease_in_species, maps)

    end_to_end = compute_metrics(true_global, pred_global, num_classes=maps.num_classes)
    species = compute_metrics(true_species, pred_species, num_classes=maps.num_species)

    # --- L2 quality isolated from propagation ----------------------------
    species_ok = pred_species == true_species
    if species_ok.any():
        dgcs = compute_metrics(
            true_disease_in_species[species_ok],
            pred_disease_in_species[species_ok],
        )
    else:
        dgcs = compute_metrics(np.array([0]), np.array([0]))

    # --- error-propagation accounting ------------------------------------
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
