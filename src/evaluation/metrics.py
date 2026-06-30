"""Metrike klasifikacije — po dizajnu osetljive na neuravnoteženost.

Pravilo projekta (mentor): **accuracy nikada nije primarna metrika** jer je
dataset neuravnotežen na oba nivoa hijerarhije. Primarne metrike su:

* **macro F1**     — neponderisana srednja vrednost F1 po klasi (sve klase jednako);
* **weighted F1**  — F1 ponderisan support-om (realističan prikaz);
* **balanced accuracy** — alternativa nalik accuracy-ju (srednji recall po klasi);
* **precision / recall / F1 po klasi** — gde se neuravnoteženost zaista vidi;
* **confusion matrix**.

Obična accuracy se takođe računa, ali samo kao sekundarna vrednost koja se
izveštava uz napomenu da precenjuje većinske klase.

``compute_metrics`` vraća ravan dict skalara (lak za logovanje po epoch-u) plus
nizove po klasi pod ``per_class``. Kriterijum za checkpoint/early-stopping uzvodno
je ``macro_f1``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


@dataclass
class MetricResult:
    """Kontejner za snimak metrika (jedan prolaz evaluacije)."""

    macro_f1: float
    weighted_f1: float
    balanced_accuracy: float
    accuracy: float  # samo sekundarno
    per_class: dict[str, np.ndarray] = field(default_factory=dict)

    def scalars(self) -> dict[str, float]:
        """Ravan dict skalara za logovanje (bez nizova po klasi)."""
        return {
            "macro_f1": self.macro_f1,
            "weighted_f1": self.weighted_f1,
            "balanced_accuracy": self.balanced_accuracy,
            "accuracy": self.accuracy,
        }


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    num_classes: int | None = None,
) -> MetricResult:
    """Izračunaj skup metrika osetljivih na neuravnoteženost iz celobrojnih labela.

    ``num_classes`` obezbeđuje da odsutne klase i dalje budu prisutne (sa nulama) u
    nizovima po klasi i u confusion matrici — važno kada val batch/podskup slučajno
    izostavi retku klasu.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = list(range(num_classes)) if num_classes is not None else None

    macro = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    weighted = f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)

    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )

    return MetricResult(
        macro_f1=float(macro),
        weighted_f1=float(weighted),
        balanced_accuracy=float(bal_acc),
        accuracy=float(acc),
        per_class={
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "support": support,
        },
    )


def confusion(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    num_classes: int | None = None,
    normalize: str | None = None,
) -> np.ndarray:
    """Confusion matrix; ``normalize`` u {None, 'true', 'pred', 'all'}."""
    labels = list(range(num_classes)) if num_classes is not None else None
    return confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize)


def per_class_table(
    result: MetricResult,
    class_names: list[str] | None = None,
):
    """Napravi DataFrame sa precision/recall/F1/support po klasi (za izveštaje)."""
    import pandas as pd

    pc = result.per_class
    n = len(pc["f1"])
    names = class_names if class_names is not None else [str(i) for i in range(n)]
    return pd.DataFrame(
        {
            "class": names,
            "precision": pc["precision"],
            "recall": pc["recall"],
            "f1": pc["f1"],
            "support": pc["support"],
        }
    ).sort_values("f1").reset_index(drop=True)
