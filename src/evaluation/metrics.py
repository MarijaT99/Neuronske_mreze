"""Classification metrics — imbalance-aware by design.

Project rule (supervisor): **accuracy is never the primary metric** because the
dataset is imbalanced at both hierarchy levels. The primary metrics are:

* **macro F1**     — unweighted mean of per-class F1 (treats every class equally);
* **weighted F1**  — F1 weighted by support (realistic view);
* **balanced accuracy** — the accuracy-like alternative (mean per-class recall);
* **per-class precision / recall / F1** — where the imbalance actually shows;
* **confusion matrix**.

Plain accuracy is computed too, but only as a secondary figure to be reported
with the caveat that it over-credits the majority classes.

``compute_metrics`` returns a flat dict of scalars (easy to log per-epoch) plus
the per-class arrays under ``per_class``. The checkpoint/early-stopping criterion
upstream is ``macro_f1``.
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
    """Container for a metrics snapshot (one eval pass)."""

    macro_f1: float
    weighted_f1: float
    balanced_accuracy: float
    accuracy: float  # secondary only
    per_class: dict[str, np.ndarray] = field(default_factory=dict)

    def scalars(self) -> dict[str, float]:
        """Flat scalar dict for logging (excludes per-class arrays)."""
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
    """Compute the imbalance-aware metric suite from integer labels.

    ``num_classes`` ensures absent classes still appear (with zeros) in per-class
    arrays and the confusion matrix — important when a val batch/subset happens
    to miss a rare class.
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
    """Confusion matrix; ``normalize`` in {None, 'true', 'pred', 'all'}."""
    labels = list(range(num_classes)) if num_classes is not None else None
    return confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize)


def per_class_table(
    result: MetricResult,
    class_names: list[str] | None = None,
):
    """Build a per-class precision/recall/F1/support DataFrame (for reports)."""
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
