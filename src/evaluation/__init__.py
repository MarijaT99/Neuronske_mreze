"""Paket za evaluaciju: metrike (osetljive na neuravnoteženost), hijerarhijska evaluacija, analiza grešaka."""

from src.evaluation.error_analysis import (
    find_misclassified,
    gradcam_overlay,
    most_confused_pairs,
    plot_confusion_matrix,
)
from src.evaluation.hierarchical_eval import (
    HierarchicalEvalResult,
    evaluate_hierarchical,
)
from src.evaluation.metrics import (
    MetricResult,
    compute_metrics,
    confusion,
    per_class_table,
)

__all__ = [
    "MetricResult",
    "compute_metrics",
    "confusion",
    "per_class_table",
    "HierarchicalEvalResult",
    "evaluate_hierarchical",
    "plot_confusion_matrix",
    "most_confused_pairs",
    "find_misclassified",
    "gradcam_overlay",
]
