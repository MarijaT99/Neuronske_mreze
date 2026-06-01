"""Evaluation package: metrics (imbalance-aware), hierarchical eval, error analysis."""

from src.evaluation.metrics import (
    MetricResult,
    compute_metrics,
    confusion,
    per_class_table,
)

__all__ = ["MetricResult", "compute_metrics", "confusion", "per_class_table"]
