"""Evaluation metrics: macro-F1, per-class precision/recall, confusion matrix."""

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


@dataclass
class MetricsResult:
    macro_f1: float
    accuracy: float
    per_class_f1: np.ndarray
    per_class_precision: np.ndarray
    per_class_recall: np.ndarray
    confusion_mat: np.ndarray
    report: str


def compute_metrics(
    labels: list[int],
    preds: list[int],
    class_names: list[str] | None = None,
) -> MetricsResult:
    # Build explicit label set so arrays always align with class_names,
    # even when a split is missing some classes.
    all_labels = list(range(len(class_names))) if class_names else None

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, labels=all_labels, average=None, zero_division=0
    )

    report = classification_report(
        labels, preds,
        labels=all_labels,
        target_names=class_names,
        zero_division=0,
    )

    return MetricsResult(
        macro_f1=f1_score(labels, preds, labels=all_labels, average="macro", zero_division=0),
        accuracy=accuracy_score(labels, preds),
        per_class_f1=f1,
        per_class_precision=precision,
        per_class_recall=recall,
        confusion_mat=confusion_matrix(labels, preds, labels=all_labels),
        report=report,
    )
