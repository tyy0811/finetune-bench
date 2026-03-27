"""Confidence calibration metrics and reliability diagrams."""

import numpy as np


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error for multiclass (top-label).

    Args:
        y_true: (n_samples,) true class indices.
        y_prob: (n_samples, n_classes) softmax probabilities.
        n_bins: number of confidence bins.

    Returns:
        ECE score (lower is better, 0 = perfectly calibrated).
    """
    confidences = y_prob.max(axis=1)
    predictions = y_prob.argmax(axis=1)
    accuracies = (predictions == y_true)

    ece = 0.0
    for bin_lower, bin_upper in zip(
        np.linspace(0, 1, n_bins + 1)[:-1],
        np.linspace(0, 1, n_bins + 1)[1:],
    ):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        if in_bin.sum() == 0:
            continue
        bin_accuracy = accuracies[in_bin].mean()
        bin_confidence = confidences[in_bin].mean()
        bin_weight = in_bin.sum() / len(y_true)
        ece += bin_weight * abs(bin_accuracy - bin_confidence)

    return float(ece)


def plot_reliability_diagram(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    n_bins: int = 10,
    save_path: str | None = None,
):
    """Top-label reliability diagram for multiclass classifier."""
    import matplotlib.pyplot as plt

    confidences = y_prob.max(axis=1)
    predictions = y_prob.argmax(axis=1)
    accuracies = (predictions == y_true).astype(float)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_accs, bin_confs, bin_counts = [], [], []

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            bin_accs.append(0)
            bin_confs.append((lo + hi) / 2)
            bin_counts.append(0)
        else:
            bin_accs.append(accuracies[mask].mean())
            bin_confs.append(confidences[mask].mean())
            bin_counts.append(mask.sum())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.bar(bin_confs, bin_accs, width=1 / n_bins, alpha=0.7,
            edgecolor="black", label="Model")
    ax1.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax1.set_xlabel("Mean predicted confidence")
    ax1.set_ylabel("Fraction of positives (accuracy)")
    ax1.set_title(f"{model_name} — Reliability Diagram")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.bar(bin_confs, bin_counts, width=1 / n_bins, alpha=0.7,
            edgecolor="black")
    ax2.set_xlabel("Mean predicted confidence")
    ax2.set_ylabel("Count")
    ax2.set_title("Prediction Confidence Distribution")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
