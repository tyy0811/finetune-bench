"""Generate privacy-utility tradeoff charts from artifacts/dp_results.json."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"
RESULTS_DIR = Path(__file__).parent.parent / "results"


def load_dp_results():
    dp_path = ARTIFACTS_DIR / "dp_results.json"
    return json.loads(dp_path.read_text())


def load_baseline_f1():
    """Load non-DP M2 baseline F1 from results/all_results.json."""
    all_path = RESULTS_DIR / "all_results.json"
    if not all_path.exists():
        return None, None
    data = json.loads(all_path.read_text())
    f1s = [r["test_macro_f1"] for r in data if r["variant"] == "M2"]
    if not f1s:
        return None, None
    return float(np.mean(f1s)), float(np.std(f1s))


def plot_privacy_utility_tradeoff():
    """Plot epsilon vs Macro-F1 with error bars."""
    dp_data = load_dp_results()
    baseline_f1, baseline_std = load_baseline_f1()

    fig, ax = plt.subplots(figsize=(8, 5))

    # DP results
    configs = dp_data["results"]
    epsilons = [r["epsilon_actual"] for r in configs]
    f1s = [r["val_macro_f1"] for r in configs]
    stds = [r.get("val_macro_f1_std", 0) for r in configs]
    labels = [r["config"] for r in configs]

    ax.errorbar(epsilons, f1s, yerr=stds, fmt="o-", color="#e74c3c",
                capsize=5, linewidth=2, markersize=8, label="DP-SGD (full model)")

    for eps, f1, label in zip(epsilons, f1s, labels):
        ax.annotate(label, (eps, f1), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=8)

    # Non-DP baseline as horizontal line
    if baseline_f1 is not None:
        ax.axhline(y=baseline_f1, color="#2ecc71", linestyle="--", linewidth=2,
                   label=f"M2 baseline (no DP): {baseline_f1:.4f}")
        if baseline_std:
            ax.axhspan(baseline_f1 - baseline_std, baseline_f1 + baseline_std,
                       alpha=0.1, color="#2ecc71")

    ax.set_xlabel("Privacy Budget (ε)", fontsize=12)
    ax.set_ylabel("Macro-F1", fontsize=12)
    ax.set_title("Privacy-Utility Tradeoff: DP-SGD via dp-transformers", fontsize=14)
    ax.legend(loc="center right", fontsize=10)
    ax.set_xscale("log")
    ax.set_ylim(0, max(0.75, (baseline_f1 or 0.7) + 0.1))
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = ARTIFACTS_DIR / "privacy_utility_tradeoff.png"
    fig.savefig(out_path, dpi=150)
    print(f"Chart saved to {out_path}")
    plt.close()


def plot_mia_results():
    """Plot MIA AUC across model variants."""
    mia_path = ARTIFACTS_DIR / "mia_results.json"
    if not mia_path.exists():
        print("No mia_results.json found, skipping MIA chart")
        return

    mia_data = json.loads(mia_path.read_text())
    results = mia_data["results"]

    fig, ax = plt.subplots(figsize=(8, 5))

    models = [r["model"] for r in results]
    aucs = [r["mia_auc"] for r in results]
    colors = ["#e74c3c" if r.get("epsilon", "inf") not in ("inf",) else "#2ecc71"
              for r in results]

    bars = ax.bar(models, aucs, color=colors, alpha=0.8, edgecolor="black")
    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=1, label="Random guess (0.5)")
    ax.set_ylabel("MIA AUC", fontsize=12)
    ax.set_title("Membership Inference Attack: AUC by Model Variant", fontsize=14)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    for bar, auc in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{auc:.3f}", ha="center", fontsize=10)

    fig.tight_layout()
    out_path = ARTIFACTS_DIR / "mia_auc_chart.png"
    fig.savefig(out_path, dpi=150)
    print(f"Chart saved to {out_path}")
    plt.close()


if __name__ == "__main__":
    plot_privacy_utility_tradeoff()
    plot_mia_results()
