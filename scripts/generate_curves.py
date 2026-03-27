"""Generate training curves from results/all_results.json."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).parent.parent / "results"

VARIANT_STYLES = {
    "M1": {"color": "#1f77b4", "label": "M1: Text-only"},
    "M2": {"color": "#ff7f0e", "label": "M2: Fusion"},
    "M3": {"color": "#2ca02c", "label": "M3: Fusion+Dropout"},
}

SEED_MARKERS = {42: "o", 123: "s", 456: "^"}


def load_epoch_data():
    with open(RESULTS_DIR / "all_results.json") as f:
        results = json.load(f)
    return [r for r in results if "epochs" in r]


def plot_training_curves(results):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for r in results:
        variant = r["variant"]
        seed = r["seed"]
        style = VARIANT_STYLES[variant]
        epochs = [e["epoch"] for e in r["epochs"]]
        losses = [e["train_loss"] for e in r["epochs"]]
        f1s = [e["val_macro_f1"] for e in r["epochs"]]

        label = f"{style['label']} (s={seed})"
        marker = SEED_MARKERS.get(seed, "o")

        axes[0].plot(epochs, losses, color=style["color"], marker=marker,
                     label=label, alpha=0.7, linewidth=1.5, markersize=5)
        axes[1].plot(epochs, f1s, color=style["color"], marker=marker,
                     label=label, alpha=0.7, linewidth=1.5, markersize=5)

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Training Loss")
    axes[0].set_title("Training Loss")
    axes[0].set_xticks([1, 2, 3])
    axes[0].legend(fontsize=7, ncol=1, loc="upper right")
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation Macro-F1")
    axes[1].set_title("Validation Macro-F1")
    axes[1].set_xticks([1, 2, 3])
    axes[1].legend(fontsize=7, ncol=1, loc="lower right")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = RESULTS_DIR / "training_curves.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved training curves to {out_path}")
    plt.close()


def plot_robustness_chart():
    with open(RESULTS_DIR / "robustness_results.json") as f:
        rob = json.load(f)

    corruptions = [
        ("clean", "Clean"),
        ("typo_10", "Typo 10%"),
        ("typo_20", "Typo 20%"),
        ("token_drop_20", "TokDrop 20%"),
        ("token_drop_40", "TokDrop 40%"),
        ("truncate_32", "Trunc 32"),
    ]

    x = np.arange(len(corruptions))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (variant, style) in enumerate(VARIANT_STYLES.items()):
        f1s = [rob[variant].get(key, {}).get("macro_f1", 0) for key, _ in corruptions]
        ax.bar(x + i * width, f1s, width, label=style["label"],
               color=style["color"], alpha=0.8)

    ax.set_xlabel("Corruption")
    ax.set_ylabel("Macro-F1")
    ax.set_title("Robustness: Macro-F1 Under Text Corruption")
    ax.set_xticks(x + width)
    ax.set_xticklabels([label for _, label in corruptions], rotation=15)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 0.75)

    plt.tight_layout()
    out_path = RESULTS_DIR / "robustness_chart.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved robustness chart to {out_path}")
    plt.close()


if __name__ == "__main__":
    results = load_epoch_data()
    plot_training_curves(results)
    plot_robustness_chart()
