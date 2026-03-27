"""Generate markdown ablation tables and training curves from results."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np

RESULTS_DIR = Path("results")


def load_results() -> list[dict]:
    path = RESULTS_DIR / "all_results.json"
    with open(path) as f:
        return json.load(f)


def generate_table1(results: list[dict]) -> str:
    """Ablation Table 1: Component contribution on clean data."""
    lines = [
        "# Ablation Table 1 -- Component Contribution (clean data, random split)\n",
        "| Variant | Macro-F1 | Accuracy | Per-class F1 range | Delta vs B1 |",
        "|---------|----------|----------|-------------------|-------------|",
    ]

    b1 = next((r for r in results if r["variant"] == "B1"), None)
    b1_f1 = b1["test_macro_f1"] if b1 else 0

    for variant_id in ["B1", "B2", "M1", "M2", "M3"]:
        variant_results = [r for r in results if r["variant"] == variant_id]
        if not variant_results:
            continue

        f1_scores = [r["test_macro_f1"] for r in variant_results]
        acc_scores = [r["test_accuracy"] for r in variant_results]

        if len(f1_scores) > 1:
            f1_str = f"{np.mean(f1_scores):.4f} +/- {np.std(f1_scores):.4f}"
            acc_str = f"{np.mean(acc_scores):.4f} +/- {np.std(acc_scores):.4f}"
        else:
            f1_str = f"{f1_scores[0]:.4f}"
            acc_str = f"{acc_scores[0]:.4f}"

        all_per_class = []
        for r in variant_results:
            all_per_class.extend(r.get("per_class_f1", []))
        range_str = f"{min(all_per_class):.2f}-{max(all_per_class):.2f}" if all_per_class else "--"

        delta = np.mean(f1_scores) - b1_f1
        delta_str = f"+{delta:.4f}" if variant_id != "B1" else "--"

        names = {
            "B1": "B1: TF-IDF + LogReg",
            "B2": "B2: Tabular-only LightGBM",
            "M1": "M1: DistilBERT text-only",
            "M2": "M2: Full fusion",
            "M3": "M3: Fusion + dropout",
        }
        lines.append(f"| {names[variant_id]} | {f1_str} | {acc_str} | {range_str} | {delta_str} |")

    lines.append("\n*DL variants report mean +/- std over 3 seeds.*")
    return "\n".join(lines)


def generate_table2(results: list[dict]) -> str:
    """Ablation Table 2: Robustness under corruption."""
    rob_path = RESULTS_DIR / "robustness_results.json"
    if not rob_path.exists():
        return "# Ablation Table 2 -- Robustness Under Corruption\n\n*Run robustness evaluation first.*"

    with open(rob_path) as f:
        rob_results = json.load(f)

    corruptions = [
        ("None (clean)", "--", "clean"),
        ("Typo injection", "10%", "typo_10"),
        ("Typo injection", "20%", "typo_20"),
        ("Token dropout", "20%", "token_drop_20"),
        ("Token dropout", "40%", "token_drop_40"),
        ("Truncation", "32 tokens", "truncate_32"),
        ("Tabular dropout", "50%", "tabular_drop_50"),
        ("Full tabular ablation", "--", "tabular_ablation"),
    ]

    lines = [
        "# Ablation Table 2 -- Robustness Under Corruption\n",
        "| Corruption | Rate | M1: Text-only | M2: Fusion | M3: Fusion+Dropout | Delta (M3 vs M1) |",
        "|------------|------|---------------|------------|-------------------|-------------------|",
    ]

    for display_name, rate, key in corruptions:
        cells = [display_name, rate]
        m1_f1 = None
        m3_f1 = None
        for variant in ["M1", "M2", "M3"]:
            entry = rob_results.get(variant, {}).get(key)
            if entry is None:
                cells.append("N/A")
            else:
                f1 = entry["macro_f1"]
                cells.append(f"{f1:.4f}")
                if variant == "M1":
                    m1_f1 = f1
                if variant == "M3":
                    m3_f1 = f1

        if m1_f1 is not None and m3_f1 is not None:
            delta = m3_f1 - m1_f1
            cells.append(f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}")
        else:
            cells.append("--")

        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


def generate_training_curves():
    """Generate training curves from MLflow metrics."""
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("finetune-bench")
    if experiment is None:
        print("No MLflow experiment found. Skipping training curves.")
        return

    runs = client.search_runs(experiment.experiment_id)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for run in runs:
        name = run.info.run_name
        if not name or not any(name.startswith(v) for v in ["M1_", "M2_", "M3_"]):
            continue

        # Loss curve
        loss_history = client.get_metric_history(run.info.run_id, "train_loss_epoch")
        if loss_history:
            epochs = [m.step for m in loss_history]
            losses = [m.value for m in loss_history]
            axes[0].plot(epochs, losses, label=name, alpha=0.7)

        # F1 curve
        f1_history = client.get_metric_history(run.info.run_id, "val_macro_f1")
        if f1_history:
            epochs = [m.step for m in f1_history]
            f1s = [m.value for m in f1_history]
            axes[1].plot(epochs, f1s, label=name, alpha=0.7)

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Training Loss")
    axes[0].set_title("Training Loss")
    axes[0].legend(fontsize=8)

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation Macro-F1")
    axes[1].set_title("Validation Macro-F1")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "training_curves.png", dpi=150)
    print(f"Saved training curves to {RESULTS_DIR / 'training_curves.png'}")
    plt.close()


def main():
    results = load_results()

    table1 = generate_table1(results)
    with open(RESULTS_DIR / "ablation_component.md", "w") as f:
        f.write(table1)
    print(table1)

    print()

    table2 = generate_table2(results)
    with open(RESULTS_DIR / "ablation_robustness.md", "w") as f:
        f.write(table2)
    print(table2)

    print()
    generate_training_curves()


if __name__ == "__main__":
    main()
