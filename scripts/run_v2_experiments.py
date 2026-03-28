"""Run v2 experiments: M2b no-company, temporal split, 50K scaling, calibration, per-class robustness."""

import json
from pathlib import Path

import numpy as np
import torch
from transformers import DistilBertTokenizer

from adapters.cfpb import CFPBAdapter
from evaluation.calibration import compute_ece, plot_reliability_diagram
from evaluation.robustness import run_robustness_eval
from models.fusion_model import MultimodalClassifier
from training.config import TrainConfig
from training.train import evaluate, train

RESULTS_DIR = Path("results")
SEEDS = [42, 123, 456]


def run_m2b_no_company():
    """Tier 1.1: No-company diagnostic — one seed."""
    print("\n" + "=" * 60)
    print("M2b: Fusion WITHOUT company features (seed=42)")
    print("=" * 60)

    config = TrainConfig(
        variant="M2",
        seed=42,
        sample_size=20_000,
        num_epochs=3,
        exclude_features=["company", "company_complaint_volume"],
        run_name="M2b_no_company_seed42",
    )
    result = train(config)
    result["variant"] = "M2b"
    return result


def run_temporal_split():
    """Tier 1.2: Temporal split diagnostic — M2, one seed."""
    print("\n" + "=" * 60)
    print("Temporal split: M2 (train: pre-2023, test: post-2023)")
    print("=" * 60)

    config = TrainConfig(
        variant="M2",
        seed=42,
        sample_size=20_000,
        num_epochs=3,
        split_strategy="temporal",
        cutoff_date="2023-01-01",
        run_name="M2_temporal_seed42",
    )
    result = train(config)
    result["variant"] = "M2_temporal"
    return result


def run_50k_scaling():
    """Tier 1.3: 50K scaling check — M1 and M2, 3 seeds each."""
    print("\n" + "=" * 60)
    print("50K scaling check")
    print("=" * 60)

    results = []
    for variant in ["M1", "M2"]:
        for seed in SEEDS:
            print(f"\n--- {variant} seed={seed} @ 50K ---")
            config = TrainConfig(
                variant=variant,
                seed=seed,
                sample_size=50_000,
                num_epochs=3,
                run_name=f"{variant}_50k_seed{seed}",
            )
            result = train(config)
            result["variant"] = f"{variant}_50k"
            results.append(result)
    return results


def run_calibration():
    """Tier 2.1: Confidence calibration for M2 and M3."""
    print("\n" + "=" * 60)
    print("Calibration: M2 and M3")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adapter = CFPBAdapter(sample_size=20_000, seed=42)
    splits = adapter.preprocess()
    tabular_dim = splits["train"]["tabular_features"].shape[1]

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    from torch.utils.data import DataLoader

    from training.train import ComplaintDataset

    test_ds = ComplaintDataset(
        splits["test"]["narratives"],
        splits["test"]["tabular_features"],
        splits["test"]["labels"],
        tokenizer,
        max_length=128,
    )
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

    cal_results = {}
    for variant in ["M2", "M3"]:
        model_path = RESULTS_DIR / f"{variant}_seed42_best.pt"
        if not model_path.exists():
            model_path = RESULTS_DIR / f"{variant}_seed42.pt"
        if not model_path.exists():
            print(f"Skipping calibration for {variant}: no model at {model_path}")
            continue

        model = MultimodalClassifier(
            num_classes=len(adapter.class_names),
            tabular_input_dim=tabular_dim,
            modality_dropout=(variant == "M3"),
            dropout=0.0,
        )
        model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
        model.to(device)
        model.eval()

        metrics, probs = evaluate(model, test_loader, adapter.class_names, device, return_probs=True)
        y_true = np.array(splits["test"]["labels"])

        ece = compute_ece(y_true, probs)
        print(f"  {variant}: ECE={ece:.4f}, Test F1={metrics.macro_f1:.4f}")

        plot_reliability_diagram(
            y_true, probs, f"{variant}: {'Fusion' if variant == 'M2' else 'Fusion+Dropout'}",
            save_path=str(RESULTS_DIR / f"calibration_{variant.lower()}.png"),
        )

        cal_results[variant] = {
            "ece": ece,
            "test_macro_f1": metrics.macro_f1,
            "test_accuracy": metrics.accuracy,
        }

    with open(RESULTS_DIR / "calibration_results.json", "w") as f:
        json.dump(cal_results, f, indent=2)
    print(f"Calibration results saved to {RESULTS_DIR / 'calibration_results.json'}")
    return cal_results


def run_per_class_robustness():
    """Tier 2.2: Per-class robustness heatmap for M3."""
    print("\n" + "=" * 60)
    print("Per-class robustness heatmap: M3")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 42
    adapter = CFPBAdapter(sample_size=20_000, seed=seed)
    splits = adapter.preprocess()
    tabular_dim = splits["train"]["tabular_features"].shape[1]
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    model_path = RESULTS_DIR / f"M3_seed{seed}_best.pt"
    if not model_path.exists():
        model_path = RESULTS_DIR / f"M3_seed{seed}.pt"
    if not model_path.exists():
        print(f"No M3 model found at {model_path}")
        return None

    model = MultimodalClassifier(
        num_classes=len(adapter.class_names),
        tabular_input_dim=tabular_dim,
        modality_dropout=False,
        dropout=0.0,
    )
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
    model.to(device)
    model.eval()

    # Run robustness eval — we need per-class F1, so we use the existing
    # run_robustness_eval and extend it
    run_robustness_eval(
        model=model,
        test_narratives=splits["test"]["narratives"],
        test_tabular=splits["test"]["tabular_features"],
        test_labels=splits["test"]["labels"].tolist(),
        class_names=adapter.class_names,
        tokenizer=tokenizer,
        is_text_only=False,
        seed=seed,
    )

    # Now run again but capture per-class F1 for each corruption
    from evaluation.metrics import compute_metrics
    from evaluation.robustness import (
        inject_typos,
        tabular_ablation,
        tabular_dropout,
        token_dropout,
        truncate_text,
    )

    test_tabular_t = torch.tensor(
        splits["test"]["tabular_features"], dtype=torch.float32
    )

    corruptions = [
        ("clean", {}),
        ("typo_10", {"type": "typo", "rate": 0.1}),
        ("typo_20", {"type": "typo", "rate": 0.2}),
        ("token_drop_20", {"type": "token_drop", "rate": 0.2}),
        ("token_drop_40", {"type": "token_drop", "rate": 0.4}),
        ("truncate_32", {"type": "truncate", "max_tokens": 32}),
        ("tabular_drop_50", {"type": "tabular_drop", "rate": 0.5}),
        ("tabular_ablation", {"type": "tabular_ablation"}),
    ]

    heatmap_data = {}
    test_narratives = splits["test"]["narratives"]
    test_labels = splits["test"]["labels"].tolist()

    for name, params in corruptions:
        corruption_type = params.get("type", "")
        all_preds = []

        if corruption_type == "typo":
            texts = [inject_typos(t, rate=params["rate"], seed=seed + i)
                     for i, t in enumerate(test_narratives)]
        elif corruption_type == "truncate":
            texts = [truncate_text(t, max_tokens=params["max_tokens"])
                     for t in test_narratives]
        else:
            texts = test_narratives

        batch_size = 16
        for start in range(0, len(texts), batch_size):
            end = min(start + batch_size, len(texts))
            batch_texts = texts[start:end]
            batch_tab = test_tabular_t[start:end].to(device)

            encodings = tokenizer(
                batch_texts, max_length=128, padding="max_length",
                truncation=True, return_tensors="pt",
            )
            text_inputs = {k: v.to(device) for k, v in encodings.items()}

            if corruption_type == "token_drop":
                text_inputs["input_ids"], text_inputs["attention_mask"] = token_dropout(
                    text_inputs["input_ids"],
                    attention_mask=text_inputs["attention_mask"],
                    rate=params["rate"], seed=seed + start,
                )

            if corruption_type == "tabular_drop":
                batch_tab = tabular_dropout(batch_tab, rate=params["rate"], seed=seed + start)
            elif corruption_type == "tabular_ablation":
                batch_tab = tabular_ablation(batch_tab)

            with torch.no_grad():
                logits = model(text_inputs, batch_tab)
                all_preds.extend(logits.argmax(dim=-1).cpu().tolist())

        metrics = compute_metrics(test_labels, all_preds, adapter.class_names)
        heatmap_data[name] = {
            cn: float(f1) for cn, f1 in zip(adapter.class_names, metrics.per_class_f1)
        }
        print(f"  {name}: per-class F1 computed")

    with open(RESULTS_DIR / "per_class_robustness.json", "w") as f:
        json.dump(heatmap_data, f, indent=2)

    # Generate heatmap
    _plot_heatmap(heatmap_data, adapter.class_names)
    return heatmap_data


def _plot_heatmap(heatmap_data, class_names):
    import matplotlib.pyplot as plt

    corruptions = list(heatmap_data.keys())
    matrix = np.array([[heatmap_data[c].get(cls, 0) for cls in class_names]
                        for c in corruptions])

    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(corruptions)))
    ax.set_yticklabels(corruptions, fontsize=9)

    for i in range(len(corruptions)):
        for j in range(len(class_names)):
            val = matrix[i, j]
            color = "white" if val < 0.4 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color=color, fontsize=7)

    ax.set_title("M3 Per-Class F1 Under Corruption")
    fig.colorbar(im, ax=ax, label="F1 Score")
    plt.tight_layout()
    out_path = RESULTS_DIR / "robustness_heatmap.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved heatmap to {out_path}")
    plt.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run v2 experiments")
    parser.add_argument("--skip-m2b", action="store_true")
    parser.add_argument("--skip-temporal", action="store_true")
    parser.add_argument("--skip-50k", action="store_true")
    parser.add_argument("--skip-calibration", action="store_true")
    parser.add_argument("--skip-heatmap", action="store_true")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    v2_results = {}

    if not args.skip_m2b:
        v2_results["m2b"] = run_m2b_no_company()

    if not args.skip_temporal:
        v2_results["temporal"] = run_temporal_split()

    if not args.skip_50k:
        v2_results["scaling_50k"] = run_50k_scaling()

    if not args.skip_calibration:
        v2_results["calibration"] = run_calibration()

    if not args.skip_heatmap:
        v2_results["per_class_robustness"] = run_per_class_robustness()

    with open(RESULTS_DIR / "v2_results.json", "w") as f:
        json.dump(v2_results, f, indent=2, default=str)
    print(f"\nAll v2 results saved to {RESULTS_DIR / 'v2_results.json'}")


if __name__ == "__main__":
    main()
