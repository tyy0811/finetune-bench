"""Run full ablation matrix: baselines, M1, M2, M3 across seeds.

Also runs robustness evaluation and ONNX export/latency benchmarking.
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import DistilBertTokenizer

from adapters.cfpb import CFPBAdapter
from evaluation.export import benchmark_latency, export_to_onnx
from evaluation.robustness import run_robustness_eval
from models.baselines import train_lightgbm, train_tfidf_logreg
from models.fusion_model import MultimodalClassifier
from training.config import TrainConfig
from training.train import train

SEEDS = [42, 123, 456]
RESULTS_DIR = Path("results")


def run_baselines(sample_size: int = 20_000, seed: int = 42):
    """Run B1 and B2 baselines."""
    adapter = CFPBAdapter(sample_size=sample_size, seed=seed)
    splits = adapter.preprocess()

    print("\n" + "=" * 60)
    print("Running B1: TF-IDF + LogReg")
    print("=" * 60)
    b1_results = train_tfidf_logreg(
        splits["train"]["narratives"],
        splits["train"]["labels"],
        splits["test"]["narratives"],
        splits["test"]["labels"],
        adapter.class_names,
    )

    print("\n" + "=" * 60)
    print("Running B2: LightGBM (tabular-only)")
    print("=" * 60)
    b2_results = train_lightgbm(
        splits["train"]["tabular_features"],
        splits["train"]["labels"],
        splits["val"]["tabular_features"],
        splits["val"]["labels"],
        splits["test"]["tabular_features"],
        splits["test"]["labels"],
        adapter.class_names,
    )

    return b1_results, b2_results


def run_dl_variants(sample_size: int = 20_000, num_epochs: int = 3):
    """Run M1, M2, M3 across all seeds."""
    all_results = []

    for variant in ["M1", "M2", "M3"]:
        for seed in SEEDS:
            print("\n" + "=" * 60)
            print(f"Running {variant} seed={seed}")
            print("=" * 60)

            config = TrainConfig(
                variant=variant,
                seed=seed,
                sample_size=sample_size,
                num_epochs=num_epochs,
            )
            result = train(config)
            all_results.append(result)

    return all_results


def run_robustness(sample_size: int = 20_000, seed: int = 42):
    """Run robustness evaluation for M1, M2, M3 (first seed)."""
    adapter = CFPBAdapter(sample_size=sample_size, seed=seed)
    splits = adapter.preprocess()
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    tabular_dim = splits["train"]["tabular_features"].shape[1]

    robustness_results = {}
    for variant in ["M1", "M2", "M3"]:
        run_name = f"{variant}_seed{seed}"
        model_path = RESULTS_DIR / f"{run_name}_best.pt"
        if not model_path.exists():
            model_path = RESULTS_DIR / f"{run_name}.pt"
        if not model_path.exists():
            print(f"Skipping robustness for {variant}: no saved model at {model_path}")
            continue

        print(f"\n{'=' * 60}")
        print(f"Robustness evaluation: {variant}")
        print("=" * 60)

        model = MultimodalClassifier(
            num_classes=len(adapter.class_names),
            tabular_input_dim=tabular_dim,
            modality_dropout=False,
            dropout=0.0,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
        model.to(device)
        model.eval()

        rob = run_robustness_eval(
            model=model,
            test_narratives=splits["test"]["narratives"],
            test_tabular=splits["test"]["tabular_features"],
            test_labels=splits["test"]["labels"].tolist(),
            class_names=adapter.class_names,
            tokenizer=tokenizer,
            is_text_only=(variant == "M1"),
            seed=seed,
        )
        robustness_results[variant] = rob

    with open(RESULTS_DIR / "robustness_results.json", "w") as f:
        json.dump(robustness_results, f, indent=2)
    print(f"\nRobustness results saved to {RESULTS_DIR / 'robustness_results.json'}")

    return robustness_results


def run_onnx_export(sample_size: int = 20_000, seed: int = 42):
    """Export best M2 model (highest test F1 across seeds) to ONNX and benchmark."""
    # Find best M2 across seeds by test_macro_f1 BEFORE building adapter
    results_path = RESULTS_DIR / "all_results.json"
    best_seed = seed
    if results_path.exists():
        with open(results_path) as f:
            all_results = json.load(f)
        m2_results = [r for r in all_results if r.get("variant") == "M2"]
        if m2_results:
            best = max(m2_results, key=lambda r: r["test_macro_f1"])
            best_seed = best["seed"]
            print(f"Best M2: seed={best_seed}, test_macro_f1={best['test_macro_f1']:.4f}")

    # Build adapter with the SAME seed used to train the best model
    adapter = CFPBAdapter(sample_size=sample_size, seed=best_seed)
    splits = adapter.preprocess()
    tabular_dim = splits["train"]["tabular_features"].shape[1]

    run_name = f"M2_seed{best_seed}"
    model_path = RESULTS_DIR / f"{run_name}_best.pt"
    if not model_path.exists():
        model_path = RESULTS_DIR / f"{run_name}.pt"
    if not model_path.exists():
        print(f"Skipping ONNX export: no saved model at {model_path}")
        return None

    print(f"\n{'=' * 60}")
    print(f"ONNX Export and Latency Benchmark (M2 seed={best_seed})")
    print("=" * 60)

    model = MultimodalClassifier(
        num_classes=len(adapter.class_names),
        tabular_input_dim=tabular_dim,
        modality_dropout=False,
        dropout=0.0,
    )
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    onnx_path = str(RESULTS_DIR / "model_m2.onnx")
    export_to_onnx(model, onnx_path, tabular_dim=tabular_dim)
    print(f"ONNX model exported to {onnx_path}")

    latency = benchmark_latency(model, onnx_path, tabular_dim=tabular_dim)
    print(f"Latency results: {latency}")

    with open(RESULTS_DIR / "latency_results.json", "w") as f:
        json.dump(latency, f, indent=2)

    return latency


def main():
    parser = argparse.ArgumentParser(description="Run finetune-bench ablation matrix")
    parser.add_argument(
        "--sample-size", type=int, default=20_000,
        help="Training data size (default: 20000, use 50000 for final runs)",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument(
        "--skip-robustness", action="store_true",
        help="Skip robustness evaluation",
    )
    parser.add_argument(
        "--skip-onnx", action="store_true",
        help="Skip ONNX export and latency benchmarking",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)

    print("=== BASELINES ===")
    b1, b2 = run_baselines(sample_size=args.sample_size)

    print("\n=== DL VARIANTS ===")
    dl_results = run_dl_variants(
        sample_size=args.sample_size, num_epochs=args.epochs,
    )

    all_results = [b1, b2] + dl_results
    with open(RESULTS_DIR / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to {RESULTS_DIR / 'all_results.json'}")

    if not args.skip_robustness:
        print("\n=== ROBUSTNESS EVALUATION ===")
        run_robustness(sample_size=args.sample_size)

    if not args.skip_onnx:
        print("\n=== ONNX EXPORT ===")
        run_onnx_export(sample_size=args.sample_size)


if __name__ == "__main__":
    main()
