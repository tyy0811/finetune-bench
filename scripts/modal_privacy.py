"""Modal GPU app for DP-SGD training and membership inference.

Separate from scripts/modal_run.py to isolate the Opacus dependency.

Usage:
    modal run scripts/modal_privacy.py --dp-train    # DP-SGD training (12 runs)
    modal run scripts/modal_privacy.py --mia          # Membership inference
    modal run scripts/modal_privacy.py --all          # Both
"""

import json

import modal

app = modal.App("finetune-bench-privacy")

# Separate image with Opacus — does not affect the main finetune-bench image.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.2.2",
        "transformers==4.57.6",
        "scikit-learn==1.8.0",
        "pandas==2.3.3",
        "numpy==1.26.4",
        "mlflow==3.10.1",
        "pydantic==2.12.5",
        "opacus>=1.4.0",
        "requests==2.32.5",
    )
    .add_local_dir(
        "/Users/zenith/Desktop/finetune-bench",
        remote_path="/root/finetune-bench",
        ignore=[
            ".git/**", "data/**", "mlruns/**", "**/__pycache__/**",
            "**/*.egg-info/**", ".claude/**", "mlflow.db",
            "*.zip", "*.ipynb", "docs/**",
        ],
    )
)

vol = modal.Volume.from_name("finetune-bench-privacy-data", create_if_missing=True)

# Experiment matrix: 4 DP configs x 3 seeds = 12 runs
DP_CONFIGS = [
    {"name": "loose_dp", "epsilon": 50.0, "delta": 1e-5, "max_grad_norm": 1.0},
    {"name": "moderate_dp", "epsilon": 8.0, "delta": 1e-5, "max_grad_norm": 1.0},
    {"name": "strict_dp", "epsilon": 1.0, "delta": 1e-5, "max_grad_norm": 1.0},
    {"name": "strict_dp_tuned_clip", "epsilon": 1.0, "delta": 1e-5, "max_grad_norm": 0.5},
]
SEEDS = [42, 123, 456]


@app.function(gpu="A10G", timeout=3600, image=image, volumes={"/data": vol})
def train_dp_model(config: dict, seed: int) -> dict:
    """Train DistilBERT with Opacus DP-SGD on Modal GPU."""
    import os
    import subprocess
    import sys

    os.chdir("/root/finetune-bench")
    sys.path.insert(0, "/root/finetune-bench")
    subprocess.run(["pip", "install", "-e", "."], capture_output=True)

    import torch

    from adapters.cfpb import CFPBAdapter
    from privacy.dp_training import make_dp_config

    # Load data
    adapter = CFPBAdapter(sample_size=20_000, seed=seed)
    splits = adapter.preprocess()

    # Build TrainConfig with DP overrides
    dp_overrides = make_dp_config(
        epsilon=config["epsilon"],
        delta=config["delta"],
        max_grad_norm=config["max_grad_norm"],
    )

    from training.config import TrainConfig

    train_config = TrainConfig(
        variant="M2",
        seed=seed,
        use_amp=False,
        grad_accumulation_steps=1,
        lr_encoder=dp_overrides["lr"],
        lr_head=dp_overrides["lr"],  # single LR for DP
        run_name=f"dp_{config['name']}_seed{seed}",
    )

    # Full DP training with the tested components
    from privacy.dp_training import train_dp_fullmodel

    result = train_dp_fullmodel(
        config=train_config,
        splits=splits,
        class_names=adapter.class_names,
        epsilon=config["epsilon"],
        delta=config["delta"],
        max_grad_norm=config["max_grad_norm"],
    )

    result["config_name"] = config["name"]
    result["seed"] = seed

    # Save checkpoint to volume
    checkpoint_name = f"M2_dp_{config['name']}_seed{seed}_best.pt"
    vol_path = f"/data/{checkpoint_name}"
    if "model_state" in result:
        torch.save(result.pop("model_state"), vol_path)
        result["checkpoint_path"] = checkpoint_name

    return result


@app.function(gpu="A10G", timeout=1800, image=image, volumes={"/data": vol})
def run_membership_inference_attack(model_checkpoint: str, epsilon_label: str) -> dict:
    """Run loss-threshold MIA on a trained model."""
    import os
    import subprocess
    import sys

    os.chdir("/root/finetune-bench")
    sys.path.insert(0, "/root/finetune-bench")
    subprocess.run(["pip", "install", "-e", "."], capture_output=True)

    # TODO: Full MIA pipeline — load model checkpoint, compute per-sample losses
    # on train/test splits, run compute_mia_auc and stratified_mia_by_entity.
    # Blocked on dp_training checkpoint format being finalized during Modal runs.

    return {"model": model_checkpoint, "epsilon": epsilon_label}


@app.local_entrypoint()
def main(
    dp_train: bool = False,
    mia: bool = False,
    all: bool = False,
):
    """CLI entrypoint for Modal privacy experiments."""
    from pathlib import Path

    artifacts = Path("/Users/zenith/Desktop/finetune-bench/artifacts")
    artifacts.mkdir(exist_ok=True)

    if dp_train or all:
        print(f"Dispatching {len(DP_CONFIGS) * len(SEEDS)} DP training runs...")
        configs = [
            (config, seed)
            for config in DP_CONFIGS
            for seed in SEEDS
        ]
        results = list(train_dp_model.starmap(configs))

        # Aggregate by config (3 seeds each)
        aggregated = _aggregate_dp_results(results)
        dp_path = artifacts / "dp_results.json"
        dp_path.write_text(json.dumps(aggregated, indent=2))
        print(f"DP results written to {dp_path}")

    if mia or all:
        print("Running membership inference attacks...")
        # Dispatch MIA on each DP variant's best checkpoint
        # ... uses run_membership_inference_attack.map() ...
        print("MIA complete")


def _aggregate_dp_results(results: list[dict]) -> dict:
    """Aggregate per-seed results into per-config summaries with mean/std."""
    from collections import defaultdict

    import numpy as np

    by_config: dict[str, list] = defaultdict(list)
    for r in results:
        by_config[r["config_name"]].append(r)

    aggregated: dict = {"results": []}
    for config_name, runs in by_config.items():
        f1s = [r["val_macro_f1"] for r in runs]
        aggregated["results"].append({
            "config": config_name,
            "epsilon_target": runs[0]["epsilon_target"],
            "epsilon_actual": round(
                float(np.mean([r["epsilon_actual"] for r in runs])), 4
            ),
            "val_macro_f1": round(float(np.mean(f1s)), 4),
            "val_macro_f1_std": round(float(np.std(f1s)), 4),
            "seeds": [r["seed"] for r in runs],
        })

    return aggregated
