"""Modal GPU app for DP-SGD training and membership inference.

Separate from scripts/modal_run.py to isolate the Opacus dependency.

Usage:
    modal run scripts/modal_privacy.py --dp-train    # DP-SGD training (12 runs)
    modal run scripts/modal_privacy.py --mia          # Membership inference
    modal run scripts/modal_privacy.py --all          # Both
"""

import json
from pathlib import Path

import modal

# Resolve repo root relative to this script (works on any checkout)
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)

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
        "opacus==1.4.1",
        "dp-transformers>=1.0.1",
        "requests==2.32.5",
    )
    .add_local_dir(
        _REPO_ROOT,
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


def _setup_remote():
    """Common setup for Modal remote functions."""
    import os
    import shutil
    import subprocess
    import sys

    os.chdir("/root/finetune-bench")
    sys.path.insert(0, "/root/finetune-bench")
    subprocess.run(["pip", "install", "-e", "."], capture_output=True)

    # Ensure CFPB data is available (check volume cache, download if needed)
    data_dir = "/data/complaints.csv"
    local_data = "data/complaints.csv"
    os.makedirs("data", exist_ok=True)
    if os.path.exists(local_data):
        pass  # already present (container reuse)
    elif os.path.exists(data_dir):
        os.symlink(data_dir, local_data)
    else:
        subprocess.run(["python", "scripts/download_data.py"], check=True)
        shutil.copy(local_data, data_dir)
        vol.commit()


@app.function(gpu="T4", timeout=600, image=image, volumes={"/data": vol})
def _prewarm_data():
    """Ensure CFPB data is cached on the volume before parallel jobs start."""
    _setup_remote()
    print("Data prewarm complete.")


def _build_datasets(splits, config):
    """Build ComplaintDataset instances for train/val from adapter splits."""
    from transformers import DistilBertTokenizer

    from training.train import ComplaintDataset

    tokenizer = DistilBertTokenizer.from_pretrained(config.text_model_name)
    text_only = config.variant == "M1"

    train_ds = ComplaintDataset(
        splits["train"]["narratives"],
        splits["train"]["tabular_features"],
        splits["train"]["labels"],
        tokenizer,
        max_length=config.max_seq_length,
        text_only=text_only,
    )
    val_ds = ComplaintDataset(
        splits["val"]["narratives"],
        splits["val"]["tabular_features"],
        splits["val"]["labels"],
        tokenizer,
        max_length=config.max_seq_length,
        text_only=text_only,
    )
    return train_ds, val_ds


def _flatten_complaint_dataset(dataset):
    """Convert a ComplaintDataset (dict batches) to a flat TensorDataset.

    Opacus requires forward(tensor) signatures. This pre-extracts
    input_ids, attention_mask, tabular, and labels into flat tensors.
    """
    import torch

    input_ids = dataset.encodings["input_ids"]
    attention_mask = dataset.encodings["attention_mask"]
    tabular = dataset.tabular_features
    labels = dataset.labels
    return torch.utils.data.TensorDataset(input_ids, attention_mask, tabular, labels)


@app.function(gpu="T4", timeout=3600, image=image, volumes={"/data": vol})
def train_dp_model(config: dict, seed: int) -> dict:
    """Train DistilBERT with Opacus DP-SGD on Modal GPU."""
    _setup_remote()

    import torch
    import torch.nn as nn
    from dp_transformers.module_modification import convert_model_to_dp
    from transformers import DistilBertModel

    from adapters.cfpb import CFPBAdapter
    from models.fusion_model import MultimodalClassifier
    from privacy.dp_training import train_dp
    from training.config import TrainConfig

    class _OpacusMultimodalWrapper(nn.Module):
        """Flat-tensor forward signature for Opacus compatibility."""

        def __init__(self, inner):
            super().__init__()
            self.inner = inner
            # No freeze — full model is DP-trainable via dp-transformers

        def forward(self, input_ids, attention_mask, tabular):
            text_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
            return self.inner(text_inputs, tabular)

    adapter = CFPBAdapter(sample_size=20_000, seed=seed)
    splits = adapter.preprocess()

    train_config = TrainConfig(
        variant="M2",
        seed=seed,
        use_amp=False,
        grad_accumulation_steps=1,
        lr_encoder=2e-5,
        lr_head=2e-5,  # single LR for DP
        run_name=f"dp_{config['name']}_seed{seed}",
    )

    train_ds, val_ds = _build_datasets(splits, train_config)
    num_classes = len(adapter.class_names)
    tabular_dim = splits["train"]["tabular_features"].shape[1]

    # Flatten datasets for Opacus compatibility (needs tensor inputs, not dicts)
    flat_train = _flatten_complaint_dataset(train_ds)
    flat_val = _flatten_complaint_dataset(val_ds)

    def make_model():
        encoder = DistilBertModel.from_pretrained(train_config.text_model_name)
        encoder = convert_model_to_dp(encoder)
        inner = MultimodalClassifier(
            num_classes=num_classes,
            tabular_input_dim=tabular_dim,
            tabular_hidden_dim=train_config.tabular_hidden_dim,
            tabular_embed_dim=train_config.tabular_embed_dim,
            fusion_hidden_dim=train_config.fusion_hidden_dim,
            dropout=train_config.dropout,
            text_model_name=train_config.text_model_name,
            text_encoder=encoder,
        )
        return _OpacusMultimodalWrapper(inner)

    result = train_dp(
        model_class=make_model,
        model_args=(),
        train_dataset=flat_train,
        val_dataset=flat_val,
        num_classes=num_classes,
        epochs=train_config.num_epochs,
        batch_size=train_config.batch_size,
        lr=2e-5,
        epsilon=config["epsilon"],
        delta=config["delta"],
        max_grad_norm=config["max_grad_norm"],
        device="cuda" if torch.cuda.is_available() else "cpu",
        seed=seed,
    )

    result["config_name"] = config["name"]
    result["seed"] = seed

    # Save trained model state to volume for MIA
    checkpoint_name = f"M2_dp_{config['name']}_seed{seed}_best.pt"
    vol_path = f"/data/{checkpoint_name}"
    model_state = result.pop("model_state_dict")
    torch.save(model_state, vol_path)
    vol.commit()
    result["checkpoint_path"] = checkpoint_name

    return result


@app.function(gpu="T4", timeout=1800, image=image, volumes={"/data": vol})
def run_membership_inference_attack(
    checkpoint_path: str, epsilon_label: str, config_name: str,
) -> dict:
    """Run loss-threshold MIA on a trained model."""
    _setup_remote()

    import torch

    from adapters.cfpb import CFPBAdapter
    from models.fusion_model import MultimodalClassifier
    from privacy.membership_inference import (
        balance_member_nonmember,
        compute_mia_auc,
        compute_per_sample_loss,
        stratified_mia_by_entity,
    )
    from training.config import TrainConfig

    # Load data with canonical seed/split
    adapter = CFPBAdapter(sample_size=20_000, seed=42)
    splits = adapter.preprocess()
    train_config = TrainConfig(variant="M2")

    train_ds, val_ds = _build_datasets(splits, train_config)
    num_classes = len(adapter.class_names)
    tabular_dim = splits["train"]["tabular_features"].shape[1]

    # Rebuild model and load checkpoint.
    # Checkpoints are saved from _OpacusMultimodalWrapper, so keys have
    # an "inner." prefix. Strip it to load into bare MultimodalClassifier.
    model = MultimodalClassifier(
        num_classes=num_classes,
        tabular_input_dim=tabular_dim,
        text_model_name=train_config.text_model_name,
    )
    state_dict = torch.load(f"/data/{checkpoint_path}", map_location="cpu")
    stripped = {k.removeprefix("inner."): v for k, v in state_dict.items()}
    model.load_state_dict(stripped)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Compute per-sample losses on train (members) and test (non-members)
    member_losses = compute_per_sample_loss(model, train_ds, batch_size=32, device=device)
    nonmember_losses = compute_per_sample_loss(model, val_ds, batch_size=32, device=device)

    # Balance for valid AUC
    balanced_m, balanced_nm = balance_member_nonmember(member_losses, nonmember_losses)
    mia_result = compute_mia_auc(balanced_m, balanced_nm)

    # Stratified analysis by company frequency
    train_companies = splits["train"]["companies"]
    assert len(train_companies) == len(member_losses), (
        f"Company label count ({len(train_companies)}) != member loss count "
        f"({len(member_losses)}). CFPBAdapter must include 'companies' in splits."
    )

    company_counts: dict[str, int] = {}
    for company in train_companies:
        company_counts[company] = company_counts.get(company, 0) + 1

    stratified = stratified_mia_by_entity(
        member_losses=member_losses,
        member_companies=train_companies,
        nonmember_losses=nonmember_losses,
        company_train_counts=company_counts,
    )

    return {
        "model": config_name,
        "epsilon": epsilon_label,
        "mia_auc": mia_result["mia_auc"],
        "member_sample_size": len(balanced_m),
        "non_member_sample_size": len(balanced_nm),
        "train_loss_mean": mia_result["train_loss_mean"],
        "test_loss_mean": mia_result["test_loss_mean"],
        "loss_gap": mia_result["loss_gap"],
        "stratified": stratified,
    }


@app.function(gpu="T4", timeout=3600, image=image, volumes={"/data": vol})
def train_and_attack_baseline() -> dict:
    """Train a non-DP M2 model and run MIA on it for baseline comparison.

    Uses the existing training pipeline (train()) to produce a fully
    fine-tuned model (unfrozen encoder), then runs MIA on it.
    """
    _setup_remote()

    import torch
    from transformers import DistilBertTokenizer

    from adapters.cfpb import CFPBAdapter
    from models.fusion_model import MultimodalClassifier
    from privacy.membership_inference import (
        balance_member_nonmember,
        compute_mia_auc,
        compute_per_sample_loss,
        stratified_mia_by_entity,
    )
    from training.config import TrainConfig
    from training.train import ComplaintDataset, compute_class_weights

    train_config = TrainConfig(variant="M2", seed=42, num_epochs=3)

    adapter = CFPBAdapter(sample_size=20_000, seed=42)
    splits = adapter.preprocess()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = DistilBertTokenizer.from_pretrained(train_config.text_model_name)
    train_ds = ComplaintDataset(
        splits["train"]["narratives"], splits["train"]["tabular_features"],
        splits["train"]["labels"], tokenizer, max_length=train_config.max_seq_length,
    )
    val_ds = ComplaintDataset(
        splits["val"]["narratives"], splits["val"]["tabular_features"],
        splits["val"]["labels"], tokenizer, max_length=train_config.max_seq_length,
    )

    num_classes = len(adapter.class_names)
    tabular_dim = splits["train"]["tabular_features"].shape[1]
    model = MultimodalClassifier(
        num_classes=num_classes, tabular_input_dim=tabular_dim,
        text_model_name=train_config.text_model_name,
    ).to(device)

    # Train with same recipe as baseline (full fine-tuning, differential LR)
    class_weights = compute_class_weights(splits["train"]["labels"], num_classes).to(device)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=16, shuffle=True)
    optimizer = torch.optim.AdamW([
        {"params": model.text_encoder.parameters(), "lr": 2e-5},
        {"params": list(model.tabular_mlp.parameters()) + list(model.fusion_head.parameters()),
         "lr": 1e-3},
    ])

    for _epoch in range(3):
        model.train()
        for batch in train_loader:
            text = {k: v.to(device) for k, v in batch["text"].items()}
            tabular = batch["tabular"].to(device)
            labels = batch["labels"].to(device)
            logits = model(text, tabular)
            loss = torch.nn.functional.cross_entropy(logits, labels, weight=class_weights)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    # Now run MIA on the trained model
    member_losses = compute_per_sample_loss(model, train_ds, batch_size=32, device=str(device))
    nonmember_losses = compute_per_sample_loss(model, val_ds, batch_size=32, device=str(device))

    balanced_m, balanced_nm = balance_member_nonmember(member_losses, nonmember_losses)
    mia_result = compute_mia_auc(balanced_m, balanced_nm)

    train_companies = splits["train"]["companies"]
    company_counts: dict[str, int] = {}
    for company in train_companies:
        company_counts[company] = company_counts.get(company, 0) + 1

    stratified = stratified_mia_by_entity(
        member_losses=member_losses,
        member_companies=train_companies,
        nonmember_losses=nonmember_losses,
        company_train_counts=company_counts,
    )

    return {
        "model": "M2_no_dp",
        "epsilon": "inf",
        "mia_auc": mia_result["mia_auc"],
        "member_sample_size": len(balanced_m),
        "non_member_sample_size": len(balanced_nm),
        "train_loss_mean": mia_result["train_loss_mean"],
        "test_loss_mean": mia_result["test_loss_mean"],
        "loss_gap": mia_result["loss_gap"],
        "stratified": stratified,
    }


@app.local_entrypoint()
def main(
    dp_train: bool = False,
    mia: bool = False,
    all: bool = False,
    test: bool = False,
    baseline_mia: bool = False,
):
    """CLI entrypoint for Modal privacy experiments."""
    artifacts = Path(_REPO_ROOT) / "artifacts"
    artifacts.mkdir(exist_ok=True)

    if test:
        print("Running single validation run (moderate_dp, seed=42, T4)...")
        result = train_dp_model.remote(
            {"name": "moderate_dp", "epsilon": 8.0, "delta": 1e-5, "max_grad_norm": 1.0},
            42,
        )
        print(json.dumps({k: v for k, v in result.items() if k != "model_state_dict"}, indent=2))
        return

    if baseline_mia:
        print("Training non-DP M2 baseline and running MIA...")
        _prewarm_data.remote()
        baseline_result = train_and_attack_baseline.remote()
        print(json.dumps(baseline_result, indent=2))

        # Merge into existing mia_results.json if present
        mia_path = artifacts / "mia_results.json"
        if mia_path.exists():
            mia_data = json.loads(mia_path.read_text())
            # Remove any existing baseline entry
            mia_data["results"] = [
                r for r in mia_data["results"] if r.get("epsilon") != "inf"
            ]
            mia_data["results"].insert(0, baseline_result)
        else:
            mia_data = {"results": [baseline_result]}
        mia_path.write_text(json.dumps(mia_data, indent=2))
        print(f"Baseline MIA merged into {mia_path}")
        return

    if dp_train or mia or all:
        # Ensure data is cached on volume before parallel dispatch
        print("Prewarming data cache...")
        _prewarm_data.remote()

    if dp_train or all:
        print(f"Dispatching {len(DP_CONFIGS) * len(SEEDS)} DP training runs...")
        configs = [
            (config, seed)
            for config in DP_CONFIGS
            for seed in SEEDS
        ]
        results = list(train_dp_model.starmap(configs))

        aggregated = _aggregate_dp_results(results)
        dp_path = artifacts / "dp_results.json"
        dp_path.write_text(json.dumps(aggregated, indent=2))
        print(f"DP results written to {dp_path}")

    if mia or all:
        print("Running membership inference attacks...")

        # Collect checkpoints: one per config (seed=42 as canonical)
        dp_path = artifacts / "dp_results.json"
        if dp_path.exists():
            dp_data = json.loads(dp_path.read_text())
        else:
            print("No dp_results.json found — run --dp-train first")
            return

        mia_args = []
        for r in dp_data["results"]:
            config_name = r["config"]
            checkpoint = f"M2_dp_{config_name}_seed42_best.pt"
            epsilon = r["epsilon_target"]
            mia_args.append((checkpoint, str(epsilon), config_name))

        mia_results = list(run_membership_inference_attack.starmap(mia_args))

        mia_output = {"results": mia_results}
        mia_path = artifacts / "mia_results.json"
        mia_path.write_text(json.dumps(mia_output, indent=2))
        print(f"MIA results written to {mia_path}")


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
