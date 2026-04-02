"""Modal GPU app for vmap DP-SGD training and membership inference.

Uses manual DP-SGD via torch.func.vmap with per-group gradient clipping.

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

# Separate image with Opacus (for RDP accounting) — does not affect the main finetune-bench image.
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
        "peft>=0.7",
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
    {"name": "loose_dp", "epsilon": 50.0, "delta": 1e-5},
    {"name": "moderate_dp", "epsilon": 8.0, "delta": 1e-5},
    {"name": "strict_dp", "epsilon": 1.0, "delta": 1e-5},
    {"name": "strict_dp_tuned_clip", "epsilon": 1.0, "delta": 1e-5,
     "lora_clip": 0.05, "head_clip": 0.5},
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

    vmap DP-SGD needs flat tensor batches (not dicts). This pre-extracts
    input_ids, attention_mask, tabular, and labels into flat tensors.
    """
    import torch

    input_ids = dataset.encodings["input_ids"]
    attention_mask = dataset.encodings["attention_mask"]
    tabular = dataset.tabular_features
    labels = dataset.labels
    return torch.utils.data.TensorDataset(input_ids, attention_mask, tabular, labels)


@app.function(gpu="T4", timeout=86400, image=image, volumes={"/data": vol})
def train_dp_model(config: dict, seed: int) -> dict:
    """Train DistilBERT+LoRA with vmap DP-SGD and per-group clipping."""
    _setup_remote()

    import torch
    import torch.nn as nn
    from peft import LoraConfig, get_peft_model
    from transformers import DistilBertModel

    from adapters.cfpb import CFPBAdapter
    from models.fusion_model import MultimodalClassifier
    from privacy.vmap_dp import train_dp_vmap
    from training.config import TrainConfig
    from training.train import compute_class_weights

    adapter = CFPBAdapter(sample_size=20_000, seed=seed)
    splits = adapter.preprocess()
    train_config = TrainConfig(
        variant="M2", seed=seed, use_amp=False,
        run_name=f"dp_{config['name']}_seed{seed}",
    )

    train_ds, val_ds = _build_datasets(splits, train_config)
    num_classes = len(adapter.class_names)
    tabular_dim = splits["train"]["tabular_features"].shape[1]

    flat_train = _flatten_complaint_dataset(train_ds)
    flat_val = _flatten_complaint_dataset(val_ds)

    lora_rank = config.get("lora_rank", 8)

    # Build model: eager attention (required for vmap) + LoRA
    encoder = DistilBertModel.from_pretrained(
        train_config.text_model_name, attn_implementation="eager",
    )
    model = MultimodalClassifier(
        num_classes=num_classes,
        tabular_input_dim=tabular_dim,
        tabular_hidden_dim=train_config.tabular_hidden_dim,
        tabular_embed_dim=train_config.tabular_embed_dim,
        fusion_hidden_dim=train_config.fusion_hidden_dim,
        dropout=train_config.dropout,
        text_encoder=encoder,
    )
    lora_config = LoraConfig(
        r=lora_rank, lora_alpha=lora_rank * 2,
        lora_dropout=0.0,  # must be 0 for vmap (no data-dependent control flow)
        target_modules=["q_lin", "v_lin"], bias="none",
    )
    model.text_encoder = get_peft_model(model.text_encoder, lora_config)

    # Freeze base encoder, train LoRA + head
    for name, param in model.named_parameters():
        is_lora = "lora_" in name
        is_head = any(x in name for x in ["tabular_mlp", "fusion_head"])
        param.requires_grad = is_lora or is_head

    # Per-group clipping config
    lora_names = [n for n, p in model.named_parameters() if "lora_" in n and p.requires_grad]
    head_names = [n for n, p in model.named_parameters()
                  if any(x in n for x in ["tabular_mlp", "fusion_head"]) and p.requires_grad]
    groups = {
        "lora": {"params": lora_names, "clip_norm": config.get("lora_clip", 0.1)},
        "head": {"params": head_names, "clip_norm": config.get("head_clip", 1.0)},
    }

    # Class weights for imbalanced data
    all_labels = [int(flat_train[i][-1]) for i in range(len(flat_train))]
    class_weights = torch.tensor(
        [len(all_labels) / (num_classes * max(all_labels.count(c), 1)) for c in range(num_classes)],
        dtype=torch.float32,
    )

    # Loss function for vmap (per-sample signature)
    def loss_fn(trainable, frozen, bufs, input_ids, attention_mask, tabular, label):
        all_p = {**frozen, **trainable}
        out = torch.func.functional_call(
            model, (all_p, bufs), args=(),
            kwargs={
                "text_inputs": {
                    "input_ids": input_ids.unsqueeze(0),
                    "attention_mask": attention_mask.unsqueeze(0),
                },
                "tabular_features": tabular.unsqueeze(0),
            },
        )
        return torch.nn.functional.cross_entropy(
            out, label.unsqueeze(0), weight=class_weights.to(out.device),
        )

    # Predict function for evaluation (batch-level, dict inputs)
    def predict_fn(m, input_ids, attention_mask, tabular):
        return m(
            {"input_ids": input_ids, "attention_mask": attention_mask},
            tabular,
        )

    result = train_dp_vmap(
        model=model, loss_fn=loss_fn,
        train_dataset=flat_train, val_dataset=flat_val,
        groups=groups, num_classes=num_classes,
        epochs=config.get("epochs", 10),
        batch_size=config.get("batch_size", 16),
        lr=config.get("lr", 2e-5),
        epsilon=config["epsilon"], delta=config["delta"],
        device="cuda" if torch.cuda.is_available() else "cpu",
        seed=seed, class_weights=class_weights,
        predict_fn=predict_fn,
        optimizer_type=config.get("optimizer", "sgd"),
    )

    result["config_name"] = config["name"]
    result["seed"] = seed
    result["lora_rank"] = lora_rank

    # Save checkpoint for MIA
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
    lora_rank: int = 8,
) -> dict:
    """Run loss-threshold MIA on a trained model."""
    _setup_remote()

    import torch
    from peft import LoraConfig, get_peft_model

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

    # Rebuild model with LoRA to match checkpoint's state_dict keys.
    # vmap checkpoints are saved directly from MultimodalClassifier (no wrapper prefix).
    model = MultimodalClassifier(
        num_classes=num_classes,
        tabular_input_dim=tabular_dim,
        text_model_name=train_config.text_model_name,
    )
    lora_config = LoraConfig(
        r=lora_rank, lora_alpha=lora_rank * 2, lora_dropout=0.0,
        target_modules=["q_lin", "v_lin"], bias="none",
    )
    model.text_encoder = get_peft_model(model.text_encoder, lora_config)

    state_dict = torch.load(f"/data/{checkpoint_path}", map_location="cpu")
    model.load_state_dict(state_dict)
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


@app.function(gpu="T4", timeout=3600, image=image, volumes={"/data": vol})
def train_lora_baseline(seed: int) -> dict:
    """Train LoRA model WITHOUT DP-SGD to establish the LoRA utility ceiling."""
    _setup_remote()

    import torch
    import torch.nn as nn
    from peft import LoraConfig, get_peft_model

    from adapters.cfpb import CFPBAdapter
    from evaluation.metrics import compute_metrics
    from models.fusion_model import MultimodalClassifier
    from training.config import TrainConfig
    from training.train import compute_class_weights

    adapter = CFPBAdapter(sample_size=20_000, seed=seed)
    splits = adapter.preprocess()
    train_config = TrainConfig(variant="M2", seed=seed, num_epochs=3)
    train_ds, val_ds = _build_datasets(splits, train_config)
    num_classes = len(adapter.class_names)
    tabular_dim = splits["train"]["tabular_features"].shape[1]

    flat_train = _flatten_complaint_dataset(train_ds)
    flat_val = _flatten_complaint_dataset(val_ds)

    model = MultimodalClassifier(
        num_classes=num_classes, tabular_input_dim=tabular_dim,
        text_model_name=train_config.text_model_name,
    )
    lora_config = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.1,
        target_modules=["q_lin", "v_lin"], bias="none",
    )
    model.text_encoder = get_peft_model(model.text_encoder, lora_config)

    for name, param in model.named_parameters():
        is_lora = "lora_" in name
        is_head = any(x in name for x in ["tabular_mlp", "fusion_head"])
        param.requires_grad = is_lora or is_head

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=2e-5,
    )
    train_loader = torch.utils.data.DataLoader(flat_train, batch_size=16, shuffle=True)
    val_loader = torch.utils.data.DataLoader(flat_val, batch_size=32)

    all_labels = [int(flat_train[i][-1]) for i in range(len(flat_train))]
    class_weights = compute_class_weights(all_labels, num_classes).to(device)

    for _epoch in range(train_config.num_epochs):
        model.train()
        for batch in train_loader:
            ids, mask, tab, labels = [t.to(device) for t in batch]
            logits = model({"input_ids": ids, "attention_mask": mask}, tab)
            loss = nn.functional.cross_entropy(logits, labels, weight=class_weights)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            ids, mask, tab, labels = [t.to(device) for t in batch]
            logits = model({"input_ids": ids, "attention_mask": mask}, tab)
            all_preds.extend(logits.argmax(dim=1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    class_names = [str(i) for i in range(num_classes)]
    metrics = compute_metrics(all_labels, all_preds, class_names)

    # Save warm-start checkpoint to volume for DP stage 2
    checkpoint_name = f"lora_warmstart_seed{seed}.pt"
    torch.save(model.state_dict(), f"/data/{checkpoint_name}")
    vol.commit()
    print(f"Warm-start checkpoint saved: {checkpoint_name}")

    return {
        "config_name": "lora_baseline",
        "seed": seed,
        "val_macro_f1": metrics.macro_f1,
        "epsilon_target": float("inf"),
        "epsilon_actual": float("inf"),
        "lora_rank": 8,
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "checkpoint_path": checkpoint_name,
    }


@app.local_entrypoint()
def main(
    dp_train: bool = False,
    mia: bool = False,
    all: bool = False,
    test: bool = False,
    baseline_mia: bool = False,
    lora_baseline: bool = False,
    diag: bool = False,
):
    """CLI entrypoint for Modal privacy experiments."""
    artifacts = Path(_REPO_ROOT) / "artifacts"
    artifacts.mkdir(exist_ok=True)

    if test:
        print("Running 3 diagnostic configs in parallel (seed=42, T4)...")
        _prewarm_data.remote()
        diag_configs = [
            # Round 3: find the ceiling and test the curve
            # Stop criterion: F1 >= 0.30 at ε=50/10ep, else Path D
            ({"name": "e50_adam_10ep", "epsilon": 50.0, "delta": 1e-5,
              "epochs": 10, "optimizer": "adam", "lr": 1e-3}, 42),
            ({"name": "e50_adam_20ep", "epsilon": 50.0, "delta": 1e-5,
              "epochs": 20, "optimizer": "adam", "lr": 1e-3}, 42),
            ({"name": "e8_adam_10ep", "epsilon": 8.0, "delta": 1e-5,
              "epochs": 10, "optimizer": "adam", "lr": 1e-3}, 42),
        ]
        results = list(train_dp_model.starmap(diag_configs))
        print("\n=== DIAGNOSTIC RESULTS ===")
        for r in results:
            print("{:<25} F1={:.4f}  acc={:.4f}  loss={:.4f}  eps={:.2f}  per_class_f1={}".format(
                r["config_name"], r["val_macro_f1"], r["val_accuracy"],
                r["train_loss"], r["epsilon_actual"],
                [round(x, 3) for x in r["per_class_f1"]],
            ))
        return

    if diag:
        print("Running 4 diagnostic configs in parallel (seed=42, T4)...")
        _prewarm_data.remote()
        # Stage 1: warm-start (non-DP LoRA+head training, saves checkpoint)
        print("Stage 1: warm-start training...")
        ws_result = train_lora_baseline.remote(42)
        ws_checkpoint = ws_result["checkpoint_path"]
        print(f"  Warm-start F1={ws_result['val_macro_f1']:.4f}, checkpoint={ws_checkpoint}")

        # Critical test: does LoRA contribute, or is the frozen head doing everything?
        # "reset_lora" loads warm-start but re-randomizes LoRA weights before DP training
        diag_configs = [
            ({"name": "diag_ws_e8", "epsilon": 8.0, "delta": 1e-5,
              "max_grad_norm": 1.0, "lora_rank": 8, "batch_size": 128, "epochs": 3,
              "warmstart": ws_checkpoint}, 42),
            ({"name": "diag_ws_e8_reset_lora", "epsilon": 8.0, "delta": 1e-5,
              "max_grad_norm": 1.0, "lora_rank": 8, "batch_size": 128, "epochs": 3,
              "warmstart": ws_checkpoint, "reset_lora": True}, 42),
            ({"name": "diag_head_only_no_dp", "epsilon": 10000.0, "delta": 1e-5,
              "max_grad_norm": 1.0, "lora_rank": 8, "batch_size": 128, "epochs": 1,
              "warmstart": ws_checkpoint, "reset_lora": True}, 42),
        ]
        results = list(train_dp_model.starmap(diag_configs))
        print("\n=== DIAGNOSTIC RESULTS ===")
        for r in results:
            print("{:<30} F1={:.4f}  acc={:.4f}  loss={:.4f}  eps={:.2f}  warmstart={}".format(
                r["config_name"], r["val_macro_f1"], r["val_accuracy"],
                r["train_loss"], r["epsilon_actual"], r.get("warmstart", "none"),
            ))
        return

    if lora_baseline:
        print("Training non-DP LoRA baselines (3 seeds)...")
        _prewarm_data.remote()
        results = list(train_lora_baseline.starmap([(s,) for s in SEEDS]))
        import numpy as np
        f1s = [r["val_macro_f1"] for r in results]
        summary = {
            "config": "lora_baseline",
            "epsilon_target": "inf",
            "epsilon_actual": "inf",
            "val_macro_f1": round(float(np.mean(f1s)), 4),
            "val_macro_f1_std": round(float(np.std(f1s)), 4),
            "lora_rank": 8,
            "seeds": SEEDS,
        }
        print(json.dumps(summary, indent=2))

        # Merge into dp_results.json as the first entry
        dp_path = artifacts / "dp_results.json"
        if dp_path.exists():
            dp_data = json.loads(dp_path.read_text())
            dp_data["results"] = [
                r for r in dp_data["results"] if r["config"] != "lora_baseline"
            ]
            dp_data["results"].insert(0, summary)
        else:
            dp_data = {"results": [summary]}
        dp_path.write_text(json.dumps(dp_data, indent=2))
        print(f"LoRA baseline merged into {dp_path}")
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
        # Single-stage DP training: LoRA + head from scratch, encoder frozen.
        configs = [
            ({**config, "epochs": config.get("epochs", 10)}, seed)
            for config in DP_CONFIGS
            for seed in SEEDS
        ]

        print(f"Dispatching {len(configs)} DP training runs in 2 batches...")
        mid = len(configs) // 2
        print(f"  Batch 1: {mid} runs")
        results = list(train_dp_model.starmap(configs[:mid]))
        print(f"  Batch 2: {len(configs) - mid} runs")
        results.extend(train_dp_model.starmap(configs[mid:]))

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

        mia_args = build_mia_args(dp_data)
        mia_results = list(run_membership_inference_attack.starmap(mia_args))

        mia_output = {"results": mia_results}
        mia_path = artifacts / "mia_results.json"
        mia_path.write_text(json.dumps(mia_output, indent=2))
        print(f"MIA results written to {mia_path}")


def build_mia_args(dp_data: dict) -> list[tuple]:
    """Build MIA attack args from dp_results, skipping entries without checkpoints."""
    mia_args = []
    for r in dp_data["results"]:
        config_name = r["config"]
        if config_name == "lora_baseline":
            continue  # no checkpoint saved for non-DP baseline
        checkpoint = f"M2_dp_{config_name}_seed42_best.pt"
        epsilon = r["epsilon_target"]
        lora_rank = r.get("lora_rank", 8)
        mia_args.append((checkpoint, str(epsilon), config_name, lora_rank))
    return mia_args


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
            "lora_rank": runs[0].get("lora_rank", 8),
            "seeds": [r["seed"] for r in runs],
        })

    return aggregated
