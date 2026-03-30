"""Custom PyTorch training loop with MLflow tracking.

Core deliverable: explicit backward/step/clip mechanics, differential
learning rates, gradient accumulation, and early stopping.
"""

import json
import random as python_random
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizer, get_cosine_schedule_with_warmup

from evaluation.metrics import MetricsResult, compute_metrics
from models.fusion_model import MultimodalClassifier
from training.config import TrainConfig
from training.gpu_profiler import GPUProfiler


class ComplaintDataset(Dataset):
    """Dataset for complaint narratives + tabular features.

    Pre-tokenizes all narratives in __init__ to avoid redundant
    tokenization across epochs (~3x speedup on CPU).
    """

    def __init__(
        self,
        narratives: list[str],
        tabular_features: np.ndarray,
        labels: np.ndarray,
        tokenizer: DistilBertTokenizer,
        max_length: int = 128,
        text_only: bool = False,
    ):
        # Pre-tokenize all narratives once
        self.encodings = tokenizer(
            narratives,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        if text_only:
            self.tabular_features = torch.zeros(
                len(labels), tabular_features.shape[1], dtype=torch.float32,
            )
        else:
            self.tabular_features = torch.tensor(tabular_features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "text": {k: v[idx] for k, v in self.encodings.items()},
            "tabular": self.tabular_features[idx],
            "labels": self.labels[idx],
        }


class EarlyStopper:
    """Early stopping on validation metric (higher is better)."""

    def __init__(self, patience: int = 2):
        self.patience = patience
        self.best_score = -float("inf")
        self.counter = 0

    def should_stop(self, score: float) -> bool:
        if score > self.best_score:
            self.best_score = score
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


def compute_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights for imbalanced data."""
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    counts[counts == 0] = 1.0
    weights = len(labels) / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def _set_seeds(seed: int):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    python_random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model: MultimodalClassifier,
    train_loader: DataLoader,
    optimizer: AdamW,
    scheduler,
    class_weights: torch.Tensor,
    config: TrainConfig,
    epoch: int,
    device: torch.device,
    profiler: GPUProfiler | None = None,
) -> float:
    """Train for one epoch. Returns average loss."""
    model.train()
    epoch_loss = 0.0
    num_batches = 0

    for step, batch in enumerate(train_loader):
        text_inputs = {k: v.to(device) for k, v in batch["text"].items()}
        tabular_features = batch["tabular"].to(device)
        labels = batch["labels"].to(device)

        logits = model(text_inputs, tabular_features)
        loss = F.cross_entropy(logits, labels, weight=class_weights.to(device))
        loss = loss / config.grad_accumulation_steps

        loss.backward()

        if (step + 1) % config.grad_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=config.max_grad_norm
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        batch_loss = loss.item() * config.grad_accumulation_steps
        epoch_loss += batch_loss
        num_batches += 1

        global_step = epoch * len(train_loader) + step
        mlflow.log_metric("train_loss_step", batch_loss, step=global_step)

        if profiler is not None:
            profiler.on_step_end(epoch, step)

    # Flush any remaining accumulated gradients from incomplete final batch
    if (step + 1) % config.grad_accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=config.max_grad_norm
        )
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return epoch_loss / max(num_batches, 1)


def evaluate(
    model: MultimodalClassifier,
    loader: DataLoader,
    class_names: list[str],
    device: torch.device,
    return_probs: bool = False,
) -> MetricsResult | tuple[MetricsResult, np.ndarray]:
    """Evaluate model on a data loader.

    If return_probs=True, also returns (n_samples, n_classes) softmax array.
    """
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in loader:
            text_inputs = {k: v.to(device) for k, v in batch["text"].items()}
            tabular_features = batch["tabular"].to(device)
            logits = model(text_inputs, tabular_features)
            all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
            all_labels.extend(batch["labels"].tolist())
            if return_probs:
                probs = torch.nn.functional.softmax(logits, dim=-1)
                all_probs.extend(probs.cpu().numpy())

    metrics = compute_metrics(all_labels, all_preds, class_names)
    if return_probs:
        return metrics, np.array(all_probs)
    return metrics


def train(config: TrainConfig) -> dict:
    """Full training pipeline. Returns test metrics dict."""
    _set_seeds(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    from adapters.cfpb import CFPBAdapter

    adapter = CFPBAdapter(
        sample_size=config.sample_size,
        seed=config.seed,
        split_strategy=config.split_strategy,
        cutoff_date=config.cutoff_date,
        exclude_features=config.exclude_features,
    )
    splits = adapter.preprocess()

    tokenizer = DistilBertTokenizer.from_pretrained(config.text_model_name)

    text_only = config.variant == "M1"

    train_ds = ComplaintDataset(
        splits["train"]["narratives"],
        splits["train"]["tabular_features"],
        splits["train"]["labels"],
        tokenizer,
        config.max_seq_length,
        text_only=text_only,
    )
    val_ds = ComplaintDataset(
        splits["val"]["narratives"],
        splits["val"]["tabular_features"],
        splits["val"]["labels"],
        tokenizer,
        config.max_seq_length,
        text_only=text_only,
    )
    test_ds = ComplaintDataset(
        splits["test"]["narratives"],
        splits["test"]["tabular_features"],
        splits["test"]["labels"],
        tokenizer,
        config.max_seq_length,
        text_only=text_only,
    )

    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size, num_workers=0,
    )
    test_loader = DataLoader(
        test_ds, batch_size=config.batch_size, num_workers=0,
    )

    # Model
    tabular_dim = splits["train"]["tabular_features"].shape[1]

    model = MultimodalClassifier(
        num_classes=len(adapter.class_names),
        tabular_input_dim=tabular_dim,
        tabular_hidden_dim=config.tabular_hidden_dim,
        tabular_embed_dim=config.tabular_embed_dim,
        fusion_hidden_dim=config.fusion_hidden_dim,
        dropout=config.dropout,
        modality_dropout=(config.variant == "M3"),
        text_model_name=config.text_model_name,
        text_dropout_prob=config.text_dropout_prob,
        tabular_dropout_prob=config.tabular_dropout_prob,
    ).to(device)

    class_weights = compute_class_weights(
        splits["train"]["labels"], len(adapter.class_names)
    )

    # Differential learning rates
    encoder_params = list(model.text_encoder.parameters())
    head_params = (
        list(model.tabular_mlp.parameters())
        + list(model.fusion_head.parameters())
    )
    optimizer = AdamW([
        {"params": encoder_params, "lr": config.lr_encoder},
        {"params": head_params, "lr": config.lr_head},
    ])

    # Ceiling division: account for partial final accumulation batch
    # that train_one_epoch explicitly flushes
    steps_per_epoch = -(-len(train_loader) // config.grad_accumulation_steps)
    total_steps = steps_per_epoch * config.num_epochs
    warmup_steps = int(total_steps * config.warmup_fraction)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    early_stopper = EarlyStopper(patience=config.early_stopping_patience)
    profiler = GPUProfiler(enabled=torch.cuda.is_available())

    results_dir = Path(config.results_dir)
    results_dir.mkdir(exist_ok=True)

    run_name = config.run_name or f"{config.variant}_seed{config.seed}"

    mlflow.set_experiment(config.experiment_name)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(config.model_dump())

        best_val_f1 = 0.0
        for epoch in range(config.num_epochs):
            profiler.on_epoch_start(epoch)
            train_loss = train_one_epoch(
                model, train_loader, optimizer, scheduler,
                class_weights, config, epoch, device,
                profiler=profiler,
            )
            profiler.on_epoch_end(epoch)
            val_metrics = evaluate(model, val_loader, adapter.class_names, device)

            print(
                f"Epoch {epoch + 1}/{config.num_epochs} | "
                f"Loss: {train_loss:.4f} | "
                f"Val F1: {val_metrics.macro_f1:.4f} | "
                f"Val Acc: {val_metrics.accuracy:.4f}"
            )

            mlflow.log_metrics(
                {
                    "val_macro_f1": val_metrics.macro_f1,
                    "val_accuracy": val_metrics.accuracy,
                    "train_loss_epoch": train_loss,
                },
                step=epoch,
            )

            if val_metrics.macro_f1 > best_val_f1:
                best_val_f1 = val_metrics.macro_f1
                model_path = results_dir / f"{run_name}_best.pt"
                torch.save(model.state_dict(), model_path)

            if early_stopper.should_stop(val_metrics.macro_f1):
                print(f"Early stopping at epoch {epoch + 1}")
                mlflow.log_metric("stopped_epoch", epoch + 1)
                break

        # Load best model for test evaluation
        best_path = results_dir / f"{run_name}_best.pt"
        if best_path.exists():
            model.load_state_dict(torch.load(best_path, weights_only=True))

        test_metrics = evaluate(model, test_loader, adapter.class_names, device)
        print(f"\nTest Macro-F1: {test_metrics.macro_f1:.4f}")
        print(f"Test Accuracy: {test_metrics.accuracy:.4f}")
        print(f"\n{test_metrics.report}")

        mlflow.log_metrics({
            "test_macro_f1": test_metrics.macro_f1,
            "test_accuracy": test_metrics.accuracy,
        })

        # Log GPU profiling metrics
        gpu_summary = profiler.summary()
        for key, value in gpu_summary.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(f"gpu/{key}", value)

        final_path = results_dir / f"{run_name}.pt"
        torch.save(model.state_dict(), final_path)
        mlflow.log_artifact(str(final_path))

        class_names_path = results_dir / "class_names.json"
        with open(class_names_path, "w") as f:
            json.dump(adapter.class_names, f)

    return {
        "test_macro_f1": test_metrics.macro_f1,
        "test_accuracy": test_metrics.accuracy,
        "per_class_f1": test_metrics.per_class_f1.tolist(),
        "class_names": adapter.class_names,
        "variant": config.variant,
        "seed": config.seed,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="M2", choices=["M1", "M2", "M3"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-size", type=int, default=20000)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    cfg = TrainConfig(
        variant=args.variant,
        seed=args.seed,
        sample_size=args.sample_size,
        num_epochs=args.epochs,
    )
    train(cfg)
