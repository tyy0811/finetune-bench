"""Differential privacy training via Opacus DP-SGD."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset


def make_dp_config(
    epsilon: float,
    delta: float,
    max_grad_norm: float,
    lr: float = 2e-5,
) -> dict[str, Any]:
    """Build config overrides for DP training.

    Disables AMP and gradient accumulation to ensure correct privacy accounting.
    Uses a single learning rate (Opacus may flatten param groups).
    """
    return {
        "use_amp": False,
        "grad_accumulation_steps": 1,
        "lr": lr,
        "epsilon": epsilon,
        "delta": delta,
        "max_grad_norm": max_grad_norm,
    }


def create_dp_training_components(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: DataLoader,
    epochs: int,
    epsilon: float,
    delta: float,
    max_grad_norm: float,
) -> tuple[torch.nn.Module, torch.optim.Optimizer, DataLoader, object]:
    """Wrap model, optimizer, and dataloader with Opacus DP-SGD.

    Returns (model, optimizer, dataloader, privacy_engine).
    Opacus replaces the model with a GradSampleModule (per-sample gradients),
    wraps the optimizer to clip + add noise, and wraps the DataLoader
    for Poisson sampling (required for privacy accounting).
    """
    from opacus import PrivacyEngine

    privacy_engine = PrivacyEngine()
    dp_model, dp_optimizer, dp_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=data_loader,
        epochs=epochs,
        target_epsilon=epsilon,
        target_delta=delta,
        max_grad_norm=max_grad_norm,
    )
    return dp_model, dp_optimizer, dp_loader, privacy_engine


def _collect_gradient_norms(model: torch.nn.Module) -> dict[str, float]:
    """Collect per-layer gradient norms after backward, before optimizer step.

    Attempts to read .grad_sample (Opacus per-sample gradients).
    Falls back to .grad (post-clip) if .grad_sample is unavailable.
    """
    norms = {}
    # Try the underlying module if wrapped by GradSampleModule
    target = getattr(model, "_module", model)
    for name, param in target.named_parameters():
        if hasattr(param, "grad_sample") and param.grad_sample is not None:
            # Per-sample gradient: take norm per sample, then mean
            per_sample_norms = param.grad_sample.flatten(1).norm(2, dim=1)
            norms[name] = per_sample_norms.mean().item()
        elif param.grad is not None:
            norms[name] = param.grad.flatten().norm(2).item()
    return norms


def _forward_batch(
    model: torch.nn.Module, batch: tuple | dict, device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unpack a batch and run the model forward pass.

    Supports two batch formats:
    - Tuple/list: (inputs, labels) — used by TensorDataset
    - Dict with 'text', 'tabular', 'labels' keys — used by ComplaintDataset
    """
    if isinstance(batch, dict):
        text_inputs = {k: v.to(device) for k, v in batch["text"].items()}
        tabular = batch["tabular"].to(device)
        labels = batch["labels"].to(device)
        logits = model(text_inputs, tabular)
    else:
        inputs, labels = batch[0].to(device), batch[1].to(device)
        logits = model(inputs)
    return logits, labels


def train_dp(
    model_class: type,
    model_args: tuple,
    train_dataset: Dataset,
    val_dataset: Dataset,
    num_classes: int,
    epochs: int,
    batch_size: int,
    lr: float,
    epsilon: float,
    delta: float,
    max_grad_norm: float,
    device: str = "cpu",
    seed: int = 42,
) -> dict:
    """Train a model with Opacus DP-SGD and return metrics.

    This is a self-contained training loop — AMP disabled, no gradient
    accumulation, single learning rate. Designed for correct privacy
    accounting with minimal interaction effects.
    """
    from evaluation.metrics import compute_metrics

    torch.manual_seed(seed)

    model = model_class(*model_args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    dp_model, dp_optimizer, dp_loader, privacy_engine = create_dp_training_components(
        model=model,
        optimizer=optimizer,
        data_loader=train_loader,
        epochs=epochs,
        epsilon=epsilon,
        delta=delta,
        max_grad_norm=max_grad_norm,
    )
    dp_model = dp_model.to(device)

    epoch_grad_norms: list[dict[str, float]] = []
    epoch_losses: list[float] = []

    for _epoch in range(epochs):
        dp_model.train()
        total_loss = 0.0
        step_count = 0

        for batch in dp_loader:
            logits, labels = _forward_batch(dp_model, batch, device)
            # Poisson sampling can yield empty batches — skip them
            if labels.shape[0] == 0:
                continue
            loss = torch.nn.functional.cross_entropy(logits, labels)
            loss.backward()

            grad_norms = _collect_gradient_norms(dp_model)
            if grad_norms:
                epoch_grad_norms.append(grad_norms)

            dp_optimizer.step()
            dp_optimizer.zero_grad()

            total_loss += loss.item()
            step_count += 1

        avg_loss = total_loss / max(step_count, 1)
        epoch_losses.append(avg_loss)

    # Evaluate on validation set
    dp_model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in val_loader:
            logits, labels = _forward_batch(dp_model, batch, device)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    class_names = [str(i) for i in range(num_classes)]
    metrics = compute_metrics(all_labels, all_preds, class_names)

    eps_actual = privacy_engine.get_epsilon(delta)

    return {
        "epsilon_target": epsilon,
        "epsilon_actual": eps_actual,
        "delta": delta,
        "max_grad_norm": max_grad_norm,
        "train_loss": epoch_losses[-1] if epoch_losses else 0.0,
        "val_macro_f1": metrics.macro_f1,
        "val_accuracy": metrics.accuracy,
        "per_class_f1": metrics.per_class_f1.tolist(),
        "epoch_losses": epoch_losses,
        "gradient_norms_sample": epoch_grad_norms[:5] if epoch_grad_norms else [],
    }
