"""Differential privacy training via Opacus DP-SGD."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader


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
) -> tuple[torch.nn.Module, torch.optim.Optimizer, DataLoader]:
    """Wrap model, optimizer, and dataloader with Opacus DP-SGD.

    Returns the wrapped (model, optimizer, dataloader) tuple.
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
    return dp_model, dp_optimizer, dp_loader
