"""Shared batch-handling utilities for privacy modules."""

from __future__ import annotations

import torch


def forward_batch(
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
    elif len(batch) == 2:
        # Simple (inputs, labels) from TensorDataset
        inputs, labels = batch[0].to(device), batch[1].to(device)
        logits = model(inputs)
    else:
        # Multi-tensor tuple: all but last are model inputs, last is labels
        model_inputs = [t.to(device) for t in batch[:-1]]
        labels = batch[-1].to(device)
        logits = model(*model_inputs)
    return logits, labels


def batch_size(batch: tuple | dict) -> int:
    """Return the number of samples in a batch (tuple or dict format)."""
    if isinstance(batch, dict):
        return batch["labels"].shape[0]
    return batch[0].shape[0]
