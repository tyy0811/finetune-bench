"""Membership inference attack via loss-threshold method."""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

from privacy._batch import forward_batch


def balance_member_nonmember(
    members: list,
    nonmembers: list,
    seed: int = 42,
) -> tuple[list, list]:
    """Subsample the larger set to match the smaller set size.

    Ensures 50/50 class balance for valid AUC computation.
    """
    rng = random.Random(seed)
    n = min(len(members), len(nonmembers))
    balanced_m = rng.sample(members, n) if len(members) > n else list(members)
    balanced_nm = rng.sample(nonmembers, n) if len(nonmembers) > n else list(nonmembers)
    return balanced_m, balanced_nm


def compute_mia_auc(
    member_losses: list[float],
    nonmember_losses: list[float],
) -> dict:
    """Compute membership inference AUC from per-sample losses.

    Lower loss = higher membership confidence (model is more confident
    on data it trained on). AUC ~ 0.5 means no leakage; AUC >> 0.5
    means the model memorizes training data.
    """
    labels = [1] * len(member_losses) + [0] * len(nonmember_losses)
    # Negative loss as score: lower loss -> higher score -> more likely member
    scores = [-loss for loss in member_losses] + [-loss for loss in nonmember_losses]

    auc = roc_auc_score(labels, scores)
    train_mean = float(np.mean(member_losses))
    test_mean = float(np.mean(nonmember_losses))

    return {
        "mia_auc": round(float(auc), 4),
        "train_loss_mean": round(train_mean, 4),
        "test_loss_mean": round(test_mean, 4),
        "loss_gap": round(test_mean - train_mean, 4),
    }


def stratified_mia_by_entity(
    member_losses: list[float],
    member_companies: list[str],
    nonmember_losses: list[float],
    company_train_counts: dict[str, int],
    top_n: int = 10,
) -> dict:
    """Run MIA separately on high-frequency and low-frequency company subgroups.

    This tests whether entity memorization (Finding #7) drives the MIA signal.
    High-frequency companies appear often in training -> model may memorize them more.
    """
    # Identify top-N companies by training count
    sorted_companies = sorted(company_train_counts.items(), key=lambda x: x[1], reverse=True)
    high_freq_set = {name for name, _ in sorted_companies[:top_n]}

    high_freq_losses = []
    low_freq_losses = []
    for loss, company in zip(member_losses, member_companies):
        if company in high_freq_set:
            high_freq_losses.append(loss)
        else:
            low_freq_losses.append(loss)

    result: dict = {
        "high_freq_count": len(high_freq_losses),
        "low_freq_count": len(low_freq_losses),
    }

    # Need at least 2 samples in each stratum + nonmembers for AUC
    if len(high_freq_losses) >= 2 and len(nonmember_losses) >= 2:
        balanced_hf, balanced_nm = balance_member_nonmember(
            high_freq_losses, nonmember_losses
        )
        hf_result = compute_mia_auc(balanced_hf, balanced_nm)
        result["high_freq_company_auc"] = hf_result["mia_auc"]
    else:
        result["high_freq_company_auc"] = None

    if len(low_freq_losses) >= 2 and len(nonmember_losses) >= 2:
        balanced_lf, balanced_nm = balance_member_nonmember(
            low_freq_losses, nonmember_losses
        )
        lf_result = compute_mia_auc(balanced_lf, balanced_nm)
        result["low_freq_company_auc"] = lf_result["mia_auc"]
    else:
        result["low_freq_company_auc"] = None

    return result


def compute_per_sample_loss(
    model: torch.nn.Module,
    dataset: Dataset,
    batch_size: int = 32,
    device: str = "cpu",
    forward_fn: Callable[..., Any] | None = None,
) -> list[float]:
    """Compute cross-entropy loss for each sample in the dataset.

    Handles both TensorDataset (inputs, labels) tuples and dict-style
    batches from ComplaintDataset ({"text": ..., "tabular": ..., "labels": ...}).
    Pass a custom forward_fn(model, batch, device) -> (logits, labels) to
    override the default batch handling.
    """
    if forward_fn is None:
        forward_fn = forward_batch

    model = model.to(device)
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_losses: list[float] = []

    with torch.no_grad():
        for batch in loader:
            logits, labels = forward_fn(model, batch, device)
            # Per-sample loss (no reduction)
            losses = torch.nn.functional.cross_entropy(
                logits, labels, reduction="none"
            )
            all_losses.extend(losses.cpu().tolist())

    return all_losses
