"""Membership inference attack via loss-threshold method."""

from __future__ import annotations

import random

import numpy as np
from sklearn.metrics import roc_auc_score


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
