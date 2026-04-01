"""Tests for privacy.membership_inference module."""

import numpy as np

from privacy.membership_inference import balance_member_nonmember, compute_mia_auc


class TestComputeMiaAuc:
    def test_perfect_separation(self):
        """When members have low loss and non-members have high loss, AUC ~ 1.0."""
        member_losses = [0.1, 0.2, 0.15, 0.05]
        nonmember_losses = [2.0, 1.8, 2.5, 1.9]
        result = compute_mia_auc(member_losses, nonmember_losses)
        assert result["mia_auc"] > 0.95

    def test_no_separation(self):
        """When losses overlap completely, AUC ~ 0.5."""
        rng = np.random.RandomState(42)
        member_losses = rng.normal(1.0, 0.5, 200).tolist()
        nonmember_losses = rng.normal(1.0, 0.5, 200).tolist()
        result = compute_mia_auc(member_losses, nonmember_losses)
        assert 0.35 < result["mia_auc"] < 0.65

    def test_output_keys(self):
        member_losses = [0.5, 0.6]
        nonmember_losses = [0.7, 0.8]
        result = compute_mia_auc(member_losses, nonmember_losses)
        assert "mia_auc" in result
        assert "train_loss_mean" in result
        assert "test_loss_mean" in result
        assert "loss_gap" in result

    def test_loss_gap_is_difference(self):
        member_losses = [0.3, 0.5]
        nonmember_losses = [0.9, 1.1]
        result = compute_mia_auc(member_losses, nonmember_losses)
        expected_gap = np.mean(nonmember_losses) - np.mean(member_losses)
        assert abs(result["loss_gap"] - expected_gap) < 1e-6


class TestBalanceMemberNonmember:
    def test_subsamples_to_match(self):
        members = list(range(100))
        nonmembers = list(range(20))
        balanced_m, balanced_nm = balance_member_nonmember(members, nonmembers, seed=42)
        assert len(balanced_m) == len(balanced_nm) == 20

    def test_keeps_all_when_equal(self):
        members = list(range(50))
        nonmembers = list(range(50))
        balanced_m, balanced_nm = balance_member_nonmember(members, nonmembers, seed=42)
        assert len(balanced_m) == len(balanced_nm) == 50

    def test_subsamples_nonmembers_when_fewer_members(self):
        members = list(range(10))
        nonmembers = list(range(100))
        balanced_m, balanced_nm = balance_member_nonmember(members, nonmembers, seed=42)
        assert len(balanced_m) == len(balanced_nm) == 10

    def test_deterministic_with_seed(self):
        members = list(range(100))
        nonmembers = list(range(50))
        r1 = balance_member_nonmember(members, nonmembers, seed=42)
        r2 = balance_member_nonmember(members, nonmembers, seed=42)
        assert r1[0] == r2[0]
