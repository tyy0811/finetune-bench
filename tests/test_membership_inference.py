"""Tests for privacy.membership_inference module."""

import numpy as np
import torch

from privacy.membership_inference import (
    balance_member_nonmember,
    compute_mia_auc,
    compute_per_sample_loss,
    stratified_mia_by_entity,
)


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


class TestStratifiedMia:
    def test_splits_by_company_frequency(self):
        member_losses = [0.1, 0.2, 0.8, 0.9, 0.3, 0.4]
        member_companies = ["BigCo", "BigCo", "SmallCo", "SmallCo", "BigCo", "TinyCo"]
        nonmember_losses = [0.5, 0.6, 0.7, 0.8, 0.5, 0.6]
        company_counts = {"BigCo": 5000, "SmallCo": 100, "TinyCo": 50}

        result = stratified_mia_by_entity(
            member_losses=member_losses,
            member_companies=member_companies,
            nonmember_losses=nonmember_losses,
            company_train_counts=company_counts,
            top_n=1,
        )
        assert "high_freq_company_auc" in result
        assert "low_freq_company_auc" in result
        assert "high_freq_count" in result
        assert "low_freq_count" in result

    def test_high_freq_count_is_correct(self):
        member_losses = [0.1, 0.2, 0.3, 0.4]
        member_companies = ["BigCo", "BigCo", "SmallCo", "SmallCo"]
        nonmember_losses = [0.5, 0.6]
        company_counts = {"BigCo": 5000, "SmallCo": 100}

        result = stratified_mia_by_entity(
            member_losses=member_losses,
            member_companies=member_companies,
            nonmember_losses=nonmember_losses,
            company_train_counts=company_counts,
            top_n=1,
        )
        assert result["high_freq_count"] == 2  # two BigCo members
        assert result["low_freq_count"] == 2   # two SmallCo members


class TestPerSampleLoss:
    def test_returns_one_loss_per_sample(self):
        model = torch.nn.Linear(10, 3)
        dataset = torch.utils.data.TensorDataset(
            torch.randn(16, 10), torch.randint(0, 3, (16,))
        )
        losses = compute_per_sample_loss(model, dataset, batch_size=4, device="cpu")
        assert len(losses) == 16

    def test_losses_are_positive(self):
        model = torch.nn.Linear(10, 3)
        dataset = torch.utils.data.TensorDataset(
            torch.randn(8, 10), torch.randint(0, 3, (8,))
        )
        losses = compute_per_sample_loss(model, dataset, batch_size=4, device="cpu")
        assert all(loss > 0 for loss in losses)

    def test_overfit_model_has_low_loss_on_train(self):
        """A model overfit to its training data should have low loss on that data."""
        torch.manual_seed(42)
        model = torch.nn.Linear(10, 3)
        data_x = torch.randn(8, 10)
        data_y = torch.randint(0, 3, (8,))
        dataset = torch.utils.data.TensorDataset(data_x, data_y)

        # Overfit
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        for _ in range(200):
            logits = model(data_x)
            loss = torch.nn.functional.cross_entropy(logits, data_y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        losses = compute_per_sample_loss(model, dataset, batch_size=8, device="cpu")
        assert np.mean(losses) < 0.5  # should be very low after overfitting
