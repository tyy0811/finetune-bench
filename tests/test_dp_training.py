"""Tests for privacy.dp_training module."""

import pytest
import torch

from privacy.dp_training import create_dp_training_components, make_dp_config


class TestDpConfig:
    def test_amp_disabled(self):
        dp_config = make_dp_config(epsilon=8.0, delta=1e-5, max_grad_norm=1.0)
        assert dp_config["use_amp"] is False

    def test_grad_accumulation_is_one(self):
        dp_config = make_dp_config(epsilon=8.0, delta=1e-5, max_grad_norm=1.0)
        assert dp_config["grad_accumulation_steps"] == 1

    def test_epsilon_stored(self):
        dp_config = make_dp_config(epsilon=1.0, delta=1e-5, max_grad_norm=0.5)
        assert dp_config["epsilon"] == 1.0
        assert dp_config["delta"] == 1e-5
        assert dp_config["max_grad_norm"] == 0.5


class TestDpTrainingComponents:
    """Test that Opacus wraps model/optimizer/dataloader correctly.

    These tests require opacus to be installed. Skip if not available.
    """

    @pytest.fixture
    def _small_components(self):
        """Minimal model, optimizer, dataloader for Opacus testing."""
        pytest.importorskip("opacus")

        model = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 3),
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        data = torch.utils.data.TensorDataset(
            torch.randn(32, 10), torch.randint(0, 3, (32,))
        )
        loader = torch.utils.data.DataLoader(data, batch_size=8)
        return model, optimizer, loader

    def test_privacy_engine_attaches(self, _small_components):
        opacus = pytest.importorskip("opacus")
        model, optimizer, loader = _small_components

        dp_model, dp_optimizer, dp_loader = create_dp_training_components(
            model=model,
            optimizer=optimizer,
            data_loader=loader,
            epochs=3,
            epsilon=8.0,
            delta=1e-5,
            max_grad_norm=1.0,
        )
        # Opacus wraps model in GradSampleModule
        assert isinstance(dp_model, opacus.GradSampleModule)

    def test_dp_model_trains_one_step(self, _small_components):
        pytest.importorskip("opacus")
        model, optimizer, loader = _small_components

        dp_model, dp_optimizer, dp_loader = create_dp_training_components(
            model=model,
            optimizer=optimizer,
            data_loader=loader,
            epochs=3,
            epsilon=50.0,
            delta=1e-5,
            max_grad_norm=1.0,
        )
        # One forward-backward-step cycle should not crash
        batch_x, batch_y = next(iter(dp_loader))
        logits = dp_model(batch_x)
        loss = torch.nn.functional.cross_entropy(logits, batch_y)
        loss.backward()
        dp_optimizer.step()
        dp_optimizer.zero_grad()

    def test_privacy_budget_consumed(self, _small_components):
        pytest.importorskip("opacus")
        model, optimizer, loader = _small_components

        dp_model, dp_optimizer, dp_loader = create_dp_training_components(
            model=model,
            optimizer=optimizer,
            data_loader=loader,
            epochs=3,
            epsilon=50.0,
            delta=1e-5,
            max_grad_norm=1.0,
        )
        # Train one full epoch
        for batch_x, batch_y in dp_loader:
            logits = dp_model(batch_x)
            loss = torch.nn.functional.cross_entropy(logits, batch_y)
            loss.backward()
            dp_optimizer.step()
            dp_optimizer.zero_grad()

        eps = dp_optimizer.privacy_engine.get_epsilon(1e-5)
        assert eps > 0  # some budget consumed
