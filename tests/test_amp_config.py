"""Tests for AMP configuration — all CPU-safe."""

import torch

from training.config import TrainConfig


def test_use_amp_default_false():
    """use_amp defaults to False to preserve existing fp32 behavior."""
    config = TrainConfig()
    assert config.use_amp is False


def test_use_amp_serializes_to_dict():
    """use_amp appears in model_dump() for MLflow param logging."""
    config = TrainConfig(use_amp=True)
    params = config.model_dump()
    assert params["use_amp"] is True


def test_gradscaler_disabled_is_noop():
    """GradScaler with enabled=False passes tensors through unchanged."""
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    t = torch.tensor(1.0)
    assert scaler.scale(t) == t
