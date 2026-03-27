"""Test that model produces identical outputs given identical seeds."""

import torch

from models.fusion_model import MultimodalClassifier


def _create_inputs(seed: int, batch_size: int = 2, seq_len: int = 16, tab_dim: int = 60):
    torch.manual_seed(seed)
    text_inputs = {
        "input_ids": torch.randint(0, 1000, (batch_size, seq_len)),
        "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
    }
    tabular = torch.randn(batch_size, tab_dim)
    return text_inputs, tabular


def test_deterministic_forward():
    """Two forward passes with same seed produce identical output."""
    tab_dim = 60

    torch.manual_seed(42)
    model1 = MultimodalClassifier(num_classes=5, tabular_input_dim=tab_dim, modality_dropout=False)
    model1.eval()
    text1, tab1 = _create_inputs(seed=99, tab_dim=tab_dim)
    with torch.no_grad():
        out1 = model1(text1, tab1)

    torch.manual_seed(42)
    model2 = MultimodalClassifier(num_classes=5, tabular_input_dim=tab_dim, modality_dropout=False)
    model2.eval()
    text2, tab2 = _create_inputs(seed=99, tab_dim=tab_dim)
    with torch.no_grad():
        out2 = model2(text2, tab2)

    assert torch.allclose(out1, out2, atol=1e-6), (
        f"Non-deterministic output: max diff = {(out1 - out2).abs().max().item()}"
    )
