"""Tests for fusion model forward pass shapes."""

import torch
import pytest

from models.fusion_model import MultimodalClassifier


@pytest.fixture
def model():
    return MultimodalClassifier(
        num_classes=10,
        tabular_input_dim=120,
        tabular_hidden_dim=128,
        tabular_embed_dim=64,
        fusion_hidden_dim=256,
        dropout=0.3,
        modality_dropout=False,
    )


@pytest.fixture
def model_with_dropout():
    return MultimodalClassifier(
        num_classes=10,
        tabular_input_dim=120,
        modality_dropout=True,
    )


class TestFusionModel:
    def test_forward_shape(self, model):
        batch_size = 4
        text_inputs = {
            "input_ids": torch.randint(0, 1000, (batch_size, 32)),
            "attention_mask": torch.ones(batch_size, 32, dtype=torch.long),
        }
        tabular = torch.randn(batch_size, 120)
        logits = model(text_inputs, tabular)
        assert logits.shape == (batch_size, 10)

    def test_text_only_mode(self, model):
        """When tabular features are zero, model still produces valid logits."""
        batch_size = 4
        text_inputs = {
            "input_ids": torch.randint(0, 1000, (batch_size, 32)),
            "attention_mask": torch.ones(batch_size, 32, dtype=torch.long),
        }
        tabular = torch.zeros(batch_size, 120)
        logits = model(text_inputs, tabular)
        assert logits.shape == (batch_size, 10)

    def test_no_token_type_ids(self, model):
        """DistilBERT does NOT use token_type_ids — model must strip them."""
        batch_size = 2
        text_inputs = {
            "input_ids": torch.randint(0, 1000, (batch_size, 16)),
            "attention_mask": torch.ones(batch_size, 16, dtype=torch.long),
            "token_type_ids": torch.zeros(batch_size, 16, dtype=torch.long),
        }
        tabular = torch.randn(batch_size, 120)
        logits = model(text_inputs, tabular)
        assert logits.shape == (batch_size, 10)

    def test_modality_dropout_train_mode(self, model_with_dropout):
        """Modality dropout only active during training."""
        model_with_dropout.train()
        batch_size = 4
        text_inputs = {
            "input_ids": torch.randint(0, 1000, (batch_size, 32)),
            "attention_mask": torch.ones(batch_size, 32, dtype=torch.long),
        }
        tabular = torch.randn(batch_size, 120)
        logits = model_with_dropout(text_inputs, tabular)
        assert logits.shape == (batch_size, 10)

    def test_modality_dropout_eval_mode(self, model_with_dropout):
        """Modality dropout inactive during eval — output is deterministic."""
        model_with_dropout.eval()
        batch_size = 4
        text_inputs = {
            "input_ids": torch.randint(0, 1000, (batch_size, 32)),
            "attention_mask": torch.ones(batch_size, 32, dtype=torch.long),
        }
        tabular = torch.randn(batch_size, 120)
        logits = model_with_dropout(text_inputs, tabular)
        assert logits.shape == (batch_size, 10)
