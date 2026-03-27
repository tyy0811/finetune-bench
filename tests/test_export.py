"""Tests for ONNX export."""

import os
import tempfile

import numpy as np
import torch
import pytest

from models.fusion_model import MultimodalClassifier
from evaluation.export import export_to_onnx


def test_onnx_export_produces_file():
    """ONNX export should produce a valid .onnx file."""
    model = MultimodalClassifier(
        num_classes=5,
        tabular_input_dim=20,
        tabular_hidden_dim=32,
        tabular_embed_dim=16,
        fusion_hidden_dim=32,
    )
    model.eval()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "model.onnx")
        export_to_onnx(model, output_path=path, tabular_dim=20, seq_length=16)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0


def test_onnx_inference_matches_pytorch():
    """ONNX output should approximate PyTorch output."""
    import onnxruntime as ort

    torch.manual_seed(42)
    model = MultimodalClassifier(
        num_classes=5,
        tabular_input_dim=20,
        tabular_hidden_dim=32,
        tabular_embed_dim=16,
        fusion_hidden_dim=32,
    )
    model.eval()

    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    tabular = torch.randn(batch_size, 20)

    with torch.no_grad():
        pt_out = model(
            {"input_ids": input_ids, "attention_mask": attention_mask},
            tabular,
        ).numpy()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "model.onnx")
        export_to_onnx(model, path, tabular_dim=20, seq_length=seq_len)

        session = ort.InferenceSession(path)
        onnx_out = session.run(
            None,
            {
                "input_ids": input_ids.numpy(),
                "attention_mask": attention_mask.numpy(),
                "tabular_features": tabular.numpy(),
            },
        )[0]

    np.testing.assert_allclose(pt_out, onnx_out, atol=1e-4)
