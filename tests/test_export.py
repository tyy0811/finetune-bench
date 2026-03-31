"""Tests for ONNX export."""

import os
import tempfile

import numpy as np
import torch

from evaluation.export import convert_onnx_to_fp16, export_to_onnx
from models.fusion_model import MultimodalClassifier


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


def test_onnx_fp16_conversion():
    """fp16 ONNX model is smaller than fp32 and numerically close."""
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
    # Fixed inputs for deterministic comparison
    torch.manual_seed(42)
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    tabular = torch.randn(batch_size, 20)

    with tempfile.TemporaryDirectory() as tmpdir:
        fp32_path = os.path.join(tmpdir, "model.onnx")
        fp16_path = os.path.join(tmpdir, "model_fp16.onnx")

        export_to_onnx(model, fp32_path, tabular_dim=20, seq_length=seq_len)
        convert_onnx_to_fp16(fp32_path, fp16_path)

        assert os.path.exists(fp16_path)
        assert os.path.getsize(fp16_path) < os.path.getsize(fp32_path)

        onnx_inputs = {
            "input_ids": input_ids.numpy(),
            "attention_mask": attention_mask.numpy(),
            "tabular_features": tabular.numpy(),
        }

        # Run both fp32 and fp16 ONNX with same inputs
        fp32_session = ort.InferenceSession(fp32_path)
        fp32_out = fp32_session.run(None, onnx_inputs)[0]

        fp16_session = ort.InferenceSession(fp16_path)
        fp16_out = fp16_session.run(None, onnx_inputs)[0]

        assert fp16_out.shape == (batch_size, 5)
        # fp16 precision: atol=1e-2 accounts for half-precision rounding
        np.testing.assert_allclose(fp16_out, fp32_out, atol=1e-2)
