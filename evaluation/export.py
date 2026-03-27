"""ONNX export and latency benchmarking."""

import os
import time

import numpy as np
import torch

from models.fusion_model import MultimodalClassifier


class _ExportWrapper(torch.nn.Module):
    """Wrapper that takes flat tensor inputs for ONNX export."""

    def __init__(self, model: MultimodalClassifier):
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        tabular_features: torch.Tensor,
    ) -> torch.Tensor:
        text_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return self.model(text_inputs, tabular_features)


def export_to_onnx(
    model: MultimodalClassifier,
    output_path: str,
    tabular_dim: int,
    seq_length: int = 128,
    batch_size: int = 1,
) -> None:
    """Export model to ONNX format."""
    model.eval()
    wrapper = _ExportWrapper(model)

    dummy_input_ids = torch.randint(0, 1000, (batch_size, seq_length))
    dummy_attention_mask = torch.ones(batch_size, seq_length, dtype=torch.long)
    dummy_tabular = torch.randn(batch_size, tabular_dim)

    torch.onnx.export(
        wrapper,
        (dummy_input_ids, dummy_attention_mask, dummy_tabular),
        output_path,
        input_names=["input_ids", "attention_mask", "tabular_features"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size"},
            "attention_mask": {0: "batch_size"},
            "tabular_features": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
        opset_version=14,
    )


def benchmark_latency(
    model: MultimodalClassifier,
    onnx_path: str,
    tabular_dim: int,
    seq_length: int = 128,
    n_runs: int = 50,
) -> dict:
    """Compare PyTorch vs ONNX latency. Returns timing dict."""
    import onnxruntime as ort

    model.eval()

    # Single-sample inputs
    input_ids = torch.randint(0, 1000, (1, seq_length))
    attention_mask = torch.ones(1, seq_length, dtype=torch.long)
    tabular = torch.randn(1, tabular_dim)

    # Batch inputs
    input_ids_batch = torch.randint(0, 1000, (32, seq_length))
    attention_mask_batch = torch.ones(32, seq_length, dtype=torch.long)
    tabular_batch = torch.randn(32, tabular_dim)

    # PyTorch latency
    with torch.no_grad():
        for _ in range(5):  # warmup
            model({"input_ids": input_ids, "attention_mask": attention_mask}, tabular)

        start = time.perf_counter()
        for _ in range(n_runs):
            model({"input_ids": input_ids, "attention_mask": attention_mask}, tabular)
        pt_single = (time.perf_counter() - start) / n_runs * 1000

        start = time.perf_counter()
        for _ in range(n_runs):
            model(
                {"input_ids": input_ids_batch, "attention_mask": attention_mask_batch},
                tabular_batch,
            )
        pt_batch = (time.perf_counter() - start) / n_runs * 1000

    # ONNX latency
    session = ort.InferenceSession(onnx_path)

    for _ in range(5):  # warmup
        session.run(
            None,
            {
                "input_ids": input_ids.numpy(),
                "attention_mask": attention_mask.numpy(),
                "tabular_features": tabular.numpy(),
            },
        )

    start = time.perf_counter()
    for _ in range(n_runs):
        session.run(
            None,
            {
                "input_ids": input_ids.numpy(),
                "attention_mask": attention_mask.numpy(),
                "tabular_features": tabular.numpy(),
            },
        )
    onnx_single = (time.perf_counter() - start) / n_runs * 1000

    start = time.perf_counter()
    for _ in range(n_runs):
        session.run(
            None,
            {
                "input_ids": input_ids_batch.numpy(),
                "attention_mask": attention_mask_batch.numpy(),
                "tabular_features": tabular_batch.numpy(),
            },
        )
    onnx_batch = (time.perf_counter() - start) / n_runs * 1000

    onnx_size_mb = os.path.getsize(onnx_path) / 1e6

    return {
        "pytorch_single_ms": round(pt_single, 2),
        "pytorch_batch32_ms": round(pt_batch, 2),
        "onnx_single_ms": round(onnx_single, 2),
        "onnx_batch32_ms": round(onnx_batch, 2),
        "onnx_size_mb": round(onnx_size_mb, 1),
    }
