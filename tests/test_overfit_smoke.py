"""Smoke test: 1-batch overfit. Loss should drop to near zero."""

import torch
import torch.nn.functional as F
from transformers import DistilBertTokenizer

from models.fusion_model import MultimodalClassifier


def test_overfit_one_batch():
    """Train on a single batch for many steps. Loss must approach 0."""
    torch.manual_seed(42)

    num_classes = 3
    batch_size = 4
    tabular_dim = 10
    seq_len = 16

    model = MultimodalClassifier(
        num_classes=num_classes,
        tabular_input_dim=tabular_dim,
        tabular_hidden_dim=32,
        tabular_embed_dim=16,
        fusion_hidden_dim=32,
        dropout=0.0,  # No dropout for overfitting
        modality_dropout=False,
    )

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    texts = [
        "This is a complaint about my mortgage payment",
        "Credit card fraud on my account",
        "Student loan billing issue",
        "Debt collector harassment calls",
    ]
    encodings = tokenizer(
        texts,
        max_length=seq_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    text_inputs = {k: v for k, v in encodings.items()}
    tabular = torch.randn(batch_size, tabular_dim)
    labels = torch.tensor([0, 1, 2, 0])

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    initial_loss = None
    for step in range(50):
        optimizer.zero_grad()
        logits = model(text_inputs, tabular)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()

        if initial_loss is None:
            initial_loss = loss.item()

    final_loss = loss.item()

    assert final_loss < initial_loss * 0.1, (
        f"Loss did not converge: {initial_loss:.4f} -> {final_loss:.4f}"
    )
    assert final_loss < 0.1, f"Final loss too high: {final_loss:.4f}"
