"""Multimodal classifier: DistilBERT + tabular MLP + fusion head."""

import random

import torch
import torch.nn as nn
from transformers import DistilBertModel


class MultimodalClassifier(nn.Module):
    """Text+tabular fusion classifier.

    Text branch: DistilBERT [CLS] embedding (768-dim)
    Tabular branch: MLP -> 64-dim embedding
    Fusion: concat -> FC -> ReLU -> Dropout -> FC -> logits
    """

    def __init__(
        self,
        num_classes: int,
        tabular_input_dim: int,
        tabular_hidden_dim: int = 128,
        tabular_embed_dim: int = 64,
        fusion_hidden_dim: int = 256,
        dropout: float = 0.3,
        modality_dropout: bool = False,
        text_model_name: str = "distilbert-base-uncased",
        text_dropout_prob: float = 0.1,
        tabular_dropout_prob: float = 0.1,
        text_encoder: DistilBertModel | None = None,
    ):
        super().__init__()

        self.text_encoder = text_encoder if text_encoder is not None else DistilBertModel.from_pretrained(text_model_name)
        text_dim = self.text_encoder.config.dim  # 768

        self.tabular_mlp = nn.Sequential(
            nn.Linear(tabular_input_dim, tabular_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(tabular_hidden_dim, tabular_embed_dim),
        )

        fusion_input_dim = text_dim + tabular_embed_dim  # 768 + 64 = 832
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, num_classes),
        )

        self.modality_dropout = modality_dropout
        self.text_dropout_prob = text_dropout_prob
        self.tabular_dropout_prob = tabular_dropout_prob

    def forward(
        self,
        text_inputs: dict[str, torch.Tensor],
        tabular_features: torch.Tensor,
    ) -> torch.Tensor:
        # DistilBERT does NOT use token_type_ids
        text_inputs = {
            k: v for k, v in text_inputs.items() if k != "token_type_ids"
        }

        text_output = self.text_encoder(**text_inputs)
        # last_hidden_state is first element whether output is tuple or dict
        hidden_states = (
            text_output.last_hidden_state
            if hasattr(text_output, "last_hidden_state")
            else text_output[0]
        )
        cls_embedding = hidden_states[:, 0, :]  # (B, 768)

        tabular_embedding = self.tabular_mlp(tabular_features)  # (B, 64)

        # Modality dropout: mutually exclusive, training only
        if self.training and self.modality_dropout:
            drop_roll = random.random()
            if drop_roll < self.text_dropout_prob:
                cls_embedding = torch.zeros_like(cls_embedding)
            elif drop_roll < self.text_dropout_prob + self.tabular_dropout_prob:
                tabular_embedding = torch.zeros_like(tabular_embedding)

        fused = torch.cat([cls_embedding, tabular_embedding], dim=-1)
        return self.fusion_head(fused)
