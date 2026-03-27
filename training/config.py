"""Training configuration dataclass."""

from pydantic import BaseModel, Field


class TrainConfig(BaseModel):
    """Configuration for training runs."""

    # Variant
    variant: str = "M2"  # B1, B2, M1, M2, M3
    seed: int = 42

    # Data
    sample_size: int = 20_000
    max_seq_length: int = 128
    split_strategy: str = "random"
    cutoff_date: str | None = None
    exclude_features: list[str] = Field(default_factory=list)

    # Model
    text_model_name: str = "distilbert-base-uncased"
    tabular_hidden_dim: int = 128
    tabular_embed_dim: int = 64
    fusion_hidden_dim: int = 256
    text_dropout_prob: float = 0.1
    tabular_dropout_prob: float = 0.1

    # Training
    lr_encoder: float = 2e-5
    lr_head: float = 1e-3
    batch_size: int = 16
    grad_accumulation_steps: int = 2
    num_epochs: int = 3
    max_grad_norm: float = 1.0
    warmup_fraction: float = 0.1
    dropout: float = 0.3
    early_stopping_patience: int = 2

    # MLflow
    experiment_name: str = "finetune-bench"
    run_name: str | None = None

    # Paths
    results_dir: str = "results"
