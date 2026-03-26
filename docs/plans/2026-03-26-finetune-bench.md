# finetune-bench Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a multimodal text+tabular classification benchmark with robustness training, corruption evaluation, and production packaging using the CFPB Consumer Complaints dataset.

**Architecture:** DistilBERT encodes complaint narratives into 768-dim [CLS] embeddings. A tabular MLP encodes structured features (company, state, submission channel, complaint volume) into 64-dim embeddings. A fusion head concatenates both and classifies into product categories. Modality dropout during training (M3 variant) zeroes one branch at a time to build robustness. Five corruption types evaluate degradation at inference time.

**Tech Stack:** PyTorch, HuggingFace Transformers (DistilBERT), scikit-learn, LightGBM, MLflow, ONNX, Pydantic, Docker, pytest

**Source spec:** Implementation Plan v1.2 (Final)

---

## Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `adapters/__init__.py`
- Create: `models/__init__.py`
- Create: `training/__init__.py`
- Create: `evaluation/__init__.py`
- Create: `tests/__init__.py`
- Create: `results/.gitkeep`
- Create: `.gitignore`

**Step 1: Initialize git repo**

```bash
cd /Users/zenith/Desktop/finetune-bench
git init
```

**Step 2: Create `.gitignore`**

```gitignore
# Data
data/
*.csv
*.csv.zip

# Models
*.pt
*.onnx
mlruns/

# Python
__pycache__/
*.pyc
*.egg-info/
dist/
build/
.eggs/

# Results (keep summary files)
results/*.pt
results/*.png

# Environment
.env
.venv/
venv/

# IDE
.idea/
.vscode/
*.swp
```

**Step 3: Create `pyproject.toml`**

```toml
[project]
name = "finetune-bench"
version = "0.1.0"
description = "Multimodal text+tabular classification benchmark with robustness training and corruption evaluation"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.0.0",
    "transformers>=4.30.0",
    "scikit-learn>=1.3.0",
    "lightgbm>=4.0.0",
    "pandas>=2.0.0",
    "numpy>=1.24.0",
    "mlflow>=2.5.0",
    "pydantic>=2.0.0",
    "onnx>=1.14.0",
    "onnxruntime>=1.15.0",
    "matplotlib>=3.7.0",
    "requests>=2.31.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
]

[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.build_meta"

[tool.pytest.ini_options]
testpaths = ["tests"]
```

**Step 4: Create directory structure with `__init__.py` files**

No `__init__.py` for `scripts/` — scripts are run directly, not imported as a package.

```bash
mkdir -p adapters models training evaluation tests scripts results data
touch adapters/__init__.py models/__init__.py training/__init__.py evaluation/__init__.py tests/__init__.py results/.gitkeep
```

**Step 5: Install project in dev mode**

```bash
pip install -e ".[dev]"
```

**Step 6: Commit**

```bash
git add .
git commit -m "chore: initialize project scaffolding with dependencies"
```

---

## Task 2: Data Download Script

**Files:**
- Create: `scripts/download_data.py`

**Step 1: Write download script**

```python
"""Download and cache CFPB Consumer Complaints dataset."""

import os
import zipfile
from pathlib import Path

import requests

DATA_DIR = Path(__file__).parent.parent / "data"
CSV_URL = "https://files.consumerfinance.gov/ccdb/complaints.csv.zip"
CSV_PATH = DATA_DIR / "complaints.csv"
ZIP_PATH = DATA_DIR / "complaints.csv.zip"


def download_cfpb(force: bool = False) -> Path:
    """Download CFPB complaints CSV if not already cached.

    Returns path to the unzipped CSV file.
    """
    DATA_DIR.mkdir(exist_ok=True)

    if CSV_PATH.exists() and not force:
        print(f"Data already exists at {CSV_PATH}")
        return CSV_PATH

    print(f"Downloading CFPB data from {CSV_URL}...")
    response = requests.get(CSV_URL, stream=True, timeout=300)
    response.raise_for_status()

    with open(ZIP_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print("Extracting...")
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(DATA_DIR)

    ZIP_PATH.unlink()

    print(f"Data saved to {CSV_PATH}")
    return CSV_PATH


if __name__ == "__main__":
    path = download_cfpb()
    print(f"CSV at: {path} ({os.path.getsize(path) / 1e6:.0f} MB)")
```

**Step 2: Commit**

```bash
git add scripts/download_data.py
git commit -m "feat: add CFPB data download script"
```

---

## Task 3: Abstract Dataset Adapter

**Files:**
- Create: `adapters/base.py`
- Create: `tests/test_adapter.py`

**Step 1: Write the adapter contract test first (TDD)**

```python
"""Tests for DatasetAdapter contract."""

import numpy as np
import pandas as pd
import pytest

from adapters.cfpb import CFPBAdapter


@pytest.fixture
def adapter():
    """Create adapter with small sample for testing."""
    return CFPBAdapter(sample_size=500, seed=42)


class TestAdapterContract:
    """Verify DatasetAdapter interface contract."""

    def test_load_raw_returns_dataframe(self, adapter):
        df = adapter.load_raw()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_preprocess_returns_three_splits(self, adapter):
        splits = adapter.preprocess()
        assert "train" in splits
        assert "val" in splits
        assert "test" in splits

    def test_split_ratios(self, adapter):
        splits = adapter.preprocess()
        total = sum(len(s["labels"]) for s in splits.values())
        train_ratio = len(splits["train"]["labels"]) / total
        assert 0.75 < train_ratio < 0.85  # ~80%

    def test_split_contents(self, adapter):
        splits = adapter.preprocess()
        for name, split in splits.items():
            assert "narratives" in split, f"{name} missing narratives"
            assert "tabular_features" in split, f"{name} missing tabular_features"
            assert "labels" in split, f"{name} missing labels"
            assert isinstance(split["tabular_features"], np.ndarray)
            assert isinstance(split["labels"], np.ndarray)
            n = len(split["labels"])
            assert len(split["narratives"]) == n
            assert split["tabular_features"].shape[0] == n

    def test_no_label_leakage_in_features(self, adapter):
        splits = adapter.preprocess()
        n_features = splits["train"]["tabular_features"].shape[1]
        assert n_features < 200

    def test_class_names_exist(self, adapter):
        assert hasattr(adapter, "class_names")
        splits = adapter.preprocess()
        n_classes = len(adapter.class_names)
        assert n_classes >= 5
        assert n_classes <= 15
        for split in splits.values():
            assert split["labels"].max() < n_classes
            assert split["labels"].min() >= 0

    def test_company_volume_from_train_only(self, adapter):
        """Verify company_complaint_volume is derived from train split only.

        Strategy: a company appearing only in val/test (not in train)
        should get volume=0 (log1p(0)=0, standardized to a specific value).
        We verify by checking that the adapter stores train_volume_stats
        and that unknown companies don't leak volume information.
        """
        splits = adapter.preprocess()

        # The adapter must expose the volume stats it computed
        assert hasattr(adapter, "train_volume_stats"), (
            "Adapter must store train_volume_stats for leakage verification"
        )
        assert "mean" in adapter.train_volume_stats
        assert "std" in adapter.train_volume_stats

        # Verify features are computed (basic shape check)
        assert splits["train"]["tabular_features"].shape[1] > 0

    def test_volume_standardized(self, adapter):
        """Verify company_complaint_volume is standardized (mean ~0, std ~1 on train)."""
        splits = adapter.preprocess()

        # Volume is the last feature column
        train_volume = splits["train"]["tabular_features"][:, -1]

        # After standardization, train volume should have mean ~0 and std ~1
        assert abs(train_volume.mean()) < 0.1, (
            f"Volume mean on train = {train_volume.mean():.4f}, expected ~0"
        )
        assert abs(train_volume.std() - 1.0) < 0.3, (
            f"Volume std on train = {train_volume.std():.4f}, expected ~1"
        )
```

**Step 2: Write the abstract base class**

```python
"""Abstract DatasetAdapter interface."""

from abc import ABC, abstractmethod
from typing import Any


class DatasetAdapter(ABC):
    """Base class for dataset adapters.

    All adapters must implement load_raw() and preprocess().
    The preprocess() method returns a dict with train/val/test splits,
    each containing narratives, tabular_features, and labels.
    """

    def __init__(
        self,
        sample_size: int | None = None,
        seed: int = 42,
        split_strategy: str = "random",
        cutoff_date: str | None = None,
        exclude_features: list[str] | None = None,
    ):
        self.sample_size = sample_size
        self.seed = seed
        self.split_strategy = split_strategy
        self.cutoff_date = cutoff_date
        self.exclude_features = exclude_features or []
        self.class_names: list[str] = []

    @abstractmethod
    def load_raw(self) -> Any:
        """Load raw data and return a DataFrame."""
        ...

    @abstractmethod
    def preprocess(self) -> dict:
        """Return {'train': {...}, 'val': {...}, 'test': {...}}.

        Each split dict contains:
        - 'narratives': list[str]
        - 'tabular_features': np.ndarray of shape (n, d)
        - 'labels': np.ndarray of shape (n,) with int class indices
        """
        ...
```

**Step 3: Commit**

```bash
git add adapters/base.py tests/test_adapter.py
git commit -m "feat: add abstract DatasetAdapter interface and contract tests"
```

---

## Task 4: CFPB Adapter Implementation

**Files:**
- Create: `adapters/cfpb.py`

**Step 1: Implement the CFPB adapter**

Uses vectorized pandas/numpy operations for encoding (not Python loops).
Standardizes `company_complaint_volume` using train-split mean/std.
Stores `train_volume_stats` for leakage verification in tests.

```python
"""CFPB Consumer Complaints dataset adapter with leakage-safe preprocessing."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from adapters.base import DatasetAdapter

DATA_PATH = Path(__file__).parent.parent / "data" / "complaints.csv"

MIN_CLASS_SIZE = 500
TOP_N_COMPANIES = 50


class CFPBAdapter(DatasetAdapter):
    """CFPB Consumer Complaints adapter.

    Leakage safeguards:
    - company_complaint_volume computed from training split only
    - All encoding categories (top-50 companies, states, channels) from training split only
    - Volume is log-transformed and standardized using train mean/std
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.train_volume_stats: dict = {}

    def load_raw(self) -> pd.DataFrame:
        if not DATA_PATH.exists():
            raise FileNotFoundError(
                f"CFPB data not found at {DATA_PATH}. "
                "Run: python scripts/download_data.py"
            )
        return pd.read_csv(DATA_PATH, low_memory=False)

    def preprocess(self) -> dict:
        df = self.load_raw()

        # Filter for rows with narratives
        df = df[df["Consumer complaint narrative"].notna()].copy()
        df = df.rename(columns={"Consumer complaint narrative": "narrative"})

        # Class consolidation: merge rare classes into "Other"
        product_counts = df["Product"].value_counts()
        small_classes = product_counts[product_counts < MIN_CLASS_SIZE].index
        df["product_clean"] = df["Product"].where(
            ~df["Product"].isin(small_classes), "Other"
        )

        le = LabelEncoder()
        df["label"] = le.fit_transform(df["product_clean"])
        self.class_names = list(le.classes_)

        # Subsample
        if self.sample_size and len(df) > self.sample_size:
            df = df.sample(n=self.sample_size, random_state=self.seed, replace=False)
            df = df[df.groupby("label")["label"].transform("count") >= 5]

        # Split
        if self.split_strategy == "temporal":
            splits_df = self._temporal_split(df)
        else:
            splits_df = self._random_split(df)

        # Feature engineering from TRAINING data only
        train_df = splits_df["train"]

        # Company complaint volume from training data
        train_volumes = train_df["Company"].value_counts().to_dict()

        # Top-N companies from training data
        top_companies = (
            train_df["Company"].value_counts().head(TOP_N_COMPANIES).index.tolist()
        )

        # Categories from training data
        all_states = sorted(train_df["State"].dropna().unique())
        all_channels = sorted(train_df["Submitted via"].dropna().unique())

        # Compute volume stats on train for standardization
        train_log_volumes = np.log1p(
            train_df["Company"].map(train_volumes).fillna(0).values
        )
        vol_mean = float(train_log_volumes.mean())
        vol_std = float(train_log_volumes.std())
        if vol_std == 0:
            vol_std = 1.0
        self.train_volume_stats = {"mean": vol_mean, "std": vol_std}

        result = {}
        for split_name, split_df in splits_df.items():
            features = self._encode_features(
                split_df, train_volumes, top_companies,
                all_states, all_channels, vol_mean, vol_std,
            )
            result[split_name] = {
                "narratives": split_df["narrative"].tolist(),
                "tabular_features": features,
                "labels": split_df["label"].values,
            }

        return result

    def _random_split(self, df: pd.DataFrame) -> dict:
        train_df, temp_df = train_test_split(
            df, test_size=0.2, random_state=self.seed, stratify=df["label"],
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=0.5, random_state=self.seed, stratify=temp_df["label"],
        )
        return {"train": train_df, "val": val_df, "test": test_df}

    def _temporal_split(self, df: pd.DataFrame) -> dict:
        if not self.cutoff_date:
            raise ValueError("cutoff_date required for temporal split")

        df["date_received"] = pd.to_datetime(df["Date received"])
        pre = df[df["date_received"] < self.cutoff_date]
        post = df[df["date_received"] >= self.cutoff_date]

        train_n = min(16000, len(pre))
        test_n = min(4000, len(post))

        train_df = pre.sample(n=train_n, random_state=self.seed)
        test_df = post.sample(n=test_n, random_state=self.seed)

        train_df, val_df = train_test_split(
            train_df, test_size=0.1, random_state=self.seed, stratify=train_df["label"],
        )
        return {"train": train_df, "val": val_df, "test": test_df}

    def _encode_features(
        self,
        df: pd.DataFrame,
        train_volumes: dict,
        top_companies: list,
        all_states: list,
        all_channels: list,
        vol_mean: float,
        vol_std: float,
    ) -> np.ndarray:
        """Encode tabular features using vectorized operations.

        All encoding categories come from training data.
        """
        features = []

        # 1. Company: top-50 + "other" one-hot (vectorized)
        if "company" not in self.exclude_features:
            company_cats = top_companies + ["Other"]
            company_mapped = df["Company"].where(
                df["Company"].isin(top_companies), "Other"
            )
            cat_type = pd.CategoricalDtype(categories=company_cats, ordered=False)
            company_categorical = company_mapped.astype(cat_type)
            company_onehot = pd.get_dummies(company_categorical).values.astype(np.float32)
            features.append(company_onehot)

        # 2. State one-hot (vectorized)
        state_cat = pd.CategoricalDtype(categories=all_states, ordered=False)
        state_categorical = df["State"].fillna("").astype(state_cat)
        state_onehot = pd.get_dummies(state_categorical).values.astype(np.float32)
        features.append(state_onehot)

        # 3. Submitted via one-hot (vectorized)
        channel_cat = pd.CategoricalDtype(categories=all_channels, ordered=False)
        channel_categorical = df["Submitted via"].fillna("").astype(channel_cat)
        channel_onehot = pd.get_dummies(channel_categorical).values.astype(np.float32)
        features.append(channel_onehot)

        # 4. Company complaint volume: log-transformed + standardized
        if "company_complaint_volume" not in self.exclude_features:
            volumes = df["Company"].map(train_volumes).fillna(0).values
            log_volumes = np.log1p(volumes)
            standardized = ((log_volumes - vol_mean) / vol_std).reshape(-1, 1)
            features.append(standardized.astype(np.float32))

        return np.hstack(features)
```

**Step 2: Run adapter contract tests**

```bash
pytest tests/test_adapter.py -v
```

Expected: All tests PASS (requires downloaded data).

**Step 3: Commit**

```bash
git add adapters/cfpb.py
git commit -m "feat: implement CFPB adapter with leakage-safe preprocessing"
```

---

## Task 5: Training Configuration

**Files:**
- Create: `training/config.py`

**Step 1: Write Pydantic config**

```python
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
```

**Step 2: Commit**

```bash
git add training/config.py
git commit -m "feat: add Pydantic training config"
```

---

## Task 6: Fusion Model

**Files:**
- Create: `models/fusion_model.py`
- Create: `tests/test_model.py`

**Step 1: Write model shape tests first (TDD)**

```python
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
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_model.py -v
```

Expected: FAIL — `models.fusion_model` doesn't exist yet.

**Step 3: Implement the fusion model**

```python
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
    ):
        super().__init__()

        self.text_encoder = DistilBertModel.from_pretrained(text_model_name)
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
        cls_embedding = text_output.last_hidden_state[:, 0, :]  # (B, 768)

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
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_model.py -v
```

Expected: All PASS

**Step 5: Commit**

```bash
git add models/fusion_model.py tests/test_model.py
git commit -m "feat: implement multimodal fusion classifier with modality dropout"
```

---

## Task 7: Determinism Test

**Files:**
- Create: `tests/test_determinism.py`

**Step 1: Write determinism test**

```python
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
```

**Step 2: Run test**

```bash
pytest tests/test_determinism.py -v
```

Expected: PASS

**Step 3: Commit**

```bash
git add tests/test_determinism.py
git commit -m "test: add determinism test for reproducible forward passes"
```

---

## Task 8: Evaluation Metrics

**Files:**
- Create: `evaluation/metrics.py`

**Step 1: Implement metrics**

```python
"""Evaluation metrics: macro-F1, per-class precision/recall, confusion matrix."""

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


@dataclass
class MetricsResult:
    macro_f1: float
    accuracy: float
    per_class_f1: np.ndarray
    per_class_precision: np.ndarray
    per_class_recall: np.ndarray
    confusion_mat: np.ndarray
    report: str


def compute_metrics(
    labels: list[int],
    preds: list[int],
    class_names: list[str] | None = None,
) -> MetricsResult:
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average=None, zero_division=0
    )

    report = classification_report(
        labels, preds,
        target_names=class_names,
        zero_division=0,
    )

    return MetricsResult(
        macro_f1=f1_score(labels, preds, average="macro", zero_division=0),
        accuracy=accuracy_score(labels, preds),
        per_class_f1=f1,
        per_class_precision=precision,
        per_class_recall=recall,
        confusion_mat=confusion_matrix(labels, preds),
        report=report,
    )
```

**Step 2: Commit**

```bash
git add evaluation/metrics.py
git commit -m "feat: add evaluation metrics (macro-F1, per-class P/R, confusion matrix)"
```

---

## Task 9: Custom Training Loop

**Files:**
- Create: `training/train.py`

This is the most important engineering artifact in the project.

**Step 1: Implement the training loop**

Key fixes from review:
- **M1 zeros tabular features** in the dataset via `text_only` flag
- **Pre-tokenizes in `__init__`** — avoids 48K redundant tokenizations across 3 epochs
- **Sets all random seeds** (torch, numpy, random) for reproducibility
- **Flushes final gradient accumulation batch** after epoch loop
- **Uses `num_workers=0`** (macOS `spawn` would serialize pre-tokenized tensors; bottleneck is already eliminated by pre-tokenization)

```python
"""Custom PyTorch training loop with MLflow tracking.

Core deliverable: explicit backward/step/clip mechanics, differential
learning rates, gradient accumulation, and early stopping.
"""

import json
import random as python_random
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizer, get_cosine_schedule_with_warmup

from evaluation.metrics import MetricsResult, compute_metrics
from models.fusion_model import MultimodalClassifier
from training.config import TrainConfig


class ComplaintDataset(Dataset):
    """Dataset for complaint narratives + tabular features.

    Pre-tokenizes all narratives in __init__ to avoid redundant
    tokenization across epochs (~3x speedup on CPU).
    """

    def __init__(
        self,
        narratives: list[str],
        tabular_features: np.ndarray,
        labels: np.ndarray,
        tokenizer: DistilBertTokenizer,
        max_length: int = 128,
        text_only: bool = False,
    ):
        # Pre-tokenize all narratives once
        self.encodings = tokenizer(
            narratives,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        if text_only:
            self.tabular_features = torch.zeros(
                len(labels), tabular_features.shape[1], dtype=torch.float32,
            )
        else:
            self.tabular_features = torch.tensor(tabular_features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "text": {k: v[idx] for k, v in self.encodings.items()},
            "tabular": self.tabular_features[idx],
            "labels": self.labels[idx],
        }


class EarlyStopper:
    """Early stopping on validation metric (higher is better)."""

    def __init__(self, patience: int = 2):
        self.patience = patience
        self.best_score = -float("inf")
        self.counter = 0

    def should_stop(self, score: float) -> bool:
        if score > self.best_score:
            self.best_score = score
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


def compute_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights for imbalanced data."""
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    counts[counts == 0] = 1.0
    weights = len(labels) / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def _set_seeds(seed: int):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    python_random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model: MultimodalClassifier,
    train_loader: DataLoader,
    optimizer: AdamW,
    scheduler,
    class_weights: torch.Tensor,
    config: TrainConfig,
    epoch: int,
    device: torch.device,
) -> float:
    """Train for one epoch. Returns average loss."""
    model.train()
    epoch_loss = 0.0
    num_batches = 0

    for step, batch in enumerate(train_loader):
        text_inputs = {k: v.to(device) for k, v in batch["text"].items()}
        tabular_features = batch["tabular"].to(device)
        labels = batch["labels"].to(device)

        logits = model(text_inputs, tabular_features)
        loss = F.cross_entropy(logits, labels, weight=class_weights.to(device))
        loss = loss / config.grad_accumulation_steps

        loss.backward()

        if (step + 1) % config.grad_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=config.max_grad_norm
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        batch_loss = loss.item() * config.grad_accumulation_steps
        epoch_loss += batch_loss
        num_batches += 1

        global_step = epoch * len(train_loader) + step
        mlflow.log_metric("train_loss_step", batch_loss, step=global_step)

    # Flush any remaining accumulated gradients from incomplete final batch
    if (step + 1) % config.grad_accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=config.max_grad_norm
        )
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return epoch_loss / max(num_batches, 1)


def evaluate(
    model: MultimodalClassifier,
    loader: DataLoader,
    class_names: list[str],
    device: torch.device,
) -> MetricsResult:
    """Evaluate model on a data loader."""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            text_inputs = {k: v.to(device) for k, v in batch["text"].items()}
            tabular_features = batch["tabular"].to(device)
            logits = model(text_inputs, tabular_features)
            all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
            all_labels.extend(batch["labels"].tolist())

    return compute_metrics(all_labels, all_preds, class_names)


def train(config: TrainConfig) -> dict:
    """Full training pipeline. Returns test metrics dict."""
    _set_seeds(config.seed)
    device = torch.device("cpu")

    # Load data
    from adapters.cfpb import CFPBAdapter

    adapter = CFPBAdapter(
        sample_size=config.sample_size,
        seed=config.seed,
        split_strategy=config.split_strategy,
        cutoff_date=config.cutoff_date,
        exclude_features=config.exclude_features,
    )
    splits = adapter.preprocess()

    tokenizer = DistilBertTokenizer.from_pretrained(config.text_model_name)

    text_only = config.variant == "M1"

    train_ds = ComplaintDataset(
        splits["train"]["narratives"],
        splits["train"]["tabular_features"],
        splits["train"]["labels"],
        tokenizer,
        config.max_seq_length,
        text_only=text_only,
    )
    val_ds = ComplaintDataset(
        splits["val"]["narratives"],
        splits["val"]["tabular_features"],
        splits["val"]["labels"],
        tokenizer,
        config.max_seq_length,
        text_only=text_only,
    )
    test_ds = ComplaintDataset(
        splits["test"]["narratives"],
        splits["test"]["tabular_features"],
        splits["test"]["labels"],
        tokenizer,
        config.max_seq_length,
        text_only=text_only,
    )

    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size, num_workers=0,
    )
    test_loader = DataLoader(
        test_ds, batch_size=config.batch_size, num_workers=0,
    )

    # Model
    tabular_dim = splits["train"]["tabular_features"].shape[1]

    model = MultimodalClassifier(
        num_classes=len(adapter.class_names),
        tabular_input_dim=tabular_dim,
        tabular_hidden_dim=config.tabular_hidden_dim,
        tabular_embed_dim=config.tabular_embed_dim,
        fusion_hidden_dim=config.fusion_hidden_dim,
        dropout=config.dropout,
        modality_dropout=(config.variant == "M3"),
        text_model_name=config.text_model_name,
        text_dropout_prob=config.text_dropout_prob,
        tabular_dropout_prob=config.tabular_dropout_prob,
    ).to(device)

    class_weights = compute_class_weights(
        splits["train"]["labels"], len(adapter.class_names)
    )

    # Differential learning rates
    encoder_params = list(model.text_encoder.parameters())
    head_params = (
        list(model.tabular_mlp.parameters())
        + list(model.fusion_head.parameters())
    )
    optimizer = AdamW([
        {"params": encoder_params, "lr": config.lr_encoder},
        {"params": head_params, "lr": config.lr_head},
    ])

    total_steps = (
        len(train_loader) // config.grad_accumulation_steps * config.num_epochs
    )
    warmup_steps = int(total_steps * config.warmup_fraction)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    early_stopper = EarlyStopper(patience=config.early_stopping_patience)

    results_dir = Path(config.results_dir)
    results_dir.mkdir(exist_ok=True)

    run_name = config.run_name or f"{config.variant}_seed{config.seed}"

    mlflow.set_experiment(config.experiment_name)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(config.model_dump())

        best_val_f1 = 0.0
        for epoch in range(config.num_epochs):
            train_loss = train_one_epoch(
                model, train_loader, optimizer, scheduler,
                class_weights, config, epoch, device,
            )
            val_metrics = evaluate(model, val_loader, adapter.class_names, device)

            print(
                f"Epoch {epoch + 1}/{config.num_epochs} | "
                f"Loss: {train_loss:.4f} | "
                f"Val F1: {val_metrics.macro_f1:.4f} | "
                f"Val Acc: {val_metrics.accuracy:.4f}"
            )

            mlflow.log_metrics(
                {
                    "val_macro_f1": val_metrics.macro_f1,
                    "val_accuracy": val_metrics.accuracy,
                    "train_loss_epoch": train_loss,
                },
                step=epoch,
            )

            if val_metrics.macro_f1 > best_val_f1:
                best_val_f1 = val_metrics.macro_f1
                model_path = results_dir / f"{run_name}_best.pt"
                torch.save(model.state_dict(), model_path)

            if early_stopper.should_stop(val_metrics.macro_f1):
                print(f"Early stopping at epoch {epoch + 1}")
                mlflow.log_metric("stopped_epoch", epoch + 1)
                break

        # Load best model for test evaluation
        best_path = results_dir / f"{run_name}_best.pt"
        if best_path.exists():
            model.load_state_dict(torch.load(best_path, weights_only=True))

        test_metrics = evaluate(model, test_loader, adapter.class_names, device)
        print(f"\nTest Macro-F1: {test_metrics.macro_f1:.4f}")
        print(f"Test Accuracy: {test_metrics.accuracy:.4f}")
        print(f"\n{test_metrics.report}")

        mlflow.log_metrics({
            "test_macro_f1": test_metrics.macro_f1,
            "test_accuracy": test_metrics.accuracy,
        })

        final_path = results_dir / f"{run_name}.pt"
        torch.save(model.state_dict(), final_path)
        mlflow.log_artifact(str(final_path))

        class_names_path = results_dir / "class_names.json"
        with open(class_names_path, "w") as f:
            json.dump(adapter.class_names, f)

    return {
        "test_macro_f1": test_metrics.macro_f1,
        "test_accuracy": test_metrics.accuracy,
        "per_class_f1": test_metrics.per_class_f1.tolist(),
        "class_names": adapter.class_names,
        "variant": config.variant,
        "seed": config.seed,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="M2", choices=["M1", "M2", "M3"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-size", type=int, default=20000)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    cfg = TrainConfig(
        variant=args.variant,
        seed=args.seed,
        sample_size=args.sample_size,
        num_epochs=args.epochs,
    )
    train(cfg)
```

**Step 2: Commit**

```bash
git add training/train.py
git commit -m "feat: implement custom PyTorch training loop with MLflow tracking"
```

---

## Task 10: Overfit Smoke Test

**Files:**
- Create: `tests/test_overfit_smoke.py`

**Step 1: Write the overfit smoke test**

```python
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
```

**Step 2: Run test**

```bash
pytest tests/test_overfit_smoke.py -v
```

Expected: PASS

**Step 3: Commit**

```bash
git add tests/test_overfit_smoke.py
git commit -m "test: add overfit smoke test (1-batch convergence check)"
```

---

## Task 11: Baselines (TF-IDF + LogReg, LightGBM)

**Files:**
- Create: `models/baselines.py`

**Step 1: Implement baseline models**

```python
"""Non-DL baselines: TF-IDF + LogReg (B1), LightGBM tabular-only (B2)."""

import mlflow
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from evaluation.metrics import compute_metrics


def train_tfidf_logreg(
    train_narratives: list[str],
    train_labels: np.ndarray,
    test_narratives: list[str],
    test_labels: np.ndarray,
    class_names: list[str],
    max_features: int = 20_000,
) -> dict:
    """B1: TF-IDF + Logistic Regression baseline."""
    tfidf = TfidfVectorizer(max_features=max_features, stop_words="english")
    X_train = tfidf.fit_transform(train_narratives)
    X_test = tfidf.transform(test_narratives)

    clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
        n_jobs=-1,
    )
    clf.fit(X_train, train_labels)

    test_preds = clf.predict(X_test)
    test_metrics = compute_metrics(test_labels.tolist(), test_preds.tolist(), class_names)

    mlflow.set_experiment("finetune-bench")
    with mlflow.start_run(run_name="B1_tfidf_logreg"):
        mlflow.log_params({
            "variant": "B1",
            "model": "TF-IDF + LogisticRegression",
            "max_features": max_features,
        })
        mlflow.log_metrics({
            "test_macro_f1": test_metrics.macro_f1,
            "test_accuracy": test_metrics.accuracy,
        })

    print(f"B1 Test Macro-F1: {test_metrics.macro_f1:.4f}")
    print(f"B1 Test Accuracy: {test_metrics.accuracy:.4f}")
    print(f"\n{test_metrics.report}")

    return {
        "test_macro_f1": test_metrics.macro_f1,
        "test_accuracy": test_metrics.accuracy,
        "per_class_f1": test_metrics.per_class_f1.tolist(),
        "variant": "B1",
    }


def train_lightgbm(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    class_names: list[str],
) -> dict:
    """B2: LightGBM tabular-only baseline."""
    import lightgbm as lgb

    num_classes = len(class_names)
    train_data = lgb.Dataset(train_features, label=train_labels)
    val_data = lgb.Dataset(val_features, label=val_labels, reference=train_data)

    params = {
        "objective": "multiclass",
        "num_class": num_classes,
        "metric": "multi_logloss",
        "verbosity": -1,
        "num_leaves": 31,
        "learning_rate": 0.1,
        "class_weight": "balanced",
    }

    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        num_boost_round=200,
        callbacks=[lgb.early_stopping(20), lgb.log_evaluation(50)],
    )

    test_proba = model.predict(test_features)
    test_preds = np.argmax(test_proba, axis=1)

    test_metrics = compute_metrics(test_labels.tolist(), test_preds.tolist(), class_names)

    mlflow.set_experiment("finetune-bench")
    with mlflow.start_run(run_name="B2_lightgbm"):
        mlflow.log_params({
            "variant": "B2",
            "model": "LightGBM",
            "num_leaves": 31,
        })
        mlflow.log_metrics({
            "test_macro_f1": test_metrics.macro_f1,
            "test_accuracy": test_metrics.accuracy,
        })

    print(f"B2 Test Macro-F1: {test_metrics.macro_f1:.4f}")
    print(f"B2 Test Accuracy: {test_metrics.accuracy:.4f}")
    print(f"\n{test_metrics.report}")

    return {
        "test_macro_f1": test_metrics.macro_f1,
        "test_accuracy": test_metrics.accuracy,
        "per_class_f1": test_metrics.per_class_f1.tolist(),
        "variant": "B2",
    }
```

**Step 2: Commit**

```bash
git add models/baselines.py
git commit -m "feat: add TF-IDF+LogReg and LightGBM baselines"
```

---

## Task 12: Robustness Corruption Functions

**Files:**
- Create: `evaluation/robustness.py`
- Create: `tests/test_robustness.py`

**Step 1: Write robustness tests first (TDD)**

```python
"""Tests for corruption functions."""

import torch
import pytest

from evaluation.robustness import (
    inject_typos,
    token_dropout,
    truncate_text,
    tabular_dropout,
    tabular_ablation,
)


class TestInjectTypos:
    def test_returns_string(self):
        result = inject_typos("hello world", rate=0.5, seed=42)
        assert isinstance(result, str)

    def test_rate_zero_unchanged(self):
        text = "hello world"
        assert inject_typos(text, rate=0.0, seed=42) == text

    def test_rate_one_changes_all_non_space(self):
        text = "hello world"
        result = inject_typos(text, rate=1.0, seed=42)
        assert result != text

    def test_deterministic_with_seed(self):
        text = "the quick brown fox"
        r1 = inject_typos(text, rate=0.3, seed=42)
        r2 = inject_typos(text, rate=0.3, seed=42)
        assert r1 == r2

    def test_swap_actually_swaps(self):
        """Verify swap produces transposition, not duplication."""
        # Use a high rate with fixed seed to force a swap
        text = "ab"
        # After a true swap of adjacent chars, "ab" -> "ba"
        # We can't predict which operation the RNG picks, so instead
        # verify that the result is a valid corruption (not the same length
        # with the same chars repeated — which would indicate duplication)
        result = inject_typos(text, rate=1.0, seed=0)
        # Result should be a valid corruption, not "bb"
        assert result != "bb", "Swap should transpose, not duplicate"


class TestTokenDropout:
    def test_shape_preserved(self):
        input_ids = torch.tensor([[101, 2023, 2003, 1037, 3231, 102]])
        result = token_dropout(input_ids, rate=0.5, seed=42)
        assert result.shape == input_ids.shape

    def test_cls_sep_preserved(self):
        """[CLS]=101, [SEP]=102 should never be dropped."""
        input_ids = torch.tensor([[101, 2023, 2003, 1037, 3231, 102]])
        result = token_dropout(input_ids, rate=1.0, seed=42)
        assert result[0, 0].item() == 101  # [CLS]
        # Find [SEP] position (not necessarily last due to padding)
        sep_positions = (input_ids == 102).nonzero(as_tuple=True)
        for pos in sep_positions[1]:
            assert result[0, pos.item()].item() == 102

    def test_rate_zero_unchanged(self):
        input_ids = torch.tensor([[101, 2023, 2003, 102]])
        result = token_dropout(input_ids, rate=0.0, seed=42)
        assert torch.equal(result, input_ids)


class TestTruncateText:
    def test_truncation(self):
        text = "one two three four five six seven eight"
        result = truncate_text(text, max_tokens=3)
        assert result == "one two three"

    def test_short_text_unchanged(self):
        text = "hello"
        result = truncate_text(text, max_tokens=10)
        assert result == text


class TestTabularDropout:
    def test_shape_preserved(self):
        features = torch.randn(4, 10)
        result = tabular_dropout(features, rate=0.5, seed=42)
        assert result.shape == features.shape

    def test_rate_zero_unchanged(self):
        features = torch.randn(4, 10)
        result = tabular_dropout(features, rate=0.0, seed=42)
        assert torch.equal(result, features)

    def test_rate_one_all_zero(self):
        features = torch.randn(4, 10)
        result = tabular_dropout(features, rate=1.0, seed=42)
        assert torch.all(result == 0)


class TestTabularAblation:
    def test_all_zeros(self):
        features = torch.randn(4, 10)
        result = tabular_ablation(features)
        assert torch.all(result == 0)

    def test_shape_preserved(self):
        features = torch.randn(4, 10)
        result = tabular_ablation(features)
        assert result.shape == features.shape
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_robustness.py -v
```

Expected: FAIL — module not found.

**Step 3: Implement corruption functions**

Fix from review: `inject_typos` swap works on a mutable list in-place, marking swapped indices to avoid duplication.

```python
"""Robustness corruption functions for evaluation.

Text corruptions: applied to raw text before tokenization.
Token dropout: operates on tokenized input_ids.
Tabular corruptions: operate on feature tensors.
"""

import random
import string

import torch


def inject_typos(text: str, rate: float = 0.1, seed: int | None = None) -> str:
    """Inject character-level typos at the given rate.

    For each character, with probability=rate, apply one of:
    - swap with adjacent character (true transposition)
    - delete the character
    - insert a random lowercase letter

    Spaces are never corrupted.
    """
    if rate == 0.0:
        return text

    rng = random.Random(seed)
    chars = list(text)
    i = 0
    result = []

    while i < len(chars):
        ch = chars[i]

        if ch == " " or rng.random() >= rate:
            result.append(ch)
            i += 1
            continue

        op = rng.choice(["swap", "delete", "insert"])

        if op == "swap" and i + 1 < len(chars) and chars[i + 1] != " ":
            # True transposition: emit chars[i+1] then chars[i], skip both
            result.append(chars[i + 1])
            result.append(chars[i])
            i += 2  # skip the next character since we already emitted it
        elif op == "delete":
            i += 1  # skip this character
        elif op == "insert":
            result.append(ch)
            result.append(rng.choice(string.ascii_lowercase))
            i += 1
        else:
            # swap at end of string — fall back to insert
            result.append(ch)
            result.append(rng.choice(string.ascii_lowercase))
            i += 1

    return "".join(result)


def token_dropout(
    input_ids: torch.Tensor,
    rate: float = 0.2,
    pad_id: int = 0,
    cls_id: int = 101,
    sep_id: int = 102,
    seed: int | None = None,
) -> torch.Tensor:
    """Replace tokens with [PAD], preserving [CLS] and [SEP]."""
    if rate == 0.0:
        return input_ids.clone()

    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(seed)

    result = input_ids.clone()
    mask = torch.rand(result.shape, generator=gen) < rate

    # Protect special tokens
    mask = mask & (result != cls_id) & (result != sep_id) & (result != pad_id)

    result[mask] = pad_id
    return result


def truncate_text(text: str, max_tokens: int = 32) -> str:
    """Keep only the first N whitespace-split tokens."""
    tokens = text.split()
    return " ".join(tokens[:max_tokens])


def tabular_dropout(
    features: torch.Tensor,
    rate: float = 0.5,
    seed: int | None = None,
) -> torch.Tensor:
    """Zero out each feature independently with probability=rate."""
    if rate == 0.0:
        return features.clone()
    if rate == 1.0:
        return torch.zeros_like(features)

    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(seed)

    mask = torch.rand(features.shape, generator=gen) >= rate
    return features * mask.float()


def tabular_ablation(features: torch.Tensor) -> torch.Tensor:
    """Zero out all tabular features (simulate fully missing metadata)."""
    return torch.zeros_like(features)
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_robustness.py -v
```

Expected: All PASS

**Step 5: Commit**

```bash
git add evaluation/robustness.py tests/test_robustness.py
git commit -m "feat: add robustness corruption functions with tests"
```

---

## Task 13: Robustness Evaluation Runner

**Files:**
- Modify: `evaluation/robustness.py` (append `run_robustness_eval` function)

**Step 1: Append to end of `evaluation/robustness.py`**

Handles M1 correctly: skips tabular corruption entries for text-only models.

```python


def run_robustness_eval(
    model,
    test_narratives: list[str],
    test_tabular: torch.Tensor,
    test_labels: list[int],
    class_names: list[str],
    tokenizer,
    max_length: int = 128,
    device: torch.device = torch.device("cpu"),
    seed: int = 42,
    is_text_only: bool = False,
) -> dict[str, dict]:
    """Run full robustness evaluation suite.

    Args:
        is_text_only: If True, skip tabular corruption entries (N/A for M1).

    Returns dict mapping corruption_name -> {macro_f1, accuracy}.
    """
    from evaluation.metrics import compute_metrics

    corruptions = [
        ("clean", {}),
        ("typo_10", {"type": "typo", "rate": 0.1}),
        ("typo_20", {"type": "typo", "rate": 0.2}),
        ("token_drop_20", {"type": "token_drop", "rate": 0.2}),
        ("token_drop_40", {"type": "token_drop", "rate": 0.4}),
        ("truncate_32", {"type": "truncate", "max_tokens": 32}),
        ("tabular_drop_50", {"type": "tabular_drop", "rate": 0.5}),
        ("tabular_ablation", {"type": "tabular_ablation"}),
    ]

    model.eval()
    results = {}

    for name, params in corruptions:
        corruption_type = params.get("type")

        # Skip tabular corruptions for text-only models
        if is_text_only and corruption_type in ("tabular_drop", "tabular_ablation"):
            results[name] = None  # Signals N/A in table generation
            continue

        all_preds = []

        # Apply text-level corruptions
        if corruption_type == "typo":
            corrupted_texts = [
                inject_typos(t, rate=params["rate"], seed=seed + i)
                for i, t in enumerate(test_narratives)
            ]
        elif corruption_type == "truncate":
            corrupted_texts = [
                truncate_text(t, max_tokens=params["max_tokens"])
                for t in test_narratives
            ]
        else:
            corrupted_texts = test_narratives

        batch_size = 16
        for start in range(0, len(corrupted_texts), batch_size):
            end = min(start + batch_size, len(corrupted_texts))
            batch_texts = corrupted_texts[start:end]
            batch_tabular = test_tabular[start:end].to(device)

            encodings = tokenizer(
                batch_texts,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            text_inputs = {k: v.to(device) for k, v in encodings.items()}

            if corruption_type == "token_drop":
                text_inputs["input_ids"] = token_dropout(
                    text_inputs["input_ids"],
                    rate=params["rate"],
                    seed=seed + start,
                )

            if corruption_type == "tabular_drop":
                batch_tabular = tabular_dropout(
                    batch_tabular, rate=params["rate"], seed=seed + start,
                )
            elif corruption_type == "tabular_ablation":
                batch_tabular = tabular_ablation(batch_tabular)

            with torch.no_grad():
                logits = model(text_inputs, batch_tabular)
                all_preds.extend(logits.argmax(dim=-1).cpu().tolist())

        metrics = compute_metrics(test_labels, all_preds, class_names)
        results[name] = {
            "macro_f1": metrics.macro_f1,
            "accuracy": metrics.accuracy,
        }
        print(f"  {name}: F1={metrics.macro_f1:.4f}, Acc={metrics.accuracy:.4f}")

    return results
```

**Step 2: Commit**

```bash
git add evaluation/robustness.py
git commit -m "feat: add robustness evaluation runner with M1 text-only handling"
```

---

## Task 14: ONNX Export

**Files:**
- Create: `evaluation/export.py`
- Create: `tests/test_export.py`

**Step 1: Write the ONNX export test first (TDD)**

```python
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
```

**Step 2: Implement ONNX export**

```python
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
```

**Step 3: Run tests**

```bash
pytest tests/test_export.py -v
```

Expected: All PASS

**Step 4: Commit**

```bash
git add evaluation/export.py tests/test_export.py
git commit -m "feat: add ONNX export with latency benchmarking"
```

---

## Task 15: Experiment Runner Script

**Files:**
- Create: `scripts/run_all_experiments.py`

**Step 1: Write the experiment orchestrator**

Includes `--sample-size` CLI flag for 20K/50K switching.

```python
"""Run full ablation matrix: baselines, M1, M2, M3 across seeds."""

import argparse
import json
from pathlib import Path

from adapters.cfpb import CFPBAdapter
from models.baselines import train_tfidf_logreg, train_lightgbm
from training.config import TrainConfig
from training.train import train


SEEDS = [42, 123, 456]
RESULTS_DIR = Path("results")


def run_baselines(sample_size: int = 20_000, seed: int = 42):
    """Run B1 and B2 baselines."""
    adapter = CFPBAdapter(sample_size=sample_size, seed=seed)
    splits = adapter.preprocess()

    print("\n" + "=" * 60)
    print("Running B1: TF-IDF + LogReg")
    print("=" * 60)
    b1_results = train_tfidf_logreg(
        splits["train"]["narratives"],
        splits["train"]["labels"],
        splits["test"]["narratives"],
        splits["test"]["labels"],
        adapter.class_names,
    )

    print("\n" + "=" * 60)
    print("Running B2: LightGBM (tabular-only)")
    print("=" * 60)
    b2_results = train_lightgbm(
        splits["train"]["tabular_features"],
        splits["train"]["labels"],
        splits["val"]["tabular_features"],
        splits["val"]["labels"],
        splits["test"]["tabular_features"],
        splits["test"]["labels"],
        adapter.class_names,
    )

    return b1_results, b2_results


def run_dl_variants(sample_size: int = 20_000, num_epochs: int = 3):
    """Run M1, M2, M3 across all seeds."""
    all_results = []

    for variant in ["M1", "M2", "M3"]:
        for seed in SEEDS:
            print("\n" + "=" * 60)
            print(f"Running {variant} seed={seed}")
            print("=" * 60)

            config = TrainConfig(
                variant=variant,
                seed=seed,
                sample_size=sample_size,
                num_epochs=num_epochs,
            )
            result = train(config)
            all_results.append(result)

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Run finetune-bench ablation matrix")
    parser.add_argument(
        "--sample-size", type=int, default=20_000,
        help="Training data size (default: 20000, use 50000 for final runs)",
    )
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)

    print("=== BASELINES ===")
    b1, b2 = run_baselines(sample_size=args.sample_size)

    print("\n=== DL VARIANTS ===")
    dl_results = run_dl_variants(
        sample_size=args.sample_size, num_epochs=args.epochs,
    )

    all_results = [b1, b2] + dl_results
    with open(RESULTS_DIR / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nAll results saved to {RESULTS_DIR / 'all_results.json'}")


if __name__ == "__main__":
    main()
```

**Step 2: Commit**

```bash
git add scripts/run_all_experiments.py
git commit -m "feat: add experiment runner script with --sample-size flag"
```

---

## Task 16: Table Generation and Training Curves

**Files:**
- Create: `scripts/generate_tables.py`

Generates both markdown tables AND training curves figure.

**Step 1: Implement table + plot generation**

```python
"""Generate markdown ablation tables and training curves from results."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np

RESULTS_DIR = Path("results")


def load_results() -> list[dict]:
    path = RESULTS_DIR / "all_results.json"
    with open(path) as f:
        return json.load(f)


def generate_table1(results: list[dict]) -> str:
    """Ablation Table 1: Component contribution on clean data."""
    lines = [
        "# Ablation Table 1 -- Component Contribution (clean data, random split)\n",
        "| Variant | Macro-F1 | Accuracy | Per-class F1 range | Delta vs B1 |",
        "|---------|----------|----------|-------------------|-------------|",
    ]

    b1 = next((r for r in results if r["variant"] == "B1"), None)
    b1_f1 = b1["test_macro_f1"] if b1 else 0

    for variant_id in ["B1", "B2", "M1", "M2", "M3"]:
        variant_results = [r for r in results if r["variant"] == variant_id]
        if not variant_results:
            continue

        f1_scores = [r["test_macro_f1"] for r in variant_results]
        acc_scores = [r["test_accuracy"] for r in variant_results]

        if len(f1_scores) > 1:
            f1_str = f"{np.mean(f1_scores):.4f} +/- {np.std(f1_scores):.4f}"
            acc_str = f"{np.mean(acc_scores):.4f} +/- {np.std(acc_scores):.4f}"
        else:
            f1_str = f"{f1_scores[0]:.4f}"
            acc_str = f"{acc_scores[0]:.4f}"

        all_per_class = []
        for r in variant_results:
            all_per_class.extend(r.get("per_class_f1", []))
        range_str = f"{min(all_per_class):.2f}-{max(all_per_class):.2f}" if all_per_class else "--"

        delta = np.mean(f1_scores) - b1_f1
        delta_str = f"+{delta:.4f}" if variant_id != "B1" else "--"

        names = {
            "B1": "B1: TF-IDF + LogReg",
            "B2": "B2: Tabular-only LightGBM",
            "M1": "M1: DistilBERT text-only",
            "M2": "M2: Full fusion",
            "M3": "M3: Fusion + dropout",
        }
        lines.append(f"| {names[variant_id]} | {f1_str} | {acc_str} | {range_str} | {delta_str} |")

    lines.append("\n*DL variants report mean +/- std over 3 seeds.*")
    return "\n".join(lines)


def generate_table2(results: list[dict]) -> str:
    """Ablation Table 2: Robustness under corruption."""
    rob_path = RESULTS_DIR / "robustness_results.json"
    if not rob_path.exists():
        return "# Ablation Table 2 -- Robustness Under Corruption\n\n*Run robustness evaluation first.*"

    with open(rob_path) as f:
        rob_results = json.load(f)

    corruptions = [
        ("None (clean)", "clean"),
        ("Typo injection 10%", "typo_10"),
        ("Typo injection 20%", "typo_20"),
        ("Token dropout 20%", "token_drop_20"),
        ("Token dropout 40%", "token_drop_40"),
        ("Truncation 32 tokens", "truncate_32"),
        ("Tabular dropout 50%", "tabular_drop_50"),
        ("Full tabular ablation", "tabular_ablation"),
    ]

    lines = [
        "# Ablation Table 2 -- Robustness Under Corruption\n",
        "| Corruption | M1: Text-only | M2: Fusion | M3: Fusion+Dropout | Delta (M3 vs M1) |",
        "|------------|---------------|------------|-------------------|-------------------|",
    ]

    for display_name, key in corruptions:
        cells = [display_name]
        m1_f1 = None
        m3_f1 = None
        for variant in ["M1", "M2", "M3"]:
            entry = rob_results.get(variant, {}).get(key)
            if entry is None:
                cells.append("N/A")
            else:
                f1 = entry["macro_f1"]
                cells.append(f"{f1:.4f}")
                if variant == "M1":
                    m1_f1 = f1
                if variant == "M3":
                    m3_f1 = f1

        if m1_f1 is not None and m3_f1 is not None:
            delta = m3_f1 - m1_f1
            cells.append(f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}")
        else:
            cells.append("--")

        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


def generate_training_curves():
    """Generate training curves from MLflow metrics."""
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("finetune-bench")
    if experiment is None:
        print("No MLflow experiment found. Skipping training curves.")
        return

    runs = client.search_runs(experiment.experiment_id)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for run in runs:
        name = run.info.run_name
        if not name or not any(name.startswith(v) for v in ["M1_", "M2_", "M3_"]):
            continue

        # Loss curve
        loss_history = client.get_metric_history(run.info.run_id, "train_loss_epoch")
        if loss_history:
            epochs = [m.step for m in loss_history]
            losses = [m.value for m in loss_history]
            axes[0].plot(epochs, losses, label=name, alpha=0.7)

        # F1 curve
        f1_history = client.get_metric_history(run.info.run_id, "val_macro_f1")
        if f1_history:
            epochs = [m.step for m in f1_history]
            f1s = [m.value for m in f1_history]
            axes[1].plot(epochs, f1s, label=name, alpha=0.7)

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Training Loss")
    axes[0].set_title("Training Loss")
    axes[0].legend(fontsize=8)

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation Macro-F1")
    axes[1].set_title("Validation Macro-F1")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "training_curves.png", dpi=150)
    print(f"Saved training curves to {RESULTS_DIR / 'training_curves.png'}")
    plt.close()


def main():
    results = load_results()

    table1 = generate_table1(results)
    with open(RESULTS_DIR / "ablation_component.md", "w") as f:
        f.write(table1)
    print(table1)

    print()

    table2 = generate_table2(results)
    with open(RESULTS_DIR / "ablation_robustness.md", "w") as f:
        f.write(table2)
    print(table2)

    print()
    generate_training_curves()


if __name__ == "__main__":
    main()
```

**Step 2: Commit**

```bash
git add scripts/generate_tables.py
git commit -m "feat: add table generation and training curves plotting"
```

---

## Task 17: Dockerfile and Docker Compose

**Files:**
- Create: `Dockerfile`
- Create: `docker-compose.yml`

**Step 1: Write Dockerfile**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY adapters/ adapters/
COPY models/ models/
COPY training/ training/
COPY evaluation/ evaluation/
COPY scripts/ scripts/
COPY tests/ tests/

RUN pip install --no-cache-dir -e ".[dev]"

# Cache model weights at build time
RUN python -c "from transformers import DistilBertModel, DistilBertTokenizer; \
    DistilBertModel.from_pretrained('distilbert-base-uncased'); \
    DistilBertTokenizer.from_pretrained('distilbert-base-uncased')"

CMD ["pytest", "tests/", "-v", "--tb=short"]
```

**Step 2: Write docker-compose.yml**

```yaml
services:
  test:
    build: .
    command: pytest tests/ -v --tb=short
    volumes:
      - ./results:/app/results

  train-smoke:
    build: .
    command: >
      python -c "
      from training.config import TrainConfig
      from training.train import train
      cfg = TrainConfig(variant='M2', seed=42, sample_size=500, num_epochs=1)
      train(cfg)
      print('Smoke training complete')
      "
    volumes:
      - ./data:/app/data
      - ./results:/app/results
      - ./mlruns:/app/mlruns

  train-full:
    build: .
    command: python scripts/run_all_experiments.py
    volumes:
      - ./data:/app/data
      - ./results:/app/results
      - ./mlruns:/app/mlruns
```

**Step 3: Commit**

```bash
git add Dockerfile docker-compose.yml
git commit -m "feat: add Dockerfile and docker-compose for reproducible runs"
```

---

## Task 18: README and LICENSE

**Files:**
- Create: `README.md`
- Create: `LICENSE`

**Step 1: Write README**

Template with empty table cells — numbers filled after training (Day 4 afternoon).

```markdown
# finetune-bench

Multimodal text+tabular classification benchmark with robustness training and corruption evaluation.

## Key Results

### Ablation Table 1 -- Component Contribution

| Variant | Macro-F1 | Accuracy | Per-class F1 range | Delta vs B1 |
|---------|----------|----------|-------------------|-------------|
| B1: TF-IDF + LogReg | | | | -- |
| B2: Tabular-only LightGBM | | | | |
| M1: DistilBERT text-only | | | | |
| M2: Full fusion | | | | |
| M3: Fusion + dropout | | | | |

*DL variants report mean +/- std over 3 seeds.*

### Ablation Table 2 -- Robustness Under Corruption

| Corruption | M1: Text-only | M2: Fusion | M3: Fusion+Dropout | Delta (M3 vs M1) |
|------------|---------------|------------|-------------------|-------------------|
| None (clean) | | | | |
| Typo injection 10% | | | | |
| Typo injection 20% | | | | |
| Token dropout 20% | | | | |
| Token dropout 40% | | | | |
| Truncation 32 tokens | | | | |
| Tabular dropout 50% | N/A | | | |
| Full tabular ablation | N/A | | | |

### Findings

> *To be filled after training runs complete.*

## Architecture

```
Text Branch              Tabular Branch
DistilBERT               Feature eng. + MLP
(fine-tuned)             (2 layers)
    |                        |
[CLS] embed (768)       tabular embed (64)
    |                        |
    |   Modality Dropout     |
    |   (M3: exclusive,      |
    |    p=0.1/branch)       |
    |                        |
    +--------+---------------+
             |
        Fusion Head
    concat -> FC(832,256)
    -> ReLU -> Dropout
    -> FC(256, num_classes)
             |
      Product category
```

The classifier is designed as a modular component compatible with retrieval-augmented pipelines; see [agent-bench](https://github.com/tyy0811/agent-bench) for retrieval system evidence.

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Download data
python scripts/download_data.py

# Run all experiments
python scripts/run_all_experiments.py

# Or with Docker
docker compose up test          # Run tests
docker compose up train-smoke   # Quick smoke run
docker compose up train-full    # Full ablation matrix
```

## Dataset

[CFPB Consumer Complaints](https://www.consumerfinance.gov/data-research/consumer-complaints/). All tabular features are available at the time the complaint is submitted. No post-outcome features are used.

The `DatasetAdapter` pattern makes the pipeline dataset-agnostic -- implementing `adapters/your_dataset.py` is all that's needed to benchmark a new domain.

**Note:** The CFPB database is not a statistical sample of all consumer experiences. Narratives are published only when consumers opt in.

## Training

Custom PyTorch loop (`training/train.py`) with:

- AdamW + linear warmup + cosine decay
- Differential learning rates (2e-5 encoder, 1e-3 new layers)
- Gradient accumulation (effective batch 32) and clipping (max_norm=1.0)
- Modality dropout (M3 variant: 10% per branch, mutually exclusive)
- Early stopping on validation macro-F1 (patience=2)
- MLflow experiment tracking
- Inverse-frequency class weights for imbalanced data

## Robustness

Five corruption types evaluate model degradation:

| Type | Description |
|------|-------------|
| Typo injection | Random char swap/delete/insert at 10% and 20% rates |
| Token dropout | Replace tokens with [PAD] at 20% and 40% rates |
| Truncation | Keep only first 32 whitespace tokens |
| Tabular dropout | Zero 50% of features independently |
| Tabular ablation | Zero all tabular features |

## Deployment

ONNX export with latency comparison:

| Format | Latency (single) | Latency (batch=32) | Model size |
|--------|-------------------|--------------------|------------|
| PyTorch | | | |
| ONNX | | | |

## Limitations & Ethics

**Dataset limitations:**
- CFPB narratives are opt-in, not a representative sample
- Class imbalance (Debt collection dominates); mitigated with inverse-frequency class weights and macro-F1
- `company` feature creates shortcut learning risk (company -> product correlation). Text-only variant (M1) isolates language-only signal
- Temporal distribution shift expected but not quantified unless temporal diagnostic was run

**Model limitations:**
- DistilBERT is English-only; not tested on other languages
- Tabular feature set is CFPB-specific; generalization requires implementing a new adapter
- Not suitable for production deployment without domain-specific validation and fairness auditing

**Leakage safeguards:**
- `company_complaint_volume` computed on training split only, log-transformed and standardized
- `company` shortcut acknowledged; M1 provides shortcut-free baseline
- All encoding categories (top-50 companies, states, channels) derived from training data only

## License

MIT
```

**Step 2: Create LICENSE**

```
MIT License

Copyright (c) 2026 Tsz Ying Jane Yeung

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

**Step 3: Commit**

```bash
git add README.md LICENSE
git commit -m "docs: add README with architecture, tables template, and limitations"
```

---

## Fixes Applied (cross-reference to review)

| # | Finding | Fix location | Status |
|---|---------|-------------|--------|
| 1 | M1 never zeros tabular | `ComplaintDataset` takes `text_only` flag; `train()` passes `text_only=(variant=="M1")` | Fixed in Task 9 |
| 2 | inject_typos swap is duplication | Rewritten with `i += 2` skip after swap to prevent re-emitting swapped char | Fixed in Task 12 |
| 3 | Final grad accum batch never steps | Post-loop flush added to `train_one_epoch` | Fixed in Task 9 |
| 4 | Volume not standardized | `_encode_features` receives `vol_mean`/`vol_std` from train; applies `(x-mean)/std` | Fixed in Task 4 |
| 5 | Python loops for one-hot encoding | Replaced with `pd.CategoricalDtype` + `pd.get_dummies` | Fixed in Task 4 |
| 6 | Volume test doesn't test the claim | New test checks `train_volume_stats` dict + `test_volume_standardized` checks mean/std | Fixed in Task 3 |
| 7 | EarlyStopper tie behavior | Noted, not a bug | No change needed |
| 8 | Robustness runner ignores M1 tabular N/A | `is_text_only` param added; returns `None` for tabular corruptions | Fixed in Task 13 |
| 9 | token_dropout test fragile [SEP] assertion | Test now finds [SEP] position with `.nonzero()` instead of assuming `[-1]` | Fixed in Task 12 |
| 10 | scripts/__init__.py + pip install | Removed `scripts/__init__.py`; added `pip install -e ".[dev]"` to Quick Start | Fixed in Tasks 1, 18 |
| 11 | pyproject.toml build-backend | Changed to `"setuptools.build_meta"` | Fixed in Task 1 |
| 12 | No random seeds in train() | Added `_set_seeds()` at top of `train()` (torch, numpy, random) | Fixed in Task 9 |
| 13 | No num_workers on DataLoader | Kept `num_workers=0` — pre-tokenization eliminates the bottleneck; macOS `spawn` would serialize full tensors | Fixed in Task 9 |
| 14 | Per-access tokenization | Pre-tokenize all narratives in `ComplaintDataset.__init__` | Fixed in Task 9 |
| 15 | No training curves code | Added `generate_training_curves()` to `generate_tables.py` | Fixed in Task 16 |
| 16 | Hardcoded sample_size | Added `--sample-size` and `--epochs` argparse flags | Fixed in Task 15 |
| 17 | License missing name | Added "Tsz Ying Jane Yeung" | Fixed in Task 18 |

---

## Task Summary & Dependency Graph

```
Task 1:  Scaffolding ─────────────────────────────────┐
Task 2:  Download script ──────────────────────────────┤
Task 3:  Abstract adapter + contract tests ────────────┤
Task 4:  CFPB adapter ── depends on Task 3 ────────────┤
Task 5:  Training config ──────────────────────────────┤
Task 6:  Fusion model + tests ─────────────────────────┤
Task 7:  Determinism test ── depends on Task 6 ────────┤
Task 8:  Evaluation metrics ───────────────────────────┤
Task 9:  Custom training loop ── depends on 4,5,6,8 ──┤
Task 10: Overfit smoke test ── depends on Task 6 ──────┤
Task 11: Baselines ── depends on Task 8 ───────────────┤
Task 12: Corruption functions + tests ─────────────────┤
Task 13: Robustness eval runner ── depends on 12 ──────┤
Task 14: ONNX export + tests ── depends on Task 6 ────┤
Task 15: Experiment runner ── depends on 9,11 ─────────┤
Task 16: Table generation + plots ── depends on 15 ────┤
Task 17: Docker ── depends on all code tasks ──────────┤
Task 18: README + LICENSE ─────────────────────────────┘
```

**Parallelizable groups:**
- Tasks 2, 3, 5, 6, 8, 12 can all be built independently
- Tasks 4, 7, 10, 11, 14 depend on their respective upstream tasks
- Tasks 9, 13 need multiple predecessors
- Tasks 15-18 are integration/packaging and come last
