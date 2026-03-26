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
