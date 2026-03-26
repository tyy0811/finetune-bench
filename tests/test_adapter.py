"""Tests for DatasetAdapter contract.

These tests require:
1. adapters/cfpb.py to be implemented (Task 4)
2. CFPB data downloaded to data/complaints.csv (scripts/download_data.py)

Tests are skipped cleanly if either dependency is missing.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_DATA_PATH = Path(__file__).parent.parent / "data" / "complaints.csv"

try:
    from adapters.cfpb import CFPBAdapter
    _HAS_CFPB = True
except ImportError:
    _HAS_CFPB = False

_skip_reason = (
    "adapters.cfpb not yet implemented" if not _HAS_CFPB
    else f"CFPB data not found at {_DATA_PATH}" if not _DATA_PATH.exists()
    else None
)

pytestmark = pytest.mark.skipif(
    _skip_reason is not None, reason=_skip_reason or ""
)


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

        Checks that companies absent from the training split get the
        standardized-zero volume in val/test, not a value leaked from
        those splits.
        """
        splits = adapter.preprocess()

        assert hasattr(adapter, "_train_volumes"), (
            "Adapter must expose _train_volumes dict for leakage verification"
        )
        train_volumes = adapter._train_volumes

        vol_col = adapter.volume_col_idx
        if vol_col is None:
            pytest.skip("company_complaint_volume excluded from features")

        vol_mean = adapter.train_volume_stats["mean"]
        vol_std = adapter.train_volume_stats["std"]
        expected_unknown = (np.log1p(0) - vol_mean) / vol_std

        # Check val and test splits: for every company NOT in train_volumes,
        # its volume feature must equal the standardized-zero value.
        for split_name in ("val", "test"):
            features = splits[split_name]["tabular_features"]
            companies = adapter._split_companies[split_name]

            for i, company in enumerate(companies):
                if company not in train_volumes:
                    actual_vol = features[i, vol_col]
                    assert abs(actual_vol - expected_unknown) < 1e-5, (
                        f"{split_name}[{i}] company={company!r} not in train, "
                        f"expected volume={expected_unknown:.5f}, got {actual_vol:.5f}"
                    )

        # Also verify that at least one known company in train gets a
        # non-zero volume (i.e., the feature is actually doing something)
        train_features = splits["train"]["tabular_features"]
        assert (train_features[:, vol_col] != expected_unknown).any(), (
            "All train volumes equal to unknown-company value — feature is degenerate"
        )

    def test_volume_standardized(self, adapter):
        """Verify company_complaint_volume is standardized (mean ~0, std ~1 on train)."""
        splits = adapter.preprocess()

        vol_col = adapter.volume_col_idx
        if vol_col is None:
            pytest.skip("company_complaint_volume excluded from features")

        train_volume = splits["train"]["tabular_features"][:, vol_col]

        assert abs(train_volume.mean()) < 0.1, (
            f"Volume mean on train = {train_volume.mean():.4f}, expected ~0"
        )
        assert abs(train_volume.std() - 1.0) < 0.3, (
            f"Volume std on train = {train_volume.std():.4f}, expected ~1"
        )
