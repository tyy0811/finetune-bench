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

# CFPB has renamed product categories over time; merge historical variants
_PRODUCT_MERGE_MAP = {
    "Credit reporting": "Credit reporting",
    "Credit reporting, credit repair services, or other personal consumer reports": "Credit reporting",
    "Credit reporting or other personal consumer reports": "Credit reporting",
    "Credit card": "Credit card",
    "Credit card or prepaid card": "Credit card",
    "Prepaid card": "Credit card",
    "Payday loan": "Payday loan",
    "Payday loan, title loan, or personal loan": "Payday loan",
    "Payday loan, title loan, personal loan, or advance loan": "Payday loan",
    "Consumer Loan": "Payday loan",
    "Money transfer, virtual currency, or money service": "Money transfer",
    "Money transfers": "Money transfer",
    "Virtual currency": "Money transfer",
    "Bank account or service": "Bank account",
    "Checking or savings account": "Bank account",
    "Debt collection": "Debt collection",
    "Debt or credit management": "Debt collection",
}


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
        self._train_volumes: dict = {}
        self._split_companies: dict[str, list[str]] = {}
        self.volume_col_idx: int | None = None

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

        # Class consolidation: merge historically-renamed CFPB products,
        # then fold remaining rare classes into "Other"
        df["product_clean"] = df["Product"].replace(_PRODUCT_MERGE_MAP)
        product_counts = df["product_clean"].value_counts()
        small_classes = product_counts[product_counts < MIN_CLASS_SIZE].index
        df["product_clean"] = df["product_clean"].where(
            ~df["product_clean"].isin(small_classes), "Other"
        )

        le = LabelEncoder()
        df["label"] = le.fit_transform(df["product_clean"])

        # Split (temporal partitions BEFORE subsampling; random subsamples first)
        if self.split_strategy == "temporal":
            splits_df = self._temporal_split(df)
        else:
            # Stratified subsample for random path
            if self.sample_size and len(df) > self.sample_size:
                df = self._stratified_subsample(df, self.sample_size)
            splits_df = self._random_split(df)

        # Re-fit label encoder on actual data present after subsampling/splitting
        # to prevent orphan classes in class_names
        all_labels_present = set()
        for split_df in splits_df.values():
            all_labels_present.update(split_df["product_clean"].unique())
        le_final = LabelEncoder()
        le_final.fit(sorted(all_labels_present))
        self.class_names = list(le_final.classes_)
        for split_name in splits_df:
            splits_df[split_name] = splits_df[split_name].copy()
            splits_df[split_name]["label"] = le_final.transform(
                splits_df[split_name]["product_clean"]
            )

        # Feature engineering from TRAINING data only
        train_df = splits_df["train"]

        # Company complaint volume from training data
        self._train_volumes = train_df["Company"].value_counts().to_dict()

        # Top-N companies from training data
        top_companies = (
            train_df["Company"].value_counts().head(TOP_N_COMPANIES).index.tolist()
        )

        # Categories from training data
        all_states = sorted(train_df["State"].dropna().unique())
        all_channels = sorted(train_df["Submitted via"].dropna().unique())

        # Compute volume stats on train for standardization
        train_log_volumes = np.log1p(
            train_df["Company"].map(self._train_volumes).fillna(0).values
        )
        vol_mean = float(train_log_volumes.mean())
        vol_std = float(train_log_volumes.std())
        if vol_std == 0:
            vol_std = 1.0
        self.train_volume_stats = {"mean": vol_mean, "std": vol_std}

        result = {}
        for split_name, split_df in splits_df.items():
            # Store company names per split for leakage test verification
            self._split_companies[split_name] = split_df["Company"].tolist()

            features = self._encode_features(
                split_df, top_companies,
                all_states, all_channels, vol_mean, vol_std,
            )
            result[split_name] = {
                "narratives": split_df["narrative"].tolist(),
                "tabular_features": features,
                "labels": split_df["label"].values,
            }

        return result

    def _stratified_subsample(self, df: pd.DataFrame, target_n: int) -> pd.DataFrame:
        """Stratified subsample preserving class ratios with min 2 per class."""
        n_total = len(df)
        parts = []
        for _, group in df.groupby("label"):
            n_prop = max(2, round(len(group) / n_total * target_n))
            n_prop = min(n_prop, len(group))
            parts.append(group.sample(n=n_prop, random_state=self.seed))
        return pd.concat(parts)

    @staticmethod
    def _safe_stratified_split(df, test_size, random_state):
        """Stratified split with fallback to non-stratified when class counts < 2."""
        min_count = df["label"].value_counts().min()
        stratify = df["label"] if min_count >= 2 else None
        return train_test_split(
            df, test_size=test_size, random_state=random_state, stratify=stratify,
        )

    def _random_split(self, df: pd.DataFrame) -> dict:
        train_df, temp_df = self._safe_stratified_split(df, 0.2, self.seed)
        val_df, test_df = self._safe_stratified_split(temp_df, 0.5, self.seed)
        return {"train": train_df, "val": val_df, "test": test_df}

    def _temporal_split(self, df: pd.DataFrame) -> dict:
        """Partition by cutoff date first, then stratified-subsample within each."""
        if not self.cutoff_date:
            raise ValueError("cutoff_date required for temporal split")

        df = df.copy()
        df["date_received"] = pd.to_datetime(df["Date received"])
        pre = df[df["date_received"] < self.cutoff_date]
        post = df[df["date_received"] >= self.cutoff_date]

        # Stratified subsample within each temporal partition
        train_n = min(16000, len(pre))
        test_n = min(4000, len(post))

        if train_n < len(pre):
            pre = self._stratified_subsample(pre, train_n)
        if test_n < len(post):
            post = self._stratified_subsample(post, test_n)

        train_df, val_df = self._safe_stratified_split(pre, 0.1, self.seed)
        return {"train": train_df, "val": val_df, "test": post}

    def _encode_features(
        self,
        df: pd.DataFrame,
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
        col_offset = 0

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
            col_offset += company_onehot.shape[1]

        # 2. State one-hot (vectorized)
        state_cat = pd.CategoricalDtype(categories=all_states, ordered=False)
        state_categorical = df["State"].fillna("").astype(state_cat)
        state_onehot = pd.get_dummies(state_categorical).values.astype(np.float32)
        features.append(state_onehot)
        col_offset += state_onehot.shape[1]

        # 3. Submitted via one-hot (vectorized)
        channel_cat = pd.CategoricalDtype(categories=all_channels, ordered=False)
        channel_categorical = df["Submitted via"].fillna("").astype(channel_cat)
        channel_onehot = pd.get_dummies(channel_categorical).values.astype(np.float32)
        features.append(channel_onehot)
        col_offset += channel_onehot.shape[1]

        # 4. Company complaint volume: log-transformed + standardized
        if "company_complaint_volume" not in self.exclude_features:
            volumes = df["Company"].map(self._train_volumes).fillna(0).values
            log_volumes = np.log1p(volumes)
            standardized = ((log_volumes - vol_mean) / vol_std).reshape(-1, 1)
            features.append(standardized.astype(np.float32))
            self.volume_col_idx = col_offset
        else:
            self.volume_col_idx = None

        return np.hstack(features)
