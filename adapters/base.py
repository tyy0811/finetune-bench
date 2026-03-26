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
