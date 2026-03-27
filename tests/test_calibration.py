"""Tests for calibration metrics."""

import numpy as np
import pytest

from evaluation.calibration import compute_ece


def test_ece_perfect():
    """ECE should be 0 for perfectly calibrated predictions."""
    y_true = np.array([0, 1, 0, 1])
    y_prob = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    assert compute_ece(y_true, y_prob) == pytest.approx(0.0, abs=1e-6)


def test_ece_worst():
    """ECE should be high for maximally overconfident wrong predictions."""
    y_true = np.array([0, 0, 0, 0])
    y_prob = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
    assert compute_ece(y_true, y_prob) > 0.9


def test_ece_uniform():
    """Uniform predictions on a balanced binary problem should have moderate ECE."""
    y_true = np.array([0, 1, 0, 1])
    y_prob = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
    ece = compute_ece(y_true, y_prob)
    assert 0.0 <= ece <= 0.5


def test_ece_returns_float():
    y_true = np.array([0, 1, 2])
    y_prob = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.3, 0.5]])
    assert isinstance(compute_ece(y_true, y_prob), float)
