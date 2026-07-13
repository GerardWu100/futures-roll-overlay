"""Tests for walk-forward evaluation utilities."""

from __future__ import annotations

import pandas as pd
import pytest

from futures_roll_overlay.evaluation.walk_forward import generate_walk_forward_splits


def test_generate_walk_forward_splits_produces_non_overlapping_windows() -> None:
    """Split generator should produce ordered train/test index windows."""
    index = pd.date_range("2024-01-01", periods=30, freq="B")
    splits = generate_walk_forward_splits(
        date_index=index,
        train_size=10,
        test_size=5,
        step_size=5,
        label_horizon_days=3,
    )
    assert len(splits) > 0
    first = splits[0]
    assert first.train_end == first.test_start - 3


def test_generate_walk_forward_splits_purges_overlapping_labels() -> None:
    """No training target may extend beyond the first test forecast origin."""
    index = pd.date_range("2024-01-01", periods=20, freq="B")
    splits = generate_walk_forward_splits(
        date_index=index,
        train_size=10,
        test_size=5,
        step_size=5,
        label_horizon_days=2,
    )

    for split in splits:
        latest_training_label_end = split.train_end + 2
        assert latest_training_label_end == split.test_start


def test_generate_walk_forward_splits_rejects_empty_purged_training_window() -> None:
    """A label horizon cannot exceed the nominal training boundary."""
    index = pd.date_range("2024-01-01", periods=20, freq="B")

    with pytest.raises(ValueError, match="cannot exceed train_size"):
        generate_walk_forward_splits(
            date_index=index,
            train_size=2,
            test_size=5,
            step_size=5,
            label_horizon_days=3,
        )
