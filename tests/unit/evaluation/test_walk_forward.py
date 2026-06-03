"""Tests for walk-forward evaluation utilities."""

from __future__ import annotations

import pandas as pd

from futures_roll_overlay.evaluation.walk_forward import generate_walk_forward_splits


def test_generate_walk_forward_splits_produces_non_overlapping_windows() -> None:
    """Split generator should produce ordered train/test index windows."""
    index = pd.date_range("2024-01-01", periods=30, freq="B")
    splits = generate_walk_forward_splits(
        date_index=index,
        train_size=10,
        test_size=5,
        step_size=5,
    )
    assert len(splits) > 0
    first = splits[0]
    assert first.train_end < first.test_start
