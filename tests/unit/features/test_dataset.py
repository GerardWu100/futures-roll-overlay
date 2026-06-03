"""Tests for feature-and-target panel assembly."""

from __future__ import annotations

import pandas as pd

from futures_roll_overlay.features.dataset import build_asset_dataset


def test_build_asset_dataset_merges_term_features_and_rv_target() -> None:
    """Dataset builder should produce a date-indexed model table."""
    term_features = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            "spread": [1.0, 0.5, -0.1],
            "roll_yield": [0.02, 0.01, -0.01],
            "slope": [0.01, 0.005, -0.002],
            "regime": ["contango", "contango", "backwardation"],
        }
    )
    realized_variance = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            "daily_realized_variance": [0.0001, 0.0002, 0.0003],
            "forward_realized_variance": [0.0005, 0.0006, 0.0007],
            "forward_realized_variance_annualized": [0.06, 0.07, 0.08],
        }
    )
    dataset = build_asset_dataset(
        asset="ES",
        term_features=term_features,
        realized_variance=realized_variance,
        lag_days=1,
    )
    assert "asset" in dataset.columns
    assert "target_forward_rv_annualized" in dataset.columns
    assert dataset["asset"].nunique() == 1
