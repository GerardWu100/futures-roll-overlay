"""Tests for naive realized-variance baselines."""

from __future__ import annotations

import pandas as pd

from futures_roll_overlay.models.baselines import trailing_variance_persistence


def test_persistence_baseline_uses_observable_trailing_variance() -> None:
    """Persistence should use the known trailing window without a lead label."""
    trailing_variance = pd.Series(
        [0.10, 0.12, 0.14],
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
        name="known_trailing_rv_annualized",
    )
    prediction = trailing_variance_persistence(trailing_variance=trailing_variance)
    pd.testing.assert_series_equal(
        prediction,
        trailing_variance.rename("prediction"),
    )
