"""Tests for realized-variance target construction."""

from __future__ import annotations

import pandas as pd

from futures_roll_overlay.features.realized_variance import (
    build_forward_realized_variance,
)


def test_build_forward_realized_variance_adds_daily_and_forward_columns() -> None:
    """Target dates should contain only returns strictly after that date."""
    returns = pd.Series(
        [0.01, -0.02, 0.03, 0.01],
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]),
        name="log_return",
    )
    frame = build_forward_realized_variance(
        log_returns=returns,
        horizon_days=2,
        annualization_factor=252,
    )
    assert "daily_realized_variance" in frame.columns
    assert "forward_realized_variance" in frame.columns
    assert "forward_realized_variance_annualized" in frame.columns
    assert frame.loc[pd.Timestamp("2024-01-02"), "daily_realized_variance"] == 0.01**2
    expected_forward_variance = pd.Series(
        [(-0.02) ** 2 + 0.03**2, 0.03**2 + 0.01**2, None, None],
        index=returns.index.rename("date"),
        name="forward_realized_variance",
        dtype=float,
    )
    pd.testing.assert_series_equal(
        frame["forward_realized_variance"],
        expected_forward_variance,
    )
    assert frame.loc[
        pd.Timestamp("2024-01-03"), "known_trailing_rv_annualized"
    ] == 126 * (0.01**2 + (-0.02) ** 2)
