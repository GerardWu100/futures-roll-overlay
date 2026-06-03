"""Tests for term-structure feature construction."""

from __future__ import annotations

import pandas as pd

from futures_roll_overlay.features.term_structure import compute_term_structure_features


def _curve_data() -> pd.DataFrame:
    """Create one-day, three-contract curve data."""
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-02"]),
            "symbol": ["AA_H24", "AA_M24", "AA_U24"],
            "close": [100.0, 101.0, 102.0],
        }
    )


def _contract_end_dates() -> pd.Series:
    """Create end-date metadata for contract timing denominators."""
    return pd.Series(
        {
            "AA_H24": pd.Timestamp("2024-02-16"),
            "AA_M24": pd.Timestamp("2024-03-15"),
            "AA_U24": pd.Timestamp("2024-06-14"),
        }
    )


def test_compute_term_structure_features_outputs_expected_columns() -> None:
    """Feature builder should return spread, roll yield, slope, and regime."""
    frame = compute_term_structure_features(
        daily_data=_curve_data(),
        contract_end_dates=_contract_end_dates(),
        flat_threshold=0.001,
        num_contracts=3,
        regime_persistence_days=1,
    )
    assert list(frame.columns) == ["date", "spread", "roll_yield", "slope", "regime"]
    assert len(frame) == 1
    assert frame.loc[0, "spread"] == 1.0
    assert frame.loc[0, "regime"] == "contango"
