"""Tests for continuous futures construction.

The continuous-series builder should stitch contract-level prices into one
asset-level close series with explicit roll metadata.
"""

from __future__ import annotations

import pandas as pd

from futures_roll_overlay.continuous_futures.build import build_continuous


def _daily_contract_data() -> pd.DataFrame:
    """Create overlapping contract prices for one synthetic asset."""
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2024-01-02",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-03",
                    "2024-01-04",
                    "2024-01-04",
                ]
            ),
            "symbol": ["ZZ_H24", "ZZ_M24", "ZZ_H24", "ZZ_M24", "ZZ_H24", "ZZ_M24"],
            "open": [100.0, 101.0, 101.0, 102.0, 102.0, 103.0],
            "high": [101.0, 102.0, 102.0, 103.0, 103.0, 104.0],
            "low": [99.0, 100.0, 100.0, 101.0, 101.0, 102.0],
            "close": [100.0, 101.0, 101.0, 102.0, 102.0, 103.0],
            "volume": [1_000.0, 900.0, 950.0, 1_050.0, 900.0, 1_200.0],
        }
    )


def _roll_calendar() -> pd.DataFrame:
    """Create a two-contract roll calendar with one roll event."""
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            "contract": ["ZZ_H24", "ZZ_H24", "ZZ_M24"],
        }
    )


def test_build_continuous_returns_expected_columns_and_roll_dates() -> None:
    """Calendar + ratio mode should output a stitched series and roll date."""
    result = build_continuous(
        daily_data=_daily_contract_data(),
        roll_calendar=_roll_calendar(),
        method="calendar",
        adjustment="ratio",
    )
    assert list(result.prices.columns) == [
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]
    assert result.roll_dates == [pd.Timestamp("2024-01-04")]
    assert result.active_contracts.loc[pd.Timestamp("2024-01-04")] == "ZZ_M24"


def test_build_continuous_volume_mode_switches_on_higher_volume() -> None:
    """Volume mode should switch active contract once deferred volume is larger."""
    result = build_continuous(
        daily_data=_daily_contract_data(),
        roll_calendar=_roll_calendar(),
        method="volume",
        adjustment="ratio",
    )
    assert pd.Timestamp("2024-01-03") in result.roll_dates
