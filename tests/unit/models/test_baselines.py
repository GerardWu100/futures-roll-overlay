"""Tests for naive realized-variance baselines."""

from __future__ import annotations

import pandas as pd

from futures_roll_overlay.models.baselines import persistence_baseline


def test_persistence_baseline_uses_lagged_target() -> None:
    """Persistence baseline should shift the target by one observation."""
    target = pd.Series(
        [0.10, 0.12, 0.14],
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
        name="target_forward_rv_annualized",
    )
    prediction = persistence_baseline(target=target)
    assert prediction.loc[pd.Timestamp("2024-01-03")] == 0.10
    assert prediction.loc[pd.Timestamp("2024-01-04")] == 0.12
