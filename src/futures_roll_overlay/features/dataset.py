"""Dataset assembly for realized-variance forecasting models."""

from __future__ import annotations

import pandas as pd


def _encode_regime_columns(term_features: pd.DataFrame) -> pd.DataFrame:
    """Create one-hot regime indicator columns for linear model input."""
    frame = term_features.copy()
    if "regime" not in frame.columns:
        return frame
    dummies = pd.get_dummies(frame["regime"], prefix="regime", dtype=float)
    return pd.concat([frame.drop(columns=["regime"]), dummies], axis=1)


def build_asset_dataset(
    asset: str,
    term_features: pd.DataFrame,
    realized_variance: pd.DataFrame,
    lag_days: int,
) -> pd.DataFrame:
    """Merge term-structure and realized-variance inputs for one asset.

    Parameters
    ----------
    asset : str
        Asset root identifier, for example ``ES``.
    term_features : pd.DataFrame
        Frame with ``date`` plus term-structure features.
    realized_variance : pd.DataFrame
        Frame with ``date`` plus realized-variance target columns.
    lag_days : int
        Lag depth for autoregressive realized-variance feature terms.

    Returns
    -------
    pd.DataFrame
        Date-indexed model dataset with explanatory features and one target
        column named ``target_forward_rv_annualized``.
    """
    if "date" not in realized_variance.columns:
        realized_variance_frame = realized_variance.reset_index()
    else:
        realized_variance_frame = realized_variance.copy()

    merged = term_features.merge(realized_variance_frame, on="date", how="inner")
    merged = merged.sort_values("date").reset_index(drop=True)
    merged = _encode_regime_columns(term_features=merged)

    # Lagged realized variance features capture persistence in volatility.
    for lag in range(1, lag_days + 1):
        merged[f"lag_rv_{lag}"] = merged["daily_realized_variance"].shift(lag)

    if "log_return" in merged.columns:
        merged["lag_log_return_1"] = merged["log_return"].shift(1)
    else:
        # When log returns are absent, use signed sqrt-RV as a return proxy.
        merged["lag_log_return_1"] = merged["daily_realized_variance"].pow(0.5).shift(1)
    merged["asset"] = asset
    merged["target_forward_rv_annualized"] = merged[
        "forward_realized_variance_annualized"
    ]

    dataset = merged.dropna().reset_index(drop=True)
    drop_columns = ["forward_realized_variance_annualized"]
    if "log_return" in dataset.columns:
        drop_columns.append("log_return")
    return dataset.drop(columns=drop_columns)


def build_pooled_dataset(asset_datasets: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate per-asset datasets into one pooled training panel."""
    if not asset_datasets:
        return pd.DataFrame()
    return pd.concat(asset_datasets, axis=0, ignore_index=True)
