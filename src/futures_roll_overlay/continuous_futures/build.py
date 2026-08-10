"""Continuous futures construction from contract-level bars and roll calendars."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

PRICE_COLUMNS = ["open", "high", "low", "close"]
OUTPUT_COLUMNS = ["date", "open", "high", "low", "close", "volume"]
RollAdjustment = Literal["ratio", "panama"]


@dataclass
class ContinuousFutures:
    """Container for one asset's continuous futures output.

    Attributes
    ----------
    prices : pd.DataFrame
        Adjusted continuous series with columns
        ``date, open, high, low, close, volume``.
    roll_dates : list[pd.Timestamp]
        Dates where active contract changes.
    active_contracts : pd.Series
        Series indexed by date with active contract symbol values.
    unadjusted : pd.DataFrame
        Unadjusted front-contract panel before roll-gap adjustment.
    """

    prices: pd.DataFrame
    roll_dates: list[pd.Timestamp]
    active_contracts: pd.Series
    unadjusted: pd.DataFrame


def _prepare_daily_data(daily_data: pd.DataFrame) -> pd.DataFrame:
    """Normalize contract-level futures input before contract selection."""
    prepared = daily_data.copy()
    prepared["date"] = pd.to_datetime(prepared["date"])
    return prepared.sort_values(["date", "symbol"]).reset_index(drop=True)


def _prepare_roll_calendar(roll_calendar: pd.DataFrame) -> pd.DataFrame:
    """Normalize roll-calendar input before contract selection."""
    prepared = roll_calendar.copy()
    prepared["date"] = pd.to_datetime(prepared["date"])
    return prepared.sort_values(["date", "contract"]).reset_index(drop=True)


def _extract_roll_dates(active_contracts: pd.Series) -> list[pd.Timestamp]:
    """Extract roll timestamps where active contract changes."""
    changed = active_contracts != active_contracts.shift(1)
    if len(changed) > 0:
        changed.iloc[0] = False
    return [pd.Timestamp(item) for item in active_contracts.index[changed]]


def _calendar_roll(
    daily_data: pd.DataFrame,
    roll_calendar: pd.DataFrame,
) -> tuple[pd.DataFrame, list[pd.Timestamp], pd.Series]:
    """Select front contract from explicit roll calendar mapping."""
    merged = roll_calendar.merge(
        daily_data,
        how="left",
        left_on=["date", "contract"],
        right_on=["date", "symbol"],
    )
    merged = merged.dropna(subset=["close"]).reset_index(drop=True)
    active_contracts = merged.set_index("date")["contract"]
    roll_dates = _extract_roll_dates(active_contracts=active_contracts)
    front_prices = merged.loc[:, OUTPUT_COLUMNS].copy()
    return front_prices, roll_dates, active_contracts


def _volume_roll(
    daily_data: pd.DataFrame,
    roll_calendar: pd.DataFrame,
) -> tuple[pd.DataFrame, list[pd.Timestamp], pd.Series]:
    """Select front contract by daily volume leadership among eligible contracts."""
    eligible_contracts = set(roll_calendar["contract"])
    filtered = daily_data[daily_data["symbol"].isin(eligible_contracts)].copy()
    # Highest volume wins; symbol breaks ties deterministically on the same date.
    filtered = filtered.sort_values(
        ["date", "volume", "symbol"],
        ascending=[True, False, True],
    )
    front_prices = filtered.drop_duplicates(subset=["date"], keep="first")
    active_contracts = front_prices.set_index("date")["symbol"].rename("contract")
    roll_dates = _extract_roll_dates(active_contracts=active_contracts)
    return front_prices.loc[:, OUTPUT_COLUMNS].copy(), roll_dates, active_contracts


def _contract_close_panel(daily_data: pd.DataFrame) -> pd.DataFrame:
    """Reshape contract-level bars into a wide close panel.

    Returns
    -------
    pd.DataFrame
        Frame indexed by trade date with one column per contract symbol and
        close prices as values. Dates where a contract did not trade hold
        ``NaN``.
    """
    return daily_data.pivot_table(index="date", columns="symbol", values="close")


def _active_contract_on(
    active_contracts: pd.Series,
    trade_date: pd.Timestamp,
) -> str | None:
    """Return the front contract symbol on one date, or ``None`` if absent."""
    if trade_date not in active_contracts.index:
        return None
    value = active_contracts.loc[trade_date]
    if isinstance(value, pd.Series):
        value = value.iloc[0]
    return str(value)


def _roll_boundary_closes(
    adjusted: pd.DataFrame,
    current_index: int,
    roll_timestamp: pd.Timestamp,
    active_contracts: pd.Series,
    contract_closes: pd.DataFrame,
) -> tuple[float, float]:
    """Return ``(outgoing_close, incoming_close)`` for one roll boundary.

    Both legs are read from the last session before the roll, where the
    outgoing and the incoming contract normally both trade. Measuring them on
    the same session isolates the price difference between the two contracts,
    which is the only part a back-adjustment should remove.

    Reading the outgoing close on one session and the incoming close on the
    next session instead would fold that session's genuine market move into
    the adjustment factor and force the roll-day return of the adjusted series
    to exactly zero.

    The cross-session pair taken from the stitched front series is used only as
    a fallback, when the incoming contract has no bar on the session before the
    roll.
    """
    fallback = (
        float(adjusted.loc[current_index - 1, "close"]),
        float(adjusted.loc[current_index, "close"]),
    )
    previous_date = pd.Timestamp(adjusted.loc[current_index - 1, "date"])
    if previous_date not in contract_closes.index:
        return fallback

    outgoing_symbol = _active_contract_on(active_contracts, previous_date)
    incoming_symbol = _active_contract_on(active_contracts, roll_timestamp)
    if outgoing_symbol is None or incoming_symbol is None:
        return fallback

    overlap_row = contract_closes.loc[previous_date]
    if outgoing_symbol not in overlap_row.index:
        return fallback
    if incoming_symbol not in overlap_row.index:
        return fallback

    outgoing_close = overlap_row[outgoing_symbol]
    incoming_close = overlap_row[incoming_symbol]
    if pd.isna(outgoing_close) or pd.isna(incoming_close):
        return fallback
    return float(outgoing_close), float(incoming_close)


def _apply_roll_adjustment(
    prices: pd.DataFrame,
    roll_dates: list[pd.Timestamp],
    adjustment: RollAdjustment,
    active_contracts: pd.Series,
    contract_closes: pd.DataFrame,
) -> pd.DataFrame:
    """Back-adjust historical prices so roll gaps do not create artificial jumps.

    Walk roll dates from newest to oldest so each earlier segment is scaled or
    shifted by the outgoing-to-incoming price difference measured on the last
    session before that roll.
    """
    adjusted = prices.copy().sort_values("date").reset_index(drop=True)
    for roll_timestamp in sorted(roll_dates, reverse=True):
        roll_index = adjusted.index[adjusted["date"] == roll_timestamp]
        if len(roll_index) == 0:
            continue
        current_index = int(roll_index[0])
        if current_index == 0:
            continue
        old_close, new_close = _roll_boundary_closes(
            adjusted=adjusted,
            current_index=current_index,
            roll_timestamp=roll_timestamp,
            active_contracts=active_contracts,
            contract_closes=contract_closes,
        )
        if adjustment == "ratio":
            if old_close == 0.0:
                continue
            # Multiply all prior OHLC levels by the contract-to-contract ratio.
            factor = new_close / old_close
            adjusted.loc[: current_index - 1, PRICE_COLUMNS] = (
                adjusted.loc[: current_index - 1, PRICE_COLUMNS] * factor
            )
        else:
            # Panama: add the contract-to-contract gap to all prior OHLC levels.
            gap = new_close - old_close
            adjusted.loc[: current_index - 1, PRICE_COLUMNS] = (
                adjusted.loc[: current_index - 1, PRICE_COLUMNS] + gap
            )
    return adjusted


def build_continuous(
    daily_data: pd.DataFrame,
    roll_calendar: pd.DataFrame,
    method: str,
    adjustment: str,
) -> ContinuousFutures:
    """Build one asset-level continuous futures series.

    Parameters
    ----------
    daily_data : pd.DataFrame
        Contract-level futures bars for one asset root.
    roll_calendar : pd.DataFrame
        Roll schedule with columns ``date`` and ``contract``.
    method : str
        Contract-selection method; valid values are ``calendar`` and ``volume``.
    adjustment : str
        Price-adjustment method; valid values are ``ratio`` and ``panama``.

    Returns
    -------
    ContinuousFutures
        Continuous-series output plus roll metadata.
    """
    prepared_daily = _prepare_daily_data(daily_data=daily_data)
    prepared_calendar = _prepare_roll_calendar(roll_calendar=roll_calendar)

    if method == "calendar":
        front_prices, roll_dates, active_contracts = _calendar_roll(
            daily_data=prepared_daily,
            roll_calendar=prepared_calendar,
        )
    elif method == "volume":
        front_prices, roll_dates, active_contracts = _volume_roll(
            daily_data=prepared_daily,
            roll_calendar=prepared_calendar,
        )
    else:
        raise ValueError(f"Unsupported roll method: {method}")

    unadjusted = front_prices.copy()

    if adjustment not in ("ratio", "panama"):
        raise ValueError(f"Unsupported adjustment method: {adjustment}")
    resolved_adjustment: RollAdjustment = adjustment  # validated above
    adjusted = _apply_roll_adjustment(
        prices=front_prices,
        roll_dates=roll_dates,
        adjustment=resolved_adjustment,
        active_contracts=active_contracts,
        contract_closes=_contract_close_panel(daily_data=prepared_daily),
    )

    return ContinuousFutures(
        prices=adjusted,
        roll_dates=roll_dates,
        active_contracts=active_contracts,
        unadjusted=unadjusted,
    )
