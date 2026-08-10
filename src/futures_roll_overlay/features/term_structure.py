"""Term-structure feature engineering from futures contract curves."""

from __future__ import annotations

import numpy as np
import pandas as pd

MONTH_CODE_TO_MONTH: dict[str, int] = {
    "F": 1,
    "G": 2,
    "H": 3,
    "J": 4,
    "K": 5,
    "M": 6,
    "N": 7,
    "Q": 8,
    "U": 9,
    "V": 10,
    "X": 11,
    "Z": 12,
}


def _parse_contract_expiry(contract_symbol: str) -> pd.Timestamp:
    """Parse a contract code such as ``ES_H24`` into an expiry proxy date."""
    _root, expiry_code = contract_symbol.split("_", maxsplit=1)
    month_code = expiry_code[0]
    year_suffix = expiry_code[1:]
    month_number = MONTH_CODE_TO_MONTH[month_code]
    year_number = 2000 + int(year_suffix)
    return pd.Timestamp(year=year_number, month=month_number, day=1)


def _ordered_contract_slice(
    daily_data: pd.DataFrame, num_contracts: int
) -> pd.DataFrame:
    """Order by contract expiry and retain nearest maturities per date."""
    ordered = daily_data.copy()
    ordered["date"] = pd.to_datetime(ordered["date"])
    ordered["expiry"] = ordered["symbol"].map(_parse_contract_expiry)
    ordered = ordered.sort_values(["date", "expiry"]).reset_index(drop=True)
    ordered["contract_rank"] = ordered.groupby("date").cumcount()
    return ordered[ordered["contract_rank"] < num_contracts].copy()


def _compute_curve_slope(closes: np.ndarray) -> float:
    """Compute normalized linear slope over maturity ranks."""
    if len(closes) < 2:
        return 0.0
    if closes[0] == 0.0:
        return 0.0
    normalized = closes / closes[0]
    maturity_axis = np.arange(len(closes), dtype=float)
    slope, _ = np.polyfit(maturity_axis, normalized, deg=1)
    return float(slope)


def _maturity_gap_days(
    front_contract: str,
    second_contract: str,
    contract_end_dates: pd.Series,
) -> int:
    """Return the positive day gap between two contracts' end dates.

    Both ends are read from the same timing source. The roll-calendar
    lifecycle map is preferred, because it dates the session on which each
    contract stopped being the front contract. When either symbol is missing
    from that map, both ends fall back to month-code expiry parsing.

    Reading one end from the calendar and the other from the month-code proxy
    would put the two legs on different scales. The calendar dates a roll-out
    session; the proxy dates the first day of the delivery month. Mixing them
    produced gaps of a few weeks between contracts that are a full quarter
    apart, which then inflated the annualized roll yield several times over.

    Parameters
    ----------
    front_contract : str
        Nearest-maturity contract symbol, for example ``ES_Z24``.
    second_contract : str
        Next-maturity contract symbol, for example ``ES_H25``.
    contract_end_dates : pd.Series
        Series mapping contract symbol to lifecycle end date.

    Returns
    -------
    int
        Day gap, floored at 1 so it can be used as a denominator.
    """
    both_in_calendar = (
        front_contract in contract_end_dates.index
        and second_contract in contract_end_dates.index
    )
    if both_in_calendar:
        front_end = pd.Timestamp(contract_end_dates.loc[front_contract])
        second_end = pd.Timestamp(contract_end_dates.loc[second_contract])
    else:
        front_end = _parse_contract_expiry(front_contract)
        second_end = _parse_contract_expiry(second_contract)
    return max((second_end - front_end).days, 1)


def _apply_regime_persistence(
    raw_regimes: pd.Series, persistence_days: int
) -> pd.Series:
    """Smooth regime labels with a persistence threshold before switching."""
    persisted: list[str] = []
    active_regime: str | None = None
    candidate_regime: str | None = None
    candidate_count = 0

    for raw_regime in raw_regimes:
        regime_label = str(raw_regime)
        if active_regime is None:
            active_regime = regime_label
            candidate_regime = regime_label
            candidate_count = 0
            persisted.append(regime_label)
            continue

        if regime_label == active_regime:
            candidate_regime = regime_label
            candidate_count = 0
            persisted.append(active_regime)
            continue

        # Count consecutive days of the same challenger regime.
        if regime_label == candidate_regime:
            candidate_count += 1
        else:
            candidate_regime = regime_label
            candidate_count = 1

        if candidate_count >= persistence_days:
            active_regime = regime_label
            candidate_count = 0

        persisted.append(active_regime)

    return pd.Series(persisted, index=raw_regimes.index, name="regime")


def compute_term_structure_features(
    daily_data: pd.DataFrame,
    contract_end_dates: pd.Series,
    flat_threshold: float,
    num_contracts: int,
    regime_persistence_days: int,
) -> pd.DataFrame:
    """Compute daily term-structure features for one asset root.

    Let ``F1`` be the front-contract close and ``F2`` be the second-contract
    close. The spread is ``F2 - F1``. The annualized roll yield is
    ``((F1 - F2) / F2) * (365 / day_gap)``, where ``day_gap`` is the positive
    day difference between configured contract end dates for the first two
    maturities.

    Parameters
    ----------
    daily_data : pd.DataFrame
        Contract-level close panel with columns ``date``, ``symbol``, ``close``.
    contract_end_dates : pd.Series
        Series mapping contract symbol to lifecycle end date.
    flat_threshold : float
        Absolute roll-yield threshold for ``flat`` regime labels.
    num_contracts : int
        Number of nearest contracts to retain per date for slope estimation.
    regime_persistence_days : int
        Number of consecutive observations required before regime switching.

    Returns
    -------
    pd.DataFrame
        Date-indexed feature panel with columns
        ``date, spread, roll_yield, slope, regime``.
    """
    ordered = _ordered_contract_slice(
        daily_data=daily_data, num_contracts=num_contracts
    )
    rows: list[dict[str, object]] = []

    for trade_date, date_slice in ordered.groupby("date"):
        sorted_slice = date_slice.sort_values("expiry").reset_index(drop=True)
        if len(sorted_slice) < 2:
            continue
        front_contract = str(sorted_slice.loc[0, "symbol"])
        second_contract = str(sorted_slice.loc[1, "symbol"])

        front_close = float(sorted_slice.loc[0, "close"])
        second_close = float(sorted_slice.loc[1, "close"])
        spread = second_close - front_close

        day_gap = _maturity_gap_days(
            front_contract=front_contract,
            second_contract=second_contract,
            contract_end_dates=contract_end_dates,
        )
        roll_yield = ((front_close - second_close) / second_close) * (365.0 / day_gap)

        close_curve = sorted_slice["close"].to_numpy(dtype=float)
        slope = _compute_curve_slope(closes=close_curve)

        if abs(roll_yield) <= flat_threshold:
            raw_regime = "flat"
        elif spread > 0.0:
            raw_regime = "contango"
        else:
            raw_regime = "backwardation"

        rows.append(
            {
                "date": pd.Timestamp(trade_date),
                "spread": spread,
                "roll_yield": roll_yield,
                "slope": slope,
                "raw_regime": raw_regime,
            }
        )

    if not rows:
        return pd.DataFrame(columns=["date", "spread", "roll_yield", "slope", "regime"])

    frame = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    frame = frame.set_index("date")
    frame["regime"] = _apply_regime_persistence(
        raw_regimes=frame["raw_regime"],
        persistence_days=regime_persistence_days,
    )
    frame = frame.drop(columns=["raw_regime"]).reset_index()
    return frame
