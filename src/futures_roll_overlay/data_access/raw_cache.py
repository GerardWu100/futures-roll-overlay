"""Offline raw-cache access for the futures research pipeline.

This module defines the default runtime data contract for the repository.
Normal research runs read only from local Parquet files under ``data/raw``.
There is no ClickHouse dependency in these code paths.
"""

from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.toml"

FUTURES_COLUMNS = ["symbol", "date", "open", "high", "low", "close", "volume"]
ROLL_CALENDAR_COLUMNS = ["date", "contract"]


@dataclass(frozen=True)
class RawDataSettings:
    """Resolved raw-data paths from configuration.

    Attributes
    ----------
    futures_dir : Path
        Directory containing one futures Parquet file per asset root.
    roll_calendars_dir : Path
        Directory containing roll-calendar Parquet files.
    manifest_path : Path
        Path to the machine-readable raw-data manifest JSON file.
    roll_calendar_default_variant : str
        Default roll-calendar variant name, for example ``sample``.
    """

    futures_dir: Path
    roll_calendars_dir: Path
    manifest_path: Path
    roll_calendar_default_variant: str


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load TOML configuration for the research pipeline.

    Parameters
    ----------
    config_path : Path | None, default None
        Optional explicit path to a TOML file.

    Returns
    -------
    dict[str, Any]
        Parsed configuration dictionary.
    """
    resolved_path = config_path or DEFAULT_CONFIG_PATH
    with resolved_path.open("rb") as config_file:
        return tomllib.load(config_file)


def resolve_project_path(path_text: str) -> Path:
    """Resolve absolute or project-relative config paths."""
    path = Path(path_text)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _raw_data_settings(config: dict[str, Any]) -> RawDataSettings:
    """Resolve raw-data settings from the ``raw_data`` configuration section."""
    section = config.get("raw_data", {})
    futures_dir = resolve_project_path(section.get("futures_dir", "data/raw/futures"))
    roll_calendars_dir = resolve_project_path(
        section.get("roll_calendars_dir", "data/raw/roll_calendars")
    )
    manifest_path = resolve_project_path(
        section.get("manifest_path", "data/raw/manifest/dataset_manifest.json")
    )
    roll_calendar_default_variant = str(
        section.get("roll_calendar_default_variant", "sample")
    )
    return RawDataSettings(
        futures_dir=futures_dir,
        roll_calendars_dir=roll_calendars_dir,
        manifest_path=manifest_path,
        roll_calendar_default_variant=roll_calendar_default_variant,
    )


def _read_parquet_with_columns(
    parquet_path: Path,
    required_columns: list[str],
    table_name: str,
) -> pd.DataFrame:
    """Read one Parquet file and validate that required columns exist.

    Parameters
    ----------
    parquet_path : Path
        Path to the Parquet file.
    required_columns : list[str]
        Required output columns in canonical order.
    table_name : str
        Logical table name for error messages.

    Returns
    -------
    pd.DataFrame
        Parquet contents with columns reordered to canonical order.

    Raises
    ------
    FileNotFoundError
        Raised when the Parquet file does not exist.
    ValueError
        Raised when required schema columns are missing.
    """
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Missing offline Parquet for table='{table_name}': {parquet_path}"
        )
    frame = pd.read_parquet(parquet_path)
    missing_columns = [
        column_name
        for column_name in required_columns
        if column_name not in frame.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Parquet schema mismatch for table='{table_name}', path='{parquet_path}', "
            f"missing columns={missing_columns}"
        )
    return frame.loc[:, required_columns].copy()


def _apply_date_filter(
    frame: pd.DataFrame,
    date_column: str,
    start: str | None,
    end: str | None,
) -> pd.DataFrame:
    """Apply inclusive date filtering to one DataFrame."""
    filtered = frame.copy()
    filtered[date_column] = pd.to_datetime(filtered[date_column])
    if start is not None:
        filtered = filtered[filtered[date_column] >= pd.Timestamp(start)]
    if end is not None:
        filtered = filtered[filtered[date_column] <= pd.Timestamp(end)]
    return filtered


def get_futures_daily(
    root: str,
    start: str,
    end: str,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Load daily futures contract bars from local raw Parquet.

    Parameters
    ----------
    root : str
        Futures root symbol, for example ``ES``.
    start : str
        Inclusive start date with ``YYYY-MM-DD`` format.
    end : str
        Inclusive end date with ``YYYY-MM-DD`` format.
    config : dict[str, Any] | None, default None
        Optional loaded config dictionary.

    Returns
    -------
    pd.DataFrame
        Contract-level futures panel with canonical columns.
    """
    resolved_config = config or load_config()
    settings = _raw_data_settings(config=resolved_config)
    parquet_path = settings.futures_dir / f"{root}.parquet"
    frame = _read_parquet_with_columns(
        parquet_path=parquet_path,
        required_columns=FUTURES_COLUMNS,
        table_name="futures",
    )
    filtered = _apply_date_filter(
        frame=frame,
        date_column="date",
        start=start,
        end=end,
    )
    return filtered.sort_values(["date", "symbol"]).reset_index(drop=True)


def get_roll_calendar(
    root: str,
    config: dict[str, Any] | None = None,
    start: str | None = None,
    end: str | None = None,
    variant: str | None = None,
) -> pd.DataFrame:
    """Load roll-calendar dates from local raw Parquet.

    Parameters
    ----------
    root : str
        Futures root symbol, for example ``ES``.
    config : dict[str, Any] | None, default None
        Optional loaded config dictionary.
    start : str | None, default None
        Optional inclusive lower bound date.
    end : str | None, default None
        Optional inclusive upper bound date.
    variant : str | None, default None
        Roll-calendar file variant, for example ``sample`` or ``full``.

    Returns
    -------
    pd.DataFrame
        Roll schedule with columns ``date`` and ``contract``.
    """
    resolved_config = config or load_config()
    settings = _raw_data_settings(config=resolved_config)
    resolved_variant = variant or settings.roll_calendar_default_variant
    parquet_path = settings.roll_calendars_dir / f"{root}_{resolved_variant}.parquet"
    frame = _read_parquet_with_columns(
        parquet_path=parquet_path,
        required_columns=ROLL_CALENDAR_COLUMNS,
        table_name="roll_calendars",
    )
    filtered = _apply_date_filter(
        frame=frame,
        date_column="date",
        start=start,
        end=end,
    )
    return filtered.sort_values(["date", "contract"]).reset_index(drop=True)


def list_raw_inventory(config: dict[str, Any] | None = None) -> pd.DataFrame:
    """Read the raw-data manifest as a tabular inventory.

    Parameters
    ----------
    config : dict[str, Any] | None, default None
        Optional loaded config dictionary.

    Returns
    -------
    pd.DataFrame
        One row per file listed in the manifest ``tables`` collection.
    """
    resolved_config = config or load_config()
    settings = _raw_data_settings(config=resolved_config)
    if not settings.manifest_path.exists():
        raise FileNotFoundError(f"Missing raw-data manifest: {settings.manifest_path}")
    payload = json.loads(settings.manifest_path.read_text(encoding="utf-8"))
    table_rows = payload.get("tables", [])
    return pd.DataFrame(table_rows)
