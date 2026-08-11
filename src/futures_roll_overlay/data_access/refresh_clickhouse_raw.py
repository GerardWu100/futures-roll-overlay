"""Optional one-time ClickHouse refresh for local raw Parquet inputs.

This module is intentionally separate from the default offline research path.
Use it only when you need to repopulate ``data/raw`` from ClickHouse.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import clickhouse_connect
import pandas as pd
from dotenv import dotenv_values

from futures_roll_overlay.data_access.raw_cache import (
    FUTURES_COLUMNS,
    ROLL_CALENDAR_COLUMNS,
    load_config,
    resolve_project_path,
)


@dataclass(frozen=True)
class RefreshSettings:
    """Configuration for raw-data refresh job execution.

    Attributes
    ----------
    clickhouse_env : Path
        Path to environment file containing ClickHouse credentials.
    futures_dir : Path
        Output directory for futures Parquet files.
    roll_calendars_dir : Path
        Output directory for roll-calendar Parquet files.
    manifest_path : Path
        Output path for manifest JSON file.
    assets : list[str]
        Futures roots to refresh.
    start_date : str
        Inclusive date lower bound.
    end_date : str
        Inclusive date upper bound.
    """

    clickhouse_env: Path
    futures_dir: Path
    roll_calendars_dir: Path
    manifest_path: Path
    assets: list[str]
    start_date: str
    end_date: str


def _load_refresh_settings(config: dict[str, Any]) -> RefreshSettings:
    """Extract refresh-specific settings from repository config."""
    raw_data = config["raw_data"]
    refresh = config["raw_refresh"]
    research = config["research"]
    return RefreshSettings(
        clickhouse_env=resolve_project_path(refresh["clickhouse_env"]),
        futures_dir=resolve_project_path(raw_data["futures_dir"]),
        roll_calendars_dir=resolve_project_path(raw_data["roll_calendars_dir"]),
        manifest_path=resolve_project_path(raw_data["manifest_path"]),
        assets=list(research["assets"]),
        start_date=str(research["start_date"]),
        end_date=str(research["end_date"]),
    )


def _get_client(clickhouse_env: Path) -> clickhouse_connect.driver.Client:
    """Create ClickHouse client from env-file credentials."""
    env_values = dotenv_values(clickhouse_env)
    return clickhouse_connect.get_client(
        host=env_values["CLICKHOUSE_HOST"],
        port=int(env_values["CLICKHOUSE_PORT"]),
        username=env_values["CLICKHOUSE_USER"],
        password=env_values["CLICKHOUSE_PASSWORD"],
        secure=env_values.get("CLICKHOUSE_SECURE", "false").lower() == "true",
        verify=env_values.get("CLICKHOUSE_VERIFY", "false").lower() == "true",
    )


def _fetch_futures_daily(
    client: clickhouse_connect.driver.Client,
    root: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Fetch daily futures bars for one root from ClickHouse."""
    query = f"""
    SELECT
        symbol,
        toDate(ts) AS date,
        argMin(open, ts) AS open,
        max(high) AS high,
        min(low) AS low,
        argMax(close, ts) AS close,
        sum(volume) AS volume
    FROM firstrate.futures
    WHERE startsWith(symbol, '{root}_')
      AND toDate(ts) >= toDate('{start_date}')
      AND toDate(ts) <= toDate('{end_date}')
    GROUP BY symbol, date
    ORDER BY date, symbol
    """
    frame = client.query_df(query)
    missing_columns = [
        column for column in FUTURES_COLUMNS if column not in frame.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Futures query missing required columns for {root}: {missing_columns}"
        )
    frame = frame.loc[:, FUTURES_COLUMNS].copy()
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def _fetch_roll_calendar(
    client: clickhouse_connect.driver.Client,
    root: str,
) -> pd.DataFrame:
    """Fetch full roll calendar for one root from ClickHouse."""
    query = f"""
    SELECT
        date,
        contract
    FROM firstrate.future_dates
    WHERE startsWith(contract, '{root}_')
    ORDER BY date, contract
    """
    frame = client.query_df(query)
    missing_columns = [
        column for column in ROLL_CALENDAR_COLUMNS if column not in frame.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Roll-calendar query missing required columns for {root}: {missing_columns}"
        )
    frame = frame.loc[:, ROLL_CALENDAR_COLUMNS].copy()
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def _manifest_row(
    *,
    asset: str,
    table: str,
    file_path: Path,
    frame: pd.DataFrame,
    columns: list[str],
    variant: str | None = None,
) -> dict[str, object]:
    """Build one manifest table entry with coverage and file-size metadata."""
    row: dict[str, object] = {
        "asset": asset,
        "table": table,
        "file": file_path.as_posix(),
        "start_date": frame["date"].min().strftime("%Y-%m-%d"),
        "end_date": frame["date"].max().strftime("%Y-%m-%d"),
        "row_count": len(frame),
        "columns": columns,
        "file_size_bytes": int(file_path.stat().st_size),
    }
    if variant is not None:
        row["variant"] = variant
    return row


def _write_manifest(manifest_path: Path, rows: list[dict[str, object]]) -> None:
    """Write machine-readable manifest with payload metadata."""
    total_size = int(sum(int(row["file_size_bytes"]) for row in rows))
    payload = {
        "dataset_name": "portable_futures_demo",
        "description": "Offline futures and roll-calendar cache for realized variance forecasting demo",
        "total_files": len(rows),
        "total_raw_payload_bytes": total_size,
        "tables": rows,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def refresh_raw_cache(config_path: Path | None = None) -> None:
    """Refresh local raw Parquet inputs from ClickHouse.

    Parameters
    ----------
    config_path : Path | None, default None
        Optional path to alternate project config.
    """
    config = load_config(config_path=config_path)
    settings = _load_refresh_settings(config=config)
    client = _get_client(clickhouse_env=settings.clickhouse_env)

    settings.futures_dir.mkdir(parents=True, exist_ok=True)
    settings.roll_calendars_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, object]] = []

    for asset in settings.assets:
        futures_frame = _fetch_futures_daily(
            client=client,
            root=asset,
            start_date=settings.start_date,
            end_date=settings.end_date,
        )
        futures_path = settings.futures_dir / f"{asset}.parquet"
        futures_frame.to_parquet(futures_path, index=False)
        manifest_rows.append(
            _manifest_row(
                asset=asset,
                table="futures",
                file_path=futures_path,
                frame=futures_frame,
                columns=FUTURES_COLUMNS,
            )
        )

        full_roll = _fetch_roll_calendar(client=client, root=asset)
        full_roll_path = settings.roll_calendars_dir / f"{asset}_full.parquet"
        full_roll.to_parquet(full_roll_path, index=False)
        manifest_rows.append(
            _manifest_row(
                asset=asset,
                table="roll_calendars",
                file_path=full_roll_path,
                frame=full_roll,
                columns=ROLL_CALENDAR_COLUMNS,
                variant="full",
            )
        )

        sample_roll = full_roll[
            (full_roll["date"] >= pd.Timestamp(settings.start_date))
            & (full_roll["date"] <= pd.Timestamp(settings.end_date))
        ].copy()
        sample_roll_path = settings.roll_calendars_dir / f"{asset}_sample.parquet"
        sample_roll.to_parquet(sample_roll_path, index=False)
        manifest_rows.append(
            _manifest_row(
                asset=asset,
                table="roll_calendars",
                file_path=sample_roll_path,
                frame=sample_roll,
                columns=ROLL_CALENDAR_COLUMNS,
                variant="sample",
            )
        )

    _write_manifest(manifest_path=settings.manifest_path, rows=manifest_rows)


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for raw-cache refresh command."""
    parser = argparse.ArgumentParser(
        description="Refresh local raw Parquet cache from ClickHouse."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional path to alternate config.toml.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for optional ClickHouse refresh workflow."""
    args = _parse_args()
    refresh_raw_cache(config_path=args.config)


if __name__ == "__main__":
    main()
