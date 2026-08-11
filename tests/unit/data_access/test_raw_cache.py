"""Tests for offline raw Parquet access.

The new project contract requires that default research runs read only from
``data/raw``. These tests verify the local-file loader behavior and manifest
introspection behavior.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from futures_roll_overlay.data_access.raw_cache import (
    get_futures_daily,
    get_roll_calendar,
    list_raw_inventory,
    load_config,
)


def test_load_config_reads_project_defaults() -> None:
    """The default configuration should expose the raw-data section."""
    config = load_config()
    assert "raw_data" in config
    assert "research" in config


def test_get_futures_daily_reads_local_parquet_with_date_filter() -> None:
    """Futures loader should read one asset file and apply an inclusive window."""
    config = load_config()
    result = get_futures_daily(
        root="ES",
        start="2024-01-02",
        end="2024-01-03",
        config=config,
    )
    assert not result.empty
    assert list(result.columns) == [
        "symbol",
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]
    assert pd.to_datetime(result["date"]).min() == pd.Timestamp("2024-01-02")
    assert pd.to_datetime(result["date"]).max() == pd.Timestamp("2024-01-03")


def test_get_roll_calendar_supports_sample_and_full_variants() -> None:
    """Roll calendar loader should resolve the requested variant file."""
    config = load_config()
    sample_frame = get_roll_calendar(root="CL", config=config, variant="sample")
    full_frame = get_roll_calendar(root="CL", config=config, variant="full")
    assert len(sample_frame) > 0
    assert len(full_frame) > 0
    assert len(full_frame) >= len(sample_frame)
    assert list(sample_frame.columns) == ["date", "contract"]


def test_list_raw_inventory_reads_manifest_table() -> None:
    """Manifest loader should produce one row per tracked raw file."""
    config = load_config()
    inventory = list_raw_inventory(config=config)
    assert not inventory.empty
    assert "asset" in inventory.columns
    assert "table" in inventory.columns
    assert "row_count" in inventory.columns


def test_get_futures_daily_raises_when_asset_file_is_missing(tmp_path: Path) -> None:
    """Missing asset Parquet file should raise a clear file-not-found error."""
    config = {
        "raw_data": {
            "futures_dir": str(tmp_path / "futures"),
            "roll_calendars_dir": str(tmp_path / "roll_calendars"),
            "manifest_path": str(tmp_path / "manifest.json"),
            "roll_calendar_default_variant": "sample",
        }
    }
    with pytest.raises(FileNotFoundError):
        get_futures_daily(
            root="ES",
            start="2024-01-02",
            end="2024-01-03",
            config=config,
        )
