# Raw Cache Contract

Default runtime reads only from `data/raw`.

Required structure:

- `data/raw/futures/<ASSET>.parquet`
- `data/raw/roll_calendars/<ASSET>_sample.parquet`
- `data/raw/roll_calendars/<ASSET>_full.parquet`
- `data/raw/manifest/dataset_manifest.json`

`dataset_manifest.json` fields per table entry:

- `asset`
- `table`
- optional `variant`
- `file`
- `start_date`
- `end_date`
- `row_count`
- `columns`
- `file_size_bytes`

The contract is considered valid when all required files exist and can be read with expected columns.

Optional refresh command:

`uv run python -m futures_roll_overlay.data_access.refresh_clickhouse_raw`

This command is separate from default offline pipeline execution.
