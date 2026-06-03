# Data Access Guide

This folder defines data boundaries for the project.

- `raw_cache.py`: default offline loaders for futures and roll calendars from `data/raw`.
- `refresh_clickhouse_raw.py`: optional one-time ClickHouse refresh command that rewrites `data/raw` and manifest metadata.

Interview story for this layer:

1. Default research path is offline and deterministic.
2. Database calls are explicit and separate from normal runs.
3. Raw-data contract is visible as files plus manifest, not hidden state.
