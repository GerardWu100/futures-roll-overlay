#!/usr/bin/env bash
# Thin wrapper: run the offline research pipeline from repo root.
set -euo pipefail
cd "$(dirname "$0")/.."
exec uv run futures-roll-research "$@"
