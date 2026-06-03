"""Integration test for offline research pipeline execution."""

from __future__ import annotations

from pathlib import Path

from futures_roll_overlay.pipeline.run_research_pipeline import run_research_pipeline


def test_run_research_pipeline_writes_expected_outputs(tmp_path: Path) -> None:
    """Pipeline should run from local raw cache and write run artifacts."""
    result = run_research_pipeline(output_root=tmp_path)
    run_dir = result["run_dir"]
    assert (run_dir / "dataset.parquet").exists()
    assert (run_dir / "predictions.parquet").exists()
    assert (run_dir / "notebook_dataset.parquet").exists()
    assert (run_dir / "notebook_predictions.parquet").exists()
    assert (run_dir / "metrics.csv").exists()
    assert (run_dir / "feature_importance.csv").exists()
