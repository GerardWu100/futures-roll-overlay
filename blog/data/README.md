# Frozen blog data

These files were copied from `outputs/runs/2026-07-13_001/` after running
`uv run futures-roll-research` on 2026-07-13:

- `reported_metrics.csv`: metrics from the production pipeline's current target.
- `reported_predictions.parquet`: predictions from that production run.
- `feature_importance.csv`: coefficients from the full pooled ridge fit.

The source review found that the production target is shifted by one session,
that the persistence baseline uses a forward label before it is observable,
and that the final training label in each fold overlaps the test period.
`blog/build_corrected_results.py` corrects the target, uses known trailing
two-session variance for persistence, purges unavailable training labels, and
freezes the audited evidence as:

- `corrected_metrics.csv`: mean walk-forward fold metrics for the future-only target.
- `corrected_predictions.parquet`: out-of-sample predictions for that target.

The run used the repository's tracked `config.toml` and local Parquet inputs.
The technical figures can be regenerated with:

```bash
uv run python blog/build_corrected_results.py
uv run python blog/generate_charts.py
```
