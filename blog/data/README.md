# Frozen Blog Evidence

`metrics.csv`, `predictions.parquet`, and `feature_importance.csv` are frozen
copies from the production research pipeline. The production path enforces the
three timing invariants discussed in the article:

1. A target dated `t` uses squared returns from `t + 1` through `t + H`.
2. Persistence uses the trailing `H` squared returns observable at `t`.
3. Each training fold ends at or before `T - H`, where `T` is the first test
   date, so every fitted label is known at the forecast origin.

Regenerate the evidence and charts from the project root:

```bash
uv run python blog/build_blog_results.py
uv run python blog/generate_charts.py
```

Variance and root mean squared error use annualized decimal variance units with
252 trading sessions per year.
