# Features Guide

This folder constructs model inputs and realized-variance targets.

- `term_structure.py`: spread, roll yield, slope, and regime labels.
- `realized_variance.py`: daily variance and forward annualized realized variance targets.
- `dataset.py`: per-asset feature panel assembly, lagged features, and pooled dataset construction.

Interview story for this layer:

1. Define the target mathematically before modeling.
2. Use explainable feature families tied to futures market structure.
3. Keep transformations modular so each stage is auditable.
