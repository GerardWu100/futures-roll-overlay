# Continuous Futures Guide

This folder builds continuous futures series from contract-level bars.

- `build.py`:
  - contract selection (`calendar` or `volume`),
  - roll-gap adjustment (`ratio` or `panama`),
  - output dataclass with adjusted prices, unadjusted prices, active contracts, and roll dates.

Interview story for this layer:

1. Raw contracts become one tradable time series per asset.
2. Adjustment method is explicit and configurable.
3. Roll metadata is preserved for downstream diagnostics and interpretation.
