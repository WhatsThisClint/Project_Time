# Cleaning Playbook

Use this checklist when preparing time-series data.

1. Confirm which column is the timestamp.
2. Confirm which column is the forecast target.
3. Use a series ID column when the file has multiple wells, sensors, stores, or locations.
4. Let Project Time collapse duplicate timestamps with `mean` unless the value should be additive, in which case use `sum`.
5. Prefer interpolation for environmental measurements sampled on a regular cadence.
6. Use `drop` when missing values should not be imputed.
7. Use `iqr` or `winsorize` only when clear measurement spikes are present.
8. Check the cleaning report before forecasting.
9. Run a holdout forecast to compare methods before trusting a future forecast.
10. Use TimesFM for rich historical signals or many related series, and keep lightweight baselines as a sanity check.
