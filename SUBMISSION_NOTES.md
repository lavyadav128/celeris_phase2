# Submission notes

## Final files
- `outputs/submission.csv` is the required file to upload to the hackathon portal.
- `outputs/forecast_with_intervals.csv` is the companion planning output.
- `outputs/run_summary.json` contains reproducibility and validation metadata.
- `README.md` documents setup, assumptions, validation and limitations.

## Results snapshot
- Test rows: 58,219
- Forecast period: Nov 2026–May 2028
- Forecast total: 2,546,506 units (P50 integer submission)
- One-step LightGBM WAPE: 37.60%; bias: −1.72%
- Fixed-origin 19-month holdout WAPE: 55.62%; bias: −11.80%
- Submission keys: exact test key order; no duplicates, missing values or negative forecasts

## Important disclosure
P10/P50/P90 come from quantile regression. Their empirical segment coverage is not yet calibrated. The 19-month holdout is a single fixed-origin direct-horizon evaluation, not a full set of rolling-origin retrains. Forecast revenue uses last known SKU price. These are stated explicitly in the README and dashboard.
