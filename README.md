# Celeris Automobili Industrial — Demand Forecasting & Planning Control Tower

**MKS AI/ML Hackathon · Phase 2 submission**  
A reproducible end-to-end demand data pipeline, global machine-learning forecast, planner dashboard, data-quality register and planning analytics for the supplied synthetic ERP dataset.

## Deliverables

- `pipeline.py` — cleaning, reconciliation, feature engineering, baseline comparison, holdout validation, quantile forecasts and submission generation.
- `expert_outputs.py` — optional planning analytics generated from pipeline outputs.
- `app.py` — Streamlit dashboard with the five required views and additional planning views.
- `data/` — the seven original supplied CSV files.
- `outputs/submission.csv` — final four-column hackathon submission.
- `outputs/forecast_with_intervals.csv` — complete forecast with P10/P50/P90 and planning revenue.
- `outputs/cleaned_train.csv`, `cleaned_test.csv`, `cleaned_macro.csv` — reproducible cleaned data.
- `outputs/data_quality_report.csv` and `data_quality_profile.csv` — issue register and before/after file profile.
- `outputs/anomaly_register.csv` — anomaly candidates with reason/action fields.
- `outputs/backtest_metrics.csv`, `backtest_predictions.csv`, `rolling_origin_19m_metrics.csv` — model validation artifacts.
- `outputs/ordering_cycle_diagnostics.csv`, `macro_lead_lag_correlations.csv`, `procurement_plan_p90.csv`, `lifecycle_forecast.csv`, `stockout_lost_sales_summary.csv`, `planner_exceptions.csv` — optional planner analytics.
- `outputs/forecast_by_*.csv` — forecast totals reconciled by month, family, region and channel.

## Run locally

Python 3.10+ is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python pipeline.py
streamlit run app.py
```

The pipeline reads only from `data/` and writes generated artifacts to `outputs/`. The dashboard reads generated outputs; it does not contain manually typed business metrics.

## Data and business-rule treatment

1. `Date` is authoritative; `Year` and `Month` are rebuilt from it.
2. Duplicate `Record_ID` rows retain the latest `Load_Timestamp`; duplicate monthly `Date-SKU_ID-Region` keys are reduced to the latest loaded record.
3. Text fields are trimmed and missing master attributes are filled from `sku_master.csv` where available.
4. Numeric fields are parsed with thousands separators supported. Negative units are marked missing, not converted to zero.
5. Stockout records use `Requested_Units` as the observed unconstrained-demand target for the matching key. These records are flagged; the stockout file is acknowledged as incomplete.
6. Invalid/extreme unit prices are replaced with the SKU median, then product-family median. Recalculated revenue uses unconstrained demand × cleaned price; future revenue is a planning estimate using the last known SKU price.
7. Lifecycle conflicts are flagged and excluded from training targets. Forecasts are forced to zero before a planned launch and after EOL.
8. Missing historical rows remain missing in the panel; they are not silently interpreted as zero demand.
9. Event flags respect the calendar's date window and Global, Region, Product_Family, Channel or Customer scope.
10. Macro indicators are cleaned and joined with 0–3 month lags. The macro lead/lag table is descriptive correlation, not a causal estimate.
11. Replacement-SKU historical demand and recent predecessor demand are included as transfer features where a replacement relationship is present.
12. ADI/CV² and median observed order-gap diagnostics describe intermittent ordering. They use observed records only; absent ERP rows are not imputed as zero orders.

## Forecasting and validation

- **Baselines:** seasonal-naive lag-12 and recent 3-month rolling mean.
- **Main benchmark:** global LightGBM Tweedie regression, evaluated on a one-step holdout from Nov 2025 to Oct 2026.
- **Forecast distribution:** three global LightGBM quantile-regression models estimate P10, P50 and P90. Quantile crossings are sorted to enforce P10 ≤ P50 ≤ P90. P50 is rounded to a non-negative integer in `submission.csv`.
- **Features:** SKU/region/product/customer/channel identifiers, calendar seasonality, demand lags and rolling statistics, intermittency class, macro lags, event flags, lifecycle age/EOL, and replacement-SKU history.
- **Long-horizon test:** a fixed-origin 19-month holdout trains through Mar 2024 and predicts Apr 2024–Oct 2025 with lag state frozen at the origin. This tests the direct multi-horizon strategy, not 19 separate rolling-origin retrains.

### Recorded validation results

| Validation | WAPE | Bias |
|---|---:|---:|
| Seasonal naive, one-step holdout | 47.07% | +6.49% |
| Recent 3-month mean, one-step holdout | 60.31% | −1.37% |
| LightGBM Tweedie, one-step holdout | 37.60% | −1.72% |
| LightGBM Tweedie, fixed-origin 19-month holdout | 55.62% | −11.80% |

WAPE = Σ|actual − forecast| / Σactual. Bias = (Σforecast − Σactual) / Σactual. The long-horizon score is materially weaker than the one-step score; this limitation is disclosed rather than hidden. P10/P50/P90 are quantile-regression estimates, but segment-level empirical coverage calibration has not yet been established. They must not be described as guaranteed confidence intervals.

## Dashboard views

**Required:** Executive overview; Data health; Demand explorer; Anomaly monitor; Forecast & accuracy.  
**Additional:** Macro lead/lag diagnostics; procurement P90 plan; lifecycle forecast; stockout/lost-sales summary; planner exceptions; CSV downloads.

The dashboard uses common family/SKU/region/channel/customer filters for its history and forecast views. Historical and forecast time series are displayed on a continuous timeline with the forecast start marked.

## Submission checks

The generated submission has been checked against the supplied `test.csv`:

- 58,219 rows, matching the test file.
- Exact `Date`, `SKU_ID`, `Region` key order.
- No duplicate submission keys.
- No missing or negative forecast values.
- Forecast dates span 2026-11 through 2028-05.
- Interval file has 58,219 rows, no missing P10/P50/P90 values, and no quantile crossings.

## Important limitations

- The 19-month validation WAPE is 55.62%; the long-horizon forecast remains uncertain and should be reviewed at family/SKU level.
- Quantile intervals are model estimates, not yet calibrated for nominal 10%/90% coverage by segment.
- Future revenue uses last observed cleaned SKU price; future annual price changes and surcharges are not forecast.
- Macro lead/lag correlations are exploratory and not causal. The what-if PMI feature is therefore not presented as a causal demand simulator.
- Procurement P90 is a demand-planning proxy; lead times, MOQ, supplier capacity, safety stock and inventory on hand are not included in the supplied problem data.
- Anomaly flags identify review candidates, not proof of data corruption. Genuine one-off orders and events should be retained when supported by the event/customer context.

All company names and data are fictional and synthetic, as stated in the problem statement.
