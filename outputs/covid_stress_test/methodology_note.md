# COVID Shock Forecasting Experiment

## Purpose

This is a separate forecasting stress-test experiment. It does not replace the production Layer 1 model, which may be trained on the full available historical dataset for normal project outputs.

## Observed Dataset Coverage

- First available month: 2015-10-01
- Final available month: 2026-05-01
- Observed monthly periods in project range: 128
- Missing calendar months in project range: 0

## Experimental Split

- Core pre-COVID training period for validation model: through 2019-02-01
- Pre-COVID validation period: 2019-03-01 to 2020-02-01
- Final pre-COVID training period for shock-period model: through 2020-02-01
- COVID shock test period: 2020-03-01 to 2021-09-01
- Recovery period: from 2021-10-01, when observations exist after the shock window

The maximum final training forecast month is earlier than the minimum COVID shock forecast month. All feature scaling statistics for the final shock-period model are fit using rows no later than 2020-02-01.

## Models Compared

- Custom PyTorch TCN quantile regressor with P10, P50 and P90 outputs.
- Seasonal naive baseline using the same month one year earlier, with P10 and P90 formed from pre-COVID residual quantiles.
- Random forest baseline using lagged target values and origin-month operational/calendar features.

## Forecast Protocol

Forecasts are rolling one-month-ahead predictions. For each forecast month, model inputs use observed information available up to the forecast origin, which is the previous month. Model parameters and preprocessing objects are not fit on COVID-period or recovery-period targets.

## Limitations

This experiment tests degradation when pre-COVID fitted models are applied during a severe service disruption. It is not a causal estimate of COVID effects, and it does not simulate counterfactual operational policy. Later COVID-period one-step forecasts may use already-observed previous COVID months as context, but those observations are never used to fit model parameters or scalers.
