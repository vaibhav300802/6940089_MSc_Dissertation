# NHS RTT MSc Data Science Project

This folder contains the complete four-layer NHS RTT dissertation project with one shared file contract across forecasting, SHAP, optimisation, and the Streamlit dashboard.

## Canonical Project Structure

```text
nhs_rtt_msc_project/
  config/
    pipeline_config.json
  requirements.txt
  environment.yml
  pyproject.toml
  run_pipeline.py
  download_rtt_data.py
  validate_project.py
  project_notebooks/
    layer1_nhs_rtt_tcn.ipynb
    layer1c_nhs_rtt_rolling_origin_validation.ipynb
    layer1b_nhs_rtt_covid_shock_experiment.ipynb
    layer1d_nhs_rtt_ablation_study.ipynb
    layer2_nhs_rtt_tcn_shap.ipynb
    layer3_nhs_rtt_lp_optimisation.ipynb
  nhs_rtt_pipeline/
    config.py
    data_quality.py
    download.py
    rtt_parsing.py
    schemas.py
    sequences.py
    settings.py
    reproducibility.py
    forecasting_baselines.py
    modeling.py
    optimisation.py
    preprocessing.py
    rolling_origin.py
  data/
    processed/
      rtt_clean.parquet
  models/
    tcn_state_dict.pt
    tcn_dta_state_dict.pt
    model_config.json
    dta_model_config.json
    feature_metadata.json
    dta_feature_metadata.json
    data_dictionary.csv
    surgical_specialties.csv
  outputs/
    logs/
    run_summaries/
    backtest_predictions.parquet
    future_forecasts.parquet
    future_optimisation_forecasts.parquet
    forecast_metrics.csv
    data_quality_report.csv
    data_quality_summary.md
    missingness_by_series.csv
    trust_identifier_changes.csv
    model_comparison.csv
    model_comparison_by_horizon.csv
    model_comparison_by_specialty.csv
    model_comparison_predictions.parquet
    model_comparison_plots/
    rolling_origin_validation/
      rolling_origin_predictions.parquet
      rolling_origin_metrics_overall.csv
      rolling_origin_metrics_by_origin.csv
      rolling_origin_metrics_by_horizon.csv
      rolling_origin_metrics_by_trust.csv
      rolling_origin_metrics_by_specialty.csv
      rolling_origin_metrics_by_waiting_size_group.csv
      rolling_origin_reliability_summary.csv
      rolling_origin_quantile_crossing_report.csv
      rolling_origin_summary.md
      calibration_expected_vs_empirical.png
      interval_width_vs_coverage.png
    part2a_coverage_report.csv
    shap_values.npy
    shap_values_long.parquet
    shap_global_summary.png
    shap_global_feature_importance.csv
    shap_horizon_feature_importance.csv
    shap_local_trust_specialty_explanations.parquet
    shap_local_consistency_report.csv
    shap_audit_log.csv
    shap_trust_interpretations.csv
    lp_allocation_output.csv
    lp_sensitivity.png
    lp_uncertainty_comparison.csv
    lp_covid_stress_test.csv
    covid_stress_test/
      covid_shock_predictions.parquet
      covid_shock_metrics.csv
      covid_shock_degradation.csv
      split_summary.json
      methodology_note.md
  dashboard/
    app.py
    startup_validation.py
    requirements.txt
  tests/
    test_*.py
```

The folders `data/processed`, `models`, and `outputs` are created by the notebooks.

## Execution Order

For a local run with the monthly NHS RTT ZIP files already placed in `data/raw/zips/`, use this route:

```bash
python -m pip install -r requirements.txt
python run_pipeline.py validate --smoke-test
python run_pipeline.py train --force-prepare --force-retrain
python run_pipeline.py rolling_origin
python run_pipeline.py covid_shock
python run_pipeline.py explain
python run_pipeline.py optimise
python run_pipeline.py validate
```

The first heavy command rebuilds the processed dataset and retrains both TCN models from the local raw ZIP files. This is intentional: it prevents stale processed data, model state dictionaries, and feature metadata from being mixed together. If the official NHS RTT files have not been downloaded yet, run `python run_pipeline.py download` first, then run the commands above.

`run_pipeline.py` supports these stages:

```text
download
prepare
train
backtest
forecast
rolling_origin
explain
optimise
covid_shock
validate
all
```

`download` only fetches the raw NHS RTT ZIP datasets into `data/raw/zips/` and writes `data/raw/rtt_full_csv_manifest.csv`. The Layer 1 stages `prepare`, `train`, `backtest`, and `forecast` are implemented by the complete Layer 1 training script, so running any of those executes the full Layer 1 data/preprocessing/training/forecasting path once and writes the canonical Layer 1 outputs. The `--force-prepare` flag rebuilds `data/processed/rtt_clean.parquet`; the `--force-retrain` flag rebuilds `models/tcn_state_dict.pt`, `models/tcn_dta_state_dict.pt`, and their matching metadata fingerprints. The `rolling_origin` stage creates the multi-origin validation outputs required by the project validator. The `covid_shock` stage creates the separate pre-COVID training and COVID-period stress-test outputs. The runner stops immediately if a stage fails, writes a timestamped log to `outputs/logs/`, and writes a machine-readable run summary to `outputs/run_summaries/`.

The notebook route is:

1. `project_notebooks/layer1_nhs_rtt_tcn.ipynb`
2. Rolling-origin validation: `python run_pipeline.py rolling_origin` or `project_notebooks/layer1c_nhs_rtt_rolling_origin_validation.ipynb`
3. COVID shock experiment: `python run_pipeline.py covid_shock` or `project_notebooks/layer1b_nhs_rtt_covid_shock_experiment.ipynb`
4. `project_notebooks/layer2_nhs_rtt_tcn_shap.ipynb`
5. `project_notebooks/layer3_nhs_rtt_lp_optimisation.ipynb`
6. `python validate_project.py`
7. `python dashboard/startup_validation.py`
8. `streamlit run dashboard/app.py`

## Installation

Local virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Conda alternative:

```bash
conda env create -f environment.yml
conda activate nhs-rtt-msc
```

The root `requirements.txt` is for the full reproducible project. `dashboard/requirements.txt` is a smaller Streamlit-only dependency file.

## Central Configuration

Project settings live in:

```text
config/pipeline_config.json
```

This file controls data/output directories, random seed, deterministic PyTorch mode, train/test horizons, model hyperparameters, SHAP sample sizes, and optimisation scenario assumptions. Shared modules read the same settings through `nhs_rtt_pipeline/settings.py`.

Reproducibility helpers are in `nhs_rtt_pipeline/reproducibility.py`. They set Python, NumPy, and PyTorch seeds. Deterministic PyTorch mode can be enabled in the JSON config, but GPU kernels, multiprocessing data loaders, and the CBC solver may still introduce small platform-level variation.

## Testing

Run syntax checks and automated tests:

```bash
python -m compileall -q .
python -m unittest discover -s tests
```

Run the project smoke test:

```bash
python run_pipeline.py validate --smoke-test
```

The test suite covers RTT parsing, date extraction, signed `net_inflow`, unreported removals, sliding-window sequence creation, custom TCN forward-pass output shape, future-date leakage checks, SHAP model loading, optimisation constraints, dashboard startup status, and generated-output schema validation. Tests that require optional heavy packages such as Torch, SHAP, or PuLP skip cleanly when those packages are not installed in the local runtime.

On Windows:

```powershell
.\run_dashboard.bat
```

The helper scripts still work, but the exact Streamlit command is:

```bash
streamlit run dashboard/app.py
```

## Layer 1 Outputs

To download only the raw NHS RTT source ZIPs before running notebooks:

```bash
python download_rtt_data.py
```

or through the staged runner:

```bash
python run_pipeline.py download
```

The downloader scrapes the official NHS England RTT waiting-times pages for monthly “Full CSV data file” ZIP links from October 2015 onward, stores the ZIP files in `data/raw/zips/`, and writes:

- `data/raw/rtt_full_csv_manifest.csv`
- `data/raw/rtt_download_summary.json`

Layer 1 retains the custom PyTorch TCN model and writes:

- `data/processed/rtt_clean.parquet`
- `models/tcn_state_dict.pt`
- `models/tcn_dta_state_dict.pt`
- `models/model_config.json`
- `models/dta_model_config.json`
- `models/feature_metadata.json`
- `models/dta_feature_metadata.json`
- `models/data_dictionary.csv`
- `models/surgical_specialties.csv`
- `outputs/backtest_predictions.parquet`
- `outputs/future_forecasts.parquet`
- `outputs/future_optimisation_forecasts.parquet`
- `outputs/forecast_metrics.csv`
- `outputs/forecast_metrics_by_horizon.csv`
- `outputs/dta_backtest_predictions.parquet`
- `outputs/dta_forecast_metrics.csv`
- `outputs/dta_forecast_metrics_by_horizon.csv`
- `outputs/data_quality_report.csv`
- `outputs/data_quality_summary.md`
- `outputs/missingness_by_series.csv`
- `outputs/trust_identifier_changes.csv`
- `outputs/model_comparison_predictions.parquet`
- `outputs/model_comparison.csv`
- `outputs/model_comparison_by_horizon.csv`
- `outputs/model_comparison_by_specialty.csv`
- `outputs/model_comparison_by_trust_size.csv`
- `outputs/model_comparison_by_covid_period.csv`
- `outputs/model_comparison_paired_error_analysis.csv`
- `outputs/model_comparison_audit_log.csv`
- `outputs/model_comparison_summary.md`
- `outputs/dta_model_comparison_predictions.parquet`
- `outputs/dta_model_comparison.csv`
- `outputs/dta_model_comparison_by_horizon.csv`
- `outputs/dta_model_comparison_by_specialty.csv`
- `outputs/dta_model_comparison_by_trust_size.csv`
- `outputs/dta_model_comparison_by_covid_period.csv`
- `outputs/dta_model_comparison_paired_error_analysis.csv`
- `outputs/dta_model_comparison_audit_log.csv`
- `outputs/dta_model_comparison_summary.md`
- `outputs/model_comparison_plots/overall_model_comparison.png`
- `outputs/model_comparison_plots/mae_by_horizon.png`
- `outputs/model_comparison_plots/tcn_vs_seasonal_naive_abs_error.png`
- `outputs/rolling_origin_validation/rolling_origin_predictions.parquet`
- `outputs/rolling_origin_validation/rolling_origin_metrics_overall.csv`
- `outputs/rolling_origin_validation/rolling_origin_metrics_by_origin.csv`
- `outputs/rolling_origin_validation/rolling_origin_metrics_by_horizon.csv`
- `outputs/rolling_origin_validation/rolling_origin_metrics_by_trust.csv`
- `outputs/rolling_origin_validation/rolling_origin_metrics_by_specialty.csv`
- `outputs/rolling_origin_validation/rolling_origin_metrics_by_waiting_size_group.csv`
- `outputs/rolling_origin_validation/rolling_origin_reliability_summary.csv`
- `outputs/rolling_origin_validation/rolling_origin_quantile_crossing_report.csv`
- `outputs/rolling_origin_validation/rolling_origin_summary.md`
- `outputs/rolling_origin_validation/calibration_expected_vs_empirical.png`
- `outputs/rolling_origin_validation/interval_width_vs_coverage.png`
- `outputs/part2a_coverage_report.csv`
- `outputs/net_inflow_data_quality_summary.csv`
- `outputs/rtt_flow_reconciliation_quality_report.json`

The project preserves both RTT Part 2 and Part 2A measures:

```text
incomplete_total = Part 2 total incomplete RTT pathways
incomplete_decision_to_admit = Part 2A incomplete RTT pathways with a decision to admit
```

`outputs/future_forecasts.parquet` forecasts `incomplete_total` for general RTT forecasting analysis. It must not be interpreted as a surgical waiting list.

`outputs/future_optimisation_forecasts.parquet` forecasts `incomplete_decision_to_admit` for capacity optimisation. If Part 2A is unavailable for a Trust-specialty series, the pipeline does not substitute `incomplete_total`.

`outputs/part2a_coverage_report.csv` reports Part 2A availability by Trust and specialty.

`net_inflow` is signed and is retained as a backward-compatible alias of `reported_net_inflow`:

```text
reported_net_inflow = new_rtt_periods - completed_total
net_inflow = reported_net_inflow
```

Negative values are preserved because they mean completions exceeded new RTT periods. Only genuinely non-negative variables such as `waiting_list`, `new_rtt_periods`, `completed_admitted`, `completed_non_admitted`, and `completed_total` are clipped/validated as non-negative.

Layer 1 also creates RTT accounting reconciliation features:

```text
opening_waiting_list = previous month's closing_waiting_list within each trust-specialty series
closing_waiting_list = current month's reported waiting_list
unreported_removals = opening_waiting_list + new_rtt_periods - completed_total - closing_waiting_list
```

`unreported_removals` is signed. Negative, zero, and positive values are preserved because it is a residual accounting feature. Rows with unavailable opening waiting list, closing waiting list, new RTT periods, or completed pathway components are flagged with missingness indicators and are excluded from residual reconciliation. Missing flow components are not filled with raw zero; model tensors use training-mean imputation after transformation and carry explicit missingness indicators.

`models/data_dictionary.csv` describes the generated accounting variables and model features without causal wording. `outputs/rtt_flow_reconciliation_quality_report.json` contains:

- rows successfully reconciled;
- rows that could not be reconciled;
- unreported removals distribution;
- largest absolute reconciliation discrepancies;
- affected Trusts and specialties.

## Layer 1 Data Quality

Layer 1 starts scraping from financial year `2015-16` and then applies an explicit post-load date filter:

```text
month >= 2015-10-01
```

The preprocessing layer writes:

- `outputs/data_quality_report.csv`
- `outputs/data_quality_summary.md`
- `outputs/missingness_by_series.csv`
- `outputs/trust_identifier_changes.csv`

The data-quality report covers source files loaded, publication month, table type, row counts, duplicate Trust-specialty-month rows before aggregation, missing values, invalid negative counts, Trust code/name changes, specialty-code changes, discontinued series, and unexpectedly large jumps.

Identifier harmonisation is deliberately conservative: Trust and specialty codes are trimmed and uppercased, names are whitespace-normalised, and original source identifiers are retained as audit columns. The pipeline does not automatically merge organisations unless a mapping is explicitly added and documented.

Missing-month completion uses variable-specific rules. Stock waiting-list fields may be forward-filled for inserted missing publication months and receive imputation flags. Activity flow variables such as new RTT periods and completed pathways are not forward-filled; they remain missing and are represented through missingness indicators.

The processed Parquet preserves source provenance columns including source ZIPs, CSVs, URLs, publication months, source row counts, and harmonisation-rule labels. The notebook prints the final retained number of Trusts, specialties, months, and Trust-specialty series after preprocessing.

## Layer 2 Outputs

Layer 2 imports the same `TCNQuantileRegressor` class from `nhs_rtt_pipeline/modeling.py` and loads:

- `models/tcn_state_dict.pt`
- `models/model_config.json`
- `models/feature_metadata.json`
- `data/processed/rtt_clean.parquet`

It writes:

- `outputs/shap_values.npy`
- `outputs/shap_values_aggregated.npy`
- `outputs/shap_values_long.parquet`
- `outputs/shap_feature_names.json`
- `outputs/shap_feature_groups.json`
- `outputs/shap_global_feature_importance.csv`
- `outputs/shap_horizon_feature_importance.csv`
- `outputs/shap_group_importance.csv`
- `outputs/shap_local_trust_specialty_explanations.parquet`
- `outputs/shap_local_consistency_report.csv`
- `outputs/shap_audit_log.csv`
- `outputs/shap_methodology_note.md`
- `outputs/shap_global_summary.png`
- `outputs/shap_trust_interpretations.csv`
- `outputs/shap_trust_{trust_name}.png`

Layer 2 does not use PyTorch Forecasting, Lightning checkpoints, `TimeSeriesDataSet`, or `load_from_checkpoint()`. It loads the custom PyTorch TCN state dictionary, reconstructs the same transformed encoder tensor used in training, and explains the inverse-transformed P50 median forecast.

The SHAP method is `shap.KernelExplainer` wrapped around the custom TCN. The wrapper exposes interpretable lagged encoder feature values and Trust/specialty embedding identifiers, then rebuilds the scaled model tensor using `models/feature_metadata.json`. The default explained horizons are 1, 6, and 12 months when those horizons are present in the trained model.

Layer 2 writes `outputs/shap_local_consistency_report.csv`, which checks:

```text
base value + sum(feature contributions) approximately equals model P50 output
```

The report includes the approximation error for every explained Trust-specialty context and forecast horizon. Missing or non-finite model predictions raise a clear error and are recorded in `outputs/shap_audit_log.csv`; the code does not replace missing predictions with actual target values.

SHAP feature metadata groups model inputs into:

- waiting-list lags;
- referral or new RTT-period features;
- completed-pathway features;
- unreported-removal features;
- calendar features;
- Trust and specialty identifiers or embeddings.

Generated interpretation text uses associational wording such as “was associated with a higher model forecast” or “contributed positively to this model prediction.” It does not claim that a feature caused a waiting-list change. Completed-pathway contributions are interpreted together with the feature values; a positive SHAP value for completed pathways is not automatically interpreted as reduced throughput.

## Layer 3 Outputs

Layer 3 reads:

- `outputs/future_optimisation_forecasts.parquet`

It imports optimisation logic from `nhs_rtt_pipeline/optimisation.py` and writes:

- `outputs/lp_allocation_output.csv`
- `outputs/lp_sensitivity.csv`
- `outputs/lp_sensitivity.png`
- `outputs/lp_uncertainty_comparison.csv`
- `outputs/lp_covid_stress_test.csv`

The optimisation is restricted to Trust-specialty rows where:

- `forecast_target = incomplete_decision_to_admit`;
- Part 2A data are available;
- the treatment-function code is included in `models/surgical_specialties.csv`.

The configured surgical-specialty list includes treatment-function codes whose planned admitted pathways commonly require additional treatment capacity. Clearly non-surgical medical, diagnostic, and therapy specialties are excluded unless explicitly added to the mapping. The session-to-pathway parameter is a scenario assumption for incomplete decision-to-admit pathways, not a claim that one treatment session directly removes people from the total incomplete RTT pathway forecast.

## Canonical Forecast Columns

`outputs/backtest_predictions.parquet` is for evaluation only. It contains historical test-period predictions where actual outcomes are available:

```text
trust_code
trust_name
specialty_code
specialty_name
forecast_origin
forecast_month
horizon
p10
p50
p90
actual
```

`outputs/future_forecasts.parquet` is the genuine forward-looking forecast. It starts after the final observed month in `data/processed/rtt_clean.parquet` and does not contain an `actual` column:

```text
trust_code
trust_name
specialty_code
specialty_name
forecast_origin
forecast_month
horizon
p10
p50
p90
latest_observed_waiting_list
```

`outputs/future_optimisation_forecasts.parquet` is the forward-looking Part 2A forecast used by the LP:

```text
trust_code
trust_name
specialty_code
specialty_name
forecast_origin
forecast_month
horizon
p10
p50
p90
latest_observed_incomplete_decision_to_admit
forecast_target
is_surgical_specialty
specialty_inclusion_criteria
```

Layer 1 validates that every future `forecast_month` is later than `forecast_origin`, backtest rows contain actual values, future rows contain no actual outcomes, and optimisation rows target `incomplete_decision_to_admit`.

## Forecasting Baselines

Layer 1 also compares the custom TCN against defensible baselines on exactly the same held-out backtest rows:

- naive last-value forecast;
- seasonal naive forecast using the value from 12 months earlier;
- historical seasonal mean forecast using only history available at the forecast origin;
- optional `HistGradientBoostingRegressor` quantile baseline.

The expensive machine-learning baseline is controlled by:

```python
Layer1Config.enable_hist_gradient_boosting_baseline
Layer1Config.hist_gradient_boosting_max_train_rows
Layer1Config.hist_gradient_boosting_max_iter
```

All baseline feature construction uses observations no later than each forecast origin. The HistGradientBoosting baseline is fit on training-period rows only. Deterministic baseline P10/P90 intervals are calibrated from validation-period residuals, not from the test period. The model comparison files report MAE, RMSE, sMAPE, WAPE, pinball loss, P10-P90 coverage, and interval width overall and by horizon, specialty, Trust-size group, and COVID/non-COVID period when applicable.

`outputs/model_comparison_summary.md` states which baseline is strongest and whether the TCN actually outperforms seasonal naive. Do not claim TCN superiority unless the relevant comparison table supports it.

## Layer 1C Rolling-Origin Validation

`project_notebooks/layer1c_nhs_rtt_rolling_origin_validation.ipynb` performs rolling-origin backtesting with multiple forecast origins. For each origin it:

- fits feature scalers using that fold's training period only;
- trains a fresh custom PyTorch TCN using targets that end before the forecast origin;
- forecasts the configured horizon after the origin;
- saves raw quantiles before ordering correction and corrected quantiles for display;
- records the raw quantile-crossing rate before any correction.

Default rolling-origin settings are:

```text
forecast horizon: 12 months
origin spacing: 6 months
requested origins: 3
minimum training coverage before each origin: 36 months
```

The origin selector also enforces enough history for the configured encoder length, forecast horizon, and internal validation period. With the default 24-month encoder, 12-month horizon, and 12-month internal validation window, a fold needs at least 48 months of coverage before it is eligible.

Outputs are saved under:

```text
outputs/rolling_origin_validation/
  rolling_origin_predictions.parquet
  rolling_origin_metrics_overall.csv
  rolling_origin_metrics_by_origin.csv
  rolling_origin_metrics_by_horizon.csv
  rolling_origin_metrics_by_trust.csv
  rolling_origin_metrics_by_specialty.csv
  rolling_origin_metrics_by_waiting_size_group.csv
  rolling_origin_reliability_summary.csv
  rolling_origin_quantile_crossing_report.csv
  rolling_origin_summary.md
  calibration_expected_vs_empirical.png
  interval_width_vs_coverage.png
```

The rolling-origin prediction file includes:

```text
forecast_origin
forecast_month
horizon
model_name
trust_code
trust_name
specialty_code
specialty_name
p10_raw
p50_raw
p90_raw
quantile_crossing_raw
p10
p50
p90
actual
```

The summary report states whether the nominal 80% P10-P90 interval is close to empirical coverage. Rolling-origin fold models are validation-only; the final production model artifacts in `models/` are trained separately and are the artifacts used by SHAP, optimisation, and the dashboard.

## Layer 1B COVID Shock Forecasting Experiment

`project_notebooks/layer1b_nhs_rtt_covid_shock_experiment.ipynb` is a separate experiment, not the production forecasting pipeline. It trains stress-test models using only pre-COVID behaviour and evaluates performance during the COVID elective-care disruption.

The default split is:

```text
core pre-COVID training: through February 2019, used for validation-model fitting
pre-COVID validation: March 2019 to February 2020
final pre-COVID training: through February 2020
COVID shock test: March 2020 to September 2021
recovery period: October 2021 onward, if observations exist
```

All scalers and preprocessing statistics for the final shock experiment are fit using rows no later than February 2020. The notebook writes leakage checks proving that the maximum training target month is earlier than the minimum COVID test target month.

Outputs are saved under:

```text
outputs/covid_stress_test/
  covid_shock_predictions.parquet
  covid_shock_metrics.csv
  covid_shock_degradation.csv
  split_summary.json
  methodology_note.md
  actual_vs_forecast_covid_shock.png
  prediction_intervals_covid_shock_tcn.png
  error_over_time.png
  performance_by_trust.png
  performance_by_specialty.png
```

The experiment compares the custom PyTorch TCN with a seasonal naive baseline and a random forest baseline. It reports MAE, RMSE, sMAPE, pinball loss, P10-P90 empirical coverage, average interval width, and percentage degradation from normal pre-COVID validation to the COVID shock period.

## Dashboard

The dashboard reads the canonical files directly from:

- `data/processed/rtt_clean.parquet`
- `models/tcn_state_dict.pt`
- `models/model_config.json`
- `outputs/future_forecasts.parquet`
- `outputs/future_optimisation_forecasts.parquet`
- `outputs/model_comparison.csv`
- `outputs/model_comparison_by_horizon.csv`
- `outputs/data_quality_summary.md`
- `outputs/shap_*`
- `outputs/lp_*`

The dashboard displays a setup-status page when required outputs are missing. It does not retrain models. Interactive capacity scenarios import the shared mixed-integer optimisation function from `nhs_rtt_pipeline/optimisation.py`.

For the map, provide Trust coordinates in either:

- latitude and longitude columns inside `outputs/future_forecasts.parquet`, or
- `dashboard/data/nhs_trust_coordinates.csv`

with columns:

```csv
trust_code,trust_name,latitude,longitude
```

`trust_code` is preferred. If it is missing, the dashboard tries to match by `trust_name`.

If real Trust boundary polygons are unavailable, the dashboard labels the map as a Trust location map and uses point coordinates only. The point map must not be interpreted as Trust catchment or boundary geography.

## One-Command Local Run

After all notebooks have been run:

```bash
bash run_dashboard.sh
```

The script installs dashboard dependencies and starts Streamlit. To run the dashboard startup validation separately:

```bash
python dashboard/startup_validation.py
```

## Streamlit Cloud

Upload the complete `nhs_rtt_msc_project` folder, not only the `dashboard` folder. Set the app entry point to:

```text
dashboard/app.py
```

The shared package `nhs_rtt_pipeline` must remain beside the `dashboard` folder.
