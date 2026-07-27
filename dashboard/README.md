# RTT Waiting List Forecast Dashboard

This dashboard presents the saved dissertation outputs: national forecasts, Trust-level forecasts, model evidence, feature-contribution outputs, capacity scenarios, and data checks.

Run the startup validation from the project root after all notebooks have completed:

```bash
python dashboard/startup_validation.py
```

Launch the dashboard from the project root:

```bash
streamlit run dashboard/app.py
```

The helper script also works:

```bash
bash run_dashboard.sh
```

The dashboard reads the standard project outputs directly:

- `../data/processed/rtt_clean.parquet`
- `../models/tcn_state_dict.pt`
- `../models/model_config.json`
- `../outputs/future_forecasts.parquet`
- `../outputs/model_comparison.csv`
- `../outputs/model_comparison_by_horizon.csv`
- `../outputs/future_optimisation_forecasts.parquet`
- `../outputs/data_quality_summary.md`
- `../outputs/shap_*`
- `../outputs/lp_*`

`../outputs/future_forecasts.parquet` must be the genuine future file with `latest_observed_waiting_list` and no `actual` column. Do not point the dashboard at `backtest_predictions.parquet`.

The dashboard does not retrain forecasting models. Interactive capacity scenarios call the shared optimisation function in `nhs_rtt_pipeline/optimisation.py`.

Optional map coordinates can be placed at:

```text
dashboard/data/nhs_trust_coordinates.csv
```

Recommended columns:

```csv
trust_code,trust_name,latitude,longitude
```

To rebuild the coordinate file from NHS ODS Trust postcodes:

```bash
python dashboard/build_trust_coordinates.py
```

The map uses point locations only. It does not show Trust boundary polygons.

For Streamlit Cloud, upload the complete project folder and set the app entry point to:

```text
dashboard/app.py
```
