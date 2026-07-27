from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nhs_rtt_pipeline.config import (
    BACKTEST_PREDICTION_COLUMNS,
    CLEAN_RTT_COLUMNS,
    COLUMNS,
    FUTURE_FORECAST_COLUMNS,
    LP_ALLOCATION_COLUMNS,
    OPTIMISATION_FORECAST_COLUMNS,
    ROLLING_ORIGIN_PREDICTION_COLUMNS,
    get_paths,
    validate_backtest_predictions_frame,
    validate_future_forecast_frame,
    validate_optimisation_forecast_frame,
)
from nhs_rtt_pipeline.preprocessing import assert_flow_reconciliation_integrity, assert_net_inflow_integrity
from nhs_rtt_pipeline.rolling_origin import validate_rolling_origin_predictions


PATHS = get_paths(PROJECT_ROOT)

REQUIRED_FILES = {
    "clean processed RTT data": PATHS.clean_parquet,
    "TCN state dict": PATHS.tcn_state_dict,
    "DTA TCN state dict": PATHS.dta_tcn_state_dict,
    "model config": PATHS.model_config,
    "DTA model config": PATHS.dta_model_config,
    "feature metadata": PATHS.feature_metadata,
    "DTA feature metadata": PATHS.dta_feature_metadata,
    "data dictionary": PATHS.data_dictionary,
    "backtest predictions": PATHS.backtest_predictions,
    "DTA backtest predictions": PATHS.dta_backtest_predictions,
    "future forecasts": PATHS.future_forecasts,
    "future Part 2A optimisation forecasts": PATHS.future_optimisation_forecasts,
    "forecast metrics": PATHS.forecast_metrics,
    "DTA forecast metrics": PATHS.dta_forecast_metrics,
    "DTA forecast metrics by horizon": PATHS.dta_forecast_metrics_by_horizon,
    "model comparison": PATHS.model_comparison,
    "DTA model comparison": PATHS.dta_model_comparison,
    "model comparison by horizon": PATHS.model_comparison_by_horizon,
    "DTA model comparison by horizon": PATHS.dta_model_comparison_by_horizon,
    "model comparison by specialty": PATHS.model_comparison_by_specialty,
    "DTA model comparison by specialty": PATHS.dta_model_comparison_by_specialty,
    "model comparison predictions": PATHS.model_comparison_predictions,
    "DTA model comparison predictions": PATHS.dta_model_comparison_predictions,
    "model comparison summary": PATHS.model_comparison_summary,
    "DTA model comparison summary": PATHS.dta_model_comparison_summary,
    "model comparison overall plot": PATHS.model_comparison_overall_png,
    "model comparison by horizon plot": PATHS.model_comparison_by_horizon_png,
    "TCN vs seasonal naive paired error plot": PATHS.model_comparison_tcn_vs_seasonal_png,
    "rolling-origin predictions": PATHS.rolling_origin_predictions,
    "rolling-origin overall metrics": PATHS.rolling_origin_metrics,
    "rolling-origin metrics by origin": PATHS.rolling_origin_metrics_by_origin,
    "rolling-origin metrics by horizon": PATHS.rolling_origin_metrics_by_horizon,
    "rolling-origin metrics by Trust": PATHS.rolling_origin_metrics_by_trust,
    "rolling-origin metrics by specialty": PATHS.rolling_origin_metrics_by_specialty,
    "rolling-origin metrics by waiting-list-size group": PATHS.rolling_origin_metrics_by_waiting_size,
    "rolling-origin reliability summary": PATHS.rolling_origin_reliability,
    "rolling-origin quantile-crossing report": PATHS.rolling_origin_quantile_crossing,
    "rolling-origin report": PATHS.rolling_origin_summary,
    "rolling-origin calibration plot": PATHS.rolling_origin_calibration_png,
    "rolling-origin interval width coverage plot": PATHS.rolling_origin_interval_width_coverage_png,
    "Part 2A coverage report": PATHS.part2a_coverage_report,
    "surgical specialty inclusion mapping": PATHS.surgical_specialties,
    "net inflow data-quality summary": PATHS.net_inflow_quality,
    "RTT flow reconciliation quality report": PATHS.flow_reconciliation_quality,
    "data-quality report": PATHS.data_quality_report,
    "data-quality summary": PATHS.data_quality_summary,
    "missingness by series": PATHS.missingness_by_series,
    "trust identifier changes": PATHS.trust_identifier_changes,
    "SHAP values": PATHS.shap_values,
    "SHAP global summary image": PATHS.shap_global_summary,
    "SHAP Trust interpretations": PATHS.shap_interpretations,
    "LP allocation output": PATHS.lp_allocation_output,
    "LP sensitivity chart": PATHS.lp_sensitivity_png,
    "LP uncertainty comparison": PATHS.lp_uncertainty_comparison,
    "LP COVID stress test": PATHS.lp_covid_stress_test,
}

OPTIONAL_FILES = {
    "Trust coordinates": PATHS.dashboard_data_dir / "nhs_trust_coordinates.csv",
    "SHAP waterfall index": PATHS.shap_waterfall_index,
    "SHAP long-format values": PATHS.shap_values_long,
    "SHAP global feature importance": PATHS.shap_global_importance,
    "SHAP horizon feature importance": PATHS.shap_horizon_importance,
    "SHAP local Trust-specialty explanations": PATHS.shap_local_explanations,
    "SHAP local consistency report": PATHS.shap_consistency_report,
    "SHAP audit log": PATHS.shap_audit_log,
    "SHAP methodology note": PATHS.shap_methodology_note,
    "COVID shock forecasting predictions": PATHS.covid_predictions,
    "COVID shock forecasting metrics": PATHS.covid_metrics,
    "COVID shock degradation summary": PATHS.covid_degradation,
    "COVID shock split summary": PATHS.covid_split_summary,
    "COVID shock methodology note": PATHS.covid_methodology_note,
}

MODEL_CONFIG_KEYS = [
    "model_class",
    "format",
    "n_features",
    "n_trusts",
    "n_specialties",
    "prediction_length",
    "quantiles",
    "hidden_channels",
    "tcn_levels",
    "kernel_size",
    "dropout",
    "embedding_dim",
]

FEATURE_METADATA_KEYS = [
    "feature_columns",
    "raw_feature_columns",
    "feature_stats",
    "trust_to_idx",
    "specialty_to_idx",
    "quantiles",
    "config",
    "model_config",
    "feature_groups",
    "feature_group_names",
    "missingness_feature_columns",
    "data_dictionary",
]


def check_file_exists(label: str, path: Path, required: bool = True) -> bool:
    exists = path.exists()
    if exists:
        print(f"OK: {label}: {path}")
        return True
    if required:
        print(f"MISSING: {label}: {path}")
        return False
    print(f"optional missing: {label}: {path}")
    return True


def check_columns(path: Path, expected_columns: list[str], loader: str, label: str) -> bool:
    if not path.exists():
        return False
    try:
        if loader == "csv":
            frame = pd.read_csv(path, nrows=20)
        elif loader == "parquet":
            frame = pd.read_parquet(path).head(20)
        else:
            raise ValueError(loader)
    except Exception as exc:
        print(f"BAD FILE: {label}: could not read {path}: {exc}")
        return False
    missing = [column for column in expected_columns if column not in frame.columns]
    if missing:
        print(f"BAD SCHEMA: {label}: missing columns: {missing}")
        return False
    print(f"OK SCHEMA: {label}")
    return True


def check_backtest_contract(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        frame = validate_backtest_predictions_frame(pd.read_parquet(path), "outputs/backtest_predictions.parquet")
    except Exception as exc:
        print(f"BAD BACKTEST CONTRACT: {exc}")
        return False
    if frame[COLUMNS.actual].isna().any():
        print("BAD BACKTEST CONTRACT: actual contains missing values")
        return False
    print("OK CONTRACT: backtest predictions contain historical actual values")
    return True


def check_future_contract(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        frame = validate_future_forecast_frame(pd.read_parquet(path), "outputs/future_forecasts.parquet")
    except Exception as exc:
        print(f"BAD FUTURE CONTRACT: {exc}")
        return False
    if COLUMNS.actual in frame.columns:
        print("BAD FUTURE CONTRACT: future forecasts contain an actual column")
        return False
    if not (frame[COLUMNS.forecast_month] > frame[COLUMNS.forecast_origin]).all():
        print("BAD FUTURE CONTRACT: forecast_month must be later than forecast_origin")
        return False
    print("OK CONTRACT: future forecasts contain no actual outcomes and all months are after origin")
    return True


def check_optimisation_forecast_contract(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        frame = validate_optimisation_forecast_frame(pd.read_parquet(path), "outputs/future_optimisation_forecasts.parquet")
    except Exception as exc:
        print(f"BAD OPTIMISATION FORECAST CONTRACT: {exc}")
        return False
    if not (frame[COLUMNS.forecast_target] == COLUMNS.incomplete_decision_to_admit).all():
        print("BAD OPTIMISATION FORECAST CONTRACT: forecast_target is not incomplete_decision_to_admit")
        return False
    print("OK CONTRACT: optimisation forecasts use Part 2A decision-to-admit rows for surgical specialties")
    return True


def check_net_inflow_contract(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        frame = pd.read_parquet(path)
        assert_net_inflow_integrity(frame)
    except Exception as exc:
        print(f"BAD NET INFLOW CONTRACT: {exc}")
        return False
    print("OK CONTRACT: net_inflow preserves signed values and equals new_rtt_periods - completed_total")
    return True


def check_flow_reconciliation_contract(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        frame = pd.read_parquet(path)
        assert_flow_reconciliation_integrity(frame)
    except Exception as exc:
        print(f"BAD RTT FLOW RECONCILIATION CONTRACT: {exc}")
        return False
    print("OK CONTRACT: RTT flow reconciliation residuals are signed and internally consistent")
    return True


def check_json_keys(path: Path, expected_keys: list[str], label: str) -> bool:
    if not path.exists():
        return False
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        print(f"BAD JSON: {label}: {exc}")
        return False
    missing = [key for key in expected_keys if key not in payload]
    if missing:
        print(f"BAD JSON SCHEMA: {label}: missing keys: {missing}")
        return False
    print(f"OK JSON: {label}")
    return True


def read_json_file(path: Path, label: str) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        print(f"BAD JSON: {label}: {exc}")
        return None
    if not isinstance(payload, dict):
        print(f"BAD JSON: {label}: expected a JSON object")
        return None
    return payload


def check_model_artifact_contract(model_config_path: Path, feature_metadata_path: Path, label: str) -> bool:
    model_config = read_json_file(model_config_path, f"{label} model config")
    metadata = read_json_file(feature_metadata_path, f"{label} feature metadata")
    if model_config is None or metadata is None:
        return False

    model_fingerprint = model_config.get("artifact_fingerprint")
    metadata_fingerprint = metadata.get("artifact_fingerprint")
    if not model_fingerprint or not metadata_fingerprint:
        print(
            f"BAD MODEL ARTIFACT CONTRACT: {label} artifacts do not contain an artifact_fingerprint. "
            "Re-run the training stage with --force-retrain."
        )
        return False
    if model_fingerprint != metadata_fingerprint:
        print(
            f"BAD MODEL ARTIFACT CONTRACT: {label} model_config and feature_metadata fingerprints differ. "
            "Re-run the training stage with --force-prepare --force-retrain."
        )
        return False

    required_fingerprint_keys = [
        "fingerprint_version",
        "target_column",
        "row_count",
        "series_count",
        "min_month",
        "max_month",
        "boundaries",
        "sample_counts",
        "feature_columns_sha256",
        "feature_stats_sha256",
        "prepared_target_sha256",
        "model_config_sha256",
    ]
    missing = [key for key in required_fingerprint_keys if key not in model_fingerprint]
    if missing:
        print(f"BAD MODEL ARTIFACT CONTRACT: {label} fingerprint is missing keys: {missing}")
        return False

    if int(model_config.get("n_features", -1)) != len(metadata.get("feature_columns", [])):
        print(
            f"BAD MODEL ARTIFACT CONTRACT: {label} n_features does not match feature_columns length "
            f"({model_config.get('n_features')} vs {len(metadata.get('feature_columns', []))})"
        )
        return False
    if model_config.get("quantiles") != metadata.get("quantiles"):
        print(f"BAD MODEL ARTIFACT CONTRACT: {label} quantiles differ between config and metadata")
        return False

    print(f"OK MODEL ARTIFACT CONTRACT: {label} config and feature metadata are fingerprint-matched")
    return True


def check_flow_report(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        print(f"BAD JSON: outputs/rtt_flow_reconciliation_quality_report.json: {exc}")
        return False
    required = [
        "summary",
        "missingness_indicators",
        "unreported_removals_distribution",
        "largest_absolute_reconciliation_discrepancies",
    ]
    missing = [key for key in required if key not in payload]
    if missing:
        print(f"BAD FLOW REPORT: missing keys: {missing}")
        return False
    summary = payload["summary"]
    for key in ["rows_successfully_reconciled", "rows_could_not_be_reconciled"]:
        if key not in summary:
            print(f"BAD FLOW REPORT: summary missing {key}")
            return False
    print("OK JSON: outputs/rtt_flow_reconciliation_quality_report.json")
    return True


def check_covid_split_summary(path: Path) -> bool:
    if not path.exists():
        return True
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        print(f"BAD COVID SPLIT SUMMARY: {exc}")
        return False
    checks = payload.get("leakage_checks", {})
    if not checks.get("training_before_covid_test", False):
        print("BAD COVID SPLIT SUMMARY: training_before_covid_test is not true")
        return False
    if not checks.get("scaler_fit_before_covid_test", False):
        print("BAD COVID SPLIT SUMMARY: scaler_fit_before_covid_test is not true")
        return False
    print("OK OPTIONAL CONTRACT: COVID shock experiment split records pre-COVID-only fitting")
    return True


def check_rolling_origin_contract(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        frame = validate_rolling_origin_predictions(pd.read_parquet(path))
    except Exception as exc:
        print(f"BAD ROLLING-ORIGIN CONTRACT: {exc}")
        return False
    if frame[COLUMNS.actual].isna().any():
        print("BAD ROLLING-ORIGIN CONTRACT: actual contains missing values")
        return False
    if not (frame[COLUMNS.forecast_month] > frame[COLUMNS.forecast_origin]).all():
        print("BAD ROLLING-ORIGIN CONTRACT: forecast months must be later than origins")
        return False
    origin_count = int(frame[COLUMNS.forecast_origin].nunique())
    if origin_count < 3:
        print(f"BAD ROLLING-ORIGIN CONTRACT: expected at least three forecast origins, found {origin_count}")
        return False
    print(
        "OK CONTRACT: rolling-origin predictions contain actual values, non-leaking dates, "
        f"and {origin_count} forecast origins"
    )
    return True


def check_rolling_origin_summary(path: Path) -> bool:
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8")
    required_phrases = [
        "Nominal P10-P90 coverage",
        "Empirical P10-P90 coverage",
        "final production model is trained separately",
    ]
    missing = [phrase for phrase in required_phrases if phrase not in text]
    if missing:
        print(f"BAD ROLLING-ORIGIN SUMMARY: missing phrases: {missing}")
        return False
    print("OK REPORT: rolling-origin summary states empirical interval coverage and production-model separation")
    return True


def check_optional_shap_contracts() -> bool:
    ok = True
    if PATHS.shap_consistency_report.exists():
        ok = check_columns(
            PATHS.shap_consistency_report,
            [
                "context_id",
                COLUMNS.trust_code,
                COLUMNS.trust_name,
                COLUMNS.specialty_code,
                COLUMNS.specialty_name,
                COLUMNS.horizon,
                "base_value",
                "sum_feature_contributions",
                "base_plus_sum",
                "model_output",
                "absolute_approximation_error",
            ],
            "csv",
            "outputs/shap_local_consistency_report.csv",
        ) and ok
    if PATHS.shap_local_explanations.exists():
        ok = check_columns(
            PATHS.shap_local_explanations,
            [
                COLUMNS.trust_code,
                COLUMNS.trust_name,
                COLUMNS.specialty_code,
                COLUMNS.specialty_name,
                COLUMNS.horizon,
                "feature_group",
                "mean_shap_value",
                "mean_abs_shap_value",
                "interpretation_note",
            ],
            "parquet",
            "outputs/shap_local_trust_specialty_explanations.parquet",
        ) and ok
    return ok


def main() -> int:
    print(f"Project root: {PROJECT_ROOT}")
    print("Canonical file contract:")
    print(f"- processed data: {PATHS.clean_parquet}")
    print(f"- model state dict: {PATHS.tcn_state_dict}")
    print(f"- model config: {PATHS.model_config}")
    print(f"- feature metadata: {PATHS.feature_metadata}")
    print(f"- future forecasts: {PATHS.future_forecasts}")
    print(f"- future optimisation forecasts: {PATHS.future_optimisation_forecasts}")

    ok = True
    for label, path in REQUIRED_FILES.items():
        ok = check_file_exists(label, path, required=True) and ok
    for label, path in OPTIONAL_FILES.items():
        check_file_exists(label, path, required=False)

    ok = check_columns(PATHS.clean_parquet, CLEAN_RTT_COLUMNS, "parquet", "data/processed/rtt_clean.parquet") and ok
    ok = check_net_inflow_contract(PATHS.clean_parquet) and ok
    ok = check_flow_reconciliation_contract(PATHS.clean_parquet) and ok
    ok = check_columns(
        PATHS.net_inflow_quality,
        [
            "min_net_inflow",
            "max_net_inflow",
            "negative_observations",
            "negative_observations_pct",
            "zero_observations",
            "zero_observations_pct",
            "positive_observations",
            "positive_observations_pct",
        ],
        "csv",
        "outputs/net_inflow_data_quality_summary.csv",
    ) and ok
    ok = check_flow_report(PATHS.flow_reconciliation_quality) and ok
    ok = check_columns(
        PATHS.data_quality_report,
        ["audit_section", "issue_type", "severity", "metric", "value", "details"],
        "csv",
        "outputs/data_quality_report.csv",
    ) and ok
    ok = check_columns(
        PATHS.missingness_by_series,
        [
            COLUMNS.series_id,
            COLUMNS.trust_code,
            COLUMNS.trust_name,
            COLUMNS.specialty_code,
            COLUMNS.specialty_name,
            "missing_months",
            "max_consecutive_missing_months",
            "missing_pct",
            "excluded_for_insufficient_history",
        ],
        "csv",
        "outputs/missingness_by_series.csv",
    ) and ok
    ok = check_columns(
        PATHS.trust_identifier_changes,
        ["change_type", "first_month", "last_month", "distinct_values", "details"],
        "csv",
        "outputs/trust_identifier_changes.csv",
    ) and ok
    if PATHS.data_quality_summary.exists():
        text = PATHS.data_quality_summary.read_text(encoding="utf-8")
        if "Retained Trusts/providers" not in text or "Retained Trust-specialty series" not in text:
            print("BAD DATA QUALITY SUMMARY: retained-count lines are missing")
            ok = False
        else:
            print("OK REPORT: outputs/data_quality_summary.md includes retained dataset counts")
    ok = check_columns(
        PATHS.data_dictionary,
        ["feature", "feature_group", "description", "signed", "missing_allowed"],
        "csv",
        "models/data_dictionary.csv",
    ) and ok
    ok = check_columns(
        PATHS.backtest_predictions,
        BACKTEST_PREDICTION_COLUMNS,
        "parquet",
        "outputs/backtest_predictions.parquet",
    ) and ok
    ok = check_columns(
        PATHS.future_forecasts,
        FUTURE_FORECAST_COLUMNS,
        "parquet",
        "outputs/future_forecasts.parquet",
    ) and ok
    ok = check_columns(
        PATHS.future_optimisation_forecasts,
        OPTIMISATION_FORECAST_COLUMNS,
        "parquet",
        "outputs/future_optimisation_forecasts.parquet",
    ) and ok
    ok = check_backtest_contract(PATHS.backtest_predictions) and ok
    ok = check_backtest_contract(PATHS.dta_backtest_predictions) and ok
    ok = check_future_contract(PATHS.future_forecasts) and ok
    ok = check_optimisation_forecast_contract(PATHS.future_optimisation_forecasts) and ok
    ok = check_columns(
        PATHS.part2a_coverage_report,
        [
            "trust_code",
            "trust_name",
            "specialty_code",
            "specialty_name",
            "part2a_available_months",
            "part2a_coverage_pct",
            "eligible_for_capacity_simulation",
        ],
        "csv",
        "outputs/part2a_coverage_report.csv",
    ) and ok
    ok = check_columns(
        PATHS.surgical_specialties,
        ["specialty_code", "specialty_inclusion_criteria", "configured_use"],
        "csv",
        "models/surgical_specialties.csv",
    ) and ok
    ok = check_columns(PATHS.forecast_metrics, ["metric", "value"], "csv", "outputs/forecast_metrics.csv") and ok
    ok = check_columns(PATHS.dta_forecast_metrics, ["metric", "value"], "csv", "outputs/dta_forecast_metrics.csv") and ok
    ok = check_columns(
        PATHS.dta_forecast_metrics_by_horizon,
        ["horizon", "pinball_mean", "rmse_median", "mae_median"],
        "csv",
        "outputs/dta_forecast_metrics_by_horizon.csv",
    ) and ok
    ok = check_columns(
        PATHS.model_comparison,
        [
            "model",
            "n_rows",
            "mae",
            "rmse",
            "smape",
            "wape",
            "pinball_mean",
            "p10_p90_coverage",
            "average_interval_width",
        ],
        "csv",
        "outputs/model_comparison.csv",
    ) and ok
    ok = check_columns(
        PATHS.dta_model_comparison,
        [
            "model",
            "n_rows",
            "mae",
            "rmse",
            "smape",
            "wape",
            "pinball_mean",
            "p10_p90_coverage",
            "average_interval_width",
        ],
        "csv",
        "outputs/dta_model_comparison.csv",
    ) and ok
    ok = check_columns(
        PATHS.model_comparison_by_horizon,
        ["model", COLUMNS.horizon, "n_rows", "mae", "rmse", "smape", "wape"],
        "csv",
        "outputs/model_comparison_by_horizon.csv",
    ) and ok
    ok = check_columns(
        PATHS.dta_model_comparison_by_horizon,
        ["model", COLUMNS.horizon, "n_rows", "mae", "rmse", "smape", "wape"],
        "csv",
        "outputs/dta_model_comparison_by_horizon.csv",
    ) and ok
    ok = check_columns(
        PATHS.model_comparison_by_specialty,
        ["model", COLUMNS.specialty_code, COLUMNS.specialty_name, "n_rows", "mae", "rmse", "smape", "wape"],
        "csv",
        "outputs/model_comparison_by_specialty.csv",
    ) and ok
    ok = check_columns(
        PATHS.dta_model_comparison_by_specialty,
        ["model", COLUMNS.specialty_code, COLUMNS.specialty_name, "n_rows", "mae", "rmse", "smape", "wape"],
        "csv",
        "outputs/dta_model_comparison_by_specialty.csv",
    ) and ok
    ok = check_columns(
        PATHS.model_comparison_predictions,
        [
            "model",
            COLUMNS.trust_code,
            COLUMNS.specialty_code,
            COLUMNS.forecast_origin,
            COLUMNS.forecast_month,
            COLUMNS.horizon,
            COLUMNS.actual,
            COLUMNS.p10,
            COLUMNS.p50,
            COLUMNS.p90,
        ],
        "parquet",
        "outputs/model_comparison_predictions.parquet",
    ) and ok
    ok = check_columns(
        PATHS.dta_model_comparison_predictions,
        [
            "model",
            COLUMNS.trust_code,
            COLUMNS.specialty_code,
            COLUMNS.forecast_origin,
            COLUMNS.forecast_month,
            COLUMNS.horizon,
            COLUMNS.actual,
            COLUMNS.p10,
            COLUMNS.p50,
            COLUMNS.p90,
        ],
        "parquet",
        "outputs/dta_model_comparison_predictions.parquet",
    ) and ok
    ok = check_columns(
        PATHS.rolling_origin_predictions,
        ROLLING_ORIGIN_PREDICTION_COLUMNS,
        "parquet",
        "outputs/rolling_origin_validation/rolling_origin_predictions.parquet",
    ) and ok
    ok = check_rolling_origin_contract(PATHS.rolling_origin_predictions) and ok
    rolling_metric_columns = [
        "n_rows",
        "mae",
        "rmse",
        "smape",
        "pinball_q10",
        "pinball_q50",
        "pinball_q90",
        "pinball_mean",
        "p10_p90_coverage",
        "average_interval_width",
        "winkler_score_80",
        "quantile_crossing_rate_raw",
    ]
    ok = check_columns(
        PATHS.rolling_origin_metrics,
        ["metric_group", *rolling_metric_columns],
        "csv",
        "outputs/rolling_origin_validation/rolling_origin_metrics_overall.csv",
    ) and ok
    ok = check_columns(
        PATHS.rolling_origin_metrics_by_origin,
        [COLUMNS.forecast_origin, *rolling_metric_columns],
        "csv",
        "outputs/rolling_origin_validation/rolling_origin_metrics_by_origin.csv",
    ) and ok
    ok = check_columns(
        PATHS.rolling_origin_metrics_by_horizon,
        [COLUMNS.horizon, *rolling_metric_columns],
        "csv",
        "outputs/rolling_origin_validation/rolling_origin_metrics_by_horizon.csv",
    ) and ok
    ok = check_columns(
        PATHS.rolling_origin_metrics_by_trust,
        [COLUMNS.trust_code, COLUMNS.trust_name, *rolling_metric_columns],
        "csv",
        "outputs/rolling_origin_validation/rolling_origin_metrics_by_trust.csv",
    ) and ok
    ok = check_columns(
        PATHS.rolling_origin_metrics_by_specialty,
        [COLUMNS.specialty_code, COLUMNS.specialty_name, *rolling_metric_columns],
        "csv",
        "outputs/rolling_origin_validation/rolling_origin_metrics_by_specialty.csv",
    ) and ok
    ok = check_columns(
        PATHS.rolling_origin_metrics_by_waiting_size,
        ["waiting_list_size_group", *rolling_metric_columns],
        "csv",
        "outputs/rolling_origin_validation/rolling_origin_metrics_by_waiting_size_group.csv",
    ) and ok
    ok = check_columns(
        PATHS.rolling_origin_reliability,
        ["model_name", "target", "expected_coverage", "empirical_coverage", "coverage_error", "n_rows"],
        "csv",
        "outputs/rolling_origin_validation/rolling_origin_reliability_summary.csv",
    ) and ok
    ok = check_columns(
        PATHS.rolling_origin_quantile_crossing,
        ["model_name", COLUMNS.forecast_origin, COLUMNS.horizon, "n_rows", "crossing_rows", "quantile_crossing_rate_raw"],
        "csv",
        "outputs/rolling_origin_validation/rolling_origin_quantile_crossing_report.csv",
    ) and ok
    ok = check_rolling_origin_summary(PATHS.rolling_origin_summary) and ok
    ok = check_columns(PATHS.lp_allocation_output, LP_ALLOCATION_COLUMNS, "csv", "outputs/lp_allocation_output.csv") and ok
    ok = check_json_keys(PATHS.model_config, MODEL_CONFIG_KEYS, "models/model_config.json") and ok
    ok = check_json_keys(PATHS.feature_metadata, FEATURE_METADATA_KEYS, "models/feature_metadata.json") and ok
    ok = check_json_keys(PATHS.dta_model_config, MODEL_CONFIG_KEYS, "models/dta_model_config.json") and ok
    ok = check_json_keys(PATHS.dta_feature_metadata, FEATURE_METADATA_KEYS, "models/dta_feature_metadata.json") and ok
    ok = check_model_artifact_contract(PATHS.model_config, PATHS.feature_metadata, "incomplete-total TCN") and ok
    ok = check_model_artifact_contract(PATHS.dta_model_config, PATHS.dta_feature_metadata, "decision-to-admit TCN") and ok
    ok = check_covid_split_summary(PATHS.covid_split_summary) and ok
    ok = check_optional_shap_contracts() and ok

    shap_plots = sorted(PATHS.outputs_dir.glob("shap_trust_*.png"))
    if shap_plots:
        print(f"OK: found {len(shap_plots)} Trust SHAP waterfall PNG files")
    else:
        print(f"MISSING: no shap_trust_*.png files found in {PATHS.outputs_dir}")
        ok = False

    if ok:
        print("Preflight passed.")
        return 0
    print("Preflight failed. Re-run the Colab notebooks in order before launching the dashboard.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
