from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pandas as pd


@dataclass(frozen=True)
class CanonicalColumns:
    trust_code: str = "trust_code"
    trust_name: str = "trust_name"
    specialty_code: str = "specialty_code"
    specialty_name: str = "specialty_name"
    forecast_month: str = "forecast_month"
    p10: str = "p10"
    p50: str = "p50"
    p90: str = "p90"
    actual: str = "actual"
    forecast_origin: str = "forecast_origin"
    horizon: str = "horizon"
    series_id: str = "series_id"
    current_waiting_list_size: str = "current_waiting_list_size"
    latest_observed_waiting_list: str = "latest_observed_waiting_list"
    incomplete_total: str = "incomplete_total"
    incomplete_decision_to_admit: str = "incomplete_decision_to_admit"
    latest_observed_incomplete_decision_to_admit: str = "latest_observed_incomplete_decision_to_admit"
    forecast_target: str = "forecast_target"
    is_surgical_specialty: str = "is_surgical_specialty"
    specialty_inclusion_criteria: str = "specialty_inclusion_criteria"


COLUMNS = CanonicalColumns()

CLEAN_RTT_COLUMNS = [
    "month",
    "source_trust_code",
    "source_trust_name",
    "source_specialty_code",
    "source_specialty_name",
    COLUMNS.trust_code,
    COLUMNS.trust_name,
    COLUMNS.specialty_code,
    COLUMNS.specialty_name,
    "trust_identifier_harmonisation_rule",
    "specialty_identifier_harmonisation_rule",
    "source_file_count",
    "source_row_count",
    "source_zips",
    "source_csvs",
    "source_urls",
    "source_publication_months",
    "waiting_list",
    "incomplete_total",
    "completed_admitted",
    "completed_non_admitted",
    "waiting_list_with_dta",
    "incomplete_decision_to_admit",
    "incomplete_total_source_available",
    "incomplete_decision_to_admit_source_available",
    "new_rtt_periods",
    "completed_total",
    "net_inflow",
    "opening_waiting_list",
    "closing_waiting_list",
    "reported_net_inflow",
    "unreported_removals",
    "reconciliation_error",
    "reconciliation_error_abs",
    "new_rtt_periods_missing",
    "completed_admitted_missing",
    "completed_non_admitted_missing",
    "completed_total_missing",
    "opening_waiting_list_missing",
    "closing_waiting_list_missing",
    "reported_net_inflow_missing",
    "unreported_removals_missing",
    "flow_components_missing",
    "new_rtt_periods_lag1",
    "new_rtt_periods_lag3",
    "new_rtt_periods_lag6",
    "completed_total_lag1",
    "completed_total_lag3",
    "completed_total_lag6",
    "reported_net_inflow_lag1",
    "reported_net_inflow_lag3",
    "reported_net_inflow_lag6",
    "unreported_removals_lag1",
    "unreported_removals_lag3",
    "unreported_removals_lag6",
    "new_rtt_periods_lag1_missing",
    "new_rtt_periods_lag3_missing",
    "new_rtt_periods_lag6_missing",
    "completed_total_lag1_missing",
    "completed_total_lag3_missing",
    "completed_total_lag6_missing",
    "reported_net_inflow_lag1_missing",
    "reported_net_inflow_lag3_missing",
    "reported_net_inflow_lag6_missing",
    "unreported_removals_lag1_missing",
    "unreported_removals_lag3_missing",
    "unreported_removals_lag6_missing",
    "is_imputed_month",
    "missing_month",
    "observed_month",
    "waiting_list_imputed",
    "waiting_list_with_dta_imputed",
    "completed_admitted_imputed",
    "completed_non_admitted_imputed",
    "new_rtt_periods_imputed",
    "series_id",
    "time_idx",
    "calendar_month",
    "month_sin",
    "month_cos",
    COLUMNS.is_surgical_specialty,
    COLUMNS.specialty_inclusion_criteria,
]

BACKTEST_PREDICTION_COLUMNS = [
    COLUMNS.trust_code,
    COLUMNS.trust_name,
    COLUMNS.specialty_code,
    COLUMNS.specialty_name,
    COLUMNS.forecast_origin,
    COLUMNS.forecast_month,
    COLUMNS.horizon,
    COLUMNS.p10,
    COLUMNS.p50,
    COLUMNS.p90,
    COLUMNS.actual,
]

FUTURE_FORECAST_COLUMNS = [
    COLUMNS.trust_code,
    COLUMNS.trust_name,
    COLUMNS.specialty_code,
    COLUMNS.specialty_name,
    COLUMNS.forecast_origin,
    COLUMNS.forecast_month,
    COLUMNS.horizon,
    COLUMNS.p10,
    COLUMNS.p50,
    COLUMNS.p90,
    COLUMNS.latest_observed_waiting_list,
]

OPTIMISATION_FORECAST_COLUMNS = [
    COLUMNS.trust_code,
    COLUMNS.trust_name,
    COLUMNS.specialty_code,
    COLUMNS.specialty_name,
    COLUMNS.forecast_origin,
    COLUMNS.forecast_month,
    COLUMNS.horizon,
    COLUMNS.p10,
    COLUMNS.p50,
    COLUMNS.p90,
    COLUMNS.latest_observed_incomplete_decision_to_admit,
    COLUMNS.forecast_target,
    COLUMNS.is_surgical_specialty,
    COLUMNS.specialty_inclusion_criteria,
]

FORECAST_COLUMNS = BACKTEST_PREDICTION_COLUMNS

ROLLING_ORIGIN_PREDICTION_COLUMNS = [
    "origin_index",
    "model_name",
    COLUMNS.series_id,
    COLUMNS.trust_code,
    COLUMNS.trust_name,
    "trust",
    COLUMNS.specialty_code,
    COLUMNS.specialty_name,
    "specialty",
    COLUMNS.forecast_origin,
    COLUMNS.forecast_month,
    COLUMNS.horizon,
    "target_column",
    "training_end_month",
    "scaler_fit_end_month",
    "p10_raw",
    "p50_raw",
    "p90_raw",
    "quantile_crossing_raw",
    COLUMNS.p10,
    COLUMNS.p50,
    COLUMNS.p90,
    COLUMNS.actual,
]

LP_ALLOCATION_COLUMNS = [
    "trust_code",
    "trust_name",
    "specialty_code",
    "specialty_name",
    "forecast_month",
    "baseline_predicted_backlog",
    "patients_completed_per_session",
    "max_feasible_additional_sessions",
    "sessions_allocated",
    "simulated_completed_pathways",
    "remaining_backlog",
    "unused_feasible_sessions",
    "allocation_cost",
    "percent_reduction",
]

SURGICAL_SPECIALTY_INCLUSION: Mapping[str, str] = {
    "100": "Included: General Surgery treatment function commonly includes operative admitted pathways.",
    "101": "Included: Urology treatment function commonly includes operative admitted pathways.",
    "102": "Included: Transplantation Surgery treatment function.",
    "103": "Included: Breast Surgery treatment function.",
    "104": "Included: Colorectal Surgery treatment function.",
    "105": "Included: Hepatobiliary and Pancreatic Surgery treatment function.",
    "106": "Included: Upper Gastrointestinal Surgery treatment function.",
    "107": "Included: Vascular Surgery treatment function.",
    "108": "Included: Spinal Surgery treatment function.",
    "110": "Included: Trauma and Orthopaedics treatment function commonly includes admitted operative pathways.",
    "120": "Included: Ear, Nose and Throat treatment function commonly includes operative pathways.",
    "130": "Included: Ophthalmology treatment function commonly includes operative pathways.",
    "140": "Included: Oral Surgery treatment function.",
    "145": "Included: Oral and Maxillofacial Surgery treatment function.",
    "150": "Included: Neurosurgery treatment function.",
    "160": "Included: Plastic Surgery treatment function.",
    "170": "Included: Cardiothoracic Surgery treatment function.",
    "171": "Included: Paediatric Surgery treatment function.",
    "172": "Included: Cardiac Surgery treatment function.",
    "173": "Included: Thoracic Surgery treatment function.",
    "502": "Included: Gynaecology treatment function has admitted operative pathways.",
}

SURGICAL_SPECIALTY_INCLUSION_CRITERIA = (
    "Included specialties are RTT treatment-function codes whose planned admitted pathways commonly require "
    "additional treatment capacity. Clearly non-surgical medical, diagnostic, and therapy specialties are excluded "
    "from the decision-to-admit capacity simulation unless explicitly added to SURGICAL_SPECIALTY_INCLUSION."
)


@dataclass(frozen=True)
class ProjectPaths:
    project_root: Path
    config_dir: Path
    data_dir: Path
    raw_dir: Path
    raw_zip_dir: Path
    processed_dir: Path
    models_dir: Path
    outputs_dir: Path
    logs_dir: Path
    run_summaries_dir: Path
    dashboard_dir: Path
    dashboard_data_dir: Path
    forecast_plot_dir: Path
    pipeline_config: Path
    clean_parquet: Path
    tcn_state_dict: Path
    dta_tcn_state_dict: Path
    model_config: Path
    dta_model_config: Path
    feature_metadata: Path
    dta_feature_metadata: Path
    data_dictionary: Path
    capacity_scenario_config: Path
    training_history: Path
    dta_training_history: Path
    dta_backtest_predictions: Path
    dta_forecast_metrics: Path
    dta_forecast_metrics_by_horizon: Path
    dta_model_comparison_predictions: Path
    dta_model_comparison: Path
    dta_model_comparison_by_horizon: Path
    dta_model_comparison_by_specialty: Path
    dta_model_comparison_by_trust_size: Path
    dta_model_comparison_by_covid_period: Path
    dta_model_comparison_paired_errors: Path
    dta_model_comparison_audit_log: Path
    dta_model_comparison_summary: Path
    dta_reliability_summary: Path
    net_inflow_quality: Path
    flow_reconciliation_quality: Path
    data_quality_report: Path
    data_quality_summary: Path
    missingness_by_series: Path
    trust_identifier_changes: Path
    backtest_predictions: Path
    future_forecasts: Path
    future_optimisation_forecasts: Path
    part2a_coverage_report: Path
    surgical_specialties: Path
    forecast_metrics: Path
    forecast_metrics_by_horizon: Path
    forecast_plot_index: Path
    model_comparison_predictions: Path
    model_comparison: Path
    model_comparison_by_horizon: Path
    model_comparison_by_specialty: Path
    model_comparison_by_trust_size: Path
    model_comparison_by_covid_period: Path
    model_comparison_paired_errors: Path
    model_comparison_audit_log: Path
    model_comparison_summary: Path
    model_comparison_plot_dir: Path
    model_comparison_overall_png: Path
    model_comparison_by_horizon_png: Path
    model_comparison_tcn_vs_seasonal_png: Path
    rolling_origin_dir: Path
    rolling_origin_predictions: Path
    rolling_origin_metrics: Path
    rolling_origin_metrics_by_origin: Path
    rolling_origin_metrics_by_horizon: Path
    rolling_origin_metrics_by_trust: Path
    rolling_origin_metrics_by_specialty: Path
    rolling_origin_metrics_by_waiting_size: Path
    rolling_origin_reliability: Path
    rolling_origin_quantile_crossing: Path
    rolling_origin_summary: Path
    rolling_origin_calibration_png: Path
    rolling_origin_interval_width_coverage_png: Path
    shap_values: Path
    shap_values_aggregated: Path
    shap_feature_names_json: Path
    shap_feature_names_npy: Path
    shap_feature_groups_json: Path
    shap_aggregated_feature_names_json: Path
    shap_background: Path
    shap_test_matrix: Path
    shap_context_index: Path
    shap_expected_values: Path
    shap_model_outputs: Path
    shap_values_long: Path
    shap_global_importance: Path
    shap_horizon_importance: Path
    shap_local_explanations: Path
    shap_consistency_report: Path
    shap_audit_log: Path
    shap_methodology_note: Path
    shap_global_summary: Path
    shap_group_importance: Path
    shap_latest_predictions_by_series: Path
    shap_top_trusts: Path
    shap_interpretations: Path
    shap_waterfall_index: Path
    lp_allocation_output: Path
    lp_reduction_by_trust_specialty: Path
    lp_sensitivity_output: Path
    lp_sensitivity_png: Path
    lp_uncertainty_comparison: Path
    lp_covid_stress_test: Path
    covid_stress_test_dir: Path
    covid_predictions: Path
    covid_metrics: Path
    covid_degradation: Path
    covid_split_summary: Path
    covid_methodology_note: Path
    covid_actual_vs_forecast_png: Path
    covid_prediction_intervals_png: Path
    covid_error_over_time_png: Path
    covid_performance_by_trust_png: Path
    covid_performance_by_specialty_png: Path


def detect_project_root() -> Path:
    env_value = os.environ.get("NHS_RTT_PROJECT_DIR")
    if env_value:
        return Path(env_value).expanduser().resolve()

    package_project_root = Path(__file__).resolve().parents[1]
    if (package_project_root / "nhs_rtt_pipeline").exists():
        return package_project_root

    for candidate in [Path.cwd(), Path.cwd().parent, Path("/content/nhs_rtt_msc_project"), Path("/content")]:
        if (candidate / "nhs_rtt_pipeline").exists() or (candidate / "project_notebooks").exists():
            return candidate.resolve()
    return Path("/content").resolve()


def get_paths(project_root: str | Path | None = None) -> ProjectPaths:
    root = Path(project_root).expanduser().resolve() if project_root is not None else detect_project_root()
    
    def resolved_path(env_name: str, default_relative: str | Path) -> Path:
        configured = os.environ.get(env_name)
        value = Path(configured).expanduser() if configured else Path(default_relative)
        if not value.is_absolute():
            value = root / value
        return value.resolve()

    config_dir = resolved_path("NHS_RTT_CONFIG_DIR", "config")
    data_dir = resolved_path("NHS_RTT_DATA_DIR", "data")
    raw_dir = resolved_path("NHS_RTT_RAW_DIR", data_dir / "raw")
    processed_dir = resolved_path("NHS_RTT_PROCESSED_DIR", data_dir / "processed")
    models_dir = resolved_path("NHS_RTT_MODELS_DIR", "models")
    outputs_dir = resolved_path("NHS_RTT_OUTPUTS_DIR", "outputs")
    logs_dir = resolved_path("NHS_RTT_LOGS_DIR", outputs_dir / "logs")
    run_summaries_dir = resolved_path("NHS_RTT_RUN_SUMMARIES_DIR", outputs_dir / "run_summaries")
    dashboard_data_dir = resolved_path("NHS_RTT_DASHBOARD_DATA_DIR", "dashboard/data")
    covid_stress_test_dir = outputs_dir / "covid_stress_test"
    rolling_origin_dir = outputs_dir / "rolling_origin_validation"
    return ProjectPaths(
        project_root=root,
        config_dir=config_dir,
        data_dir=data_dir,
        raw_dir=raw_dir,
        raw_zip_dir=resolved_path("NHS_RTT_RAW_ZIP_DIR", raw_dir / "zips"),
        processed_dir=processed_dir,
        models_dir=models_dir,
        outputs_dir=outputs_dir,
        logs_dir=logs_dir,
        run_summaries_dir=run_summaries_dir,
        dashboard_dir=root / "dashboard",
        dashboard_data_dir=dashboard_data_dir,
        forecast_plot_dir=outputs_dir / "forecast_plots_by_trust",
        pipeline_config=config_dir / "pipeline_config.json",
        clean_parquet=processed_dir / "rtt_clean.parquet",
        tcn_state_dict=models_dir / "tcn_state_dict.pt",
        dta_tcn_state_dict=models_dir / "tcn_dta_state_dict.pt",
        model_config=models_dir / "model_config.json",
        dta_model_config=models_dir / "dta_model_config.json",
        feature_metadata=models_dir / "feature_metadata.json",
        dta_feature_metadata=models_dir / "dta_feature_metadata.json",
        data_dictionary=models_dir / "data_dictionary.csv",
        capacity_scenario_config=models_dir / "capacity_productivity_scenarios.json",
        training_history=outputs_dir / "training_history.csv",
        dta_training_history=outputs_dir / "dta_training_history.csv",
        dta_backtest_predictions=outputs_dir / "dta_backtest_predictions.parquet",
        dta_forecast_metrics=outputs_dir / "dta_forecast_metrics.csv",
        dta_forecast_metrics_by_horizon=outputs_dir / "dta_forecast_metrics_by_horizon.csv",
        dta_model_comparison_predictions=outputs_dir / "dta_model_comparison_predictions.parquet",
        dta_model_comparison=outputs_dir / "dta_model_comparison.csv",
        dta_model_comparison_by_horizon=outputs_dir / "dta_model_comparison_by_horizon.csv",
        dta_model_comparison_by_specialty=outputs_dir / "dta_model_comparison_by_specialty.csv",
        dta_model_comparison_by_trust_size=outputs_dir / "dta_model_comparison_by_trust_size.csv",
        dta_model_comparison_by_covid_period=outputs_dir / "dta_model_comparison_by_covid_period.csv",
        dta_model_comparison_paired_errors=outputs_dir / "dta_model_comparison_paired_error_analysis.csv",
        dta_model_comparison_audit_log=outputs_dir / "dta_model_comparison_audit_log.csv",
        dta_model_comparison_summary=outputs_dir / "dta_model_comparison_summary.md",
        dta_reliability_summary=outputs_dir / "dta_reliability_summary.csv",
        net_inflow_quality=outputs_dir / "net_inflow_data_quality_summary.csv",
        flow_reconciliation_quality=outputs_dir / "rtt_flow_reconciliation_quality_report.json",
        data_quality_report=outputs_dir / "data_quality_report.csv",
        data_quality_summary=outputs_dir / "data_quality_summary.md",
        missingness_by_series=outputs_dir / "missingness_by_series.csv",
        trust_identifier_changes=outputs_dir / "trust_identifier_changes.csv",
        backtest_predictions=outputs_dir / "backtest_predictions.parquet",
        future_forecasts=outputs_dir / "future_forecasts.parquet",
        future_optimisation_forecasts=outputs_dir / "future_optimisation_forecasts.parquet",
        part2a_coverage_report=outputs_dir / "part2a_coverage_report.csv",
        surgical_specialties=models_dir / "surgical_specialties.csv",
        forecast_metrics=outputs_dir / "forecast_metrics.csv",
        forecast_metrics_by_horizon=outputs_dir / "forecast_metrics_by_horizon.csv",
        forecast_plot_index=outputs_dir / "forecast_plot_index.csv",
        model_comparison_predictions=outputs_dir / "model_comparison_predictions.parquet",
        model_comparison=outputs_dir / "model_comparison.csv",
        model_comparison_by_horizon=outputs_dir / "model_comparison_by_horizon.csv",
        model_comparison_by_specialty=outputs_dir / "model_comparison_by_specialty.csv",
        model_comparison_by_trust_size=outputs_dir / "model_comparison_by_trust_size.csv",
        model_comparison_by_covid_period=outputs_dir / "model_comparison_by_covid_period.csv",
        model_comparison_paired_errors=outputs_dir / "model_comparison_paired_error_analysis.csv",
        model_comparison_audit_log=outputs_dir / "model_comparison_audit_log.csv",
        model_comparison_summary=outputs_dir / "model_comparison_summary.md",
        model_comparison_plot_dir=outputs_dir / "model_comparison_plots",
        model_comparison_overall_png=outputs_dir / "model_comparison_plots" / "overall_model_comparison.png",
        model_comparison_by_horizon_png=outputs_dir / "model_comparison_plots" / "mae_by_horizon.png",
        model_comparison_tcn_vs_seasonal_png=outputs_dir / "model_comparison_plots" / "tcn_vs_seasonal_naive_abs_error.png",
        rolling_origin_dir=rolling_origin_dir,
        rolling_origin_predictions=rolling_origin_dir / "rolling_origin_predictions.parquet",
        rolling_origin_metrics=rolling_origin_dir / "rolling_origin_metrics_overall.csv",
        rolling_origin_metrics_by_origin=rolling_origin_dir / "rolling_origin_metrics_by_origin.csv",
        rolling_origin_metrics_by_horizon=rolling_origin_dir / "rolling_origin_metrics_by_horizon.csv",
        rolling_origin_metrics_by_trust=rolling_origin_dir / "rolling_origin_metrics_by_trust.csv",
        rolling_origin_metrics_by_specialty=rolling_origin_dir / "rolling_origin_metrics_by_specialty.csv",
        rolling_origin_metrics_by_waiting_size=rolling_origin_dir / "rolling_origin_metrics_by_waiting_size_group.csv",
        rolling_origin_reliability=rolling_origin_dir / "rolling_origin_reliability_summary.csv",
        rolling_origin_quantile_crossing=rolling_origin_dir / "rolling_origin_quantile_crossing_report.csv",
        rolling_origin_summary=rolling_origin_dir / "rolling_origin_summary.md",
        rolling_origin_calibration_png=rolling_origin_dir / "calibration_expected_vs_empirical.png",
        rolling_origin_interval_width_coverage_png=rolling_origin_dir / "interval_width_vs_coverage.png",
        shap_values=outputs_dir / "shap_values.npy",
        shap_values_aggregated=outputs_dir / "shap_values_aggregated.npy",
        shap_feature_names_json=outputs_dir / "shap_feature_names.json",
        shap_feature_names_npy=outputs_dir / "shap_feature_names.npy",
        shap_feature_groups_json=outputs_dir / "shap_feature_groups.json",
        shap_aggregated_feature_names_json=outputs_dir / "shap_aggregated_feature_names.json",
        shap_background=outputs_dir / "shap_background.npy",
        shap_test_matrix=outputs_dir / "shap_test_matrix.npy",
        shap_context_index=outputs_dir / "shap_explained_instances.csv",
        shap_expected_values=outputs_dir / "shap_expected_values.npy",
        shap_model_outputs=outputs_dir / "shap_model_outputs.npy",
        shap_values_long=outputs_dir / "shap_values_long.parquet",
        shap_global_importance=outputs_dir / "shap_global_feature_importance.csv",
        shap_horizon_importance=outputs_dir / "shap_horizon_feature_importance.csv",
        shap_local_explanations=outputs_dir / "shap_local_trust_specialty_explanations.parquet",
        shap_consistency_report=outputs_dir / "shap_local_consistency_report.csv",
        shap_audit_log=outputs_dir / "shap_audit_log.csv",
        shap_methodology_note=outputs_dir / "shap_methodology_note.md",
        shap_global_summary=outputs_dir / "shap_global_summary.png",
        shap_group_importance=outputs_dir / "shap_group_importance.csv",
        shap_latest_predictions_by_series=outputs_dir / "shap_latest_predictions_by_series.csv",
        shap_top_trusts=outputs_dir / "shap_top_predicted_trusts.csv",
        shap_interpretations=outputs_dir / "shap_trust_interpretations.csv",
        shap_waterfall_index=outputs_dir / "shap_trust_waterfall_index.csv",
        lp_allocation_output=outputs_dir / "lp_allocation_output.csv",
        lp_reduction_by_trust_specialty=outputs_dir / "lp_reduction_by_trust_specialty.csv",
        lp_sensitivity_output=outputs_dir / "lp_sensitivity.csv",
        lp_sensitivity_png=outputs_dir / "lp_sensitivity.png",
        lp_uncertainty_comparison=outputs_dir / "lp_uncertainty_comparison.csv",
        lp_covid_stress_test=outputs_dir / "lp_covid_stress_test.csv",
        covid_stress_test_dir=covid_stress_test_dir,
        covid_predictions=covid_stress_test_dir / "covid_shock_predictions.parquet",
        covid_metrics=covid_stress_test_dir / "covid_shock_metrics.csv",
        covid_degradation=covid_stress_test_dir / "covid_shock_degradation.csv",
        covid_split_summary=covid_stress_test_dir / "split_summary.json",
        covid_methodology_note=covid_stress_test_dir / "methodology_note.md",
        covid_actual_vs_forecast_png=covid_stress_test_dir / "actual_vs_forecast_covid_shock.png",
        covid_prediction_intervals_png=covid_stress_test_dir / "prediction_intervals_covid_shock_tcn.png",
        covid_error_over_time_png=covid_stress_test_dir / "error_over_time.png",
        covid_performance_by_trust_png=covid_stress_test_dir / "performance_by_trust.png",
        covid_performance_by_specialty_png=covid_stress_test_dir / "performance_by_specialty.png",
    )


def ensure_directories(paths: ProjectPaths) -> None:
    for directory in [
        paths.raw_dir,
        paths.raw_zip_dir,
        paths.processed_dir,
        paths.models_dir,
        paths.outputs_dir,
        paths.config_dir,
        paths.logs_dir,
        paths.run_summaries_dir,
        paths.forecast_plot_dir,
        paths.model_comparison_plot_dir,
        paths.rolling_origin_dir,
        paths.dashboard_data_dir,
        paths.covid_stress_test_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def require_file(path: str | Path, label: str) -> Path:
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Missing required {label}: {resolved}")
    return resolved


def require_columns(frame: pd.DataFrame, required_columns: Sequence[str], label: str) -> None:
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def normalise_text(value: object) -> str:
    return " ".join(str(value).strip().split())


def rename_legacy_forecast_columns(frame: pd.DataFrame) -> pd.DataFrame:
    legacy_mapping: Mapping[str, str] = {
        "month": COLUMNS.forecast_month,
        "predicted_p10": COLUMNS.p10,
        "predicted_p50": COLUMNS.p50,
        "predicted_p90": COLUMNS.p90,
        "current_waiting_list_size": COLUMNS.actual,
        "specialty": COLUMNS.specialty_name,
    }
    rename_map = {
        old: new
        for old, new in legacy_mapping.items()
        if old in frame.columns and new not in frame.columns
    }
    return frame.rename(columns=rename_map)


def validate_forecast_frame(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    return validate_backtest_predictions_frame(frame, label)


def _validate_common_forecast_fields(frame: pd.DataFrame, required_columns: Sequence[str], label: str) -> pd.DataFrame:
    frame = frame.copy()
    require_columns(frame, required_columns, label)
    frame[COLUMNS.forecast_month] = pd.to_datetime(frame[COLUMNS.forecast_month], errors="coerce")
    frame[COLUMNS.forecast_origin] = pd.to_datetime(frame[COLUMNS.forecast_origin], errors="coerce")
    bad_dates = frame[COLUMNS.forecast_month].isna().sum() + frame[COLUMNS.forecast_origin].isna().sum()
    if bad_dates:
        raise ValueError(f"{label} contains {int(bad_dates)} invalid forecast/origin date values.")
    if (frame[COLUMNS.forecast_month] <= frame[COLUMNS.forecast_origin]).any():
        raise ValueError(f"{label} contains forecast_month values that are not later than forecast_origin.")
    for column in [COLUMNS.p10, COLUMNS.p50, COLUMNS.p90]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[[COLUMNS.p10, COLUMNS.p50, COLUMNS.p90]].isna().any().any():
        raise ValueError(f"{label} contains missing quantile forecast values.")
    frame[COLUMNS.horizon] = pd.to_numeric(frame[COLUMNS.horizon], errors="coerce")
    if frame[COLUMNS.horizon].isna().any():
        raise ValueError(f"{label} contains missing or non-numeric horizon values.")
    frame[COLUMNS.horizon] = frame[COLUMNS.horizon].astype(int)
    if (frame[COLUMNS.horizon] <= 0).any():
        raise ValueError(f"{label} contains non-positive horizon values.")
    month_delta = (
        (frame[COLUMNS.forecast_month].dt.year - frame[COLUMNS.forecast_origin].dt.year) * 12
        + (frame[COLUMNS.forecast_month].dt.month - frame[COLUMNS.forecast_origin].dt.month)
    ).astype(int)
    if not (month_delta == frame[COLUMNS.horizon]).all():
        raise ValueError(f"{label} contains horizon values inconsistent with forecast_origin and forecast_month.")
    for column in [COLUMNS.trust_code, COLUMNS.trust_name, COLUMNS.specialty_code, COLUMNS.specialty_name]:
        frame[column] = frame[column].map(normalise_text)
    return frame


def validate_backtest_predictions_frame(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    frame = rename_legacy_forecast_columns(frame.copy())
    frame = _validate_common_forecast_fields(frame, BACKTEST_PREDICTION_COLUMNS, label)
    frame[COLUMNS.actual] = pd.to_numeric(frame[COLUMNS.actual], errors="coerce")
    if frame[COLUMNS.actual].isna().any():
        missing_count = int(frame[COLUMNS.actual].isna().sum())
        raise ValueError(f"{label} contains {missing_count} backtest rows without actual values.")
    return frame


def validate_future_forecast_frame(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    frame = frame.copy()
    if COLUMNS.actual in frame.columns:
        raise ValueError(f"{label} must not contain an '{COLUMNS.actual}' column.")
    frame = _validate_common_forecast_fields(frame, FUTURE_FORECAST_COLUMNS, label)
    frame[COLUMNS.latest_observed_waiting_list] = pd.to_numeric(
        frame[COLUMNS.latest_observed_waiting_list],
        errors="coerce",
    )
    if frame[COLUMNS.latest_observed_waiting_list].isna().any():
        missing_count = int(frame[COLUMNS.latest_observed_waiting_list].isna().sum())
        raise ValueError(f"{label} contains {missing_count} rows without latest observed waiting list values.")
    return frame


def validate_optimisation_forecast_frame(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    frame = frame.copy()
    if COLUMNS.actual in frame.columns:
        raise ValueError(f"{label} must not contain an '{COLUMNS.actual}' column.")
    frame = _validate_common_forecast_fields(frame, OPTIMISATION_FORECAST_COLUMNS, label)
    frame[COLUMNS.latest_observed_incomplete_decision_to_admit] = pd.to_numeric(
        frame[COLUMNS.latest_observed_incomplete_decision_to_admit],
        errors="coerce",
    )
    if frame[COLUMNS.latest_observed_incomplete_decision_to_admit].isna().any():
        missing_count = int(frame[COLUMNS.latest_observed_incomplete_decision_to_admit].isna().sum())
        raise ValueError(f"{label} contains {missing_count} rows without latest observed Part 2A values.")
    frame[COLUMNS.forecast_target] = frame[COLUMNS.forecast_target].astype(str)
    wrong_target = frame[COLUMNS.forecast_target] != COLUMNS.incomplete_decision_to_admit
    if wrong_target.any():
        bad_values = sorted(frame.loc[wrong_target, COLUMNS.forecast_target].dropna().unique().tolist())
        raise ValueError(
            f"{label} must contain only forecast_target='{COLUMNS.incomplete_decision_to_admit}', "
            f"but found {bad_values}."
        )
    frame[COLUMNS.is_surgical_specialty] = frame[COLUMNS.is_surgical_specialty].astype(bool)
    if not frame[COLUMNS.is_surgical_specialty].all():
        bad_count = int((~frame[COLUMNS.is_surgical_specialty]).sum())
        raise ValueError(f"{label} contains {bad_count} non-surgical rows.")
    if frame[COLUMNS.specialty_inclusion_criteria].astype(str).str.strip().eq("").any():
        raise ValueError(f"{label} contains empty specialty inclusion criteria.")
    return frame


def read_parquet_checked(path: str | Path, required_columns: Iterable[str], label: str) -> pd.DataFrame:
    require_file(path, label)
    frame = pd.read_parquet(path)
    require_columns(frame, list(required_columns), label)
    return frame


def read_csv_checked(path: str | Path, required_columns: Iterable[str], label: str) -> pd.DataFrame:
    require_file(path, label)
    frame = pd.read_csv(path)
    require_columns(frame, list(required_columns), label)
    return frame
