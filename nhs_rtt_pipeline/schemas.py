from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

import pandas as pd

from .config import (
    BACKTEST_PREDICTION_COLUMNS,
    CLEAN_RTT_COLUMNS,
    COLUMNS,
    FUTURE_FORECAST_COLUMNS,
    LP_ALLOCATION_COLUMNS,
    OPTIMISATION_FORECAST_COLUMNS,
    get_paths,
    require_columns,
    validate_backtest_predictions_frame,
    validate_future_forecast_frame,
    validate_optimisation_forecast_frame,
)


@dataclass(frozen=True)
class OutputSchema:
    name: str
    path_attr: str
    required_columns: tuple[str, ...]
    file_type: str
    validator: Optional[Callable[[pd.DataFrame, str], pd.DataFrame]] = None
    required_for_dashboard: bool = False


OUTPUT_SCHEMAS: tuple[OutputSchema, ...] = (
    OutputSchema("processed RTT data", "clean_parquet", tuple(CLEAN_RTT_COLUMNS), "parquet", None, True),
    OutputSchema("backtest predictions", "backtest_predictions", tuple(BACKTEST_PREDICTION_COLUMNS), "parquet", validate_backtest_predictions_frame, False),
    OutputSchema("DTA backtest predictions", "dta_backtest_predictions", tuple(BACKTEST_PREDICTION_COLUMNS), "parquet", validate_backtest_predictions_frame, False),
    OutputSchema("future forecasts", "future_forecasts", tuple(FUTURE_FORECAST_COLUMNS), "parquet", validate_future_forecast_frame, True),
    OutputSchema("future Part 2A optimisation forecasts", "future_optimisation_forecasts", tuple(OPTIMISATION_FORECAST_COLUMNS), "parquet", validate_optimisation_forecast_frame, True),
    OutputSchema("forecast metrics", "forecast_metrics", ("metric",), "csv", None, False),
    OutputSchema("DTA forecast metrics", "dta_forecast_metrics", ("metric",), "csv", None, False),
    OutputSchema("model comparison", "model_comparison", ("model",), "csv", None, False),
    OutputSchema("DTA model comparison", "dta_model_comparison", ("model",), "csv", None, False),
    OutputSchema("model comparison by horizon", "model_comparison_by_horizon", ("model", COLUMNS.horizon), "csv", None, False),
    OutputSchema("DTA model comparison by horizon", "dta_model_comparison_by_horizon", ("model", COLUMNS.horizon), "csv", None, False),
    OutputSchema("model comparison by specialty", "model_comparison_by_specialty", ("model", COLUMNS.specialty_code), "csv", None, False),
    OutputSchema("DTA model comparison by specialty", "dta_model_comparison_by_specialty", ("model", COLUMNS.specialty_code), "csv", None, False),
    OutputSchema("model comparison predictions", "model_comparison_predictions", ("model", COLUMNS.forecast_month, COLUMNS.actual), "parquet", None, False),
    OutputSchema("DTA model comparison predictions", "dta_model_comparison_predictions", ("model", COLUMNS.forecast_month, COLUMNS.actual), "parquet", None, False),
    OutputSchema("SHAP global feature importance", "shap_global_importance", ("feature_name",), "csv", None, False),
    OutputSchema("SHAP grouped feature importance", "shap_group_importance", ("feature_group",), "csv", None, False),
    OutputSchema("SHAP local explanations", "shap_local_explanations", ("trust_code", "specialty_code"), "parquet", None, False),
    OutputSchema("optimisation allocation", "lp_allocation_output", tuple(LP_ALLOCATION_COLUMNS), "csv", None, False),
    OutputSchema("data quality report", "data_quality_report", ("audit_section", "issue_type", "severity"), "csv", None, False),
    OutputSchema("missingness by series", "missingness_by_series", (COLUMNS.series_id,), "csv", None, False),
    OutputSchema("trust identifier changes", "trust_identifier_changes", (COLUMNS.trust_code,), "csv", None, False),
)


def validate_dataframe_schema(frame: pd.DataFrame, required_columns: Iterable[str], label: str) -> pd.DataFrame:
    required = list(required_columns)
    require_columns(frame, required, label)
    return frame


def _read_frame(path: Path, file_type: str) -> pd.DataFrame:
    if file_type == "parquet":
        return pd.read_parquet(path)
    if file_type == "csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported schema file type: {file_type}")


def _date_range_summary(frame: pd.DataFrame) -> str:
    date_columns = [column for column in ["month", COLUMNS.forecast_month, COLUMNS.forecast_origin] if column in frame.columns]
    summaries: list[str] = []
    for column in date_columns:
        values = pd.to_datetime(frame[column], errors="coerce")
        if values.notna().any():
            summaries.append(f"{column}:{values.min().date()}..{values.max().date()}")
    return "; ".join(summaries)


def validate_output_schema(schema: OutputSchema, project_root: str | Path | None = None) -> dict[str, object]:
    paths = get_paths(project_root)
    path = getattr(paths, schema.path_attr)
    row: dict[str, object] = {
        "artifact": schema.name,
        "path": str(path),
        "required_for_dashboard": bool(schema.required_for_dashboard),
        "exists": path.exists(),
        "status": "missing",
        "row_count": None,
        "date_range": "",
        "missing_columns": "",
        "warning": "",
    }
    if not path.exists():
        return row
    try:
        frame = _read_frame(path, schema.file_type)
        validate_dataframe_schema(frame, schema.required_columns, schema.name)
        if schema.validator is not None:
            frame = schema.validator(frame, schema.name)
        row["status"] = "ok"
        row["row_count"] = int(len(frame))
        row["date_range"] = _date_range_summary(frame)
        return row
    except Exception as exc:
        missing = []
        try:
            frame = _read_frame(path, schema.file_type)
            missing = [column for column in schema.required_columns if column not in frame.columns]
        except Exception:
            pass
        row["status"] = "failed"
        row["missing_columns"] = ", ".join(missing)
        row["warning"] = str(exc)
        return row


def validate_generated_outputs(project_root: str | Path | None = None) -> pd.DataFrame:
    return pd.DataFrame([validate_output_schema(schema, project_root) for schema in OUTPUT_SCHEMAS])


def assert_required_outputs_valid(project_root: str | Path | None = None) -> pd.DataFrame:
    report = validate_generated_outputs(project_root)
    failed = report[(report["required_for_dashboard"]) & (report["status"] != "ok")]
    if not failed.empty:
        details = failed[["artifact", "path", "status", "warning"]].to_dict(orient="records")
        raise FileNotFoundError(f"Required dashboard outputs are missing or invalid: {details}")
    return report
