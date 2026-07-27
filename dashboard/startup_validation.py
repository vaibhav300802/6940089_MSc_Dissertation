from __future__ import annotations

import sys
from pathlib import Path
from typing import Mapping

import pandas as pd


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nhs_rtt_pipeline.config import (
    COLUMNS,
    get_paths,
    validate_future_forecast_frame,
    validate_optimisation_forecast_frame,
)


PATHS = get_paths(PROJECT_ROOT)


REQUIRED_FILES: Mapping[str, Path] = {
    "future_forecasts": PATHS.future_forecasts,
    "historical_processed_data": PATHS.clean_parquet,
    "model_config": PATHS.model_config,
    "model_comparison": PATHS.model_comparison,
    "model_comparison_by_horizon": PATHS.model_comparison_by_horizon,
    "future_optimisation_forecasts": PATHS.future_optimisation_forecasts,
    "capacity_scenario_config": PATHS.capacity_scenario_config,
}


OPTIONAL_FILES: Mapping[str, Path] = {
    "data_quality_summary": PATHS.data_quality_summary,
    "shap_global_importance": PATHS.shap_global_importance,
    "shap_group_importance": PATHS.shap_group_importance,
    "shap_local_explanations": PATHS.shap_local_explanations,
    "lp_allocation_output": PATHS.lp_allocation_output,
    "lp_sensitivity": PATHS.lp_sensitivity_output,
    "rolling_origin_reliability": PATHS.rolling_origin_reliability,
}


def build_startup_status(
    required_files: Mapping[str, Path] | None = None,
    optional_files: Mapping[str, Path] | None = None,
) -> pd.DataFrame:
    required_files = REQUIRED_FILES if required_files is None else required_files
    optional_files = OPTIONAL_FILES if optional_files is None else optional_files
    rows = []
    for required, files in [(True, required_files), (False, optional_files)]:
        for label, path in files.items():
            rows.append(
                {
                    "artifact": label,
                    "required": bool(required),
                    "exists": Path(path).exists(),
                    "path": str(path),
                }
            )
    return pd.DataFrame(rows)


def validate_dashboard_inputs(paths=PATHS) -> list[str]:
    errors: list[str] = []
    if paths.future_forecasts.exists():
        try:
            future = validate_future_forecast_frame(pd.read_parquet(paths.future_forecasts), str(paths.future_forecasts))
            if COLUMNS.actual in future.columns:
                errors.append("future_forecasts contains an actual column.")
        except Exception as exc:
            errors.append(f"future_forecasts could not be validated: {exc}")
    if paths.future_optimisation_forecasts.exists():
        try:
            validate_optimisation_forecast_frame(
                pd.read_parquet(paths.future_optimisation_forecasts),
                str(paths.future_optimisation_forecasts),
            )
        except Exception as exc:
            errors.append(f"future_optimisation_forecasts could not be validated: {exc}")
    if paths.model_comparison.exists():
        try:
            comparison = pd.read_csv(paths.model_comparison, nrows=5)
            for column in ["model", "mae", "rmse"]:
                if column not in comparison.columns:
                    errors.append(f"model_comparison is missing column: {column}")
        except Exception as exc:
            errors.append(f"model_comparison could not be read: {exc}")
    return errors


def main() -> int:
    status = build_startup_status()
    missing_required = status[status["required"] & ~status["exists"]]
    print("Dashboard startup status")
    print(status.to_string(index=False))
    validation_errors = validate_dashboard_inputs()
    if not missing_required.empty:
        print("\nMissing required dashboard files:")
        print(missing_required[["artifact", "path"]].to_string(index=False))
    if validation_errors:
        print("\nValidation errors:")
        for error in validation_errors:
            print(f"- {error}")
    if not missing_required.empty or validation_errors:
        return 1
    print("\nDashboard startup validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
