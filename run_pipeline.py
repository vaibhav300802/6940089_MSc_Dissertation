from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping

from nhs_rtt_pipeline.config import ensure_directories, get_paths, require_file
from nhs_rtt_pipeline.reproducibility import set_global_seed
from nhs_rtt_pipeline.schemas import validate_generated_outputs
from nhs_rtt_pipeline.settings import load_pipeline_settings, write_effective_settings_summary


PROJECT_ROOT = Path(__file__).resolve().parent
PATHS = get_paths(PROJECT_ROOT)

STAGE_ORDER = [
    "download",
    "prepare",
    "train",
    "backtest",
    "forecast",
    "rolling_origin",
    "explain",
    "optimise",
    "covid_shock",
    "validate",
]
STAGE_ALIASES = {
    "rolling-origin": "rolling_origin",
    "rolling_origin_validation": "rolling_origin",
    "covid": "covid_shock",
    "covid-shock": "covid_shock",
    "covid_shock_experiment": "covid_shock",
}
LAYER1_STAGES = {"prepare", "train", "backtest", "forecast"}
LAYER1_MODEL_STAGES = {"train", "backtest", "forecast"}


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def configure_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def merge_env(*parts: Mapping[str, str] | None) -> dict[str, str]:
    merged: dict[str, str] = {}
    for part in parts:
        if part:
            merged.update({str(key): str(value) for key, value in part.items()})
    return merged


def resolve_project_path(value: str | os.PathLike[str], root: Path = PROJECT_ROOT) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    return str(path.resolve())


def settings_environment(settings_config_path: Path, settings: object) -> dict[str, str]:
    data_dirs = dict(getattr(settings, "data_dirs", {}) or {})
    output_dirs = dict(getattr(settings, "output_dirs", {}) or {})
    env = {
        "NHS_RTT_PROJECT_DIR": str(PROJECT_ROOT),
        "NHS_RTT_PIPELINE_CONFIG": str(settings_config_path),
    }
    directory_mapping = {
        "NHS_RTT_RAW_DIR": data_dirs.get("raw"),
        "NHS_RTT_PROCESSED_DIR": data_dirs.get("processed"),
        "NHS_RTT_MODELS_DIR": output_dirs.get("models"),
        "NHS_RTT_OUTPUTS_DIR": output_dirs.get("outputs"),
        "NHS_RTT_LOGS_DIR": output_dirs.get("logs"),
        "NHS_RTT_RUN_SUMMARIES_DIR": output_dirs.get("run_summaries"),
        "NHS_RTT_DASHBOARD_DATA_DIR": output_dirs.get("dashboard_data"),
    }
    for env_name, configured_path in directory_mapping.items():
        if configured_path:
            env[env_name] = resolve_project_path(str(configured_path))
    return env


def run_command(command: list[str], stage: str, extra_env: dict[str, str] | None = None) -> None:
    logging.info("Starting stage command for %s: %s", stage, " ".join(command))
    command_env = None
    if extra_env:
        command_env = os.environ.copy()
        command_env.update(extra_env)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True, env=command_env)
    logging.info("Completed stage command for %s", stage)


def require_prerequisites(stage: str) -> None:
    if stage == "download":
        required = {}
    elif stage == "explain":
        required = {
            "processed RTT data": PATHS.clean_parquet,
            "TCN state dictionary": PATHS.tcn_state_dict,
            "TCN model configuration": PATHS.model_config,
            "feature metadata": PATHS.feature_metadata,
        }
    elif stage == "optimise":
        required = {
            "future Part 2A optimisation forecasts": PATHS.future_optimisation_forecasts,
        }
    elif stage == "rolling_origin":
        required = {
            "processed RTT data": PATHS.clean_parquet,
        }
    elif stage == "covid_shock":
        required = {
            "processed RTT data": PATHS.clean_parquet,
        }
    elif stage == "validate":
        required = {}
    elif stage in LAYER1_STAGES:
        required = {}
    else:
        raise ValueError(f"Unknown stage: {stage}")

    for label, path in required.items():
        require_file(path, label)


def execute_stage(stage: str, base_env: Mapping[str, str] | None = None) -> list[str]:
    require_prerequisites(stage)
    produced: list[str] = []

    if stage == "download":
        run_command([sys.executable, "download_rtt_data.py"], stage, merge_env(base_env))
        produced.extend(
            [
                str(PATHS.raw_dir / "rtt_full_csv_manifest.csv"),
                str(PATHS.raw_dir / "rtt_download_summary.json"),
                str(PATHS.raw_zip_dir),
            ]
        )
    elif stage == "prepare":
        run_command(
            [sys.executable, "project_notebooks/layer1_nhs_rtt_tcn.py"],
            stage,
            merge_env(base_env, {"NHS_RTT_LAYER1_STAGE": "prepare"}),
        )
        produced.extend(
            [
                str(PATHS.clean_parquet),
                str(PATHS.net_inflow_quality),
                str(PATHS.flow_reconciliation_quality),
                str(PATHS.data_quality_report),
                str(PATHS.data_quality_summary),
                str(PATHS.missingness_by_series),
                str(PATHS.trust_identifier_changes),
                str(PATHS.part2a_coverage_report),
                str(PATHS.surgical_specialties),
                str(PATHS.data_dictionary),
            ]
        )
    elif stage in LAYER1_MODEL_STAGES:
        run_command(
            [sys.executable, "project_notebooks/layer1_nhs_rtt_tcn.py"],
            stage,
            merge_env(base_env, {"NHS_RTT_LAYER1_STAGE": stage}),
        )
        produced.extend(
            [
                str(PATHS.clean_parquet),
                str(PATHS.tcn_state_dict),
                str(PATHS.model_config),
                str(PATHS.feature_metadata),
                str(PATHS.dta_tcn_state_dict),
                str(PATHS.dta_model_config),
                str(PATHS.dta_feature_metadata),
                str(PATHS.backtest_predictions),
                str(PATHS.dta_backtest_predictions),
                str(PATHS.future_forecasts),
                str(PATHS.future_optimisation_forecasts),
                str(PATHS.forecast_metrics),
                str(PATHS.dta_forecast_metrics),
                str(PATHS.model_comparison),
                str(PATHS.dta_model_comparison),
            ]
        )
    elif stage == "explain":
        run_command([sys.executable, "project_notebooks/layer2_nhs_rtt_tcn_shap.py"], stage, merge_env(base_env))
        produced.extend(
            [
                str(PATHS.shap_values),
                str(PATHS.shap_global_importance),
                str(PATHS.shap_local_explanations),
                str(PATHS.shap_global_summary),
            ]
        )
    elif stage == "optimise":
        run_command([sys.executable, "project_notebooks/layer3_nhs_rtt_lp_optimisation.py"], stage, merge_env(base_env))
        produced.extend(
            [
                str(PATHS.lp_allocation_output),
                str(PATHS.lp_sensitivity_output),
                str(PATHS.lp_sensitivity_png),
                str(PATHS.lp_uncertainty_comparison),
                str(PATHS.lp_covid_stress_test),
            ]
        )
    elif stage == "rolling_origin":
        run_command([sys.executable, "project_notebooks/layer1c_nhs_rtt_rolling_origin_validation.py"], stage, merge_env(base_env))
        produced.extend(
            [
                str(PATHS.rolling_origin_predictions),
                str(PATHS.rolling_origin_metrics),
                str(PATHS.rolling_origin_metrics_by_origin),
                str(PATHS.rolling_origin_metrics_by_horizon),
                str(PATHS.rolling_origin_metrics_by_trust),
                str(PATHS.rolling_origin_metrics_by_specialty),
                str(PATHS.rolling_origin_metrics_by_waiting_size),
                str(PATHS.rolling_origin_reliability),
                str(PATHS.rolling_origin_quantile_crossing),
                str(PATHS.rolling_origin_summary),
                str(PATHS.rolling_origin_calibration_png),
                str(PATHS.rolling_origin_interval_width_coverage_png),
            ]
        )
    elif stage == "covid_shock":
        run_command([sys.executable, "project_notebooks/layer1b_nhs_rtt_covid_shock_experiment.py"], stage, merge_env(base_env))
        produced.extend(
            [
                str(PATHS.covid_predictions),
                str(PATHS.covid_metrics),
                str(PATHS.covid_degradation),
                str(PATHS.covid_split_summary),
                str(PATHS.covid_methodology_note),
                str(PATHS.covid_actual_vs_forecast_png),
                str(PATHS.covid_prediction_intervals_png),
                str(PATHS.covid_error_over_time_png),
                str(PATHS.covid_performance_by_trust_png),
                str(PATHS.covid_performance_by_specialty_png),
            ]
        )
    elif stage == "validate":
        run_command([sys.executable, "-m", "compileall", "-q", "."], stage, merge_env(base_env))
        run_command([sys.executable, "validate_project.py"], stage, merge_env(base_env))
        run_command([sys.executable, "dashboard/startup_validation.py"], stage, merge_env(base_env))
        run_command([sys.executable, "-m", "unittest", "discover", "-s", "tests"], stage, merge_env(base_env))
    return produced


def run_smoke_test() -> dict[str, object]:
    logging.info("Running smoke-test mode")
    run_command([sys.executable, "-m", "unittest", "discover", "-s", "tests"], "smoke-test")
    schema_report = validate_generated_outputs(PROJECT_ROOT)
    return {
        "mode": "smoke-test",
        "tests": "unittest discover completed",
        "existing_artifacts_checked": int(len(schema_report)),
        "existing_artifacts_ok": int((schema_report["status"] == "ok").sum()),
        "missing_artifacts": int((schema_report["status"] == "missing").sum()),
    }


def requested_stages(stage: str) -> list[str]:
    stage = STAGE_ALIASES.get(stage, stage)
    if stage == "all":
        return STAGE_ORDER
    return [stage]


def coalesce_layer1_stages(stages: Iterable[str]) -> list[str]:
    result: list[str] = []
    layer1_model_added = False
    for stage in stages:
        if stage in LAYER1_MODEL_STAGES:
            if not layer1_model_added:
                result.append(stage)
                layer1_model_added = True
        else:
            result.append(stage)
    return result


def write_run_summary(
    summary_path: Path,
    completed_stages: list[str],
    produced_files: list[str],
    status: str,
    started_at: str,
    finished_at: str,
    warnings: list[str],
    smoke_summary: dict[str, object] | None = None,
) -> None:
    schema_report = validate_generated_outputs(PROJECT_ROOT)
    schema_records = schema_report.to_dict(orient="records")
    payload = {
        "project_root": str(PROJECT_ROOT),
        "status": status,
        "started_at": started_at,
        "finished_at": finished_at,
        "stages_completed": completed_stages,
        "files_produced": produced_files,
        "schema_report": schema_records,
        "warnings": warnings,
        "smoke_summary": smoke_summary or {},
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)

    markdown_path = summary_path.with_suffix(".md")
    lines = [
        "# NHS RTT Pipeline Run Summary",
        "",
        f"- status: {status}",
        f"- started_at: {started_at}",
        f"- finished_at: {finished_at}",
        f"- stages_completed: {', '.join(completed_stages) if completed_stages else 'none'}",
        "",
        "## Files Produced",
        "",
    ]
    lines.extend([f"- `{path}`" for path in produced_files] or ["- none recorded"])
    lines.extend(["", "## Schema Status", ""])
    for row in schema_records:
        lines.append(f"- {row['artifact']}: {row['status']} ({row['row_count']} rows)")
    if warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend([f"- {warning}" for warning in warnings])
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    global PATHS
    parser = argparse.ArgumentParser(description="Run the NHS RTT MSc project pipeline.")
    parser.add_argument(
        "stage",
        choices=[*STAGE_ORDER, *STAGE_ALIASES.keys(), "all"],
        help=(
            "Pipeline stage to run. The all stage runs data preparation, training, backtesting, "
            "future forecasting, rolling-origin validation, SHAP, optimisation, COVID shock validation, and final validation."
        ),
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run lightweight module/tests/schema checks instead of the full data/model pipeline.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional path to a pipeline_config.json file.",
    )
    parser.add_argument(
        "--force-prepare",
        action="store_true",
        help="Rebuild data/processed/rtt_clean.parquet from raw RTT ZIP files even if it already exists.",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Retrain TCN model artifacts instead of reusing saved state dictionaries.",
    )
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve() if args.config is not None else PATHS.pipeline_config
    settings = load_pipeline_settings(config_path, PROJECT_ROOT)
    base_env = settings_environment(config_path, settings)
    if args.force_prepare:
        base_env["NHS_RTT_FORCE_PREPARE"] = "1"
    if args.force_retrain:
        base_env["NHS_RTT_FORCE_RETRAIN"] = "1"
    os.environ.update(base_env)
    PATHS = get_paths(PROJECT_ROOT)

    ensure_directories(PATHS)
    run_id = timestamp()
    log_path = PATHS.logs_dir / f"pipeline_{run_id}.log"
    summary_path = PATHS.run_summaries_dir / f"run_summary_{run_id}.json"
    configure_logging(log_path)

    started_at = datetime.now().isoformat(timespec="seconds")
    completed_stages: list[str] = []
    produced_files: list[str] = []
    warnings: list[str] = []
    smoke_summary: dict[str, object] | None = None
    status = "failed"

    try:
        seed_report = set_global_seed(settings.random_seed, settings.deterministic_torch)
        effective_settings_path = PATHS.run_summaries_dir / f"effective_settings_{run_id}.json"
        write_effective_settings_summary(settings, effective_settings_path)
        produced_files.append(str(effective_settings_path))
        warnings.extend(seed_report.get("limitations", []))
        logging.info("Loaded central settings from %s", config_path)
        logging.info("Seed report: %s", json.dumps(seed_report, default=str))

        if args.smoke_test:
            smoke_summary = run_smoke_test()
            completed_stages.append("smoke-test")
            status = "ok"
            return 0

        stages = coalesce_layer1_stages(requested_stages(args.stage))
        for stage in stages:
            stage_start = time.time()
            produced_files.extend(execute_stage(stage, base_env))
            if stage in LAYER1_MODEL_STAGES:
                completed_stages.extend(["train", "backtest", "forecast"])
            else:
                completed_stages.append(stage)
            logging.info("Stage %s finished in %.1f seconds", stage, time.time() - stage_start)
        status = "ok"
        return 0
    except subprocess.CalledProcessError as exc:
        logging.exception("Pipeline stage failed with exit code %s", exc.returncode)
        warnings.append(f"Command failed with exit code {exc.returncode}: {' '.join(exc.cmd)}")
        return int(exc.returncode or 1)
    except Exception as exc:
        logging.exception("Pipeline failed")
        warnings.append(str(exc))
        return 1
    finally:
        finished_at = datetime.now().isoformat(timespec="seconds")
        write_run_summary(
            summary_path=summary_path,
            completed_stages=completed_stages,
            produced_files=produced_files,
            status=status,
            started_at=started_at,
            finished_at=finished_at,
            warnings=warnings,
            smoke_summary=smoke_summary,
        )
        logging.info("Run summary written to %s", summary_path)
        logging.info("Log file written to %s", log_path)


if __name__ == "__main__":
    raise SystemExit(main())
