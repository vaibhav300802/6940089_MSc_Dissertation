from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import pulp
except ImportError:  # pragma: no cover - exercised in environments without PuLP.
    pulp = None

from .config import COLUMNS, LP_ALLOCATION_COLUMNS, validate_future_forecast_frame, validate_optimisation_forecast_frame


DEFAULT_AVAILABLE_SESSIONS = 5000
DEFAULT_TOTAL_EXTRA_SESSIONS = DEFAULT_AVAILABLE_SESSIONS
DEFAULT_MAX_CAPACITY_INCREASE_PER_TRUST = 50
DEFAULT_PRODUCTIVITY_SCENARIO = "central"
LONG_WAIT_CANDIDATE_COLUMNS = [
    "predicted_long_wait_backlog",
    "long_wait_backlog",
    "waiting_over_threshold",
    "patients_waiting_over_threshold",
]


def default_capacity_scenario_config() -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "assumption_label": (
            "Hypothetical scenario parameters for dissertation optimisation. These are not observed NHS "
            "productivity, staffing, bed, or cost facts unless separately replaced with sourced local data."
        ),
        "session_unit_label": "simulated additional treatment session",
        "default_available_sessions": DEFAULT_AVAILABLE_SESSIONS,
        "default_max_sessions_per_trust_specialty_month": 12,
        "default_session_cost": 1.0,
        "capacity_budgets": [500, 2500, 5000, 10000, 25000, 50000],
        "productivity_scenarios": {
            "low": {
                "default_patients_completed_per_session": 5.0,
                "specialty_overrides": {
                    "130": 7.0,
                    "110": 4.0,
                    "100": 5.0,
                    "502": 5.0,
                },
            },
            "central": {
                "default_patients_completed_per_session": 8.0,
                "specialty_overrides": {
                    "130": 10.0,
                    "110": 6.0,
                    "100": 8.0,
                    "502": 7.0,
                },
            },
            "high": {
                "default_patients_completed_per_session": 11.0,
                "specialty_overrides": {
                    "130": 14.0,
                    "110": 8.0,
                    "100": 11.0,
                    "502": 10.0,
                },
            },
        },
        "specialty_priority_weights": {
            "default": 1.0,
            "130": 1.05,
            "110": 1.10,
            "100": 1.00,
            "502": 1.00,
        },
        "specialty_session_costs": {
            "default": 1.0,
            "130": 0.9,
            "110": 1.2,
            "100": 1.0,
            "502": 1.0,
        },
        "objective_weights": {
            "remaining_backlog": 1.0,
            "long_wait_backlog": 0.0,
            "weighted_backlog": 0.0,
            "trust_inequality": 0.0,
            "capacity_cost": 0.001,
        },
    }


def write_default_capacity_scenario_config(path: str | Path, overwrite: bool = False) -> Path:
    resolved = Path(path)
    if resolved.exists() and not overwrite:
        return resolved
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with open(resolved, "w", encoding="utf-8") as handle:
        json.dump(default_capacity_scenario_config(), handle, indent=2)
    return resolved


def load_capacity_scenario_config(path: str | Path) -> Dict[str, Any]:
    resolved = Path(path)
    if not resolved.exists():
        write_default_capacity_scenario_config(resolved, overwrite=False)
    with open(resolved, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    return merge_capacity_config(config)


def merge_capacity_config(config: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    merged = default_capacity_scenario_config()
    if not config:
        return merged
    for key, value in config.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            nested = dict(merged[key])
            nested.update(value)
            merged[key] = nested
        else:
            merged[key] = value
    return merged


def _lookup_numeric(mapping: Mapping[str, Any], key: object, default_key: str = "default", default: float = 0.0) -> float:
    key_string = str(key)
    if key_string in mapping:
        return float(mapping[key_string])
    if default_key in mapping:
        return float(mapping[default_key])
    return float(default)


def productivity_for_specialty(
    specialty_code: object,
    capacity_config: Mapping[str, Any],
    productivity_scenario: str,
) -> float:
    scenarios = capacity_config.get("productivity_scenarios", {})
    if productivity_scenario not in scenarios:
        raise ValueError(
            f"Unknown productivity scenario '{productivity_scenario}'. "
            f"Available scenarios: {sorted(scenarios)}"
        )
    scenario = scenarios[productivity_scenario]
    default_value = float(scenario.get("default_patients_completed_per_session", 1.0))
    value = _lookup_numeric(
        scenario.get("specialty_overrides", {}),
        specialty_code,
        default=default_value,
    )
    if value <= 0:
        raise ValueError(f"Productivity must be positive for specialty {specialty_code}: {value}")
    return value


def load_future_forecasts(path: str | Path) -> pd.DataFrame:
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Missing future forecast file: {resolved}")
    frame = pd.read_parquet(resolved)
    return validate_future_forecast_frame(frame, f"future forecasts ({resolved})")


def load_optimisation_forecasts(path: str | Path) -> pd.DataFrame:
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Missing Part 2A optimisation forecast file: {resolved}")
    frame = pd.read_parquet(resolved)
    return validate_optimisation_forecast_frame(frame, f"Part 2A optimisation forecasts ({resolved})")


def filter_forecast_period(
    forecast_df: pd.DataFrame,
    start_month: Optional[str] = None,
    end_month: Optional[str] = None,
    use_latest_month: bool = True,
) -> pd.DataFrame:
    frame = validate_optimisation_forecast_frame(forecast_df, "Part 2A optimisation forecast data")
    if start_month is not None:
        frame = frame[frame[COLUMNS.forecast_month] >= pd.Timestamp(start_month)]
    if end_month is not None:
        frame = frame[frame[COLUMNS.forecast_month] <= pd.Timestamp(end_month)]
    if frame.empty:
        return frame
    if use_latest_month:
        latest_month = frame[COLUMNS.forecast_month].max()
        frame = frame[frame[COLUMNS.forecast_month] == latest_month].copy()
    return frame


def _first_existing_column(frame: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for column in candidates:
        if column in frame.columns:
            return column
    return None


def _normalise_limit_mapping(limit: Optional[int | float | Mapping[str, int | float]]) -> Optional[Dict[str, float] | float]:
    if limit is None:
        return None
    if isinstance(limit, Mapping):
        return {str(key): float(value) for key, value in limit.items()}
    return float(limit)


def _limit_for_key(limit: Optional[Dict[str, float] | float], key: str) -> Optional[float]:
    if limit is None:
        return None
    if isinstance(limit, dict):
        return limit.get(str(key))
    return float(limit)


def prepare_trust_specialty_month_forecast(
    forecast_df: pd.DataFrame,
    scenario_column: str = COLUMNS.p50,
    capacity_config: Optional[Mapping[str, Any]] = None,
    productivity_scenario: str = DEFAULT_PRODUCTIVITY_SCENARIO,
    start_month: Optional[str] = None,
    end_month: Optional[str] = None,
    use_latest_month: bool = True,
) -> pd.DataFrame:
    config = merge_capacity_config(capacity_config)
    frame = filter_forecast_period(
        forecast_df,
        start_month=start_month,
        end_month=end_month,
        use_latest_month=use_latest_month,
    )
    if scenario_column not in frame.columns:
        raise ValueError(f"Scenario column is missing from optimisation forecast data: {scenario_column}")
    if frame.empty:
        return pd.DataFrame()

    rows = frame.copy()
    rows["baseline_predicted_backlog"] = pd.to_numeric(rows[scenario_column], errors="coerce").fillna(0.0).clip(lower=0.0)
    rows = rows[rows["baseline_predicted_backlog"] > 0].copy()
    if rows.empty:
        return pd.DataFrame()

    rows["patients_completed_per_session"] = [
        productivity_for_specialty(code, config, productivity_scenario)
        for code in rows[COLUMNS.specialty_code].astype(str)
    ]
    rows["specialty_priority_weight"] = [
        _lookup_numeric(config.get("specialty_priority_weights", {}), code, default=1.0)
        for code in rows[COLUMNS.specialty_code].astype(str)
    ]
    rows["session_cost"] = [
        _lookup_numeric(config.get("specialty_session_costs", {}), code, default=float(config.get("default_session_cost", 1.0)))
        for code in rows[COLUMNS.specialty_code].astype(str)
    ]
    if "max_feasible_additional_sessions" in rows.columns:
        base_max = pd.to_numeric(rows["max_feasible_additional_sessions"], errors="coerce")
    else:
        base_max = pd.Series(float(config.get("default_max_sessions_per_trust_specialty_month", 12)), index=rows.index)
    backlog_limited_max = np.ceil(
        rows["baseline_predicted_backlog"].to_numpy(dtype=float)
        / rows["patients_completed_per_session"].to_numpy(dtype=float)
    )
    rows["max_feasible_additional_sessions"] = np.minimum(
        base_max.fillna(0.0).clip(lower=0.0).to_numpy(dtype=float),
        backlog_limited_max,
    ).astype(int)
    rows = rows[rows["max_feasible_additional_sessions"] > 0].copy()

    long_wait_column = _first_existing_column(rows, LONG_WAIT_CANDIDATE_COLUMNS)
    if long_wait_column is not None:
        rows["long_wait_backlog"] = pd.to_numeric(rows[long_wait_column], errors="coerce").fillna(0.0).clip(lower=0.0)
        rows["long_wait_backlog"] = np.minimum(rows["long_wait_backlog"], rows["baseline_predicted_backlog"])
    else:
        rows["long_wait_backlog"] = 0.0

    return rows.sort_values(
        [COLUMNS.trust_code, COLUMNS.specialty_code, COLUMNS.forecast_month],
    ).reset_index(drop=True)


def prepare_trust_level_forecast(
    forecast_df: pd.DataFrame,
    scenario_column: str = COLUMNS.p50,
    start_month: Optional[str] = None,
    end_month: Optional[str] = None,
    use_latest_month: bool = True,
    average_monthly: bool = False,
) -> pd.DataFrame:
    rows = prepare_trust_specialty_month_forecast(
        forecast_df,
        scenario_column=scenario_column,
        start_month=start_month,
        end_month=end_month,
        use_latest_month=use_latest_month,
    )
    if rows.empty:
        return pd.DataFrame(
            columns=[
                "trust_code",
                "trust_name",
                "current_incomplete_decision_to_admit",
                COLUMNS.p10,
                COLUMNS.p50,
                COLUMNS.p90,
                "objective_incomplete_decision_to_admit",
                "period_start",
                "period_end",
                "n_months",
            ]
        )
    aggregation = "mean" if average_monthly else "sum"
    trust_level = (
        rows.groupby(["trust_code", "trust_name"], as_index=False, observed=True)
        .agg(
            current_incomplete_decision_to_admit=(COLUMNS.latest_observed_incomplete_decision_to_admit, aggregation),
            objective_incomplete_decision_to_admit=("baseline_predicted_backlog", aggregation),
            period_start=(COLUMNS.forecast_month, "min"),
            period_end=(COLUMNS.forecast_month, "max"),
            n_months=(COLUMNS.forecast_month, "nunique"),
        )
        .sort_values("objective_incomplete_decision_to_admit", ascending=False)
        .reset_index(drop=True)
    )
    return trust_level


def pathway_reduction(
    trust_name: str,
    extra_sessions: int,
    forecast_df: pd.DataFrame,
    pathways_addressed_per_session: int = 8,
) -> float:
    if extra_sessions < 0:
        raise ValueError("extra_sessions must be non-negative.")
    trust_level = prepare_trust_level_forecast(forecast_df, scenario_column=COLUMNS.p50)
    trust_rows = trust_level[trust_level["trust_name"].astype(str) == str(trust_name)]
    if trust_rows.empty:
        raise ValueError(f"Trust was not found in Part 2A optimisation forecast data: {trust_name}")
    predicted_dta = float(trust_rows["objective_incomplete_decision_to_admit"].sum())
    linear_reduction = float(extra_sessions) * float(pathways_addressed_per_session)
    return float(min(linear_reduction, predicted_dta))


def _require_pulp() -> Any:
    if pulp is None:
        raise ImportError("PuLP is required for mixed-integer linear optimisation. Install pulp>=2.8.0.")
    return pulp


def solve_milp_allocation(
    forecast_df: pd.DataFrame,
    available_sessions: Optional[int] = None,
    total_extra_sessions: Optional[int] = None,
    scenario_column: str = COLUMNS.p50,
    capacity_config: Optional[Mapping[str, Any]] = None,
    productivity_scenario: str = DEFAULT_PRODUCTIVITY_SCENARIO,
    objective_weights: Optional[Mapping[str, float]] = None,
    max_sessions_per_trust: Optional[int | Mapping[str, int]] = None,
    max_sessions_per_specialty: Optional[int | Mapping[str, int]] = None,
    minimum_regional_allocation: Optional[Mapping[str, int]] = None,
    region_column: str = "region",
    staff_bed_capacity_column: Optional[str] = None,
    staff_bed_capacity_by_trust: Optional[Mapping[str, int]] = None,
    budget: Optional[float] = None,
    start_month: Optional[str] = None,
    end_month: Optional[str] = None,
    use_latest_month: bool = True,
    solver_msg: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any], Any]:
    pulp_module = _require_pulp()
    config = merge_capacity_config(capacity_config)
    if available_sessions is None:
        available_sessions = total_extra_sessions
    if available_sessions is None:
        available_sessions = int(config.get("default_available_sessions", DEFAULT_AVAILABLE_SESSIONS))
    available_sessions = int(available_sessions)
    if available_sessions < 0:
        raise ValueError("available_sessions must be non-negative.")

    weights = dict(config.get("objective_weights", {}))
    if objective_weights:
        weights.update({key: float(value) for key, value in objective_weights.items()})
    for key in ["remaining_backlog", "long_wait_backlog", "weighted_backlog", "trust_inequality", "capacity_cost"]:
        weights[key] = float(weights.get(key, 0.0))

    rows = prepare_trust_specialty_month_forecast(
        forecast_df,
        scenario_column=scenario_column,
        capacity_config=config,
        productivity_scenario=productivity_scenario,
        start_month=start_month,
        end_month=end_month,
        use_latest_month=use_latest_month,
    )
    if rows.empty:
        raise ValueError("No Trust-specialty-month Part 2A forecast rows are available for MILP optimisation.")

    max_sessions_per_trust = _normalise_limit_mapping(max_sessions_per_trust)
    max_sessions_per_specialty = _normalise_limit_mapping(max_sessions_per_specialty)
    if minimum_regional_allocation and region_column not in rows.columns:
        raise ValueError(
            f"minimum_regional_allocation was supplied, but region column '{region_column}' is missing from forecast data."
        )
    if staff_bed_capacity_column is not None and staff_bed_capacity_column not in rows.columns:
        raise ValueError(
            f"staff_bed_capacity_column='{staff_bed_capacity_column}' is missing from forecast data. "
            "Do not invent staff or bed capacity; provide a labelled scenario column if this constraint is needed."
        )

    indices = list(rows.index)
    problem = pulp_module.LpProblem("NHS_RTT_Part2A_Trust_Specialty_Month_MILP", pulp_module.LpMinimize)
    sessions = {
        i: pulp_module.LpVariable(f"sessions_allocated_{i}", lowBound=0, upBound=int(rows.loc[i, "max_feasible_additional_sessions"]), cat=pulp_module.LpInteger)
        for i in indices
    }
    completed = {i: pulp_module.LpVariable(f"simulated_completed_pathways_{i}", lowBound=0, cat=pulp_module.LpContinuous) for i in indices}
    remaining = {i: pulp_module.LpVariable(f"remaining_backlog_{i}", lowBound=0, cat=pulp_module.LpContinuous) for i in indices}
    long_wait_remaining = {
        i: pulp_module.LpVariable(f"long_wait_remaining_{i}", lowBound=0, cat=pulp_module.LpContinuous)
        for i in indices
    }

    problem += pulp_module.lpSum(sessions[i] for i in indices) <= available_sessions, "available_sessions_upper_bound"

    for i in indices:
        backlog = float(rows.loc[i, "baseline_predicted_backlog"])
        productivity = float(rows.loc[i, "patients_completed_per_session"])
        long_wait_backlog = float(rows.loc[i, "long_wait_backlog"])
        problem += completed[i] <= productivity * sessions[i], f"productivity_link_{i}"
        problem += completed[i] <= backlog, f"cannot_complete_more_than_forecast_backlog_{i}"
        problem += remaining[i] == backlog - completed[i], f"remaining_backlog_balance_{i}"
        problem += long_wait_remaining[i] >= long_wait_backlog - completed[i], f"long_wait_remaining_lower_{i}"
        problem += long_wait_remaining[i] <= long_wait_backlog, f"long_wait_remaining_upper_{i}"

    for trust_code, group in rows.groupby(COLUMNS.trust_code, observed=True):
        limit = _limit_for_key(max_sessions_per_trust, str(trust_code))
        if limit is not None:
            problem += (
                pulp_module.lpSum(sessions[i] for i in group.index) <= int(limit)
            ), f"max_sessions_trust_{trust_code}"

    for specialty_code, group in rows.groupby(COLUMNS.specialty_code, observed=True):
        limit = _limit_for_key(max_sessions_per_specialty, str(specialty_code))
        if limit is not None:
            problem += (
                pulp_module.lpSum(sessions[i] for i in group.index) <= int(limit)
            ), f"max_sessions_specialty_{specialty_code}"

    if minimum_regional_allocation:
        for region, minimum in minimum_regional_allocation.items():
            group_indices = rows.index[rows[region_column].astype(str).eq(str(region))].tolist()
            if not group_indices:
                raise ValueError(f"Minimum regional allocation was requested for unknown region '{region}'.")
            problem += (
                pulp_module.lpSum(sessions[i] for i in group_indices) >= int(minimum)
            ), f"minimum_sessions_region_{region}"

    if staff_bed_capacity_column is not None:
        for i in indices:
            proxy_value = float(pd.to_numeric(pd.Series([rows.loc[i, staff_bed_capacity_column]]), errors="coerce").fillna(0).iloc[0])
            problem += sessions[i] <= int(max(0, math.floor(proxy_value))), f"staff_bed_proxy_row_{i}"

    if staff_bed_capacity_by_trust:
        for trust_code, limit in staff_bed_capacity_by_trust.items():
            group_indices = rows.index[rows[COLUMNS.trust_code].astype(str).eq(str(trust_code))].tolist()
            if group_indices:
                problem += (
                    pulp_module.lpSum(sessions[i] for i in group_indices) <= int(limit)
                ), f"staff_bed_proxy_trust_{trust_code}"

    total_cost_expr = pulp_module.lpSum(float(rows.loc[i, "session_cost"]) * sessions[i] for i in indices)
    if budget is not None:
        problem += total_cost_expr <= float(budget), "scenario_budget_limit"

    trust_remaining = {}
    for trust_code, group in rows.groupby(COLUMNS.trust_code, observed=True):
        trust_remaining[str(trust_code)] = pulp_module.lpSum(remaining[i] for i in group.index)
    max_trust_remaining = pulp_module.LpVariable("max_trust_remaining", lowBound=0, cat=pulp_module.LpContinuous)
    min_trust_remaining = pulp_module.LpVariable("min_trust_remaining", lowBound=0, cat=pulp_module.LpContinuous)
    for trust_code, expr in trust_remaining.items():
        problem += max_trust_remaining >= expr, f"max_trust_remaining_{trust_code}"
        problem += min_trust_remaining <= expr, f"min_trust_remaining_{trust_code}"

    objective = (
        weights["remaining_backlog"] * pulp_module.lpSum(remaining[i] for i in indices)
        + weights["long_wait_backlog"] * pulp_module.lpSum(long_wait_remaining[i] for i in indices)
        + weights["weighted_backlog"]
        * pulp_module.lpSum(float(rows.loc[i, "specialty_priority_weight"]) * remaining[i] for i in indices)
        + weights["trust_inequality"] * (max_trust_remaining - min_trust_remaining)
        + weights["capacity_cost"] * total_cost_expr
    )
    problem += objective, "weighted_milp_capacity_allocation_objective"

    status_code = problem.solve(pulp_module.PULP_CBC_CMD(msg=solver_msg))
    status = pulp_module.LpStatus[status_code]
    if status != "Optimal":
        raise RuntimeError(f"PuLP MILP optimisation failed with solver status '{status}'.")

    allocation = rows.copy()
    allocation["sessions_allocated"] = [int(round(float(pulp_module.value(sessions[i])))) for i in indices]
    allocation["simulated_completed_pathways"] = [float(pulp_module.value(completed[i])) for i in indices]
    allocation["remaining_backlog"] = [float(pulp_module.value(remaining[i])) for i in indices]
    allocation["unused_feasible_sessions"] = (
        allocation["max_feasible_additional_sessions"].astype(float) - allocation["sessions_allocated"].astype(float)
    ).clip(lower=0.0)
    allocation["allocation_cost"] = allocation["sessions_allocated"].astype(float) * allocation["session_cost"].astype(float)
    allocation["percent_reduction"] = np.where(
        allocation["baseline_predicted_backlog"] > 0,
        100.0 * allocation["simulated_completed_pathways"] / allocation["baseline_predicted_backlog"],
        0.0,
    )
    allocation = allocation[LP_ALLOCATION_COLUMNS].sort_values(
        ["simulated_completed_pathways", "baseline_predicted_backlog"],
        ascending=[False, False],
    ).reset_index(drop=True)

    baseline = float(allocation["baseline_predicted_backlog"].sum())
    completed_total = float(allocation["simulated_completed_pathways"].sum())
    remaining_total = float(allocation["remaining_backlog"].sum())
    sessions_used = int(allocation["sessions_allocated"].sum())
    metadata: Dict[str, Any] = {
        "method": "mixed-integer linear programming",
        "status": status,
        "objective_value": float(pulp_module.value(problem.objective)),
        "available_sessions": int(available_sessions),
        "sessions_used": sessions_used,
        "unused_sessions": int(max(0, available_sessions - sessions_used)),
        "baseline_predicted_backlog": baseline,
        "simulated_completed_pathways": completed_total,
        "remaining_backlog": remaining_total,
        "national_percent_reduction": 100.0 * completed_total / baseline if baseline > 0 else 0.0,
        "total_allocation_cost": float(allocation["allocation_cost"].sum()),
        "productivity_scenario": productivity_scenario,
        "objective_weights": weights,
        "long_wait_objective_available": bool(rows["long_wait_backlog"].sum() > 0),
        "scenario_assumption_label": str(config.get("assumption_label", "")),
    }
    metadata["total_pathways_addressed"] = metadata["simulated_completed_pathways"]
    metadata["national_incomplete_decision_to_admit_no_intervention"] = metadata["baseline_predicted_backlog"]
    metadata["national_incomplete_decision_to_admit_post_allocation"] = metadata["remaining_backlog"]
    return allocation, metadata, problem


def solve_lp_allocation(*args: Any, **kwargs: Any) -> Tuple[pd.DataFrame, Dict[str, Any], Any]:
    return solve_milp_allocation(*args, **kwargs)


def reduction_by_trust_specialty(allocation: pd.DataFrame) -> pd.DataFrame:
    required = [
        "trust_code",
        "trust_name",
        "specialty_code",
        "specialty_name",
        "baseline_predicted_backlog",
        "sessions_allocated",
        "simulated_completed_pathways",
        "remaining_backlog",
    ]
    missing = [column for column in required if column not in allocation.columns]
    if missing:
        raise ValueError(f"Allocation output is missing required columns: {missing}")
    grouped = (
        allocation.groupby(["trust_code", "trust_name", "specialty_code", "specialty_name"], as_index=False, observed=True)
        .agg(
            baseline_predicted_backlog=("baseline_predicted_backlog", "sum"),
            sessions_allocated=("sessions_allocated", "sum"),
            simulated_completed_pathways=("simulated_completed_pathways", "sum"),
            remaining_backlog=("remaining_backlog", "sum"),
            allocation_cost=("allocation_cost", "sum"),
        )
        .sort_values("simulated_completed_pathways", ascending=False)
        .reset_index(drop=True)
    )
    grouped["percent_reduction"] = np.where(
        grouped["baseline_predicted_backlog"] > 0,
        100.0 * grouped["simulated_completed_pathways"] / grouped["baseline_predicted_backlog"],
        0.0,
    )
    return grouped


def run_sensitivity_analysis(
    forecast_df: pd.DataFrame,
    available_session_values: Optional[Sequence[int]] = None,
    total_sessions_values: Optional[Sequence[int]] = None,
    productivity_scenarios: Sequence[str] = ("low", "central", "high"),
    capacity_config: Optional[Mapping[str, Any]] = None,
    scenario_column: str = COLUMNS.p50,
    **solve_kwargs: Any,
) -> pd.DataFrame:
    config = merge_capacity_config(capacity_config)
    if available_session_values is None:
        available_session_values = total_sessions_values
    if available_session_values is None:
        available_session_values = list(config.get("capacity_budgets", [500, 2500, 5000, 10000, 25000, 50000]))
    rows = []
    for productivity_scenario in productivity_scenarios:
        for available_sessions in available_session_values:
            try:
                _, metadata, _ = solve_milp_allocation(
                    forecast_df,
                    available_sessions=int(available_sessions),
                    scenario_column=scenario_column,
                    capacity_config=config,
                    productivity_scenario=productivity_scenario,
                    **solve_kwargs,
                )
                rows.append(
                    {
                        "productivity_scenario": productivity_scenario,
                        "available_sessions": int(available_sessions),
                        "solver_status": metadata["status"],
                        "sessions_used": metadata["sessions_used"],
                        "unused_sessions": metadata["unused_sessions"],
                        "baseline_predicted_backlog": metadata["baseline_predicted_backlog"],
                        "simulated_completed_pathways": metadata["simulated_completed_pathways"],
                        "remaining_backlog": metadata["remaining_backlog"],
                        "national_percent_reduction": metadata["national_percent_reduction"],
                        "total_allocation_cost": metadata["total_allocation_cost"],
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "productivity_scenario": productivity_scenario,
                        "available_sessions": int(available_sessions),
                        "solver_status": f"failed: {exc}",
                        "sessions_used": np.nan,
                        "unused_sessions": np.nan,
                        "baseline_predicted_backlog": np.nan,
                        "simulated_completed_pathways": np.nan,
                        "remaining_backlog": np.nan,
                        "national_percent_reduction": np.nan,
                        "total_allocation_cost": np.nan,
                    }
                )
    return pd.DataFrame(rows)


def run_uncertainty_comparison(
    forecast_df: pd.DataFrame,
    available_sessions: int = DEFAULT_AVAILABLE_SESSIONS,
    total_extra_sessions: Optional[int] = None,
    capacity_config: Optional[Mapping[str, Any]] = None,
    productivity_scenario: str = DEFAULT_PRODUCTIVITY_SCENARIO,
    **solve_kwargs: Any,
) -> pd.DataFrame:
    if total_extra_sessions is not None:
        available_sessions = int(total_extra_sessions)
    scenario_columns = {
        "optimistic_p10": COLUMNS.p10,
        "median_p50": COLUMNS.p50,
        "pessimistic_p90": COLUMNS.p90,
    }
    scenario_frames = []
    for scenario_name, scenario_column in scenario_columns.items():
        allocation, metadata, _ = solve_milp_allocation(
            forecast_df,
            available_sessions=available_sessions,
            scenario_column=scenario_column,
            capacity_config=capacity_config,
            productivity_scenario=productivity_scenario,
            **solve_kwargs,
        )
        scenario = reduction_by_trust_specialty(allocation)
        scenario.insert(0, "scenario", scenario_name)
        scenario["solver_status"] = metadata["status"]
        scenario["sessions_used"] = metadata["sessions_used"]
        scenario["unused_sessions"] = metadata["unused_sessions"]
        scenario["baseline_predicted_backlog_national"] = metadata["baseline_predicted_backlog"]
        scenario["remaining_backlog_national"] = metadata["remaining_backlog"]
        scenario["simulated_completed_pathways_national"] = metadata["simulated_completed_pathways"]
        scenario_frames.append(scenario)
    return pd.concat(scenario_frames, ignore_index=True)


def run_covid_stress_test(
    forecast_df: pd.DataFrame,
    available_sessions: int = DEFAULT_AVAILABLE_SESSIONS,
    total_extra_sessions: Optional[int] = None,
    capacity_config: Optional[Mapping[str, Any]] = None,
    productivity_scenario: str = DEFAULT_PRODUCTIVITY_SCENARIO,
    covid_start: str = "2020-03-01",
    covid_end: str = "2021-09-30",
    **solve_kwargs: Any,
) -> pd.DataFrame:
    frame = validate_optimisation_forecast_frame(forecast_df, "Part 2A optimisation forecast data")
    if total_extra_sessions is not None:
        available_sessions = int(total_extra_sessions)
    forecast_min = pd.to_datetime(frame[COLUMNS.forecast_month]).min()
    forecast_max = pd.to_datetime(frame[COLUMNS.forecast_month]).max()
    covid_start_ts = pd.Timestamp(covid_start)
    covid_end_ts = pd.Timestamp(covid_end)
    covid_mask = (frame[COLUMNS.forecast_month] >= covid_start_ts) & (frame[COLUMNS.forecast_month] <= covid_end_ts)
    covid_rows = int(covid_mask.sum())
    return pd.DataFrame(
        [
            {
                "scenario": "optimisation_historical_covid_period_audit",
                "status": "not_run_for_future_forecast_file",
                "forecast_file_month_start": forecast_min.date().isoformat(),
                "forecast_file_month_end": forecast_max.date().isoformat(),
                "requested_covid_period_start": covid_start_ts.date().isoformat(),
                "requested_covid_period_end": covid_end_ts.date().isoformat(),
                "matching_forecast_rows": covid_rows,
                "available_sessions_requested": int(available_sessions),
                "productivity_scenario": productivity_scenario,
                "simulated_completed_pathways": np.nan,
                "baseline_predicted_backlog": np.nan,
                "remaining_backlog": np.nan,
                "national_percent_reduction": np.nan,
                "covid_worse_than_normal_absolute": np.nan,
                "covid_worse_than_normal_percent": np.nan,
                "normal_minus_covid_percent_reduction_points": np.nan,
                "interpretation": (
                    "This optimisation stage uses genuine future Part 2A forecasts, so historical COVID months "
                    "are not optimised here. Pandemic-period forecasting performance is evaluated under "
                    "outputs/covid_stress_test/."
                ),
            }
        ]
    )
