# %% Cell 1
import importlib.util
import subprocess
import sys

PIP_PACKAGES = [
    "numpy>=1.23.0",
    "pandas>=2.0.0",
    "pyarrow>=10.0.0",
    "pulp>=2.8.0",
    "plotly>=5.18.0",
    "kaleido==0.2.1",
]

IMPORT_CHECKS = {
    "numpy": "numpy",
    "pandas": "pandas",
    "pyarrow": "pyarrow",
    "pulp": "pulp",
    "plotly": "plotly",
    "kaleido": "kaleido",
}

missing_packages = []
for package_name, module_name in IMPORT_CHECKS.items():
    if importlib.util.find_spec(module_name) is None:
        missing_packages.append(package_name)

if missing_packages:
    packages_to_install = [
        package_spec
        for package_spec in PIP_PACKAGES
        if package_spec.split(">=")[0].split("==")[0] in missing_packages
    ]
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *packages_to_install])

# %% Cell 2
import os
import sys
from pathlib import Path

import pandas as pd

try:
    from IPython.display import display
except Exception:
    def display(obj: object) -> None:
        if isinstance(obj, pd.DataFrame):
            print(obj.head(25).to_string())
        else:
            print(obj)

PROJECT_ROOT_CANDIDATES = [
    Path.cwd(),
    Path.cwd().parent,
    Path("/content/nhs_rtt_msc_project"),
    Path("/content"),
]
for candidate_root in PROJECT_ROOT_CANDIDATES:
    if (candidate_root / "nhs_rtt_pipeline").exists():
        sys.path.insert(0, str(candidate_root))
        break
else:
    raise FileNotFoundError(
        "Could not find the shared nhs_rtt_pipeline package. "
        "Run this notebook from the project root or upload the full nhs_rtt_msc_project folder to Colab."
    )

from nhs_rtt_pipeline.config import COLUMNS, ensure_directories, get_paths
from nhs_rtt_pipeline.optimisation import (
    DEFAULT_AVAILABLE_SESSIONS,
    load_capacity_scenario_config,
    load_optimisation_forecasts,
)
from nhs_rtt_pipeline.settings import load_pipeline_settings

PATHS = get_paths()
ensure_directories(PATHS)
try:
    PROJECT_SETTINGS = load_pipeline_settings(os.environ.get("NHS_RTT_PIPELINE_CONFIG"))
    OPTIMISATION_SETTINGS = PROJECT_SETTINGS.optimisation
except FileNotFoundError:
    PROJECT_SETTINGS = None
    OPTIMISATION_SETTINGS = {}

# Step 1 - MILP formulation.
# Index set: K = Trust-specialty-month rows with surgical specialty and Part 2A decision-to-admit forecasts.
# Decision variables: sessions_allocated[k] = integer simulated additional treatment sessions for row k.
# Constraint (a): sum_k sessions_allocated[k] <= available_sessions.
# Constraint (b): sessions_allocated[k] >= 0 and integer-valued.
# Constraint (c): sessions_allocated[k] <= max_feasible_additional_sessions[k].
# Constraint (d): simulated_completed_pathways[k] <= patients_completed_per_session[specialty] * sessions_allocated[k].
# Constraint (e): simulated_completed_pathways[k] <= forecast incomplete_decision_to_admit backlog[k].
# Optional constraints supported by the shared module include max sessions per Trust, max sessions per specialty,
# minimum regional allocation when a region column exists, staff/bed scenario proxies, and budget.
# Objective weights are configurable for remaining backlog, long-wait backlog if available, weighted backlog,
# inequality between Trusts, and capacity cost.
# The full incomplete RTT pathway total is not used as the capacity-simulation objective.

capacity_config = load_capacity_scenario_config(PATHS.capacity_scenario_config)
DEFAULT_SCENARIO_SESSIONS = int(OPTIMISATION_SETTINGS.get("default_available_sessions", DEFAULT_AVAILABLE_SESSIONS))
DEFAULT_PRODUCTIVITY_SCENARIO = str(OPTIMISATION_SETTINGS.get("productivity_scenario", "central"))
future_optimisation_forecasts = load_optimisation_forecasts(PATHS.future_optimisation_forecasts)
print(f"Loaded MILP scenario config: {PATHS.capacity_scenario_config}")
print(f"Loaded Part 2A optimisation forecasts: {PATHS.future_optimisation_forecasts}")
print(f"Rows: {len(future_optimisation_forecasts):,}")
print(f"Trusts: {future_optimisation_forecasts[COLUMNS.trust_code].nunique():,}")
print(
    "Forecast range:",
    future_optimisation_forecasts[COLUMNS.forecast_month].min().date(),
    "to",
    future_optimisation_forecasts[COLUMNS.forecast_month].max().date(),
)
display(future_optimisation_forecasts.head(10))

# %% Cell 3
from nhs_rtt_pipeline.optimisation import reduction_by_trust_specialty, solve_milp_allocation


allocation_output, allocation_metadata, milp_problem = solve_milp_allocation(
    future_optimisation_forecasts,
    available_sessions=DEFAULT_SCENARIO_SESSIONS,
    capacity_config=capacity_config,
    productivity_scenario=DEFAULT_PRODUCTIVITY_SCENARIO,
    scenario_column=COLUMNS.p50,
)
reduction_output = reduction_by_trust_specialty(allocation_output)

allocation_output.to_csv(PATHS.lp_allocation_output, index=False)
reduction_output.to_csv(PATHS.lp_reduction_by_trust_specialty, index=False)
print(f"Solver status: {allocation_metadata['status']}")
print(f"Saved allocation output to: {PATHS.lp_allocation_output}")
print(f"Saved Trust-specialty reduction output to: {PATHS.lp_reduction_by_trust_specialty}")
print(allocation_metadata)
display(allocation_output.head(20))
display(reduction_output.head(20))

# %% Cell 4
import plotly.graph_objects as go

from nhs_rtt_pipeline.optimisation import run_sensitivity_analysis


sensitivity_results = run_sensitivity_analysis(
    future_optimisation_forecasts,
    available_session_values=OPTIMISATION_SETTINGS.get("capacity_budgets"),
    capacity_config=capacity_config,
    productivity_scenarios=("low", "central", "high"),
)
sensitivity_results.to_csv(PATHS.lp_sensitivity_output, index=False)

sensitivity_fig = go.Figure()
for scenario_name, scenario_frame in sensitivity_results.groupby("productivity_scenario", observed=True):
    sensitivity_fig.add_trace(
        go.Scatter(
            x=scenario_frame["available_sessions"],
            y=scenario_frame["simulated_completed_pathways"],
            mode="lines+markers",
            line=dict(width=3),
            marker=dict(size=8),
            name=str(scenario_name),
        )
    )
sensitivity_fig.update_layout(
    title="MILP Sensitivity: Productivity Scenario and Available Session Budget",
    xaxis_title="Available simulated additional treatment sessions",
    yaxis_title="Simulated completed Part 2A pathways",
    template="plotly_white",
    width=1000,
    height=560,
)
sensitivity_fig.write_image(str(PATHS.lp_sensitivity_png), scale=2)
print(f"Saved sensitivity table to: {PATHS.lp_sensitivity_output}")
print(f"Saved sensitivity chart to: {PATHS.lp_sensitivity_png}")
display(sensitivity_results)

# %% Cell 5
from nhs_rtt_pipeline.optimisation import run_uncertainty_comparison


uncertainty_comparison = run_uncertainty_comparison(
    future_optimisation_forecasts,
    available_sessions=DEFAULT_SCENARIO_SESSIONS,
    capacity_config=capacity_config,
    productivity_scenario=DEFAULT_PRODUCTIVITY_SCENARIO,
)
uncertainty_comparison.to_csv(PATHS.lp_uncertainty_comparison, index=False)
print(f"Saved uncertainty comparison to: {PATHS.lp_uncertainty_comparison}")
display(uncertainty_comparison.head(30))

# %% Cell 6
from nhs_rtt_pipeline.optimisation import run_covid_stress_test


covid_stress_test_results = run_covid_stress_test(
    future_optimisation_forecasts,
    available_sessions=DEFAULT_SCENARIO_SESSIONS,
    capacity_config=capacity_config,
    productivity_scenario=DEFAULT_PRODUCTIVITY_SCENARIO,
    covid_start="2020-03-01",
    covid_end="2021-09-30",
)
covid_stress_test_results.to_csv(PATHS.lp_covid_stress_test, index=False)
print(f"Saved COVID stress test results to: {PATHS.lp_covid_stress_test}")
display(covid_stress_test_results)
