from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nhs_rtt_pipeline.config import COLUMNS


def _forecast() -> pd.DataFrame:
    rows = []
    for specialty_code, p50 in [("100", 1000.0), ("110", 500.0)]:
        rows.append(
            {
                COLUMNS.trust_code: "R00",
                COLUMNS.trust_name: "Example Trust",
                COLUMNS.specialty_code: specialty_code,
                COLUMNS.specialty_name: f"Specialty {specialty_code}",
                COLUMNS.forecast_origin: "2024-01-01",
                COLUMNS.forecast_month: "2024-02-01",
                COLUMNS.horizon: 1,
                COLUMNS.p10: max(0.0, p50 - 5.0),
                COLUMNS.p50: p50,
                COLUMNS.p90: p50 + 5.0,
                COLUMNS.latest_observed_incomplete_decision_to_admit: p50,
                COLUMNS.forecast_target: COLUMNS.incomplete_decision_to_admit,
                COLUMNS.is_surgical_specialty: True,
                COLUMNS.specialty_inclusion_criteria: "Included for test scenario.",
                "region": "North",
            }
        )
    return pd.DataFrame(rows)


@unittest.skipIf(importlib.util.find_spec("pulp") is None, "PuLP is not installed in this local test runtime")
@unittest.skipIf(importlib.util.find_spec("streamlit") is None, "streamlit is not installed in this local test runtime")
class DashboardCapacityScenarioWiringTests(unittest.TestCase):
    """Confirms the dashboard's capacity-slider callback genuinely wires through
    to the shared nhs_rtt_pipeline.optimisation MILP solver, rather than only
    being checked for file existence (as test_dashboard_startup.py already does)."""

    def test_dashboard_cached_solver_matches_direct_solver_call(self) -> None:
        import dashboard.app as app

        from nhs_rtt_pipeline.optimisation import default_capacity_scenario_config, solve_milp_allocation

        forecast = _forecast()
        capacity_config = default_capacity_scenario_config()

        dashboard_allocation, dashboard_metadata = app.solve_capacity_scenario_cached(
            forecast,
            available_sessions=100,
            scenario_column=COLUMNS.p50,
            capacity_config=capacity_config,
            productivity_scenario="central",
        )
        direct_allocation, direct_metadata, _ = solve_milp_allocation(
            forecast,
            available_sessions=100,
            scenario_column=COLUMNS.p50,
            capacity_config=capacity_config,
            productivity_scenario="central",
            solver_msg=False,
        )

        pd.testing.assert_frame_equal(
            dashboard_allocation.reset_index(drop=True),
            direct_allocation.reset_index(drop=True),
        )
        self.assertEqual(dashboard_metadata["status"], "Optimal")
        self.assertEqual(int(dashboard_metadata["sessions_used"]), int(direct_metadata["sessions_used"]))

    def test_slider_at_zero_sessions_allocates_nothing_via_dashboard_path(self) -> None:
        import dashboard.app as app

        from nhs_rtt_pipeline.optimisation import default_capacity_scenario_config

        allocation, metadata = app.solve_capacity_scenario_cached(
            _forecast(),
            available_sessions=0,
            scenario_column=COLUMNS.p50,
            capacity_config=default_capacity_scenario_config(),
            productivity_scenario="central",
        )

        self.assertEqual(int(metadata["sessions_used"]), 0)
        self.assertEqual(int(allocation["sessions_allocated"].sum()), 0)

    def test_different_productivity_scenarios_change_the_dashboard_result(self) -> None:
        # Guards against the slider/dropdown being wired to a value that the
        # solver silently ignores (e.g. a stale default always being used).
        import dashboard.app as app

        from nhs_rtt_pipeline.optimisation import default_capacity_scenario_config

        capacity_config = default_capacity_scenario_config()
        _, low_metadata = app.solve_capacity_scenario_cached(
            _forecast(), available_sessions=100, scenario_column=COLUMNS.p50,
            capacity_config=capacity_config, productivity_scenario="low",
        )
        _, high_metadata = app.solve_capacity_scenario_cached(
            _forecast(), available_sessions=100, scenario_column=COLUMNS.p50,
            capacity_config=capacity_config, productivity_scenario="high",
        )

        self.assertNotEqual(
            low_metadata["simulated_completed_pathways"],
            high_metadata["simulated_completed_pathways"],
        )


if __name__ == "__main__":
    unittest.main()
