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


@unittest.skipIf(importlib.util.find_spec("pulp") is None, "PuLP is not installed in this local test runtime")
class OptimisationConstraintTests(unittest.TestCase):
    def _forecast(self) -> pd.DataFrame:
        rows = []
        for specialty_code, p50 in [("100", 20.0), ("110", 5.0)]:
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

    def test_zero_available_sessions_allocates_nothing(self) -> None:
        from nhs_rtt_pipeline.optimisation import solve_milp_allocation

        allocation, metadata, _ = solve_milp_allocation(self._forecast(), available_sessions=0)

        self.assertEqual(int(metadata["sessions_used"]), 0)
        self.assertEqual(int(allocation["sessions_allocated"].sum()), 0)
        self.assertEqual(float(allocation["simulated_completed_pathways"].sum()), 0.0)

    def test_excess_sessions_do_not_complete_more_than_waiting_list(self) -> None:
        from nhs_rtt_pipeline.optimisation import solve_milp_allocation

        allocation, metadata, _ = solve_milp_allocation(self._forecast(), available_sessions=100)

        self.assertLessEqual(float(allocation["simulated_completed_pathways"].sum()), float(allocation["baseline_predicted_backlog"].sum()))
        self.assertGreaterEqual(int(metadata["unused_sessions"]), 0)
        self.assertTrue((allocation["sessions_allocated"] <= allocation["max_feasible_additional_sessions"]).all())

    def test_infeasible_minimum_region_constraint_raises(self) -> None:
        from nhs_rtt_pipeline.optimisation import solve_milp_allocation

        with self.assertRaises(RuntimeError):
            solve_milp_allocation(
                self._forecast(),
                available_sessions=1,
                minimum_regional_allocation={"North": 5},
            )


if __name__ == "__main__":
    unittest.main()
