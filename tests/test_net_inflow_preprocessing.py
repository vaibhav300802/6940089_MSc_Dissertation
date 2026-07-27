from __future__ import annotations

import unittest
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nhs_rtt_pipeline.config import COLUMNS, validate_optimisation_forecast_frame
from nhs_rtt_pipeline.preprocessing import (
    assert_flow_reconciliation_integrity,
    assert_net_inflow_integrity,
    clean_rtt_operational_features,
    flow_reconciliation_quality_report,
    net_inflow_quality_summary,
)


class NetInflowPreprocessingTests(unittest.TestCase):
    def test_negative_net_inflow_is_preserved_after_cleaning(self) -> None:
        sample = pd.DataFrame(
            [
                {
                    "month": "2024-01-01",
                    "trust_code": "R00",
                    "specialty_code": "100",
                    "waiting_list": 5000,
                    "completed_admitted": 600,
                    "completed_non_admitted": 500,
                    "waiting_list_with_dta": 120,
                    "new_rtt_periods": 900,
                }
            ]
        )

        cleaned = clean_rtt_operational_features(sample)

        self.assertEqual(float(cleaned.loc[0, "completed_total"]), 1100.0)
        self.assertEqual(float(cleaned.loc[0, "incomplete_total"]), 5000.0)
        self.assertEqual(float(cleaned.loc[0, "incomplete_decision_to_admit"]), 120.0)
        self.assertEqual(float(cleaned.loc[0, "net_inflow"]), -200.0)
        self.assertEqual(float(cleaned.loc[0, "reported_net_inflow"]), -200.0)
        assert_net_inflow_integrity(cleaned)

    def test_net_inflow_quality_summary_counts_signed_values(self) -> None:
        sample = pd.DataFrame(
            [
                {"net_inflow": -200},
                {"net_inflow": 0},
                {"net_inflow": 300},
            ]
        )

        summary = net_inflow_quality_summary(sample).iloc[0]

        self.assertEqual(float(summary["min_net_inflow"]), -200.0)
        self.assertEqual(float(summary["max_net_inflow"]), 300.0)
        self.assertEqual(int(summary["negative_observations"]), 1)
        self.assertEqual(int(summary["zero_observations"]), 1)
        self.assertEqual(int(summary["positive_observations"]), 1)

    def test_flow_reconciliation_calculates_unreported_removals(self) -> None:
        sample = pd.DataFrame(
            [
                {
                    "month": "2024-01-01",
                    "trust_code": "R00",
                    "trust_name": "Example Trust",
                    "specialty_code": "100",
                    "specialty_name": "General Surgery",
                    "waiting_list": 1000,
                    "completed_admitted": 40,
                    "completed_non_admitted": 60,
                    "waiting_list_with_dta": 10,
                    "new_rtt_periods": 120,
                    "waiting_list_source_available": 1,
                    "completed_admitted_source_available": 1,
                    "completed_non_admitted_source_available": 1,
                    "new_rtt_periods_source_available": 1,
                },
                {
                    "month": "2024-02-01",
                    "trust_code": "R00",
                    "trust_name": "Example Trust",
                    "specialty_code": "100",
                    "specialty_name": "General Surgery",
                    "waiting_list": 700,
                    "completed_admitted": 600,
                    "completed_non_admitted": 500,
                    "waiting_list_with_dta": 12,
                    "new_rtt_periods": 900,
                    "waiting_list_source_available": 1,
                    "completed_admitted_source_available": 1,
                    "completed_non_admitted_source_available": 1,
                    "new_rtt_periods_source_available": 1,
                },
            ]
        )

        cleaned = clean_rtt_operational_features(sample)
        second = cleaned.iloc[1]

        self.assertEqual(float(second["opening_waiting_list"]), 1000.0)
        self.assertEqual(float(second["closing_waiting_list"]), 700.0)
        self.assertEqual(float(second["reported_net_inflow"]), -200.0)
        self.assertEqual(float(second["unreported_removals"]), 100.0)
        self.assertAlmostEqual(float(second["reconciliation_error"]), 0.0, places=6)
        assert_flow_reconciliation_integrity(cleaned)

    def test_negative_unreported_removals_are_preserved(self) -> None:
        sample = pd.DataFrame(
            [
                {
                    "month": "2024-01-01",
                    "trust_code": "R00",
                    "trust_name": "Example Trust",
                    "specialty_code": "100",
                    "specialty_name": "General Surgery",
                    "waiting_list": 1000,
                    "completed_admitted": 50,
                    "completed_non_admitted": 50,
                    "waiting_list_with_dta": 10,
                    "new_rtt_periods": 100,
                    "waiting_list_source_available": 1,
                    "completed_admitted_source_available": 1,
                    "completed_non_admitted_source_available": 1,
                    "new_rtt_periods_source_available": 1,
                },
                {
                    "month": "2024-02-01",
                    "trust_code": "R00",
                    "trust_name": "Example Trust",
                    "specialty_code": "100",
                    "specialty_name": "General Surgery",
                    "waiting_list": 1200,
                    "completed_admitted": 20,
                    "completed_non_admitted": 30,
                    "waiting_list_with_dta": 12,
                    "new_rtt_periods": 100,
                    "waiting_list_source_available": 1,
                    "completed_admitted_source_available": 1,
                    "completed_non_admitted_source_available": 1,
                    "new_rtt_periods_source_available": 1,
                },
            ]
        )

        cleaned = clean_rtt_operational_features(sample)

        self.assertEqual(float(cleaned.loc[1, "unreported_removals"]), -150.0)
        self.assertLess(float(cleaned.loc[1, "unreported_removals"]), 0.0)
        assert_flow_reconciliation_integrity(cleaned)

    def test_missing_source_component_does_not_create_zero_removal(self) -> None:
        sample = pd.DataFrame(
            [
                {
                    "month": "2024-01-01",
                    "trust_code": "R00",
                    "trust_name": "Example Trust",
                    "specialty_code": "100",
                    "specialty_name": "General Surgery",
                    "waiting_list": 1000,
                    "completed_admitted": 50,
                    "completed_non_admitted": 50,
                    "waiting_list_with_dta": 10,
                    "new_rtt_periods": 100,
                    "waiting_list_source_available": 1,
                    "completed_admitted_source_available": 1,
                    "completed_non_admitted_source_available": 1,
                    "new_rtt_periods_source_available": 1,
                },
                {
                    "month": "2024-02-01",
                    "trust_code": "R00",
                    "trust_name": "Example Trust",
                    "specialty_code": "100",
                    "specialty_name": "General Surgery",
                    "waiting_list": 900,
                    "completed_admitted": 60,
                    "completed_non_admitted": 40,
                    "waiting_list_with_dta": 12,
                    "new_rtt_periods": 100,
                    "waiting_list_source_available": 1,
                    "completed_admitted_source_available": 1,
                    "completed_non_admitted_source_available": 1,
                    "new_rtt_periods_source_available": 0,
                },
            ]
        )

        cleaned = clean_rtt_operational_features(sample)
        report = flow_reconciliation_quality_report(cleaned)

        self.assertEqual(int(cleaned.loc[1, "new_rtt_periods_missing"]), 1)
        self.assertEqual(int(cleaned.loc[1, "flow_components_missing"]), 1)
        self.assertTrue(pd.isna(cleaned.loc[1, "unreported_removals"]))
        self.assertEqual(int(report["summary"]["rows_could_not_be_reconciled"]), 2)

    def test_optimisation_forecast_requires_part2a_target(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    COLUMNS.trust_code: "R00",
                    COLUMNS.trust_name: "Example Trust",
                    COLUMNS.specialty_code: "100",
                    COLUMNS.specialty_name: "General Surgery",
                    COLUMNS.forecast_origin: "2024-01-01",
                    COLUMNS.forecast_month: "2024-02-01",
                    COLUMNS.horizon: 1,
                    COLUMNS.p10: 80,
                    COLUMNS.p50: 100,
                    COLUMNS.p90: 130,
                    COLUMNS.latest_observed_incomplete_decision_to_admit: 95,
                    COLUMNS.forecast_target: COLUMNS.incomplete_decision_to_admit,
                    COLUMNS.is_surgical_specialty: True,
                    COLUMNS.specialty_inclusion_criteria: "Included: General Surgery treatment function.",
                }
            ]
        )

        checked = validate_optimisation_forecast_frame(frame, "test optimisation forecast")

        self.assertEqual(str(checked.loc[0, COLUMNS.forecast_target]), COLUMNS.incomplete_decision_to_admit)
        self.assertTrue(bool(checked.loc[0, COLUMNS.is_surgical_specialty]))

    def test_optimisation_forecast_rejects_total_incomplete_target(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    COLUMNS.trust_code: "R00",
                    COLUMNS.trust_name: "Example Trust",
                    COLUMNS.specialty_code: "100",
                    COLUMNS.specialty_name: "General Surgery",
                    COLUMNS.forecast_origin: "2024-01-01",
                    COLUMNS.forecast_month: "2024-02-01",
                    COLUMNS.horizon: 1,
                    COLUMNS.p10: 800,
                    COLUMNS.p50: 1000,
                    COLUMNS.p90: 1300,
                    COLUMNS.latest_observed_incomplete_decision_to_admit: 95,
                    COLUMNS.forecast_target: COLUMNS.incomplete_total,
                    COLUMNS.is_surgical_specialty: True,
                    COLUMNS.specialty_inclusion_criteria: "Included: General Surgery treatment function.",
                }
            ]
        )

        with self.assertRaises(ValueError):
            validate_optimisation_forecast_frame(frame, "bad optimisation forecast")


if __name__ == "__main__":
    unittest.main()
