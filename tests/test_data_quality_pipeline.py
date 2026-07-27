from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nhs_rtt_pipeline.config import COLUMNS
from nhs_rtt_pipeline.data_quality import (
    aggregate_monthly_records,
    duplicate_group_audit,
    harmonise_trust_and_specialty_identifiers,
    missingness_by_series_report,
    trust_identifier_changes_report,
)


class DataQualityPipelineTests(unittest.TestCase):
    def _raw_rows(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "month": "2015-10-01",
                    COLUMNS.trust_code: " r00 ",
                    COLUMNS.trust_name: " Example   Trust ",
                    COLUMNS.specialty_code: "100",
                    COLUMNS.specialty_name: "General Surgery",
                    "waiting_list": 100.0,
                    "new_rtt_periods": 50.0,
                    "completed_admitted": 10.0,
                    "completed_non_admitted": 20.0,
                    "source_zip": "a.zip",
                    "source_csv": "a.csv",
                    "source_url": "https://example.test/a.zip",
                    "source_publication_month": "2015-10-01",
                },
                {
                    "month": "2015-10-01",
                    COLUMNS.trust_code: "R00",
                    COLUMNS.trust_name: "Example Trust",
                    COLUMNS.specialty_code: "100",
                    COLUMNS.specialty_name: "General Surgery",
                    "waiting_list": 120.0,
                    "new_rtt_periods": 50.0,
                    "completed_admitted": 10.0,
                    "completed_non_admitted": 20.0,
                    "source_zip": "b.zip",
                    "source_csv": "b.csv",
                    "source_url": "https://example.test/b.zip",
                    "source_publication_month": "2015-10-01",
                },
            ]
        )

    def test_harmonisation_preserves_original_identifiers_without_merging(self) -> None:
        harmonised = harmonise_trust_and_specialty_identifiers(self._raw_rows())

        self.assertIn("source_trust_code", harmonised.columns)
        self.assertEqual(set(harmonised[COLUMNS.trust_code]), {"R00"})
        self.assertTrue(
            harmonised["trust_identifier_harmonisation_rule"]
            .eq("identity_trim_upper_code_normalise_name_no_organisation_merge")
            .all()
        )

    def test_duplicate_conflicts_are_logged_before_aggregation(self) -> None:
        harmonised = harmonise_trust_and_specialty_identifiers(self._raw_rows())
        value_columns = ["waiting_list", "new_rtt_periods", "completed_admitted", "completed_non_admitted"]
        availability_columns = [f"{column}_source_available" for column in value_columns]
        for column in availability_columns:
            harmonised[column] = 1

        duplicate_audit = duplicate_group_audit(harmonised, value_columns=value_columns)
        monthly = aggregate_monthly_records(harmonised, value_columns=value_columns, availability_columns=availability_columns)

        self.assertFalse(duplicate_audit.empty)
        self.assertIn("waiting_list", duplicate_audit["conflicting_value_columns"].iloc[0])
        self.assertFalse(monthly.duplicated(["month", COLUMNS.trust_code, COLUMNS.specialty_code]).any())
        self.assertEqual(float(monthly["waiting_list"].iloc[0]), 120.0)
        self.assertIn("source_zips", monthly.columns)

    def test_missingness_report_flags_insufficient_history(self) -> None:
        completed = pd.DataFrame(
            [
                {
                    "month": pd.Timestamp("2015-10-01") + pd.DateOffset(months=idx),
                    COLUMNS.series_id: "R00__100",
                    COLUMNS.trust_code: "R00",
                    COLUMNS.trust_name: "Example Trust",
                    COLUMNS.specialty_code: "100",
                    COLUMNS.specialty_name: "General Surgery",
                    "observed_month": 0 if idx == 1 else 1,
                }
                for idx in range(3)
            ]
        )

        report = missingness_by_series_report(completed, min_series_length=6)

        self.assertEqual(int(report["missing_months"].iloc[0]), 1)
        self.assertEqual(int(report["max_consecutive_missing_months"].iloc[0]), 1)
        self.assertTrue(bool(report["excluded_for_insufficient_history"].iloc[0]))

    def test_empty_identifier_change_report_has_schema(self) -> None:
        harmonised = harmonise_trust_and_specialty_identifiers(self._raw_rows().head(1))
        changes = trust_identifier_changes_report(harmonised)

        self.assertTrue(changes.empty)
        self.assertIn("change_type", changes.columns)
        self.assertIn("distinct_values", changes.columns)


if __name__ == "__main__":
    unittest.main()
