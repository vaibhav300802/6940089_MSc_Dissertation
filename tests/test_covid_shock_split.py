from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nhs_rtt_pipeline.covid_shock import compute_covid_shock_split_boundaries


DEFAULT_KWARGS = dict(
    train_end="2020-02-01",
    covid_test_start="2020-03-01",
    covid_test_end="2021-09-01",
    recovery_start="2021-10-01",
    validation_months=12,
)
WIDE_COVERAGE = {"minimum_month": "2015-10-01", "maximum_month": "2026-05-01"}


class CovidShockSplitBoundaryTests(unittest.TestCase):
    def test_default_project_config_matches_observed_pipeline_output(self) -> None:
        # These are the project's actual default config values (matches the
        # CovidShockExperimentConfig defaults in the layer1b script), and the
        # expected boundaries match what a real successful pipeline run wrote
        # to outputs/covid_stress_test/split_summary.json.
        boundaries = compute_covid_shock_split_boundaries(date_coverage=WIDE_COVERAGE, **DEFAULT_KWARGS)

        self.assertEqual(boundaries.core_train_end, pd.Timestamp("2019-02-01"))
        self.assertEqual(boundaries.validation_start, pd.Timestamp("2019-03-01"))
        self.assertEqual(boundaries.train_end, pd.Timestamp("2020-02-01"))
        self.assertEqual(boundaries.covid_start, pd.Timestamp("2020-03-01"))
        self.assertEqual(boundaries.covid_end, pd.Timestamp("2021-09-01"))
        self.assertEqual(boundaries.recovery_start, pd.Timestamp("2021-10-01"))

    def test_no_leakage_strict_ordering_holds(self) -> None:
        # Encodes the methodological claim directly: core-train < validation <
        # final-pre-COVID-train < COVID-shock-test, with no overlap, for a range
        # of validation window lengths.
        for validation_months in (1, 6, 12, 18, 24):
            kwargs = dict(DEFAULT_KWARGS)
            kwargs["validation_months"] = validation_months
            boundaries = compute_covid_shock_split_boundaries(date_coverage=WIDE_COVERAGE, **kwargs)

            self.assertLess(boundaries.core_train_end, boundaries.validation_start)
            # A 1-month validation window legitimately has validation_start == train_end.
            self.assertLessEqual(boundaries.validation_start, boundaries.train_end)
            self.assertLess(boundaries.train_end, boundaries.covid_start)
            self.assertLessEqual(boundaries.covid_start, boundaries.covid_end)
            self.assertLess(boundaries.covid_end, boundaries.recovery_start)

    def test_train_end_not_before_covid_start_raises(self) -> None:
        kwargs = dict(DEFAULT_KWARGS)
        kwargs["train_end"] = "2020-06-01"  # after covid_test_start of 2020-03-01
        with self.assertRaises(ValueError):
            compute_covid_shock_split_boundaries(date_coverage=WIDE_COVERAGE, **kwargs)

    def test_train_end_equal_to_covid_start_raises(self) -> None:
        kwargs = dict(DEFAULT_KWARGS)
        kwargs["train_end"] = kwargs["covid_test_start"]
        with self.assertRaises(ValueError):
            compute_covid_shock_split_boundaries(date_coverage=WIDE_COVERAGE, **kwargs)

    def test_dataset_ending_before_covid_start_raises(self) -> None:
        short_coverage = {"minimum_month": "2015-10-01", "maximum_month": "2020-01-01"}
        with self.assertRaises(ValueError):
            compute_covid_shock_split_boundaries(date_coverage=short_coverage, **DEFAULT_KWARGS)

    def test_dataset_starting_after_core_train_end_raises(self) -> None:
        late_start_coverage = {"minimum_month": "2019-06-01", "maximum_month": "2026-05-01"}
        with self.assertRaises(ValueError):
            compute_covid_shock_split_boundaries(date_coverage=late_start_coverage, **DEFAULT_KWARGS)


if __name__ == "__main__":
    unittest.main()
