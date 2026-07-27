from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nhs_rtt_pipeline.config import COLUMNS
from nhs_rtt_pipeline.schemas import validate_dataframe_schema, validate_generated_outputs
from nhs_rtt_pipeline.settings import load_pipeline_settings


class SchemaValidationTests(unittest.TestCase):
    def test_dataframe_schema_reports_missing_columns(self) -> None:
        frame = pd.DataFrame({COLUMNS.trust_code: ["R00"]})

        with self.assertRaises(ValueError):
            validate_dataframe_schema(frame, [COLUMNS.trust_code, COLUMNS.specialty_code], "toy frame")

    def test_generated_output_schema_report_is_machine_readable(self) -> None:
        report = validate_generated_outputs(PROJECT_ROOT)

        self.assertIn("artifact", report.columns)
        self.assertIn("status", report.columns)
        self.assertGreaterEqual(len(report), 5)

    def test_central_settings_file_loads(self) -> None:
        settings = load_pipeline_settings(project_root=PROJECT_ROOT)

        self.assertEqual(settings.random_seed, 42)
        self.assertEqual(settings.forecasting["forecast_horizon"], 12)
        self.assertEqual(settings.model["model_class"], "TCNQuantileRegressor")


if __name__ == "__main__":
    unittest.main()
