from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nhs_rtt_pipeline.config import COLUMNS, validate_backtest_predictions_frame, validate_future_forecast_frame


class FutureValidationTests(unittest.TestCase):
    def _future_row(self) -> dict[str, object]:
        return {
            COLUMNS.trust_code: "R00",
            COLUMNS.trust_name: "Example Trust",
            COLUMNS.specialty_code: "100",
            COLUMNS.specialty_name: "General Surgery",
            COLUMNS.forecast_origin: "2024-01-01",
            COLUMNS.forecast_month: "2024-02-01",
            COLUMNS.horizon: 1,
            COLUMNS.p10: 90.0,
            COLUMNS.p50: 100.0,
            COLUMNS.p90: 120.0,
            COLUMNS.latest_observed_waiting_list: 95.0,
        }

    def test_future_forecast_rejects_actual_column(self) -> None:
        frame = pd.DataFrame([{**self._future_row(), COLUMNS.actual: 101.0}])

        with self.assertRaises(ValueError):
            validate_future_forecast_frame(frame, "bad future forecast")

    def test_future_forecast_rejects_non_future_month(self) -> None:
        row = self._future_row()
        row[COLUMNS.forecast_month] = "2024-01-01"
        frame = pd.DataFrame([row])

        with self.assertRaises(ValueError):
            validate_future_forecast_frame(frame, "bad future date")

    def test_backtest_requires_actual_values(self) -> None:
        row = self._future_row()
        row.pop(COLUMNS.latest_observed_waiting_list)
        row[COLUMNS.actual] = None
        frame = pd.DataFrame([row])

        with self.assertRaises(ValueError):
            validate_backtest_predictions_frame(frame, "bad backtest")


if __name__ == "__main__":
    unittest.main()
