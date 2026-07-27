from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nhs_rtt_pipeline.config import COLUMNS
from nhs_rtt_pipeline.rolling_origin import (
    RollingOriginConfig,
    calibration_summary,
    quantile_crossing_report,
    rolling_metrics_by_group,
    select_rolling_origins,
    validate_rolling_origin_predictions,
)


class RollingOriginValidationTests(unittest.TestCase):
    def test_selects_configured_origins_without_future_target_overlap(self) -> None:
        months = pd.date_range("2020-01-01", periods=60, freq="MS")
        frame = pd.DataFrame({"month": months})
        config = RollingOriginConfig(
            forecast_horizon=6,
            origin_step_months=3,
            requested_origins=4,
            min_train_months=24,
        )

        origins = select_rolling_origins(frame, config)

        self.assertEqual(len(origins), 4)
        self.assertEqual(origins, sorted(origins))
        self.assertTrue(all(origin + pd.DateOffset(months=6) <= months.max() for origin in origins))
        self.assertTrue(all(origin >= months.min() + pd.DateOffset(months=23) for origin in origins))

    def test_metrics_report_raw_crossing_before_corrected_quantiles(self) -> None:
        rows = []
        origins = pd.date_range("2022-12-01", periods=3, freq="6MS")
        for origin_index, origin in enumerate(origins, start=1):
            for horizon in [1, 2]:
                forecast_month = origin + pd.DateOffset(months=horizon)
                raw_p10 = 120.0 if origin_index == 1 and horizon == 1 else 90.0
                raw_p50 = 100.0
                raw_p90 = 110.0
                rows.append(
                    {
                        "origin_index": origin_index,
                        "model_name": "TCN",
                        COLUMNS.series_id: "R00__100",
                        COLUMNS.trust_code: "R00",
                        COLUMNS.trust_name: "Example Trust",
                        "trust": "Example Trust",
                        COLUMNS.specialty_code: "100",
                        COLUMNS.specialty_name: "General Surgery",
                        "specialty": "General Surgery",
                        COLUMNS.forecast_origin: origin,
                        COLUMNS.forecast_month: forecast_month,
                        COLUMNS.horizon: horizon,
                        "target_column": COLUMNS.incomplete_total,
                        "training_end_month": origin - pd.DateOffset(months=12),
                        "scaler_fit_end_month": origin - pd.DateOffset(months=12),
                        "p10_raw": raw_p10,
                        "p50_raw": raw_p50,
                        "p90_raw": raw_p90,
                        "quantile_crossing_raw": raw_p10 > raw_p50 or raw_p50 > raw_p90,
                        COLUMNS.p10: min(raw_p10, raw_p50, raw_p90),
                        COLUMNS.p50: sorted([raw_p10, raw_p50, raw_p90])[1],
                        COLUMNS.p90: max(raw_p10, raw_p50, raw_p90),
                        COLUMNS.actual: 105.0 + horizon,
                    }
                )
        predictions = pd.DataFrame(rows)

        checked = validate_rolling_origin_predictions(predictions)
        overall = rolling_metrics_by_group(checked, [])
        reliability = calibration_summary(checked)
        crossing = quantile_crossing_report(checked)

        self.assertEqual(len(checked), 6)
        self.assertAlmostEqual(float(overall["quantile_crossing_rate_raw"].iloc[0]), 1.0 / 6.0)
        self.assertIn("winkler_score_80", overall.columns)
        self.assertIn("p10_p90_interval", set(reliability["target"]))
        self.assertEqual(int(crossing["crossing_rows"].sum()), 1)


if __name__ == "__main__":
    unittest.main()
