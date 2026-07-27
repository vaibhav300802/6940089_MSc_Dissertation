from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nhs_rtt_pipeline.config import COLUMNS
from nhs_rtt_pipeline.forecasting_baselines import (
    BaselineComparisonConfig,
    add_trust_size_groups,
    build_split_rows,
    prepare_target_frame,
    run_baseline_comparison,
)


class ForecastingBaselineTests(unittest.TestCase):
    def _sample_clean_data(self) -> pd.DataFrame:
        rows = []
        for month_idx in range(14):
            rows.append(
                {
                    "month": pd.Timestamp("2024-01-01") + pd.DateOffset(months=month_idx),
                    COLUMNS.trust_code: "R00",
                    COLUMNS.trust_name: "Example Trust",
                    COLUMNS.specialty_code: "100",
                    COLUMNS.specialty_name: "General Surgery",
                    COLUMNS.series_id: "R00__100",
                    "time_idx": month_idx,
                    COLUMNS.incomplete_total: 1000 + 10 * month_idx + (month_idx % 3) * 5,
                }
            )
        return pd.DataFrame(rows)

    def _metadata(self) -> dict:
        return {
            "target_column": COLUMNS.incomplete_total,
            "config": {"encoder_length": 3, "prediction_length": 2},
            "boundaries": {
                "validation_start_idx": 5,
                "test_start_idx": 8,
                "max_time_idx": 13,
            },
        }

    def _backtest_predictions(self, clean: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        target_frame = prepare_target_frame(
            clean,
            COLUMNS.incomplete_total,
            pd.DataFrame(
                [
                    {
                        COLUMNS.trust_code: "R00",
                        COLUMNS.specialty_code: "100",
                        COLUMNS.forecast_origin: "2024-01-01",
                        COLUMNS.forecast_month: "2024-02-01",
                        COLUMNS.horizon: 1,
                        COLUMNS.p10: 0,
                        COLUMNS.p50: 0,
                        COLUMNS.p90: 0,
                        COLUMNS.actual: 0,
                    }
                ]
            ),
        )
        test_rows = build_split_rows(
            target_frame,
            metadata,
            split="test",
            prediction_length=2,
            encoder_length=3,
            allowed_series_ids={"R00__100"},
        )
        backtest = test_rows[
            [
                COLUMNS.trust_code,
                COLUMNS.trust_name,
                COLUMNS.specialty_code,
                COLUMNS.specialty_name,
                COLUMNS.forecast_origin,
                COLUMNS.forecast_month,
                COLUMNS.horizon,
                COLUMNS.actual,
            ]
        ].copy()
        backtest[COLUMNS.p50] = backtest[COLUMNS.actual] + 3.0
        backtest[COLUMNS.p10] = np.maximum(0.0, backtest[COLUMNS.p50] - 20.0)
        backtest[COLUMNS.p90] = backtest[COLUMNS.p50] + 20.0
        return backtest

    def test_core_baselines_align_to_tcn_backtest_rows(self) -> None:
        clean = self._sample_clean_data()
        metadata = self._metadata()
        backtest = self._backtest_predictions(clean, metadata)
        config = BaselineComparisonConfig(
            seasonal_period=3,
            enable_hist_gradient_boosting=False,
            require_complete_core_baselines=True,
        )

        results = run_baseline_comparison(clean, backtest, metadata, config)

        self.assertEqual(
            set(results.predictions["model"]),
            {"TCN", "naive_last_value", "seasonal_naive_12m", "historical_seasonal_mean"},
        )
        counts = results.predictions.groupby("model").size()
        self.assertEqual(counts.nunique(), 1)
        self.assertEqual(int(counts.iloc[0]), len(backtest))
        self.assertIn("mae", results.model_comparison.columns)
        self.assertIn(COLUMNS.horizon, results.by_horizon.columns)
        self.assertIn(COLUMNS.specialty_name, results.by_specialty.columns)
        self.assertFalse(results.paired_errors.empty)
        self.assertIn("Strongest model by overall MAE", results.summary_text)

    def test_trust_size_grouping_does_not_duplicate_changed_trust_names(self) -> None:
        rows = []
        for model_name in ["TCN", "naive_last_value"]:
            for month_idx, trust_name in enumerate(["Old Trust Name", "New Trust Name"]):
                rows.append(
                    {
                        "model": model_name,
                        COLUMNS.series_id: "R00__100",
                        COLUMNS.trust_code: "R00",
                        COLUMNS.trust_name: trust_name,
                        COLUMNS.specialty_code: "100",
                        COLUMNS.specialty_name: "General Surgery",
                        COLUMNS.forecast_origin: pd.Timestamp("2025-01-01"),
                        COLUMNS.forecast_month: pd.Timestamp("2025-02-01") + pd.DateOffset(months=month_idx),
                        COLUMNS.horizon: month_idx + 1,
                        COLUMNS.actual: 100.0 + month_idx,
                        COLUMNS.p10: 90.0,
                        COLUMNS.p50: 100.0,
                        COLUMNS.p90: 110.0,
                    }
                )
        predictions = pd.DataFrame(rows)
        grouped = add_trust_size_groups(predictions)

        self.assertEqual(len(grouped), len(predictions))
        self.assertIn("trust_size_group", grouped.columns)


if __name__ == "__main__":
    unittest.main()
