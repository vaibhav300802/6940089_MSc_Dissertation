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
from nhs_rtt_pipeline.sequences import make_sliding_window_indices


class SequenceCreationTests(unittest.TestCase):
    def test_sequence_windows_respect_encoder_and_forecast_lengths(self) -> None:
        frame = pd.DataFrame(
            {
                COLUMNS.series_id: ["R00__100"] * 6,
                "time_idx": list(range(6)),
            }
        )

        windows = make_sliding_window_indices(
            frame,
            group_columns=[COLUMNS.series_id],
            time_column="time_idx",
            encoder_length=3,
            prediction_length=2,
        )

        self.assertEqual(len(windows), 2)
        self.assertEqual((windows[0].encoder_start, windows[0].encoder_end), (0, 2))
        self.assertEqual((windows[0].target_start, windows[0].target_end), (3, 4))
        self.assertEqual((windows[1].encoder_start, windows[1].encoder_end), (1, 3))
        self.assertEqual((windows[1].target_start, windows[1].target_end), (4, 5))

    def test_sequence_windows_reject_non_contiguous_time(self) -> None:
        frame = pd.DataFrame(
            {
                COLUMNS.series_id: ["R00__100"] * 5,
                "time_idx": [0, 1, 3, 4, 5],
            }
        )

        with self.assertRaises(ValueError):
            make_sliding_window_indices(
                frame,
                group_columns=[COLUMNS.series_id],
                time_column="time_idx",
                encoder_length=2,
                prediction_length=2,
            )


@unittest.skipIf(importlib.util.find_spec("torch") is None, "torch is not installed in this local test runtime")
class ModelForwardTests(unittest.TestCase):
    def test_custom_tcn_forward_returns_expected_quantile_shape(self) -> None:
        import torch

        from nhs_rtt_pipeline.modeling import TCNQuantileRegressor
        from nhs_rtt_pipeline.reproducibility import set_global_seed

        set_global_seed(42, deterministic_torch=True)
        model = TCNQuantileRegressor(
            n_features=3,
            n_trusts=2,
            n_specialties=4,
            prediction_length=5,
            quantiles=[0.1, 0.5, 0.9],
            hidden_channels=6,
            tcn_levels=2,
            kernel_size=2,
            dropout=0.0,
            embedding_dim=3,
        )
        x = torch.randn(7, 8, 3)
        trust_idx = torch.tensor([0, 1, 0, 1, 0, 1, 0], dtype=torch.long)
        specialty_idx = torch.tensor([0, 1, 2, 3, 0, 1, 2], dtype=torch.long)

        output = model(x, trust_idx, specialty_idx)

        self.assertEqual(tuple(output.shape), (7, 5, 3))
        self.assertTrue(torch.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()
