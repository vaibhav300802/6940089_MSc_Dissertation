from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("torch is not installed in this local test runtime")

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nhs_rtt_pipeline.config import COLUMNS
from nhs_rtt_pipeline.explainability import (
    CustomTCNShapExplainer,
    compute_kernel_shap,
    load_custom_tcn_explainer_bundle,
)
from nhs_rtt_pipeline.modeling import TCNQuantileRegressor, build_tcn_model_config


class CustomTCNShapTests(unittest.TestCase):
    def _write_toy_artifacts(self, directory: Path) -> tuple[Path, Path, Path, pd.DataFrame]:
        torch.manual_seed(7)
        model_config = build_tcn_model_config(
            n_features=2,
            n_trusts=1,
            n_specialties=1,
            prediction_length=2,
            quantiles=[0.1, 0.5, 0.9],
            hidden_channels=4,
            tcn_levels=1,
            kernel_size=2,
            dropout=0.0,
            embedding_dim=2,
        )
        model = TCNQuantileRegressor(**{key: model_config[key] for key in [
            "n_features",
            "n_trusts",
            "n_specialties",
            "prediction_length",
            "quantiles",
            "hidden_channels",
            "tcn_levels",
            "kernel_size",
            "dropout",
            "embedding_dim",
        ]})
        state_path = directory / "tcn_state_dict.pt"
        config_path = directory / "model_config.json"
        metadata_path = directory / "feature_metadata.json"
        torch.save(model.state_dict(), state_path)
        metadata = {
            "feature_columns": ["waiting_list_model", "completed_total_model"],
            "raw_feature_columns": ["waiting_list", "completed_total"],
            "feature_stats": {
                "waiting_list": {"mean": 5.0, "std": 1.0, "transform": "log1p_non_negative"},
                "completed_total": {"mean": 4.0, "std": 1.0, "transform": "log1p_non_negative"},
            },
            "trust_to_idx": {"R00": 0},
            "specialty_to_idx": {"100": 0},
            "config": {"encoder_length": 3, "prediction_length": 2},
            "quantiles": [0.1, 0.5, 0.9],
        }
        toy_fingerprint = {
            "fingerprint_version": 1,
            "target_column": "waiting_list",
            "row_count": 5,
            "series_count": 1,
            "min_month": "2024-01-01",
            "max_month": "2024-05-01",
            "feature_columns_sha256": "toy",
            "prepared_target_sha256": "toy",
        }
        model_config["artifact_fingerprint"] = toy_fingerprint
        metadata["artifact_fingerprint"] = toy_fingerprint
        config_path.write_text(json.dumps(model_config), encoding="utf-8")
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        clean = pd.DataFrame(
            [
                {
                    "month": f"2024-0{month}-01",
                    COLUMNS.trust_code: "R00",
                    COLUMNS.trust_name: "Example Trust",
                    COLUMNS.specialty_code: "100",
                    COLUMNS.specialty_name: "General Surgery",
                    COLUMNS.series_id: "R00__100",
                    "time_idx": month - 1,
                    "waiting_list": 100.0 + month,
                    "completed_total": 20.0 + month,
                }
                for month in range(1, 6)
            ]
        )
        return state_path, config_path, metadata_path, clean

    def test_saved_custom_tcn_loads_and_reconstructs_encoder_prediction(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path, config_path, metadata_path, clean = self._write_toy_artifacts(Path(temp_dir))
            model, model_config, metadata = load_custom_tcn_explainer_bundle(
                state_path,
                config_path,
                metadata_path,
                device="cpu",
            )
            explainer = CustomTCNShapExplainer(model, model_config, metadata, device="cpu", horizons=(1, 2))
            prepared = explainer.prepare_model_frame(clean)
            contexts = explainer.build_contexts(prepared, [3])
            self.assertEqual(len(contexts), 1)
            encoder_matrix = contexts[0]["encoder"][explainer.feature_columns].to_numpy(dtype=np.float32)
            prediction = explainer.predict_q50_from_encoder(
                encoder_matrix,
                trust_idx=0,
                specialty_idx=0,
                horizons=(1, 2),
            )
            self.assertEqual(prediction.shape, (2,))
            self.assertTrue(np.isfinite(prediction).all())

    @unittest.skipIf(importlib.util.find_spec("shap") is None, "shap is not installed in this local test runtime")
    def test_saved_custom_tcn_produces_one_kernel_shap_explanation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path, config_path, metadata_path, clean = self._write_toy_artifacts(Path(temp_dir))
            model, model_config, metadata = load_custom_tcn_explainer_bundle(
                state_path,
                config_path,
                metadata_path,
                device="cpu",
            )
            explainer = CustomTCNShapExplainer(model, model_config, metadata, device="cpu", horizons=(1,))
            prepared = explainer.prepare_model_frame(clean)
            contexts = explainer.build_contexts(prepared, [3])
            specs = explainer.build_feature_specs(recent_lags=1, include_identifiers=True)
            matrix = explainer.contexts_to_matrix(contexts, specs)
            shap_values, expected_values, model_outputs, audit = compute_kernel_shap(
                explainer=explainer,
                contexts=contexts,
                specs=specs,
                background_values=matrix,
                test_values=matrix,
                configured_nsamples=16,
            )
            self.assertEqual(shap_values.shape, (1, 1, len(specs)))
            self.assertEqual(expected_values.shape, (1, 1))
            self.assertEqual(model_outputs.shape, (1, 1))
            self.assertEqual(str(audit.loc[0, "status"]), "explained")
            reconstructed = expected_values[0, 0] + shap_values[0, 0].sum()
            self.assertTrue(np.isfinite(reconstructed))


if __name__ == "__main__":
    unittest.main()
