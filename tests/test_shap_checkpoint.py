from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

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
    replace_with_retry,
)
from nhs_rtt_pipeline.modeling import TCNQuantileRegressor, build_tcn_model_config


def _write_toy_artifacts(directory: Path) -> tuple[Path, Path, Path, pd.DataFrame]:
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


@unittest.skipIf(importlib.util.find_spec("shap") is None, "shap is not installed in this local test runtime")
class ShapCheckpointResumeTests(unittest.TestCase):
    def _build_inputs(self, temp_dir: Path):
        state_path, config_path, metadata_path, clean = _write_toy_artifacts(temp_dir)
        model, model_config, metadata = load_custom_tcn_explainer_bundle(
            state_path, config_path, metadata_path, device="cpu",
        )
        explainer = CustomTCNShapExplainer(model, model_config, metadata, device="cpu", horizons=(1, 2))
        prepared = explainer.prepare_model_frame(clean)
        contexts = explainer.build_contexts(prepared, [2, 3])
        specs = explainer.build_feature_specs(recent_lags=1, include_identifiers=True)
        matrix = explainer.contexts_to_matrix(contexts, specs)
        return explainer, contexts, specs, matrix

    def test_checkpoint_file_written_after_full_run(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            explainer, contexts, specs, matrix = self._build_inputs(temp_path)
            checkpoint_path = temp_path / "checkpoint.npz"

            compute_kernel_shap(
                explainer=explainer,
                contexts=contexts,
                specs=specs,
                background_values=matrix,
                test_values=matrix,
                configured_nsamples=16,
                checkpoint_path=checkpoint_path,
                checkpoint_every=1,
            )

            self.assertTrue(checkpoint_path.exists())
            with np.load(checkpoint_path) as saved:
                self.assertTrue(bool(saved["completed_mask"].all()))

    def test_resuming_complete_checkpoint_never_recomputes_shap(self) -> None:
        # This is the exact regression this test guards: previously, resuming a
        # fully-complete checkpoint still triggered an unconditional checkpoint
        # re-save, which crashed on a transient Windows file lock. It must also
        # never re-invoke the expensive KernelExplainer for already-complete rows.
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            explainer, contexts, specs, matrix = self._build_inputs(temp_path)
            checkpoint_path = temp_path / "checkpoint.npz"

            compute_kernel_shap(
                explainer=explainer,
                contexts=contexts,
                specs=specs,
                background_values=matrix,
                test_values=matrix,
                configured_nsamples=16,
                checkpoint_path=checkpoint_path,
                checkpoint_every=1,
            )
            mtime_before = checkpoint_path.stat().st_mtime_ns

            with mock.patch("shap.KernelExplainer", side_effect=AssertionError("SHAP recomputed a completed context")):
                shap_values, expected_values, model_outputs, audit = compute_kernel_shap(
                    explainer=explainer,
                    contexts=contexts,
                    specs=specs,
                    background_values=matrix,
                    test_values=matrix,
                    configured_nsamples=16,
                    checkpoint_path=checkpoint_path,
                    checkpoint_every=1,
                )

            self.assertEqual(shap_values.shape, (len(contexts), 2, len(specs)))
            self.assertTrue(np.isfinite(shap_values).all())
            # Nothing changed, so the checkpoint file must not have been rewritten either.
            self.assertEqual(checkpoint_path.stat().st_mtime_ns, mtime_before)

    def test_resuming_from_checkpoint_releases_the_file_handle(self) -> None:
        # Regression test: np.load() on a .npz keeps the underlying file open
        # until closed. If the resume path doesn't close it, the checkpoint file
        # cannot be deleted/replaced afterwards on Windows even though nothing
        # else is using it.
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            explainer, contexts, specs, matrix = self._build_inputs(temp_path)
            checkpoint_path = temp_path / "checkpoint.npz"

            compute_kernel_shap(
                explainer=explainer,
                contexts=contexts,
                specs=specs,
                background_values=matrix,
                test_values=matrix,
                configured_nsamples=16,
                checkpoint_path=checkpoint_path,
                checkpoint_every=1,
            )
            compute_kernel_shap(
                explainer=explainer,
                contexts=contexts,
                specs=specs,
                background_values=matrix,
                test_values=matrix,
                configured_nsamples=16,
                checkpoint_path=checkpoint_path,
                checkpoint_every=1,
            )

            # If the resume path left the file handle open, this unlink would
            # raise PermissionError on Windows.
            checkpoint_path.unlink()
            self.assertFalse(checkpoint_path.exists())

    def test_checkpoint_context_id_mismatch_raises(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            explainer, contexts, specs, matrix = self._build_inputs(temp_path)
            checkpoint_path = temp_path / "checkpoint.npz"

            compute_kernel_shap(
                explainer=explainer,
                contexts=contexts,
                specs=specs,
                background_values=matrix,
                test_values=matrix,
                configured_nsamples=16,
                checkpoint_path=checkpoint_path,
                checkpoint_every=1,
            )

            mutated_contexts = [dict(context) for context in contexts]
            mutated_contexts[0]["context_id"] = int(mutated_contexts[0]["context_id"]) + 999

            with self.assertRaises(ValueError):
                compute_kernel_shap(
                    explainer=explainer,
                    contexts=mutated_contexts,
                    specs=specs,
                    background_values=matrix,
                    test_values=matrix,
                    configured_nsamples=16,
                    checkpoint_path=checkpoint_path,
                    checkpoint_every=1,
                )

    def test_checkpoint_shape_mismatch_raises(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            explainer, contexts, specs, matrix = self._build_inputs(temp_path)
            checkpoint_path = temp_path / "checkpoint.npz"

            # Craft a checkpoint whose context_ids match the real contexts but
            # whose shap_values array has an extra feature column, simulating a
            # checkpoint left over from a run with a different SHAP feature set.
            context_ids = np.asarray([int(context["context_id"]) for context in contexts], dtype=np.int64)
            horizons_array = np.asarray(list(explainer.horizons), dtype=np.int64)
            wrong_feature_count = len(specs) + 1
            np.savez_compressed(
                checkpoint_path,
                shap_values=np.zeros((len(contexts), len(explainer.horizons), wrong_feature_count), dtype=np.float64),
                expected_values=np.zeros((len(contexts), len(explainer.horizons)), dtype=np.float64),
                model_outputs=np.zeros((len(contexts), len(explainer.horizons)), dtype=np.float64),
                completed_mask=np.ones(len(contexts), dtype=bool),
                context_ids=context_ids,
                horizons=horizons_array,
                nsamples=np.asarray([16], dtype=np.int64),
            )

            with self.assertRaises(ValueError):
                compute_kernel_shap(
                    explainer=explainer,
                    contexts=contexts,
                    specs=specs,
                    background_values=matrix,
                    test_values=matrix,
                    configured_nsamples=16,
                    checkpoint_path=checkpoint_path,
                    checkpoint_every=1,
                )


class ReplaceWithRetryTests(unittest.TestCase):
    def test_succeeds_immediately_when_no_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source = temp_path / "source.txt"
            destination = temp_path / "destination.txt"
            source.write_text("payload", encoding="utf-8")

            replace_with_retry(source, destination, attempts=3, delay_seconds=0.01)

            self.assertFalse(source.exists())
            self.assertEqual(destination.read_text(encoding="utf-8"), "payload")

    def test_retries_and_recovers_from_transient_permission_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source = temp_path / "source.txt"
            destination = temp_path / "destination.txt"
            source.write_text("payload", encoding="utf-8")

            real_replace = Path.replace
            call_count = {"n": 0}

            def flaky_replace(self, target):
                call_count["n"] += 1
                if call_count["n"] < 3:
                    raise PermissionError("simulated OneDrive/antivirus file lock")
                return real_replace(self, target)

            with mock.patch.object(Path, "replace", flaky_replace):
                replace_with_retry(source, destination, attempts=5, delay_seconds=0.01)

            self.assertEqual(call_count["n"], 3)
            self.assertEqual(destination.read_text(encoding="utf-8"), "payload")

    def test_raises_clear_runtime_error_after_exhausting_attempts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source = temp_path / "source.txt"
            destination = temp_path / "destination.txt"
            source.write_text("payload", encoding="utf-8")

            def always_locked(self, target):
                raise PermissionError("simulated permanent file lock")

            with mock.patch.object(Path, "replace", always_locked):
                with self.assertRaises(RuntimeError) as ctx:
                    replace_with_retry(source, destination, attempts=3, delay_seconds=0.01)

            self.assertIn("locked", str(ctx.exception))
            self.assertIsInstance(ctx.exception.__cause__, PermissionError)


if __name__ == "__main__":
    unittest.main()
