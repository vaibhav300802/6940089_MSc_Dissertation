from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from .config import COLUMNS
from .modeling import TCNQuantileRegressor, load_json, load_tcn_from_artifacts
from .preprocessing import feature_group_for_column


SHAP_GROUP_WAITING_LIST = "waiting-list lags"
SHAP_GROUP_REFERRAL = "referral or new RTT-period features"
SHAP_GROUP_COMPLETED = "completed-pathway features"
SHAP_GROUP_UNREPORTED = "unreported-removal features"
SHAP_GROUP_CALENDAR = "calendar features"
SHAP_GROUP_IDENTIFIER = "trust and specialty identifiers or embeddings"
SHAP_GROUP_DATA_AVAILABILITY = "data availability indicators"


DISPLAY_ALIASES: Mapping[str, str] = {
    "waiting_list": "total_waiting",
    "incomplete_total": "incomplete_total",
    "opening_waiting_list": "opening_waiting_list",
    "closing_waiting_list": "closing_waiting_list",
    "completed_total": "total_completed_pathways",
    "new_rtt_periods": "new_rtt_periods",
    "reported_net_inflow": "reported_net_inflow",
    "net_inflow": "net_inflow",
    "unreported_removals": "unreported_removals",
    "waiting_list_with_dta": "waiting_list_with_dta",
    "incomplete_decision_to_admit": "incomplete_decision_to_admit",
    "completed_admitted": "completed_admitted",
    "completed_non_admitted": "completed_non_admitted",
    "month_sin": "calendar_month_sin",
    "month_cos": "calendar_month_cos",
    "time_idx": "time_index",
}


PREFERRED_RAW_FEATURES = [
    "waiting_list",
    "incomplete_total",
    "opening_waiting_list",
    "closing_waiting_list",
    "new_rtt_periods",
    "completed_total",
    "completed_admitted",
    "completed_non_admitted",
    "reported_net_inflow",
    "net_inflow",
    "unreported_removals",
    "waiting_list_with_dta",
    "incomplete_decision_to_admit",
    "new_rtt_periods_lag1",
    "new_rtt_periods_lag3",
    "new_rtt_periods_lag6",
    "completed_total_lag1",
    "completed_total_lag3",
    "completed_total_lag6",
    "reported_net_inflow_lag1",
    "reported_net_inflow_lag3",
    "reported_net_inflow_lag6",
    "unreported_removals_lag1",
    "unreported_removals_lag3",
    "unreported_removals_lag6",
    "month_sin",
    "month_cos",
    "time_idx",
]


@dataclass(frozen=True)
class ShapLayerConfig:
    n_background: int = 50
    n_explain: int = 100
    min_distinct_trusts: int = 10
    selected_trust_count: int = 5
    priority_trust_codes: Tuple[str, ...] = ()
    priority_trust_names: Tuple[str, ...] = ()
    recent_lags: int = 3
    nsamples: int = 192
    horizons: Tuple[int, ...] = (1, 6, 12)
    random_seed: int = 42
    max_waterfall_features: int = 14


@dataclass(frozen=True)
class ShapFeatureSpec:
    feature_name: str
    display_name: str
    raw_column: str
    model_column: Optional[str]
    feature_group: str
    lag: int
    kind: str


def safe_filename(value: object, max_length: int = 140) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_")
    return (cleaned or "unnamed")[:max_length]


def inverse_target(values: np.ndarray) -> np.ndarray:
    return np.clip(np.expm1(np.maximum(values, 0.0)), 0.0, None)


def canonical_shap_group(raw_column: str) -> str:
    column = str(raw_column)
    if column in {"trust_idx", "specialty_idx"}:
        return SHAP_GROUP_IDENTIFIER
    if column in {"month_sin", "month_cos", "time_idx", "calendar_month"}:
        return SHAP_GROUP_CALENDAR
    if "unreported_removals" in column:
        return SHAP_GROUP_UNREPORTED
    if column.startswith("completed") or "completed_total" in column:
        return SHAP_GROUP_COMPLETED
    if column.startswith("new_rtt") or "reported_net_inflow" in column or column == "net_inflow":
        return SHAP_GROUP_REFERRAL
    if "waiting_list" in column or "incomplete" in column or column.startswith("opening_") or column.startswith("closing_"):
        return SHAP_GROUP_WAITING_LIST
    if column.endswith("_missing") or "missing" in column:
        return SHAP_GROUP_DATA_AVAILABILITY
    local_group = feature_group_for_column(column)
    mapping = {
        "waiting-list history": SHAP_GROUP_WAITING_LIST,
        "referral or clock-start pressure": SHAP_GROUP_REFERRAL,
        "completed-pathway throughput": SHAP_GROUP_COMPLETED,
        "unreported removals": SHAP_GROUP_UNREPORTED,
        "calendar effects": SHAP_GROUP_CALENDAR,
    }
    return mapping.get(local_group, local_group)


def load_custom_tcn_explainer_bundle(
    state_dict_path: str | Path,
    model_config_path: str | Path,
    feature_metadata_path: str | Path,
    device: torch.device | str = "cpu",
) -> Tuple[TCNQuantileRegressor, Dict[str, Any], Dict[str, Any]]:
    model, model_config = load_tcn_from_artifacts(state_dict_path, model_config_path, device=device)
    metadata_path = Path(feature_metadata_path)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing fitted feature metadata: {metadata_path}")
    metadata = load_json(metadata_path)
    required = [
        "feature_columns",
        "raw_feature_columns",
        "feature_stats",
        "trust_to_idx",
        "specialty_to_idx",
        "config",
    ]
    missing = [key for key in required if key not in metadata]
    if missing:
        raise ValueError(f"Feature metadata is missing required keys: {missing}")
    if int(model_config["n_features"]) != len(metadata["feature_columns"]):
        raise ValueError(
            "Model config n_features does not match feature metadata. "
            f"model_config={model_config['n_features']} metadata={len(metadata['feature_columns'])}"
        )
    model_fingerprint = model_config.get("artifact_fingerprint")
    metadata_fingerprint = metadata.get("artifact_fingerprint")
    if model_fingerprint is None or metadata_fingerprint is None:
        raise ValueError(
            "The TCN model config and feature metadata must include matching artifact fingerprints. "
            "Re-run Layer 1 with `python run_pipeline.py train --force-retrain` before running SHAP."
        )
    if model_fingerprint != metadata_fingerprint:
        raise ValueError(
            "The TCN model config and feature metadata fingerprints do not match. "
            "Re-run Layer 1 with `python run_pipeline.py train --force-prepare --force-retrain`."
        )
    return model, model_config, metadata


class CustomTCNShapExplainer:
    def __init__(
        self,
        model: TCNQuantileRegressor,
        model_config: Mapping[str, Any],
        feature_metadata: Mapping[str, Any],
        device: torch.device | str = "cpu",
        horizons: Optional[Sequence[int]] = None,
    ) -> None:
        self.model = model
        self.model_config = dict(model_config)
        self.metadata = dict(feature_metadata)
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

        self.feature_columns = list(self.metadata["feature_columns"])
        self.raw_feature_columns = list(self.metadata["raw_feature_columns"])
        self.raw_to_model_column = dict(zip(self.raw_feature_columns, self.feature_columns))
        self.feature_stats = dict(self.metadata["feature_stats"])
        self.trust_to_idx = {str(key): int(value) for key, value in self.metadata["trust_to_idx"].items()}
        self.specialty_to_idx = {str(key): int(value) for key, value in self.metadata["specialty_to_idx"].items()}
        self.encoder_length = int(self.metadata.get("config", {}).get("encoder_length", 24))
        self.prediction_length = int(
            self.model_config.get("prediction_length", self.metadata.get("config", {}).get("prediction_length", 12))
        )
        self.quantiles = [float(q) for q in self.model_config.get("quantiles", self.metadata.get("quantiles", [0.1, 0.5, 0.9]))]
        self.q50_index = int(np.argmin(np.abs(np.asarray(self.quantiles, dtype=float) - 0.5)))
        requested_horizons = tuple(int(horizon) for horizon in (horizons or [self.prediction_length]))
        self.horizons = tuple(sorted({h for h in requested_horizons if 1 <= h <= self.prediction_length}))
        if not self.horizons:
            raise ValueError("At least one explanation horizon must be between 1 and prediction_length.")
        self.feature_column_positions = {column: idx for idx, column in enumerate(self.feature_columns)}

    def transform_feature(self, raw_column: str, values: pd.Series) -> pd.Series:
        if raw_column not in self.feature_stats:
            raise ValueError(f"No fitted preprocessing metadata found for raw feature: {raw_column}")
        stats = self.feature_stats[raw_column]
        numeric = pd.to_numeric(values, errors="coerce").astype(float)
        transform_name = str(
            stats.get(
                "transform",
                "log1p_non_negative" if bool(stats.get("log1p", False)) else "identity",
            )
        )
        if transform_name == "log1p_non_negative":
            numeric = np.log1p(numeric.clip(lower=0.0))
        elif transform_name in {"identity", "identity_signed"}:
            numeric = numeric
        else:
            raise ValueError(f"Unsupported fitted feature transform for {raw_column}: {transform_name}")
        mean = float(stats["mean"])
        std = float(stats["std"])
        if not np.isfinite(std) or std < 1.0e-8:
            std = 1.0
        return ((numeric.fillna(mean) - mean) / std).astype("float32")

    def raw_imputation_value(self, raw_column: str) -> float:
        stats = self.feature_stats[raw_column]
        mean = float(stats["mean"])
        transform_name = str(
            stats.get(
                "transform",
                "log1p_non_negative" if bool(stats.get("log1p", False)) else "identity",
            )
        )
        if transform_name == "log1p_non_negative":
            return float(np.expm1(mean))
        return mean

    def prepare_model_frame(self, clean_frame: pd.DataFrame) -> pd.DataFrame:
        frame = clean_frame.copy()
        required_columns = [
            "month",
            COLUMNS.trust_code,
            COLUMNS.trust_name,
            COLUMNS.specialty_code,
            COLUMNS.specialty_name,
            COLUMNS.series_id,
            "time_idx",
            *self.raw_feature_columns,
        ]
        missing = [column for column in required_columns if column not in frame.columns]
        if missing:
            raise ValueError(f"Clean RTT data is missing columns required for TCN SHAP reconstruction: {missing}")
        frame["month"] = pd.to_datetime(frame["month"], errors="coerce")
        if frame["month"].isna().any():
            raise ValueError("Clean RTT data contains invalid month values.")
        frame[COLUMNS.trust_code] = frame[COLUMNS.trust_code].astype(str)
        frame[COLUMNS.specialty_code] = frame[COLUMNS.specialty_code].astype(str)
        frame[COLUMNS.series_id] = frame[COLUMNS.series_id].astype(str)
        frame = frame[
            frame[COLUMNS.trust_code].isin(self.trust_to_idx)
            & frame[COLUMNS.specialty_code].isin(self.specialty_to_idx)
        ].copy()
        if frame.empty:
            raise ValueError("No clean RTT rows match the fitted model Trust/specialty indices.")
        frame["trust_idx"] = frame[COLUMNS.trust_code].map(self.trust_to_idx).astype("int64")
        frame["specialty_idx"] = frame[COLUMNS.specialty_code].map(self.specialty_to_idx).astype("int64")
        for raw_column in self.raw_feature_columns:
            model_column = self.raw_to_model_column[raw_column]
            frame[model_column] = self.transform_feature(raw_column, frame[raw_column])
        return frame.sort_values([COLUMNS.series_id, "time_idx"]).reset_index(drop=True)

    def selected_raw_features(self) -> List[str]:
        selected: List[str] = []
        for column in PREFERRED_RAW_FEATURES:
            if column in self.raw_feature_columns and column not in selected:
                selected.append(column)
        for column in self.raw_feature_columns:
            if column in selected:
                continue
            group = canonical_shap_group(column)
            if group in {
                SHAP_GROUP_WAITING_LIST,
                SHAP_GROUP_REFERRAL,
                SHAP_GROUP_COMPLETED,
                SHAP_GROUP_UNREPORTED,
                SHAP_GROUP_CALENDAR,
            }:
                selected.append(column)
        if not selected:
            raise RuntimeError("No raw features were selected for SHAP explanation.")
        return selected

    def build_feature_specs(self, recent_lags: int, include_identifiers: bool = True) -> List[ShapFeatureSpec]:
        lag_count = max(1, min(int(recent_lags), self.encoder_length))
        specs: List[ShapFeatureSpec] = []
        for raw_column in self.selected_raw_features():
            display_name = DISPLAY_ALIASES.get(raw_column, raw_column)
            model_column = self.raw_to_model_column[raw_column]
            for lag in range(lag_count):
                specs.append(
                    ShapFeatureSpec(
                        feature_name=f"{display_name}__lag_{lag:02d}",
                        display_name=display_name,
                        raw_column=raw_column,
                        model_column=model_column,
                        feature_group=canonical_shap_group(raw_column),
                        lag=lag,
                        kind="raw_encoder_feature",
                    )
                )
        if include_identifiers:
            specs.extend(
                [
                    ShapFeatureSpec(
                        feature_name="trust_identifier_embedding",
                        display_name="trust_identifier_embedding",
                        raw_column="trust_idx",
                        model_column=None,
                        feature_group=SHAP_GROUP_IDENTIFIER,
                        lag=0,
                        kind="trust_identifier",
                    ),
                    ShapFeatureSpec(
                        feature_name="specialty_identifier_embedding",
                        display_name="specialty_identifier_embedding",
                        raw_column="specialty_idx",
                        model_column=None,
                        feature_group=SHAP_GROUP_IDENTIFIER,
                        lag=0,
                        kind="specialty_identifier",
                    ),
                ]
            )
        return specs

    def predict_q50_from_encoder(
        self,
        encoder_matrix: np.ndarray,
        trust_idx: int,
        specialty_idx: int,
        horizons: Optional[Sequence[int]] = None,
    ) -> np.ndarray:
        requested = tuple(int(h) for h in (horizons or self.horizons))
        x = torch.tensor(encoder_matrix[None, :, :], dtype=torch.float32, device=self.device)
        trust_tensor = torch.tensor([int(trust_idx)], dtype=torch.long, device=self.device)
        specialty_tensor = torch.tensor([int(specialty_idx)], dtype=torch.long, device=self.device)
        with torch.no_grad():
            prediction_log = self.model(x, trust_tensor, specialty_tensor).detach().cpu().numpy()
        prediction_log.sort(axis=-1)
        values = []
        for horizon in requested:
            horizon_index = min(max(int(horizon) - 1, 0), prediction_log.shape[1] - 1)
            values.append(float(prediction_log[0, horizon_index, self.q50_index]))
        return inverse_target(np.asarray(values, dtype=float))

    def predict_q50_from_encoder_batch(
        self,
        encoder_matrices: np.ndarray,
        trust_indices: np.ndarray,
        specialty_indices: np.ndarray,
        horizons: Optional[Sequence[int]] = None,
        batch_size: int = 1024,
    ) -> np.ndarray:
        requested = tuple(int(h) for h in (horizons or self.horizons))
        matrices = np.asarray(encoder_matrices, dtype=np.float32)
        if matrices.ndim != 3:
            raise ValueError(f"Expected encoder_matrices with shape (n, sequence, features), received {matrices.shape}.")
        trust_array = np.asarray(trust_indices, dtype=np.int64).reshape(-1)
        specialty_array = np.asarray(specialty_indices, dtype=np.int64).reshape(-1)
        if len(trust_array) != matrices.shape[0] or len(specialty_array) != matrices.shape[0]:
            raise ValueError("Batch encoder, Trust index and specialty index lengths do not match.")

        outputs: List[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, matrices.shape[0], int(batch_size)):
                end = min(start + int(batch_size), matrices.shape[0])
                x = torch.tensor(matrices[start:end], dtype=torch.float32, device=self.device)
                trust_tensor = torch.tensor(trust_array[start:end], dtype=torch.long, device=self.device)
                specialty_tensor = torch.tensor(specialty_array[start:end], dtype=torch.long, device=self.device)
                prediction_log = self.model(x, trust_tensor, specialty_tensor).detach().cpu().numpy()
                prediction_log.sort(axis=-1)
                selected = []
                for horizon in requested:
                    horizon_index = min(max(int(horizon) - 1, 0), prediction_log.shape[1] - 1)
                    selected.append(prediction_log[:, horizon_index, self.q50_index])
                outputs.append(np.vstack(selected).T)
        if not outputs:
            return np.empty((0, len(requested)), dtype=np.float64)
        return inverse_target(np.vstack(outputs).astype(np.float64))

    def build_contexts(self, frame: pd.DataFrame, forecast_start_indices: Iterable[int]) -> List[Dict[str, Any]]:
        contexts: List[Dict[str, Any]] = []
        grouped = frame.groupby(COLUMNS.series_id, sort=False, observed=True)
        for series_id, group in grouped:
            group = group.sort_values("time_idx").reset_index(drop=True)
            available = set(group["time_idx"].astype(int).tolist())
            for forecast_start_idx in forecast_start_indices:
                forecast_start = int(forecast_start_idx)
                encoder_start = forecast_start - self.encoder_length
                encoder_required = set(range(encoder_start, forecast_start))
                if not encoder_required.issubset(available):
                    continue
                encoder = group[
                    (group["time_idx"] >= encoder_start)
                    & (group["time_idx"] < forecast_start)
                ].copy()
                if len(encoder) != self.encoder_length:
                    continue
                if not np.all(np.diff(encoder["time_idx"].to_numpy(dtype=int)) == 1):
                    continue
                latest = encoder.iloc[-1]
                forecast_origin = pd.to_datetime(encoder["month"].iloc[-1])
                contexts.append(
                    {
                        "context_id": len(contexts),
                        "series_id": str(series_id),
                        COLUMNS.trust_code: str(latest[COLUMNS.trust_code]),
                        COLUMNS.trust_name: str(latest[COLUMNS.trust_name]),
                        COLUMNS.specialty_code: str(latest[COLUMNS.specialty_code]),
                        COLUMNS.specialty_name: str(latest[COLUMNS.specialty_name]),
                        "trust_idx": int(latest["trust_idx"]),
                        "specialty_idx": int(latest["specialty_idx"]),
                        "forecast_start_idx": forecast_start,
                        COLUMNS.forecast_origin: forecast_origin,
                        "encoder": encoder.reset_index(drop=True),
                    }
                )
        return contexts

    def context_feature_vector(self, context: Mapping[str, Any], specs: Sequence[ShapFeatureSpec]) -> np.ndarray:
        encoder = context["encoder"]
        values = []
        for spec in specs:
            if spec.kind == "trust_identifier":
                values.append(float(context["trust_idx"]))
                continue
            if spec.kind == "specialty_identifier":
                values.append(float(context["specialty_idx"]))
                continue
            position = self.encoder_length - 1 - int(spec.lag)
            raw_value = pd.to_numeric(pd.Series([encoder.loc[position, spec.raw_column]]), errors="coerce").iloc[0]
            if pd.isna(raw_value):
                raw_value = self.raw_imputation_value(spec.raw_column)
            values.append(float(raw_value))
        return np.asarray(values, dtype=np.float64)

    def contexts_to_matrix(self, contexts: Sequence[Mapping[str, Any]], specs: Sequence[ShapFeatureSpec]) -> np.ndarray:
        if not contexts:
            raise ValueError("No contexts were supplied for SHAP matrix construction.")
        return np.vstack([self.context_feature_vector(context, specs) for context in contexts]).astype(np.float64)

    def wrapper_for_context(self, context: Mapping[str, Any], specs: Sequence[ShapFeatureSpec]) -> "TCNKernelContextWrapper":
        return TCNKernelContextWrapper(self, context, specs)


class TCNKernelContextWrapper:
    def __init__(
        self,
        explainer: CustomTCNShapExplainer,
        context: Mapping[str, Any],
        specs: Sequence[ShapFeatureSpec],
    ) -> None:
        self.explainer = explainer
        self.context = context
        self.specs = list(specs)
        self.base_encoder = context["encoder"].copy()
        self.base_matrix = self.base_encoder[explainer.feature_columns].to_numpy(dtype=np.float32)
        self.n_trusts = int(explainer.model_config["n_trusts"])
        self.n_specialties = int(explainer.model_config["n_specialties"])

    @staticmethod
    def _bounded_index(value: float, upper: int) -> int:
        if not np.isfinite(value):
            return 0
        return int(min(max(round(float(value)), 0), upper - 1))

    @staticmethod
    def _bounded_indices(values: np.ndarray, upper: int) -> np.ndarray:
        numeric = np.asarray(values, dtype=np.float64)
        numeric = np.where(np.isfinite(numeric), numeric, 0.0)
        return np.rint(np.clip(numeric, 0, int(upper) - 1)).astype(np.int64)

    def __call__(self, x_numpy: np.ndarray) -> np.ndarray:
        values = np.asarray(x_numpy, dtype=np.float64)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.shape[0] == 0:
            return np.empty((0, len(self.explainer.horizons)), dtype=np.float64)

        matrices = np.repeat(self.base_matrix[None, :, :], values.shape[0], axis=0).astype(np.float32, copy=False)
        trust_indices = np.full(values.shape[0], int(self.context["trust_idx"]), dtype=np.int64)
        specialty_indices = np.full(values.shape[0], int(self.context["specialty_idx"]), dtype=np.int64)

        for feature_position, spec in enumerate(self.specs):
            column_values = values[:, feature_position]
            if spec.kind == "trust_identifier":
                trust_indices = self._bounded_indices(column_values, self.n_trusts)
                continue
            if spec.kind == "specialty_identifier":
                specialty_indices = self._bounded_indices(column_values, self.n_specialties)
                continue
            if spec.model_column is None:
                continue
            encoder_position = self.explainer.encoder_length - 1 - int(spec.lag)
            model_feature_index = self.explainer.feature_column_positions[spec.model_column]
            transformed = self.explainer.transform_feature(spec.raw_column, pd.Series(column_values)).to_numpy(dtype=np.float32)
            matrices[:, encoder_position, model_feature_index] = transformed

        return self.explainer.predict_q50_from_encoder_batch(
            matrices,
            trust_indices,
            specialty_indices,
            horizons=self.explainer.horizons,
        ).astype(np.float64)


def normalise_shap_values(values: Any, n_horizons: int, n_features: int) -> np.ndarray:
    if isinstance(values, list):
        if len(values) != n_horizons:
            raise ValueError(f"Expected {n_horizons} SHAP output arrays, received {len(values)}.")
        stacked = []
        for output_values in values:
            arr = np.asarray(output_values, dtype=np.float64)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            stacked.append(arr[0])
        return np.vstack(stacked)

    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1 and n_horizons == 1:
        return arr.reshape(1, -1)
    if arr.ndim == 2 and n_horizons == 1:
        return arr[0].reshape(1, -1)
    if arr.ndim == 3:
        if arr.shape[0] == 1 and arr.shape[1] == n_features and arr.shape[2] == n_horizons:
            return arr[0].T
        if arr.shape[0] == n_horizons and arr.shape[-1] == n_features:
            return arr[:, 0, :]
    raise ValueError(f"Unsupported SHAP value shape {arr.shape} for {n_horizons} horizons and {n_features} features.")


def normalise_expected_values(expected_value: Any, n_horizons: int) -> np.ndarray:
    arr = np.asarray(expected_value, dtype=np.float64).reshape(-1)
    if len(arr) == 1 and n_horizons > 1:
        return np.repeat(arr[0], n_horizons)
    if len(arr) != n_horizons:
        raise ValueError(f"Expected {n_horizons} SHAP base values, received {len(arr)}.")
    return arr


def kernel_nsamples(n_features: int, configured_nsamples: int) -> int:
    return int(max(configured_nsamples, 2 * n_features + 32))


def replace_with_retry(source: Path, destination: Path, attempts: int = 8, delay_seconds: float = 1.5) -> None:
    """Atomically replace ``destination`` with ``source``, retrying on transient Windows file locks.

    On Windows, cloud-sync clients (OneDrive) and antivirus real-time scanning can transiently
    hold an open handle on the destination file, which makes ``os.replace()`` raise
    ``PermissionError`` (WinError 5) even though nothing is actually wrong with the run. Retry
    briefly instead of losing hours of completed SHAP computation to a momentary file lock.
    """
    last_error: Optional[OSError] = None
    for attempt in range(1, attempts + 1):
        try:
            source.replace(destination)
            return
        except PermissionError as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(delay_seconds)
    raise RuntimeError(
        f"Could not replace {destination} after {attempts} attempts because the file stayed locked "
        "(commonly OneDrive sync, Windows Search Indexer, or antivirus real-time scanning). "
        "The in-memory SHAP progress for this run is intact; rerun the explain stage once the file "
        "is free, or pause OneDrive/Defender scanning for this project folder."
    ) from last_error


def compute_kernel_shap(
    explainer: CustomTCNShapExplainer,
    contexts: Sequence[Mapping[str, Any]],
    specs: Sequence[ShapFeatureSpec],
    background_values: np.ndarray,
    test_values: np.ndarray,
    configured_nsamples: int,
    checkpoint_path: Optional[str | Path] = None,
    audit_checkpoint_path: Optional[str | Path] = None,
    checkpoint_every: int = 1,
    progress_every: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    import shap

    n_contexts = len(contexts)
    n_horizons = len(explainer.horizons)
    n_features = len(specs)
    shap_values = np.zeros((n_contexts, n_horizons, n_features), dtype=np.float64)
    expected_values = np.zeros((n_contexts, n_horizons), dtype=np.float64)
    model_outputs = np.zeros((n_contexts, n_horizons), dtype=np.float64)
    completed_mask = np.zeros(n_contexts, dtype=bool)
    nsamples = kernel_nsamples(n_features, configured_nsamples)

    audit_rows = []
    checkpoint = Path(checkpoint_path) if checkpoint_path is not None else None
    audit_checkpoint = Path(audit_checkpoint_path) if audit_checkpoint_path is not None else None
    context_ids = np.asarray([int(context["context_id"]) for context in contexts], dtype=np.int64)
    horizons_array = np.asarray(list(explainer.horizons), dtype=np.int64)

    if checkpoint is not None and checkpoint.exists():
        # Use a `with` block so the underlying file handle is released immediately
        # after the arrays are copied out. Left open (as a bare `np.load()` would
        # leave it), this handle survives for the rest of the function — on
        # Windows that then blocks every later attempt to replace this same file
        # during checkpointing, regardless of OneDrive/antivirus behaviour.
        with np.load(checkpoint, allow_pickle=False) as checkpoint_data:
            existing_shape = tuple(checkpoint_data["shap_values"].shape)
            expected_shape = tuple(shap_values.shape)
            if existing_shape != expected_shape:
                raise ValueError(
                    f"Existing SHAP checkpoint has shape {existing_shape}; expected {expected_shape}. "
                    f"Remove the stale checkpoint if you intentionally changed SHAP settings: {checkpoint}"
                )
            saved_context_ids = checkpoint_data["context_ids"].astype(np.int64)
            if not np.array_equal(saved_context_ids, context_ids):
                raise ValueError("Existing SHAP checkpoint context ids do not match the current selected contexts.")
            shap_values[:] = checkpoint_data["shap_values"]
            expected_values[:] = checkpoint_data["expected_values"]
            model_outputs[:] = checkpoint_data["model_outputs"]
            completed_mask[:] = checkpoint_data["completed_mask"].astype(bool)
        if audit_checkpoint is not None and audit_checkpoint.exists():
            audit_rows = pd.read_csv(audit_checkpoint).to_dict("records")
        print(
            f"Resuming SHAP from checkpoint {checkpoint}: "
            f"{int(completed_mask.sum())}/{n_contexts} contexts already complete.",
            flush=True,
        )

    checkpoint_dirty = checkpoint is not None and not checkpoint.exists()

    def save_checkpoint(force: bool = False) -> None:
        nonlocal checkpoint_dirty
        if checkpoint is None:
            return
        if not force and not checkpoint_dirty:
            return
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        temporary = checkpoint.with_name(f"{checkpoint.name}.tmp.npz")
        np.savez_compressed(
            temporary,
            shap_values=shap_values,
            expected_values=expected_values,
            model_outputs=model_outputs,
            completed_mask=completed_mask,
            context_ids=context_ids,
            horizons=horizons_array,
            nsamples=np.asarray([nsamples], dtype=np.int64),
        )
        replace_with_retry(temporary, checkpoint)
        if audit_checkpoint is not None:
            pd.DataFrame(audit_rows).to_csv(audit_checkpoint, index=False)
        checkpoint_dirty = False

    start_time = time.time()
    for row_index, context in enumerate(contexts):
        if completed_mask[row_index]:
            continue
        row_started = time.time()
        if progress_every > 0 and (row_index == 0 or row_index % int(progress_every) == 0):
            print(
                f"SHAP context {row_index + 1}/{n_contexts} "
                f"(context_id={context['context_id']}, nsamples={nsamples}) started.",
                flush=True,
            )
        wrapper = explainer.wrapper_for_context(context, specs)
        instance = test_values[row_index : row_index + 1, :]
        output = wrapper(instance).reshape(-1)
        if not np.isfinite(output).all():
            audit_rows.append(
                {
                    "context_id": context["context_id"],
                    "status": "excluded",
                    "reason": "model prediction contained non-finite values",
                }
            )
            raise RuntimeError(f"Missing or non-finite prediction for SHAP context {context['context_id']}.")
        kernel_explainer = shap.KernelExplainer(wrapper, background_values, link="identity")
        raw_values = kernel_explainer.shap_values(
            instance,
            nsamples=nsamples,
            l1_reg="num_features(10)",
            silent=True,
        )
        values = normalise_shap_values(raw_values, n_horizons=n_horizons, n_features=n_features)
        base = normalise_expected_values(kernel_explainer.expected_value, n_horizons=n_horizons)
        shap_values[row_index] = values
        expected_values[row_index] = base
        model_outputs[row_index] = output
        completed_mask[row_index] = True
        checkpoint_dirty = True
        audit_rows.append(
            {
                "context_id": context["context_id"],
                "status": "explained",
                "reason": "",
                "nsamples": nsamples,
                "explained_horizons": ",".join(str(h) for h in explainer.horizons),
                "row_index": row_index,
                "elapsed_seconds": round(time.time() - row_started, 3),
            }
        )
        if checkpoint_every > 0 and (int(completed_mask.sum()) % int(checkpoint_every) == 0):
            save_checkpoint()
        if progress_every > 0 and ((row_index + 1) % int(progress_every) == 0 or int(completed_mask.sum()) == n_contexts):
            completed = int(completed_mask.sum())
            elapsed = time.time() - start_time
            rate = completed / max(elapsed, 1.0)
            remaining = (n_contexts - completed) / rate if rate > 0 else float("nan")
            print(
                f"SHAP context {row_index + 1}/{n_contexts} complete. "
                f"Completed={completed}/{n_contexts}; elapsed={elapsed / 60.0:.1f} min; "
                f"ETA={remaining / 60.0:.1f} min.",
                flush=True,
            )
    save_checkpoint()
    if not completed_mask.all():
        missing = np.where(~completed_mask)[0].tolist()
        raise RuntimeError(f"SHAP did not complete all contexts. Missing row positions: {missing[:20]}")
    return shap_values, expected_values, model_outputs, pd.DataFrame(audit_rows)


def context_index_frame(
    contexts: Sequence[Mapping[str, Any]],
    model_outputs: Optional[np.ndarray] = None,
    horizons: Optional[Sequence[int]] = None,
) -> pd.DataFrame:
    rows = []
    horizons = tuple(horizons or [])
    for context_index, context in enumerate(contexts):
        row = {
            "context_id": int(context["context_id"]),
            COLUMNS.series_id: context["series_id"],
            COLUMNS.trust_code: context[COLUMNS.trust_code],
            COLUMNS.trust_name: context[COLUMNS.trust_name],
            COLUMNS.specialty_code: context[COLUMNS.specialty_code],
            COLUMNS.specialty_name: context[COLUMNS.specialty_name],
            "trust_idx": int(context["trust_idx"]),
            "specialty_idx": int(context["specialty_idx"]),
            "forecast_start_idx": int(context["forecast_start_idx"]),
            COLUMNS.forecast_origin: pd.to_datetime(context[COLUMNS.forecast_origin]),
        }
        for horizon_index, horizon in enumerate(horizons):
            row[f"forecast_month_h{horizon}"] = pd.to_datetime(context[COLUMNS.forecast_origin]) + pd.DateOffset(months=int(horizon))
            if model_outputs is not None:
                row[f"p50_h{horizon}"] = float(model_outputs[context_index, horizon_index])
        rows.append(row)
    return pd.DataFrame(rows)


def shap_values_long_frame(
    shap_values: np.ndarray,
    expected_values: np.ndarray,
    model_outputs: np.ndarray,
    feature_values: np.ndarray,
    contexts: Sequence[Mapping[str, Any]],
    specs: Sequence[ShapFeatureSpec],
    horizons: Sequence[int],
) -> pd.DataFrame:
    rows = []
    for context_position, context in enumerate(contexts):
        for horizon_position, horizon in enumerate(horizons):
            forecast_month = pd.to_datetime(context[COLUMNS.forecast_origin]) + pd.DateOffset(months=int(horizon))
            for feature_position, spec in enumerate(specs):
                shap_value = float(shap_values[context_position, horizon_position, feature_position])
                rows.append(
                    {
                        "context_id": int(context["context_id"]),
                        COLUMNS.series_id: context["series_id"],
                        COLUMNS.trust_code: context[COLUMNS.trust_code],
                        COLUMNS.trust_name: context[COLUMNS.trust_name],
                        COLUMNS.specialty_code: context[COLUMNS.specialty_code],
                        COLUMNS.specialty_name: context[COLUMNS.specialty_name],
                        COLUMNS.forecast_origin: pd.to_datetime(context[COLUMNS.forecast_origin]),
                        COLUMNS.forecast_month: forecast_month,
                        COLUMNS.horizon: int(horizon),
                        COLUMNS.p50: float(model_outputs[context_position, horizon_position]),
                        "base_value": float(expected_values[context_position, horizon_position]),
                        "feature_name": spec.feature_name,
                        "display_name": spec.display_name,
                        "raw_column": spec.raw_column,
                        "feature_group": spec.feature_group,
                        "lag": int(spec.lag),
                        "feature_kind": spec.kind,
                        "feature_value": float(feature_values[context_position, feature_position]),
                        "shap_value": shap_value,
                        "abs_shap_value": abs(shap_value),
                    }
                )
    return pd.DataFrame(rows)


def consistency_report_frame(
    shap_values: np.ndarray,
    expected_values: np.ndarray,
    model_outputs: np.ndarray,
    contexts: Sequence[Mapping[str, Any]],
    horizons: Sequence[int],
) -> pd.DataFrame:
    rows = []
    approx = expected_values + shap_values.sum(axis=2)
    error = approx - model_outputs
    for context_position, context in enumerate(contexts):
        for horizon_position, horizon in enumerate(horizons):
            rows.append(
                {
                    "context_id": int(context["context_id"]),
                    COLUMNS.trust_code: context[COLUMNS.trust_code],
                    COLUMNS.trust_name: context[COLUMNS.trust_name],
                    COLUMNS.specialty_code: context[COLUMNS.specialty_code],
                    COLUMNS.specialty_name: context[COLUMNS.specialty_name],
                    COLUMNS.horizon: int(horizon),
                    "base_value": float(expected_values[context_position, horizon_position]),
                    "sum_feature_contributions": float(shap_values[context_position, horizon_position].sum()),
                    "base_plus_sum": float(approx[context_position, horizon_position]),
                    "model_output": float(model_outputs[context_position, horizon_position]),
                    "approximation_error": float(error[context_position, horizon_position]),
                    "absolute_approximation_error": float(abs(error[context_position, horizon_position])),
                }
            )
    return pd.DataFrame(rows)


def aggregate_by_display(
    shap_values: np.ndarray,
    feature_values: np.ndarray,
    specs: Sequence[ShapFeatureSpec],
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    display_names: List[str] = []
    display_groups: List[str] = []
    for spec in specs:
        if spec.display_name not in display_names:
            display_names.append(spec.display_name)
            display_groups.append(spec.feature_group)
    aggregated_shap = np.zeros((shap_values.shape[0], shap_values.shape[1], len(display_names)), dtype=np.float64)
    aggregated_features = np.zeros((feature_values.shape[0], len(display_names)), dtype=np.float64)
    for display_index, display_name in enumerate(display_names):
        indices = [idx for idx, spec in enumerate(specs) if spec.display_name == display_name]
        aggregated_shap[:, :, display_index] = shap_values[:, :, indices].sum(axis=2)
        aggregated_features[:, display_index] = feature_values[:, indices].mean(axis=1)
    return aggregated_shap, aggregated_features, display_names, display_groups


def feature_importance_frames(values_long: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    horizon_feature = (
        values_long.groupby(
            [
                COLUMNS.horizon,
                "feature_name",
                "display_name",
                "raw_column",
                "feature_group",
                "lag",
                "feature_kind",
            ],
            as_index=False,
            observed=True,
        )["abs_shap_value"]
        .mean()
        .rename(columns={"abs_shap_value": "mean_abs_shap"})
        .sort_values([COLUMNS.horizon, "mean_abs_shap"], ascending=[True, False])
        .reset_index(drop=True)
    )
    global_feature = (
        values_long.groupby(
            ["feature_name", "display_name", "raw_column", "feature_group", "lag", "feature_kind"],
            as_index=False,
            observed=True,
        )["abs_shap_value"]
        .mean()
        .rename(columns={"abs_shap_value": "mean_abs_shap"})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    group_importance = (
        values_long.groupby([COLUMNS.horizon, "feature_group"], as_index=False, observed=True)["abs_shap_value"]
        .mean()
        .rename(columns={"abs_shap_value": "mean_abs_shap"})
        .sort_values([COLUMNS.horizon, "mean_abs_shap"], ascending=[True, False])
        .reset_index(drop=True)
    )
    return global_feature, horizon_feature, group_importance


def local_trust_specialty_explanations(values_long: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        values_long.groupby(
            [
                COLUMNS.trust_code,
                COLUMNS.trust_name,
                COLUMNS.specialty_code,
                COLUMNS.specialty_name,
                COLUMNS.horizon,
                "feature_group",
            ],
            as_index=False,
            observed=True,
        )
        .agg(
            mean_shap_value=("shap_value", "mean"),
            mean_abs_shap_value=("abs_shap_value", "mean"),
            mean_feature_value=("feature_value", "mean"),
            mean_model_p50=(COLUMNS.p50, "mean"),
        )
        .sort_values(
            [COLUMNS.trust_name, COLUMNS.specialty_name, COLUMNS.horizon, "mean_abs_shap_value"],
            ascending=[True, True, True, False],
        )
        .reset_index(drop=True)
    )
    grouped["interpretation_note"] = grouped.apply(local_group_note, axis=1)
    return grouped


def local_group_note(row: pd.Series) -> str:
    group = str(row["feature_group"])
    contribution = float(row["mean_shap_value"])
    direction = "positively" if contribution > 0 else "negatively" if contribution < 0 else "neutrally"
    if group == SHAP_GROUP_COMPLETED:
        return (
            f"Completed-pathway features contributed {direction} to this model prediction. "
            "Interpret this together with the feature values; a positive contribution is not automatically evidence of reduced throughput."
        )
    if group == SHAP_GROUP_REFERRAL:
        return f"Referral or new RTT-period features contributed {direction} to this model prediction."
    if group == SHAP_GROUP_WAITING_LIST:
        return f"Waiting-list history features contributed {direction} to this model prediction."
    if group == SHAP_GROUP_UNREPORTED:
        return f"Unreported-removal accounting features contributed {direction} to this model prediction."
    if group == SHAP_GROUP_IDENTIFIER:
        return f"Trust or specialty embedding identifiers contributed {direction} to this model prediction."
    return f"{group} contributed {direction} to this model prediction."


def interpret_trust(
    trust_name: str,
    local_explanations: pd.DataFrame,
    horizon: Optional[int] = None,
) -> str:
    if local_explanations.empty:
        return f"Trust {trust_name} does not yet have a generated SHAP explanation."
    frame = local_explanations[local_explanations[COLUMNS.trust_name].astype(str).eq(str(trust_name))].copy()
    if frame.empty:
        return f"Trust {trust_name} does not yet have a generated SHAP explanation."
    if horizon is not None and COLUMNS.horizon in frame.columns:
        frame = frame[frame[COLUMNS.horizon].astype(int).eq(int(horizon))]
    if frame.empty:
        return f"Trust {trust_name} does not yet have a generated SHAP explanation at horizon {horizon}."
    summary = (
        frame.groupby("feature_group", as_index=False, observed=True)["mean_shap_value"]
        .mean()
        .sort_values("mean_shap_value", ascending=False)
    )
    positive = summary[summary["mean_shap_value"] > 0].sort_values("mean_shap_value", ascending=False)
    if positive.empty:
        return (
            f"Trust {trust_name}'s generated explanations do not show a dominant positive feature group; "
            "the model's median forecast is mainly moderated by the available historical inputs."
        )
    top = positive.iloc[0]
    group = str(top["feature_group"])
    if group == SHAP_GROUP_REFERRAL:
        return (
            f"For Trust {trust_name}, referral or new RTT-period features were associated with a higher model forecast "
            "in the selected explanation rows."
        )
    if group == SHAP_GROUP_COMPLETED:
        return (
            f"For Trust {trust_name}, completed-pathway features contributed positively to the selected model prediction. "
            "Review their observed feature values before interpreting the operational direction."
        )
    if group == SHAP_GROUP_WAITING_LIST:
        return (
            f"For Trust {trust_name}, waiting-list history was associated with a higher model forecast "
            "in the selected explanation rows."
        )
    if group == SHAP_GROUP_UNREPORTED:
        return (
            f"For Trust {trust_name}, unreported-removal accounting features contributed positively to the selected model prediction."
        )
    return f"For Trust {trust_name}, {group} contributed positively to the selected model prediction."


def make_trust_interpretations(local_explanations: pd.DataFrame, trust_names: Sequence[str], horizon: int) -> pd.DataFrame:
    rows = []
    for trust_name in trust_names:
        trust_frame = local_explanations[
            local_explanations[COLUMNS.trust_name].astype(str).eq(str(trust_name))
        ]
        trust_code = ""
        if not trust_frame.empty and COLUMNS.trust_code in trust_frame.columns:
            trust_code = str(trust_frame[COLUMNS.trust_code].dropna().astype(str).iloc[0])
        rows.append(
            {
                COLUMNS.trust_code: trust_code,
                COLUMNS.trust_name: str(trust_name),
                COLUMNS.horizon: int(horizon),
                "interpretation": interpret_trust(str(trust_name), local_explanations, horizon=horizon),
            }
        )
    return pd.DataFrame(rows)


def write_methodology_note(path: str | Path, config: ShapLayerConfig, horizons: Sequence[int]) -> None:
    text = f"""# Custom TCN SHAP Explainability

This explainability layer imports `TCNQuantileRegressor` from `nhs_rtt_pipeline.modeling`, loads the canonical `models/tcn_state_dict.pt`, `models/model_config.json`, and `models/feature_metadata.json`, and reconstructs the same transformed encoder tensor columns used during training.

The explained model output is the inverse-transformed median forecast, P50. The configured forecast horizons are: {', '.join(str(int(h)) for h in horizons)} month(s) ahead.

The implementation uses `shap.KernelExplainer` with a wrapper around the custom PyTorch TCN. The wrapper receives interpretable encoder feature values, reconstructs the scaled tensor inputs using fitted preprocessing metadata, passes Trust and specialty identifier indices to the model embeddings, and returns P50 forecasts for the configured horizons.

Case selection uses model predictions from held-out test-period encoder windows. It does not use actual future target values to select cases for explanation. Missing or non-finite predictions raise an error and are recorded in the audit log rather than being replaced by ground-truth values.

Local explanation consistency is checked as:

```text
base value + sum(feature contributions) approximately equals model P50 output
```

The reported approximation error is the residual from that equality for each explained Trust-specialty context and horizon.

All wording in the generated interpretation tables is associational. SHAP values describe contributions to model predictions; they are not causal claims.

Configuration:

```json
{json.dumps(asdict(config), indent=2)}
```
"""
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(text, encoding="utf-8")
