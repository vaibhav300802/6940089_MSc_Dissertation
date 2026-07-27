# %% Cell 1
import importlib.util
import subprocess
import sys

PIP_PACKAGES = [
    "pandas>=2.0.0",
    "numpy>=1.23.0",
    "pyarrow>=10.0.0",
    "matplotlib>=3.7.0",
    "tqdm>=4.66.0",
    "torch>=2.1.0",
]

IMPORT_CHECKS = {
    "pandas": "pandas",
    "numpy": "numpy",
    "pyarrow": "pyarrow",
    "matplotlib": "matplotlib",
    "tqdm": "tqdm",
    "torch": "torch",
}

missing_packages = []
for package_name, module_name in IMPORT_CHECKS.items():
    if importlib.util.find_spec(module_name) is None:
        missing_packages.append(package_name)

if missing_packages:
    packages_to_install = [
        package_spec
        for package_spec in PIP_PACKAGES
        if package_spec.split(">=")[0] in missing_packages
    ]
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *packages_to_install])

# %% Cell 2
import copy
import json
import math
import os
import random
import sys
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)

PROJECT_ROOT_CANDIDATES = [
    Path.cwd(),
    Path.cwd().parent,
    Path("/content/nhs_rtt_msc_project"),
    Path("/content"),
]
for candidate_root in PROJECT_ROOT_CANDIDATES:
    if (candidate_root / "nhs_rtt_pipeline").exists():
        sys.path.insert(0, str(candidate_root))
        break
else:
    raise FileNotFoundError(
        "Could not find the shared nhs_rtt_pipeline package. "
        "Run this notebook from the project root or upload the full nhs_rtt_msc_project folder to Colab."
    )

from nhs_rtt_pipeline.config import COLUMNS, ROLLING_ORIGIN_PREDICTION_COLUMNS, ensure_directories, get_paths
from nhs_rtt_pipeline.modeling import QuantileLoss, TCNQuantileRegressor, build_tcn_model_config
from nhs_rtt_pipeline.preprocessing import (
    FLOW_LAGGED_FEATURES,
    FLOW_MISSINGNESS_FEATURES,
    LOG1P_NON_NEGATIVE_FEATURES,
    SIGNED_OPERATIONAL_FEATURES,
    assert_net_inflow_integrity,
    data_dictionary_frame,
    feature_group_for_column,
)
from nhs_rtt_pipeline.rolling_origin import (
    RollingOriginConfig,
    save_rolling_origin_outputs,
    select_rolling_origins,
    validate_rolling_origin_predictions,
)
from nhs_rtt_pipeline.reproducibility import set_global_seed
from nhs_rtt_pipeline.settings import load_pipeline_settings


try:
    PROJECT_SETTINGS = load_pipeline_settings(os.environ.get("NHS_RTT_PIPELINE_CONFIG"))
    FORECAST_SETTINGS = PROJECT_SETTINGS.forecasting
    MODEL_SETTINGS = PROJECT_SETTINGS.model
except FileNotFoundError:
    PROJECT_SETTINGS = None
    FORECAST_SETTINGS = {}
    MODEL_SETTINGS = {}

CONFIG = RollingOriginConfig(
    target_column=COLUMNS.incomplete_total,
    model_name="TCN",
    forecast_horizon=int(FORECAST_SETTINGS.get("rolling_origin_horizon", FORECAST_SETTINGS.get("forecast_horizon", 12))),
    origin_step_months=int(FORECAST_SETTINGS.get("rolling_origin_step_months", 6)),
    requested_origins=int(FORECAST_SETTINGS.get("rolling_origin_count", 3)),
    min_train_months=36,
    encoder_length=int(FORECAST_SETTINGS.get("encoder_length", 24)),
    internal_validation_months=int(FORECAST_SETTINGS.get("validation_months", 12)),
    batch_size=int(MODEL_SETTINGS.get("batch_size", 512)),
    max_epochs=min(int(MODEL_SETTINGS.get("max_epochs", 35)), 18),
    early_stopping_patience=min(int(MODEL_SETTINGS.get("early_stopping_patience", 7)), 5),
    learning_rate=float(MODEL_SETTINGS.get("learning_rate", 1.0e-3)),
    weight_decay=float(MODEL_SETTINGS.get("weight_decay", 1.0e-4)),
    hidden_channels=int(MODEL_SETTINGS.get("hidden_channels", 96)),
    tcn_levels=int(MODEL_SETTINGS.get("tcn_levels", 5)),
    kernel_size=int(MODEL_SETTINGS.get("kernel_size", 3)),
    dropout=float(MODEL_SETTINGS.get("dropout", 0.15)),
    embedding_dim=int(MODEL_SETTINGS.get("embedding_dim", 16)),
    gradient_clip_norm=float(MODEL_SETTINGS.get("gradient_clip_norm", 1.0)),
    random_seed=int(PROJECT_SETTINGS.random_seed) if PROJECT_SETTINGS is not None else 42,
    num_workers=int(MODEL_SETTINGS.get("num_workers", 2)),
)

PATHS = get_paths()
ensure_directories(PATHS)
PATHS.rolling_origin_dir.mkdir(parents=True, exist_ok=True)

MODEL_INPUT_FEATURES = [
    "waiting_list",
    "incomplete_total",
    "opening_waiting_list",
    "closing_waiting_list",
    "completed_admitted",
    "completed_non_admitted",
    "waiting_list_with_dta",
    "incomplete_decision_to_admit",
    "new_rtt_periods",
    "completed_total",
    "net_inflow",
    "reported_net_inflow",
    "unreported_removals",
    *FLOW_LAGGED_FEATURES,
    *FLOW_MISSINGNESS_FEATURES,
    "is_imputed_month",
    "missing_month",
    "waiting_list_imputed",
    "waiting_list_with_dta_imputed",
    "completed_admitted_imputed",
    "completed_non_admitted_imputed",
    "new_rtt_periods_imputed",
    "month_sin",
    "month_cos",
    "time_idx",
]

TARGET_AVAILABLE_COLUMNS = {
    COLUMNS.incomplete_total: "incomplete_total_source_available",
    COLUMNS.incomplete_decision_to_admit: "incomplete_decision_to_admit_source_available",
}

print(json.dumps(asdict(CONFIG), indent=2))
print(f"Project root: {PATHS.project_root}")
print(f"Processed data: {PATHS.clean_parquet}")
print(f"Rolling-origin outputs: {PATHS.rolling_origin_dir}")

# %% Cell 3
import numpy as np
import pandas as pd
import torch


def set_random_seed(seed: int) -> None:
    deterministic = bool(PROJECT_SETTINGS.deterministic_torch) if PROJECT_SETTINGS is not None else False
    set_global_seed(seed, deterministic_torch=deterministic)
    if torch.cuda.is_available() and not deterministic:
        torch.backends.cudnn.benchmark = True


def month_start(value: pd.Series) -> pd.Series:
    return pd.to_datetime(value, errors="coerce").dt.to_period("M").dt.to_timestamp()


def inverse_log1p(values: np.ndarray) -> np.ndarray:
    return np.clip(np.expm1(np.maximum(values, 0.0)), 0.0, None)


def validation_start_for_origin(origin: pd.Timestamp, config: RollingOriginConfig) -> pd.Timestamp:
    return pd.Timestamp(origin) - pd.DateOffset(months=int(config.internal_validation_months) - 1)


def scaler_fit_end_for_origin(origin: pd.Timestamp, config: RollingOriginConfig) -> pd.Timestamp:
    return validation_start_for_origin(origin, config) - pd.DateOffset(months=1)


def target_available_column_for(target_column: str) -> Optional[str]:
    return TARGET_AVAILABLE_COLUMNS.get(str(target_column))


set_random_seed(CONFIG.random_seed)

clean_rtt = pd.read_parquet(PATHS.clean_parquet)
clean_rtt["month"] = month_start(clean_rtt["month"])
if clean_rtt["month"].isna().any():
    raise ValueError("Processed RTT data contains invalid month values.")
if COLUMNS.series_id not in clean_rtt.columns:
    clean_rtt[COLUMNS.series_id] = clean_rtt[COLUMNS.trust_code].astype(str) + "__" + clean_rtt[COLUMNS.specialty_code].astype(str)
assert_net_inflow_integrity(clean_rtt)

final_observed_month = pd.to_datetime(clean_rtt["month"]).max()
first_observed_month = pd.to_datetime(clean_rtt["month"]).min()
origins = select_rolling_origins(clean_rtt, CONFIG)

print(f"Monthly coverage: {first_observed_month.date()} to {final_observed_month.date()}")
print(f"Selected forecast origins: {[origin.date().isoformat() for origin in origins]}")

# %% Cell 4
import numpy as np
import pandas as pd


def prepare_origin_frame(
    frame: pd.DataFrame,
    config: RollingOriginConfig,
    origin: pd.Timestamp,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    prepared = frame.copy().sort_values([COLUMNS.series_id, "time_idx"]).reset_index(drop=True)
    prepared["month"] = month_start(prepared["month"])
    target_column = str(config.target_column)
    target_available_column = target_available_column_for(target_column)
    if target_column not in prepared.columns:
        raise ValueError(f"Target column is missing from processed RTT data: {target_column}")
    for column in MODEL_INPUT_FEATURES:
        if column not in prepared.columns:
            raise ValueError(f"Processed RTT data is missing model input feature: {column}")

    validation_start = validation_start_for_origin(origin, config)
    scaler_fit_end = scaler_fit_end_for_origin(origin, config)
    scaler_fit_period = prepared["month"] <= scaler_fit_end
    if not scaler_fit_period.any():
        raise ValueError(f"No rows are available for scaler fitting before {scaler_fit_end.date()}.")

    transformed_feature_columns = []
    feature_stats: Dict[str, Dict[str, object]] = {}
    for column in MODEL_INPUT_FEATURES:
        values = pd.to_numeric(prepared[column], errors="coerce").astype(float)
        if column in LOG1P_NON_NEGATIVE_FEATURES:
            transformed = np.log1p(values.clip(lower=0.0))
            transform_name = "log1p_non_negative"
        elif column in SIGNED_OPERATIONAL_FEATURES:
            transformed = values
            transform_name = "identity_signed"
        else:
            transformed = values
            transform_name = "identity"
        train_values = transformed[scaler_fit_period].dropna()
        mean = float(train_values.mean()) if len(train_values) else 0.0
        std = float(train_values.std(ddof=0)) if len(train_values) else 1.0
        if not np.isfinite(std) or std < 1.0e-8:
            std = 1.0
        model_column = f"{column}_model"
        prepared[model_column] = ((transformed.fillna(mean) - mean) / std).astype("float32")
        transformed_feature_columns.append(model_column)
        feature_stats[column] = {
            "mean": mean,
            "std": std,
            "transform": transform_name,
            "feature_group": feature_group_for_column(column),
            "missing_value_imputation": "rolling_origin_training_mean_after_transform",
            "scaler_fit_end_month": scaler_fit_end.date().isoformat(),
            "missing_observations": int(values.isna().sum()),
        }

    target_values = pd.to_numeric(prepared[target_column], errors="coerce")
    if target_available_column is not None and target_available_column in prepared.columns:
        available = pd.to_numeric(prepared[target_available_column], errors="coerce").fillna(0).astype(int).eq(1)
        target_values = target_values.where(available)
    if target_values[scaler_fit_period].dropna().empty:
        raise ValueError(f"No non-missing training target values are available for {target_column}.")
    if (target_values.dropna() < 0).any():
        raise ValueError(f"Target column {target_column} contains negative values.")

    prepared["target_actual"] = target_values.astype("float32")
    prepared["target_log"] = np.log1p(target_values.clip(lower=0.0)).astype("float32")

    training_rows = prepared[prepared["month"] <= pd.Timestamp(origin)]
    trust_codes = sorted(training_rows[COLUMNS.trust_code].astype(str).unique())
    specialty_codes = sorted(training_rows[COLUMNS.specialty_code].astype(str).unique())
    trust_to_idx = {code: idx for idx, code in enumerate(trust_codes)}
    specialty_to_idx = {code: idx for idx, code in enumerate(specialty_codes)}
    prepared["trust_idx"] = prepared[COLUMNS.trust_code].astype(str).map(trust_to_idx)
    prepared["specialty_idx"] = prepared[COLUMNS.specialty_code].astype(str).map(specialty_to_idx)

    metadata = {
        "forecast_origin": pd.Timestamp(origin).date().isoformat(),
        "validation_start_month": validation_start.date().isoformat(),
        "training_end_month": scaler_fit_end.date().isoformat(),
        "scaler_fit_end_month": scaler_fit_end.date().isoformat(),
        "feature_columns": transformed_feature_columns,
        "raw_feature_columns": MODEL_INPUT_FEATURES,
        "feature_stats": feature_stats,
        "feature_groups": {column: feature_group_for_column(column) for column in MODEL_INPUT_FEATURES},
        "feature_group_names": sorted({feature_group_for_column(column) for column in MODEL_INPUT_FEATURES}),
        "missingness_feature_columns": [column for column in MODEL_INPUT_FEATURES if column.endswith("_missing")],
        "data_dictionary": data_dictionary_frame(MODEL_INPUT_FEATURES).to_dict(orient="records"),
        "target_column": target_column,
        "target_available_column": target_available_column,
        "trust_to_idx": trust_to_idx,
        "specialty_to_idx": specialty_to_idx,
        "quantiles": list(config.quantiles),
        "config": asdict(config),
    }
    return prepared, metadata


def build_origin_samples(
    prepared: pd.DataFrame,
    config: RollingOriginConfig,
    origin: pd.Timestamp,
) -> Dict[str, List[Dict[str, object]]]:
    validation_start = validation_start_for_origin(origin, config)
    training_end = scaler_fit_end_for_origin(origin, config)
    samples: Dict[str, List[Dict[str, object]]] = {"train": [], "val": [], "forecast": []}
    grouped = prepared.groupby(COLUMNS.series_id, sort=False, observed=True)

    for series_id, group in tqdm(grouped, total=grouped.ngroups, desc=f"Windows for {origin.date()}"):
        group = group.sort_values("time_idx").reset_index(drop=True)
        if len(group) < int(config.encoder_length) + int(config.forecast_horizon):
            continue
        if group["trust_idx"].isna().all() or group["specialty_idx"].isna().all():
            continue
        time_values = group["time_idx"].to_numpy(dtype=int)
        if not np.all(np.diff(time_values) == 1):
            continue
        month_values = pd.to_datetime(group["month"]).dt.to_period("M").dt.to_timestamp()
        origin_positions = np.flatnonzero(month_values.eq(pd.Timestamp(origin)).to_numpy())
        for encoder_end_pos in range(int(config.encoder_length) - 1, len(group) - int(config.forecast_horizon)):
            forecast_start_pos = encoder_end_pos + 1
            forecast_end_pos = encoder_end_pos + int(config.forecast_horizon)
            target_slice = group.loc[forecast_start_pos:forecast_end_pos, "target_log"]
            if target_slice.isna().any():
                continue
            forecast_start_month = pd.Timestamp(group.loc[forecast_start_pos, "month"])
            forecast_end_month = pd.Timestamp(group.loc[forecast_end_pos, "month"])
            sample = {
                COLUMNS.series_id: str(series_id),
                "encoder_end_pos": int(encoder_end_pos),
                "forecast_start_pos": int(forecast_start_pos),
                "forecast_end_pos": int(forecast_end_pos),
                "forecast_start_month": forecast_start_month,
                "forecast_end_month": forecast_end_month,
            }
            if forecast_end_month <= training_end:
                samples["train"].append(sample)
            elif forecast_start_month >= validation_start and forecast_end_month <= pd.Timestamp(origin):
                samples["val"].append(sample)

        if len(origin_positions) == 1:
            origin_pos = int(origin_positions[0])
            forecast_start_pos = origin_pos + 1
            forecast_end_pos = origin_pos + int(config.forecast_horizon)
            if forecast_end_pos < len(group):
                target_slice = group.loc[forecast_start_pos:forecast_end_pos, "target_actual"]
                if target_slice.notna().all():
                    samples["forecast"].append(
                        {
                            COLUMNS.series_id: str(series_id),
                            "encoder_end_pos": origin_pos,
                            "forecast_start_pos": forecast_start_pos,
                            "forecast_end_pos": forecast_end_pos,
                            "forecast_start_month": pd.Timestamp(group.loc[forecast_start_pos, "month"]),
                            "forecast_end_month": pd.Timestamp(group.loc[forecast_end_pos, "month"]),
                        }
                    )

    train_series = {sample[COLUMNS.series_id] for sample in samples["train"]}
    samples["val"] = [sample for sample in samples["val"] if sample[COLUMNS.series_id] in train_series]
    samples["forecast"] = [sample for sample in samples["forecast"] if sample[COLUMNS.series_id] in train_series]

    if not samples["train"]:
        raise RuntimeError(f"No training windows were generated for origin {origin.date()}.")
    if not samples["val"]:
        raise RuntimeError(f"No validation windows were generated for origin {origin.date()}.")
    if not samples["forecast"]:
        raise RuntimeError(f"No forecast windows with observed actuals were generated for origin {origin.date()}.")

    max_train_target = max(pd.Timestamp(sample["forecast_end_month"]) for sample in samples["train"])
    max_val_target = max(pd.Timestamp(sample["forecast_end_month"]) for sample in samples["val"])
    min_forecast_target = min(pd.Timestamp(sample["forecast_start_month"]) for sample in samples["forecast"])
    assert max_train_target <= training_end, "Training targets leaked into the validation or forecast period."
    assert max_val_target <= pd.Timestamp(origin), "Validation targets leaked beyond forecast origin."
    assert min_forecast_target > pd.Timestamp(origin), "Forecast targets must be after forecast origin."

    return samples

# %% Cell 5
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class RTTWindowDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        samples: List[Dict[str, object]],
        feature_columns: Sequence[str],
        encoder_length: int,
        prediction_length: int,
    ) -> None:
        self.samples = list(samples)
        self.feature_columns = list(feature_columns)
        self.encoder_length = int(encoder_length)
        self.prediction_length = int(prediction_length)
        self.series_frames: Dict[str, pd.DataFrame] = {}
        self.series_arrays: Dict[str, Dict[str, object]] = {}
        for series_id, group in frame.groupby(COLUMNS.series_id, sort=False, observed=True):
            ordered = group.sort_values("time_idx").reset_index(drop=True)
            if ordered["trust_idx"].isna().any() or ordered["specialty_idx"].isna().any():
                continue
            self.series_frames[str(series_id)] = ordered
            self.series_arrays[str(series_id)] = {
                "features": ordered[self.feature_columns].to_numpy(dtype=np.float32),
                "target_log": ordered["target_log"].to_numpy(dtype=np.float32),
                "target_actual": ordered["target_actual"].to_numpy(dtype=np.float32),
                "trust_idx": int(ordered["trust_idx"].iloc[0]),
                "specialty_idx": int(ordered["specialty_idx"].iloc[0]),
            }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[index]
        arrays = self.series_arrays[str(sample[COLUMNS.series_id])]
        encoder_end_pos = int(sample["encoder_end_pos"])
        encoder_start_pos = encoder_end_pos - self.encoder_length + 1
        forecast_start_pos = int(sample["forecast_start_pos"])
        forecast_end_pos = int(sample["forecast_end_pos"])
        x = arrays["features"][encoder_start_pos : encoder_end_pos + 1]
        y = arrays["target_log"][forecast_start_pos : forecast_end_pos + 1]
        return {
            "x": torch.tensor(x, dtype=torch.float32),
            "y": torch.tensor(y, dtype=torch.float32),
            "trust_idx": torch.tensor(arrays["trust_idx"], dtype=torch.long),
            "specialty_idx": torch.tensor(arrays["specialty_idx"], dtype=torch.long),
            "sample_idx": torch.tensor(index, dtype=torch.long),
        }


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=int(num_workers) > 0,
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    grad_clip_norm: float,
    use_amp: bool,
) -> float:
    is_training = optimizer is not None
    model.train(is_training)
    losses = []
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and is_training)
    for batch in tqdm(loader, leave=False, desc="train" if is_training else "eval"):
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        trust_idx = batch["trust_idx"].to(device, non_blocking=True)
        specialty_idx = batch["specialty_idx"].to(device, non_blocking=True)
        if is_training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_training):
            with torch.cuda.amp.autocast(enabled=use_amp):
                prediction = model(x, trust_idx, specialty_idx)
                loss = loss_fn(prediction, y)
            if is_training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
                scaler.step(optimizer)
                scaler.update()
        losses.append(float(loss.detach().cpu().item()))
    return float(np.mean(losses))


def train_origin_tcn(
    prepared: pd.DataFrame,
    metadata: Dict[str, object],
    samples: Dict[str, List[Dict[str, object]]],
    config: RollingOriginConfig,
    origin_index: int,
) -> Tuple[nn.Module, Dict[str, object], Dict[str, RTTWindowDataset]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets = {
        split: RTTWindowDataset(
            frame=prepared,
            samples=samples[split],
            feature_columns=metadata["feature_columns"],
            encoder_length=config.encoder_length,
            prediction_length=config.forecast_horizon,
        )
        for split in ["train", "val", "forecast"]
    }
    loaders = {
        "train": make_loader(datasets["train"], config.batch_size, True, config.num_workers),
        "val": make_loader(datasets["val"], config.batch_size, False, config.num_workers),
    }
    model = TCNQuantileRegressor(
        n_features=len(metadata["feature_columns"]),
        n_trusts=len(metadata["trust_to_idx"]),
        n_specialties=len(metadata["specialty_to_idx"]),
        prediction_length=config.forecast_horizon,
        quantiles=config.quantiles,
        hidden_channels=config.hidden_channels,
        tcn_levels=config.tcn_levels,
        kernel_size=config.kernel_size,
        dropout=config.dropout,
        embedding_dim=config.embedding_dim,
    ).to(device)
    loss_fn = QuantileLoss(config.quantiles).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    use_amp = torch.cuda.is_available()

    best_val_loss = math.inf
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
    history = []
    for epoch in range(1, int(config.max_epochs) + 1):
        train_loss = run_epoch(model, loaders["train"], loss_fn, optimizer, device, config.gradient_clip_norm, use_amp)
        val_loss = run_epoch(model, loaders["val"], loss_fn, None, device, config.gradient_clip_norm, use_amp)
        scheduler.step(val_loss)
        current_lr = float(optimizer.param_groups[0]["lr"])
        history.append(
            {
                "origin_index": int(origin_index),
                "epoch": int(epoch),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": current_lr,
            }
        )
        print(
            f"Origin {origin_index} | epoch {epoch:03d} | "
            f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | lr={current_lr:.2e}"
        )
        if val_loss < best_val_loss - 1.0e-5:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= int(config.early_stopping_patience):
                print(f"Early stopping for origin {origin_index} at epoch {epoch}. Best val_loss={best_val_loss:.6f}")
                break

    model.load_state_dict(best_state)
    metadata = dict(metadata)
    metadata["best_val_loss"] = float(best_val_loss)
    metadata["training_history"] = history
    metadata["model_config"] = build_tcn_model_config(
        n_features=len(metadata["feature_columns"]),
        n_trusts=len(metadata["trust_to_idx"]),
        n_specialties=len(metadata["specialty_to_idx"]),
        prediction_length=config.forecast_horizon,
        quantiles=config.quantiles,
        hidden_channels=config.hidden_channels,
        tcn_levels=config.tcn_levels,
        kernel_size=config.kernel_size,
        dropout=config.dropout,
        embedding_dim=config.embedding_dim,
    )
    return model, metadata, datasets

# %% Cell 6
import numpy as np
import pandas as pd
import torch


def predict_origin_forecast(
    model: nn.Module,
    dataset: RTTWindowDataset,
    config: RollingOriginConfig,
    metadata: Dict[str, object],
    origin: pd.Timestamp,
    origin_index: int,
) -> pd.DataFrame:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = make_loader(dataset, config.batch_size, False, config.num_workers)
    model.eval()
    model.to(device)
    prediction_batches = []
    sample_index_batches = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Forecast origin {origin.date()}"):
            x = batch["x"].to(device, non_blocking=True)
            trust_idx = batch["trust_idx"].to(device, non_blocking=True)
            specialty_idx = batch["specialty_idx"].to(device, non_blocking=True)
            prediction_log = model(x, trust_idx, specialty_idx).detach().cpu().numpy()
            prediction_batches.append(prediction_log)
            sample_index_batches.append(batch["sample_idx"].detach().cpu().numpy())

    raw_prediction_log = np.concatenate(prediction_batches, axis=0)
    sample_indices = np.concatenate(sample_index_batches, axis=0)
    raw_crossing = (raw_prediction_log[:, :, 0] > raw_prediction_log[:, :, 1]) | (
        raw_prediction_log[:, :, 1] > raw_prediction_log[:, :, 2]
    )
    raw_prediction = inverse_log1p(raw_prediction_log)
    corrected_prediction = np.sort(raw_prediction, axis=-1)
    records: List[Dict[str, object]] = []
    training_end_month = pd.Timestamp(metadata["training_end_month"])
    scaler_fit_end_month = pd.Timestamp(metadata["scaler_fit_end_month"])

    for batch_pos, sample_idx in enumerate(sample_indices):
        sample = dataset.samples[int(sample_idx)]
        series_id = str(sample[COLUMNS.series_id])
        series_frame = dataset.series_frames[series_id]
        forecast_start_pos = int(sample["forecast_start_pos"])
        for horizon_idx in range(int(config.forecast_horizon)):
            row = series_frame.iloc[forecast_start_pos + horizon_idx]
            raw_values = raw_prediction[batch_pos, horizon_idx, :]
            corrected_values = corrected_prediction[batch_pos, horizon_idx, :]
            forecast_month = pd.Timestamp(row["month"])
            expected_horizon = (
                (forecast_month.year - pd.Timestamp(origin).year) * 12
                + (forecast_month.month - pd.Timestamp(origin).month)
            )
            record = {
                "origin_index": int(origin_index),
                "model_name": str(config.model_name),
                COLUMNS.series_id: series_id,
                COLUMNS.trust_code: str(row[COLUMNS.trust_code]),
                COLUMNS.trust_name: str(row[COLUMNS.trust_name]),
                "trust": str(row[COLUMNS.trust_name]),
                COLUMNS.specialty_code: str(row[COLUMNS.specialty_code]),
                COLUMNS.specialty_name: str(row[COLUMNS.specialty_name]),
                "specialty": str(row[COLUMNS.specialty_name]),
                COLUMNS.forecast_origin: pd.Timestamp(origin),
                COLUMNS.forecast_month: forecast_month,
                COLUMNS.horizon: int(expected_horizon),
                "target_column": str(config.target_column),
                "training_end_month": training_end_month,
                "scaler_fit_end_month": scaler_fit_end_month,
                "p10_raw": float(raw_values[0]),
                "p50_raw": float(raw_values[1]),
                "p90_raw": float(raw_values[2]),
                "quantile_crossing_raw": bool(raw_crossing[batch_pos, horizon_idx]),
                COLUMNS.p10: float(corrected_values[0]),
                COLUMNS.p50: float(corrected_values[1]),
                COLUMNS.p90: float(corrected_values[2]),
                COLUMNS.actual: float(row["target_actual"]),
            }
            records.append(record)

    frame = pd.DataFrame(records)
    if frame.empty:
        raise RuntimeError(f"No rolling-origin forecast rows were produced for origin {origin.date()}.")
    frame = frame[ROLLING_ORIGIN_PREDICTION_COLUMNS]
    return validate_rolling_origin_predictions(frame)

# %% Cell 7
import pandas as pd


all_prediction_frames = []
origin_metadata_records = []
for origin_index, origin in enumerate(origins, start=1):
    print(f"Rolling origin {origin_index}/{len(origins)}: {origin.date()}")
    fold_seed = int(CONFIG.random_seed) + origin_index
    set_random_seed(fold_seed)
    prepared_fold, fold_metadata = prepare_origin_frame(clean_rtt, CONFIG, origin)
    samples = build_origin_samples(prepared_fold, CONFIG, origin)
    print(
        {
            "origin": origin.date().isoformat(),
            "train_windows": len(samples["train"]),
            "validation_windows": len(samples["val"]),
            "forecast_windows": len(samples["forecast"]),
            "training_end_month": fold_metadata["training_end_month"],
            "scaler_fit_end_month": fold_metadata["scaler_fit_end_month"],
        }
    )
    fold_model, fold_metadata, fold_datasets = train_origin_tcn(
        prepared=prepared_fold,
        metadata=fold_metadata,
        samples=samples,
        config=CONFIG,
        origin_index=origin_index,
    )
    fold_predictions = predict_origin_forecast(
        model=fold_model,
        dataset=fold_datasets["forecast"],
        config=CONFIG,
        metadata=fold_metadata,
        origin=origin,
        origin_index=origin_index,
    )
    all_prediction_frames.append(fold_predictions)
    origin_metadata_records.append(
        {
            "origin_index": origin_index,
            "forecast_origin": origin.date().isoformat(),
            "training_end_month": fold_metadata["training_end_month"],
            "scaler_fit_end_month": fold_metadata["scaler_fit_end_month"],
            "train_windows": len(samples["train"]),
            "validation_windows": len(samples["val"]),
            "forecast_windows": len(samples["forecast"]),
            "best_val_loss": fold_metadata["best_val_loss"],
        }
    )

rolling_predictions = pd.concat(all_prediction_frames, ignore_index=True)
rolling_predictions = validate_rolling_origin_predictions(rolling_predictions)

assert rolling_predictions[COLUMNS.actual].notna().all(), "Rolling-origin predictions must contain actual values."
assert (rolling_predictions[COLUMNS.forecast_month] > rolling_predictions[COLUMNS.forecast_origin]).all(), (
    "Every rolling-origin forecast month must be after its forecast origin."
)
assert (rolling_predictions["training_end_month"] < rolling_predictions[COLUMNS.forecast_origin]).all(), (
    "Training target periods must end before each forecast origin."
)
assert (rolling_predictions["scaler_fit_end_month"] < rolling_predictions[COLUMNS.forecast_origin]).all(), (
    "Feature scalers must be fit before each forecast origin."
)

origin_metadata = pd.DataFrame(origin_metadata_records)
origin_metadata.to_csv(PATHS.rolling_origin_dir / "rolling_origin_fold_metadata.csv", index=False)
print(f"Rolling-origin rows: {len(rolling_predictions)}")

# %% Cell 8
import pandas as pd


saved = save_rolling_origin_outputs(
    predictions=rolling_predictions,
    config=CONFIG,
    origins=origins,
    paths={
        "predictions": PATHS.rolling_origin_predictions,
        "overall": PATHS.rolling_origin_metrics,
        "by_origin": PATHS.rolling_origin_metrics_by_origin,
        "by_horizon": PATHS.rolling_origin_metrics_by_horizon,
        "by_trust": PATHS.rolling_origin_metrics_by_trust,
        "by_specialty": PATHS.rolling_origin_metrics_by_specialty,
        "by_waiting_size": PATHS.rolling_origin_metrics_by_waiting_size,
        "crossing": PATHS.rolling_origin_quantile_crossing,
        "reliability": PATHS.rolling_origin_reliability,
        "summary": PATHS.rolling_origin_summary,
        "calibration_png": PATHS.rolling_origin_calibration_png,
        "width_coverage_png": PATHS.rolling_origin_interval_width_coverage_png,
    },
)

print(f"Saved predictions: {PATHS.rolling_origin_predictions}")
print(f"Saved overall metrics: {PATHS.rolling_origin_metrics}")
print(f"Saved metrics by origin: {PATHS.rolling_origin_metrics_by_origin}")
print(f"Saved metrics by horizon: {PATHS.rolling_origin_metrics_by_horizon}")
print(f"Saved metrics by Trust: {PATHS.rolling_origin_metrics_by_trust}")
print(f"Saved metrics by specialty: {PATHS.rolling_origin_metrics_by_specialty}")
print(f"Saved reliability summary: {PATHS.rolling_origin_reliability}")
print(f"Saved calibration plot: {PATHS.rolling_origin_calibration_png}")
print(f"Saved interval width plot: {PATHS.rolling_origin_interval_width_coverage_png}")
print(f"Saved report: {PATHS.rolling_origin_summary}")
print(saved["overall"].to_string(index=False))
print(saved["reliability"].to_string(index=False))
