# %% Cell 1
import importlib.util
import subprocess
import sys

PIP_PACKAGES = [
    "pandas>=2.0.0",
    "numpy>=1.23.0",
    "pyarrow>=10.0.0",
    "matplotlib>=3.7.0",
    "scikit-learn>=1.2.0",
    "torch>=2.1.0",
    "tqdm>=4.66.0",
]

IMPORT_CHECKS = {
    "pandas": "pandas",
    "numpy": "numpy",
    "pyarrow": "pyarrow",
    "matplotlib": "matplotlib",
    "scikit-learn": "sklearn",
    "torch": "torch",
    "tqdm": "tqdm",
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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    from IPython.display import display
except Exception:
    def display(obj: object) -> None:
        if isinstance(obj, pd.DataFrame):
            print(obj.head(25).to_string())
        else:
            print(obj)

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
        "Could not find nhs_rtt_pipeline. Run this notebook from the project root "
        "or upload the complete nhs_rtt_msc_project folder to Colab."
    )

from nhs_rtt_pipeline.config import COLUMNS, ensure_directories, get_paths
from nhs_rtt_pipeline.covid_shock import compute_covid_shock_split_boundaries
from nhs_rtt_pipeline.modeling import QuantileLoss, TCNQuantileRegressor, build_tcn_model_config
from nhs_rtt_pipeline.preprocessing import LOG1P_NON_NEGATIVE_FEATURES, SIGNED_OPERATIONAL_FEATURES
from nhs_rtt_pipeline.reproducibility import set_global_seed
from nhs_rtt_pipeline.settings import load_pipeline_settings


@dataclass(frozen=True)
class CovidShockExperimentConfig:
    train_end: str = "2020-02-01"
    covid_test_start: str = "2020-03-01"
    covid_test_end: str = "2021-09-01"
    recovery_start: str = "2021-10-01"
    validation_months: int = 12
    encoder_length: int = 24
    prediction_length: int = 1
    min_series_length: int = 36
    batch_size: int = 1024
    max_epochs_validation_model: int = 18
    max_epochs_final_model: int = 22
    early_stopping_patience: int = 5
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    hidden_channels: int = 64
    tcn_levels: int = 4
    kernel_size: int = 3
    dropout: float = 0.15
    embedding_dim: int = 12
    gradient_clip_norm: float = 1.0
    num_workers: int = 2
    random_seed: int = 42
    random_forest_estimators: int = 250
    random_forest_min_samples_leaf: int = 5
    plot_top_n: int = 20


try:
    PROJECT_SETTINGS = load_pipeline_settings(os.environ.get("NHS_RTT_PIPELINE_CONFIG"))
    MODEL_SETTINGS = PROJECT_SETTINGS.model
    FORECAST_SETTINGS = PROJECT_SETTINGS.forecasting
    CONFIG = CovidShockExperimentConfig(
        validation_months=int(FORECAST_SETTINGS.get("validation_months", 12)),
        encoder_length=int(FORECAST_SETTINGS.get("encoder_length", 24)),
        batch_size=int(MODEL_SETTINGS.get("batch_size", 1024)),
        learning_rate=float(MODEL_SETTINGS.get("learning_rate", 1.0e-3)),
        weight_decay=float(MODEL_SETTINGS.get("weight_decay", 1.0e-4)),
        hidden_channels=min(int(MODEL_SETTINGS.get("hidden_channels", 96)), 64),
        tcn_levels=min(int(MODEL_SETTINGS.get("tcn_levels", 5)), 4),
        kernel_size=int(MODEL_SETTINGS.get("kernel_size", 3)),
        dropout=float(MODEL_SETTINGS.get("dropout", 0.15)),
        embedding_dim=min(int(MODEL_SETTINGS.get("embedding_dim", 16)), 12),
        gradient_clip_norm=float(MODEL_SETTINGS.get("gradient_clip_norm", 1.0)),
        num_workers=int(MODEL_SETTINGS.get("num_workers", 2)),
        random_seed=int(PROJECT_SETTINGS.random_seed),
    )
except FileNotFoundError:
    PROJECT_SETTINGS = None
    CONFIG = CovidShockExperimentConfig()
QUANTILES = (0.1, 0.5, 0.9)
PATHS = get_paths()
ensure_directories(PATHS)
PATHS.covid_stress_test_dir.mkdir(parents=True, exist_ok=True)


def set_random_seed(seed: int) -> None:
    deterministic = bool(PROJECT_SETTINGS.deterministic_torch) if PROJECT_SETTINGS is not None else False
    set_global_seed(seed, deterministic_torch=deterministic)
    if torch.cuda.is_available() and not deterministic:
        torch.backends.cudnn.benchmark = True


set_random_seed(CONFIG.random_seed)

print(json.dumps(asdict(CONFIG), indent=2))
print(f"Project root: {PATHS.project_root}")
print(f"Processed data: {PATHS.clean_parquet}")
print(f"COVID experiment outputs: {PATHS.covid_stress_test_dir}")
print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

# %% Cell 3
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


BASE_FEATURE_CANDIDATES = [
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
    "new_rtt_periods_missing",
    "completed_admitted_missing",
    "completed_non_admitted_missing",
    "completed_total_missing",
    "opening_waiting_list_missing",
    "closing_waiting_list_missing",
    "reported_net_inflow_missing",
    "unreported_removals_missing",
    "flow_components_missing",
    "is_imputed_month",
    "month_sin",
    "month_cos",
    "time_idx",
]


def month_start(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce").dt.to_period("M").dt.to_timestamp()


def load_processed_rtt(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing processed RTT data: {path}")
    frame = pd.read_parquet(path)
    required = ["month", COLUMNS.trust_code, COLUMNS.trust_name, COLUMNS.specialty_code, COLUMNS.specialty_name]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Processed RTT data is missing required columns: {missing}")
    frame = frame.copy()
    frame["month"] = month_start(frame["month"])
    if frame["month"].isna().any():
        raise ValueError("Processed RTT data contains invalid month values.")
    if COLUMNS.series_id not in frame.columns:
        frame[COLUMNS.series_id] = (
            frame[COLUMNS.trust_code].astype(str) + "__" + frame[COLUMNS.specialty_code].astype(str)
        )
    if "calendar_month" not in frame.columns:
        frame["calendar_month"] = frame["month"].dt.month.astype(int)
    if "month_sin" not in frame.columns:
        frame["month_sin"] = np.sin(2.0 * np.pi * frame["calendar_month"].astype(float) / 12.0)
    if "month_cos" not in frame.columns:
        frame["month_cos"] = np.cos(2.0 * np.pi * frame["calendar_month"].astype(float) / 12.0)
    if "time_idx" not in frame.columns:
        month_index = {month: idx for idx, month in enumerate(sorted(frame["month"].dropna().unique()))}
        frame["time_idx"] = frame["month"].map(month_index).astype(int)
    return frame.sort_values([COLUMNS.series_id, "month"]).reset_index(drop=True)


def choose_target_column(frame: pd.DataFrame) -> str:
    for candidate in [COLUMNS.incomplete_total, "waiting_list"]:
        if candidate in frame.columns and pd.to_numeric(frame[candidate], errors="coerce").notna().any():
            return candidate
    raise ValueError("No usable incomplete RTT target was found. Expected incomplete_total or waiting_list.")


def select_feature_columns(frame: pd.DataFrame, target_column: str) -> List[str]:
    columns = []
    for column in [target_column, *BASE_FEATURE_CANDIDATES]:
        if column in frame.columns and column not in columns:
            columns.append(column)
    if len(columns) < 4:
        raise ValueError(f"Too few usable model features were found: {columns}")
    return columns


def summarize_month_coverage(frame: pd.DataFrame) -> Dict[str, object]:
    months = pd.Series(sorted(frame["month"].dropna().unique()))
    month_periods = months.dt.to_period("M")
    expected = pd.period_range(month_periods.min(), month_periods.max(), freq="M")
    missing_months = sorted(set(expected.astype(str)) - set(month_periods.astype(str)))
    return {
        "minimum_month": month_periods.min().to_timestamp().date().isoformat(),
        "maximum_month": month_periods.max().to_timestamp().date().isoformat(),
        "observed_month_count": int(month_periods.nunique()),
        "missing_calendar_months_in_project_range": missing_months,
    }


raw_rtt = load_processed_rtt(PATHS.clean_parquet)
TARGET_COLUMN = choose_target_column(raw_rtt)
FEATURE_COLUMNS = select_feature_columns(raw_rtt, TARGET_COLUMN)

date_coverage = summarize_month_coverage(raw_rtt)
_split_boundaries = compute_covid_shock_split_boundaries(
    train_end=CONFIG.train_end,
    covid_test_start=CONFIG.covid_test_start,
    covid_test_end=CONFIG.covid_test_end,
    recovery_start=CONFIG.recovery_start,
    validation_months=CONFIG.validation_months,
    date_coverage=date_coverage,
)
train_end = _split_boundaries.train_end
covid_start = _split_boundaries.covid_start
covid_end = _split_boundaries.covid_end
recovery_start = _split_boundaries.recovery_start
validation_start = _split_boundaries.validation_start
core_train_end = _split_boundaries.core_train_end

split_summary = {
    "target_column": TARGET_COLUMN,
    "feature_columns": FEATURE_COLUMNS,
    "date_coverage": date_coverage,
    "core_training_period": {
        "end_month": core_train_end.date().isoformat(),
        "purpose": "Fit validation model and all preprocessing used for pre-COVID validation.",
    },
    "pre_covid_validation_period": {
        "start_month": validation_start.date().isoformat(),
        "end_month": train_end.date().isoformat(),
    },
    "full_pre_covid_training_period": {
        "end_month": train_end.date().isoformat(),
        "purpose": "Fit final shock experiment model and all preprocessing used for COVID/recovery forecasts.",
    },
    "covid_shock_period": {
        "start_month": covid_start.date().isoformat(),
        "end_month": covid_end.date().isoformat(),
    },
    "recovery_period": {
        "start_month": recovery_start.date().isoformat(),
        "available_if_data_extend_to_or_after": recovery_start.date().isoformat(),
    },
}

with open(PATHS.covid_split_summary, "w", encoding="utf-8") as handle:
    json.dump(split_summary, handle, indent=2)

print(json.dumps(split_summary, indent=2))

# %% Cell 4
import math
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


def prepare_experiment_frame(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    target_column: str,
    scaler_fit_end_month: pd.Timestamp,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    prepared = frame.copy().sort_values([COLUMNS.series_id, "month"]).reset_index(drop=True)
    prepared["target_actual"] = pd.to_numeric(prepared[target_column], errors="coerce").astype(float)
    prepared["target_log"] = np.log1p(prepared["target_actual"].clip(lower=0.0)).astype("float32")
    fit_mask = prepared["month"] <= pd.Timestamp(scaler_fit_end_month)
    if not fit_mask.any():
        raise ValueError(f"No rows are available for preprocessing fit through {scaler_fit_end_month.date()}.")

    log1p_features = set(LOG1P_NON_NEGATIVE_FEATURES) | {
        "waiting_list",
        COLUMNS.incomplete_total,
        COLUMNS.incomplete_decision_to_admit,
        "opening_waiting_list",
        "closing_waiting_list",
    }
    signed_features = set(SIGNED_OPERATIONAL_FEATURES)
    transformed_columns = []
    feature_stats: Dict[str, Dict[str, object]] = {}

    for column in feature_columns:
        values = pd.to_numeric(prepared[column], errors="coerce").astype(float)
        if column in log1p_features:
            transformed = np.log1p(values.clip(lower=0.0))
            transform_name = "log1p_non_negative"
        elif column in signed_features:
            transformed = values
            transform_name = "identity_signed"
        else:
            transformed = values
            transform_name = "identity"

        fit_values = transformed[fit_mask].dropna()
        mean = float(fit_values.mean()) if len(fit_values) else 0.0
        std = float(fit_values.std(ddof=0)) if len(fit_values) else 1.0
        if not np.isfinite(std) or std < 1.0e-8:
            std = 1.0
        model_column = f"{column}_covid_model"
        prepared[model_column] = ((transformed.fillna(mean) - mean) / std).astype("float32")
        transformed_columns.append(model_column)
        feature_stats[column] = {
            "mean": mean,
            "std": std,
            "transform": transform_name,
            "fit_end_month": pd.Timestamp(scaler_fit_end_month).date().isoformat(),
            "missing_observations": int(values.isna().sum()),
        }

    trust_codes = sorted(prepared[COLUMNS.trust_code].astype(str).unique())
    specialty_codes = sorted(prepared[COLUMNS.specialty_code].astype(str).unique())
    trust_to_idx = {code: idx for idx, code in enumerate(trust_codes)}
    specialty_to_idx = {code: idx for idx, code in enumerate(specialty_codes)}
    prepared["trust_idx"] = prepared[COLUMNS.trust_code].astype(str).map(trust_to_idx).astype("int64")
    prepared["specialty_idx"] = prepared[COLUMNS.specialty_code].astype(str).map(specialty_to_idx).astype("int64")

    metadata = {
        "feature_columns": transformed_columns,
        "raw_feature_columns": list(feature_columns),
        "feature_stats": feature_stats,
        "target_column": target_column,
        "scaler_fit_end_month": pd.Timestamp(scaler_fit_end_month).date().isoformat(),
        "trust_to_idx": trust_to_idx,
        "specialty_to_idx": specialty_to_idx,
        "quantiles": list(QUANTILES),
    }
    max_fit_month = prepared.loc[fit_mask, "month"].max()
    assert max_fit_month <= pd.Timestamp(scaler_fit_end_month)
    return prepared, metadata


validation_prepared, validation_metadata = prepare_experiment_frame(
    raw_rtt,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    scaler_fit_end_month=core_train_end,
)
final_prepared, final_metadata = prepare_experiment_frame(
    raw_rtt,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    scaler_fit_end_month=train_end,
)

assert pd.Timestamp(validation_metadata["scaler_fit_end_month"]) < covid_start
assert pd.Timestamp(final_metadata["scaler_fit_end_month"]) < covid_start
print("Preprocessing/scaling fit checks passed.")

# %% Cell 5
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def filter_sample_series(samples: List[Dict[str, object]], allowed_series_ids: set[str]) -> List[Dict[str, object]]:
    return [sample for sample in samples if str(sample["series_id"]) in allowed_series_ids]


def build_tcn_samples(
    frame: pd.DataFrame,
    start_month: Optional[pd.Timestamp],
    end_month: Optional[pd.Timestamp],
    encoder_length: int,
    prediction_length: int,
) -> List[Dict[str, object]]:
    samples: List[Dict[str, object]] = []
    grouped = frame.groupby(COLUMNS.series_id, sort=False, observed=True)
    for series_id, group in grouped:
        group = group.sort_values("month").reset_index(drop=True)
        if len(group) < encoder_length + prediction_length:
            continue
        month_values = pd.to_datetime(group["month"]).reset_index(drop=True)
        for encoder_end_pos in range(encoder_length - 1, len(group) - prediction_length):
            forecast_start_pos = encoder_end_pos + 1
            forecast_end_pos = encoder_end_pos + prediction_length
            forecast_months = month_values.iloc[forecast_start_pos : forecast_end_pos + 1]
            target_month = pd.Timestamp(forecast_months.iloc[-1])
            forecast_origin = pd.Timestamp(month_values.iloc[encoder_end_pos])
            if start_month is not None and target_month < pd.Timestamp(start_month):
                continue
            if end_month is not None and target_month > pd.Timestamp(end_month):
                continue
            if group.loc[forecast_start_pos:forecast_end_pos, "target_log"].isna().any():
                continue
            if group.loc[encoder_end_pos - encoder_length + 1 : encoder_end_pos, "target_actual"].isna().any():
                continue
            samples.append(
                {
                    "series_id": str(series_id),
                    "encoder_end_pos": int(encoder_end_pos),
                    "forecast_start_pos": int(forecast_start_pos),
                    "forecast_end_pos": int(forecast_end_pos),
                    "forecast_origin": forecast_origin,
                    "forecast_month": target_month,
                }
            )
    return samples


class CovidTCNDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        samples: Sequence[Dict[str, object]],
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
            group = group.sort_values("month").reset_index(drop=True)
            self.series_frames[str(series_id)] = group
            self.series_arrays[str(series_id)] = {
                "features": group[self.feature_columns].to_numpy(dtype=np.float32),
                "target_log": group["target_log"].to_numpy(dtype=np.float32),
                "trust_idx": int(group["trust_idx"].iloc[0]),
                "specialty_idx": int(group["specialty_idx"].iloc[0]),
            }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[index]
        arrays = self.series_arrays[str(sample["series_id"])]
        encoder_end_pos = int(sample["encoder_end_pos"])
        encoder_start_pos = encoder_end_pos - self.encoder_length + 1
        forecast_start_pos = int(sample["forecast_start_pos"])
        forecast_end_pos = int(sample["forecast_end_pos"])
        return {
            "x": torch.tensor(arrays["features"][encoder_start_pos : encoder_end_pos + 1], dtype=torch.float32),
            "y": torch.tensor(arrays["target_log"][forecast_start_pos : forecast_end_pos + 1], dtype=torch.float32),
            "trust_idx": torch.tensor(arrays["trust_idx"], dtype=torch.long),
            "specialty_idx": torch.tensor(arrays["specialty_idx"], dtype=torch.long),
            "sample_idx": torch.tensor(index, dtype=torch.long),
        }


validation_train_samples = build_tcn_samples(
    validation_prepared,
    start_month=None,
    end_month=core_train_end,
    encoder_length=CONFIG.encoder_length,
    prediction_length=CONFIG.prediction_length,
)
validation_eval_samples = build_tcn_samples(
    validation_prepared,
    start_month=validation_start,
    end_month=train_end,
    encoder_length=CONFIG.encoder_length,
    prediction_length=CONFIG.prediction_length,
)
final_train_samples = build_tcn_samples(
    final_prepared,
    start_month=None,
    end_month=train_end,
    encoder_length=CONFIG.encoder_length,
    prediction_length=CONFIG.prediction_length,
)
shock_samples = build_tcn_samples(
    final_prepared,
    start_month=covid_start,
    end_month=min(covid_end, pd.to_datetime(final_prepared["month"]).max()),
    encoder_length=CONFIG.encoder_length,
    prediction_length=CONFIG.prediction_length,
)
recovery_samples = build_tcn_samples(
    final_prepared,
    start_month=recovery_start,
    end_month=pd.to_datetime(final_prepared["month"]).max(),
    encoder_length=CONFIG.encoder_length,
    prediction_length=CONFIG.prediction_length,
)

validation_train_series = {sample["series_id"] for sample in validation_train_samples}
final_train_series = {sample["series_id"] for sample in final_train_samples}
validation_eval_samples = filter_sample_series(validation_eval_samples, validation_train_series)
shock_samples = filter_sample_series(shock_samples, final_train_series)
recovery_samples = filter_sample_series(recovery_samples, final_train_series)

if not validation_train_samples:
    raise RuntimeError("No pre-COVID core training samples were generated.")
if not validation_eval_samples:
    raise RuntimeError("No pre-COVID validation samples were generated.")
if not final_train_samples:
    raise RuntimeError("No full pre-COVID training samples were generated.")
if not shock_samples:
    raise RuntimeError("No COVID shock-period samples were generated.")

assert max(sample["forecast_month"] for sample in validation_train_samples) <= core_train_end
assert min(sample["forecast_month"] for sample in validation_eval_samples) >= validation_start
assert max(sample["forecast_month"] for sample in validation_eval_samples) <= train_end
assert max(sample["forecast_month"] for sample in final_train_samples) <= train_end
assert min(sample["forecast_month"] for sample in shock_samples) >= covid_start
assert max(sample["forecast_month"] for sample in shock_samples) <= covid_end
assert max(sample["forecast_month"] for sample in final_train_samples) < min(
    sample["forecast_month"] for sample in shock_samples
)

sample_summary = {
    "validation_training_samples": len(validation_train_samples),
    "pre_covid_validation_samples": len(validation_eval_samples),
    "final_pre_covid_training_samples": len(final_train_samples),
    "covid_shock_samples": len(shock_samples),
    "recovery_samples": len(recovery_samples),
    "trust_specialty_series_forecast": len(final_train_series),
}
split_summary.update(sample_summary)
with open(PATHS.covid_split_summary, "w", encoding="utf-8") as handle:
    json.dump(split_summary, handle, indent=2)

print(json.dumps(sample_summary, indent=2))

# %% Cell 6
import copy
import math
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=CONFIG.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=CONFIG.num_workers > 0,
    )


def run_tcn_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
) -> float:
    is_training = optimizer is not None
    model.train(is_training)
    losses = []
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available() and is_training)
    for batch in tqdm(loader, leave=False, desc="train" if is_training else "eval"):
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        trust_idx = batch["trust_idx"].to(device, non_blocking=True)
        specialty_idx = batch["specialty_idx"].to(device, non_blocking=True)
        if is_training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_training):
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                prediction = model(x, trust_idx, specialty_idx)
                loss = loss_fn(prediction, y)
            if is_training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), CONFIG.gradient_clip_norm)
                scaler.step(optimizer)
                scaler.update()
        losses.append(float(loss.detach().cpu().item()))
    return float(np.mean(losses))


def build_tcn(metadata: Dict[str, object]) -> TCNQuantileRegressor:
    return TCNQuantileRegressor(
        n_features=len(metadata["feature_columns"]),
        n_trusts=len(metadata["trust_to_idx"]),
        n_specialties=len(metadata["specialty_to_idx"]),
        prediction_length=CONFIG.prediction_length,
        quantiles=QUANTILES,
        hidden_channels=CONFIG.hidden_channels,
        tcn_levels=CONFIG.tcn_levels,
        kernel_size=CONFIG.kernel_size,
        dropout=CONFIG.dropout,
        embedding_dim=CONFIG.embedding_dim,
    )


def train_tcn_for_experiment(
    train_dataset: CovidTCNDataset,
    metadata: Dict[str, object],
    max_epochs: int,
    validation_dataset: Optional[CovidTCNDataset] = None,
) -> Tuple[TCNQuantileRegressor, pd.DataFrame]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_tcn(metadata).to(device)
    loss_fn = QuantileLoss(QUANTILES).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG.learning_rate, weight_decay=CONFIG.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    train_loader = make_loader(train_dataset, CONFIG.batch_size, shuffle=True)
    validation_loader = make_loader(validation_dataset, CONFIG.batch_size, shuffle=False) if validation_dataset else None

    best_monitor = math.inf
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, int(max_epochs) + 1):
        train_loss = run_tcn_epoch(model, train_loader, loss_fn, optimizer, device)
        if validation_loader is not None:
            monitor_loss = run_tcn_epoch(model, validation_loader, loss_fn, None, device)
            scheduler.step(monitor_loss)
        else:
            monitor_loss = train_loss
            scheduler.step(monitor_loss)
        current_lr = float(optimizer.param_groups[0]["lr"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "monitor_loss": monitor_loss,
                "learning_rate": current_lr,
                "used_validation_monitor": validation_loader is not None,
            }
        )
        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | "
            f"monitor_loss={monitor_loss:.6f} | lr={current_lr:.2e}"
        )
        if monitor_loss < best_monitor - 1.0e-5:
            best_monitor = monitor_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if validation_loader is not None and epochs_without_improvement >= CONFIG.early_stopping_patience:
                print(f"Early stopping at epoch {epoch}. Best monitor loss={best_monitor:.6f}")
                break

    model.load_state_dict(best_state)
    model.eval()
    return model, pd.DataFrame(history)


validation_train_dataset = CovidTCNDataset(
    validation_prepared,
    validation_train_samples,
    validation_metadata["feature_columns"],
    CONFIG.encoder_length,
    CONFIG.prediction_length,
)
validation_eval_dataset = CovidTCNDataset(
    validation_prepared,
    validation_eval_samples,
    validation_metadata["feature_columns"],
    CONFIG.encoder_length,
    CONFIG.prediction_length,
)
final_train_dataset = CovidTCNDataset(
    final_prepared,
    final_train_samples,
    final_metadata["feature_columns"],
    CONFIG.encoder_length,
    CONFIG.prediction_length,
)
shock_dataset = CovidTCNDataset(
    final_prepared,
    shock_samples,
    final_metadata["feature_columns"],
    CONFIG.encoder_length,
    CONFIG.prediction_length,
)
recovery_dataset = CovidTCNDataset(
    final_prepared,
    recovery_samples,
    final_metadata["feature_columns"],
    CONFIG.encoder_length,
    CONFIG.prediction_length,
) if recovery_samples else None

validation_tcn, validation_tcn_history = train_tcn_for_experiment(
    validation_train_dataset,
    validation_metadata,
    max_epochs=CONFIG.max_epochs_validation_model,
    validation_dataset=validation_eval_dataset,
)
final_tcn, final_tcn_history = train_tcn_for_experiment(
    final_train_dataset,
    final_metadata,
    max_epochs=CONFIG.max_epochs_final_model,
    validation_dataset=None,
)

validation_tcn_history.to_csv(PATHS.covid_stress_test_dir / "tcn_validation_model_training_history.csv", index=False)
final_tcn_history.to_csv(PATHS.covid_stress_test_dir / "tcn_final_precovid_training_history.csv", index=False)

covid_model_config = build_tcn_model_config(
    n_features=len(final_metadata["feature_columns"]),
    n_trusts=len(final_metadata["trust_to_idx"]),
    n_specialties=len(final_metadata["specialty_to_idx"]),
    prediction_length=CONFIG.prediction_length,
    quantiles=QUANTILES,
    hidden_channels=CONFIG.hidden_channels,
    tcn_levels=CONFIG.tcn_levels,
    kernel_size=CONFIG.kernel_size,
    dropout=CONFIG.dropout,
    embedding_dim=CONFIG.embedding_dim,
)
torch.save(final_tcn.state_dict(), PATHS.covid_stress_test_dir / "tcn_precovid_state_dict.pt")
with open(PATHS.covid_stress_test_dir / "tcn_precovid_model_config.json", "w", encoding="utf-8") as handle:
    json.dump(covid_model_config, handle, indent=2)
with open(PATHS.covid_stress_test_dir / "tcn_precovid_feature_metadata.json", "w", encoding="utf-8") as handle:
    json.dump(final_metadata, handle, indent=2)

# %% Cell 7
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm


def inverse_log1p(values: np.ndarray) -> np.ndarray:
    return np.clip(np.expm1(np.maximum(values, 0.0)), 0.0, None)


def predict_tcn(
    model: TCNQuantileRegressor,
    dataset: CovidTCNDataset,
    model_name: str,
    period_name: str,
) -> pd.DataFrame:
    if len(dataset) == 0:
        return pd.DataFrame()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    loader = make_loader(dataset, CONFIG.batch_size, shuffle=False)
    prediction_batches = []
    sample_index_batches = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Predicting {model_name} {period_name}"):
            x = batch["x"].to(device, non_blocking=True)
            trust_idx = batch["trust_idx"].to(device, non_blocking=True)
            specialty_idx = batch["specialty_idx"].to(device, non_blocking=True)
            prediction_log = model(x, trust_idx, specialty_idx).detach().cpu().numpy()
            prediction_log.sort(axis=-1)
            prediction_batches.append(prediction_log)
            sample_index_batches.append(batch["sample_idx"].detach().cpu().numpy())
    prediction_log_all = np.concatenate(prediction_batches, axis=0)
    sample_indices = np.concatenate(sample_index_batches, axis=0)
    prediction_all = inverse_log1p(prediction_log_all)
    records: List[Dict[str, object]] = []
    for batch_pos, sample_idx in enumerate(sample_indices):
        sample = dataset.samples[int(sample_idx)]
        series_id = str(sample["series_id"])
        series_frame = dataset.series_frames[series_id]
        forecast_start_pos = int(sample["forecast_start_pos"])
        row = series_frame.iloc[forecast_start_pos]
        records.append(
            {
                "model": model_name,
                "period": period_name,
                COLUMNS.trust_code: str(row[COLUMNS.trust_code]),
                COLUMNS.trust_name: str(row[COLUMNS.trust_name]),
                COLUMNS.specialty_code: str(row[COLUMNS.specialty_code]),
                COLUMNS.specialty_name: str(row[COLUMNS.specialty_name]),
                COLUMNS.forecast_origin: pd.Timestamp(sample["forecast_origin"]),
                COLUMNS.forecast_month: pd.Timestamp(row["month"]),
                COLUMNS.horizon: 1,
                COLUMNS.actual: float(row["target_actual"]),
                COLUMNS.p10: float(prediction_all[batch_pos, 0, 0]),
                COLUMNS.p50: float(prediction_all[batch_pos, 0, 1]),
                COLUMNS.p90: float(prediction_all[batch_pos, 0, 2]),
            }
        )
    return pd.DataFrame(records)


tcn_validation_predictions = predict_tcn(validation_tcn, validation_eval_dataset, "TCN", "pre_covid_validation")
tcn_shock_predictions = predict_tcn(final_tcn, shock_dataset, "TCN", "covid_shock")
tcn_recovery_predictions = (
    predict_tcn(final_tcn, recovery_dataset, "TCN", "recovery")
    if recovery_dataset is not None
    else pd.DataFrame()
)

print(
    {
        "TCN validation rows": len(tcn_validation_predictions),
        "TCN COVID shock rows": len(tcn_shock_predictions),
        "TCN recovery rows": len(tcn_recovery_predictions),
    }
)

# %% Cell 8
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


TABULAR_TARGET_LAGS = (1, 2, 3, 6, 12)
TABULAR_ORIGIN_FEATURES = [
    "new_rtt_periods",
    "completed_total",
    "reported_net_inflow",
    "net_inflow",
    "unreported_removals",
    "opening_waiting_list",
    "closing_waiting_list",
    "month_sin",
    "month_cos",
    "time_idx",
]


def build_tabular_rows(
    frame: pd.DataFrame,
    target_column: str,
    start_month: Optional[pd.Timestamp],
    end_month: Optional[pd.Timestamp],
) -> Tuple[pd.DataFrame, List[str]]:
    rows = []
    feature_columns = [f"target_lag_{lag}" for lag in TABULAR_TARGET_LAGS]
    available_origin_features = [column for column in TABULAR_ORIGIN_FEATURES if column in frame.columns]
    feature_columns.extend([f"origin_{column}" for column in available_origin_features])
    feature_columns.extend(["forecast_month_sin", "forecast_month_cos", "forecast_month_index"])

    for series_id, group in frame.groupby(COLUMNS.series_id, sort=False, observed=True):
        group = group.sort_values("month").reset_index(drop=True).copy()
        group["target_actual"] = pd.to_numeric(group[target_column], errors="coerce").astype(float)
        for lag in TABULAR_TARGET_LAGS:
            group[f"target_lag_{lag}"] = group["target_actual"].shift(lag)
        for column in available_origin_features:
            group[f"origin_{column}"] = pd.to_numeric(group[column], errors="coerce").shift(1)
        group["forecast_month_sin"] = np.sin(2.0 * np.pi * group["month"].dt.month.astype(float) / 12.0)
        group["forecast_month_cos"] = np.cos(2.0 * np.pi * group["month"].dt.month.astype(float) / 12.0)
        group["forecast_month_index"] = group["time_idx"].astype(float)
        mask = group["target_actual"].notna()
        if start_month is not None:
            mask &= group["month"] >= pd.Timestamp(start_month)
        if end_month is not None:
            mask &= group["month"] <= pd.Timestamp(end_month)
        usable = group.loc[mask].copy()
        usable = usable.dropna(subset=[f"target_lag_{lag}" for lag in TABULAR_TARGET_LAGS])
        if usable.empty:
            continue
        for row in usable.itertuples(index=False):
            record = {
                "series_id": str(getattr(row, COLUMNS.series_id)),
                "model": "Random forest",
                COLUMNS.trust_code: str(getattr(row, COLUMNS.trust_code)),
                COLUMNS.trust_name: str(getattr(row, COLUMNS.trust_name)),
                COLUMNS.specialty_code: str(getattr(row, COLUMNS.specialty_code)),
                COLUMNS.specialty_name: str(getattr(row, COLUMNS.specialty_name)),
                COLUMNS.forecast_origin: pd.Timestamp(getattr(row, "month")) - pd.DateOffset(months=1),
                COLUMNS.forecast_month: pd.Timestamp(getattr(row, "month")),
                COLUMNS.horizon: 1,
                COLUMNS.actual: float(getattr(row, "target_actual")),
            }
            for feature in feature_columns:
                record[feature] = float(getattr(row, feature)) if hasattr(row, feature) else np.nan
            rows.append(record)
    return pd.DataFrame(rows), feature_columns


def fit_random_forest_baseline(
    frame: pd.DataFrame,
    target_column: str,
    train_end_month: pd.Timestamp,
) -> Dict[str, object]:
    train_rows, feature_columns = build_tabular_rows(frame, target_column, start_month=None, end_month=train_end_month)
    if train_rows.empty:
        raise RuntimeError("No rows are available for the random forest baseline.")
    X_train = train_rows[feature_columns].to_numpy(dtype=float)
    feature_medians = np.nanmedian(X_train, axis=0)
    feature_medians = np.where(np.isfinite(feature_medians), feature_medians, 0.0)
    X_train = np.where(np.isfinite(X_train), X_train, feature_medians)
    y_train = np.log1p(train_rows[COLUMNS.actual].to_numpy(dtype=float).clip(min=0.0))
    model = RandomForestRegressor(
        n_estimators=CONFIG.random_forest_estimators,
        min_samples_leaf=CONFIG.random_forest_min_samples_leaf,
        random_state=CONFIG.random_seed,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    max_training_month = pd.to_datetime(train_rows[COLUMNS.forecast_month]).max()
    assert max_training_month <= pd.Timestamp(train_end_month)
    return {"model": model, "feature_columns": feature_columns, "feature_medians": feature_medians}


def predict_random_forest_baseline(
    fitted: Dict[str, object],
    frame: pd.DataFrame,
    target_column: str,
    start_month: pd.Timestamp,
    end_month: pd.Timestamp,
    period_name: str,
) -> pd.DataFrame:
    rows, feature_columns = build_tabular_rows(frame, target_column, start_month=start_month, end_month=end_month)
    if rows.empty:
        return pd.DataFrame()
    expected_features = list(fitted["feature_columns"])
    missing = [column for column in expected_features if column not in rows.columns]
    if missing:
        raise ValueError(f"Random forest prediction rows are missing fitted features: {missing}")
    X = rows[expected_features].to_numpy(dtype=float)
    medians = np.asarray(fitted["feature_medians"], dtype=float)
    X = np.where(np.isfinite(X), X, medians)
    model: RandomForestRegressor = fitted["model"]
    tree_predictions = np.vstack([estimator.predict(X) for estimator in model.estimators_])
    quantile_predictions = np.quantile(tree_predictions, [0.1, 0.5, 0.9], axis=0)
    predictions = inverse_log1p(quantile_predictions)
    output = rows[
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
    output["model"] = "Random forest"
    output["period"] = period_name
    output[COLUMNS.p10] = predictions[0]
    output[COLUMNS.p50] = predictions[1]
    output[COLUMNS.p90] = predictions[2]
    return output[
        [
            "model",
            "period",
            COLUMNS.trust_code,
            COLUMNS.trust_name,
            COLUMNS.specialty_code,
            COLUMNS.specialty_name,
            COLUMNS.forecast_origin,
            COLUMNS.forecast_month,
            COLUMNS.horizon,
            COLUMNS.actual,
            COLUMNS.p10,
            COLUMNS.p50,
            COLUMNS.p90,
        ]
    ]


rf_validation_fit = fit_random_forest_baseline(raw_rtt, TARGET_COLUMN, core_train_end)
rf_final_fit = fit_random_forest_baseline(raw_rtt, TARGET_COLUMN, train_end)

rf_validation_predictions = predict_random_forest_baseline(
    rf_validation_fit,
    raw_rtt,
    TARGET_COLUMN,
    validation_start,
    train_end,
    "pre_covid_validation",
)
rf_shock_predictions = predict_random_forest_baseline(
    rf_final_fit,
    raw_rtt,
    TARGET_COLUMN,
    covid_start,
    min(covid_end, pd.to_datetime(raw_rtt["month"]).max()),
    "covid_shock",
)
rf_recovery_predictions = (
    predict_random_forest_baseline(
        rf_final_fit,
        raw_rtt,
        TARGET_COLUMN,
        recovery_start,
        pd.to_datetime(raw_rtt["month"]).max(),
        "recovery",
    )
    if pd.to_datetime(raw_rtt["month"]).max() >= recovery_start
    else pd.DataFrame()
)

print(
    {
        "RF validation rows": len(rf_validation_predictions),
        "RF COVID shock rows": len(rf_shock_predictions),
        "RF recovery rows": len(rf_recovery_predictions),
    }
)

# %% Cell 9
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


def seasonal_naive_residual_quantiles(
    frame: pd.DataFrame,
    target_column: str,
    train_end_month: pd.Timestamp,
) -> Tuple[float, float]:
    residuals = []
    for _, group in frame.groupby(COLUMNS.series_id, sort=False, observed=True):
        group = group.sort_values("month").reset_index(drop=True).copy()
        target = pd.to_numeric(group[target_column], errors="coerce").astype(float)
        lag_12 = target.shift(12)
        mask = (group["month"] <= pd.Timestamp(train_end_month)) & target.notna() & lag_12.notna()
        residuals.extend((target[mask] - lag_12[mask]).tolist())
    if not residuals:
        return 0.0, 0.0
    return float(np.quantile(residuals, 0.1)), float(np.quantile(residuals, 0.9))


def predict_seasonal_naive(
    frame: pd.DataFrame,
    target_column: str,
    residual_q10: float,
    residual_q90: float,
    start_month: pd.Timestamp,
    end_month: pd.Timestamp,
    period_name: str,
) -> pd.DataFrame:
    rows = []
    for _, group in frame.groupby(COLUMNS.series_id, sort=False, observed=True):
        group = group.sort_values("month").reset_index(drop=True).copy()
        target = pd.to_numeric(group[target_column], errors="coerce").astype(float)
        group["lag12"] = target.shift(12)
        mask = (
            (group["month"] >= pd.Timestamp(start_month))
            & (group["month"] <= pd.Timestamp(end_month))
            & target.notna()
            & group["lag12"].notna()
        )
        for row in group.loc[mask].itertuples(index=False):
            p50 = float(getattr(row, "lag12"))
            p10 = max(0.0, p50 + float(residual_q10))
            p90 = max(0.0, p50 + float(residual_q90))
            p10, p50_sorted, p90 = sorted([p10, p50, p90])
            rows.append(
                {
                    "model": "Seasonal naive",
                    "period": period_name,
                    COLUMNS.trust_code: str(getattr(row, COLUMNS.trust_code)),
                    COLUMNS.trust_name: str(getattr(row, COLUMNS.trust_name)),
                    COLUMNS.specialty_code: str(getattr(row, COLUMNS.specialty_code)),
                    COLUMNS.specialty_name: str(getattr(row, COLUMNS.specialty_name)),
                    COLUMNS.forecast_origin: pd.Timestamp(getattr(row, "month")) - pd.DateOffset(months=1),
                    COLUMNS.forecast_month: pd.Timestamp(getattr(row, "month")),
                    COLUMNS.horizon: 1,
                    COLUMNS.actual: float(getattr(row, target_column)),
                    COLUMNS.p10: float(p10),
                    COLUMNS.p50: float(p50_sorted),
                    COLUMNS.p90: float(p90),
                }
            )
    return pd.DataFrame(rows)


naive_validation_q10, naive_validation_q90 = seasonal_naive_residual_quantiles(raw_rtt, TARGET_COLUMN, core_train_end)
naive_final_q10, naive_final_q90 = seasonal_naive_residual_quantiles(raw_rtt, TARGET_COLUMN, train_end)

naive_validation_predictions = predict_seasonal_naive(
    raw_rtt,
    TARGET_COLUMN,
    naive_validation_q10,
    naive_validation_q90,
    validation_start,
    train_end,
    "pre_covid_validation",
)
naive_shock_predictions = predict_seasonal_naive(
    raw_rtt,
    TARGET_COLUMN,
    naive_final_q10,
    naive_final_q90,
    covid_start,
    min(covid_end, pd.to_datetime(raw_rtt["month"]).max()),
    "covid_shock",
)
naive_recovery_predictions = (
    predict_seasonal_naive(
        raw_rtt,
        TARGET_COLUMN,
        naive_final_q10,
        naive_final_q90,
        recovery_start,
        pd.to_datetime(raw_rtt["month"]).max(),
        "recovery",
    )
    if pd.to_datetime(raw_rtt["month"]).max() >= recovery_start
    else pd.DataFrame()
)

print(
    {
        "Seasonal naive validation rows": len(naive_validation_predictions),
        "Seasonal naive COVID shock rows": len(naive_shock_predictions),
        "Seasonal naive recovery rows": len(naive_recovery_predictions),
    }
)

# %% Cell 10
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


PREDICTION_COLUMNS = [
    "model",
    "period",
    COLUMNS.trust_code,
    COLUMNS.trust_name,
    COLUMNS.specialty_code,
    COLUMNS.specialty_name,
    COLUMNS.forecast_origin,
    COLUMNS.forecast_month,
    COLUMNS.horizon,
    COLUMNS.actual,
    COLUMNS.p10,
    COLUMNS.p50,
    COLUMNS.p90,
]


def pinball_loss_np(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> np.ndarray:
    error = y_true - y_pred
    return np.maximum(quantile * error, (quantile - 1.0) * error)


def smape_np(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    values = np.full_like(y_true, np.nan, dtype=float)
    mask = denominator > 0.0
    values[mask] = 100.0 * np.abs(y_pred[mask] - y_true[mask]) / denominator[mask]
    return values


def normalise_prediction_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=PREDICTION_COLUMNS)
    output = frame.copy()
    output = output[PREDICTION_COLUMNS]
    for column in [COLUMNS.forecast_origin, COLUMNS.forecast_month]:
        output[column] = pd.to_datetime(output[column]).dt.to_period("M").dt.to_timestamp()
    for column in [COLUMNS.actual, COLUMNS.p10, COLUMNS.p50, COLUMNS.p90]:
        output[column] = pd.to_numeric(output[column], errors="coerce")
    output[[COLUMNS.p10, COLUMNS.p50, COLUMNS.p90]] = np.sort(
        output[[COLUMNS.p10, COLUMNS.p50, COLUMNS.p90]].to_numpy(dtype=float),
        axis=1,
    )
    output = output.dropna(subset=[COLUMNS.actual, COLUMNS.p10, COLUMNS.p50, COLUMNS.p90])
    return output.reset_index(drop=True)


all_predictions = pd.concat(
    [
        tcn_validation_predictions,
        tcn_shock_predictions,
        tcn_recovery_predictions,
        rf_validation_predictions,
        rf_shock_predictions,
        rf_recovery_predictions,
        naive_validation_predictions,
        naive_shock_predictions,
        naive_recovery_predictions,
    ],
    ignore_index=True,
)
all_predictions = normalise_prediction_frame(all_predictions)

if all_predictions.empty:
    raise RuntimeError("No COVID shock experiment predictions were produced.")

assert pd.to_datetime(all_predictions.loc[all_predictions["period"].eq("pre_covid_validation"), COLUMNS.forecast_month]).max() <= train_end
assert pd.to_datetime(all_predictions.loc[all_predictions["period"].eq("covid_shock"), COLUMNS.forecast_month]).min() >= covid_start
assert pd.to_datetime(all_predictions.loc[all_predictions["period"].eq("covid_shock"), COLUMNS.forecast_month]).max() <= covid_end
assert train_end < pd.to_datetime(all_predictions.loc[all_predictions["period"].eq("covid_shock"), COLUMNS.forecast_month]).min()

all_predictions.to_parquet(PATHS.covid_predictions, index=False)
print(f"Saved predictions: {PATHS.covid_predictions}")
display(all_predictions.head(20))

# %% Cell 11
from typing import Dict, List

import numpy as np
import pandas as pd


def metric_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model_name, period_name), group in frame.groupby(["model", "period"], observed=True):
        y = group[COLUMNS.actual].to_numpy(dtype=float)
        p10 = group[COLUMNS.p10].to_numpy(dtype=float)
        p50 = group[COLUMNS.p50].to_numpy(dtype=float)
        p90 = group[COLUMNS.p90].to_numpy(dtype=float)
        errors = p50 - y
        pinball_q10 = pinball_loss_np(y, p10, 0.1)
        pinball_q50 = pinball_loss_np(y, p50, 0.5)
        pinball_q90 = pinball_loss_np(y, p90, 0.9)
        smape_values = smape_np(y, p50)
        rows.append(
            {
                "model": model_name,
                "period": period_name,
                "n": int(len(group)),
                "mae": float(np.mean(np.abs(errors))),
                "rmse": float(np.sqrt(np.mean(errors**2))),
                "smape": float(np.nanmean(smape_values)) if np.isfinite(smape_values).any() else np.nan,
                "pinball_q10": float(np.mean(pinball_q10)),
                "pinball_q50": float(np.mean(pinball_q50)),
                "pinball_q90": float(np.mean(pinball_q90)),
                "pinball_mean": float(np.mean(np.vstack([pinball_q10, pinball_q50, pinball_q90]))),
                "p10_p90_empirical_coverage": float(np.mean((y >= p10) & (y <= p90))),
                "average_interval_width": float(np.mean(p90 - p10)),
            }
        )
    return pd.DataFrame(rows).sort_values(["model", "period"]).reset_index(drop=True)


def degradation_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    validation = metrics[metrics["period"].eq("pre_covid_validation")]
    shock = metrics[metrics["period"].eq("covid_shock")]
    for model_name in sorted(set(validation["model"]).intersection(set(shock["model"]))):
        normal_row = validation[validation["model"].eq(model_name)].iloc[0]
        shock_row = shock[shock["model"].eq(model_name)].iloc[0]
        record = {"model": model_name}
        for metric in ["mae", "rmse", "smape", "pinball_mean", "average_interval_width"]:
            normal_value = float(normal_row[metric])
            shock_value = float(shock_row[metric])
            if np.isfinite(normal_value) and abs(normal_value) > 1.0e-12:
                pct = 100.0 * (shock_value - normal_value) / normal_value
            else:
                pct = np.nan
            record[f"{metric}_pre_covid_validation"] = normal_value
            record[f"{metric}_covid_shock"] = shock_value
            record[f"{metric}_pct_degradation"] = pct
        record["p10_p90_coverage_pre_covid_validation"] = float(normal_row["p10_p90_empirical_coverage"])
        record["p10_p90_coverage_covid_shock"] = float(shock_row["p10_p90_empirical_coverage"])
        record["p10_p90_coverage_change_points"] = float(
            shock_row["p10_p90_empirical_coverage"] - normal_row["p10_p90_empirical_coverage"]
        )
        rows.append(record)
    return pd.DataFrame(rows)


metrics = metric_summary(all_predictions)
degradation = degradation_summary(metrics)

metrics.to_csv(PATHS.covid_metrics, index=False)
degradation.to_csv(PATHS.covid_degradation, index=False)

print(f"Saved metrics: {PATHS.covid_metrics}")
print(f"Saved degradation summary: {PATHS.covid_degradation}")
display(metrics)
display(degradation)

# %% Cell 12
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_aggregate_actual_vs_forecast(predictions: pd.DataFrame, output_path: Path) -> None:
    shock = predictions[predictions["period"].eq("covid_shock")].copy()
    if shock.empty:
        return
    actual = (
        shock[shock["model"].eq("TCN")]
        .groupby(COLUMNS.forecast_month, as_index=False, observed=True)[[COLUMNS.actual, COLUMNS.p10, COLUMNS.p50, COLUMNS.p90]]
        .sum()
        .sort_values(COLUMNS.forecast_month)
    )
    model_lines = (
        shock.groupby(["model", COLUMNS.forecast_month], as_index=False, observed=True)[COLUMNS.p50]
        .sum()
        .sort_values(COLUMNS.forecast_month)
    )
    fig, ax = plt.subplots(figsize=(12, 6))
    if not actual.empty:
        x = pd.to_datetime(actual[COLUMNS.forecast_month]).to_numpy()
        ax.fill_between(
            x,
            actual[COLUMNS.p10].to_numpy(dtype=float),
            actual[COLUMNS.p90].to_numpy(dtype=float),
            color="#9ecae1",
            alpha=0.35,
            label="TCN P10-P90 interval",
        )
        ax.plot(x, actual[COLUMNS.actual].to_numpy(dtype=float), color="#111111", linewidth=2.2, label="Actual")
    for model_name, group in model_lines.groupby("model", observed=True):
        ax.plot(
            pd.to_datetime(group[COLUMNS.forecast_month]).to_numpy(),
            group[COLUMNS.p50].to_numpy(dtype=float),
            linewidth=1.9,
            label=f"{model_name} median",
        )
    ax.set_title("COVID Shock Period: National Actual vs Forecast")
    ax.set_xlabel("Forecast month")
    ax.set_ylabel("Incomplete RTT pathways")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_tcn_interval_plot(predictions: pd.DataFrame, output_path: Path) -> None:
    shock = predictions[(predictions["period"].eq("covid_shock")) & (predictions["model"].eq("TCN"))].copy()
    if shock.empty:
        return
    monthly = (
        shock.groupby(COLUMNS.forecast_month, as_index=False, observed=True)[[COLUMNS.actual, COLUMNS.p10, COLUMNS.p50, COLUMNS.p90]]
        .sum()
        .sort_values(COLUMNS.forecast_month)
    )
    x = pd.to_datetime(monthly[COLUMNS.forecast_month]).to_numpy()
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.fill_between(
        x,
        monthly[COLUMNS.p10].to_numpy(dtype=float),
        monthly[COLUMNS.p90].to_numpy(dtype=float),
        color="#bcbddc",
        alpha=0.45,
        label="P10-P90 interval",
    )
    ax.plot(x, monthly[COLUMNS.p50].to_numpy(dtype=float), color="#54278f", linewidth=2.3, label="TCN median")
    ax.plot(x, monthly[COLUMNS.actual].to_numpy(dtype=float), color="#111111", linewidth=2.0, linestyle="--", label="Actual")
    ax.set_title("TCN Prediction Interval During COVID Shock Period")
    ax.set_xlabel("Forecast month")
    ax.set_ylabel("Incomplete RTT pathways")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_error_over_time_plot(predictions: pd.DataFrame, output_path: Path) -> None:
    working = predictions.copy()
    working["absolute_error"] = (working[COLUMNS.p50] - working[COLUMNS.actual]).abs()
    monthly = (
        working.groupby(["model", "period", COLUMNS.forecast_month], as_index=False, observed=True)["absolute_error"]
        .mean()
        .sort_values(COLUMNS.forecast_month)
    )
    fig, ax = plt.subplots(figsize=(12, 6))
    for (model_name, period_name), group in monthly.groupby(["model", "period"], observed=True):
        ax.plot(
            pd.to_datetime(group[COLUMNS.forecast_month]).to_numpy(),
            group["absolute_error"].to_numpy(dtype=float),
            linewidth=1.8,
            label=f"{model_name} - {period_name}",
        )
    ax.axvline(covid_start, color="#b2182b", linestyle="--", linewidth=1.4, label="COVID shock test start")
    ax.set_title("Mean Absolute Error Over Time")
    ax.set_xlabel("Forecast month")
    ax.set_ylabel("Mean absolute error")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_performance_bar_plot(
    predictions: pd.DataFrame,
    group_columns: List[str],
    label_column: str,
    output_path: Path,
    title: str,
) -> None:
    shock = predictions[(predictions["period"].eq("covid_shock")) & (predictions["model"].eq("TCN"))].copy()
    if shock.empty:
        return
    shock["absolute_error"] = (shock[COLUMNS.p50] - shock[COLUMNS.actual]).abs()
    grouped = (
        shock.groupby(group_columns, as_index=False, observed=True)["absolute_error"]
        .mean()
        .sort_values("absolute_error", ascending=False)
        .head(CONFIG.plot_top_n)
    )
    labels = grouped[label_column].astype(str).to_list()
    values = grouped["absolute_error"].to_numpy(dtype=float)
    fig_height = max(6, 0.35 * len(grouped) + 2)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    ax.barh(labels[::-1], values[::-1], color="#3182bd")
    ax.set_title(title)
    ax.set_xlabel("Mean absolute error")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


save_aggregate_actual_vs_forecast(all_predictions, PATHS.covid_actual_vs_forecast_png)
save_tcn_interval_plot(all_predictions, PATHS.covid_prediction_intervals_png)
save_error_over_time_plot(all_predictions, PATHS.covid_error_over_time_png)
save_performance_bar_plot(
    all_predictions,
    [COLUMNS.trust_code, COLUMNS.trust_name],
    COLUMNS.trust_name,
    PATHS.covid_performance_by_trust_png,
    "COVID Shock Period TCN Performance by Trust",
)
save_performance_bar_plot(
    all_predictions,
    [COLUMNS.specialty_code, COLUMNS.specialty_name],
    COLUMNS.specialty_name,
    PATHS.covid_performance_by_specialty_png,
    "COVID Shock Period TCN Performance by Specialty",
)

print("Saved plots:")
for plot_path in [
    PATHS.covid_actual_vs_forecast_png,
    PATHS.covid_prediction_intervals_png,
    PATHS.covid_error_over_time_png,
    PATHS.covid_performance_by_trust_png,
    PATHS.covid_performance_by_specialty_png,
]:
    print(plot_path)

# %% Cell 13
import json
from pathlib import Path

import pandas as pd


methodology_note = f"""# COVID Shock Forecasting Experiment

## Purpose

This is a separate forecasting stress-test experiment. It does not replace the production Layer 1 model, which may be trained on the full available historical dataset for normal project outputs.

## Observed Dataset Coverage

- First available month: {date_coverage["minimum_month"]}
- Final available month: {date_coverage["maximum_month"]}
- Observed monthly periods in project range: {date_coverage["observed_month_count"]}
- Missing calendar months in project range: {len(date_coverage["missing_calendar_months_in_project_range"])}

## Experimental Split

- Core pre-COVID training period for validation model: through {core_train_end.date().isoformat()}
- Pre-COVID validation period: {validation_start.date().isoformat()} to {train_end.date().isoformat()}
- Final pre-COVID training period for shock-period model: through {train_end.date().isoformat()}
- COVID shock test period: {covid_start.date().isoformat()} to {covid_end.date().isoformat()}
- Recovery period: from {recovery_start.date().isoformat()}, when observations exist after the shock window

The maximum final training forecast month is earlier than the minimum COVID shock forecast month. All feature scaling statistics for the final shock-period model are fit using rows no later than {train_end.date().isoformat()}.

## Models Compared

- Custom PyTorch TCN quantile regressor with P10, P50 and P90 outputs.
- Seasonal naive baseline using the same month one year earlier, with P10 and P90 formed from pre-COVID residual quantiles.
- Random forest baseline using lagged target values and origin-month operational/calendar features.

## Forecast Protocol

Forecasts are rolling one-month-ahead predictions. For each forecast month, model inputs use observed information available up to the forecast origin, which is the previous month. Model parameters and preprocessing objects are not fit on COVID-period or recovery-period targets.

## Limitations

This experiment tests degradation when pre-COVID fitted models are applied during a severe service disruption. It is not a causal estimate of COVID effects, and it does not simulate counterfactual operational policy. Later COVID-period one-step forecasts may use already-observed previous COVID months as context, but those observations are never used to fit model parameters or scalers.
"""

with open(PATHS.covid_methodology_note, "w", encoding="utf-8") as handle:
    handle.write(methodology_note)

leakage_checks = {
    "max_validation_model_training_target_month": max(sample["forecast_month"] for sample in validation_train_samples).date().isoformat(),
    "min_pre_covid_validation_target_month": min(sample["forecast_month"] for sample in validation_eval_samples).date().isoformat(),
    "max_final_model_training_target_month": max(sample["forecast_month"] for sample in final_train_samples).date().isoformat(),
    "min_covid_shock_target_month": min(sample["forecast_month"] for sample in shock_samples).date().isoformat(),
    "final_model_scaler_fit_end_month": final_metadata["scaler_fit_end_month"],
    "training_before_covid_test": bool(
        max(sample["forecast_month"] for sample in final_train_samples)
        < min(sample["forecast_month"] for sample in shock_samples)
    ),
    "scaler_fit_before_covid_test": bool(pd.Timestamp(final_metadata["scaler_fit_end_month"]) < covid_start),
}
assert leakage_checks["training_before_covid_test"]
assert leakage_checks["scaler_fit_before_covid_test"]

split_summary["leakage_checks"] = leakage_checks
split_summary["outputs"] = {
    "predictions": str(PATHS.covid_predictions),
    "metrics": str(PATHS.covid_metrics),
    "degradation": str(PATHS.covid_degradation),
    "methodology_note": str(PATHS.covid_methodology_note),
    "plots": [
        str(PATHS.covid_actual_vs_forecast_png),
        str(PATHS.covid_prediction_intervals_png),
        str(PATHS.covid_error_over_time_png),
        str(PATHS.covid_performance_by_trust_png),
        str(PATHS.covid_performance_by_specialty_png),
    ],
}
with open(PATHS.covid_split_summary, "w", encoding="utf-8") as handle:
    json.dump(split_summary, handle, indent=2)

summary = {
    "final_observed_month": date_coverage["maximum_month"],
    "pre_covid_training_end": train_end.date().isoformat(),
    "covid_test_start": covid_start.date().isoformat(),
    "covid_test_end": covid_end.date().isoformat(),
    "trust_specialty_series_forecast": sample_summary["trust_specialty_series_forecast"],
    "prediction_rows_produced": int(len(all_predictions)),
    "metrics_rows_produced": int(len(metrics)),
    "leakage_checks": leakage_checks,
}

print(json.dumps(summary, indent=2))
print(f"Saved methodology note: {PATHS.covid_methodology_note}")
print(f"Saved split summary: {PATHS.covid_split_summary}")
