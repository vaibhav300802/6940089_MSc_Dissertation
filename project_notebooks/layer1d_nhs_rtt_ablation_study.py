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
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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
        "Could not find the shared nhs_rtt_pipeline package. "
        "Run this notebook from the project root or upload the full nhs_rtt_msc_project folder to Colab."
    )

from nhs_rtt_pipeline.config import COLUMNS, ensure_directories, get_paths
from nhs_rtt_pipeline.modeling import QuantileLoss, TCNQuantileRegressor, build_tcn_model_config
from nhs_rtt_pipeline.preprocessing import (
    FLOW_LAGGED_FEATURES,
    FLOW_MISSINGNESS_FEATURES,
    LOG1P_NON_NEGATIVE_FEATURES,
    SIGNED_OPERATIONAL_FEATURES,
    assert_net_inflow_integrity,
    feature_group_for_column,
)
from nhs_rtt_pipeline.reproducibility import set_global_seed
from nhs_rtt_pipeline.settings import load_pipeline_settings

# %% Cell 3
@dataclass(frozen=True)
class AblationStudyConfig:
    target_column: str = "incomplete_total"
    encoder_length: int = 24
    prediction_length: int = 12
    validation_months: int = 12
    test_months: int = 12
    batch_size: int = 512
    max_epochs: int = 20
    early_stopping_patience: int = 5
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    hidden_channels: int = 96
    tcn_levels: int = 5
    kernel_size: int = 3
    dropout: float = 0.15
    embedding_dim: int = 16
    gradient_clip_norm: float = 1.0
    random_seed: int = 42
    num_workers: int = 2


try:
    PROJECT_SETTINGS = load_pipeline_settings(os.environ.get("NHS_RTT_PIPELINE_CONFIG"))
    MODEL_SETTINGS = PROJECT_SETTINGS.model
    CONFIG = AblationStudyConfig(
        encoder_length=int(MODEL_SETTINGS.get("encoder_length", 24)),
        prediction_length=int(MODEL_SETTINGS.get("prediction_length", 12)),
        hidden_channels=int(MODEL_SETTINGS.get("hidden_channels", 96)),
        tcn_levels=int(MODEL_SETTINGS.get("tcn_levels", 5)),
        kernel_size=int(MODEL_SETTINGS.get("kernel_size", 3)),
        dropout=float(MODEL_SETTINGS.get("dropout", 0.15)),
        embedding_dim=int(MODEL_SETTINGS.get("embedding_dim", 16)),
        random_seed=int(PROJECT_SETTINGS.random_seed),
    )
except FileNotFoundError:
    PROJECT_SETTINGS = None
    CONFIG = AblationStudyConfig()

if os.name == "nt" and CONFIG.num_workers != 0:
    CONFIG = AblationStudyConfig(**{**asdict(CONFIG), "num_workers": 0})
    print("Windows execution detected; using num_workers=0 for safe PyTorch DataLoader startup.")

QUANTILES = (0.1, 0.5, 0.9)
PATHS = get_paths()
ensure_directories(PATHS)
ABLATION_DIR = PATHS.outputs_dir / "ablation_study"
ABLATION_DIR.mkdir(parents=True, exist_ok=True)


def set_random_seed(seed: int) -> None:
    deterministic = bool(PROJECT_SETTINGS.deterministic_torch) if PROJECT_SETTINGS is not None else False
    set_global_seed(seed, deterministic_torch=deterministic)
    if torch.cuda.is_available() and not deterministic:
        torch.backends.cudnn.benchmark = True


set_random_seed(CONFIG.random_seed)
print(json.dumps(asdict(CONFIG), indent=2))
print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
print(f"Ablation study outputs: {ABLATION_DIR}")

# %% Cell 4
BASE_MODEL_INPUT_FEATURES = [
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

MISSINGNESS_FLAG_FEATURES = [
    *FLOW_MISSINGNESS_FEATURES,
    "is_imputed_month",
    "missing_month",
    "waiting_list_imputed",
    "waiting_list_with_dta_imputed",
    "completed_admitted_imputed",
    "completed_non_admitted_imputed",
    "new_rtt_periods_imputed",
]

# Each variant isolates exactly one change from the full production model, so any accuracy
# difference can be attributed to that single component rather than to a confound.
ABLATION_VARIANTS = [
    {
        "name": "full_model",
        "label": "Full model (baseline)",
        "drop_features": [],
        "neutralise_entities": False,
        "description": "The production model exactly as trained in Layer 1; not retrained here, its "
        "existing saved backtest predictions are reused so the comparison is against the real deployed model.",
    },
    {
        "name": "no_entity_identity",
        "label": "No Trust/specialty identity",
        "drop_features": [],
        "neutralise_entities": True,
        "description": "Retrained with every Trust and specialty index forced to the same constant value, "
        "so the entity embedding layer cannot distinguish one Trust or specialty from another.",
    },
    {
        "name": "no_lagged_features",
        "label": "No lagged features",
        "drop_features": list(FLOW_LAGGED_FEATURES),
        "neutralise_entities": False,
        "description": "Retrained with every 1/3/6-month lagged referral, completion and net-inflow "
        "feature removed from the input list.",
    },
    {
        "name": "no_missingness_flags",
        "label": "No missingness/imputation flags",
        "drop_features": list(MISSINGNESS_FLAG_FEATURES),
        "neutralise_entities": False,
        "description": "Retrained with every missingness indicator and imputation flag removed from the "
        "input list, so the model can no longer tell an observed month from a filled one.",
    },
]

print(f"{len(ABLATION_VARIANTS)} ablation variants configured:")
for variant in ABLATION_VARIANTS:
    print(f"  - {variant['name']}: {variant['label']}")

# %% Cell 5
def split_boundaries(frame: pd.DataFrame, config: AblationStudyConfig) -> Dict[str, int]:
    max_time_idx = int(frame["time_idx"].max())
    test_start_idx = max_time_idx - config.test_months + 1
    validation_start_idx = test_start_idx - config.validation_months
    if validation_start_idx <= int(frame["time_idx"].min()):
        raise ValueError("Not enough monthly history for the requested train/validation/test split.")
    return {
        "max_time_idx": max_time_idx,
        "validation_start_idx": validation_start_idx,
        "test_start_idx": test_start_idx,
    }


def prepare_ablation_frame(
    frame: pd.DataFrame,
    config: AblationStudyConfig,
    feature_columns: Sequence[str],
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    prepared = frame.copy().sort_values(["series_id", "time_idx"]).reset_index(drop=True)
    assert_net_inflow_integrity(prepared)
    boundaries = split_boundaries(prepared, config)
    train_period = prepared["time_idx"] < boundaries["validation_start_idx"]
    transformed_feature_columns = []

    for column in feature_columns:
        if column not in prepared.columns:
            raise ValueError(f"Prepared RTT frame is missing model input feature: {column}")
        values = pd.to_numeric(prepared[column], errors="coerce").astype(float)
        if column in LOG1P_NON_NEGATIVE_FEATURES:
            transformed = np.log1p(values.clip(lower=0.0))
        elif column in SIGNED_OPERATIONAL_FEATURES:
            transformed = values
        else:
            transformed = values
        train_values = transformed[train_period].dropna()
        mean = float(train_values.mean()) if len(train_values) else 0.0
        std = float(train_values.std(ddof=0)) if len(train_values) else 1.0
        if not np.isfinite(std) or std < 1.0e-8:
            std = 1.0
        model_column = f"{column}_ablation_model"
        prepared[model_column] = ((transformed.fillna(mean) - mean) / std).astype("float32")
        transformed_feature_columns.append(model_column)

    target_column = config.target_column
    target_values = pd.to_numeric(prepared[target_column], errors="coerce")
    available_column = f"{target_column}_source_available"
    if available_column in prepared.columns:
        available = pd.to_numeric(prepared[available_column], errors="coerce").fillna(0).astype(int).eq(1)
        target_values = target_values.where(available)
    prepared["target_actual"] = target_values.astype("float32")
    prepared["target_log"] = np.log1p(target_values.clip(lower=0.0)).astype("float32")

    trust_codes = sorted(prepared["trust_code"].astype(str).unique())
    specialty_codes = sorted(prepared["specialty_code"].astype(str).unique())
    trust_to_idx = {code: idx for idx, code in enumerate(trust_codes)}
    specialty_to_idx = {code: idx for idx, code in enumerate(specialty_codes)}
    prepared["trust_idx"] = prepared["trust_code"].map(trust_to_idx).astype("int64")
    prepared["specialty_idx"] = prepared["specialty_code"].map(specialty_to_idx).astype("int64")

    metadata = {
        "boundaries": boundaries,
        "feature_columns": transformed_feature_columns,
        "raw_feature_columns": list(feature_columns),
        "target_column": target_column,
        "trust_to_idx": trust_to_idx,
        "specialty_to_idx": specialty_to_idx,
        "quantiles": list(QUANTILES),
    }
    return prepared, metadata


def make_supervised_samples(
    frame: pd.DataFrame, config: AblationStudyConfig, boundaries: Dict[str, int]
) -> Dict[str, List[Dict[str, object]]]:
    samples = {"train": [], "val": [], "test": []}
    grouped = frame.groupby("series_id", sort=False, observed=True)

    for series_id, group in tqdm(grouped, total=grouped.ngroups, desc="Building supervised windows"):
        group = group.sort_values("time_idx").reset_index(drop=True)
        time_values = group["time_idx"].to_numpy()
        if len(group) < config.encoder_length + config.prediction_length:
            continue
        if not np.all(np.diff(time_values) == 1):
            continue

        for encoder_end_pos in range(config.encoder_length - 1, len(group) - config.prediction_length):
            forecast_start_pos = encoder_end_pos + 1
            forecast_end_pos = encoder_end_pos + config.prediction_length
            if group.loc[forecast_start_pos:forecast_end_pos, "target_log"].isna().any():
                continue
            forecast_start_idx = int(time_values[forecast_start_pos])
            forecast_end_idx = int(time_values[forecast_end_pos])
            sample = {
                "series_id": series_id,
                "encoder_end_pos": encoder_end_pos,
                "forecast_start_pos": forecast_start_pos,
                "forecast_end_pos": forecast_end_pos,
            }
            if forecast_end_idx < boundaries["validation_start_idx"]:
                samples["train"].append(sample)
            elif forecast_start_idx >= boundaries["validation_start_idx"] and forecast_end_idx < boundaries["test_start_idx"]:
                samples["val"].append(sample)
            elif forecast_start_idx >= boundaries["test_start_idx"] and forecast_end_idx <= boundaries["max_time_idx"]:
                samples["test"].append(sample)

    train_series_ids = {sample["series_id"] for sample in samples["train"]}
    samples["val"] = [sample for sample in samples["val"] if sample["series_id"] in train_series_ids]
    samples["test"] = [sample for sample in samples["test"] if sample["series_id"] in train_series_ids]

    if not samples["train"] or not samples["val"] or not samples["test"]:
        raise RuntimeError("An ablation variant produced empty train/val/test windows.")
    return samples

# %% Cell 6
class AblationDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        samples: List[Dict[str, object]],
        feature_columns: Sequence[str],
        encoder_length: int,
        prediction_length: int,
        neutralise_entities: bool,
    ) -> None:
        self.samples = samples
        self.feature_columns = list(feature_columns)
        self.encoder_length = encoder_length
        self.prediction_length = prediction_length
        self.neutralise_entities = bool(neutralise_entities)
        self.series_frames: Dict[str, pd.DataFrame] = {}
        self.series_arrays: Dict[str, Dict[str, object]] = {}

        for series_id, group in frame.groupby("series_id", sort=False, observed=True):
            group = group.sort_values("time_idx").reset_index(drop=True)
            self.series_frames[series_id] = group
            self.series_arrays[series_id] = {
                "features": group[self.feature_columns].to_numpy(dtype=np.float32),
                "target_log": group["target_log"].to_numpy(dtype=np.float32),
                "target_actual": group["target_actual"].to_numpy(dtype=np.float32),
                "trust_idx": 0 if self.neutralise_entities else int(group["trust_idx"].iloc[0]),
                "specialty_idx": 0 if self.neutralise_entities else int(group["specialty_idx"].iloc[0]),
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
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    grad_clip_norm: float,
) -> float:
    is_training = optimizer is not None
    model.train(is_training)
    losses = []
    for batch in tqdm(loader, leave=False, desc="train" if is_training else "eval"):
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        trust_idx = batch["trust_idx"].to(device, non_blocking=True)
        specialty_idx = batch["specialty_idx"].to(device, non_blocking=True)
        if is_training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_training):
            prediction = model(x, trust_idx, specialty_idx)
            loss = loss_fn(prediction, y)
            if is_training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
    return float(np.mean(losses))

# %% Cell 7
def inverse_log1p(values: np.ndarray) -> np.ndarray:
    return np.clip(np.expm1(np.maximum(values, 0.0)), 0.0, None)


def predict_ablation_quantiles(
    model: nn.Module, dataset: AblationDataset, config: AblationStudyConfig
) -> pd.DataFrame:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = make_loader(dataset, config.batch_size, False, config.num_workers)
    model.eval()
    model.to(device)
    prediction_batches = []
    sample_index_batches = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting"):
            x = batch["x"].to(device, non_blocking=True)
            trust_idx = batch["trust_idx"].to(device, non_blocking=True)
            specialty_idx = batch["specialty_idx"].to(device, non_blocking=True)
            prediction = model(x, trust_idx, specialty_idx).detach().cpu().numpy()
            prediction.sort(axis=-1)
            prediction_batches.append(prediction)
            sample_index_batches.append(batch["sample_idx"].detach().cpu().numpy())

    predictions_log = np.concatenate(prediction_batches, axis=0)
    sample_indices = np.concatenate(sample_index_batches, axis=0)
    predictions = inverse_log1p(predictions_log)
    records: List[Dict[str, object]] = []
    for batch_pos, sample_idx in enumerate(sample_indices):
        sample = dataset.samples[int(sample_idx)]
        series_id = str(sample["series_id"])
        series_frame = dataset.series_frames[series_id]
        forecast_start_pos = int(sample["forecast_start_pos"])
        for horizon_idx in range(dataset.prediction_length):
            row = series_frame.iloc[forecast_start_pos + horizon_idx]
            records.append(
                {
                    "series_id": series_id,
                    "horizon": horizon_idx + 1,
                    "actual": float(row["target_actual"]),
                    "p10": float(predictions[batch_pos, horizon_idx, 0]),
                    "p50": float(predictions[batch_pos, horizon_idx, 1]),
                    "p90": float(predictions[batch_pos, horizon_idx, 2]),
                }
            )
    return pd.DataFrame(records)


def pinball_loss_np(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> np.ndarray:
    error = y_true - y_pred
    return np.maximum(quantile * error, (quantile - 1.0) * error)


def compute_ablation_metrics(predictions: pd.DataFrame) -> Dict[str, float]:
    y = predictions["actual"].to_numpy(dtype=float)
    p10 = predictions["p10"].to_numpy(dtype=float)
    p50 = predictions["p50"].to_numpy(dtype=float)
    p90 = predictions["p90"].to_numpy(dtype=float)
    errors = p50 - y
    pinball_mean = float(
        np.mean(
            np.vstack(
                [
                    pinball_loss_np(y, p10, 0.1),
                    pinball_loss_np(y, p50, 0.5),
                    pinball_loss_np(y, p90, 0.9),
                ]
            )
        )
    )
    return {
        "n_rows": int(len(predictions)),
        "mae_median": float(np.mean(np.abs(errors))),
        "rmse_median": float(np.sqrt(np.mean(errors**2))),
        "pinball_mean": pinball_mean,
        "p10_p90_coverage": float(np.mean((y >= p10) & (y <= p90))),
    }


def train_ablation_variant(
    prepared: pd.DataFrame,
    metadata: Dict[str, object],
    samples: Dict[str, List[Dict[str, object]]],
    config: AblationStudyConfig,
    neutralise_entities: bool,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets = {
        split: AblationDataset(
            frame=prepared,
            samples=samples[split],
            feature_columns=metadata["feature_columns"],
            encoder_length=config.encoder_length,
            prediction_length=config.prediction_length,
            neutralise_entities=neutralise_entities,
        )
        for split in ["train", "val", "test"]
    }
    loaders = {
        "train": make_loader(datasets["train"], config.batch_size, True, config.num_workers),
        "val": make_loader(datasets["val"], config.batch_size, False, config.num_workers),
    }
    model = TCNQuantileRegressor(
        n_features=len(metadata["feature_columns"]),
        n_trusts=len(metadata["trust_to_idx"]),
        n_specialties=len(metadata["specialty_to_idx"]),
        prediction_length=config.prediction_length,
        quantiles=QUANTILES,
        hidden_channels=config.hidden_channels,
        tcn_levels=config.tcn_levels,
        kernel_size=config.kernel_size,
        dropout=config.dropout,
        embedding_dim=config.embedding_dim,
    ).to(device)
    loss_fn = QuantileLoss(QUANTILES).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_val_loss = math.inf
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
    for epoch in range(1, config.max_epochs + 1):
        epoch_start = time.time()
        train_loss = run_epoch(model, loaders["train"], loss_fn, optimizer, device, config.gradient_clip_norm)
        val_loss = run_epoch(model, loaders["val"], loss_fn, None, device, config.gradient_clip_norm)
        scheduler.step(val_loss)
        epoch_seconds = time.time() - epoch_start
        now = time.strftime("%H:%M:%S")
        print(
            f"  [{now}] epoch {epoch:03d}/{config.max_epochs} | train_loss={train_loss:.6f} | "
            f"val_loss={val_loss:.6f} | {epoch_seconds:.1f}s this epoch"
        )
        if val_loss < best_val_loss - 1.0e-5:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.early_stopping_patience:
                print(f"  early stopping at epoch {epoch}. Best val_loss={best_val_loss:.6f}")
                break

    model.load_state_dict(best_state)
    test_predictions = predict_ablation_quantiles(model, datasets["test"], config)
    metrics = compute_ablation_metrics(test_predictions)
    metrics["best_val_loss"] = float(best_val_loss)
    return test_predictions, metrics

# %% Cell 8
clean_rtt = pd.read_parquet(PATHS.clean_parquet)

baseline_backtest = pd.read_parquet(PATHS.backtest_predictions)
baseline_predictions_for_metrics = pd.DataFrame(
    {
        "series_id": baseline_backtest[COLUMNS.trust_code].astype(str) + "__" + baseline_backtest[COLUMNS.specialty_code].astype(str),
        "horizon": baseline_backtest[COLUMNS.horizon],
        "actual": baseline_backtest[COLUMNS.actual],
        "p10": baseline_backtest[COLUMNS.p10],
        "p50": baseline_backtest[COLUMNS.p50],
        "p90": baseline_backtest[COLUMNS.p90],
    }
)
baseline_metrics = compute_ablation_metrics(baseline_predictions_for_metrics)
baseline_metrics["best_val_loss"] = float("nan")
print("Baseline (full model) metrics recomputed from the existing saved backtest predictions:")
print(json.dumps(baseline_metrics, indent=2))

# %% Cell 9
# Every variant's result is written to disk the moment it finishes, and re-running this cell
# skips any variant whose checkpoint already exists. If the run is interrupted (closed terminal,
# machine sleep, crash) partway through, nothing already completed is lost -- just re-run the
# script and it picks up from the next untrained variant instead of starting over.
CHECKPOINT_DIR = ABLATION_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def checkpoint_paths(variant_name: str) -> Tuple[Path, Path]:
    return (
        CHECKPOINT_DIR / f"{variant_name}_predictions.parquet",
        CHECKPOINT_DIR / f"{variant_name}_metrics.json",
    )


variant_predictions: Dict[str, pd.DataFrame] = {"full_model": baseline_predictions_for_metrics}
variant_metrics: Dict[str, Dict[str, float]] = {"full_model": baseline_metrics}
full_model_predictions_path, full_model_metrics_path = checkpoint_paths("full_model")
baseline_predictions_for_metrics.to_parquet(full_model_predictions_path, index=False)
with open(full_model_metrics_path, "w", encoding="utf-8") as handle:
    json.dump(baseline_metrics, handle, indent=2)

run_start = time.time()
for variant in ABLATION_VARIANTS:
    if variant["name"] == "full_model":
        continue

    predictions_path, metrics_path = checkpoint_paths(variant["name"])
    if predictions_path.exists() and metrics_path.exists():
        print(f"\n=== Skipping {variant['name']}: checkpoint already exists at {predictions_path} ===")
        variant_predictions[variant["name"]] = pd.read_parquet(predictions_path)
        with open(metrics_path, "r", encoding="utf-8") as handle:
            variant_metrics[variant["name"]] = json.load(handle)
        continue

    print(f"\n=== Training ablation variant: {variant['name']} ({variant['label']}) ===")
    print(f"  [{time.strftime('%H:%M:%S')}] started | {time.time() - run_start:.0f}s since this cell began")
    variant_start = time.time()
    set_random_seed(CONFIG.random_seed)
    feature_columns = [
        column for column in BASE_MODEL_INPUT_FEATURES if column not in variant["drop_features"]
    ]
    prepared, metadata = prepare_ablation_frame(clean_rtt, CONFIG, feature_columns)
    samples = make_supervised_samples(prepared, CONFIG, metadata["boundaries"])
    print(
        f"  features: {len(feature_columns)} (dropped {len(variant['drop_features'])}) | "
        f"train/val/test windows: {len(samples['train'])}/{len(samples['val'])}/{len(samples['test'])}"
    )
    predictions, metrics = train_ablation_variant(
        prepared=prepared,
        metadata=metadata,
        samples=samples,
        config=CONFIG,
        neutralise_entities=variant["neutralise_entities"],
    )
    variant_predictions[variant["name"]] = predictions
    variant_metrics[variant["name"]] = metrics

    predictions.to_parquet(predictions_path, index=False)
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    variant_minutes = (time.time() - variant_start) / 60.0
    print(f"  {variant['name']} metrics: {json.dumps(metrics, indent=2)}")
    print(f"  Checkpoint saved: {predictions_path}")
    print(f"  [{time.strftime('%H:%M:%S')}] finished {variant['name']} in {variant_minutes:.1f} minutes")

# %% Cell 10
comparison_rows = []
baseline_row_metrics = variant_metrics["full_model"]
for variant in ABLATION_VARIANTS:
    metrics = variant_metrics[variant["name"]]
    row = {
        "variant": variant["name"],
        "label": variant["label"],
        "description": variant["description"],
        "n_test_rows": metrics["n_rows"],
        "mae_median": metrics["mae_median"],
        "rmse_median": metrics["rmse_median"],
        "pinball_mean": metrics["pinball_mean"],
        "p10_p90_coverage": metrics["p10_p90_coverage"],
    }
    for metric_name in ["mae_median", "rmse_median", "pinball_mean"]:
        baseline_value = baseline_row_metrics[metric_name]
        variant_value = metrics[metric_name]
        row[f"{metric_name}_pct_change_vs_full_model"] = (
            100.0 * (variant_value - baseline_value) / baseline_value if baseline_value else float("nan")
        )
    comparison_rows.append(row)

ablation_comparison = pd.DataFrame(comparison_rows)
ablation_comparison_path = ABLATION_DIR / "ablation_study_comparison.csv"
ablation_comparison.to_csv(ablation_comparison_path, index=False)
print(f"Saved ablation comparison table to: {ablation_comparison_path}")
display(ablation_comparison)

# %% Cell 11
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(9, 5.5))
labels = ablation_comparison["label"].tolist()
mae_values = ablation_comparison["mae_median"].to_numpy(dtype=float)
colors = ["#08519c" if name == "full_model" else "#b2182b" for name in ablation_comparison["variant"]]
ax.barh(labels[::-1], mae_values[::-1], color=colors[::-1])
ax.set_xlabel("Median-forecast MAE on the held-out test period")
ax.set_title("Ablation Study: Effect of Removing Each Component on Forecast Accuracy")
ax.grid(axis="x", alpha=0.25)
fig.tight_layout()
ablation_chart_path = ABLATION_DIR / "ablation_study_mae_comparison.png"
fig.savefig(ablation_chart_path, dpi=150)
plt.close(fig)
print(f"Saved ablation comparison chart to: {ablation_chart_path}")

# %% Cell 12
methodology_note = f"""# Ablation Study Methodology

## Purpose

This experiment measures how much each of three model components contributes to forecast
accuracy, by retraining the TCN with that component removed and comparing it against the
full production model on the same held-out test period.

## Variants

{chr(10).join(f"- **{variant['label']}**: {variant['description']}" for variant in ABLATION_VARIANTS)}

## Protocol

Every variant uses the identical encoder length, forecast horizon, train/validation/test
split, model architecture (hidden channels, TCN levels, kernel size, dropout, embedding
dimension) and random seed as the production Layer 1 model. Only the single named component
differs, so any change in accuracy can be attributed to that component rather than to a
confound. The full-model row is not retrained; it is the existing saved production model's
backtest predictions, recomputed through the same metric functions used for every variant
so the comparison is on equal footing.

## Result

{ablation_comparison[["label", "mae_median", "mae_median_pct_change_vs_full_model"]].to_string(index=False)}
"""

ablation_methodology_path = ABLATION_DIR / "ablation_study_methodology_note.md"
with open(ablation_methodology_path, "w", encoding="utf-8") as handle:
    handle.write(methodology_note)
print(f"Saved methodology note to: {ablation_methodology_path}")

print("\nAblation study complete.")
print(f"Comparison table: {ablation_comparison_path}")
print(f"Chart: {ablation_chart_path}")
print(f"Methodology note: {ablation_methodology_path}")
