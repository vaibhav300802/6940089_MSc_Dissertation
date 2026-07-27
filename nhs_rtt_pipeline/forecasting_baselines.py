from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import COLUMNS, validate_backtest_predictions_frame


QUANTILES = (0.1, 0.5, 0.9)
CORE_BASELINE_MODELS = (
    "TCN",
    "naive_last_value",
    "seasonal_naive_12m",
    "historical_seasonal_mean",
)


@dataclass(frozen=True)
class BaselineComparisonConfig:
    target_column: Optional[str] = None
    seasonal_period: int = 12
    enable_hist_gradient_boosting: bool = True
    hist_gradient_boosting_max_train_rows: int = 350_000
    hist_gradient_boosting_max_iter: int = 180
    hist_gradient_boosting_learning_rate: float = 0.05
    hist_gradient_boosting_min_samples_leaf: int = 30
    random_seed: int = 42
    covid_start: str = "2020-03-01"
    covid_end: str = "2021-09-01"
    require_complete_core_baselines: bool = True
    expensive_baseline_name: str = "hist_gradient_boosting"


@dataclass(frozen=True)
class BaselineComparisonResults:
    predictions: pd.DataFrame
    model_comparison: pd.DataFrame
    by_horizon: pd.DataFrame
    by_specialty: pd.DataFrame
    by_trust_size: pd.DataFrame
    by_covid_period: pd.DataFrame
    paired_errors: pd.DataFrame
    audit_log: pd.DataFrame
    summary_text: str


def pinball_loss_np(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> np.ndarray:
    error = y_true - y_pred
    return np.maximum(quantile * error, (quantile - 1.0) * error)


def smape_np(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    values = np.full_like(y_true, np.nan, dtype=float)
    mask = denominator > 0.0
    values[mask] = 100.0 * np.abs(y_pred[mask] - y_true[mask]) / denominator[mask]
    return values


def choose_target_column(clean_frame: pd.DataFrame, feature_metadata: Mapping[str, Any], config: BaselineComparisonConfig) -> str:
    if config.target_column is not None:
        if config.target_column not in clean_frame.columns:
            raise ValueError(f"Configured target_column is missing from clean RTT data: {config.target_column}")
        return config.target_column
    metadata_target = feature_metadata.get("target_column")
    if metadata_target and metadata_target in clean_frame.columns:
        return str(metadata_target)
    for candidate in [COLUMNS.incomplete_total, "waiting_list"]:
        if candidate in clean_frame.columns:
            return candidate
    raise ValueError("Could not infer a baseline target column from clean RTT data.")


def prepare_target_frame(clean_frame: pd.DataFrame, target_column: str, backtest: pd.DataFrame) -> pd.DataFrame:
    required = [
        "month",
        COLUMNS.trust_code,
        COLUMNS.trust_name,
        COLUMNS.specialty_code,
        COLUMNS.specialty_name,
        "time_idx",
        target_column,
    ]
    missing = [column for column in required if column not in clean_frame.columns]
    if missing:
        raise ValueError(f"Clean RTT data is missing columns needed for baseline comparison: {missing}")
    frame = clean_frame.copy()
    frame["month"] = pd.to_datetime(frame["month"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    if frame["month"].isna().any():
        raise ValueError("Clean RTT data contains invalid month values.")
    for column in [COLUMNS.trust_code, COLUMNS.specialty_code]:
        frame[column] = frame[column].astype(str)
    if COLUMNS.series_id not in frame.columns:
        frame[COLUMNS.series_id] = frame[COLUMNS.trust_code].astype(str) + "__" + frame[COLUMNS.specialty_code].astype(str)
    frame[COLUMNS.series_id] = frame[COLUMNS.series_id].astype(str)
    frame["target_actual"] = pd.to_numeric(frame[target_column], errors="coerce").astype(float)

    included_pairs = backtest[[COLUMNS.trust_code, COLUMNS.specialty_code]].drop_duplicates().copy()
    included_pairs[COLUMNS.trust_code] = included_pairs[COLUMNS.trust_code].astype(str)
    included_pairs[COLUMNS.specialty_code] = included_pairs[COLUMNS.specialty_code].astype(str)
    frame = frame.merge(included_pairs, on=[COLUMNS.trust_code, COLUMNS.specialty_code], how="inner")
    if frame.empty:
        raise ValueError("No clean RTT rows match the Trust-specialty series in TCN backtest predictions.")
    return frame.sort_values([COLUMNS.series_id, "time_idx"]).reset_index(drop=True)


def add_series_id_to_backtest(backtest: pd.DataFrame, clean_frame: pd.DataFrame) -> pd.DataFrame:
    frame = validate_backtest_predictions_frame(backtest.copy(), "TCN backtest predictions")
    for column in [COLUMNS.trust_code, COLUMNS.specialty_code]:
        frame[column] = frame[column].astype(str)
    frame[COLUMNS.forecast_month] = pd.to_datetime(frame[COLUMNS.forecast_month]).dt.to_period("M").dt.to_timestamp()
    frame[COLUMNS.forecast_origin] = pd.to_datetime(frame[COLUMNS.forecast_origin]).dt.to_period("M").dt.to_timestamp()
    lookup = (
        clean_frame[[COLUMNS.trust_code, COLUMNS.specialty_code, COLUMNS.series_id]]
        .drop_duplicates()
        .copy()
    )
    frame = frame.merge(lookup, on=[COLUMNS.trust_code, COLUMNS.specialty_code], how="left")
    if frame[COLUMNS.series_id].isna().any():
        missing_count = int(frame[COLUMNS.series_id].isna().sum())
        raise ValueError(f"Could not map {missing_count} TCN backtest rows to clean-data series_id values.")
    return frame


def build_split_rows(
    target_frame: pd.DataFrame,
    feature_metadata: Mapping[str, Any],
    split: str,
    prediction_length: int,
    encoder_length: int,
    allowed_series_ids: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    boundaries = dict(feature_metadata.get("boundaries", {}))
    if not boundaries:
        raise ValueError("Feature metadata is missing split boundaries.")
    validation_start_idx = int(boundaries["validation_start_idx"])
    test_start_idx = int(boundaries["test_start_idx"])
    max_time_idx = int(boundaries["max_time_idx"])
    allowed = {str(value) for value in allowed_series_ids} if allowed_series_ids is not None else None
    rows: List[Dict[str, Any]] = []

    for series_id, group in target_frame.groupby(COLUMNS.series_id, sort=False, observed=True):
        if allowed is not None and str(series_id) not in allowed:
            continue
        group = group.sort_values("time_idx").reset_index(drop=True)
        if len(group) < encoder_length + prediction_length:
            continue
        time_values = group["time_idx"].to_numpy(dtype=int)
        if not np.all(np.diff(time_values) == 1):
            continue
        for encoder_end_pos in range(encoder_length - 1, len(group) - prediction_length):
            forecast_start_pos = encoder_end_pos + 1
            forecast_end_pos = encoder_end_pos + prediction_length
            if group.loc[forecast_start_pos:forecast_end_pos, "target_actual"].isna().any():
                continue
            forecast_start_idx = int(time_values[forecast_start_pos])
            forecast_end_idx = int(time_values[forecast_end_pos])
            is_train = forecast_end_idx < validation_start_idx
            is_val = forecast_start_idx >= validation_start_idx and forecast_end_idx < test_start_idx
            is_test = forecast_start_idx >= test_start_idx and forecast_end_idx <= max_time_idx
            if split == "train" and not is_train:
                continue
            if split == "validation" and not is_val:
                continue
            if split == "test" and not is_test:
                continue
            origin = group.iloc[encoder_end_pos]
            for horizon in range(1, prediction_length + 1):
                target_row = group.iloc[forecast_start_pos + horizon - 1]
                rows.append(
                    {
                        COLUMNS.series_id: str(series_id),
                        COLUMNS.trust_code: str(origin[COLUMNS.trust_code]),
                        COLUMNS.trust_name: origin[COLUMNS.trust_name],
                        COLUMNS.specialty_code: str(origin[COLUMNS.specialty_code]),
                        COLUMNS.specialty_name: origin[COLUMNS.specialty_name],
                        COLUMNS.forecast_origin: pd.to_datetime(origin["month"]),
                        COLUMNS.forecast_month: pd.to_datetime(target_row["month"]),
                        COLUMNS.horizon: int(horizon),
                        COLUMNS.actual: float(target_row["target_actual"]),
                        "origin_time_idx": int(origin["time_idx"]),
                        "target_time_idx": int(target_row["time_idx"]),
                    }
                )
    return pd.DataFrame(rows)


def tcn_prediction_rows(backtest_with_series: pd.DataFrame) -> pd.DataFrame:
    frame = backtest_with_series.copy()
    frame["model"] = "TCN"
    return frame[
        [
            "model",
            COLUMNS.series_id,
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


class TargetHistoryLookup:
    def __init__(self, target_frame: pd.DataFrame) -> None:
        self.by_series: Dict[str, pd.DataFrame] = {}
        self.by_series_month: Dict[str, Dict[pd.Timestamp, float]] = {}
        for series_id, group in target_frame.groupby(COLUMNS.series_id, sort=False, observed=True):
            ordered = group.sort_values("month").reset_index(drop=True)
            self.by_series[str(series_id)] = ordered
            self.by_series_month[str(series_id)] = {
                pd.Timestamp(row.month): float(row.target_actual)
                for row in ordered.itertuples(index=False)
                if pd.notna(row.target_actual)
            }

    def value_at_month(self, series_id: str, month: pd.Timestamp) -> float:
        return float(self.by_series_month.get(str(series_id), {}).get(pd.Timestamp(month), np.nan))

    def values_through_month(self, series_id: str, month: pd.Timestamp) -> pd.DataFrame:
        group = self.by_series.get(str(series_id))
        if group is None:
            return pd.DataFrame()
        return group[group["month"] <= pd.Timestamp(month)]

    def same_calendar_history(self, series_id: str, forecast_month: pd.Timestamp, origin_month: pd.Timestamp) -> pd.Series:
        history = self.values_through_month(series_id, origin_month)
        if history.empty:
            return pd.Series(dtype=float)
        month_number = pd.Timestamp(forecast_month).month
        return pd.to_numeric(history.loc[history["month"].dt.month.eq(month_number), "target_actual"], errors="coerce").dropna()


def deterministic_baseline_p50(rows: pd.DataFrame, lookup: TargetHistoryLookup, model_name: str, seasonal_period: int) -> pd.DataFrame:
    predictions = rows.copy()
    values = []
    reasons = []
    for row in predictions.itertuples(index=False):
        series_id = str(getattr(row, COLUMNS.series_id))
        origin_month = pd.Timestamp(getattr(row, COLUMNS.forecast_origin))
        forecast_month = pd.Timestamp(getattr(row, COLUMNS.forecast_month))
        if model_name == "naive_last_value":
            value = lookup.value_at_month(series_id, origin_month)
            reason = "" if np.isfinite(value) else "missing origin value"
        elif model_name == "seasonal_naive_12m":
            source_month = forecast_month - pd.DateOffset(months=int(seasonal_period))
            if source_month > origin_month:
                value = np.nan
                reason = "seasonal source month would be after forecast origin"
            else:
                value = lookup.value_at_month(series_id, source_month)
                reason = "" if np.isfinite(value) else "missing 12-month seasonal source value"
        elif model_name == "historical_seasonal_mean":
            same_month = lookup.same_calendar_history(series_id, forecast_month, origin_month)
            if len(same_month):
                value = float(same_month.mean())
                reason = ""
            else:
                history = lookup.values_through_month(series_id, origin_month)
                historical_values = pd.to_numeric(history.get("target_actual", pd.Series(dtype=float)), errors="coerce").dropna()
                value = float(historical_values.mean()) if len(historical_values) else np.nan
                reason = "used expanding historical mean fallback" if np.isfinite(value) else "missing historical values"
        else:
            raise ValueError(f"Unsupported deterministic baseline: {model_name}")
        values.append(value)
        reasons.append(reason)
    predictions["model"] = model_name
    predictions[COLUMNS.p50] = np.asarray(values, dtype=float)
    predictions["baseline_audit_reason"] = reasons
    return predictions


def calibrate_interval_offsets(validation_predictions: pd.DataFrame) -> pd.DataFrame:
    usable = validation_predictions.dropna(subset=[COLUMNS.actual, COLUMNS.p50]).copy()
    if usable.empty:
        raise ValueError("No validation rows are available for deterministic baseline interval calibration.")
    usable["residual"] = usable[COLUMNS.actual].astype(float) - usable[COLUMNS.p50].astype(float)
    rows = []
    for (model, horizon), group in usable.groupby(["model", COLUMNS.horizon], observed=True):
        residuals = group["residual"].dropna()
        if residuals.empty:
            continue
        rows.append(
            {
                "model": model,
                COLUMNS.horizon: int(horizon),
                "residual_q10": float(residuals.quantile(0.1)),
                "residual_q90": float(residuals.quantile(0.9)),
            }
        )
    global_rows = pd.DataFrame(
        [
            {
                "model": model,
                "global_residual_q10": float(group["residual"].dropna().quantile(0.1)),
                "global_residual_q90": float(group["residual"].dropna().quantile(0.9)),
            }
            for model, group in usable.groupby("model", observed=True)
            if not group["residual"].dropna().empty
        ]
    )
    offsets = pd.DataFrame(rows)
    if offsets.empty:
        raise ValueError("No horizon-level validation residual offsets could be calculated.")
    return offsets.merge(global_rows, on="model", how="left")


def apply_interval_offsets(test_predictions: pd.DataFrame, offsets: pd.DataFrame) -> pd.DataFrame:
    frame = test_predictions.merge(offsets, on=["model", COLUMNS.horizon], how="left")
    for low_column, global_column in [
        ("residual_q10", "global_residual_q10"),
        ("residual_q90", "global_residual_q90"),
    ]:
        frame[low_column] = pd.to_numeric(frame[low_column], errors="coerce")
        frame[global_column] = pd.to_numeric(frame[global_column], errors="coerce")
        frame[low_column] = frame[low_column].fillna(frame[global_column]).fillna(0.0)
    frame[COLUMNS.p10] = frame[COLUMNS.p50].astype(float) + frame["residual_q10"].astype(float)
    frame[COLUMNS.p90] = frame[COLUMNS.p50].astype(float) + frame["residual_q90"].astype(float)
    quantiles = np.sort(frame[[COLUMNS.p10, COLUMNS.p50, COLUMNS.p90]].to_numpy(dtype=float), axis=1)
    frame[[COLUMNS.p10, COLUMNS.p50, COLUMNS.p90]] = np.clip(quantiles, 0.0, None)
    return frame.drop(columns=[column for column in ["residual_q10", "residual_q90", "global_residual_q10", "global_residual_q90"] if column in frame.columns])


def build_ml_feature_frame(rows: pd.DataFrame, lookup: TargetHistoryLookup, seasonal_period: int) -> pd.DataFrame:
    feature_rows = []
    for row in rows.itertuples(index=False):
        series_id = str(getattr(row, COLUMNS.series_id))
        origin_month = pd.Timestamp(getattr(row, COLUMNS.forecast_origin))
        forecast_month = pd.Timestamp(getattr(row, COLUMNS.forecast_month))
        history = lookup.values_through_month(series_id, origin_month).sort_values("month").reset_index(drop=True)
        target_values = pd.to_numeric(history["target_actual"], errors="coerce").dropna().to_numpy(dtype=float)
        record: Dict[str, Any] = {
            COLUMNS.series_id: series_id,
            COLUMNS.trust_code: str(getattr(row, COLUMNS.trust_code)),
            COLUMNS.trust_name: getattr(row, COLUMNS.trust_name),
            COLUMNS.specialty_code: str(getattr(row, COLUMNS.specialty_code)),
            COLUMNS.specialty_name: getattr(row, COLUMNS.specialty_name),
            COLUMNS.forecast_origin: origin_month,
            COLUMNS.forecast_month: forecast_month,
            COLUMNS.horizon: int(getattr(row, COLUMNS.horizon)),
            COLUMNS.actual: float(getattr(row, COLUMNS.actual)),
            "forecast_month_sin": math.sin(2.0 * math.pi * forecast_month.month / 12.0),
            "forecast_month_cos": math.cos(2.0 * math.pi * forecast_month.month / 12.0),
            "origin_time_idx": float(getattr(row, "origin_time_idx", np.nan)),
        }
        for lag in [0, 1, 2, 3, 6, 12]:
            record[f"target_lag_{lag}"] = float(target_values[-1 - lag]) if len(target_values) > lag else np.nan
        for window in [3, 6, 12]:
            if len(target_values):
                tail = target_values[-window:]
                record[f"rolling_mean_{window}"] = float(np.mean(tail))
                record[f"rolling_std_{window}"] = float(np.std(tail))
            else:
                record[f"rolling_mean_{window}"] = np.nan
                record[f"rolling_std_{window}"] = np.nan
        seasonal_source_month = forecast_month - pd.DateOffset(months=int(seasonal_period))
        record["seasonal_lag_for_forecast_month"] = (
            lookup.value_at_month(series_id, seasonal_source_month)
            if seasonal_source_month <= origin_month
            else np.nan
        )
        same_month = lookup.same_calendar_history(series_id, forecast_month, origin_month)
        record["historical_same_month_mean"] = float(same_month.mean()) if len(same_month) else np.nan
        feature_rows.append(record)
    return pd.DataFrame(feature_rows)


def fit_hist_gradient_boosting_baseline(
    train_rows: pd.DataFrame,
    predict_rows: pd.DataFrame,
    lookup: TargetHistoryLookup,
    config: BaselineComparisonConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
    except Exception as exc:
        raise ImportError(
            "scikit-learn is required for the HistGradientBoostingRegressor baseline. "
            "Install scikit-learn or set enable_hist_gradient_boosting=False."
        ) from exc

    train_features = build_ml_feature_frame(train_rows, lookup, config.seasonal_period)
    predict_features = build_ml_feature_frame(predict_rows, lookup, config.seasonal_period)
    if train_features.empty or predict_features.empty:
        raise ValueError("No rows are available for HistGradientBoostingRegressor baseline.")

    rng = np.random.default_rng(config.random_seed)
    if len(train_features) > config.hist_gradient_boosting_max_train_rows:
        selected = rng.choice(len(train_features), size=config.hist_gradient_boosting_max_train_rows, replace=False)
        train_features = train_features.iloc[np.sort(selected)].reset_index(drop=True)

    trust_codes = sorted(train_features[COLUMNS.trust_code].astype(str).unique())
    specialty_codes = sorted(train_features[COLUMNS.specialty_code].astype(str).unique())
    trust_to_idx = {code: idx for idx, code in enumerate(trust_codes)}
    specialty_to_idx = {code: idx for idx, code in enumerate(specialty_codes)}
    train_features["trust_code_idx"] = train_features[COLUMNS.trust_code].astype(str).map(trust_to_idx).fillna(-1).astype(float)
    train_features["specialty_code_idx"] = train_features[COLUMNS.specialty_code].astype(str).map(specialty_to_idx).fillna(-1).astype(float)
    predict_features["trust_code_idx"] = predict_features[COLUMNS.trust_code].astype(str).map(trust_to_idx).fillna(-1).astype(float)
    predict_features["specialty_code_idx"] = predict_features[COLUMNS.specialty_code].astype(str).map(specialty_to_idx).fillna(-1).astype(float)

    feature_columns = [
        COLUMNS.horizon,
        "origin_time_idx",
        "forecast_month_sin",
        "forecast_month_cos",
        "trust_code_idx",
        "specialty_code_idx",
        "target_lag_0",
        "target_lag_1",
        "target_lag_2",
        "target_lag_3",
        "target_lag_6",
        "target_lag_12",
        "rolling_mean_3",
        "rolling_std_3",
        "rolling_mean_6",
        "rolling_std_6",
        "rolling_mean_12",
        "rolling_std_12",
        "seasonal_lag_for_forecast_month",
        "historical_same_month_mean",
    ]
    X_train = train_features[feature_columns].to_numpy(dtype=float)
    X_predict = predict_features[feature_columns].to_numpy(dtype=float)
    medians = np.nanmedian(X_train, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    X_train = np.where(np.isfinite(X_train), X_train, medians)
    X_predict = np.where(np.isfinite(X_predict), X_predict, medians)
    y_train = np.log1p(np.clip(train_features[COLUMNS.actual].to_numpy(dtype=float), 0.0, None))

    prediction_columns: Dict[float, np.ndarray] = {}
    for quantile in QUANTILES:
        model = HistGradientBoostingRegressor(
            loss="quantile",
            quantile=float(quantile),
            max_iter=int(config.hist_gradient_boosting_max_iter),
            learning_rate=float(config.hist_gradient_boosting_learning_rate),
            min_samples_leaf=int(config.hist_gradient_boosting_min_samples_leaf),
            random_state=int(config.random_seed),
        )
        model.fit(X_train, y_train)
        prediction_columns[quantile] = np.expm1(np.maximum(model.predict(X_predict), 0.0))

    output = predict_features[
        [
            COLUMNS.series_id,
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
    output["model"] = config.expensive_baseline_name
    output[COLUMNS.p10] = prediction_columns[0.1]
    output[COLUMNS.p50] = prediction_columns[0.5]
    output[COLUMNS.p90] = prediction_columns[0.9]
    sorted_quantiles = np.sort(output[[COLUMNS.p10, COLUMNS.p50, COLUMNS.p90]].to_numpy(dtype=float), axis=1)
    output[[COLUMNS.p10, COLUMNS.p50, COLUMNS.p90]] = np.clip(sorted_quantiles, 0.0, None)
    output["baseline_audit_reason"] = ""
    audit = pd.DataFrame(
        [
            {
                "model": config.expensive_baseline_name,
                "status": "trained",
                "training_rows": int(len(train_features)),
                "prediction_rows": int(len(output)),
                "feature_count": int(len(feature_columns)),
                "note": "HistGradientBoostingRegressor quantile models trained on training-period rows only.",
            }
        ]
    )
    return output, audit


def prediction_key_columns() -> List[str]:
    return [
        COLUMNS.series_id,
        COLUMNS.trust_code,
        COLUMNS.specialty_code,
        COLUMNS.forecast_origin,
        COLUMNS.forecast_month,
        COLUMNS.horizon,
    ]


def assert_no_duplicate_prediction_keys(predictions: pd.DataFrame, label: str = "model comparison predictions") -> None:
    keys = ["model", *prediction_key_columns()]
    missing = [column for column in keys if column not in predictions.columns]
    if missing:
        raise ValueError(f"{label} is missing key columns needed for alignment: {missing}")

    duplicated = predictions[predictions.duplicated(keys, keep=False)]
    if not duplicated.empty:
        examples = (
            duplicated[keys]
            .sort_values(keys)
            .head(10)
            .to_dict(orient="records")
        )
        raise ValueError(
            f"{label} contains duplicate rows for the same model and forecast key. "
            f"Resolve duplicate forecast rows before calculating model metrics. Examples: {examples}"
        )


def align_prediction_rows(predictions: pd.DataFrame, required_models: Sequence[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    keys = prediction_key_columns()
    audit_rows = []
    working = predictions.copy()
    for column in [COLUMNS.forecast_origin, COLUMNS.forecast_month]:
        working[column] = pd.to_datetime(working[column]).dt.to_period("M").dt.to_timestamp()
    valid = working.dropna(subset=[COLUMNS.p10, COLUMNS.p50, COLUMNS.p90, COLUMNS.actual]).copy()
    assert_no_duplicate_prediction_keys(valid)
    for model in sorted(working["model"].astype(str).unique()):
        missing = int(len(working[working["model"].eq(model)]) - len(valid[valid["model"].eq(model)]))
        if missing:
            audit_rows.append({"model": model, "status": "excluded_rows", "rows": missing, "reason": "missing prediction or actual"})
    required_models = list(dict.fromkeys(str(model) for model in required_models))
    required_model_set = set(required_models)
    unexpected_required_missing = sorted(required_model_set - set(valid["model"].astype(str).unique()))
    if unexpected_required_missing:
        raise ValueError(f"Required model outputs are missing before alignment: {unexpected_required_missing}")

    valid = valid[valid["model"].astype(str).isin(required_model_set)].copy()
    model_key_counts = valid.groupby(keys, observed=True)["model"].nunique().reset_index(name="model_count")
    required_count = len(required_model_set)
    complete_keys = model_key_counts[model_key_counts["model_count"] == required_count][keys]
    aligned = valid.merge(complete_keys, on=keys, how="inner")
    for model in required_models:
        original_keys = valid[valid["model"].eq(model)][keys].drop_duplicates()
        aligned_keys = aligned[aligned["model"].eq(model)][keys].drop_duplicates()
        dropped = len(original_keys) - len(aligned_keys)
        if dropped:
            audit_rows.append({"model": model, "status": "excluded_keys", "rows": int(dropped), "reason": "not all required models available"})
    if aligned.empty:
        raise ValueError("No common prediction rows remain after aligning model comparison outputs.")
    aligned_key_counts = aligned.groupby("model", observed=True).size()
    if aligned_key_counts.nunique() != 1:
        raise ValueError(
            "Aligned model comparison rows are not balanced across models: "
            f"{aligned_key_counts.to_dict()}"
        )
    expected_rows = int(len(complete_keys))
    for model, row_count in aligned_key_counts.items():
        if int(row_count) != expected_rows:
            raise ValueError(
                f"Model {model} has {row_count} aligned rows but expected {expected_rows}."
            )
    return aligned.reset_index(drop=True), pd.DataFrame(audit_rows)


def metric_row(frame: pd.DataFrame, group_values: Mapping[str, Any]) -> Dict[str, Any]:
    y = frame[COLUMNS.actual].to_numpy(dtype=float)
    p10 = frame[COLUMNS.p10].to_numpy(dtype=float)
    p50 = frame[COLUMNS.p50].to_numpy(dtype=float)
    p90 = frame[COLUMNS.p90].to_numpy(dtype=float)
    errors = p50 - y
    abs_errors = np.abs(errors)
    row: Dict[str, Any] = dict(group_values)
    row.update(
        {
            "n_rows": int(len(frame)),
            "mae": float(np.mean(abs_errors)),
            "rmse": float(np.sqrt(np.mean(errors**2))),
            "smape": float(np.nanmean(smape_np(y, p50))),
            "wape": float(100.0 * np.sum(abs_errors) / np.sum(np.abs(y))) if np.sum(np.abs(y)) > 0 else np.nan,
            "pinball_q10": float(np.mean(pinball_loss_np(y, p10, 0.1))),
            "pinball_q50": float(np.mean(pinball_loss_np(y, p50, 0.5))),
            "pinball_q90": float(np.mean(pinball_loss_np(y, p90, 0.9))),
            "pinball_mean": float(np.mean(np.vstack([
                pinball_loss_np(y, p10, 0.1),
                pinball_loss_np(y, p50, 0.5),
                pinball_loss_np(y, p90, 0.9),
            ]))),
            "p10_p90_coverage": float(np.mean((y >= p10) & (y <= p90))),
            "average_interval_width": float(np.mean(p90 - p10)),
        }
    )
    return row


def metrics_by_group(predictions: pd.DataFrame, group_columns: Sequence[str]) -> pd.DataFrame:
    rows = []
    for group_key, group in predictions.groupby(list(group_columns), observed=True):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        rows.append(metric_row(group, dict(zip(group_columns, group_key))))
    return pd.DataFrame(rows)


def add_trust_size_groups(predictions: pd.DataFrame) -> pd.DataFrame:
    frame = predictions.copy()
    row_count_before = len(frame)
    tcn = frame[frame["model"].eq("TCN")]
    trust_sizes = (
        tcn.groupby(COLUMNS.trust_code, as_index=False, observed=True)[COLUMNS.actual]
        .sum()
        .rename(columns={COLUMNS.actual: "trust_test_actual_total"})
    )
    if trust_sizes["trust_test_actual_total"].nunique() >= 3:
        trust_sizes["trust_size_group"] = pd.qcut(
            trust_sizes["trust_test_actual_total"],
            q=3,
            labels=["small", "medium", "large"],
            duplicates="drop",
        ).astype(str)
    else:
        trust_sizes["trust_size_group"] = "all"
    merged = frame.merge(trust_sizes[[COLUMNS.trust_code, "trust_size_group"]], on=COLUMNS.trust_code, how="left")
    if len(merged) != row_count_before:
        raise ValueError(
            "Trust-size grouping changed the number of model-comparison rows. "
            f"Before={row_count_before}, after={len(merged)}."
        )
    return merged


def add_covid_period_labels(predictions: pd.DataFrame, config: BaselineComparisonConfig) -> pd.DataFrame:
    frame = predictions.copy()
    start = pd.Timestamp(config.covid_start)
    end = pd.Timestamp(config.covid_end)
    month = pd.to_datetime(frame[COLUMNS.forecast_month])
    frame["covid_period"] = np.where((month >= start) & (month <= end), "covid_period", "non_covid_period")
    return frame


def sign_test_two_sided(successes: int, trials: int) -> float:
    if trials <= 0:
        return np.nan
    successes = int(max(0, min(successes, trials)))
    trials = int(trials)
    try:
        from scipy.stats import binomtest

        return float(binomtest(successes, trials, p=0.5, alternative="two-sided").pvalue)
    except Exception:
        pass

    k = min(successes, trials - successes)
    if trials > 1000:
        # Normal approximation with continuity correction for very large samples.
        z = (abs(successes - (trials / 2.0)) - 0.5) / math.sqrt(trials * 0.25)
        return float(min(1.0, max(0.0, math.erfc(max(0.0, z) / math.sqrt(2.0)))))

    log_terms = [
        math.lgamma(trials + 1)
        - math.lgamma(i + 1)
        - math.lgamma(trials - i + 1)
        - (trials * math.log(2.0))
        for i in range(k + 1)
    ]
    max_log = max(log_terms)
    probability = math.exp(max_log) * sum(math.exp(term - max_log) for term in log_terms)
    return float(min(1.0, 2.0 * probability))


def paired_error_analysis(predictions: pd.DataFrame, comparator_models: Sequence[str]) -> pd.DataFrame:
    keys = prediction_key_columns()
    tcn = predictions[predictions["model"].eq("TCN")][keys + [COLUMNS.actual, COLUMNS.p50]].rename(
        columns={COLUMNS.p50: "tcn_p50"}
    )
    rows = []
    for comparator in comparator_models:
        other = predictions[predictions["model"].eq(comparator)][keys + [COLUMNS.p50]].rename(
            columns={COLUMNS.p50: "comparator_p50"}
        )
        merged = tcn.merge(other, on=keys, how="inner")
        if merged.empty:
            continue
        tcn_abs = (merged["tcn_p50"] - merged[COLUMNS.actual]).abs()
        comparator_abs = (merged["comparator_p50"] - merged[COLUMNS.actual]).abs()
        diff = comparator_abs - tcn_abs
        wins = int((diff > 0).sum())
        losses = int((diff < 0).sum())
        ties = int((diff == 0).sum())
        rows.append(
            {
                "comparison": f"TCN_vs_{comparator}",
                "n_pairs": int(len(merged)),
                "mean_abs_error_difference_comparator_minus_tcn": float(diff.mean()),
                "median_abs_error_difference_comparator_minus_tcn": float(diff.median()),
                "tcn_better_pair_count": wins,
                "tcn_worse_pair_count": losses,
                "tie_count": ties,
                "tcn_better_pair_pct": float(100.0 * wins / len(merged)),
                "two_sided_sign_test_p_value": sign_test_two_sided(wins, wins + losses),
                "interpretation": (
                    "TCN has lower absolute error on average"
                    if float(diff.mean()) > 0
                    else "Comparator has lower absolute error on average"
                    if float(diff.mean()) < 0
                    else "Mean paired absolute error is tied"
                ),
            }
        )
    return pd.DataFrame(rows)


def build_summary_text(
    model_comparison: pd.DataFrame,
    paired_errors: pd.DataFrame,
    config: BaselineComparisonConfig,
) -> str:
    ranked = model_comparison.sort_values("mae", ascending=True).reset_index(drop=True)
    strongest = ranked.iloc[0]
    seasonal = ranked[ranked["model"].eq("seasonal_naive_12m")]
    tcn = ranked[ranked["model"].eq("TCN")]
    lines = [
        "# Model Comparison Summary",
        "",
        f"Configuration: `{json.dumps(asdict(config), sort_keys=True)}`",
        "",
        f"Strongest model by overall MAE: **{strongest['model']}** with MAE {strongest['mae']:.4f}.",
    ]
    if not seasonal.empty and not tcn.empty:
        seasonal_mae = float(seasonal["mae"].iloc[0])
        tcn_mae = float(tcn["mae"].iloc[0])
        difference = seasonal_mae - tcn_mae
        if difference > 0:
            lines.append(
                f"TCN outperforms seasonal naive by {difference:.4f} MAE "
                f"({100.0 * difference / seasonal_mae:.2f}% relative improvement)."
            )
        elif difference < 0:
            lines.append(
                f"TCN does not outperform seasonal naive on overall MAE; seasonal naive is better by {-difference:.4f} MAE."
            )
        else:
            lines.append("TCN and seasonal naive are tied on overall MAE.")
    if not paired_errors.empty:
        lines.extend(["", "## Paired Error Analysis", ""])
        for row in paired_errors.itertuples(index=False):
            lines.append(
                f"- {row.comparison}: mean comparator-minus-TCN absolute-error difference "
                f"{row.mean_abs_error_difference_comparator_minus_tcn:.4f}; "
                f"TCN better on {row.tcn_better_pair_pct:.2f}% of paired rows; "
                f"sign-test p-value {row.two_sided_sign_test_p_value:.4g}."
            )
    lines.append("")
    lines.append("The TCN should only be described as superior for groups or horizons where these metrics support it.")
    return "\n".join(lines)


def run_baseline_comparison(
    clean_frame: pd.DataFrame,
    backtest_predictions: pd.DataFrame,
    feature_metadata: Mapping[str, Any],
    config: Optional[BaselineComparisonConfig] = None,
) -> BaselineComparisonResults:
    config = config or BaselineComparisonConfig()
    tcn_backtest = validate_backtest_predictions_frame(backtest_predictions.copy(), "TCN backtest predictions")
    target_column = choose_target_column(clean_frame, feature_metadata, config)
    prediction_length = int(feature_metadata.get("config", {}).get("prediction_length", int(tcn_backtest[COLUMNS.horizon].max())))
    encoder_length = int(feature_metadata.get("config", {}).get("encoder_length", 24))
    target_frame = prepare_target_frame(clean_frame, target_column, tcn_backtest)
    tcn_backtest = add_series_id_to_backtest(tcn_backtest, target_frame)
    allowed_series_ids = set(tcn_backtest[COLUMNS.series_id].astype(str).unique())

    validation_rows = build_split_rows(
        target_frame,
        feature_metadata,
        split="validation",
        prediction_length=prediction_length,
        encoder_length=encoder_length,
        allowed_series_ids=allowed_series_ids,
    )
    train_rows = build_split_rows(
        target_frame,
        feature_metadata,
        split="train",
        prediction_length=prediction_length,
        encoder_length=encoder_length,
        allowed_series_ids=allowed_series_ids,
    )
    test_rows = build_split_rows(
        target_frame,
        feature_metadata,
        split="test",
        prediction_length=prediction_length,
        encoder_length=encoder_length,
        allowed_series_ids=allowed_series_ids,
    )
    if validation_rows.empty or train_rows.empty or test_rows.empty:
        raise ValueError("Train, validation, and test rows are all required for baseline comparison.")

    keys = prediction_key_columns()
    test_rows = tcn_backtest[keys].drop_duplicates().merge(test_rows, on=keys, how="inner")
    if len(test_rows) != len(tcn_backtest):
        raise ValueError(
            "Generated baseline test rows do not exactly match the TCN backtest rows. "
            f"TCN rows={len(tcn_backtest)}, baseline rows={len(test_rows)}"
        )

    lookup = TargetHistoryLookup(target_frame)
    validation_deterministic = []
    test_deterministic = []
    for model_name in ["naive_last_value", "seasonal_naive_12m", "historical_seasonal_mean"]:
        validation_deterministic.append(
            deterministic_baseline_p50(validation_rows, lookup, model_name, config.seasonal_period)
        )
        test_deterministic.append(
            deterministic_baseline_p50(test_rows, lookup, model_name, config.seasonal_period)
        )
    validation_det = pd.concat(validation_deterministic, ignore_index=True)
    offsets = calibrate_interval_offsets(validation_det)
    test_det = apply_interval_offsets(pd.concat(test_deterministic, ignore_index=True), offsets)

    prediction_frames = [tcn_prediction_rows(tcn_backtest), test_det]
    audit_frames = []
    if config.enable_hist_gradient_boosting:
        ml_predictions, ml_audit = fit_hist_gradient_boosting_baseline(
            train_rows=train_rows,
            predict_rows=test_rows,
            lookup=lookup,
            config=config,
        )
        prediction_frames.append(ml_predictions)
        audit_frames.append(ml_audit)
    else:
        audit_frames.append(
            pd.DataFrame(
                [
                    {
                        "model": config.expensive_baseline_name,
                        "status": "disabled",
                        "training_rows": 0,
                        "prediction_rows": 0,
                        "feature_count": 0,
                        "note": "Expensive ML baseline disabled by configuration.",
                    }
                ]
            )
        )

    all_predictions = pd.concat(prediction_frames, ignore_index=True)
    all_predictions["baseline_audit_reason"] = all_predictions.get("baseline_audit_reason", "").fillna("")
    required_models = list(CORE_BASELINE_MODELS)
    if config.enable_hist_gradient_boosting:
        required_models.append(config.expensive_baseline_name)
    aligned_predictions, alignment_audit = align_prediction_rows(all_predictions, required_models)
    if config.require_complete_core_baselines:
        missing_core = sorted(set(CORE_BASELINE_MODELS) - set(aligned_predictions["model"].unique()))
        if missing_core:
            raise ValueError(f"Missing required core baseline model outputs after alignment: {missing_core}")

    aligned_predictions = add_trust_size_groups(aligned_predictions)
    aligned_predictions = add_covid_period_labels(aligned_predictions, config)
    model_comparison = metrics_by_group(aligned_predictions, ["model"]).sort_values("mae").reset_index(drop=True)
    by_horizon = metrics_by_group(aligned_predictions, ["model", COLUMNS.horizon]).sort_values(
        ["horizon", "mae"]
    ).reset_index(drop=True)
    by_specialty = metrics_by_group(
        aligned_predictions,
        ["model", COLUMNS.specialty_code, COLUMNS.specialty_name],
    ).sort_values([COLUMNS.specialty_name, "mae"]).reset_index(drop=True)
    by_trust_size = metrics_by_group(aligned_predictions, ["model", "trust_size_group"]).sort_values(
        ["trust_size_group", "mae"]
    ).reset_index(drop=True)
    by_covid_period = metrics_by_group(aligned_predictions, ["model", "covid_period"]).sort_values(
        ["covid_period", "mae"]
    ).reset_index(drop=True)
    comparator_models = [model for model in aligned_predictions["model"].unique() if model != "TCN"]
    paired_errors = paired_error_analysis(aligned_predictions, comparator_models)
    audit_log = pd.concat([alignment_audit, *audit_frames], ignore_index=True) if audit_frames else alignment_audit
    summary_text = build_summary_text(model_comparison, paired_errors, config)
    return BaselineComparisonResults(
        predictions=aligned_predictions.reset_index(drop=True),
        model_comparison=model_comparison,
        by_horizon=by_horizon,
        by_specialty=by_specialty,
        by_trust_size=by_trust_size,
        by_covid_period=by_covid_period,
        paired_errors=paired_errors,
        audit_log=audit_log,
        summary_text=summary_text,
    )


def save_baseline_comparison_outputs(
    results: BaselineComparisonResults,
    paths: Mapping[str, Path],
) -> None:
    def with_model_name(frame: pd.DataFrame) -> pd.DataFrame:
        output = frame.copy()
        if "model" in output.columns and "model_name" not in output.columns:
            model_position = output.columns.get_loc("model")
            output.insert(model_position + 1, "model_name", output["model"])
        return output

    paths["predictions"].parent.mkdir(parents=True, exist_ok=True)
    with_model_name(results.predictions).to_parquet(paths["predictions"], index=False)
    with_model_name(results.model_comparison).to_csv(paths["overall"], index=False)
    with_model_name(results.by_horizon).to_csv(paths["by_horizon"], index=False)
    with_model_name(results.by_specialty).to_csv(paths["by_specialty"], index=False)
    with_model_name(results.by_trust_size).to_csv(paths["by_trust_size"], index=False)
    with_model_name(results.by_covid_period).to_csv(paths["by_covid_period"], index=False)
    with_model_name(results.paired_errors).to_csv(paths["paired_errors"], index=False)
    with_model_name(results.audit_log).to_csv(paths["audit_log"], index=False)
    paths["summary"].write_text(results.summary_text, encoding="utf-8")


def save_model_comparison_plots(
    results: BaselineComparisonResults,
    overall_png: str | Path,
    by_horizon_png: str | Path,
    tcn_vs_seasonal_png: str | Path,
) -> None:
    import matplotlib.pyplot as plt

    overall_path = Path(overall_png)
    horizon_path = Path(by_horizon_png)
    seasonal_path = Path(tcn_vs_seasonal_png)
    overall_path.parent.mkdir(parents=True, exist_ok=True)

    overall = results.model_comparison.sort_values("mae", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(overall["model"], overall["mae"], color="#2563eb")
    ax.set_title("Overall Forecast MAE by Model")
    ax.set_xlabel("Model")
    ax.set_ylabel("MAE")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(overall_path, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6))
    for model_name, group in results.by_horizon.groupby("model", observed=True):
        group = group.sort_values(COLUMNS.horizon)
        ax.plot(group[COLUMNS.horizon], group["mae"], marker="o", linewidth=2, label=str(model_name))
    ax.set_title("Forecast MAE by Horizon")
    ax.set_xlabel("Forecast horizon (months)")
    ax.set_ylabel("MAE")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(horizon_path, dpi=180)
    plt.close(fig)

    keys = prediction_key_columns()
    tcn = results.predictions[results.predictions["model"].eq("TCN")][keys + [COLUMNS.actual, COLUMNS.p50]].rename(
        columns={COLUMNS.p50: "TCN"}
    )
    seasonal = results.predictions[results.predictions["model"].eq("seasonal_naive_12m")][keys + [COLUMNS.p50]].rename(
        columns={COLUMNS.p50: "seasonal_naive_12m"}
    )
    paired = tcn.merge(seasonal, on=keys, how="inner")
    paired["tcn_abs_error"] = (paired["TCN"] - paired[COLUMNS.actual]).abs()
    paired["seasonal_abs_error"] = (paired["seasonal_naive_12m"] - paired[COLUMNS.actual]).abs()
    paired["error_difference"] = paired["seasonal_abs_error"] - paired["tcn_abs_error"]
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.axhline(0.0, color="#111111", linewidth=1.0)
    for horizon, group in paired.groupby(COLUMNS.horizon, observed=True):
        ax.scatter(
            np.full(len(group), int(horizon)),
            group["error_difference"],
            alpha=0.35,
            s=16,
            label=f"h{int(horizon)}" if int(horizon) in {1, 6, 12} else None,
        )
    summary = paired.groupby(COLUMNS.horizon, as_index=False, observed=True)["error_difference"].mean()
    ax.plot(summary[COLUMNS.horizon], summary["error_difference"], color="#b91c1c", marker="o", linewidth=2.4, label="Mean difference")
    ax.set_title("Paired Absolute Error Difference: Seasonal Naive minus TCN")
    ax.set_xlabel("Forecast horizon (months)")
    ax.set_ylabel("Positive values mean TCN has lower absolute error")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(seasonal_path, dpi=180)
    plt.close(fig)
