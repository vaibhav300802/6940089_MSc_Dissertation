from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .config import COLUMNS, ROLLING_ORIGIN_PREDICTION_COLUMNS, require_columns


@dataclass(frozen=True)
class RollingOriginConfig:
    target_column: str = COLUMNS.incomplete_total
    model_name: str = "TCN"
    forecast_horizon: int = 12
    origin_step_months: int = 6
    requested_origins: int = 3
    min_train_months: int = 36
    encoder_length: int = 24
    internal_validation_months: int = 12
    quantiles: tuple[float, float, float] = (0.1, 0.5, 0.9)
    batch_size: int = 512
    max_epochs: int = 18
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
    nominal_interval_alpha: float = 0.20
    close_to_nominal_tolerance: float = 0.05


def month_start(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.to_period("M").dt.to_timestamp()


def month_difference(later: pd.Series, earlier: pd.Series) -> pd.Series:
    later_dt = pd.to_datetime(later)
    earlier_dt = pd.to_datetime(earlier)
    return ((later_dt.dt.year - earlier_dt.dt.year) * 12 + (later_dt.dt.month - earlier_dt.dt.month)).astype(int)


def select_rolling_origins(
    frame: pd.DataFrame,
    config: RollingOriginConfig,
    month_column: str = "month",
) -> list[pd.Timestamp]:
    require_columns(frame, [month_column], "rolling-origin input frame")
    months = pd.Series(month_start(frame[month_column]).dropna().unique()).sort_values().reset_index(drop=True)
    if months.empty:
        raise ValueError("No valid monthly dates are available for rolling-origin validation.")

    min_month = pd.Timestamp(months.iloc[0])
    max_month = pd.Timestamp(months.iloc[-1])
    effective_min_train_months = max(
        int(config.min_train_months),
        int(config.encoder_length) + int(config.forecast_horizon) + int(config.internal_validation_months),
    )
    latest_origin = max_month - pd.DateOffset(months=int(config.forecast_horizon))
    earliest_origin = min_month + pd.DateOffset(months=effective_min_train_months - 1)
    if latest_origin < earliest_origin:
        raise ValueError(
            "Not enough monthly coverage for rolling-origin validation. "
            f"First month={min_month.date()}, final month={max_month.date()}, "
            f"effective_min_train_months={effective_min_train_months}, horizon={config.forecast_horizon}."
        )

    candidates: list[pd.Timestamp] = []
    current = latest_origin
    while current >= earliest_origin:
        if current in set(pd.Timestamp(value) for value in months):
            candidates.append(pd.Timestamp(current))
        current = current - pd.DateOffset(months=int(config.origin_step_months))

    candidates = list(reversed(candidates))
    if not candidates:
        raise ValueError("No forecast origins satisfy the configured rolling-origin constraints.")

    if config.requested_origins > 0 and len(candidates) > config.requested_origins:
        candidates = candidates[-int(config.requested_origins) :]

    if len(candidates) < min(3, int(config.requested_origins)) and len(candidates) < 3:
        raise ValueError(
            "Fewer than three rolling origins are available under the current configuration. "
            "Reduce min_train_months, forecast_horizon, or origin_step_months if the dataset length permits."
        )
    return candidates


def validate_rolling_origin_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    frame = predictions.copy()
    require_columns(frame, ROLLING_ORIGIN_PREDICTION_COLUMNS, "rolling-origin predictions")
    for column in [COLUMNS.forecast_origin, COLUMNS.forecast_month, "training_end_month", "scaler_fit_end_month"]:
        frame[column] = pd.to_datetime(frame[column], errors="coerce").dt.to_period("M").dt.to_timestamp()
    if frame[[COLUMNS.forecast_origin, COLUMNS.forecast_month, "training_end_month", "scaler_fit_end_month"]].isna().any().any():
        raise ValueError("Rolling-origin predictions contain invalid origin, forecast, training, or scaler-fit dates.")
    if not (frame[COLUMNS.forecast_month] > frame[COLUMNS.forecast_origin]).all():
        raise ValueError("Rolling-origin forecast_month values must be later than forecast_origin.")
    if not (frame["training_end_month"] <= frame[COLUMNS.forecast_origin]).all():
        raise ValueError("Rolling-origin training_end_month must not be later than forecast_origin.")
    if not (frame["scaler_fit_end_month"] <= frame[COLUMNS.forecast_origin]).all():
        raise ValueError("Rolling-origin scaler_fit_end_month must not be later than forecast_origin.")

    calculated_horizon = month_difference(frame[COLUMNS.forecast_month], frame[COLUMNS.forecast_origin])
    frame[COLUMNS.horizon] = pd.to_numeric(frame[COLUMNS.horizon], errors="coerce").astype("Int64")
    if frame[COLUMNS.horizon].isna().any() or not (calculated_horizon == frame[COLUMNS.horizon].astype(int)).all():
        raise ValueError("Rolling-origin horizon values do not match forecast_origin and forecast_month.")

    numeric_columns = ["p10_raw", "p50_raw", "p90_raw", COLUMNS.p10, COLUMNS.p50, COLUMNS.p90, COLUMNS.actual]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[numeric_columns].isna().any().any():
        raise ValueError("Rolling-origin predictions contain missing quantiles or actual values.")
    if (frame[COLUMNS.actual] < 0).any():
        raise ValueError("Rolling-origin actual values must be non-negative.")

    raw_crossing = (frame["p10_raw"] > frame["p50_raw"]) | (frame["p50_raw"] > frame["p90_raw"])
    if not (raw_crossing.astype(bool).to_numpy() == frame["quantile_crossing_raw"].astype(bool).to_numpy()).all():
        raise ValueError("quantile_crossing_raw does not match the raw p10/p50/p90 ordering.")
    if ((frame[COLUMNS.p10] > frame[COLUMNS.p50]) | (frame[COLUMNS.p50] > frame[COLUMNS.p90])).any():
        raise ValueError("Corrected display quantiles still contain crossing.")
    return frame


def pinball_loss_np(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> np.ndarray:
    error = y_true - y_pred
    return np.maximum(quantile * error, (quantile - 1.0) * error)


def smape_np(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    values = np.full(y_true.shape, np.nan, dtype=float)
    mask = denominator > 0.0
    values[mask] = 100.0 * np.abs(y_pred[mask] - y_true[mask]) / denominator[mask]
    return values


def winkler_score_np(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float = 0.20) -> np.ndarray:
    width = upper - lower
    below = y_true < lower
    above = y_true > upper
    score = width.copy()
    score[below] += (2.0 / alpha) * (lower[below] - y_true[below])
    score[above] += (2.0 / alpha) * (y_true[above] - upper[above])
    return score


def add_waiting_list_size_groups(predictions: pd.DataFrame, actual_column: str = COLUMNS.actual) -> pd.DataFrame:
    frame = predictions.copy()
    series_sizes = (
        frame.groupby([COLUMNS.series_id, COLUMNS.trust_code, COLUMNS.specialty_code], as_index=False, observed=True)[actual_column]
        .mean()
        .rename(columns={actual_column: "mean_actual_waiting_list"})
    )
    if series_sizes["mean_actual_waiting_list"].nunique() >= 3 and len(series_sizes) >= 3:
        series_sizes["waiting_list_size_group"] = pd.qcut(
            series_sizes["mean_actual_waiting_list"],
            q=3,
            labels=["small", "medium", "large"],
            duplicates="drop",
        ).astype(str)
    else:
        series_sizes["waiting_list_size_group"] = "all"
    frame = frame.drop(columns=["waiting_list_size_group"], errors="ignore").merge(
        series_sizes[[COLUMNS.series_id, "waiting_list_size_group"]],
        on=COLUMNS.series_id,
        how="left",
    )
    frame["waiting_list_size_group"] = frame["waiting_list_size_group"].fillna("unknown")
    return frame


def _metric_row(frame: pd.DataFrame, group_values: Mapping[str, Any], alpha: float) -> Dict[str, Any]:
    y = frame[COLUMNS.actual].to_numpy(dtype=float)
    p10 = frame[COLUMNS.p10].to_numpy(dtype=float)
    p50 = frame[COLUMNS.p50].to_numpy(dtype=float)
    p90 = frame[COLUMNS.p90].to_numpy(dtype=float)
    errors = p50 - y
    abs_errors = np.abs(errors)
    pinball_q10 = pinball_loss_np(y, p10, 0.1)
    pinball_q50 = pinball_loss_np(y, p50, 0.5)
    pinball_q90 = pinball_loss_np(y, p90, 0.9)
    interval_width = p90 - p10
    row: Dict[str, Any] = dict(group_values)
    row.update(
        {
            "n_rows": int(len(frame)),
            "mae": float(np.mean(abs_errors)),
            "rmse": float(np.sqrt(np.mean(errors**2))),
            "smape": float(np.nanmean(smape_np(y, p50))),
            "pinball_q10": float(np.mean(pinball_q10)),
            "pinball_q50": float(np.mean(pinball_q50)),
            "pinball_q90": float(np.mean(pinball_q90)),
            "pinball_mean": float(np.mean(np.vstack([pinball_q10, pinball_q50, pinball_q90]))),
            "p10_p90_coverage": float(np.mean((y >= p10) & (y <= p90))),
            "average_interval_width": float(np.mean(interval_width)),
            "winkler_score_80": float(np.mean(winkler_score_np(y, p10, p90, alpha=alpha))),
            "quantile_crossing_rate_raw": float(np.mean(frame["quantile_crossing_raw"].astype(bool).to_numpy())),
        }
    )
    return row


def rolling_metrics_by_group(
    predictions: pd.DataFrame,
    group_columns: Sequence[str],
    alpha: float = 0.20,
) -> pd.DataFrame:
    frame = validate_rolling_origin_predictions(predictions)
    if not group_columns:
        return pd.DataFrame([_metric_row(frame, {"metric_group": "overall"}, alpha=alpha)])
    rows = []
    for group_key, group in frame.groupby(list(group_columns), observed=True):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        rows.append(_metric_row(group, dict(zip(group_columns, group_key)), alpha=alpha))
    return pd.DataFrame(rows).reset_index(drop=True)


def quantile_crossing_report(predictions: pd.DataFrame) -> pd.DataFrame:
    frame = validate_rolling_origin_predictions(predictions)
    group_columns = ["model_name", COLUMNS.forecast_origin, COLUMNS.horizon]
    rows = []
    for group_key, group in frame.groupby(group_columns, observed=True):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        crossing = group["quantile_crossing_raw"].astype(bool)
        rows.append(
            {
                **dict(zip(group_columns, group_key)),
                "n_rows": int(len(group)),
                "crossing_rows": int(crossing.sum()),
                "quantile_crossing_rate_raw": float(crossing.mean()),
            }
        )
    return pd.DataFrame(rows).reset_index(drop=True)


def calibration_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    frame = validate_rolling_origin_predictions(predictions)
    rows = []
    for model_name, group in frame.groupby("model_name", observed=True):
        y = group[COLUMNS.actual].to_numpy(dtype=float)
        for quantile, column in [(0.1, COLUMNS.p10), (0.5, COLUMNS.p50), (0.9, COLUMNS.p90)]:
            empirical = float(np.mean(y <= group[column].to_numpy(dtype=float)))
            rows.append(
                {
                    "model_name": model_name,
                    "target": f"p{int(quantile * 100):02d}",
                    "expected_coverage": float(quantile),
                    "empirical_coverage": empirical,
                    "coverage_error": empirical - float(quantile),
                    "n_rows": int(len(group)),
                }
            )
        interval_coverage = float(np.mean((y >= group[COLUMNS.p10].to_numpy(dtype=float)) & (y <= group[COLUMNS.p90].to_numpy(dtype=float))))
        rows.append(
            {
                "model_name": model_name,
                "target": "p10_p90_interval",
                "expected_coverage": 0.80,
                "empirical_coverage": interval_coverage,
                "coverage_error": interval_coverage - 0.80,
                "n_rows": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def interval_width_coverage_by_origin(predictions: pd.DataFrame) -> pd.DataFrame:
    frame = validate_rolling_origin_predictions(predictions)
    rows = []
    for group_key, group in frame.groupby(["model_name", COLUMNS.forecast_origin], observed=True):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        y = group[COLUMNS.actual].to_numpy(dtype=float)
        p10 = group[COLUMNS.p10].to_numpy(dtype=float)
        p90 = group[COLUMNS.p90].to_numpy(dtype=float)
        rows.append(
            {
                "model_name": group_key[0],
                COLUMNS.forecast_origin: group_key[1],
                "n_rows": int(len(group)),
                "p10_p90_coverage": float(np.mean((y >= p10) & (y <= p90))),
                "average_interval_width": float(np.mean(p90 - p10)),
            }
        )
    return pd.DataFrame(rows)


def build_reliability_text(
    *,
    config: RollingOriginConfig,
    origins: Sequence[pd.Timestamp],
    predictions: pd.DataFrame,
    overall_metrics: pd.DataFrame,
    reliability: pd.DataFrame,
) -> str:
    frame = validate_rolling_origin_predictions(predictions)
    interval_row = reliability[reliability["target"].eq("p10_p90_interval")]
    empirical = float(interval_row["empirical_coverage"].iloc[0]) if not interval_row.empty else math.nan
    nominal = 1.0 - float(config.nominal_interval_alpha)
    close = bool(np.isfinite(empirical) and abs(empirical - nominal) <= float(config.close_to_nominal_tolerance))
    crossing_rate = float(overall_metrics["quantile_crossing_rate_raw"].iloc[0])
    first_month = pd.to_datetime(frame[COLUMNS.forecast_month]).min().date().isoformat()
    last_month = pd.to_datetime(frame[COLUMNS.forecast_month]).max().date().isoformat()
    origin_text = ", ".join(pd.Timestamp(origin).date().isoformat() for origin in origins)
    lines = [
        "# Rolling-Origin Validation Summary",
        "",
        f"Configuration: `{json.dumps(asdict(config), sort_keys=True)}`",
        "",
        f"Forecast origins: {origin_text}",
        f"Forecast months evaluated: {first_month} to {last_month}",
        f"Rows evaluated: {len(frame)}",
        f"Raw quantile-crossing rate before correction: {crossing_rate:.4f}",
        "",
        "## Reliability",
        "",
        f"Nominal P10-P90 coverage: {nominal:.2%}",
        f"Empirical P10-P90 coverage: {empirical:.2%}",
        (
            f"The nominal 80% interval is close to empirical coverage within "
            f"+/-{config.close_to_nominal_tolerance:.0%}."
            if close
            else f"The nominal 80% interval is not close to empirical coverage within +/-{config.close_to_nominal_tolerance:.0%}."
        ),
        "",
        "Raw P10/P50/P90 values are saved alongside corrected display quantiles. Quantile crossing is detected before correction.",
        "The final production model is trained separately from these validation fold models.",
    ]
    return "\n".join(lines)


def save_calibration_plots(
    reliability: pd.DataFrame,
    width_coverage: pd.DataFrame,
    calibration_png: str | Path,
    width_coverage_png: str | Path,
) -> None:
    import matplotlib.pyplot as plt

    calibration_path = Path(calibration_png)
    width_path = Path(width_coverage_png)
    calibration_path.parent.mkdir(parents=True, exist_ok=True)

    calibration = reliability[reliability["target"].isin(["p10", "p50", "p90"])].copy()
    calibration["label"] = calibration["target"].str.upper()
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0.0, 1.0], [0.0, 1.0], color="#111111", linestyle="--", linewidth=1.2, label="Ideal")
    ax.scatter(
        calibration["expected_coverage"],
        calibration["empirical_coverage"],
        s=90,
        color="#2563eb",
        label="Observed",
    )
    for row in calibration.itertuples(index=False):
        ax.annotate(row.label, (row.expected_coverage, row.empirical_coverage), textcoords="offset points", xytext=(6, 6))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Expected quantile coverage")
    ax.set_ylabel("Empirical coverage")
    ax.set_title("Rolling-Origin Quantile Calibration")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(calibration_path, dpi=180)
    plt.close(fig)

    plot_data = width_coverage.copy()
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        plot_data["average_interval_width"],
        plot_data["p10_p90_coverage"],
        c=pd.to_datetime(plot_data[COLUMNS.forecast_origin]).map(pd.Timestamp.toordinal),
        cmap="viridis",
        s=80,
    )
    ax.axhline(0.80, color="#111111", linestyle="--", linewidth=1.2, label="Nominal 80%")
    ax.set_xlabel("Average P10-P90 interval width")
    ax.set_ylabel("Empirical P10-P90 coverage")
    ax.set_title("Interval Width Versus Coverage by Forecast Origin")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.colorbar(scatter, ax=ax, label="Forecast origin")
    fig.tight_layout()
    fig.savefig(width_path, dpi=180)
    plt.close(fig)


def save_rolling_origin_outputs(
    *,
    predictions: pd.DataFrame,
    config: RollingOriginConfig,
    origins: Sequence[pd.Timestamp],
    paths: Mapping[str, Path],
) -> Dict[str, pd.DataFrame]:
    frame = add_waiting_list_size_groups(validate_rolling_origin_predictions(predictions))
    alpha = float(config.nominal_interval_alpha)
    overall = rolling_metrics_by_group(frame, [], alpha=alpha)
    by_origin = rolling_metrics_by_group(frame, [COLUMNS.forecast_origin], alpha=alpha)
    by_horizon = rolling_metrics_by_group(frame, [COLUMNS.horizon], alpha=alpha)
    by_trust = rolling_metrics_by_group(frame, [COLUMNS.trust_code, COLUMNS.trust_name], alpha=alpha)
    by_specialty = rolling_metrics_by_group(frame, [COLUMNS.specialty_code, COLUMNS.specialty_name], alpha=alpha)
    by_waiting_size = rolling_metrics_by_group(frame, ["waiting_list_size_group"], alpha=alpha)
    crossing = quantile_crossing_report(frame)
    reliability = calibration_summary(frame)
    width_coverage = interval_width_coverage_by_origin(frame)
    summary_text = build_reliability_text(
        config=config,
        origins=origins,
        predictions=frame,
        overall_metrics=overall,
        reliability=reliability,
    )

    paths["predictions"].parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(paths["predictions"], index=False)
    overall.to_csv(paths["overall"], index=False)
    by_origin.to_csv(paths["by_origin"], index=False)
    by_horizon.to_csv(paths["by_horizon"], index=False)
    by_trust.to_csv(paths["by_trust"], index=False)
    by_specialty.to_csv(paths["by_specialty"], index=False)
    by_waiting_size.to_csv(paths["by_waiting_size"], index=False)
    crossing.to_csv(paths["crossing"], index=False)
    reliability.to_csv(paths["reliability"], index=False)
    Path(paths["summary"]).write_text(summary_text, encoding="utf-8")
    save_calibration_plots(reliability, width_coverage, paths["calibration_png"], paths["width_coverage_png"])
    width_coverage.to_csv(Path(paths["width_coverage_png"]).with_suffix(".csv"), index=False)

    return {
        "predictions": frame,
        "overall": overall,
        "by_origin": by_origin,
        "by_horizon": by_horizon,
        "by_trust": by_trust,
        "by_specialty": by_specialty,
        "by_waiting_size": by_waiting_size,
        "crossing": crossing,
        "reliability": reliability,
        "width_coverage": width_coverage,
    }
