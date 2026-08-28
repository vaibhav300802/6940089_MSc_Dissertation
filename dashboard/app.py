from __future__ import annotations

import glob
import html
import json
import re
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nhs_rtt_pipeline.config import (
    COLUMNS,
    get_paths,
    normalise_text,
    validate_backtest_predictions_frame,
    validate_future_forecast_frame,
    validate_optimisation_forecast_frame,
)
from nhs_rtt_pipeline.optimisation import (
    load_capacity_scenario_config,
    reduction_by_trust_specialty,
    solve_milp_allocation,
)


PATHS = get_paths(PROJECT_ROOT)
AUTHOR_NAME = "APPRAJIT VAIBHAV MANIKANDAN"
PROJECT_TITLE = "NHS RTT WAITING LIST FORECASTING"
PROJECT_SUBTITLE = (
    "RTT means referral-to-treatment: people waiting to start consultant-led care after referral. "
    "This dashboard turns NHS England RTT data from October 2015 onward into a calibrated 12-month "
    "forecast for every Trust and specialty in England, explains what is driving each forecast, tests "
    "how additional theatre capacity could reduce the backlog, and simulates how a new pandemic-style "
    "disruption would affect it, all without needing to write any code."
)
PAGE_LABELS = ["Overview", "Hospital profile", "Forecast drivers", "Capacity test", "Pandemic test", "Model check"]
PAGE_DISPLAY_LABELS = {
    "Overview": "OVERVIEW",
    "Hospital profile": "HOSPITAL PROFILE",
    "Forecast drivers": "FORECAST DRIVERS",
    "Capacity test": "CAPACITY TEST",
    "Pandemic test": "PANDEMIC SIMULATION",
    "Model check": "MODEL COMPARISON",
}

st.set_page_config(
    page_title="NHS RTT WAITING LIST FORECASTING",
    page_icon="NHS",
    layout="wide",
)

def apply_theme(dark_mode: bool) -> None:
    if dark_mode:
        colors = {
            "page": "#162331",
            "panel": "#1e2f40",
            "sidebar": "#1a2938",
            "border": "#36536d",
            "text": "#f1f7ff",
            "muted": "#c3d1df",
            "blue": "#78bdf8",
            "green": "#69d590",
            "note": "#233a50",
            "shadow": "rgba(0, 0, 0, 0.18)",
        }
    else:
        colors = {
            "page": "#eef5f9",
            "panel": "#ffffff",
            "sidebar": "#e7f0f6",
            "border": "#c3d8e8",
            "text": "#102033",
            "muted": "#4f6680",
            "blue": "#005eb8",
            "green": "#006b3f",
            "note": "#e2f0f8",
            "shadow": "rgba(15, 23, 42, 0.07)",
        }
    st.markdown(
        f"""
    <style>
    html, body, [data-testid="stAppViewContainer"] {{
        background: {colors["page"]};
        color: {colors["text"]};
    }}
    .block-container {{
        padding-top: 1.0rem;
        padding-bottom: 3rem;
        max-width: 1540px;
    }}
    [data-testid="stSidebar"] {{
        background: {colors["sidebar"]};
        border-right: 1px solid {colors["border"]};
    }}
    h1, h2, h3 {{
        color: {colors["text"]};
        letter-spacing: 0;
    }}
    div[data-testid="stMetric"] {{
        background: {colors["panel"]};
        border: 1px solid {colors["border"]};
        border-radius: 8px;
        padding: 0.75rem 0.9rem;
        box-shadow: 0 1px 10px {colors["shadow"]};
        min-height: 116px;
    }}
    [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p,
    [data-testid="stMetricDelta"], [data-testid="stMetricDelta"] p {{
        color: {colors["muted"]} !important;
        opacity: 1 !important;
    }}
    [data-testid="stMetricValue"], [data-testid="stMetricValue"] div {{
        color: {colors["text"]} !important;
    }}
    [data-testid="stAlert"], [data-testid="stAlert"] * {{
        color: {colors["text"]} !important;
        opacity: 1 !important;
    }}
    .plain-note {{
        border-left: 4px solid {colors["blue"]};
        background: {colors["note"]};
        color: {colors["text"]};
        padding: 0.75rem 1rem;
        margin: 0.4rem 0 1rem 0;
    }}
    .quiet {{
        color: {colors["muted"]};
        font-size: 0.92rem;
    }}
    .section-copy {{
        color: {colors["muted"]};
        font-size: 0.98rem;
        line-height: 1.55;
        max-width: 980px;
        margin: -0.25rem 0 1rem 0;
    }}
    .summary-panel {{
        background: {colors["panel"]};
        border: 1px solid {colors["border"]};
        border-radius: 8px;
        padding: 1rem 1.1rem;
        margin: 0.9rem 0 1rem 0;
        box-shadow: 0 1px 10px {colors["shadow"]};
    }}
    .summary-intro {{
        color: {colors["muted"]};
        line-height: 1.5;
        margin-bottom: 0.8rem;
    }}
    .summary-intro strong {{
        color: {colors["text"]};
        margin-right: 0.45rem;
    }}
    .summary-grid {{
        display: grid;
        grid-template-columns: repeat(3, minmax(180px, 1fr));
        gap: 0.75rem;
    }}
    .summary-item {{
        border-top: 1px solid {colors["border"]};
        padding-top: 0.65rem;
    }}
    .summary-label {{
        color: {colors["muted"]};
        font-size: 0.9rem;
        margin-bottom: 0.2rem;
    }}
    .summary-value {{
        color: {colors["text"]};
        font-size: 1.65rem;
        line-height: 1.2;
        font-weight: 500;
    }}
    .summary-delta {{
        color: {colors["muted"]};
        font-size: 0.92rem;
        margin-top: 0.25rem;
    }}
    .trend-up {{
        color: #d92d20;
        font-weight: 700;
    }}
    .trend-neutral {{
        color: {colors["muted"]};
        font-weight: 700;
    }}
    .section-title {{
        font-size: 1rem;
        font-weight: 700;
        color: {colors["text"]};
        margin: 0.35rem 0 0.5rem 0;
    }}
    .detail-panel {{
        background: {colors["panel"]};
        border: 1px solid {colors["border"]};
        border-radius: 8px;
        padding: 1rem 1.15rem;
        margin-top: 0.7rem;
        box-shadow: 0 1px 8px {colors["shadow"]};
    }}
    .detail-panel strong {{
        color: {colors["blue"]};
    }}
    .hero {{
        background: linear-gradient(135deg, {colors["panel"]} 0%, {colors["note"]} 100%);
        border: 1px solid {colors["border"]};
        border-radius: 8px;
        padding: 1.35rem 1.5rem;
        margin-bottom: 1.1rem;
        box-shadow: 0 1px 12px {colors["shadow"]};
    }}
    .hero-title {{
        font-size: 2.25rem;
        line-height: 1.08;
        font-weight: 800;
        color: {colors["text"]};
        margin-bottom: 0.35rem;
    }}
    .hero-subtitle {{
        max-width: 980px;
        color: {colors["muted"]};
        font-size: 1.02rem;
        line-height: 1.55;
    }}
    .author-line {{
        margin-top: 0.95rem;
        color: {colors["text"]};
        font-weight: 800;
        font-size: 1rem;
        letter-spacing: 0.02rem;
    }}
    .forecast-key {{
        display: grid;
        grid-template-columns: repeat(3, minmax(180px, 1fr));
        gap: 0.75rem;
        margin: 0.8rem 0 1rem 0;
    }}
    .forecast-key div {{
        background: {colors["panel"]};
        border: 1px solid {colors["border"]};
        border-radius: 8px;
        padding: 0.75rem 0.9rem;
    }}
    .forecast-key strong {{
        color: {colors["blue"]};
    }}
    [data-testid="stSidebar"] button {{
        width: 100%;
        border-radius: 8px;
        border: 0;
        background: transparent;
        color: {colors["text"]};
        min-height: 40px;
        justify-content: flex-start;
    }}
    [data-testid="stSidebar"] button:hover {{
        background: {colors["note"]};
        color: {colors["blue"]};
    }}
    .active-page {{
        border-left: 4px solid {colors["blue"]};
        background: {colors["note"]};
        color: {colors["text"]};
        padding: 0.6rem 0.75rem;
        border-radius: 8px;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }}
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] p {{
        font-size: 0.92rem;
        color: {colors["text"]};
    }}
    [data-testid="stRadio"] label, [data-testid="stRadio"] p,
    [data-testid="stRadio"] div, [data-testid="stSelectbox"] label,
    [data-testid="stSlider"] label, [data-testid="stToggle"] label {{
        color: {colors["text"]} !important;
        opacity: 1 !important;
    }}
    [data-baseweb="select"] > div, [data-testid="stNumberInput"] input {{
        background: {colors["panel"]} !important;
        color: {colors["text"]} !important;
        border-color: {colors["border"]} !important;
    }}
    [data-baseweb="select"] input, input, textarea {{
        color: {colors["text"]} !important;
        -webkit-text-fill-color: {colors["text"]} !important;
        caret-color: {colors["text"]} !important;
    }}
    div[data-baseweb="popover"] > div {{
        background: {colors["panel"]} !important;
        border: 1px solid {colors["border"]} !important;
    }}
    [data-baseweb="popover"], [data-baseweb="popover"] ul,
    [role="listbox"], [role="listbox"] ul {{
        background: {colors["panel"]} !important;
        color: {colors["text"]} !important;
    }}
    div[data-baseweb="popover"] input,
    div[data-baseweb="popover"] [role="option"],
    div[data-baseweb="popover"] [role="listbox"],
    div[data-baseweb="popover"] span,
    div[data-baseweb="popover"] div {{
        color: {colors["text"]} !important;
        -webkit-text-fill-color: {colors["text"]} !important;
    }}
    div[data-baseweb="popover"] [aria-selected="true"],
    div[data-baseweb="popover"] [role="option"]:hover {{
        background: {colors["note"]} !important;
    }}
    [role="option"], [role="option"] div, [role="option"] span {{
        background: transparent !important;
        color: {colors["text"]} !important;
        -webkit-text-fill-color: {colors["text"]} !important;
    }}
    [role="option"][aria-selected="true"], [role="option"]:hover {{
        background: {colors["note"]} !important;
    }}
    [data-testid="stDataFrame"] {{
        background: {colors["panel"]};
        border: 1px solid {colors["border"]};
        border-radius: 8px;
    }}
    [data-testid="stHeader"] {{
        display: none !important;
    }}
    [data-testid="stToolbar"], [data-testid="stDecoration"], #MainMenu, footer {{
        visibility: hidden !important;
        height: 0 !important;
    }}
    .stDeployButton, [data-testid="stDeployButton"] {{
        display: none !important;
    }}
    h1 a, h2 a, h3 a {{
        display: none !important;
    }}
    </style>
    """,
        unsafe_allow_html=True,
    )


SCENARIO_OPTIONS = {
    "Lower-bound forecast": COLUMNS.p10,
    "Median forecast": COLUMNS.p50,
    "Upper-bound forecast": COLUMNS.p90,
}

FORECAST_LABELS = {column: label for label, column in SCENARIO_OPTIONS.items()}

PLOTLY_CONFIG = {
    "displayModeBar": False,
    "displaylogo": False,
    "scrollZoom": False,
}

MAP_PLOTLY_CONFIG = {
    "displayModeBar": True,
    "displaylogo": False,
    "scrollZoom": True,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}

FORECAST_EXPLANATIONS = {
    "Lower-bound forecast": "A cautious lower estimate from the forecast range.",
    "Median forecast": "The central estimate used when one forecast value is needed.",
    "Upper-bound forecast": "A higher-pressure estimate from the forecast range.",
}

PRODUCTIVITY_EXPLANATIONS = {
    "low": "Low productivity: fewer decision-to-admit waiting-list entries are assumed to be completed per added session.",
    "central": "Central productivity: the middle scenario assumption used for the main capacity view.",
    "high": "High productivity: more decision-to-admit waiting-list entries are assumed to be completed per added session.",
}


COLUMN_LABELS = {
    COLUMNS.trust_code: "Hospital code",
    COLUMNS.trust_name: "Hospital",
    COLUMNS.specialty_code: "Specialty code",
    COLUMNS.specialty_name: "Specialty",
    COLUMNS.forecast_month: "Forecast month",
    COLUMNS.horizon: "Months ahead",
    COLUMNS.p10: "Lower-bound forecast",
    COLUMNS.p50: "Median forecast",
    COLUMNS.p90: "Upper-bound forecast",
    COLUMNS.latest_observed_waiting_list: "Latest observed waiting list",
    "model": "Model",
    "model_name": "Model",
    "n_rows": "Rows tested",
    "mae": "MAE",
    "rmse": "RMSE",
    "smape": "sMAPE",
    "wape": "WAPE",
    "pinball_mean": "Average quantile error",
    "pinball_q10": "Lower-bound quantile error",
    "pinball_q50": "Median quantile error",
    "pinball_q90": "Upper-bound quantile error",
    "p10_p90_coverage": "Forecast range coverage",
    "average_interval_width": "Average range width",
    "baseline_predicted_backlog": "Forecast backlog",
    "patients_completed_per_session": "Assumed completions per session",
    "sessions_allocated": "Sessions allocated",
    "simulated_completed_pathways": "Simulated waiting-list reduction",
    "remaining_backlog": "Remaining backlog",
    "percent_reduction": "Percent reduction",
    "missing_months": "Missing months",
    "missing_pct": "Missing percent",
    "max_consecutive_missing_months": "Longest missing run",
    "first_month": "First month",
    "last_month": "Last month",
    "change_type": "Change type",
    "details": "Details",
    "feature_group": "Historical information type",
    "feature_group_display": "Historical information type",
    "mean_shap_value": "Forecast contribution",
    "mean_abs_shap_value": "Contribution size",
    "mean_model_p50": "Median forecast",
    "mean_abs_shap": "Contribution size",
    "phase": "Scenario phase",
    "baseline_median_forecast": "Baseline median forecast",
    "additional_backlog": "Additional waiting-list pressure",
    "pandemic_type_scenario": "Pandemic-type scenario",
}


REQUIRED_FILES: Mapping[str, Path] = {
    "Future forecasts": PATHS.future_forecasts,
    "Historical processed data": PATHS.clean_parquet,
    "Model configuration": PATHS.model_config,
    "Model comparison metrics": PATHS.model_comparison,
    "Model comparison by horizon": PATHS.model_comparison_by_horizon,
    "Part 2A optimisation forecasts": PATHS.future_optimisation_forecasts,
    "Capacity scenario assumptions": PATHS.capacity_scenario_config,
}


OPTIONAL_FILES: Mapping[str, Path] = {
    "Data-quality summary": PATHS.data_quality_summary,
    "Data-quality report": PATHS.data_quality_report,
    "Missingness by series": PATHS.missingness_by_series,
    "Trust identifier changes": PATHS.trust_identifier_changes,
    "SHAP global feature importance": PATHS.shap_global_importance,
    "SHAP grouped feature importance": PATHS.shap_group_importance,
    "SHAP local explanations": PATHS.shap_local_explanations,
    "SHAP interpretations": PATHS.shap_interpretations,
    "SHAP waterfall index": PATHS.shap_waterfall_index,
    "SHAP global summary image": PATHS.shap_global_summary,
    "Optimisation allocation output": PATHS.lp_allocation_output,
    "Optimisation sensitivity CSV": PATHS.lp_sensitivity_output,
    "Optimisation sensitivity image": PATHS.lp_sensitivity_png,
    "Optimisation uncertainty comparison": PATHS.lp_uncertainty_comparison,
    "Rolling-origin reliability": PATHS.rolling_origin_reliability,
    "Rolling-origin overall metrics": PATHS.rolling_origin_metrics,
    "COVID shock predictions": PATHS.covid_predictions,
    "COVID shock metrics": PATHS.covid_metrics,
    "COVID shock degradation": PATHS.covid_degradation,
    "Trust coordinates": PATHS.dashboard_data_dir / "nhs_trust_coordinates.csv",
}


@dataclass(frozen=True)
class DashboardFilters:
    trust_label: str
    trust_code: Optional[str]
    specialty_label: str
    specialty_code: Optional[str]
    region: str
    forecast_month: Optional[pd.Timestamp]
    scenario_label: str
    scenario_column: str
    productivity_scenario: str
    available_sessions: int


def compact_number(value: object) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    number = float(value)
    if abs(number) >= 1_000_000:
        return f"{number / 1_000_000:.2f}m"
    if abs(number) >= 1_000:
        return f"{number / 1_000:.1f}k"
    return f"{number:,.0f}"


def format_month(value: object) -> str:
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return "n/a"
    return pd.Timestamp(timestamp).strftime("%B %Y")


def format_month_range(start: object, end: object) -> str:
    start_label = format_month(start)
    end_label = format_month(end)
    if start_label == "n/a" or end_label == "n/a":
        return "n/a"
    return f"{start_label} to {end_label}"


def display_hospital_name(value: object) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    cleaned = re.sub(r"\s+NHS\s+FOUNDATION\s+TRUST$", "", raw, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+NHS\s+TRUST$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+NHS\s+FOUNDATION$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if cleaned.isupper():
        cleaned = cleaned.title()
    replacements = {
        "Nhs": "NHS",
        "Rtt": "RTT",
        "Uk": "UK",
        "And": "and",
        "Of": "of",
        "The": "the",
    }
    for old, new in replacements.items():
        cleaned = re.sub(rf"\b{old}\b", new, cleaned)
    return cleaned


def display_specialty_name(value: object) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    replacements = {
        "Ear Nose and Throat Service": "Ear, Nose and Throat",
        "Neurosurgical Service": "Neurosurgery",
    }
    cleaned = replacements.get(raw, raw)
    cleaned = re.sub(r"\s+Service$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def clean_feature_label(value: object, lag: object = None) -> str:
    text = str(value or "").strip()
    lag_match = re.search(r"(?:__|_)lag_?(\d+)", text, flags=re.IGNORECASE)
    parsed_lag = int(lag_match.group(1)) if lag_match else None
    text = re.sub(r"(?:__|_)lag_?\d+", " ", text, flags=re.IGNORECASE)
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text).strip()
    replacements = {
        "rtt": "RTT",
        "dta": "decision to admit",
        "p50": "median forecast",
        "p10": "lower-bound forecast",
        "p90": "upper-bound forecast",
    }
    for old, new in replacements.items():
        text = re.sub(rf"\b{old}\b", new, text, flags=re.IGNORECASE)
    text = re.sub(r"\bmissing$", "missing-data flag", text, flags=re.IGNORECASE)
    text = re.sub(r"\bimputed$", "imputed-value flag", text, flags=re.IGNORECASE)
    direct_replacements = {
        "time index": "Long-term time pattern",
        "trust identifier embedding": "Hospital pattern",
        "specialty identifier embedding": "Specialty pattern",
        "reported net inflow": "Reported net inflow",
    }
    text = direct_replacements.get(text.lower(), text)
    if text:
        text = text[0].upper() + text[1:]
    lag_value = pd.to_numeric(pd.Series([lag]), errors="coerce").iloc[0] if lag is not None else np.nan
    resolved_lag = int(lag_value) if pd.notna(lag_value) and int(lag_value) > 0 else parsed_lag
    if resolved_lag is not None and resolved_lag > 0:
        unit = "month" if resolved_lag == 1 else "months"
        text = f"{text} ({resolved_lag} {unit} earlier)"
    return text


def clean_feature_group(value: object) -> str:
    text = str(value or "").strip().replace("-", " ")
    text = re.sub(r"\s+", " ", text)
    replacements = {
        "waiting list lags": "Waiting-list history",
        "referral or new rtt period features": "New RTT periods and net inflow",
        "completed pathway features": "Completed pathways",
        "unreported removal features": "Unreported removals",
        "calendar features": "Time and calendar patterns",
        "trust and specialty identifiers or embeddings": "Hospital and specialty patterns",
    }
    for old, new in replacements.items():
        if text.lower() == old.lower():
            return new
    return text[:1].upper() + text[1:]


def display_model_name(value: object) -> str:
    text = str(value or "").strip().replace("_", " ")
    replacements = {
        "naive last value": "Naive last value",
        "seasonal naive": "Seasonal naive",
        "historical seasonal mean": "Historical seasonal mean",
        "hist gradient boosting": "Gradient boosting",
        "tcn": "TCN",
    }
    lower = text.lower()
    return replacements.get(lower, text[:1].upper() + text[1:])


def show_png(path: Path | str) -> None:
    st.image(Image.open(path), use_column_width=True)


def chart_template() -> str:
    return "plotly_dark" if bool(st.session_state.get("dark_mode", False)) else "plotly_white"


def observed_line_color() -> str:
    return "#e5edf8" if bool(st.session_state.get("dark_mode", False)) else "#111827"


def map_style() -> str:
    return "open-street-map"


def panel_color() -> str:
    return "#0f1b2d" if bool(st.session_state.get("dark_mode", False)) else "#ffffff"


def page_color() -> str:
    return "#08111f" if bool(st.session_state.get("dark_mode", False)) else "#f4f8fb"


def chart_text_color() -> str:
    return "#f1f7ff" if bool(st.session_state.get("dark_mode", False)) else "#102033"


def chart_grid_color() -> str:
    return "#31485d" if bool(st.session_state.get("dark_mode", False)) else "#d9e5ef"


def chart_primary_color() -> str:
    return "#78bdf8" if bool(st.session_state.get("dark_mode", False)) else "#005eb8"


def chart_positive_color() -> str:
    return "#69d590" if bool(st.session_state.get("dark_mode", False)) else "#007f3b"


def chart_warning_color() -> str:
    return "#fbbf24" if bool(st.session_state.get("dark_mode", False)) else "#b45309"


def apply_chart_style(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        paper_bgcolor=panel_color(),
        plot_bgcolor=panel_color(),
        font=dict(color=chart_text_color()),
        legend=dict(font=dict(color=chart_text_color())),
        title_font=dict(color=chart_text_color()),
    )
    fig.update_xaxes(
        gridcolor=chart_grid_color(),
        zerolinecolor=chart_grid_color(),
        linecolor=chart_grid_color(),
        tickfont=dict(color=chart_text_color()),
        title_font=dict(color=chart_text_color()),
    )
    fig.update_yaxes(
        gridcolor=chart_grid_color(),
        zerolinecolor=chart_grid_color(),
        linecolor=chart_grid_color(),
        tickfont=dict(color=chart_text_color()),
        title_font=dict(color=chart_text_color()),
    )
    return fig


def clean_display_frame(frame: pd.DataFrame, columns: Optional[Sequence[str]] = None, max_rows: Optional[int] = None) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    out = frame.copy()
    if columns is not None:
        keep = [column for column in columns if column in out.columns]
        out = out[keep].copy()
    if max_rows is not None:
        out = out.head(max_rows).copy()
    for column in out.columns:
        if column == COLUMNS.trust_name:
            out[column] = out[column].map(display_hospital_name)
        if column == COLUMNS.specialty_name:
            out[column] = out[column].map(display_specialty_name)
        if column in {
            COLUMNS.forecast_month,
            COLUMNS.forecast_origin,
            "month",
            "first_month",
            "last_month",
            "publication_month",
        }:
            out[column] = pd.to_datetime(out[column], errors="coerce").map(format_month)
        elif pd.api.types.is_datetime64_any_dtype(out[column]):
            out[column] = pd.to_datetime(out[column], errors="coerce").dt.strftime("%Y-%m-%d")
        if column in {"model", "model_name"}:
            out[column] = out[column].map(display_model_name)
        if column in {"feature_name", "display_name", "raw_column"}:
            out[column] = out[column].map(clean_feature_label)
        if column == "feature_group":
            out[column] = out[column].map(clean_feature_group)
    out = out.replace({np.nan: "", pd.NaT: "", "NaT": "", "None": "", None: ""})
    keep_indices: list[int] = []
    renamed_columns: list[str] = []
    used: set[str] = set()
    for index, column in enumerate(out.columns):
        label = COLUMN_LABELS.get(column, str(column).replace("_", " ").title())
        if label in used:
            continue
        keep_indices.append(index)
        renamed_columns.append(label)
        used.add(label)
    out = out.iloc[:, keep_indices].copy()
    out.columns = renamed_columns
    return out


def show_table(
    frame: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
    max_rows: Optional[int] = None,
    max_height: int = 420,
) -> None:
    display = clean_display_frame(frame, columns=columns, max_rows=max_rows)
    table_height = min(max_height, max(110, 38 * (len(display) + 1)))
    st.dataframe(display, use_container_width=True, hide_index=True, height=table_height)


def compact_change(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    sign = "+" if value > 0 else ""
    return f"{sign}{compact_number(value)}"


def percent_change(new_value: float, old_value: float) -> str:
    if old_value in (0, 0.0) or pd.isna(old_value) or pd.isna(new_value):
        return "n/a"
    return f"{((new_value - old_value) / old_value) * 100:+.1f}%"


def friendly_interpretation(text: object) -> str:
    cleaned = str(text or "").strip()
    if not cleaned:
        return ""
    replacements = {
        "is primarily driven by increasing referral inflow": "has forecasts most strongly associated with higher referral or clock-start pressure",
        "is primarily driven by decreasing treatment throughput": "has forecasts most strongly associated with lower completed-pathway activity",
        "shows a mixed pattern of both referral surge and throughput reduction": "has forecasts linked to both referral pressure and completed-pathway activity",
        "driven by": "associated with",
        "driving": "associated with",
        "caused by": "associated with",
    }
    for old, new in replacements.items():
        cleaned = cleaned.replace(old, new)
    return cleaned


def select_display_horizon(frame: pd.DataFrame, preferred_horizon: int = 12) -> pd.DataFrame:
    if frame.empty or COLUMNS.horizon not in frame.columns:
        return frame
    horizons = pd.to_numeric(frame[COLUMNS.horizon], errors="coerce")
    if horizons.eq(preferred_horizon).any():
        return frame[horizons.eq(preferred_horizon)].copy()
    max_horizon = horizons.max()
    if pd.isna(max_horizon):
        return frame.copy()
    return frame[horizons.eq(max_horizon)].copy()


def preferred_trust_code(forecast: pd.DataFrame) -> Optional[str]:
    if forecast.empty:
        return None
    trust_codes = set(forecast[COLUMNS.trust_code].astype(str))
    if "RA2" in trust_codes:
        return "RA2"
    horizon_slice = select_display_horizon(forecast, preferred_horizon=12)
    if horizon_slice.empty:
        horizon_slice = forecast
    grouped = horizon_slice.groupby(COLUMNS.trust_code, as_index=False, observed=True)[COLUMNS.p50].sum()
    if grouped.empty:
        return None
    return str(grouped.sort_values(COLUMNS.p50, ascending=False).iloc[0][COLUMNS.trust_code])


def page_hospital_specialty_filters(
    frame: pd.DataFrame,
    filters: DashboardFilters,
    key_prefix: str,
    preferred_code: Optional[str] = None,
) -> DashboardFilters:
    required = [COLUMNS.trust_code, COLUMNS.trust_name, COLUMNS.specialty_code, COLUMNS.specialty_name]
    missing = [column for column in required if column not in frame.columns]
    if missing or frame.empty:
        return filters

    trust_options = (
        frame[[COLUMNS.trust_code, COLUMNS.trust_name]]
        .drop_duplicates()
        .sort_values(COLUMNS.trust_name)
        .reset_index(drop=True)
    )
    trust_labels = [
        f"{display_hospital_name(row.trust_name)} ({row.trust_code})"
        for row in trust_options.itertuples(index=False)
    ]
    default_code = preferred_code or filters.trust_code or preferred_trust_code(frame)
    trust_index = 0
    if default_code is not None:
        matches = trust_options.index[trust_options[COLUMNS.trust_code].astype(str).eq(str(default_code))]
        if len(matches):
            trust_index = int(matches[0])

    control_cols = st.columns(2)
    trust_label = control_cols[0].selectbox(
        "Hospital",
        trust_labels,
        index=trust_index,
        key=f"{key_prefix}_hospital",
    )
    trust_code = str(trust_options.iloc[trust_labels.index(trust_label)][COLUMNS.trust_code])

    hospital_rows = frame[frame[COLUMNS.trust_code].astype(str).eq(trust_code)]
    specialty_options = (
        hospital_rows[[COLUMNS.specialty_code, COLUMNS.specialty_name]]
        .drop_duplicates()
        .sort_values(COLUMNS.specialty_name)
        .reset_index(drop=True)
    )
    specialty_labels = ["All specialties"] + [
        f"{display_specialty_name(row.specialty_name)} ({row.specialty_code})"
        for row in specialty_options.itertuples(index=False)
    ]
    specialty_label = control_cols[1].selectbox(
        "Specialty",
        specialty_labels,
        index=0,
        key=f"{key_prefix}_specialty",
    )
    specialty_code = None if specialty_label == "All specialties" else str(
        specialty_options.iloc[specialty_labels.index(specialty_label) - 1][COLUMNS.specialty_code]
    )
    return replace(
        filters,
        trust_label=trust_label,
        trust_code=trust_code,
        specialty_label=specialty_label,
        specialty_code=specialty_code,
    )


def safe_filename(value: object, max_length: int = 140) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_")
    return (cleaned or "unnamed")[:max_length]


def file_timestamp(path: Path) -> str:
    if not path.exists():
        return "missing"
    return pd.Timestamp(path.stat().st_mtime, unit="s").strftime("%Y-%m-%d %H:%M")


def read_json_if_exists(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def status_frame() -> pd.DataFrame:
    rows = []
    for required, files in [(True, REQUIRED_FILES), (False, OPTIONAL_FILES)]:
        for label, path in files.items():
            rows.append(
                {
                    "artifact": label,
                    "required": required,
                    "status": "available" if path.exists() else "missing",
                    "path": str(path),
                    "modified": file_timestamp(path) if path.exists() else "",
                }
            )
    return pd.DataFrame(rows)


def missing_required_files(status: pd.DataFrame) -> pd.DataFrame:
    return status[status["required"].astype(bool) & status["status"].ne("available")].copy()


def first_available_path(path: Path) -> Optional[Path]:
    candidates = [path, PATHS.dashboard_data_dir / path.name, APP_DIR / "data" / path.name]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def infer_column(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    lower_lookup = {str(column).lower(): str(column) for column in columns}
    for candidate in candidates:
        if candidate.lower() in lower_lookup:
            return lower_lookup[candidate.lower()]
    for candidate in candidates:
        candidate_lower = candidate.lower()
        for column in columns:
            if candidate_lower in str(column).lower():
                return str(column)
    return None


def normalise_month_column(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    out = frame.copy()
    out[column] = pd.to_datetime(out[column], errors="coerce").dt.to_period("M").dt.to_timestamp()
    return out


@st.cache_data(show_spinner=False)
def load_future_forecasts(path_string: str) -> pd.DataFrame:
    frame = pd.read_parquet(path_string)
    frame = validate_future_forecast_frame(frame, path_string)
    for column in [COLUMNS.p10, COLUMNS.p50, COLUMNS.p90, COLUMNS.latest_observed_waiting_list]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0).clip(lower=0.0)
    return frame


@st.cache_data(show_spinner=False)
def load_historical_observations(path_string: str) -> pd.DataFrame:
    path = Path(path_string)
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_parquet(path).copy()
    value_column = COLUMNS.incomplete_total if COLUMNS.incomplete_total in frame.columns else "waiting_list"
    required = ["month", COLUMNS.trust_code, COLUMNS.trust_name, COLUMNS.specialty_code, COLUMNS.specialty_name, value_column]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{path.name} is missing historical observation columns: {missing}")
    frame = normalise_month_column(frame, "month")
    frame["observed_value"] = pd.to_numeric(frame[value_column], errors="coerce").fillna(0.0).clip(lower=0.0)
    keep_columns = [
        "month",
        COLUMNS.trust_code,
        COLUMNS.trust_name,
        COLUMNS.specialty_code,
        COLUMNS.specialty_name,
        "observed_value",
    ]
    for optional in ["completed_total", "new_rtt_periods", "region", "nhs_region", "provider_region"]:
        if optional in frame.columns:
            keep_columns.append(optional)
    for activity_column in ["completed_total", "new_rtt_periods"]:
        if activity_column in frame.columns:
            frame[activity_column] = pd.to_numeric(frame[activity_column], errors="coerce")
    return frame[keep_columns].dropna(subset=["month"]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_optimisation_forecasts(path_string: str) -> pd.DataFrame:
    frame = pd.read_parquet(path_string)
    frame = validate_optimisation_forecast_frame(frame, path_string)
    for column in [COLUMNS.p10, COLUMNS.p50, COLUMNS.p90, COLUMNS.latest_observed_incomplete_decision_to_admit]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0).clip(lower=0.0)
    return frame


@st.cache_data(show_spinner=False)
def load_backtest_predictions(path_string: str) -> pd.DataFrame:
    path = Path(path_string)
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_parquet(path)
    return validate_backtest_predictions_frame(frame, path_string)


@st.cache_data(show_spinner=False)
def load_csv(path_string: str) -> pd.DataFrame:
    path = Path(path_string)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_parquet(path_string: str) -> pd.DataFrame:
    path = Path(path_string)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def load_markdown(path_string: str) -> str:
    path = Path(path_string)
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


@st.cache_data(show_spinner=False)
def load_capacity_config(path_string: str) -> dict:
    return load_capacity_scenario_config(path_string)


@st.cache_data(show_spinner=False)
def solve_capacity_scenario_cached(
    forecast: pd.DataFrame,
    available_sessions: int,
    scenario_column: str,
    capacity_config: Mapping[str, object],
    productivity_scenario: str,
) -> tuple[pd.DataFrame, dict]:
    allocation, metadata, _ = solve_milp_allocation(
        forecast,
        available_sessions=available_sessions,
        scenario_column=scenario_column,
        capacity_config=capacity_config,
        productivity_scenario=productivity_scenario,
        solver_msg=False,
    )
    return allocation, metadata


@st.cache_data(show_spinner=False)
def load_trust_coordinates(forecast_path_string: str, coordinates_path_string: str) -> pd.DataFrame:
    forecast = pd.read_parquet(forecast_path_string)
    forecast = validate_future_forecast_frame(forecast, "future forecasts")
    coordinate_frames = []

    lat_col = infer_column(forecast.columns, ["trust_latitude", "latitude", "lat", "provider_latitude", "y"])
    lon_col = infer_column(forecast.columns, ["trust_longitude", "longitude", "lon", "lng", "provider_longitude", "x"])
    region_col = infer_column(forecast.columns, ["region", "nhs_region", "provider_region"])
    if lat_col and lon_col:
        keep = [COLUMNS.trust_code, COLUMNS.trust_name, lat_col, lon_col]
        if region_col:
            keep.append(region_col)
        coordinates = forecast[keep].drop_duplicates().rename(columns={lat_col: "latitude", lon_col: "longitude"})
        if region_col:
            coordinates = coordinates.rename(columns={region_col: "region"})
        coordinate_frames.append(coordinates)

    coordinates_path = Path(coordinates_path_string)
    if coordinates_path.exists():
        external = pd.read_csv(coordinates_path)
        trust_code = infer_column(external.columns, ["trust_code", "organisation_code", "org_code", "provider_code"])
        trust_name = infer_column(external.columns, ["trust_name", "provider_name", "organisation_name"])
        latitude = infer_column(external.columns, ["latitude", "lat", "trust_latitude"])
        longitude = infer_column(external.columns, ["longitude", "lon", "lng", "trust_longitude"])
        region = infer_column(external.columns, ["region", "nhs_region", "provider_region"])
        if trust_name and latitude and longitude:
            keep = [trust_name, latitude, longitude]
            if trust_code:
                keep.insert(0, trust_code)
            if region:
                keep.append(region)
            external = external[keep].copy()
            rename_map = {trust_name: COLUMNS.trust_name, latitude: "latitude", longitude: "longitude"}
            if trust_code:
                rename_map[trust_code] = COLUMNS.trust_code
            if region:
                rename_map[region] = "region"
            external = external.rename(columns=rename_map)
            if COLUMNS.trust_code not in external.columns:
                external[COLUMNS.trust_code] = ""
            coordinate_frames.append(external)

    if not coordinate_frames:
        return pd.DataFrame(columns=[COLUMNS.trust_code, COLUMNS.trust_name, "latitude", "longitude", "region"])

    coordinates = pd.concat(coordinate_frames, ignore_index=True)
    coordinates[COLUMNS.trust_name] = coordinates[COLUMNS.trust_name].map(normalise_text)
    coordinates[COLUMNS.trust_code] = coordinates[COLUMNS.trust_code].astype(str).map(normalise_text)
    trust_code_lookup = (
        forecast[[COLUMNS.trust_name, COLUMNS.trust_code]]
        .drop_duplicates(COLUMNS.trust_name)
        .assign(**{COLUMNS.trust_name: lambda df: df[COLUMNS.trust_name].map(normalise_text)})
        .set_index(COLUMNS.trust_name)[COLUMNS.trust_code]
        .to_dict()
    )
    missing_code = coordinates[COLUMNS.trust_code].eq("")
    coordinates.loc[missing_code, COLUMNS.trust_code] = coordinates.loc[missing_code, COLUMNS.trust_name].map(trust_code_lookup)
    coordinates["latitude"] = pd.to_numeric(coordinates["latitude"], errors="coerce")
    coordinates["longitude"] = pd.to_numeric(coordinates["longitude"], errors="coerce")
    if "region" not in coordinates.columns:
        coordinates["region"] = ""
    coordinates["region"] = coordinates["region"].fillna("").astype(str)
    coordinates = coordinates.dropna(subset=[COLUMNS.trust_code, "latitude", "longitude"])
    coordinates = coordinates[coordinates["latitude"].between(49.5, 56.5) & coordinates["longitude"].between(-6.5, 2.5)]
    return coordinates.drop_duplicates(COLUMNS.trust_code, keep="first").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def shap_plot_index(outputs_dir_string: str, dashboard_data_dir_string: str) -> Dict[str, str]:
    paths = []
    for root in [Path(outputs_dir_string), Path(dashboard_data_dir_string), APP_DIR / "data"]:
        paths.extend(glob.glob(str(root / "shap_trust_*.png")))
    mapping = {}
    for path_string in paths:
        path = Path(path_string)
        name = path.stem.replace("shap_trust_", "")
        mapping[normalise_text(name.replace("_", " ")).lower()] = str(path)
    return mapping


def add_region_from_coordinates(frame: pd.DataFrame, coordinates: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "region" in out.columns:
        out["region"] = out["region"].fillna("").astype(str)
        return out
    if coordinates.empty or "region" not in coordinates.columns:
        out["region"] = ""
        return out
    region_lookup = coordinates[[COLUMNS.trust_code, "region"]].drop_duplicates()
    out = out.merge(region_lookup, on=COLUMNS.trust_code, how="left")
    out["region"] = out["region"].fillna("").astype(str)
    return out


def apply_common_filters(
    frame: pd.DataFrame,
    filters: DashboardFilters,
    month_column: str = COLUMNS.forecast_month,
) -> pd.DataFrame:
    out = frame.copy()
    if filters.trust_code is not None and COLUMNS.trust_code in out.columns:
        out = out[out[COLUMNS.trust_code].astype(str).eq(str(filters.trust_code))]
    if filters.specialty_code is not None and COLUMNS.specialty_code in out.columns:
        out = out[out[COLUMNS.specialty_code].astype(str).eq(str(filters.specialty_code))]
    if filters.region != "All regions" and "region" in out.columns:
        out = out[out["region"].astype(str).eq(filters.region)]
    if filters.forecast_month is not None and month_column in out.columns:
        out = out[pd.to_datetime(out[month_column]).dt.to_period("M").dt.to_timestamp().eq(filters.forecast_month)]
    return out


def apply_history_filters(historical: pd.DataFrame, filters: DashboardFilters) -> pd.DataFrame:
    out = historical.copy()
    if filters.trust_code is not None:
        out = out[out[COLUMNS.trust_code].astype(str).eq(str(filters.trust_code))]
    if filters.specialty_code is not None:
        out = out[out[COLUMNS.specialty_code].astype(str).eq(str(filters.specialty_code))]
    if filters.region != "All regions" and "region" in out.columns:
        out = out[out["region"].astype(str).eq(filters.region)]
    return out


def aggregate_forecast(frame: pd.DataFrame, group_columns: Sequence[str], scenario_column: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=list(group_columns) + [COLUMNS.p10, COLUMNS.p50, COLUMNS.p90, scenario_column])
    columns = [COLUMNS.p10, COLUMNS.p50, COLUMNS.p90, COLUMNS.latest_observed_waiting_list]
    return frame.groupby(list(group_columns), as_index=False, observed=True)[columns].sum()


def forecast_metadata(forecast: pd.DataFrame, historical: pd.DataFrame, model_config: Mapping[str, object]) -> dict:
    latest_observed = (
        format_month(pd.to_datetime(historical["month"]).max())
        if not historical.empty and "month" in historical.columns
        else "n/a"
    )
    forecast_origin = (
        format_month(pd.to_datetime(forecast[COLUMNS.forecast_origin]).max())
        if not forecast.empty
        else "n/a"
    )
    forecast_start = pd.to_datetime(forecast[COLUMNS.forecast_month]).min() if not forecast.empty else pd.NaT
    forecast_end = pd.to_datetime(forecast[COLUMNS.forecast_month]).max() if not forecast.empty else pd.NaT
    horizon = int(pd.to_numeric(forecast[COLUMNS.horizon], errors="coerce").max()) if not forecast.empty else 0
    model_class = str(model_config.get("model_class", "")).lower()
    model_label = "Forecast model" if "tcn" in model_class else "Saved forecast model"
    return {
        "latest_observed_month": latest_observed,
        "forecast_origin": forecast_origin,
        "forecast_period": format_month_range(forecast_start, forecast_end),
        "forecast_horizon": f"{horizon} months" if horizon else "n/a",
        "model_version": model_label,
        "dataset_generation_timestamp": file_timestamp(PATHS.clean_parquet),
    }


def render_project_header() -> None:
    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-title">{PROJECT_TITLE}</div>
            <div class="hero-subtitle">{PROJECT_SUBTITLE}</div>
            <div class="author-line">{AUTHOR_NAME}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_copy(text: str) -> None:
    st.markdown(f"<div class='section-copy'>{text}</div>", unsafe_allow_html=True)


def render_forecast_key() -> None:
    st.markdown(
        """
        <div class="forecast-key">
            <div><strong>LOWER-BOUND FORECAST</strong><br>The lower end of the model's forecast range.</div>
            <div><strong>MEDIAN FORECAST</strong><br>The central estimate used when one forecast value is needed.</div>
            <div><strong>UPPER-BOUND FORECAST</strong><br>The upper end of the range used for pressure planning.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_overview_summary_panel(
    observed_period: str,
    forecast_period: str,
    forecast_length: str,
    current_total: float,
    p10_total: float,
    p50_total: float,
    p90_total: float,
    average_forecast: float,
    forecast_month: str,
) -> None:
    growth_value = percent_change(p50_total, current_total)
    growth_delta = compact_change(p50_total - current_total)
    trend_class = "trend-up" if p50_total > current_total else "trend-neutral"
    trend_arrow = "&#8593;" if p50_total > current_total else "&#8595;" if p50_total < current_total else "&#8594;"
    st.markdown(
        f"""
        <div class="summary-panel">
            <div class="summary-grid">
                <div class="summary-item">
                    <div class="summary-label">Observed period</div>
                    <div class="summary-value">{html.escape(observed_period)}</div>
                </div>
                <div class="summary-item">
                    <div class="summary-label">Forecast period</div>
                    <div class="summary-value">{html.escape(forecast_period)}</div>
                </div>
                <div class="summary-item">
                    <div class="summary-label">Forecast length</div>
                    <div class="summary-value">{html.escape(forecast_length)}</div>
                </div>
                <div class="summary-item">
                    <div class="summary-label">Observed waiting list</div>
                    <div class="summary-value">{compact_number(current_total)}</div>
                </div>
                <div class="summary-item">
                    <div class="summary-label">Median forecast in {html.escape(forecast_month)}</div>
                    <div class="summary-value">{compact_number(p50_total)}</div>
                    <div class="summary-delta"><span class="{trend_class}">{trend_arrow}</span> {growth_delta}</div>
                </div>
                <div class="summary-item">
                    <div class="summary-label">Growth from latest</div>
                    <div class="summary-value"><span class="{trend_class}">{trend_arrow}</span> {html.escape(growth_value)}</div>
                    <div class="summary-delta">{growth_delta}</div>
                </div>
                <div class="summary-item">
                    <div class="summary-label">Average forecast per hospital</div>
                    <div class="summary-value">{compact_number(average_forecast)}</div>
                </div>
                <div class="summary-item">
                    <div class="summary-label">Lower-bound forecast</div>
                    <div class="summary-value">{compact_number(p10_total)}</div>
                </div>
                <div class="summary-item">
                    <div class="summary-label">Upper-bound forecast</div>
                    <div class="summary-value">{compact_number(p90_total)}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_setup_status(status: pd.DataFrame) -> None:
    st.title("File check")
    missing = missing_required_files(status)
    if missing.empty:
        st.success("The core dashboard files are available.")
    else:
        st.warning("Some core files are missing. Run the pipeline stage that creates them, then refresh this page.")
        show_table(missing, ["artifact", "status"])
    display = status[["artifact", "required", "status", "modified"]].copy()
    display["required"] = display["required"].map({True: "Yes", False: "No"})
    show_table(display)


def render_header_metadata(metadata: Mapping[str, object]) -> None:
    cols = st.columns(3)
    cols[0].metric("Observed data through", str(metadata["latest_observed_month"]))
    cols[1].metric("Forecast period", str(metadata["forecast_period"]))
    cols[2].metric("Forecast length", str(metadata["forecast_horizon"]))


def render_sidebar_context(metadata: Mapping[str, object]) -> None:
    return None


def sidebar_filters(
    forecast: pd.DataFrame,
    capacity_config: Mapping[str, object],
) -> tuple[str, DashboardFilters]:
    st.sidebar.markdown("## NHS RTT DECISION SUPPORT")
    dark_mode = st.sidebar.toggle("Dark Mode", value=bool(st.session_state.get("dark_mode", False)))
    st.session_state["dark_mode"] = dark_mode
    apply_theme(dark_mode)
    if "page" not in st.session_state:
        st.session_state["page"] = "Overview"
    if st.session_state["page"] not in PAGE_LABELS:
        st.session_state["page"] = "Overview"
    for label in PAGE_LABELS:
        display_label = PAGE_DISPLAY_LABELS.get(label, label.upper())
        if st.session_state["page"] == label:
            st.sidebar.markdown(f"<div class='active-page'>{display_label}</div>", unsafe_allow_html=True)
        else:
            if st.sidebar.button(display_label, key=f"nav_{safe_filename(label)}"):
                st.session_state["page"] = label
                st.rerun()
    page = str(st.session_state["page"])
    scenario_label = "Median forecast"
    productivity_scenarios = sorted((capacity_config.get("productivity_scenarios") or {"central": {}}).keys())
    productivity_scenario = "central" if "central" in productivity_scenarios else productivity_scenarios[0]
    available_sessions = 25_000

    return page, DashboardFilters(
        trust_label="All hospitals",
        trust_code=None,
        specialty_label="All specialties",
        specialty_code=None,
        region="All regions",
        forecast_month=None,
        scenario_label=scenario_label,
        scenario_column=SCENARIO_OPTIONS[scenario_label],
        productivity_scenario=productivity_scenario,
        available_sessions=int(available_sessions),
    )


def make_location_map(forecast: pd.DataFrame, coordinates: pd.DataFrame, filters: DashboardFilters) -> tuple[go.Figure, pd.DataFrame]:
    selected = apply_common_filters(forecast, filters)
    if filters.forecast_month is None and not selected.empty:
        selected = select_display_horizon(selected, preferred_horizon=12)
    trust_values = aggregate_forecast(
        selected,
        [COLUMNS.trust_code, COLUMNS.trust_name],
        filters.scenario_column,
    )
    coordinate_columns = [column for column in coordinates.columns if column != COLUMNS.trust_name]
    map_data = trust_values.merge(coordinates[coordinate_columns].drop_duplicates(COLUMNS.trust_code), on=COLUMNS.trust_code, how="inner")
    if not map_data.empty:
        map_data["hospital_display_name"] = map_data[COLUMNS.trust_name].map(display_hospital_name)
    fig = go.Figure()
    if map_data.empty:
        top = (
            trust_values.sort_values(filters.scenario_column, ascending=False)
            .head(20)
            .sort_values(filters.scenario_column)
        )
        if not top.empty:
            fig.add_trace(
                go.Bar(
                    x=top[filters.scenario_column],
                    y=top[COLUMNS.trust_name].map(display_hospital_name),
                    orientation="h",
                    marker_color=chart_primary_color(),
                    hovertemplate="<b>%{y}</b><br>Forecast: %{x:,.0f}<extra></extra>",
                )
            )
            fig.update_layout(
                title="LARGEST FORECAST WAITING LISTS BY HOSPITAL",
                xaxis_title="Forecast waiting-list size",
                yaxis_title="",
                template=chart_template(),
                height=620,
                margin=dict(l=20, r=20, t=55, b=45),
            )
            return apply_chart_style(fig), map_data
        fig.update_layout(
            height=560,
            annotations=[
                dict(
                    text="No hospital forecast rows match the current filters.",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                    font=dict(size=16),
                )
            ],
        )
        return apply_chart_style(fig), map_data

    fig.add_trace(
        go.Scattermapbox(
            lat=map_data["latitude"],
            lon=map_data["longitude"],
            mode="markers",
            marker=dict(
                size=np.clip(np.sqrt(map_data[filters.scenario_column].clip(lower=0.0)) / 13.0, 8, 28),
                color=map_data[filters.scenario_column],
                colorscale="Blues",
                showscale=False,
                opacity=0.95,
            ),
            text=map_data["hospital_display_name"],
            customdata=map_data[
                [
                    COLUMNS.trust_code,
                    "hospital_display_name",
                    COLUMNS.latest_observed_waiting_list,
                    COLUMNS.p10,
                    COLUMNS.p50,
                    COLUMNS.p90,
                    "region",
                ]
            ].to_numpy(),
            hovertemplate=(
                "<b>%{customdata[1]}</b><br>"
                "Latest observed: %{customdata[2]:,.0f}<br>"
                "Lower-bound forecast: %{customdata[3]:,.0f}<br>"
                "Median forecast: %{customdata[4]:,.0f}<br>"
                "Upper-bound forecast: %{customdata[5]:,.0f}<extra></extra>"
            ),
            name="Hospital location",
            showlegend=False,
        )
    )
    selected_code = st.session_state.get("selected_map_trust_code")
    if selected_code and str(selected_code) in set(map_data[COLUMNS.trust_code].astype(str)):
        selected_row = map_data[map_data[COLUMNS.trust_code].astype(str).eq(str(selected_code))]
        fig.add_trace(
            go.Scattermapbox(
                lat=selected_row["latitude"],
                lon=selected_row["longitude"],
                mode="markers",
                marker=dict(size=30, color="#0f172a" if not st.session_state.get("dark_mode", False) else "#f8fafc", opacity=0.95),
                text=selected_row["hospital_display_name"],
                hovertemplate="<b>%{text}</b><extra></extra>",
                name="Selected hospital",
                showlegend=False,
            )
        )
    fig.update_layout(
        mapbox=dict(
            style=map_style(),
            center=dict(lat=52.75, lon=-1.35),
            zoom=5.55,
            bearing=0,
            pitch=0,
        ),
        title="",
        dragmode="pan",
        hovermode="closest",
        paper_bgcolor=panel_color(),
        plot_bgcolor=panel_color(),
        margin=dict(l=0, r=0, t=0, b=0),
        height=520,
        uirevision="england-hospital-location-map",
    )
    return apply_chart_style(fig), map_data


def selected_trust_code_from_event(event: object) -> Optional[str]:
    points: list[object] = []
    if isinstance(event, dict):
        points = (((event.get("selection") or {}).get("points")) or [])
    else:
        selection = getattr(event, "selection", None)
        points = getattr(selection, "points", []) if selection is not None else []
    if not points:
        return None
    point = points[0]
    if isinstance(point, dict):
        customdata = point.get("customdata") or []
    else:
        customdata = getattr(point, "customdata", []) or []
    if len(customdata) == 0:
        return None
    return str(customdata[0])


def render_selected_map_trust(
    event: object,
    map_data: pd.DataFrame,
    forecast: pd.DataFrame,
    historical: pd.DataFrame,
    filters: DashboardFilters,
) -> None:
    if map_data.empty:
        return
    clicked_code = selected_trust_code_from_event(event)
    if clicked_code:
        st.session_state["selected_map_trust_code"] = clicked_code
    selected_code = clicked_code or filters.trust_code or st.session_state.get("selected_map_trust_code")
    if not selected_code or selected_code not in set(map_data[COLUMNS.trust_code].astype(str)):
        selected_code = str(map_data.sort_values(filters.scenario_column, ascending=False).iloc[0][COLUMNS.trust_code])
    st.session_state["selected_map_trust_code"] = selected_code

    hospital_filters = replace(filters, trust_code=selected_code, specialty_code=None)
    hospital_forecasts = apply_common_filters(forecast, hospital_filters)
    if filters.forecast_month is None and not hospital_forecasts.empty:
        hospital_forecasts = select_display_horizon(hospital_forecasts, preferred_horizon=12)
    if hospital_forecasts.empty:
        return
    selected_scenario_label = filters.scenario_label
    selected_scenario_column = filters.scenario_column
    selected_forecast = hospital_forecasts
    trust_name = selected_forecast[COLUMNS.trust_name].iloc[0]
    hospital_name = display_hospital_name(trust_name)
    latest = float(selected_forecast[COLUMNS.latest_observed_waiting_list].sum())
    p10 = float(selected_forecast[COLUMNS.p10].sum())
    p50 = float(selected_forecast[COLUMNS.p50].sum())
    p90 = float(selected_forecast[COLUMNS.p90].sum())
    selected_value = float(selected_forecast[selected_scenario_column].sum())
    month_label = pd.to_datetime(selected_forecast[COLUMNS.forecast_month]).max().strftime("%B %Y")
    st.markdown("<div class='section-title'>SELECTED HOSPITAL</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='detail-panel'><strong>{hospital_name}</strong><br>"
        f"NHS organisation code: {selected_code}<br>"
        f"Waiting-list forecast month: {month_label}</div>",
        unsafe_allow_html=True,
    )
    cols = st.columns(2)
    cols[0].metric("Latest observed", compact_number(latest))
    forecast_delta = f"{percent_change(selected_value, latest)} ({compact_change(selected_value - latest)})"
    cols[1].metric(selected_scenario_label, compact_number(selected_value), forecast_delta, delta_color="inverse")
    cols = st.columns(2)
    cols[0].metric("Lower-bound forecast", compact_number(p10))
    cols[1].metric("Upper-bound forecast", compact_number(p90))


def render_selected_map_specialties(forecast: pd.DataFrame, filters: DashboardFilters) -> None:
    selected_code = filters.trust_code or st.session_state.get("selected_map_trust_code")
    if not selected_code or filters.scenario_label not in SCENARIO_OPTIONS:
        return
    selected_scenario_label = filters.scenario_label
    selected_scenario_column = filters.scenario_column
    selected_filters = replace(
        filters,
        trust_code=str(selected_code),
        specialty_code=None,
    )
    selected_forecast = apply_common_filters(forecast, selected_filters)
    if filters.forecast_month is None and not selected_forecast.empty:
        selected_forecast = select_display_horizon(selected_forecast, preferred_horizon=12)
    if selected_forecast.empty:
        return
    specialty = (
        selected_forecast.groupby([COLUMNS.specialty_code, COLUMNS.specialty_name], as_index=False, observed=True)[
            [COLUMNS.p10, COLUMNS.p50, COLUMNS.p90, COLUMNS.latest_observed_waiting_list]
        ]
        .sum()
        .sort_values(selected_scenario_column, ascending=False)
        .head(10)
    )
    if not specialty.empty:
        specialty["specialty_display"] = specialty[COLUMNS.specialty_name].map(display_specialty_name)
        fig = go.Figure(
            go.Bar(
                x=specialty[selected_scenario_column],
                y=specialty["specialty_display"],
                orientation="h",
                marker_color=chart_primary_color(),
                hovertemplate=f"<b>%{{y}}</b><br>{selected_scenario_label}: %{{x:,.0f}}<extra></extra>",
            )
        )
        fig.update_layout(
            title="LARGEST SPECIALTIES IN THE SELECTED HOSPITAL",
            xaxis_title=f"{selected_scenario_label} waiting-list size",
            yaxis_title="",
            template=chart_template(),
            height=390,
            margin=dict(l=20, r=20, t=50, b=35),
        )
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(apply_chart_style(fig), use_container_width=True, config=PLOTLY_CONFIG)
        st.markdown("<div class='section-title'>SPECIALTY VALUES</div>", unsafe_allow_html=True)
        show_table(
            specialty,
            [
                COLUMNS.specialty_name,
                COLUMNS.latest_observed_waiting_list,
                COLUMNS.p10,
                COLUMNS.p50,
                COLUMNS.p90,
            ],
            max_height=390,
        )


def make_history_forecast_chart(historical: pd.DataFrame, forecast: pd.DataFrame, filters: DashboardFilters, title: str) -> go.Figure:
    hist = apply_history_filters(historical, filters)
    fut = apply_common_filters(forecast, filters)
    hist_series = (
        hist.groupby("month", as_index=False, observed=True)["observed_value"].sum().sort_values("month")
        if not hist.empty
        else pd.DataFrame(columns=["month", "observed_value"])
    )
    future_series = (
        fut.groupby(COLUMNS.forecast_month, as_index=False, observed=True)[[COLUMNS.p10, COLUMNS.p50, COLUMNS.p90]].sum()
        .sort_values(COLUMNS.forecast_month)
        if not fut.empty
        else pd.DataFrame(columns=[COLUMNS.forecast_month, COLUMNS.p10, COLUMNS.p50, COLUMNS.p90])
    )

    fig = go.Figure()
    if not hist_series.empty:
        hist_series["hover_month"] = pd.to_datetime(hist_series["month"]).map(format_month)
        fig.add_trace(
            go.Scatter(
                x=hist_series["month"],
                y=hist_series["observed_value"],
                mode="lines",
                name="Observed NHS data",
                line=dict(color=observed_line_color(), width=2.4),
                customdata=hist_series["hover_month"],
                hovertemplate="%{customdata}<br>Observed waiting list: %{y:,.0f}<extra></extra>",
            )
        )
    if not future_series.empty:
        future_series["hover_month"] = pd.to_datetime(future_series[COLUMNS.forecast_month]).map(format_month)
        band_color = "rgba(120, 189, 248, 0.28)" if st.session_state.get("dark_mode", False) else "rgba(37, 99, 235, 0.16)"
        band_line = "rgba(120, 189, 248, 0.35)" if st.session_state.get("dark_mode", False) else "rgba(37, 99, 235, 0.18)"
        median_line = "#92d0ff" if st.session_state.get("dark_mode", False) else "#1d4ed8"
        fig.add_trace(
            go.Scatter(
                x=future_series[COLUMNS.forecast_month],
                y=future_series[COLUMNS.p90],
                mode="lines",
                line=dict(color=band_line, width=1),
                showlegend=False,
                name="Upper-bound forecast",
                customdata=future_series["hover_month"],
                hovertemplate="%{customdata}<br>Upper-bound forecast: %{y:,.0f}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=future_series[COLUMNS.forecast_month],
                y=future_series[COLUMNS.p10],
                mode="lines",
                fill="tonexty",
                fillcolor=band_color,
                line=dict(color=band_line, width=1),
                name="Forecast range",
                customdata=future_series["hover_month"],
                hovertemplate="%{customdata}<br>Lower-bound forecast: %{y:,.0f}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=future_series[COLUMNS.forecast_month],
                y=future_series[COLUMNS.p50],
                mode="lines",
                name="Median forecast",
                line=dict(color=median_line, width=3.0),
                customdata=future_series["hover_month"],
                hovertemplate="%{customdata}<br>Median forecast: %{y:,.0f}<extra></extra>",
            )
        )
    fig.update_layout(
        title=title.upper(),
        xaxis_title="Year",
        yaxis_title="Total incomplete waiting list",
        hovermode="x unified",
        template=chart_template(),
        height=470,
        margin=dict(l=20, r=20, t=60, b=40),
    )
    visible_values: list[float] = []
    if not hist_series.empty:
        visible_values.extend(pd.to_numeric(hist_series["observed_value"], errors="coerce").dropna().astype(float).tolist())
    if not future_series.empty:
        visible_values.extend(pd.to_numeric(future_series[COLUMNS.p10], errors="coerce").dropna().astype(float).tolist())
        visible_values.extend(pd.to_numeric(future_series[COLUMNS.p90], errors="coerce").dropna().astype(float).tolist())
    if visible_values:
        low = min(visible_values)
        high = max(visible_values)
        padding = max((high - low) * 0.12, high * 0.03, 1.0)
        fig.update_yaxes(range=[max(0.0, low - padding), high + padding])
    fig.update_xaxes(tickformat="%Y")
    return apply_chart_style(fig)


def local_signal_rows(local_explanations: pd.DataFrame, filters: DashboardFilters, preferred_horizon: int = 12) -> pd.DataFrame:
    if local_explanations.empty or filters.trust_code is None:
        return pd.DataFrame()
    rows = local_explanations.copy()
    if COLUMNS.trust_code in rows.columns:
        rows = rows[rows[COLUMNS.trust_code].astype(str).eq(str(filters.trust_code))]
    if filters.specialty_code is not None and COLUMNS.specialty_code in rows.columns:
        rows = rows[rows[COLUMNS.specialty_code].astype(str).eq(str(filters.specialty_code))]
    if rows.empty:
        return rows
    horizons = pd.to_numeric(rows.get(COLUMNS.horizon, pd.Series(dtype=float)), errors="coerce")
    if horizons.eq(preferred_horizon).any():
        rows = rows[horizons.eq(preferred_horizon)]
    elif horizons.notna().any():
        rows = rows[horizons.eq(horizons.max())]
    group_col = "feature_group" if "feature_group" in rows.columns else rows.columns[0]
    contribution_col = "mean_shap_value" if "mean_shap_value" in rows.columns else rows.select_dtypes("number").columns[-1]
    abs_col = "mean_abs_shap_value" if "mean_abs_shap_value" in rows.columns else contribution_col
    grouped = (
        rows.groupby(group_col, as_index=False, observed=True)
        .agg(
            mean_shap_value=(contribution_col, "sum"),
            mean_abs_shap_value=(abs_col, "sum"),
            mean_model_p50=("mean_model_p50", "mean") if "mean_model_p50" in rows.columns else (contribution_col, "size"),
        )
    )
    grouped["feature_group_display"] = grouped[group_col].map(clean_feature_group)
    grouped["absolute_contribution"] = grouped["mean_shap_value"].abs()
    return grouped.sort_values("absolute_contribution", ascending=True)


def render_local_signal_chart(local_explanations: pd.DataFrame, filters: DashboardFilters) -> bool:
    rows = local_signal_rows(local_explanations, filters)
    if rows.empty:
        return False
    colors = np.where(rows["mean_shap_value"] >= 0, chart_primary_color(), chart_warning_color())
    fig = go.Figure(
        go.Bar(
            x=rows["mean_shap_value"],
            y=rows["feature_group_display"],
            orientation="h",
            marker_color=colors,
            hovertemplate="%{x:,.1f}<extra></extra>",
        )
    )
    fig.add_vline(x=0, line_color=observed_line_color(), line_width=1)
    fig.update_layout(
        title="MODEL SIGNALS FOR THE 12-MONTH WAITING-LIST FORECAST",
        xaxis_title="Contribution to the median forecast",
        yaxis_title="",
        template=chart_template(),
        height=430,
        margin=dict(l=20, r=20, t=60, b=45),
    )
    st.plotly_chart(apply_chart_style(fig), use_container_width=True, config=PLOTLY_CONFIG)
    st.markdown("<div class='section-title'>MODEL SIGNAL SUMMARY</div>", unsafe_allow_html=True)
    section_copy(
        "The table lists the same grouped signals shown in the chart. Positive values lift the model forecast, while negative values lower it for the selected explanation case."
    )
    show_table(rows.sort_values("absolute_contribution", ascending=False), ["feature_group_display", "mean_shap_value", "mean_abs_shap_value", "mean_model_p50"])
    return True


def render_national_overview(
    forecast: pd.DataFrame,
    historical: pd.DataFrame,
    coordinates: pd.DataFrame,
    filters: DashboardFilters,
    metadata: Mapping[str, object],
) -> None:
    render_project_header()
    st.title("NATIONAL OVERVIEW")
    section_copy(
        "The national RTT waiting-list history is shown from the first loaded month, followed by a 12-month forecast. "
        "Choose a region, hospital, and forecast month before reviewing the projections. "
        "The forecast estimate selector is placed beside the selected-hospital chart below."
    )
    regions = sorted([region for region in forecast.get("region", pd.Series(dtype=str)).astype(str).unique() if region])
    control_cols = st.columns([1.0, 1.35, 1.0])
    region = control_cols[0].selectbox(
        "1. Region",
        ["All regions"] + regions,
        index=0,
        key="overview_region",
    ) if regions else "All regions"
    region_rows = forecast.copy()
    if region != "All regions" and "region" in region_rows.columns:
        region_rows = region_rows[region_rows["region"].astype(str).eq(region)]
    trust_options = (
        region_rows[[COLUMNS.trust_code, COLUMNS.trust_name]]
        .drop_duplicates()
        .sort_values(COLUMNS.trust_name)
        .reset_index(drop=True)
    )
    hospital_labels = ["All hospitals"] + [
        f"{display_hospital_name(row.trust_name)} ({row.trust_code})"
        for row in trust_options.itertuples(index=False)
    ]
    hospital_label = control_cols[1].selectbox(
        "2. Hospital",
        hospital_labels,
        index=0,
        key="overview_scope_hospital",
    )
    trust_code = None if hospital_label == "All hospitals" else str(
        trust_options.iloc[hospital_labels.index(hospital_label) - 1][COLUMNS.trust_code]
    )
    scope_rows = region_rows if trust_code is None else region_rows[region_rows[COLUMNS.trust_code].astype(str).eq(trust_code)]
    month_options = sorted(pd.to_datetime(scope_rows[COLUMNS.forecast_month]).dt.to_period("M").dt.to_timestamp().unique())
    month_labels = [format_month(month) for month in month_options]
    selected_month_label = control_cols[2].selectbox(
        "3. Forecast month",
        month_labels,
        index=len(month_labels) - 1,
        key="overview_forecast_month",
    )
    forecast_month_value = pd.Timestamp(month_options[month_labels.index(selected_month_label)])
    scenario_label = str(st.session_state.get("overview_selected_chart_estimate", filters.scenario_label))
    if scenario_label not in SCENARIO_OPTIONS:
        scenario_label = "Median forecast"
    filters = replace(
        filters,
        trust_label=hospital_label,
        trust_code=trust_code,
        scenario_label=scenario_label,
        scenario_column=SCENARIO_OPTIONS[scenario_label],
        region=region,
        forecast_month=forecast_month_value,
    )
    if trust_code is not None:
        st.session_state["selected_map_trust_code"] = trust_code
    render_forecast_key()
    selected = apply_common_filters(forecast, filters)
    if selected.empty:
        st.info("No forecast rows match the current filters.")
        return
    full_forecast_scope = apply_common_filters(forecast, replace(filters, forecast_month=None))

    hist_for_period = apply_history_filters(historical, filters)
    observed_period = (
        format_month_range(hist_for_period["month"].min(), hist_for_period["month"].max())
        if not hist_for_period.empty and "month" in hist_for_period.columns
        else str(metadata.get("latest_observed_month", "n/a"))
    )
    forecast_period = (
        format_month_range(full_forecast_scope[COLUMNS.forecast_month].min(), full_forecast_scope[COLUMNS.forecast_month].max())
        if COLUMNS.forecast_month in full_forecast_scope.columns and not full_forecast_scope.empty
        else str(metadata.get("forecast_period", "n/a"))
    )
    horizon_count = int(pd.to_numeric(full_forecast_scope.get(COLUMNS.horizon, pd.Series(dtype=float)), errors="coerce").dropna().nunique())
    forecast_length = f"{horizon_count} months" if horizon_count else str(metadata.get("forecast_horizon", "12 months"))

    selected_month = selected
    current_total = float(selected_month[COLUMNS.latest_observed_waiting_list].sum())
    p10_total = float(selected_month[COLUMNS.p10].sum())
    p50_total = float(selected_month[COLUMNS.p50].sum())
    p90_total = float(selected_month[COLUMNS.p90].sum())
    selected_total = float(selected_month[filters.scenario_column].sum())
    hospital_count = int(selected_month[COLUMNS.trust_code].nunique()) if COLUMNS.trust_code in selected_month.columns else 0
    average_forecast = selected_total / hospital_count if hospital_count else np.nan
    forecast_month = (
        format_month(pd.to_datetime(selected_month[COLUMNS.forecast_month]).max())
        if COLUMNS.forecast_month in selected_month.columns and not selected_month.empty
        else "selected month"
    )
    render_overview_summary_panel(
        observed_period=observed_period,
        forecast_period=forecast_period,
        forecast_length=forecast_length,
        current_total=current_total,
        p10_total=p10_total,
        p50_total=p50_total,
        p90_total=p90_total,
        average_forecast=average_forecast,
        forecast_month=forecast_month,
    )

    section_copy(
        "The line chart keeps observed NHS data and model forecasts separate. The shaded area is the forecast range, and the central line is the median forecast."
    )
    st.plotly_chart(
        make_history_forecast_chart(
            historical,
            forecast,
            replace(filters, forecast_month=None),
            "OBSERVED WAITING LIST AND 12-MONTH FORECAST",
        ),
        use_container_width=True,
        config=PLOTLY_CONFIG,
    )

    if coordinates.empty:
        st.markdown("<div class='plain-note'>Map coordinates are still being prepared, so the view below ranks hospitals by forecast size.</div>", unsafe_allow_html=True)
    map_fig, map_data = make_location_map(forecast, coordinates, filters)
    st.subheader("HOSPITAL LOCATION MAP")
    section_copy("Each point is an NHS organisation location in England. Selecting a point updates the hospital summary beside the map.")
    map_col, detail_col = st.columns([1.35, 1.0], gap="large")
    with map_col:
        map_event = st.plotly_chart(
            map_fig,
            use_container_width=True,
            config=MAP_PLOTLY_CONFIG,
            key="trust_location_map",
            on_select="rerun" if not map_data.empty else "ignore",
            selection_mode="points",
        )
    with detail_col:
        render_selected_map_trust(map_event, map_data, forecast, historical, filters)
    st.subheader("SELECTED-HOSPITAL SPECIALTY FORECAST")
    section_copy(
        "This section breaks the selected hospital forecast into specialties for the selected month. "
        "Changing the estimate below switches the chart between the lower-bound, median, and upper-bound forecast."
    )
    estimate_col, spacer_col = st.columns([1.0, 2.0])
    selected_chart_label = estimate_col.selectbox(
        "Forecast estimate",
        list(SCENARIO_OPTIONS.keys()),
        index=list(SCENARIO_OPTIONS.keys()).index(scenario_label),
        key="overview_selected_chart_estimate",
    )
    filters = replace(
        filters,
        scenario_label=selected_chart_label,
        scenario_column=SCENARIO_OPTIONS[selected_chart_label],
    )
    render_selected_map_specialties(forecast, filters)


def render_deep_dive(forecast: pd.DataFrame, historical: pd.DataFrame, filters: DashboardFilters) -> None:
    st.title("HOSPITAL PROFILE")
    section_copy(
        "The selected hospital is shown with its observed RTT waiting-list history and the next 12 forecast months. "
        "The shaded forecast range shows uncertainty around the median forecast."
    )
    filters = page_hospital_specialty_filters(
        forecast,
        filters,
        key_prefix="hospital_profile",
        preferred_code=preferred_trust_code(forecast),
    )
    if filters.trust_code is None:
        st.info("No hospital forecast is available for this view.")
        return
    series_filters = replace(filters, forecast_month=None)
    selected = apply_common_filters(forecast, series_filters)
    if selected.empty:
        st.info("No forecast rows match this hospital and specialty selection.")
        return
    trust_name = selected[COLUMNS.trust_name].iloc[0]
    hospital_name = display_hospital_name(trust_name)
    subtitle = hospital_name if filters.specialty_code is None else f"{hospital_name}: {display_specialty_name(selected[COLUMNS.specialty_name].iloc[0])}"
    current = selected[selected[COLUMNS.horizon].eq(selected[COLUMNS.horizon].min())][COLUMNS.latest_observed_waiting_list].sum()
    cols = st.columns(4)
    cols[0].metric("Latest observed", compact_number(current))
    for horizon, column in zip([3, 6, 12], cols[1:]):
        rows = selected[selected[COLUMNS.horizon].eq(horizon)]
        value = rows[COLUMNS.p50].sum() if not rows.empty else np.nan
        column.metric(f"Forecast at {horizon} months", compact_number(value))
    st.plotly_chart(
        make_history_forecast_chart(historical, forecast, series_filters, f"{subtitle}: observed history and forecast"),
        use_container_width=True,
        config=PLOTLY_CONFIG,
    )

    local_explanations = load_parquet(str(PATHS.shap_local_explanations))
    if not local_signal_rows(local_explanations, filters).empty:
        st.subheader("MODEL SIGNALS FOR THIS HOSPITAL")
        section_copy("These bars show recorded features that were associated with the selected model forecast. They should not be read as proof of cause.")
        render_local_signal_chart(local_explanations, filters)

    detail_columns = [
        COLUMNS.trust_name,
        COLUMNS.specialty_name,
        COLUMNS.forecast_month,
        COLUMNS.horizon,
        COLUMNS.p10,
        COLUMNS.p50,
        COLUMNS.p90,
        COLUMNS.latest_observed_waiting_list,
    ]
    st.subheader("FORECAST TABLE")
    section_copy("The table lists the forecast months, forecast range, and the latest observed waiting-list value used as the starting point.")
    show_table(selected[detail_columns].sort_values([COLUMNS.specialty_name, COLUMNS.forecast_month]), max_height=460)


def render_model_performance(filters: DashboardFilters) -> None:
    st.title("MODEL COMPARISON")
    section_copy(
        "The TCN is compared with simpler forecasting baselines using the same held-out forecast rows. "
        "Lower error values indicate closer forecasts."
    )
    comparison = load_csv(str(PATHS.model_comparison))
    by_horizon = load_csv(str(PATHS.model_comparison_by_horizon))
    rolling_reliability = load_csv(str(PATHS.rolling_origin_reliability))
    rolling_overall = load_csv(str(PATHS.rolling_origin_metrics))

    if comparison.empty:
        st.info("Model comparison metrics have not been generated yet.")
    else:
        st.subheader("MODEL COMPARED WITH SIMPLE BASELINES")
        section_copy(
            "MAE is the average size of the forecast error. RMSE gives more weight to large errors. "
            "Quantile error, commonly called pinball loss, checks the lower-bound, median, and upper-bound forecasts separately. "
            "Lower values are better for all three measures."
        )
        show_table(comparison, max_height=300)
        metric = "mae" if "mae" in comparison.columns else comparison.select_dtypes("number").columns[0]
        ordered = comparison.sort_values(metric, ascending=True).copy()
        ordered["model_display"] = ordered["model"].map(display_model_name)
        fig = go.Figure(go.Bar(x=ordered["model_display"], y=ordered[metric], marker_color=chart_primary_color(), hovertemplate="%{y:,.1f}<extra></extra>"))
        fig.update_layout(title=f"OVERALL {metric.upper()} BY MODEL", xaxis_title="Model", yaxis_title=metric.upper(), template=chart_template())
        st.plotly_chart(apply_chart_style(fig), use_container_width=True, config=PLOTLY_CONFIG)

    if not by_horizon.empty:
        st.subheader("ERROR BY FORECAST HORIZON")
        section_copy("This view shows whether error grows as the forecast moves further away from the forecast origin.")
        fig = go.Figure()
        for model_name, group in by_horizon.groupby("model", observed=True):
            group = group.sort_values(COLUMNS.horizon)
            fig.add_trace(go.Scatter(x=group[COLUMNS.horizon], y=group["mae"], mode="lines+markers", name=display_model_name(model_name), hovertemplate="%{y:,.1f}<extra></extra>"))
        fig.update_layout(xaxis_title="Horizon", yaxis_title="MAE", template=chart_template(), height=430)
        st.plotly_chart(apply_chart_style(fig), use_container_width=True, config=PLOTLY_CONFIG)
        show_table(by_horizon, max_height=460)

    st.subheader("FORECAST RANGE COVERAGE")
    section_copy("Coverage shows how often actual values fell between the lower-bound and upper-bound forecasts. The dashed line marks the nominal 80% range.")
    coverage_rows = []
    if not comparison.empty and "p10_p90_coverage" in comparison.columns:
        coverage_rows.append(comparison[["model", "p10_p90_coverage", "average_interval_width"]])
    if not rolling_reliability.empty:
        st.markdown("<div class='quiet'>Rolling-origin results use several forecast start dates.</div>", unsafe_allow_html=True)
        show_table(rolling_reliability)
    if not rolling_overall.empty:
        show_table(rolling_overall)
    if coverage_rows:
        coverage = pd.concat(coverage_rows, ignore_index=True)
        coverage = coverage.copy()
        coverage["model_display"] = coverage["model"].map(display_model_name)
        fig = go.Figure(go.Bar(x=coverage["model_display"], y=coverage["p10_p90_coverage"], marker_color=chart_positive_color(), hovertemplate="%{y:.1%}<extra></extra>"))
        fig.add_hline(y=0.80, line_dash="dash", line_color=observed_line_color(), annotation_text="Nominal 80%")
        fig.update_layout(title="FORECAST RANGE COVERAGE", yaxis_tickformat=".0%", template=chart_template())
        st.plotly_chart(apply_chart_style(fig), use_container_width=True, config=PLOTLY_CONFIG)


def find_shap_waterfall(
    trust_code: Optional[str],
    trust_name: str,
    plot_paths: Mapping[str, str],
    waterfall_index: pd.DataFrame,
) -> Optional[str]:
    if not waterfall_index.empty and "plot_path" in waterfall_index.columns:
        rows = waterfall_index.copy()
        if trust_code and COLUMNS.trust_code in rows.columns:
            code_rows = rows[rows[COLUMNS.trust_code].astype(str).eq(str(trust_code))]
            if not code_rows.empty:
                path = str(code_rows["plot_path"].iloc[0])
                if Path(path).exists():
                    return path
        if COLUMNS.trust_name in rows.columns:
            name_rows = rows[
                rows[COLUMNS.trust_name]
                .astype(str)
                .map(normalise_text)
                .eq(normalise_text(trust_name))
            ]
            if not name_rows.empty:
                path = str(name_rows["plot_path"].iloc[0])
                if Path(path).exists():
                    return path

    direct = normalise_text(safe_filename(trust_name).replace("_", " ")).lower()
    if direct in plot_paths:
        return plot_paths[direct]
    trust_key = normalise_text(trust_name).lower()
    for key, path in plot_paths.items():
        if key == trust_key or key in trust_key or trust_key in key:
            return path
    return None


def render_explainability(forecast: pd.DataFrame, filters: DashboardFilters) -> None:
    st.title("FORECAST DRIVERS")
    section_copy(
        "These views show which recorded history was used most strongly by the forecasting model. "
        "They describe model behaviour and should not be read as proof of cause."
    )
    global_importance = load_csv(str(PATHS.shap_global_importance))
    group_importance = load_csv(str(PATHS.shap_group_importance))
    local_explanations = load_parquet(str(PATHS.shap_local_explanations))
    if not local_explanations.empty:
        st.subheader("EXPLAIN A HOSPITAL'S FORECAST")
        section_copy("The hospital list contains the cases for which local model explanations were generated.")
        filters = page_hospital_specialty_filters(
            local_explanations,
            filters,
            key_prefix="forecast_drivers",
            preferred_code="RA2",
        )

    if global_importance.empty:
        st.info("Global feature importance has not been generated yet.")
    else:
        st.subheader("OVERALL MODEL SIGNALS")
        section_copy(
            "The chart ranks the historical inputs with the largest average contribution across the explanation sample. "
            "It is a model summary, not a claim that one item caused the waiting list to change."
        )
        value_col = "mean_abs_shap" if "mean_abs_shap" in global_importance.columns else (
            "mean_abs_shap_value" if "mean_abs_shap_value" in global_importance.columns else global_importance.select_dtypes("number").columns[-1]
        )
        top = global_importance.sort_values(value_col, ascending=False).head(18).copy()
        label_source = "display_name" if "display_name" in top.columns else ("feature_name" if "feature_name" in top.columns else top.columns[0])
        top["feature_label"] = [
            clean_feature_label(name, lag)
            for name, lag in zip(top[label_source], top["lag"] if "lag" in top.columns else [None] * len(top))
        ]
        top = top.sort_values(value_col)
        fig = go.Figure(
            go.Bar(
                x=top[value_col],
                y=top["feature_label"],
                orientation="h",
                marker_color=chart_primary_color(),
                hovertemplate="%{x:,.1f}<extra></extra>",
            )
        )
        fig.update_layout(
            title="LARGEST MODEL SIGNALS ACROSS ALL HOSPITALS",
            xaxis_title="Average contribution size",
            yaxis_title="",
            template=chart_template(),
            height=620,
            margin=dict(l=20, r=20, t=60, b=45),
        )
        st.plotly_chart(apply_chart_style(fig), use_container_width=True, config=PLOTLY_CONFIG)
        st.markdown("<div class='section-title'>TOP HISTORICAL INPUTS</div>", unsafe_allow_html=True)
        section_copy(
            "The table gives the same ranking with readable feature names. Larger contribution sizes mean the model relied more strongly on that recorded input."
        )
        show_table(
            top.sort_values(value_col, ascending=False),
            ["feature_label", value_col, "feature_group"],
            max_rows=25,
            max_height=460,
        )

    if group_importance.empty:
        st.info("Grouped feature importance has not been generated yet.")
    else:
        st.subheader("HISTORICAL INFORMATION GROUPS")
        section_copy(
            "The detailed model inputs are collected into readable groups: waiting-list history, new RTT periods and net inflow, "
            "completed pathways, unreported removals, time and calendar patterns, and hospital or specialty patterns."
        )
        grouped = group_importance.copy()
        horizon_values = sorted(pd.to_numeric(grouped.get(COLUMNS.horizon, pd.Series(dtype=float)), errors="coerce").dropna().astype(int).unique().tolist())
        selected_horizon = None
        if COLUMNS.horizon in grouped.columns and horizon_values:
            default_horizon = 12 if 12 in horizon_values else horizon_values[-1]
            selected_horizon = st.selectbox(
                "Forecast month ahead",
                horizon_values,
                index=horizon_values.index(default_horizon),
                key="model_signal_group_horizon",
            )
            grouped = grouped[pd.to_numeric(grouped[COLUMNS.horizon], errors="coerce").eq(selected_horizon)]
        value_col = "mean_abs_shap" if "mean_abs_shap" in grouped.columns else (
            "mean_abs_shap_value" if "mean_abs_shap_value" in grouped.columns else grouped.select_dtypes("number").columns[-1]
        )
        group_col = "feature_group" if "feature_group" in grouped.columns else grouped.columns[0]
        grouped = grouped.sort_values(value_col, ascending=False).copy()
        grouped["feature_group_display"] = grouped[group_col].map(clean_feature_group)
        grouped_plot = grouped.sort_values(value_col)
        fig = go.Figure(
            go.Bar(
                x=grouped_plot[value_col],
                y=grouped_plot["feature_group_display"],
                orientation="h",
                marker_color=chart_positive_color(),
                hovertemplate="%{x:,.1f}<extra></extra>",
            )
        )
        fig.update_layout(
            title="MODEL SIGNALS BY FEATURE GROUP",
            xaxis_title="Average contribution size",
            yaxis_title="",
            template=chart_template(),
            height=430,
            margin=dict(l=20, r=20, t=60, b=45),
        )
        st.plotly_chart(apply_chart_style(fig), use_container_width=True, config=PLOTLY_CONFIG)
        st.markdown("<div class='section-title'>GROUP CONTRIBUTION VALUES</div>", unsafe_allow_html=True)
        if selected_horizon is not None:
            section_copy(
                f"The table shows grouped contribution values for month {selected_horizon} ahead. "
                "Choose a different month above to inspect how the model's reliance changes across the forecast horizon."
            )
        else:
            section_copy("The table shows grouped contribution values for the available explanation sample.")
        show_table(grouped, ["feature_group_display", value_col, COLUMNS.horizon], max_rows=25)

    if local_explanations.empty:
        return
    local_filters = filters
    rows = local_signal_rows(local_explanations, local_filters)
    if rows.empty:
        return
    matched = local_explanations[local_explanations[COLUMNS.trust_code].astype(str).eq(str(local_filters.trust_code))]
    local_title = display_hospital_name(matched[COLUMNS.trust_name].iloc[0]) if COLUMNS.trust_name in matched.columns and not matched.empty else "Selected hospital"
    st.subheader(f"HOSPITAL MODEL SIGNALS: {local_title.upper()}")
    section_copy("This hospital-level view is shown only when local explanation rows are available for a valid forecast case.")
    render_local_signal_chart(local_explanations, local_filters)


def render_capacity_optimisation(optimisation_forecast: pd.DataFrame, filters: DashboardFilters, capacity_config: Mapping[str, object]) -> None:
    st.title("CAPACITY TEST")
    section_copy(
        "This simulator distributes a chosen number of additional treatment sessions across forecast decision-to-admit waiting lists. "
        "It shows how much backlog could be reduced under the selected assumptions; it is not an NHS operational recommendation."
    )

    st.subheader("SCENARIO SETUP")
    section_copy(
        "Choose the hospitals and specialties included in the simulation, then set the number of extra treatment sessions to test."
    )
    trust_options = (
        optimisation_forecast[[COLUMNS.trust_code, COLUMNS.trust_name]]
        .drop_duplicates()
        .sort_values(COLUMNS.trust_name)
        .reset_index(drop=True)
    )
    trust_labels = ["All hospitals"] + [
        f"{display_hospital_name(row.trust_name)} ({row.trust_code})" for row in trust_options.itertuples(index=False)
    ]
    specialty_options = (
        optimisation_forecast[[COLUMNS.specialty_code, COLUMNS.specialty_name]]
        .drop_duplicates()
        .sort_values(COLUMNS.specialty_name)
        .reset_index(drop=True)
    )
    specialty_labels = ["All specialties"] + [
        f"{display_specialty_name(row.specialty_name)} ({row.specialty_code})" for row in specialty_options.itertuples(index=False)
    ]
    regions = sorted([region for region in optimisation_forecast.get("region", pd.Series(dtype=str)).astype(str).unique() if region])

    filter_cols = st.columns(3)
    trust_label = filter_cols[0].selectbox("Hospital", trust_labels, index=0, key="capacity_hospital")
    specialty_label = filter_cols[1].selectbox("Specialty", specialty_labels, index=0, key="capacity_specialty")
    region = filter_cols[2].selectbox("Region", ["All regions"] + regions, index=0, key="capacity_region") if regions else "All regions"
    trust_code = None if trust_label == "All hospitals" else str(trust_options.iloc[trust_labels.index(trust_label) - 1][COLUMNS.trust_code])
    specialty_code = None if specialty_label == "All specialties" else str(
        specialty_options.iloc[specialty_labels.index(specialty_label) - 1][COLUMNS.specialty_code]
    )
    filters = replace(
        filters,
        trust_label=trust_label,
        trust_code=trust_code,
        specialty_label=specialty_label,
        specialty_code=specialty_code,
        region=region,
    )

    productivity_scenarios_raw = list((capacity_config.get("productivity_scenarios") or {"central": {}}).keys())
    preferred_order = ["low", "central", "high"]
    productivity_scenarios = [scenario for scenario in preferred_order if scenario in productivity_scenarios_raw]
    productivity_scenarios.extend(sorted([scenario for scenario in productivity_scenarios_raw if scenario not in productivity_scenarios]))
    control_cols = st.columns(2)
    scenario_label = control_cols[0].selectbox(
        "Forecast estimate used",
        list(SCENARIO_OPTIONS.keys()),
        index=list(SCENARIO_OPTIONS.keys()).index(filters.scenario_label) if filters.scenario_label in SCENARIO_OPTIONS else 1,
        key="capacity_forecast_estimate",
    )
    productivity_scenario = control_cols[1].selectbox(
        "Productivity assumption",
        productivity_scenarios,
        index=productivity_scenarios.index(filters.productivity_scenario) if filters.productivity_scenario in productivity_scenarios else (
            productivity_scenarios.index("central") if "central" in productivity_scenarios else 0
        ),
        key="capacity_productivity",
    )
    available_sessions = int(
        st.slider(
            "Additional treatment sessions to allocate",
            min_value=1_000,
            max_value=50_000,
            value=25_000,
            step=1_000,
            key="capacity_session_slider_v4",
        )
    )
    st.markdown(
        f"""
        <div class="forecast-key">
            <div><strong>{html.escape(scenario_label.upper())}</strong><br>{html.escape(FORECAST_EXPLANATIONS.get(scenario_label, ""))}</div>
            <div><strong>{html.escape(str(productivity_scenario).upper())} PRODUCTIVITY</strong><br>{html.escape(PRODUCTIVITY_EXPLANATIONS.get(str(productivity_scenario), "Scenario productivity values are read from the project configuration."))}</div>
            <div><strong>{available_sessions:,} ADDITIONAL TREATMENT SESSIONS</strong><br>This is the total number of extra treatment sessions available across the selected hospitals over 12 months. It is not a financial budget.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    filters = replace(
        filters,
        scenario_label=scenario_label,
        scenario_column=SCENARIO_OPTIONS[scenario_label],
        productivity_scenario=productivity_scenario,
        available_sessions=int(available_sessions),
    )
    forecast = apply_common_filters(optimisation_forecast, filters)
    if forecast.empty:
        st.info("No decision-to-admit forecast rows match the current filters.")
        return
    try:
        with st.spinner("Calculating this capacity scenario..."):
            allocation, metadata = solve_capacity_scenario_cached(
                forecast,
                available_sessions=filters.available_sessions,
                scenario_column=filters.scenario_column,
                capacity_config=capacity_config,
                productivity_scenario=filters.productivity_scenario,
            )
    except Exception as exc:
        st.error(f"The scenario could not run: {exc}")
        return

    baseline_backlog = float(metadata["baseline_predicted_backlog"])
    simulated_reduction = float(metadata["simulated_completed_pathways"])
    percent_reduction = float(metadata["national_percent_reduction"])
    cols = st.columns(5)
    cols[0].metric("Forecast backlog before scenario", compact_number(baseline_backlog))
    cols[1].metric("Simulated reduction", compact_number(simulated_reduction), f"{percent_reduction:.2f}%")
    remaining_backlog = float(metadata["remaining_backlog"])
    remaining_label = f"{remaining_backlog / 1_000_000:.3f}m" if remaining_backlog >= 1_000_000 else compact_number(remaining_backlog)
    cols[2].metric("Remaining backlog", remaining_label)
    cols[3].metric("Sessions allocated", f"{metadata['sessions_used']:,.0f}")
    cols[4].metric("Unused sessions", f"{metadata['unused_sessions']:,.0f}")

    st.subheader("ASSUMPTIONS")
    section_copy(
        "The forecast choice, specialty productivity values, available sessions, and per-row capacity limits are hypothetical scenario inputs. "
        "A mixed-integer allocation model is used throughout; these results are not observed NHS plans."
    )
    assumption_label = str(capacity_config.get("assumption_label", "")).strip()
    if assumption_label:
        st.markdown(f"<div class='quiet'>{assumption_label}</div>", unsafe_allow_html=True)

    if filters.trust_code is not None:
        chart_data = (
            allocation.groupby([COLUMNS.specialty_code, COLUMNS.specialty_name], as_index=False, observed=True)
            .agg(
                baseline_predicted_backlog=("baseline_predicted_backlog", "sum"),
                sessions_allocated=("sessions_allocated", "sum"),
                simulated_completed_pathways=("simulated_completed_pathways", "sum"),
                remaining_backlog=("remaining_backlog", "sum"),
            )
            .sort_values("simulated_completed_pathways", ascending=False)
        )
        chart_data["display_name"] = chart_data[COLUMNS.specialty_name].map(display_specialty_name)
        chart_title = "LARGEST PROPORTIONAL BACKLOG REDUCTIONS BY SPECIALTY"
    else:
        chart_data = (
            allocation.groupby([COLUMNS.trust_code, COLUMNS.trust_name], as_index=False, observed=True)
            .agg(
                baseline_predicted_backlog=("baseline_predicted_backlog", "sum"),
                sessions_allocated=("sessions_allocated", "sum"),
                simulated_completed_pathways=("simulated_completed_pathways", "sum"),
                remaining_backlog=("remaining_backlog", "sum"),
            )
            .sort_values("simulated_completed_pathways", ascending=False)
        )
        chart_data["display_name"] = chart_data[COLUMNS.trust_name].map(display_hospital_name)
        chart_title = "LARGEST PROPORTIONAL BACKLOG REDUCTIONS BY HOSPITAL"
    chart_data["percent_reduction"] = np.where(
        chart_data["baseline_predicted_backlog"] > 0,
        100.0 * chart_data["simulated_completed_pathways"] / chart_data["baseline_predicted_backlog"],
        0.0,
    )
    top = (
        chart_data[chart_data["simulated_completed_pathways"] > 0]
        .sort_values("percent_reduction", ascending=False)
        .head(20)
        .sort_values("percent_reduction")
    )
    fig = go.Figure(
        go.Bar(
            x=top["percent_reduction"],
            y=top["display_name"],
            orientation="h",
            marker_color=chart_primary_color(),
            customdata=top[["simulated_completed_pathways", "sessions_allocated", "remaining_backlog"]].to_numpy(),
            text=[f"{value:.1f}%" for value in top["percent_reduction"]],
            textposition="outside",
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Backlog reduction: %{x:.1f}%<br>"
                "Simulated waiting-list reduction: %{customdata[0]:,.0f}<br>"
                "Sessions allocated: %{customdata[1]:,.0f}<br>"
                "Remaining backlog: %{customdata[2]:,.0f}<extra></extra>"
            ),
        )
    )
    x_max = float(top["percent_reduction"].max()) if not top.empty else 1.0
    fig.update_layout(
        title=chart_title,
        xaxis_title="Backlog reduction (%)",
        yaxis_title="",
        template=chart_template(),
        height=620,
        margin=dict(l=20, r=85, t=55, b=45),
    )
    fig.update_xaxes(range=[0, x_max * 1.18 + 1])
    st.plotly_chart(apply_chart_style(fig), use_container_width=True, config=PLOTLY_CONFIG)

def render_pandemic_pressure_test(forecast: pd.DataFrame, historical: pd.DataFrame) -> None:
    st.title("PANDEMIC SIMULATION")
    section_copy(
        "A new pandemic-type disruption is applied from the start of the current 12-month forecast. "
        "Recent observed activity is used as the baseline, then the sliders test how waiting lists might move under disruption and recovery assumptions."
    )

    st.subheader("SELECT THE NHS AREA TO TEST")
    regions = sorted([region for region in forecast.get("region", pd.Series(dtype=str)).astype(str).unique() if region])
    selector_cols = st.columns(3)
    selected_region = selector_cols[0].selectbox(
        "Region",
        ["All regions"] + regions,
        index=0,
        key="pandemic_region",
    ) if regions else "All regions"
    scenario_forecast = forecast.copy()
    scenario_history = historical.copy()
    if selected_region != "All regions" and "region" in scenario_forecast.columns:
        scenario_forecast = scenario_forecast[scenario_forecast["region"].astype(str).eq(selected_region)]
        scenario_history = scenario_history[scenario_history["region"].astype(str).eq(selected_region)]

    hospital_options = (
        scenario_forecast[[COLUMNS.trust_code, COLUMNS.trust_name]]
        .drop_duplicates()
        .sort_values(COLUMNS.trust_name)
        .reset_index(drop=True)
    )
    hospital_labels = ["All hospitals"] + [
        f"{display_hospital_name(row.trust_name)} ({row.trust_code})"
        for row in hospital_options.itertuples(index=False)
    ]
    hospital_label = selector_cols[1].selectbox("Hospital", hospital_labels, index=0, key="pandemic_hospital")
    hospital_code = None if hospital_label == "All hospitals" else str(
        hospital_options.iloc[hospital_labels.index(hospital_label) - 1][COLUMNS.trust_code]
    )
    if hospital_code is not None:
        scenario_forecast = scenario_forecast[scenario_forecast[COLUMNS.trust_code].astype(str).eq(hospital_code)]
        scenario_history = scenario_history[scenario_history[COLUMNS.trust_code].astype(str).eq(hospital_code)]

    specialty_options = (
        scenario_forecast[[COLUMNS.specialty_code, COLUMNS.specialty_name]]
        .drop_duplicates()
        .sort_values(COLUMNS.specialty_name)
        .reset_index(drop=True)
    )
    specialty_labels = ["All specialties"] + [
        f"{display_specialty_name(row.specialty_name)} ({row.specialty_code})"
        for row in specialty_options.itertuples(index=False)
    ]
    specialty_label = selector_cols[2].selectbox("Specialty", specialty_labels, index=0, key="pandemic_specialty")
    specialty_code = None if specialty_label == "All specialties" else str(
        specialty_options.iloc[specialty_labels.index(specialty_label) - 1][COLUMNS.specialty_code]
    )
    if specialty_code is not None:
        scenario_forecast = scenario_forecast[scenario_forecast[COLUMNS.specialty_code].astype(str).eq(specialty_code)]
        scenario_history = scenario_history[scenario_history[COLUMNS.specialty_code].astype(str).eq(specialty_code)]

    if scenario_forecast.empty or scenario_history.empty:
        st.info("No forecast and historical activity rows match this selection.")
        return

    st.subheader("SET THE DISRUPTION AND RECOVERY ASSUMPTIONS")
    section_copy(
        "Treatment activity and new RTT periods are based on the selected area's latest 12 observed months. "
        "Every percentage below is a scenario assumption and can be changed."
    )
    parameter_cols = st.columns(2)
    disruption_months = parameter_cols[0].slider(
        "Months of severe disruption",
        min_value=1,
        max_value=12,
        value=6,
        step=1,
        key="pandemic_disruption_months",
    )
    treatment_reduction = parameter_cols[1].slider(
        "Reduction in completed pathways during disruption",
        min_value=0,
        max_value=60,
        value=35,
        step=5,
        format="%d%%",
        key="pandemic_treatment_reduction",
    )
    new_rtt_reduction = parameter_cols[0].slider(
        "Reduction in new RTT periods during disruption",
        min_value=0,
        max_value=60,
        value=20,
        step=5,
        format="%d%%",
        key="pandemic_new_rtt_reduction",
    )
    recovery_uplift = parameter_cols[1].slider(
        "Increase in completed pathways during recovery",
        min_value=0,
        max_value=50,
        value=15,
        step=5,
        format="%d%%",
        key="pandemic_recovery_uplift",
    )
    referral_rebound = parameter_cols[0].slider(
        "Increase in new RTT periods during recovery",
        min_value=0,
        max_value=40,
        value=10,
        step=5,
        format="%d%%",
        key="pandemic_referral_rebound",
    )
    recovery_months = parameter_cols[1].slider(
        "Months of recovery action",
        min_value=1,
        max_value=12,
        value=6,
        step=1,
        key="pandemic_recovery_months",
    )

    monthly_forecast = (
        scenario_forecast.groupby(COLUMNS.forecast_month, as_index=False, observed=True)[[COLUMNS.p10, COLUMNS.p50, COLUMNS.p90]]
        .sum()
        .sort_values(COLUMNS.forecast_month)
        .reset_index(drop=True)
    )
    final_observed_month = pd.to_datetime(scenario_history["month"]).max()
    reference_start = final_observed_month - pd.DateOffset(months=11)
    activity_reference = scenario_history[pd.to_datetime(scenario_history["month"]).between(reference_start, final_observed_month)].copy()
    required_activity = ["completed_total", "new_rtt_periods"]
    if any(column not in activity_reference.columns for column in required_activity):
        st.error("Completed-pathway and new-RTT-period history is required for this simulation.")
        return
    monthly_activity = activity_reference.groupby("month", as_index=False, observed=True)[required_activity].sum(min_count=1)
    average_completed = float(pd.to_numeric(monthly_activity["completed_total"], errors="coerce").mean())
    average_new_rtt = float(pd.to_numeric(monthly_activity["new_rtt_periods"], errors="coerce").mean())
    if not np.isfinite(average_completed) or not np.isfinite(average_new_rtt):
        st.error("The selected area does not have enough recent activity data for this simulation.")
        return

    additional_backlog: list[float] = []
    phases: list[str] = []
    cumulative_pressure = 0.0
    for month_index in range(len(monthly_forecast)):
        if month_index < disruption_months:
            monthly_change = (
                average_completed * treatment_reduction / 100.0
                - average_new_rtt * new_rtt_reduction / 100.0
            )
            phase = "Disruption"
        elif month_index < disruption_months + recovery_months:
            monthly_change = (
                average_new_rtt * referral_rebound / 100.0
                - average_completed * recovery_uplift / 100.0
            )
            phase = "Recovery"
        else:
            monthly_change = 0.0
            phase = "Post-recovery"
        cumulative_pressure = max(0.0, cumulative_pressure + monthly_change)
        additional_backlog.append(cumulative_pressure)
        phases.append(phase)

    monthly_forecast["phase"] = phases
    monthly_forecast["additional_backlog"] = additional_backlog
    monthly_forecast["pandemic_scenario_p10"] = monthly_forecast[COLUMNS.p10] + monthly_forecast["additional_backlog"]
    monthly_forecast["pandemic_scenario_p50"] = monthly_forecast[COLUMNS.p50] + monthly_forecast["additional_backlog"]
    monthly_forecast["pandemic_scenario_p90"] = monthly_forecast[COLUMNS.p90] + monthly_forecast["additional_backlog"]

    baseline_end = float(monthly_forecast[COLUMNS.p50].iloc[-1])
    scenario_end = float(monthly_forecast["pandemic_scenario_p50"].iloc[-1])
    additional_end = float(monthly_forecast["additional_backlog"].iloc[-1])
    peak_additional = float(monthly_forecast["additional_backlog"].max())
    metrics = st.columns(4)
    metrics[0].metric("Baseline forecast at 12 months", compact_number(baseline_end))
    metrics[1].metric("Pandemic-type scenario at 12 months", compact_number(scenario_end))
    metrics[2].metric("Additional backlog at 12 months", compact_number(additional_end))
    metrics[3].metric("Peak additional backlog", compact_number(peak_additional))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=monthly_forecast[COLUMNS.forecast_month],
            y=monthly_forecast[COLUMNS.p90],
            mode="lines",
            line=dict(color="rgba(120, 189, 248, 0.25)", width=1),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=monthly_forecast[COLUMNS.forecast_month],
            y=monthly_forecast[COLUMNS.p10],
            mode="lines",
            line=dict(color="rgba(120, 189, 248, 0.25)", width=1),
            fill="tonexty",
            fillcolor="rgba(120, 189, 248, 0.18)",
            name="Baseline forecast range",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=monthly_forecast[COLUMNS.forecast_month],
            y=monthly_forecast[COLUMNS.p50],
            mode="lines+markers",
            name="Baseline median forecast",
            line=dict(color=chart_primary_color(), width=2.5),
            hovertemplate="%{x|%B %Y}<br>Baseline forecast: %{y:,.0f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=monthly_forecast[COLUMNS.forecast_month],
            y=monthly_forecast["pandemic_scenario_p50"],
            mode="lines+markers",
            name="Pandemic-type scenario",
            line=dict(color=chart_warning_color(), width=3),
            customdata=monthly_forecast[["additional_backlog", "phase"]].to_numpy(),
            hovertemplate=(
                "%{x|%B %Y}<br>Scenario waiting list: %{y:,.0f}<br>"
                "Additional backlog: %{customdata[0]:,.0f}<br>Phase: %{customdata[1]}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title="BASELINE FORECAST COMPARED WITH A NEW PANDEMIC-TYPE DISRUPTION",
        xaxis_title="Month",
        yaxis_title="Incomplete RTT waiting list",
        template=chart_template(),
        height=520,
        margin=dict(l=20, r=20, t=65, b=45),
        hovermode="x unified",
    )
    st.plotly_chart(apply_chart_style(fig), use_container_width=True, config=PLOTLY_CONFIG)

    if additional_end <= 0.5:
        st.success("Under these assumptions, the additional backlog is cleared within the 12-month simulation.")
    else:
        st.warning(
            f"Under these assumptions, {compact_number(additional_end)} additional pathways remain at the end of the 12-month simulation. "
            "A longer recovery period or a stronger recovery uplift would be needed to close the simulated gap."
        )

    st.subheader("MONTH-BY-MONTH SCENARIO")
    section_copy("The table separates the saved baseline forecast from the additional backlog created by the selected disruption assumptions.")
    scenario_table = monthly_forecast.rename(
        columns={
            COLUMNS.p50: "baseline_median_forecast",
            "pandemic_scenario_p50": "pandemic_type_scenario",
        }
    )
    show_table(
        scenario_table,
        [COLUMNS.forecast_month, "phase", "baseline_median_forecast", "additional_backlog", "pandemic_type_scenario"],
        max_height=460,
    )


def render_data_quality(historical: pd.DataFrame) -> None:
    st.title("Data check")
    report = load_csv(str(PATHS.data_quality_report))
    missingness = load_csv(str(PATHS.missingness_by_series))
    identifier_changes = load_csv(str(PATHS.trust_identifier_changes))
    if not historical.empty:
        date_min = pd.to_datetime(historical["month"]).min().strftime("%Y-%m")
        date_max = pd.to_datetime(historical["month"]).max().strftime("%Y-%m")
        cols = st.columns(5)
        cols[0].metric("Hospitals", f"{historical[COLUMNS.trust_code].nunique():,}")
        cols[1].metric("Specialties", f"{historical[COLUMNS.specialty_code].nunique():,}")
        cols[2].metric("Months", f"{historical['month'].nunique():,}")
        cols[3].metric("Rows", f"{len(historical):,}")
        cols[4].metric("Period", f"{date_min} to {date_max}")

    tabs = st.tabs(["Summary", "Missing months", "Name changes"])
    with tabs[0]:
        if report.empty:
            st.info("The audit summary has not been generated yet.")
        else:
            summary = report[report["severity"].astype(str).str.lower().isin(["info", "warning"])].copy()
            summary = summary[summary["audit_section"].astype(str).isin(["configuration", "source_files", "row_counts", "duplicates", "missingness", "part2a_coverage"])]
            summary_columns = ["audit_section", "severity", "metric", "value", "details"]
            show_table(summary[summary_columns].head(40), summary_columns)
    with tabs[1]:
        if missingness.empty:
            st.info("The missing-month report has not been generated yet.")
        else:
            top_missing = missingness.sort_values(["missing_months", "max_consecutive_missing_months"], ascending=False)
            show_table(
                top_missing,
                [
                    COLUMNS.trust_name,
                    COLUMNS.specialty_name,
                    "first_month",
                    "last_month",
                    "observed_months",
                    "missing_months",
                    "missing_pct",
                    "max_consecutive_missing_months",
                ],
                max_rows=60,
            )
    with tabs[2]:
        if identifier_changes.empty:
            st.info("No hospital or specialty name changes were logged.")
        else:
            display = identifier_changes.copy()
            for column in [COLUMNS.trust_name, COLUMNS.specialty_name, "details"]:
                if column in display.columns:
                    display[column] = display[column].astype(str).str.replace(";", "; ", regex=False)
            show_table(
                display,
                [
                    "change_type",
                    COLUMNS.trust_code,
                    COLUMNS.trust_name,
                    COLUMNS.specialty_code,
                    COLUMNS.specialty_name,
                    "first_month",
                    "last_month",
                    "distinct_values",
                    "details",
                ],
                max_rows=80,
            )


def main() -> None:
    setup = status_frame()
    missing_required = missing_required_files(setup)
    if not missing_required.empty:
        render_setup_status(setup)
        st.stop()

    try:
        forecast = load_future_forecasts(str(PATHS.future_forecasts))
        historical = load_historical_observations(str(PATHS.clean_parquet))
        coordinates_path = first_available_path(PATHS.dashboard_data_dir / "nhs_trust_coordinates.csv")
        coordinates = load_trust_coordinates(
            str(PATHS.future_forecasts),
            str(coordinates_path) if coordinates_path is not None else str(PATHS.dashboard_data_dir / "nhs_trust_coordinates.csv"),
        )
        forecast = add_region_from_coordinates(forecast, coordinates)
        historical = add_region_from_coordinates(historical, coordinates) if not historical.empty else historical
        optimisation_forecast = add_region_from_coordinates(load_optimisation_forecasts(str(PATHS.future_optimisation_forecasts)), coordinates)
        model_config = read_json_if_exists(PATHS.model_config)
        capacity_config = load_capacity_config(str(PATHS.capacity_scenario_config))
    except Exception as exc:
        st.title("File check")
        st.error(f"The dashboard could not load its saved outputs.")
        st.info(str(exc))
        display = setup[["artifact", "required", "status", "modified"]].copy()
        display["required"] = display["required"].map({True: "Yes", False: "No"})
        show_table(display)
        st.stop()

    page, filters = sidebar_filters(forecast, capacity_config)
    metadata = forecast_metadata(forecast, historical, model_config)

    if page == "File check":
        render_setup_status(setup)
    elif page == "Overview":
        render_national_overview(forecast, historical, coordinates, filters, metadata)
    elif page == "Hospital profile":
        render_deep_dive(forecast, historical, filters)
    elif page == "Model check":
        render_model_performance(filters)
    elif page == "Forecast drivers":
        render_explainability(forecast, filters)
    elif page == "Capacity test":
        render_capacity_optimisation(optimisation_forecast, filters, capacity_config)
    elif page == "Pandemic test":
        render_pandemic_pressure_test(forecast, historical)
    elif page == "Data check":
        render_data_quality(historical)


if __name__ == "__main__":
    main()
