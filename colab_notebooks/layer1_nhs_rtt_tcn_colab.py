# %% Cell 1
import importlib.util
import subprocess
import sys

PIP_PACKAGES = [
    "pandas>=2.0.0",
    "numpy>=1.23.0",
    "pyarrow>=10.0.0",
    "beautifulsoup4>=4.12.0",
    "lxml>=4.9.0",
    "tqdm>=4.66.0",
    "matplotlib>=3.7.0",
    "scikit-learn>=1.2.0",
    "torch>=2.1.0",
]

IMPORT_CHECKS = {
    "pandas": "pandas",
    "numpy": "numpy",
    "pyarrow": "pyarrow",
    "beautifulsoup4": "bs4",
    "lxml": "lxml",
    "tqdm": "tqdm",
    "matplotlib": "matplotlib",
    "scikit-learn": "sklearn",
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
import calendar
import hashlib
import json
import math
import os
import random
import re
import sys
import time
import urllib.parse
import urllib.request
import warnings
import zipfile
from dataclasses import asdict, dataclass, replace
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from bs4 import BeautifulSoup
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

from nhs_rtt_pipeline.config import (
    BACKTEST_PREDICTION_COLUMNS,
    COLUMNS,
    FUTURE_FORECAST_COLUMNS,
    OPTIMISATION_FORECAST_COLUMNS,
    SURGICAL_SPECIALTY_INCLUSION,
    SURGICAL_SPECIALTY_INCLUSION_CRITERIA,
    ensure_directories,
    get_paths,
    validate_backtest_predictions_frame,
    validate_future_forecast_frame,
    validate_optimisation_forecast_frame,
)
from nhs_rtt_pipeline.data_quality import (
    ACTIVITY_NO_FILL_COLUMNS,
    DATA_START_MONTH,
    STOCK_FORWARD_FILL_COLUMNS,
    DataQualityConfig,
    aggregate_monthly_records,
    assert_clean_data_quality_contract,
    build_data_quality_summary_text,
    data_quality_report,
    duplicate_group_audit,
    harmonise_trust_and_specialty_identifiers,
    missingness_by_series_report,
    source_file_audit,
    trust_identifier_changes_report,
)
from nhs_rtt_pipeline.forecasting_baselines import (
    BaselineComparisonConfig,
    run_baseline_comparison,
    save_baseline_comparison_outputs,
    save_model_comparison_plots,
)
from nhs_rtt_pipeline.modeling import QuantileLoss, TCNQuantileRegressor, build_tcn_model_config
from nhs_rtt_pipeline.preprocessing import (
    FEATURE_GROUPS,
    FLOW_LAGGED_FEATURES,
    FLOW_MISSINGNESS_FEATURES,
    LOG1P_NON_NEGATIVE_FEATURES,
    NON_NEGATIVE_OPERATIONAL_FEATURES,
    SIGNED_OPERATIONAL_FEATURES,
    assert_net_inflow_integrity,
    data_dictionary_frame,
    feature_group_for_column,
    flow_reconciliation_quality_report,
    clean_rtt_operational_features,
    net_inflow_quality_summary,
)
from nhs_rtt_pipeline.reproducibility import set_global_seed
from nhs_rtt_pipeline.settings import load_pipeline_settings


@dataclass(frozen=True)
class Layer1Config:
    base_dir: str = "/content"
    start_financial_year: int = 2015
    end_financial_year: Optional[int] = None
    data_start_month: str = "2015-10-01"
    min_series_length: int = 36
    encoder_length: int = 24
    prediction_length: int = 12
    validation_months: int = 12
    test_months: int = 12
    batch_size: int = 512
    max_epochs: int = 35
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    hidden_channels: int = 96
    tcn_levels: int = 5
    kernel_size: int = 3
    dropout: float = 0.15
    embedding_dim: int = 16
    early_stopping_patience: int = 7
    gradient_clip_norm: float = 1.0
    random_seed: int = 42
    num_workers: int = 2
    plot_all_trusts: bool = True
    max_trust_plots: Optional[int] = None
    enable_hist_gradient_boosting_baseline: bool = True
    hist_gradient_boosting_max_train_rows: int = 350000
    hist_gradient_boosting_max_iter: int = 180


try:
    PROJECT_SETTINGS = load_pipeline_settings(os.environ.get("NHS_RTT_PIPELINE_CONFIG"))
    CONFIG = Layer1Config(**{**asdict(Layer1Config()), **PROJECT_SETTINGS.layer1_overrides()})
except FileNotFoundError:
    PROJECT_SETTINGS = None
    CONFIG = Layer1Config()

if os.name == "nt" and CONFIG.num_workers != 0:
    CONFIG = replace(CONFIG, num_workers=0)
    print("Windows execution detected; using num_workers=0 for safe PyTorch DataLoader startup.")

PATHS = get_paths()
ensure_directories(PATHS)

BASE_DIR = PATHS.project_root
RAW_DIR = PATHS.raw_dir
ZIP_DIR = PATHS.raw_zip_dir
CLEAN_DIR = PATHS.processed_dir
MODEL_DIR = PATHS.models_dir
OUTPUT_DIR = PATHS.outputs_dir
PLOT_DIR = PATHS.forecast_plot_dir

NHS_RTT_BASE_URL = "https://www.england.nhs.uk/statistics/statistical-work-areas/rtt-waiting-times"
MANIFEST_PATH = RAW_DIR / "rtt_full_csv_manifest.csv"
CLEAN_PARQUET_PATH = PATHS.clean_parquet
FEATURE_METADATA_PATH = PATHS.feature_metadata
DATA_DICTIONARY_PATH = PATHS.data_dictionary
MODEL_CONFIG_PATH = PATHS.model_config
DTA_MODEL_CONFIG_PATH = PATHS.dta_model_config
MODEL_STATE_DICT_PATH = PATHS.tcn_state_dict
DTA_MODEL_STATE_DICT_PATH = PATHS.dta_tcn_state_dict
MODEL_PATH = MODEL_STATE_DICT_PATH
TRAINING_HISTORY_PATH = PATHS.training_history
DTA_TRAINING_HISTORY_PATH = PATHS.dta_training_history
DTA_BACKTEST_PREDICTIONS_PATH = PATHS.dta_backtest_predictions
DTA_FORECAST_METRICS_PATH = PATHS.dta_forecast_metrics
DTA_FORECAST_METRICS_BY_HORIZON_PATH = PATHS.dta_forecast_metrics_by_horizon
DTA_MODEL_COMPARISON_PREDICTIONS_PATH = PATHS.dta_model_comparison_predictions
DTA_MODEL_COMPARISON_PATH = PATHS.dta_model_comparison
DTA_MODEL_COMPARISON_BY_HORIZON_PATH = PATHS.dta_model_comparison_by_horizon
DTA_MODEL_COMPARISON_BY_SPECIALTY_PATH = PATHS.dta_model_comparison_by_specialty
DTA_MODEL_COMPARISON_BY_TRUST_SIZE_PATH = PATHS.dta_model_comparison_by_trust_size
DTA_MODEL_COMPARISON_BY_COVID_PERIOD_PATH = PATHS.dta_model_comparison_by_covid_period
DTA_MODEL_COMPARISON_PAIRED_ERRORS_PATH = PATHS.dta_model_comparison_paired_errors
DTA_MODEL_COMPARISON_AUDIT_LOG_PATH = PATHS.dta_model_comparison_audit_log
DTA_MODEL_COMPARISON_SUMMARY_PATH = PATHS.dta_model_comparison_summary
DTA_RELIABILITY_SUMMARY_PATH = PATHS.dta_reliability_summary
BACKTEST_PREDICTIONS_PATH = PATHS.backtest_predictions
FUTURE_FORECASTS_PATH = PATHS.future_forecasts
FUTURE_OPTIMISATION_FORECASTS_PATH = PATHS.future_optimisation_forecasts
FORECAST_METRICS_PATH = PATHS.forecast_metrics
FORECAST_METRICS_BY_HORIZON_PATH = PATHS.forecast_metrics_by_horizon
FORECAST_PLOT_INDEX_PATH = PATHS.forecast_plot_index
MODEL_COMPARISON_PREDICTIONS_PATH = PATHS.model_comparison_predictions
MODEL_COMPARISON_PATH = PATHS.model_comparison
MODEL_COMPARISON_BY_HORIZON_PATH = PATHS.model_comparison_by_horizon
MODEL_COMPARISON_BY_SPECIALTY_PATH = PATHS.model_comparison_by_specialty
MODEL_COMPARISON_BY_TRUST_SIZE_PATH = PATHS.model_comparison_by_trust_size
MODEL_COMPARISON_BY_COVID_PERIOD_PATH = PATHS.model_comparison_by_covid_period
MODEL_COMPARISON_PAIRED_ERRORS_PATH = PATHS.model_comparison_paired_errors
MODEL_COMPARISON_AUDIT_LOG_PATH = PATHS.model_comparison_audit_log
MODEL_COMPARISON_SUMMARY_PATH = PATHS.model_comparison_summary
MODEL_COMPARISON_OVERALL_PNG_PATH = PATHS.model_comparison_overall_png
MODEL_COMPARISON_BY_HORIZON_PNG_PATH = PATHS.model_comparison_by_horizon_png
MODEL_COMPARISON_TCN_VS_SEASONAL_PNG_PATH = PATHS.model_comparison_tcn_vs_seasonal_png
NET_INFLOW_QUALITY_PATH = PATHS.net_inflow_quality
FLOW_RECONCILIATION_QUALITY_PATH = PATHS.flow_reconciliation_quality
DATA_QUALITY_REPORT_PATH = PATHS.data_quality_report
DATA_QUALITY_SUMMARY_PATH = PATHS.data_quality_summary
MISSINGNESS_BY_SERIES_PATH = PATHS.missingness_by_series
TRUST_IDENTIFIER_CHANGES_PATH = PATHS.trust_identifier_changes
PART2A_COVERAGE_REPORT_PATH = PATHS.part2a_coverage_report
SURGICAL_SPECIALTIES_PATH = PATHS.surgical_specialties

QUANTILES = (0.1, 0.5, 0.9)
PART_TO_FEATURE = {
    "PART_1A": "completed_admitted",
    "PART_1B": "completed_non_admitted",
    "PART_2": "waiting_list",
    "PART_2A": "waiting_list_with_dta",
    "PART_3": "new_rtt_periods",
}
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

def set_random_seed(seed: int) -> None:
    deterministic = bool(PROJECT_SETTINGS.deterministic_torch) if PROJECT_SETTINGS is not None else False
    set_global_seed(seed, deterministic_torch=deterministic)
    if torch.cuda.is_available() and not deterministic:
        torch.backends.cudnn.benchmark = True


set_random_seed(CONFIG.random_seed)
LAYER1_RUN_STAGE = os.environ.get("NHS_RTT_LAYER1_STAGE", "all").strip().lower() or "all"

print(json.dumps(asdict(CONFIG), indent=2))
print(f"Project root: {BASE_DIR}")
print(f"Using custom PyTorch TCN backend: {TCNQuantileRegressor.__module__}.{TCNQuantileRegressor.__name__}")
print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
print(f"Layer 1 execution stage: {LAYER1_RUN_STAGE}")

# %% Cell 3
import calendar
import os
import re
import time
import urllib.parse
import urllib.request
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from tqdm.auto import tqdm


MONTH_LOOKUP = {
    "JAN": 1,
    "FEB": 2,
    "MAR": 3,
    "APR": 4,
    "MAY": 5,
    "JUN": 6,
    "JUL": 7,
    "AUG": 8,
    "SEP": 9,
    "SEPT": 9,
    "OCT": 10,
    "NOV": 11,
    "DEC": 12,
}


def current_uk_financial_year_start() -> int:
    today = pd.Timestamp.today(tz="Europe/London")
    return int(today.year if today.month >= 4 else today.year - 1)


def financial_year_slug(start_year: int) -> str:
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def http_get_bytes(url: str, retries: int = 4, timeout: int = 90) -> bytes:
    last_error = None
    headers = {"User-Agent": "Mozilla/5.0 nhs-rtt-msc-data-science/1.0"}
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except Exception as exc:
            last_error = exc
            wait_seconds = min(2 ** attempt, 20)
            time.sleep(wait_seconds)
    raise RuntimeError(f"Failed to download {url}: {last_error}") from last_error


def parse_period_from_text(text: str) -> Optional[pd.Timestamp]:
    value = str(text)
    month_pattern = r"(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t|tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    match = re.search(month_pattern + r"[^A-Za-z0-9]{0,5}(\d{2,4})", value, flags=re.IGNORECASE)
    if match is None:
        return None
    month_token = match.group(1).upper()[:4]
    if month_token.startswith("SEPT"):
        month_token = "SEPT"
    else:
        month_token = month_token[:3]
    month = MONTH_LOOKUP.get(month_token)
    year_token = match.group(2)
    year = int(year_token)
    if year < 100:
        year += 2000
    if month is None:
        return None
    return pd.Timestamp(year=year, month=month, day=1)


def scrape_rtt_full_csv_manifest(
    start_financial_year: int,
    end_financial_year: Optional[int],
    manifest_path: Path,
) -> pd.DataFrame:
    final_financial_year = current_uk_financial_year_start() if end_financial_year is None else end_financial_year
    records: List[Dict[str, object]] = []

    for start_year in tqdm(range(start_financial_year, final_financial_year + 1), desc="Scraping NHS RTT pages"):
        slug = financial_year_slug(start_year)
        page_url = f"{NHS_RTT_BASE_URL}/rtt-data-{slug}/"
        try:
            html = http_get_bytes(page_url, retries=2, timeout=45).decode("utf-8", errors="replace")
        except Exception as exc:
            print(f"Skipping {page_url}: {exc}")
            continue

        soup = BeautifulSoup(html, "lxml")
        for anchor in soup.find_all("a"):
            link_text = " ".join(anchor.get_text(" ", strip=True).split())
            href = anchor.get("href")
            if href is None:
                continue
            if "full csv data file" not in link_text.lower():
                continue
            absolute_url = urllib.parse.urljoin(page_url, href)
            if ".zip" not in absolute_url.lower() and "zip" not in link_text.lower():
                continue
            period_date = parse_period_from_text(f"{link_text} {absolute_url}")
            records.append(
                {
                    "financial_year": slug,
                    "page_url": page_url,
                    "url": absolute_url,
                    "link_text": link_text,
                    "period_date": period_date,
                    "is_revised": "revised" in link_text.lower() or "revised" in absolute_url.lower(),
                }
            )

    if not records:
        raise RuntimeError("No NHS RTT full CSV ZIP links were found.")

    manifest = pd.DataFrame(records).drop_duplicates(subset=["url"]).reset_index(drop=True)
    manifest["period_date"] = pd.to_datetime(manifest["period_date"])
    with_period = manifest[manifest["period_date"].notna()].copy()
    without_period = manifest[manifest["period_date"].isna()].copy()

    if not with_period.empty:
        with_period = (
            with_period.sort_values(["period_date", "is_revised", "url"])
            .drop_duplicates(subset=["period_date"], keep="last")
            .sort_values("period_date")
        )
    manifest = pd.concat([with_period, without_period], ignore_index=True).sort_values(
        ["period_date", "financial_year", "url"], na_position="last"
    )
    manifest.to_csv(manifest_path, index=False)
    return manifest.reset_index(drop=True)


def safe_zip_filename(url: str, period_date: Optional[pd.Timestamp]) -> str:
    parsed_name = Path(urllib.parse.urlparse(url).path).name
    parsed_name = re.sub(r"[^A-Za-z0-9._-]+", "_", parsed_name)
    if period_date is not None and not pd.isna(period_date):
        return f"{pd.Timestamp(period_date).strftime('%Y-%m')}_{parsed_name}"
    return parsed_name


def download_rtt_zips(manifest: pd.DataFrame, zip_dir: Path) -> pd.DataFrame:
    manifest = manifest.copy()
    local_paths = []
    for row in tqdm(manifest.itertuples(index=False), total=len(manifest), desc="Downloading RTT ZIP files"):
        period_date = getattr(row, "period_date")
        url = getattr(row, "url")
        filename = safe_zip_filename(url, period_date)
        local_path = zip_dir / filename
        if not local_path.exists() or local_path.stat().st_size == 0:
            tmp_path = local_path.with_suffix(local_path.suffix + ".tmp")
            data = http_get_bytes(url, retries=4, timeout=120)
            tmp_path.write_bytes(data)
            tmp_path.replace(local_path)
        local_paths.append(str(local_path))
    manifest["local_path"] = local_paths
    manifest.to_csv(MANIFEST_PATH, index=False)
    return manifest


def local_rtt_zip_manifest(zip_dir: Path, manifest_path: Path) -> pd.DataFrame:
    zip_files = sorted(
        [
            path
            for path in zip_dir.glob("*.zip")
            if path.name.lower() not in {"nhs.zip", "rtt.zip", "data.zip"}
        ],
        key=lambda path: path.name.lower(),
    )
    records = []
    for path in zip_files:
        period_date = parse_period_from_text(path.name)
        records.append(
            {
                "financial_year": "",
                "page_url": "local",
                "url": path.resolve().as_uri(),
                "link_text": path.name,
                "period_date": period_date,
                "is_revised": "revised" in path.name.lower(),
                "local_path": str(path.resolve()),
            }
        )
    manifest = pd.DataFrame(records)
    if not manifest.empty:
        manifest["period_date"] = pd.to_datetime(manifest["period_date"], errors="coerce")
        manifest = manifest.sort_values(["period_date", "local_path"], na_position="last").reset_index(drop=True)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(manifest_path, index=False)
    return manifest


def find_csv_header_offset(raw: bytes) -> int:
    patterns = [
        rb'(?m)^(?:\xef\xbb\xbf)?"?Period"?,\s*"?Provider Parent Org Code"?',
        rb'(?m)^(?:\xef\xbb\xbf)?"?Year"?,\s*"?Period Name"?,\s*"?Provider Parent Org Code"?',
    ]
    for pattern in patterns:
        match = re.search(pattern, raw)
        if match is not None:
            return match.start()
    preview = raw[:500].decode("utf-8", errors="replace")
    raise ValueError(f"Could not locate an RTT CSV header. File starts with: {preview}")


def read_rtt_csv_bytes(raw: bytes) -> pd.DataFrame:
    header_offset = find_csv_header_offset(raw)
    payload = BytesIO(raw[header_offset:])
    try:
        frame = pd.read_csv(payload, dtype=str, low_memory=False, encoding="utf-8-sig")
    except UnicodeDecodeError:
        payload.seek(0)
        frame = pd.read_csv(payload, dtype=str, low_memory=False, encoding="cp1252")
    frame.columns = [str(column).strip() for column in frame.columns]
    frame = frame.dropna(how="all")
    return frame


def numeric_series(series: pd.Series) -> pd.Series:
    cleaned = series.fillna("").astype(str).str.replace(",", "", regex=False).str.strip()
    return pd.to_numeric(cleaned, errors="coerce")


def coalesce_numeric(frame: pd.DataFrame, columns: List[str]) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index, dtype="float64")
    for column in columns:
        if column in frame.columns:
            result = result.fillna(numeric_series(frame[column]))
    return result


def first_existing_column(frame: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for column in candidates:
        if column in frame.columns:
            return column
    return None


def standardise_part(value: object) -> str:
    text = str(value).strip().upper().replace("-", "_").replace(" ", "_")
    text = re.sub(r"_+", "_", text)
    return text


def parse_month_from_current_period(period: pd.Series) -> pd.Series:
    parsed = period.apply(parse_period_from_text)
    return pd.to_datetime(parsed)


def parse_month_from_legacy_period(year_series: pd.Series, period_name_series: pd.Series) -> pd.Series:
    start_year = year_series.astype(str).str.extract(r"(\d{4})", expand=False).astype("float")
    period_token = period_name_series.fillna("").astype(str).str.upper().str.extract(
        r"(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEPT|SEP|OCT|NOV|DEC)",
        expand=False,
    )
    month = period_token.map(MONTH_LOOKUP).astype("float")
    calendar_year = start_year + np.where(month <= 3, 1, 0)
    out = pd.to_datetime(
        {
            "year": calendar_year.astype("Int64"),
            "month": month.astype("Int64"),
            "day": pd.Series(1, index=year_series.index, dtype="Int64"),
        },
        errors="coerce",
    )
    return out


def extract_monthly_provider_specialty_from_zip(
    zip_path: Path,
    source_url: str,
    source_publication_month: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    extracted_frames: List[pd.DataFrame] = []
    with zipfile.ZipFile(zip_path) as archive:
        csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if not csv_names:
            raise ValueError(f"No CSV files found inside {zip_path}")
        for csv_name in csv_names:
            raw = archive.read(csv_name)
            frame = read_rtt_csv_bytes(raw)

            part_column = "RTT Part Type" if "RTT Part Type" in frame.columns else "RTT Part Name"
            specialty_code_column = first_existing_column(
                frame,
                ["Treatment Function Code", "Treatment Function Name"],
            )
            specialty_name_column = (
                "Treatment Function Name"
                if "Treatment Function Code" in frame.columns and "Treatment Function Name" in frame.columns
                else first_existing_column(frame, ["Treatment Function Description", "Treatment Function Name"])
            )
            required_columns = ["Provider Org Code", "Provider Org Name", part_column]
            missing_columns = [column for column in required_columns if column not in frame.columns]
            if specialty_code_column is None:
                missing_columns.append("Treatment Function Code or Treatment Function Name")
            if specialty_name_column is None:
                missing_columns.append("Treatment Function Name or Treatment Function Description")
            if missing_columns:
                raise ValueError(f"{zip_path}::{csv_name} is missing required columns: {missing_columns}")

            if "Period" in frame.columns:
                month = parse_month_from_current_period(frame["Period"])
            elif "Year" in frame.columns and "Period Name" in frame.columns:
                month = parse_month_from_legacy_period(frame["Year"], frame["Period Name"])
            else:
                raise ValueError(f"{zip_path}::{csv_name} has no supported period columns.")

            if "Total All" in frame.columns:
                value = numeric_series(frame["Total All"])
            else:
                total_value = coalesce_numeric(frame, ["Total"])
                unknown_clock_start_value = coalesce_numeric(frame, ["Patients with unknown clock start date"])
                value = pd.Series(np.nan, index=frame.index, dtype="float64")
                total_available = total_value.notna()
                unknown_available = unknown_clock_start_value.notna()
                value.loc[total_available & unknown_available] = (
                    total_value.loc[total_available & unknown_available]
                    + unknown_clock_start_value.loc[total_available & unknown_available]
                )
                value.loc[total_available & ~unknown_available] = total_value.loc[total_available & ~unknown_available]
                value.loc[~total_available & unknown_available] = unknown_clock_start_value.loc[
                    ~total_available & unknown_available
                ]

            specialty_code = frame[specialty_code_column].fillna("").astype(str).str.strip()
            specialty_name = frame[specialty_name_column].fillna("").astype(str).str.strip()
            is_total_specialty = specialty_code.str.upper().isin(["C_999", "999"]) | specialty_name.str.upper().eq("TOTAL")

            working = pd.DataFrame(
                {
                    "month": month,
                    "trust_code": frame["Provider Org Code"].fillna("").astype(str).str.strip(),
                    "trust_name": frame["Provider Org Name"].fillna("").astype(str).str.strip(),
                    "specialty_code": specialty_code,
                    "specialty_name": specialty_name,
                    "rtt_part": frame[part_column].map(standardise_part),
                    "value": value.astype(float),
                }
            )
            working["source_trust_code"] = working["trust_code"]
            working["source_trust_name"] = working["trust_name"]
            working["source_specialty_code"] = working["specialty_code"]
            working["source_specialty_name"] = working["specialty_name"]
            working = working[~is_total_specialty].copy()
            working = working[working["rtt_part"].isin(PART_TO_FEATURE.keys())]
            working = working[working["month"].notna()]
            working = working[(working["trust_code"] != "") & (working["specialty_code"] != "")]
            working["feature"] = working["rtt_part"].map(PART_TO_FEATURE)
            grouped = (
                working.groupby(
                    ["month", "trust_code", "trust_name", "specialty_code", "specialty_name", "feature"],
                    as_index=False,
                    observed=True,
                )["value"]
                .sum(min_count=1)
            )
            availability = (
                working.assign(source_available=working["value"].notna().astype(int))
                .groupby(
                    ["month", "trust_code", "trust_name", "specialty_code", "specialty_name", "feature"],
                    as_index=False,
                    observed=True,
                )["source_available"]
                .max()
            )
            value_pivoted = grouped.pivot_table(
                index=["month", "trust_code", "trust_name", "specialty_code", "specialty_name"],
                columns="feature",
                values="value",
                aggfunc="sum",
            ).reset_index()
            value_pivoted.columns.name = None
            availability_pivoted = availability.pivot_table(
                index=["month", "trust_code", "trust_name", "specialty_code", "specialty_name"],
                columns="feature",
                values="source_available",
                aggfunc="max",
                fill_value=0,
            ).reset_index()
            availability_pivoted.columns.name = None
            availability_pivoted = availability_pivoted.rename(
                columns={feature: f"{feature}_source_available" for feature in PART_TO_FEATURE.values()}
            )
            pivoted = availability_pivoted.merge(
                value_pivoted,
                on=["month", "trust_code", "trust_name", "specialty_code", "specialty_name"],
                how="left",
            )
            pivoted["source_trust_code"] = pivoted["trust_code"]
            pivoted["source_trust_name"] = pivoted["trust_name"]
            pivoted["source_specialty_code"] = pivoted["specialty_code"]
            pivoted["source_specialty_name"] = pivoted["specialty_name"]
            pivoted["source_zip"] = str(zip_path)
            pivoted["source_csv"] = csv_name
            pivoted["source_url"] = source_url
            pivoted["source_publication_month"] = (
                pd.Timestamp(source_publication_month).date().isoformat()
                if source_publication_month is not None and pd.notna(source_publication_month)
                else ""
            )
            pivoted["source_table_type"] = "monthly_provider_specialty_full_csv"
            pivoted["source_row_count"] = len(working)
            extracted_frames.append(pivoted)

    if not extracted_frames:
        return pd.DataFrame()
    return pd.concat(extracted_frames, ignore_index=True)


def complete_missing_months(frame: pd.DataFrame, min_series_length: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    static_columns = [
        "trust_name",
        "specialty_name",
        "source_trust_code",
        "source_trust_name",
        "source_specialty_code",
        "source_specialty_name",
        "trust_identifier_harmonisation_rule",
        "specialty_identifier_harmonisation_rule",
    ]
    count_columns = list(PART_TO_FEATURE.values())
    availability_columns = [f"{feature}_source_available" for feature in count_columns]
    provenance_columns = ["source_file_count", "source_row_count", "source_zips", "source_csvs", "source_urls", "source_publication_months"]
    completed_frames = []
    grouped = frame.groupby(["trust_code", "specialty_code"], sort=False, observed=True)

    for (trust_code, specialty_code), group in tqdm(grouped, total=grouped.ngroups, desc="Completing monthly series"):
        group = group.sort_values("month").copy()
        full_months = pd.date_range(group["month"].min(), group["month"].max(), freq="MS")
        group = group.set_index("month").reindex(full_months).rename_axis("month").reset_index()
        group["trust_code"] = trust_code
        group["specialty_code"] = specialty_code
        for column in static_columns:
            if column not in group.columns:
                group[column] = ""
            group[column] = group[column].ffill().bfill()
        for column in provenance_columns:
            if column not in group.columns:
                group[column] = 0 if column.endswith("_count") else ""
            if column.endswith("_count"):
                group[column] = pd.to_numeric(group[column], errors="coerce").fillna(0).astype(int)
            else:
                group[column] = group[column].fillna("")
        for column in count_columns:
            if column not in group.columns:
                group[column] = np.nan
            group[column] = pd.to_numeric(group[column], errors="coerce")
            imputation_flag = f"{column}_imputed"
            if column in STOCK_FORWARD_FILL_COLUMNS:
                before = group[column].copy()
                inserted_missing_month = group["observed_month"].isna()
                filled = group[column].where(~inserted_missing_month, group[column].ffill())
                group[imputation_flag] = (before.isna() & filled.notna() & inserted_missing_month).astype("int8")
                group[column] = filled
            elif column in ACTIVITY_NO_FILL_COLUMNS:
                group[imputation_flag] = 0
            else:
                group[imputation_flag] = 0
        for column in availability_columns:
            if column not in group.columns:
                group[column] = 0
            group[column] = pd.to_numeric(group[column], errors="coerce").fillna(0).astype(int)
        group["observed_month"] = group["observed_month"].fillna(0).astype(int)
        group["is_imputed_month"] = (1 - group["observed_month"]).astype(int)
        group["missing_month"] = group["is_imputed_month"].astype(int)
        completed_frames.append(group)

    completed = pd.concat(completed_frames, ignore_index=True)
    completed["series_id"] = completed["trust_code"] + "__" + completed["specialty_code"]
    min_month = completed["month"].min()
    completed["time_idx"] = (
        (completed["month"].dt.year - min_month.year) * 12 + (completed["month"].dt.month - min_month.month)
    ).astype(int)
    completed["calendar_month"] = completed["month"].dt.month.astype(int)
    completed["month_sin"] = np.sin(2 * np.pi * completed["calendar_month"] / 12.0)
    completed["month_cos"] = np.cos(2 * np.pi * completed["calendar_month"] / 12.0)

    missingness = missingness_by_series_report(completed, min_series_length=min_series_length)
    latest_observed_month = completed.loc[completed["observed_month"].eq(1), "month"].max()
    if pd.notna(latest_observed_month) and not missingness.empty:
        latest_by_series = (
            completed.loc[completed["observed_month"].eq(1)]
            .groupby("series_id", observed=True)["month"]
            .max()
            .rename("latest_observed_month")
        )
        missingness = missingness.merge(latest_by_series, on="series_id", how="left")
        missingness["discontinued_series"] = pd.to_datetime(missingness["latest_observed_month"]) < pd.Timestamp(latest_observed_month)
        missingness["latest_observed_month"] = pd.to_datetime(missingness["latest_observed_month"]).dt.date.astype(str)

    series_lengths = completed.groupby("series_id", observed=True)["month"].transform("size")
    completed = completed[series_lengths >= min_series_length].copy()
    completed = completed.sort_values(["trust_code", "specialty_code", "month"]).reset_index(drop=True)
    return completed, missingness


def build_clean_rtt_timeseries(config: Layer1Config) -> pd.DataFrame:
    dq_config = DataQualityConfig(start_month=config.data_start_month, min_series_length=config.min_series_length)
    explicit_start_month = pd.Timestamp(config.data_start_month)
    if explicit_start_month != DATA_START_MONTH:
        print(f"Using configured explicit start month {explicit_start_month.date()} instead of default {DATA_START_MONTH.date()}.")
    manifest = local_rtt_zip_manifest(ZIP_DIR, MANIFEST_PATH)
    if manifest.empty:
        print("No local RTT ZIP files found; scraping NHS England RTT pages.")
        manifest = scrape_rtt_full_csv_manifest(
            start_financial_year=config.start_financial_year,
            end_financial_year=config.end_financial_year,
            manifest_path=MANIFEST_PATH,
        )
        manifest = download_rtt_zips(manifest, ZIP_DIR)
    else:
        print(f"Using {len(manifest):,} local monthly RTT ZIP files from {ZIP_DIR}.")

    monthly_frames = []
    for row in tqdm(manifest.itertuples(index=False), total=len(manifest), desc="Parsing RTT ZIP files"):
        monthly = extract_monthly_provider_specialty_from_zip(
            Path(row.local_path),
            row.url,
            source_publication_month=getattr(row, "period_date", None),
        )
        if not monthly.empty:
            monthly_frames.append(monthly)

    if not monthly_frames:
        raise RuntimeError("No RTT monthly data could be parsed.")

    raw_monthly = pd.concat(monthly_frames, ignore_index=True)
    raw_monthly["month"] = pd.to_datetime(raw_monthly["month"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    raw_monthly = raw_monthly[raw_monthly["month"] >= explicit_start_month].copy()
    if raw_monthly.empty:
        raise RuntimeError(f"No RTT monthly data remained after explicit date filter month >= {explicit_start_month.date()}.")
    raw_monthly = harmonise_trust_and_specialty_identifiers(raw_monthly)
    for feature in PART_TO_FEATURE.values():
        if feature not in raw_monthly.columns:
            raw_monthly[feature] = np.nan
        availability_column = f"{feature}_source_available"
        if availability_column not in raw_monthly.columns:
            raw_monthly[availability_column] = raw_monthly[feature].notna().astype(int)
    raw_monthly["observed_month"] = 1

    value_columns = list(PART_TO_FEATURE.values())
    availability_columns = [f"{feature}_source_available" for feature in value_columns]
    duplicate_audit = duplicate_group_audit(raw_monthly, value_columns=value_columns)
    identifier_changes = trust_identifier_changes_report(raw_monthly)
    monthly = aggregate_monthly_records(raw_monthly, value_columns=value_columns, availability_columns=availability_columns)
    monthly["observed_month"] = (monthly["observed_month"] > 0).astype(int)
    clean, missingness_by_series = complete_missing_months(monthly, config.min_series_length)

    clean["month"] = pd.to_datetime(clean["month"])
    for column in [
        "is_imputed_month",
        "missing_month",
        "observed_month",
        "time_idx",
        "calendar_month",
        "waiting_list_imputed",
        "waiting_list_with_dta_imputed",
        "completed_admitted_imputed",
        "completed_non_admitted_imputed",
        "new_rtt_periods_imputed",
    ]:
        clean[column] = pd.to_numeric(clean[column], errors="coerce").fillna(0)
    clean = clean_rtt_operational_features(clean)
    clean["is_surgical_specialty"] = clean["specialty_code"].astype(str).isin(SURGICAL_SPECIALTY_INCLUSION)
    clean["specialty_inclusion_criteria"] = clean["specialty_code"].astype(str).map(SURGICAL_SPECIALTY_INCLUSION).fillna(
        "Excluded: treatment function is not in the configured decision-to-admit capacity-simulation specialty list."
    )
    assert_net_inflow_integrity(clean)
    assert_clean_data_quality_contract(clean, MODEL_INPUT_FEATURES, dq_config)
    net_inflow_summary = net_inflow_quality_summary(clean)
    net_inflow_summary.to_csv(NET_INFLOW_QUALITY_PATH, index=False)
    reconciliation_report = flow_reconciliation_quality_report(clean)
    with open(FLOW_RECONCILIATION_QUALITY_PATH, "w", encoding="utf-8") as handle:
        json.dump(
            reconciliation_report,
            handle,
            indent=2,
            default=lambda value: None
            if pd.isna(value)
            else float(value)
            if isinstance(value, np.floating)
            else int(value)
            if isinstance(value, np.integer)
            else str(value),
        )
    data_dictionary_frame(MODEL_INPUT_FEATURES).to_csv(DATA_DICTIONARY_PATH, index=False)
    missingness_by_series.to_csv(MISSINGNESS_BY_SERIES_PATH, index=False)
    identifier_changes.to_csv(TRUST_IDENTIFIER_CHANGES_PATH, index=False)
    source_audit = source_file_audit(raw_monthly)
    dq_report = data_quality_report(
        manifest=manifest,
        raw_monthly=raw_monthly,
        monthly=monthly,
        clean=clean,
        duplicate_audit=pd.concat([source_audit, duplicate_audit], ignore_index=True) if not source_audit.empty else duplicate_audit,
        missingness=missingness_by_series,
        identifier_changes=identifier_changes,
        config=dq_config,
    )
    dq_report.to_csv(DATA_QUALITY_REPORT_PATH, index=False)
    DATA_QUALITY_SUMMARY_PATH.write_text(
        build_data_quality_summary_text(
            clean=clean,
            missingness=missingness_by_series,
            report=dq_report,
            config=dq_config,
        ),
        encoding="utf-8",
    )
    surgical_specialties = pd.DataFrame(
        [
            {
                "specialty_code": code,
                "specialty_inclusion_criteria": reason,
                "configured_use": "decision_to_admit_capacity_simulation",
            }
            for code, reason in sorted(SURGICAL_SPECIALTY_INCLUSION.items())
        ]
    )
    surgical_specialties.to_csv(SURGICAL_SPECIALTIES_PATH, index=False)
    build_part2a_coverage_report(clean).to_csv(PART2A_COVERAGE_REPORT_PATH, index=False)
    clean.to_parquet(CLEAN_PARQUET_PATH, index=False)
    return clean


def build_part2a_coverage_report(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    if "incomplete_decision_to_admit_source_available" not in working.columns:
        working["incomplete_decision_to_admit_source_available"] = working["incomplete_decision_to_admit"].notna().astype(int)
    working["part2a_available"] = (
        pd.to_numeric(working["incomplete_decision_to_admit_source_available"], errors="coerce").fillna(0).astype(int).eq(1)
        & pd.to_numeric(working["incomplete_decision_to_admit"], errors="coerce").notna()
    )
    rows = (
        working.groupby(
            ["trust_code", "trust_name", "specialty_code", "specialty_name", "is_surgical_specialty", "specialty_inclusion_criteria"],
            as_index=False,
            observed=True,
        )
        .agg(
            total_months=("month", "size"),
            part2a_available_months=("part2a_available", "sum"),
            latest_part2a_value=("incomplete_decision_to_admit", "last"),
            latest_total_incomplete_value=("incomplete_total", "last"),
        )
        .reset_index(drop=True)
    )
    rows["part2a_available_months"] = rows["part2a_available_months"].astype(int)
    rows["part2a_coverage_pct"] = np.where(
        rows["total_months"] > 0,
        100.0 * rows["part2a_available_months"] / rows["total_months"],
        0.0,
    )
    rows["eligible_for_capacity_simulation"] = (
        rows["is_surgical_specialty"].astype(bool) & rows["part2a_available_months"].gt(0)
    )
    return rows.sort_values(
        ["eligible_for_capacity_simulation", "is_surgical_specialty", "part2a_coverage_pct", "trust_name", "specialty_name"],
        ascending=[False, False, False, True, True],
    ).reset_index(drop=True)

# %% Cell 4
import pandas as pd


FORCE_PREPARE = os.environ.get("NHS_RTT_FORCE_PREPARE", "").strip().lower() in {"1", "true", "yes"}
if CLEAN_PARQUET_PATH.exists() and not FORCE_PREPARE:
    print(f"Using existing clean monthly RTT panel: {CLEAN_PARQUET_PATH}")
    clean_rtt = pd.read_parquet(CLEAN_PARQUET_PATH)
    clean_rtt["month"] = pd.to_datetime(clean_rtt["month"], errors="coerce")
else:
    if CLEAN_PARQUET_PATH.exists() and FORCE_PREPARE:
        print(f"Force prepare requested; rebuilding clean RTT panel from raw ZIP files: {CLEAN_PARQUET_PATH}")
    clean_rtt = build_clean_rtt_timeseries(CONFIG)

print(f"Saved clean monthly RTT panel to: {CLEAN_PARQUET_PATH}")
print(f"Rows: {len(clean_rtt):,}")
print(f"Series: {clean_rtt['series_id'].nunique():,}")
print(f"Trusts/providers: {clean_rtt['trust_code'].nunique():,}")
print(f"Specialties: {clean_rtt['specialty_code'].nunique():,}")
print(f"Date range: {clean_rtt['month'].min().date()} to {clean_rtt['month'].max().date()}")
print(f"Saved net inflow data-quality summary to: {NET_INFLOW_QUALITY_PATH}")
print(f"Saved RTT flow reconciliation data-quality report to: {FLOW_RECONCILIATION_QUALITY_PATH}")
print(f"Saved data-quality report to: {DATA_QUALITY_REPORT_PATH}")
print(f"Saved data-quality summary to: {DATA_QUALITY_SUMMARY_PATH}")
print(f"Saved missingness by series to: {MISSINGNESS_BY_SERIES_PATH}")
print(f"Saved trust identifier changes to: {TRUST_IDENTIFIER_CHANGES_PATH}")
print(f"Saved Part 2A coverage report to: {PART2A_COVERAGE_REPORT_PATH}")
print(f"Saved surgical specialty inclusion mapping to: {SURGICAL_SPECIALTIES_PATH}")
print(f"Saved data dictionary to: {DATA_DICTIONARY_PATH}")
display(net_inflow_quality_summary(clean_rtt))
display(pd.DataFrame([flow_reconciliation_quality_report(clean_rtt)["summary"]]))
display(pd.read_csv(DATA_QUALITY_REPORT_PATH).head(25))
display(pd.read_csv(MISSINGNESS_BY_SERIES_PATH).sort_values("missing_months", ascending=False).head(25))
part2a_coverage = build_part2a_coverage_report(clean_rtt)
display(
    pd.DataFrame(
        [
            {
                "trust_specialty_series": int(len(part2a_coverage)),
                "series_with_part2a_available": int((part2a_coverage["part2a_available_months"] > 0).sum()),
                "eligible_surgical_series_for_optimisation": int(part2a_coverage["eligible_for_capacity_simulation"].sum()),
                "surgical_inclusion_criteria": SURGICAL_SPECIALTY_INCLUSION_CRITERIA,
            }
        ]
    )
)
display(clean_rtt.head(10))

if LAYER1_RUN_STAGE == "prepare":
    print("Layer 1 prepare stage complete. Stopping before model training.")
    sys.exit(0)

# %% Cell 5
import copy
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


def split_boundaries(frame: pd.DataFrame, config: Layer1Config) -> Dict[str, int]:
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


def prepare_model_frame(
    frame: pd.DataFrame,
    config: Layer1Config,
    target_column: str = "incomplete_total",
    target_available_column: Optional[str] = "incomplete_total_source_available",
    target_label: str = "total incomplete RTT pathways",
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    prepared = frame.copy().sort_values(["series_id", "time_idx"]).reset_index(drop=True)
    assert_net_inflow_integrity(prepared)
    if target_column not in prepared.columns:
        raise ValueError(f"Requested model target column is missing: {target_column}")
    boundaries = split_boundaries(prepared, config)
    train_period = prepared["time_idx"] < boundaries["validation_start_idx"]
    transformed_feature_columns = []
    feature_stats: Dict[str, Dict[str, object]] = {}

    for column in MODEL_INPUT_FEATURES:
        if column not in prepared.columns:
            raise ValueError(f"Prepared RTT frame is missing model input feature: {column}")
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
        train_values = transformed[train_period].dropna()
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
            "log1p": transform_name == "log1p_non_negative",
            "feature_group": feature_group_for_column(column),
            "missing_value_imputation": "training_mean_after_transform",
            "missing_observations": int(values.isna().sum()),
        }

    target_values = pd.to_numeric(prepared[target_column], errors="coerce")
    if target_available_column is not None and target_available_column in prepared.columns:
        available = pd.to_numeric(prepared[target_available_column], errors="coerce").fillna(0).astype(int).eq(1)
        target_values = target_values.where(available)
    train_target = target_values[train_period].dropna()
    if train_target.empty:
        raise ValueError(
            f"No non-missing training target values are available for {target_column}. "
            "Check Part 2A coverage before training a DTA optimisation model."
        )
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
        "raw_feature_columns": MODEL_INPUT_FEATURES,
        "feature_stats": feature_stats,
        "feature_groups": {column: feature_group_for_column(column) for column in MODEL_INPUT_FEATURES},
        "feature_group_names": sorted({feature_group_for_column(column) for column in MODEL_INPUT_FEATURES}),
        "missingness_feature_columns": [column for column in MODEL_INPUT_FEATURES if column.endswith("_missing")],
        "data_dictionary": data_dictionary_frame(MODEL_INPUT_FEATURES).to_dict(orient="records"),
        "target_column": target_column,
        "target_available_column": target_available_column,
        "target_label": target_label,
        "target_missing_observations": int(target_values.isna().sum()),
        "trust_to_idx": trust_to_idx,
        "specialty_to_idx": specialty_to_idx,
        "quantiles": list(QUANTILES),
        "config": asdict(config),
    }
    return prepared, metadata


def make_supervised_samples(frame: pd.DataFrame, config: Layer1Config, boundaries: Dict[str, int]) -> Dict[str, List[Dict[str, object]]]:
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
                "forecast_start_idx": forecast_start_idx,
                "forecast_end_idx": forecast_end_idx,
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

    if not samples["train"]:
        raise RuntimeError("No training samples were generated. Increase history or reduce encoder/prediction length.")
    if not samples["val"]:
        raise RuntimeError("No validation samples were generated. Increase validation months or reduce prediction length.")
    if not samples["test"]:
        raise RuntimeError("No test samples were generated. Increase test months or reduce prediction length.")
    return samples


class RTTSupervisedDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        samples: List[Dict[str, object]],
        feature_columns: Sequence[str],
        encoder_length: int,
        prediction_length: int,
    ) -> None:
        self.samples = samples
        self.feature_columns = list(feature_columns)
        self.encoder_length = encoder_length
        self.prediction_length = prediction_length
        self.series_frames: Dict[str, pd.DataFrame] = {}
        self.series_arrays: Dict[str, Dict[str, object]] = {}

        for series_id, group in frame.groupby("series_id", sort=False, observed=True):
            group = group.sort_values("time_idx").reset_index(drop=True)
            self.series_frames[series_id] = group
            self.series_arrays[series_id] = {
                "features": group[self.feature_columns].to_numpy(dtype=np.float32),
                "target_log": group["target_log"].to_numpy(dtype=np.float32),
                "target_actual": group["target_actual"].to_numpy(dtype=np.float32),
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
    safe_num_workers = 0 if os.name == "nt" else int(num_workers)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=safe_num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=safe_num_workers > 0,
    )


def build_supervised_datasets(
    prepared: pd.DataFrame,
    metadata: Dict[str, object],
    samples: Dict[str, List[Dict[str, object]]],
    config: Layer1Config,
) -> Dict[str, RTTSupervisedDataset]:
    return {
        split: RTTSupervisedDataset(
            frame=prepared,
            samples=samples[split],
            feature_columns=metadata["feature_columns"],
            encoder_length=config.encoder_length,
            prediction_length=config.prediction_length,
        )
        for split in ["train", "val", "test"]
    }


def build_model_from_config(model_config: Dict[str, object]) -> nn.Module:
    return TCNQuantileRegressor(
        n_features=int(model_config["n_features"]),
        n_trusts=int(model_config["n_trusts"]),
        n_specialties=int(model_config["n_specialties"]),
        prediction_length=int(model_config["prediction_length"]),
        quantiles=[float(value) for value in model_config["quantiles"]],
        hidden_channels=int(model_config["hidden_channels"]),
        tcn_levels=int(model_config["tcn_levels"]),
        kernel_size=int(model_config["kernel_size"]),
        dropout=float(model_config["dropout"]),
        embedding_dim=int(model_config["embedding_dim"]),
    )


def stable_json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def prepared_target_hash(prepared: pd.DataFrame, target_column: str) -> str:
    hash_columns = [
        column
        for column in [
            COLUMNS.series_id,
            COLUMNS.trust_code,
            COLUMNS.specialty_code,
            "month",
            "time_idx",
            target_column,
            "target_actual",
        ]
        if column in prepared.columns
    ]
    hash_frame = prepared[hash_columns].copy()
    if "month" in hash_frame.columns:
        hash_frame["month"] = pd.to_datetime(hash_frame["month"], errors="coerce").dt.strftime("%Y-%m-%d")
    row_hashes = pd.util.hash_pandas_object(hash_frame, index=False).to_numpy(dtype=np.uint64)
    return hashlib.sha256(row_hashes.tobytes()).hexdigest()


def build_artifact_fingerprint(
    prepared: pd.DataFrame,
    metadata: Dict[str, object],
    samples: Dict[str, List[Dict[str, object]]],
    config: Layer1Config,
    model_config: Dict[str, object],
) -> Dict[str, object]:
    target_column = str(metadata.get("target_column", "target_actual"))
    month_values = pd.to_datetime(prepared["month"], errors="coerce")
    model_config_core = {
        key: model_config[key]
        for key in [
            "model_class",
            "format",
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
        ]
        if key in model_config
    }
    return {
        "fingerprint_version": 1,
        "target_column": target_column,
        "target_available_column": metadata.get("target_available_column"),
        "target_label": metadata.get("target_label"),
        "row_count": int(len(prepared)),
        "series_count": int(prepared[COLUMNS.series_id].astype(str).nunique()),
        "trust_count": int(prepared[COLUMNS.trust_code].astype(str).nunique()),
        "specialty_count": int(prepared[COLUMNS.specialty_code].astype(str).nunique()),
        "min_month": month_values.min().date().isoformat(),
        "max_month": month_values.max().date().isoformat(),
        "boundaries": dict(metadata.get("boundaries", {})),
        "sample_counts": {split: int(len(split_samples)) for split, split_samples in samples.items()},
        "feature_columns_sha256": stable_json_hash(list(metadata.get("feature_columns", []))),
        "raw_feature_columns_sha256": stable_json_hash(list(metadata.get("raw_feature_columns", []))),
        "feature_stats_sha256": stable_json_hash(metadata.get("feature_stats", {})),
        "trust_mapping_sha256": stable_json_hash(metadata.get("trust_to_idx", {})),
        "specialty_mapping_sha256": stable_json_hash(metadata.get("specialty_to_idx", {})),
        "prepared_target_sha256": prepared_target_hash(prepared, target_column),
        "model_config_sha256": stable_json_hash(model_config_core),
        "runtime_config": {
            "encoder_length": int(config.encoder_length),
            "prediction_length": int(config.prediction_length),
            "validation_months": int(config.validation_months),
            "test_months": int(config.test_months),
            "random_seed": int(config.random_seed),
        },
    }


def validate_saved_artifact_fingerprint(
    prepared: pd.DataFrame,
    metadata: Dict[str, object],
    samples: Dict[str, List[Dict[str, object]]],
    config: Layer1Config,
    model_config: Dict[str, object],
    feature_metadata_path: Path,
    model_label: str,
) -> None:
    if not feature_metadata_path.exists():
        raise FileNotFoundError(f"Saved {model_label} feature metadata is missing: {feature_metadata_path}")
    with open(feature_metadata_path, "r", encoding="utf-8") as handle:
        saved_metadata = json.load(handle)
    saved_fingerprint = saved_metadata.get("artifact_fingerprint") or model_config.get("artifact_fingerprint")
    if not saved_fingerprint:
        raise ValueError(
            f"Saved {model_label} artifacts do not include a data/model fingerprint. "
            "Run `python run_pipeline.py train --force-retrain` so the model, model_config.json, "
            "and feature_metadata.json are regenerated together."
        )
    current_fingerprint = build_artifact_fingerprint(prepared, metadata, samples, config, model_config)
    compared_keys = sorted(set(saved_fingerprint) | set(current_fingerprint))
    mismatches = [
        key
        for key in compared_keys
        if saved_fingerprint.get(key) != current_fingerprint.get(key)
    ]
    if mismatches:
        mismatch_preview = {
            key: {"saved": saved_fingerprint.get(key), "current": current_fingerprint.get(key)}
            for key in mismatches[:8]
        }
        raise ValueError(
            f"Saved {model_label} artifacts do not match the current prepared data/features. "
            f"Mismatched fingerprint fields: {mismatch_preview}. "
            "Run `python run_pipeline.py train --force-prepare --force-retrain` to rebuild the full contract."
        )


def load_tcn_model_from_artifacts(
    prepared: pd.DataFrame,
    metadata: Dict[str, object],
    samples: Dict[str, List[Dict[str, object]]],
    config: Layer1Config,
    state_dict_path: Path,
    model_config_path: Path,
    feature_metadata_path: Path,
    model_label: str,
) -> Tuple[nn.Module, Dict[str, object], Dict[str, RTTSupervisedDataset]]:
    if not state_dict_path.exists() or not model_config_path.exists():
        raise FileNotFoundError(f"Saved {model_label} model artifacts are incomplete.")

    with open(model_config_path, "r", encoding="utf-8") as handle:
        model_config = json.load(handle)

    expected_features = int(model_config["n_features"])
    actual_features = len(metadata["feature_columns"])
    if expected_features != actual_features:
        raise ValueError(
            f"Saved {model_label} model expects {expected_features} features, "
            f"but the prepared frame has {actual_features} features."
        )
    validate_saved_artifact_fingerprint(
        prepared=prepared,
        metadata=metadata,
        samples=samples,
        config=config,
        model_config=model_config,
        feature_metadata_path=feature_metadata_path,
        model_label=model_label,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_from_config(model_config).to(device)
    state_dict = torch.load(state_dict_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    metadata["model_config"] = model_config
    datasets = build_supervised_datasets(prepared, metadata, samples, config)
    print(f"Loaded existing {model_label} model from: {state_dict_path}")
    return model, metadata, datasets


def train_or_load_tcn_model(
    prepared: pd.DataFrame,
    metadata: Dict[str, object],
    samples: Dict[str, List[Dict[str, object]]],
    config: Layer1Config,
    state_dict_path: Path,
    model_config_path: Path,
    feature_metadata_path: Path,
    training_history_path: Path,
    model_label: str,
) -> Tuple[nn.Module, Dict[str, object], Dict[str, RTTSupervisedDataset]]:
    force_retrain = os.environ.get("NHS_RTT_FORCE_RETRAIN", "").strip().lower() in {"1", "true", "yes"}
    if not force_retrain and state_dict_path.exists() and model_config_path.exists():
        return load_tcn_model_from_artifacts(
            prepared=prepared,
            metadata=metadata,
            samples=samples,
            config=config,
            state_dict_path=state_dict_path,
            model_config_path=model_config_path,
            feature_metadata_path=feature_metadata_path,
            model_label=model_label,
        )
    return train_tcn_model(
        prepared=prepared,
        metadata=metadata,
        samples=samples,
        config=config,
        state_dict_path=state_dict_path,
        model_config_path=model_config_path,
        feature_metadata_path=feature_metadata_path,
        training_history_path=training_history_path,
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
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()

        losses.append(float(loss.detach().cpu().item()))
    return float(np.mean(losses))


def train_tcn_model(
    prepared: pd.DataFrame,
    metadata: Dict[str, object],
    samples: Dict[str, List[Dict[str, object]]],
    config: Layer1Config,
    state_dict_path: Path = MODEL_STATE_DICT_PATH,
    model_config_path: Path = MODEL_CONFIG_PATH,
    feature_metadata_path: Path = FEATURE_METADATA_PATH,
    training_history_path: Path = TRAINING_HISTORY_PATH,
) -> Tuple[nn.Module, Dict[str, object], Dict[str, RTTSupervisedDataset]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets = {
        split: RTTSupervisedDataset(
            frame=prepared,
            samples=samples[split],
            feature_columns=metadata["feature_columns"],
            encoder_length=config.encoder_length,
            prediction_length=config.prediction_length,
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
    use_amp = torch.cuda.is_available()

    best_val_loss = math.inf
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, config.max_epochs + 1):
        train_loss = run_epoch(model, loaders["train"], loss_fn, optimizer, device, config.gradient_clip_norm, use_amp)
        val_loss = run_epoch(model, loaders["val"], loss_fn, None, device, config.gradient_clip_norm, use_amp)
        scheduler.step(val_loss)
        current_lr = float(optimizer.param_groups[0]["lr"])
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "learning_rate": current_lr})
        print(f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | lr={current_lr:.2e}")

        if val_loss < best_val_loss - 1.0e-5:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.early_stopping_patience:
                print(f"Early stopping at epoch {epoch}. Best val_loss={best_val_loss:.6f}")
                break

    model.load_state_dict(best_state)
    metadata = dict(metadata)
    metadata["best_val_loss"] = best_val_loss
    metadata["training_history"] = history
    metadata["backend"] = "manual_pytorch_tcn"
    model_config = build_tcn_model_config(
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
    )
    metadata["model_config"] = model_config
    artifact_fingerprint = build_artifact_fingerprint(
        prepared=prepared,
        metadata=metadata,
        samples=samples,
        config=config,
        model_config=model_config,
    )
    metadata["artifact_fingerprint"] = artifact_fingerprint
    model_config["artifact_fingerprint"] = artifact_fingerprint
    pd.DataFrame(history).to_csv(training_history_path, index=False)

    torch.save(model.state_dict(), state_dict_path)
    with open(model_config_path, "w", encoding="utf-8") as handle:
        json.dump(model_config, handle, indent=2)
    with open(feature_metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return model, metadata, datasets

# %% Cell 6
import pandas as pd


clean_rtt = pd.read_parquet(CLEAN_PARQUET_PATH)
prepared_rtt, feature_metadata = prepare_model_frame(
    clean_rtt,
    CONFIG,
    target_column="incomplete_total",
    target_available_column="incomplete_total_source_available",
    target_label="total incomplete RTT pathways",
)
supervised_samples = make_supervised_samples(prepared_rtt, CONFIG, feature_metadata["boundaries"])

print(
    "Samples:",
    {split: len(split_samples) for split, split_samples in supervised_samples.items()},
)
print(
    "Split boundaries:",
    feature_metadata["boundaries"],
)

tcn_model, feature_metadata, datasets = train_or_load_tcn_model(
    prepared=prepared_rtt,
    metadata=feature_metadata,
    samples=supervised_samples,
    config=CONFIG,
    state_dict_path=MODEL_STATE_DICT_PATH,
    model_config_path=MODEL_CONFIG_PATH,
    feature_metadata_path=FEATURE_METADATA_PATH,
    training_history_path=TRAINING_HISTORY_PATH,
    model_label="main TCN",
)

print(f"Saved model to: {MODEL_PATH}")
print(f"Saved model config to: {MODEL_CONFIG_PATH}")
print(f"Saved feature metadata to: {FEATURE_METADATA_PATH}")
print(f"Saved training history to: {TRAINING_HISTORY_PATH}")

dta_training_frame = clean_rtt[clean_rtt["is_surgical_specialty"].astype(bool)].copy()
if dta_training_frame.empty:
    raise RuntimeError(
        "No rows are available for configured surgical specialties. "
        "The decision-to-admit capacity simulation cannot be trained without eligible surgical series."
    )
if not pd.to_numeric(
    dta_training_frame["incomplete_decision_to_admit_source_available"],
    errors="coerce",
).fillna(0).astype(int).eq(1).any():
    raise RuntimeError(
        "No Part 2A decision-to-admit values are available for configured surgical specialties. "
        "The decision-to-admit capacity simulation cannot be trained without Part 2A coverage."
    )

prepared_dta_rtt, dta_feature_metadata = prepare_model_frame(
    dta_training_frame,
    CONFIG,
    target_column="incomplete_decision_to_admit",
    target_available_column="incomplete_decision_to_admit_source_available",
    target_label="incomplete RTT pathways with decision to admit",
)
dta_supervised_samples = make_supervised_samples(prepared_dta_rtt, CONFIG, dta_feature_metadata["boundaries"])
print(
    "DTA samples:",
    {split: len(split_samples) for split, split_samples in dta_supervised_samples.items()},
)
dta_tcn_model, dta_feature_metadata, dta_datasets = train_or_load_tcn_model(
    prepared=prepared_dta_rtt,
    metadata=dta_feature_metadata,
    samples=dta_supervised_samples,
    config=CONFIG,
    state_dict_path=DTA_MODEL_STATE_DICT_PATH,
    model_config_path=DTA_MODEL_CONFIG_PATH,
    feature_metadata_path=PATHS.dta_feature_metadata,
    training_history_path=DTA_TRAINING_HISTORY_PATH,
    model_label="DTA TCN",
)
print(f"Saved DTA model to: {DTA_MODEL_STATE_DICT_PATH}")
print(f"Saved DTA model config to: {DTA_MODEL_CONFIG_PATH}")
print(f"Saved DTA feature metadata to: {PATHS.dta_feature_metadata}")
print(f"Saved DTA training history to: {DTA_TRAINING_HISTORY_PATH}")

# %% Cell 7
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.ticker import FuncFormatter
from tqdm.auto import tqdm


def inverse_log1p(values: np.ndarray) -> np.ndarray:
    return np.clip(np.expm1(np.maximum(values, 0.0)), 0.0, None)


def predict_quantiles(
    model: torch.nn.Module,
    dataset: RTTSupervisedDataset,
    batch_size: int,
    num_workers: int,
    quantiles: Sequence[float],
) -> pd.DataFrame:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = make_loader(dataset, batch_size, False, num_workers)
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
            record = {
                "series_id": series_id,
                "trust_code": row["trust_code"],
                "trust_name": row["trust_name"],
                "specialty_code": row["specialty_code"],
                "specialty_name": row["specialty_name"],
                "forecast_month": row["month"],
                "time_idx": int(row["time_idx"]),
                "horizon": horizon_idx + 1,
                "actual": float(row["target_actual"]),
            }
            for quantile_index, quantile in enumerate(quantiles):
                record[f"q{int(round(quantile * 100)):02d}"] = float(predictions[batch_pos, horizon_idx, quantile_index])
            records.append(record)

    return pd.DataFrame(records)


def pinball_loss_np(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> np.ndarray:
    error = y_true - y_pred
    return np.maximum(quantile * error, (quantile - 1.0) * error)


def compute_metrics(predictions: pd.DataFrame, quantiles: Sequence[float]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    y = predictions["actual"].to_numpy(dtype=float)
    metric_rows = []
    pinball_values = []
    for quantile in quantiles:
        column = f"q{int(round(quantile * 100)):02d}"
        losses = pinball_loss_np(y, predictions[column].to_numpy(dtype=float), quantile)
        metric_rows.append({"metric": f"pinball_q{int(round(quantile * 100)):02d}", "value": float(np.mean(losses))})
        pinball_values.append(losses)

    median_prediction = predictions["q50"].to_numpy(dtype=float)
    errors = median_prediction - y
    crps_approx = 2.0 * np.mean(np.vstack(pinball_values), axis=0)
    metric_rows.extend(
        [
            {"metric": "pinball_mean", "value": float(np.mean(np.vstack(pinball_values)))},
            {"metric": "crps_quantile_approx", "value": float(np.mean(crps_approx))},
            {"metric": "rmse_median", "value": float(np.sqrt(np.mean(errors ** 2)))},
            {"metric": "mae_median", "value": float(np.mean(np.abs(errors)))},
        ]
    )
    metrics = pd.DataFrame(metric_rows)

    horizon_rows = []
    for horizon, group in predictions.groupby("horizon", observed=True):
        y_h = group["actual"].to_numpy(dtype=float)
        median_h = group["q50"].to_numpy(dtype=float)
        horizon_pinballs = []
        row = {"horizon": int(horizon)}
        for quantile in quantiles:
            column = f"q{int(round(quantile * 100)):02d}"
            losses = pinball_loss_np(y_h, group[column].to_numpy(dtype=float), quantile)
            row[f"pinball_q{int(round(quantile * 100)):02d}"] = float(np.mean(losses))
            horizon_pinballs.append(losses)
        row["pinball_mean"] = float(np.mean(np.vstack(horizon_pinballs)))
        row["crps_quantile_approx"] = float(np.mean(2.0 * np.mean(np.vstack(horizon_pinballs), axis=0)))
        row["rmse_median"] = float(np.sqrt(np.mean((median_h - y_h) ** 2)))
        row["mae_median"] = float(np.mean(np.abs(median_h - y_h)))
        horizon_rows.append(row)

    metrics_by_horizon = pd.DataFrame(horizon_rows).sort_values("horizon").reset_index(drop=True)
    return metrics, metrics_by_horizon


def reliability_summary(predictions: pd.DataFrame, label: str) -> pd.DataFrame:
    frame = predictions.copy()
    rename_map = {
        "q10": COLUMNS.p10,
        "q50": COLUMNS.p50,
        "q90": COLUMNS.p90,
    }
    frame = frame.rename(columns={old: new for old, new in rename_map.items() if old in frame.columns})
    required = [COLUMNS.actual, COLUMNS.p10, COLUMNS.p50, COLUMNS.p90]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} reliability summary is missing columns: {missing}")
    actual = pd.to_numeric(frame[COLUMNS.actual], errors="coerce")
    p10 = pd.to_numeric(frame[COLUMNS.p10], errors="coerce")
    p50 = pd.to_numeric(frame[COLUMNS.p50], errors="coerce")
    p90 = pd.to_numeric(frame[COLUMNS.p90], errors="coerce")
    valid = actual.notna() & p10.notna() & p50.notna() & p90.notna()
    if not valid.any():
        raise ValueError(f"{label} reliability summary has no complete prediction rows.")
    crossing_rate = float(((p10 > p50) | (p50 > p90))[valid].mean())
    coverage = float(((actual >= p10) & (actual <= p90))[valid].mean())
    return pd.DataFrame(
        [
            {
                "forecast_target": label,
                "n_rows": int(valid.sum()),
                "p10_p90_coverage": coverage,
                "average_interval_width": float((p90[valid] - p10[valid]).mean()),
                "median_absolute_error": float((p50[valid] - actual[valid]).abs().median()),
                "quantile_crossing_rate": crossing_rate,
            }
        ]
    )


def safe_filename(value: str, max_length: int = 150) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_")
    if not cleaned:
        cleaned = "unnamed"
    return cleaned[:max_length]


def y_axis_thousands(x: float, position: int) -> str:
    if abs(x) >= 1000:
        return f"{x / 1000:.0f}k"
    return f"{x:.0f}"


def plot_forecasts_by_trust(predictions: pd.DataFrame, plot_dir: Path, plot_all: bool, max_plots: Optional[int]) -> pd.DataFrame:
    plot_dir.mkdir(parents=True, exist_ok=True)
    trust_totals = (
        predictions.groupby(["trust_code", "trust_name"], as_index=False, observed=True)["actual"]
        .sum()
        .sort_values("actual", ascending=False)
    )
    if not plot_all and max_plots is not None:
        trust_totals = trust_totals.head(max_plots)
    if plot_all and max_plots is not None:
        trust_totals = trust_totals.head(max_plots)

    records = []
    for row in tqdm(trust_totals.itertuples(index=False), total=len(trust_totals), desc="Saving trust forecast plots"):
        trust_predictions = predictions[predictions["trust_code"] == row.trust_code]
        monthly = (
            trust_predictions.groupby("forecast_month", as_index=False, observed=True)[["actual", "q10", "q50", "q90"]]
            .sum()
            .sort_values("forecast_month")
        )
        fig, ax = plt.subplots(figsize=(11, 5.5))
        x_values = pd.to_datetime(monthly["forecast_month"])
        ax.fill_between(
            x_values,
            monthly["q10"].to_numpy(dtype=float),
            monthly["q90"].to_numpy(dtype=float),
            color="#9ecae1",
            alpha=0.45,
            label="10th-90th percentile",
        )
        ax.plot(x_values, monthly["q50"], color="#08519c", linewidth=2.2, label="Forecast median")
        ax.plot(x_values, monthly["actual"], color="#111111", linewidth=1.8, linestyle="--", label="Actual")
        ax.set_title(f"{row.trust_name} ({row.trust_code})")
        ax.set_xlabel("Forecast month")
        ax.set_ylabel("Waiting list size")
        ax.yaxis.set_major_formatter(FuncFormatter(y_axis_thousands))
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
        fig.autofmt_xdate()
        fig.tight_layout()
        filename = f"{safe_filename(row.trust_code)}_{safe_filename(row.trust_name)}.png"
        output_path = plot_dir / filename
        fig.savefig(output_path, dpi=160)
        plt.close(fig)
        records.append({"trust_code": row.trust_code, "trust_name": row.trust_name, "plot_path": str(output_path)})
    return pd.DataFrame(records)


def canonicalise_backtest_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    frame = predictions.rename(columns={"q10": COLUMNS.p10, "q50": COLUMNS.p50, "q90": COLUMNS.p90}).copy()
    frame[COLUMNS.forecast_month] = pd.to_datetime(frame[COLUMNS.forecast_month])
    frame[COLUMNS.forecast_origin] = [
        month - pd.DateOffset(months=int(horizon))
        for month, horizon in zip(frame[COLUMNS.forecast_month], frame["horizon"])
    ]
    required_order = [
        COLUMNS.trust_code,
        COLUMNS.trust_name,
        COLUMNS.specialty_code,
        COLUMNS.specialty_name,
        COLUMNS.forecast_origin,
        COLUMNS.forecast_month,
        COLUMNS.horizon,
        COLUMNS.p10,
        COLUMNS.p50,
        COLUMNS.p90,
        COLUMNS.actual,
    ]
    frame = frame[required_order].sort_values(
        [COLUMNS.trust_code, COLUMNS.specialty_code, COLUMNS.forecast_month]
    ).reset_index(drop=True)
    return validate_backtest_predictions_frame(frame, "backtest predictions")


def predict_future_quantiles(
    model: torch.nn.Module,
    prepared: pd.DataFrame,
    metadata: Dict[str, object],
    config: Layer1Config,
    quantiles: Sequence[float],
    latest_observed_column: str = "incomplete_total",
    latest_output_column: str = COLUMNS.latest_observed_waiting_list,
    latest_available_column: Optional[str] = "incomplete_total_source_available",
    forecast_target: str = "incomplete_total",
    optimisation_output: bool = False,
) -> pd.DataFrame:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)
    records: List[Dict[str, object]] = []
    final_observed_month = pd.to_datetime(prepared["month"]).max()

    # The custom TCN is a direct multi-horizon encoder model. It does not consume future decoder
    # operational covariates, so unknown future referrals/completions are not fabricated. The only
    # future field generated here is forecast_month; every model input comes from the observed
    # encoder window ending at forecast_origin with the same scaled features used during training.
    with torch.no_grad():
        grouped = prepared.groupby("series_id", sort=False, observed=True)
        for series_id, group in tqdm(grouped, total=grouped.ngroups, desc="Forecasting future horizon"):
            group = group.sort_values("time_idx").reset_index(drop=True)
            if len(group) < config.encoder_length:
                continue
            if pd.to_datetime(group["month"].iloc[-1]) != final_observed_month:
                continue
            encoder = group.tail(config.encoder_length)
            latest = group.iloc[-1]
            x = torch.tensor(
                encoder[metadata["feature_columns"]].to_numpy(dtype=np.float32)[None, :, :],
                dtype=torch.float32,
                device=device,
            )
            trust_idx = torch.tensor([int(latest["trust_idx"])], dtype=torch.long, device=device)
            specialty_idx = torch.tensor([int(latest["specialty_idx"])], dtype=torch.long, device=device)
            prediction_log = model(x, trust_idx, specialty_idx).detach().cpu().numpy()
            prediction_log.sort(axis=-1)
            prediction = inverse_log1p(prediction_log)[0]
            forecast_origin = pd.to_datetime(latest["month"])
            encoder_latest_values = pd.to_numeric(encoder[latest_observed_column], errors="coerce")
            if latest_available_column is not None and latest_available_column in encoder.columns:
                available_mask = pd.to_numeric(
                    encoder[latest_available_column],
                    errors="coerce",
                ).fillna(0).astype(int).eq(1)
                encoder_latest_values = encoder_latest_values.where(available_mask)
            encoder_latest_values = encoder_latest_values.dropna()
            if encoder_latest_values.empty:
                continue
            else:
                current_value = float(encoder_latest_values.iloc[-1])

            for horizon_idx in range(config.prediction_length):
                quantile_values = {
                    f"q{int(round(float(quantile) * 100)):02d}": float(prediction[horizon_idx, quantile_index])
                    for quantile_index, quantile in enumerate(quantiles)
                }
                records.append(
                    {
                        COLUMNS.series_id: str(series_id),
                        COLUMNS.trust_code: str(latest[COLUMNS.trust_code]),
                        COLUMNS.trust_name: str(latest[COLUMNS.trust_name]),
                        COLUMNS.specialty_code: str(latest[COLUMNS.specialty_code]),
                        COLUMNS.specialty_name: str(latest[COLUMNS.specialty_name]),
                        COLUMNS.forecast_origin: forecast_origin,
                        COLUMNS.forecast_month: final_observed_month + pd.DateOffset(months=horizon_idx + 1),
                        COLUMNS.horizon: horizon_idx + 1,
                        COLUMNS.p10: quantile_values.get("q10", quantile_values.get("q10", np.nan)),
                        COLUMNS.p50: quantile_values.get("q50", np.nan),
                        COLUMNS.p90: quantile_values.get("q90", quantile_values.get("q90", np.nan)),
                        latest_output_column: current_value,
                        COLUMNS.forecast_target: forecast_target,
                        COLUMNS.is_surgical_specialty: bool(latest.get(COLUMNS.is_surgical_specialty, False)),
                        COLUMNS.specialty_inclusion_criteria: str(latest.get(COLUMNS.specialty_inclusion_criteria, "")),
                    }
                )

    if not records:
        raise RuntimeError(
            "No future forecasts could be generated. No Trust-specialty series had a complete encoder "
            f"window ending at the final observed month {final_observed_month.date()}."
        )
    output_columns = OPTIMISATION_FORECAST_COLUMNS if optimisation_output else FUTURE_FORECAST_COLUMNS
    future = pd.DataFrame(records)[output_columns].sort_values(
        [COLUMNS.trust_code, COLUMNS.specialty_code, COLUMNS.forecast_month]
    ).reset_index(drop=True)
    if optimisation_output:
        return validate_optimisation_forecast_frame(future, "future optimisation forecasts")
    return validate_future_forecast_frame(future, "future forecasts")


def validate_forecast_outputs(
    backtest: pd.DataFrame,
    future: pd.DataFrame,
    final_observed_month: pd.Timestamp,
    expected_horizon: int,
) -> None:
    backtest_checked = validate_backtest_predictions_frame(backtest, "backtest predictions")
    future_checked = validate_future_forecast_frame(future, "future forecasts")

    assert backtest_checked[COLUMNS.actual].notna().all(), "Backtest predictions must contain actual values."
    assert COLUMNS.actual not in future_checked.columns, "Future forecasts must not contain an actual column."
    assert (future_checked[COLUMNS.forecast_origin] == final_observed_month).all(), (
        "Future forecast_origin must equal the final observed month."
    )
    assert (future_checked[COLUMNS.forecast_month] > future_checked[COLUMNS.forecast_origin]).all(), (
        "Every future forecast_month must be later than forecast_origin."
    )
    assert int(future_checked[COLUMNS.horizon].min()) == 1, "Future horizons must start at 1."
    assert int(future_checked[COLUMNS.horizon].max()) == int(expected_horizon), (
        f"Future horizons must end at {expected_horizon}."
    )
    month_delta = (
        (future_checked[COLUMNS.forecast_month].dt.year - future_checked[COLUMNS.forecast_origin].dt.year) * 12
        + (future_checked[COLUMNS.forecast_month].dt.month - future_checked[COLUMNS.forecast_origin].dt.month)
    ).astype(int)
    assert (month_delta == future_checked[COLUMNS.horizon]).all(), (
        "Future horizon must match the month difference between forecast_origin and forecast_month."
    )

# %% Cell 8
import pandas as pd


test_predictions = predict_quantiles(
    model=tcn_model,
    dataset=datasets["test"],
    batch_size=CONFIG.batch_size,
    num_workers=CONFIG.num_workers,
    quantiles=QUANTILES,
)
backtest_predictions = canonicalise_backtest_predictions(test_predictions)
dta_test_predictions = predict_quantiles(
    model=dta_tcn_model,
    dataset=dta_datasets["test"],
    batch_size=CONFIG.batch_size,
    num_workers=CONFIG.num_workers,
    quantiles=QUANTILES,
)
dta_backtest_predictions = canonicalise_backtest_predictions(dta_test_predictions)
future_forecasts = predict_future_quantiles(
    model=tcn_model,
    prepared=prepared_rtt,
    metadata=feature_metadata,
    config=CONFIG,
    quantiles=QUANTILES,
    latest_observed_column="incomplete_total",
    latest_output_column=COLUMNS.latest_observed_waiting_list,
    latest_available_column="incomplete_total_source_available",
    forecast_target="incomplete_total",
    optimisation_output=False,
)
future_optimisation_forecasts = predict_future_quantiles(
    model=dta_tcn_model,
    prepared=prepared_dta_rtt,
    metadata=dta_feature_metadata,
    config=CONFIG,
    quantiles=QUANTILES,
    latest_observed_column="incomplete_decision_to_admit",
    latest_output_column=COLUMNS.latest_observed_incomplete_decision_to_admit,
    latest_available_column="incomplete_decision_to_admit_source_available",
    forecast_target=COLUMNS.incomplete_decision_to_admit,
    optimisation_output=True,
)
final_observed_month = pd.to_datetime(prepared_rtt["month"]).max()
validate_forecast_outputs(
    backtest=backtest_predictions,
    future=future_forecasts,
    final_observed_month=final_observed_month,
    expected_horizon=CONFIG.prediction_length,
)
validate_optimisation_forecast_frame(future_optimisation_forecasts, "future optimisation forecasts")
test_metrics, test_metrics_by_horizon = compute_metrics(test_predictions, QUANTILES)
dta_test_metrics, dta_test_metrics_by_horizon = compute_metrics(dta_test_predictions, QUANTILES)
dta_reliability = reliability_summary(dta_test_predictions, "incomplete_decision_to_admit")
trust_plot_index = plot_forecasts_by_trust(
    predictions=test_predictions,
    plot_dir=PLOT_DIR,
    plot_all=CONFIG.plot_all_trusts,
    max_plots=CONFIG.max_trust_plots,
)
backtest_predictions.to_parquet(BACKTEST_PREDICTIONS_PATH, index=False)
dta_backtest_predictions.to_parquet(DTA_BACKTEST_PREDICTIONS_PATH, index=False)
future_forecasts.to_parquet(FUTURE_FORECASTS_PATH, index=False)
future_optimisation_forecasts.to_parquet(FUTURE_OPTIMISATION_FORECASTS_PATH, index=False)
test_metrics.to_csv(FORECAST_METRICS_PATH, index=False)
test_metrics_by_horizon.to_csv(FORECAST_METRICS_BY_HORIZON_PATH, index=False)
dta_test_metrics.to_csv(DTA_FORECAST_METRICS_PATH, index=False)
dta_test_metrics_by_horizon.to_csv(DTA_FORECAST_METRICS_BY_HORIZON_PATH, index=False)
dta_reliability.to_csv(DTA_RELIABILITY_SUMMARY_PATH, index=False)
trust_plot_index.to_csv(FORECAST_PLOT_INDEX_PATH, index=False)
print("Saved core forecast outputs before baseline comparison.")

baseline_config = BaselineComparisonConfig(
    target_column="incomplete_total",
    enable_hist_gradient_boosting=CONFIG.enable_hist_gradient_boosting_baseline,
    hist_gradient_boosting_max_train_rows=CONFIG.hist_gradient_boosting_max_train_rows,
    hist_gradient_boosting_max_iter=CONFIG.hist_gradient_boosting_max_iter,
    random_seed=CONFIG.random_seed,
)
baseline_results = run_baseline_comparison(
    clean_frame=clean_rtt,
    backtest_predictions=backtest_predictions,
    feature_metadata=feature_metadata,
    config=baseline_config,
)
save_baseline_comparison_outputs(
    baseline_results,
    paths={
        "predictions": MODEL_COMPARISON_PREDICTIONS_PATH,
        "overall": MODEL_COMPARISON_PATH,
        "by_horizon": MODEL_COMPARISON_BY_HORIZON_PATH,
        "by_specialty": MODEL_COMPARISON_BY_SPECIALTY_PATH,
        "by_trust_size": MODEL_COMPARISON_BY_TRUST_SIZE_PATH,
        "by_covid_period": MODEL_COMPARISON_BY_COVID_PERIOD_PATH,
        "paired_errors": MODEL_COMPARISON_PAIRED_ERRORS_PATH,
        "audit_log": MODEL_COMPARISON_AUDIT_LOG_PATH,
        "summary": MODEL_COMPARISON_SUMMARY_PATH,
    },
)
save_model_comparison_plots(
    baseline_results,
    overall_png=MODEL_COMPARISON_OVERALL_PNG_PATH,
    by_horizon_png=MODEL_COMPARISON_BY_HORIZON_PNG_PATH,
    tcn_vs_seasonal_png=MODEL_COMPARISON_TCN_VS_SEASONAL_PNG_PATH,
)

dta_baseline_config = BaselineComparisonConfig(
    target_column="incomplete_decision_to_admit",
    enable_hist_gradient_boosting=CONFIG.enable_hist_gradient_boosting_baseline,
    hist_gradient_boosting_max_train_rows=CONFIG.hist_gradient_boosting_max_train_rows,
    hist_gradient_boosting_max_iter=CONFIG.hist_gradient_boosting_max_iter,
    random_seed=CONFIG.random_seed,
)
dta_baseline_results = run_baseline_comparison(
    clean_frame=dta_training_frame,
    backtest_predictions=dta_backtest_predictions,
    feature_metadata=dta_feature_metadata,
    config=dta_baseline_config,
)
save_baseline_comparison_outputs(
    dta_baseline_results,
    paths={
        "predictions": DTA_MODEL_COMPARISON_PREDICTIONS_PATH,
        "overall": DTA_MODEL_COMPARISON_PATH,
        "by_horizon": DTA_MODEL_COMPARISON_BY_HORIZON_PATH,
        "by_specialty": DTA_MODEL_COMPARISON_BY_SPECIALTY_PATH,
        "by_trust_size": DTA_MODEL_COMPARISON_BY_TRUST_SIZE_PATH,
        "by_covid_period": DTA_MODEL_COMPARISON_BY_COVID_PERIOD_PATH,
        "paired_errors": DTA_MODEL_COMPARISON_PAIRED_ERRORS_PATH,
        "audit_log": DTA_MODEL_COMPARISON_AUDIT_LOG_PATH,
        "summary": DTA_MODEL_COMPARISON_SUMMARY_PATH,
    },
)

backtest_predictions.to_parquet(BACKTEST_PREDICTIONS_PATH, index=False)
dta_backtest_predictions.to_parquet(DTA_BACKTEST_PREDICTIONS_PATH, index=False)
future_forecasts.to_parquet(FUTURE_FORECASTS_PATH, index=False)
future_optimisation_forecasts.to_parquet(FUTURE_OPTIMISATION_FORECASTS_PATH, index=False)
test_metrics.to_csv(FORECAST_METRICS_PATH, index=False)
test_metrics_by_horizon.to_csv(FORECAST_METRICS_BY_HORIZON_PATH, index=False)
dta_test_metrics.to_csv(DTA_FORECAST_METRICS_PATH, index=False)
dta_test_metrics_by_horizon.to_csv(DTA_FORECAST_METRICS_BY_HORIZON_PATH, index=False)
dta_reliability.to_csv(DTA_RELIABILITY_SUMMARY_PATH, index=False)
trust_plot_index.to_csv(FORECAST_PLOT_INDEX_PATH, index=False)

print(f"Saved backtest predictions to: {BACKTEST_PREDICTIONS_PATH}")
print(f"Saved DTA backtest predictions to: {DTA_BACKTEST_PREDICTIONS_PATH}")
print(f"Saved future forecasts to: {FUTURE_FORECASTS_PATH}")
print(f"Saved future optimisation Part 2A forecasts to: {FUTURE_OPTIMISATION_FORECASTS_PATH}")
print(f"Saved forecast metrics to: {FORECAST_METRICS_PATH}")
print(f"Saved forecast metrics by horizon to: {FORECAST_METRICS_BY_HORIZON_PATH}")
print(f"Saved DTA forecast metrics to: {DTA_FORECAST_METRICS_PATH}")
print(f"Saved DTA forecast metrics by horizon to: {DTA_FORECAST_METRICS_BY_HORIZON_PATH}")
print(f"Saved DTA reliability summary to: {DTA_RELIABILITY_SUMMARY_PATH}")
print(f"Saved trust forecast plots in: {PLOT_DIR}")
print(f"Saved model comparison to: {MODEL_COMPARISON_PATH}")
print(f"Saved model comparison by horizon to: {MODEL_COMPARISON_BY_HORIZON_PATH}")
print(f"Saved model comparison by specialty to: {MODEL_COMPARISON_BY_SPECIALTY_PATH}")
print(f"Saved model comparison plots in: {PATHS.model_comparison_plot_dir}")
print(f"Saved DTA model comparison to: {DTA_MODEL_COMPARISON_PATH}")
forecast_summary = {
    "final_observed_month": final_observed_month.date().isoformat(),
    "forecast_origin": future_forecasts[COLUMNS.forecast_origin].iloc[0].date().isoformat(),
    "forecast_horizon": int(future_forecasts[COLUMNS.horizon].max()),
    "trust_specialty_series_forecast": int(
        future_forecasts[
            [COLUMNS.trust_code, COLUMNS.specialty_code]
        ].drop_duplicates().shape[0]
    ),
    "future_rows_produced": int(len(future_forecasts)),
    "part2a_surgical_series_forecast_for_optimisation": int(
        future_optimisation_forecasts[
            [COLUMNS.trust_code, COLUMNS.specialty_code]
        ].drop_duplicates().shape[0]
    ),
    "future_optimisation_rows_produced": int(len(future_optimisation_forecasts)),
    "backtest_rows_produced": int(len(backtest_predictions)),
    "dta_backtest_rows_produced": int(len(dta_backtest_predictions)),
}
print(json.dumps(forecast_summary, indent=2))
display(test_metrics)
display(test_metrics_by_horizon)
display(dta_test_metrics)
display(dta_test_metrics_by_horizon)
display(dta_reliability)
display(baseline_results.model_comparison)
display(baseline_results.by_horizon.head(30))
display(baseline_results.paired_errors)
display(dta_baseline_results.model_comparison)
display(dta_baseline_results.by_horizon.head(30))
print(baseline_results.summary_text)
print(dta_baseline_results.summary_text)
display(backtest_predictions.head(20))
display(dta_backtest_predictions.head(20))
display(future_forecasts.head(20))
display(future_optimisation_forecasts.head(20))

