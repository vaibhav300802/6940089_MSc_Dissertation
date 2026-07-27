from __future__ import annotations

import re
from io import BytesIO
from pathlib import Path
from typing import Optional

import pandas as pd


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


def month_start(value: object) -> pd.Timestamp:
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        raise ValueError(f"Could not parse month value: {value!r}")
    return pd.Timestamp(timestamp).to_period("M").to_timestamp()


def financial_year_slug(start_year: int) -> str:
    return f"{int(start_year)}-{str(int(start_year) + 1)[-2:]}"


def parse_period_from_text(text: object) -> Optional[pd.Timestamp]:
    value = str(text).upper().replace("_", " ").replace("-", " ")
    month_pattern = "|".join(sorted(MONTH_LOOKUP, key=len, reverse=True))

    month_year = re.search(rf"\b({month_pattern})[A-Z]*\s+(20\d{{2}})\b", value)
    if month_year:
        return pd.Timestamp(int(month_year.group(2)), MONTH_LOOKUP[month_year.group(1)], 1)

    year_month = re.search(rf"\b(20\d{{2}})\s+({month_pattern})[A-Z]*\b", value)
    if year_month:
        return pd.Timestamp(int(year_month.group(1)), MONTH_LOOKUP[year_month.group(2)], 1)

    compact = re.search(rf"\b({month_pattern})(20\d{{2}})\b", value)
    if compact:
        return pd.Timestamp(int(compact.group(2)), MONTH_LOOKUP[compact.group(1)], 1)

    iso = re.search(r"\b(20\d{2})[.\s]*(0?[1-9]|1[0-2])\b", value)
    if iso:
        return pd.Timestamp(int(iso.group(1)), int(iso.group(2)), 1)

    return None


def parse_month_from_current_period(value: object) -> Optional[pd.Timestamp]:
    if pd.isna(value):
        return None
    parsed = pd.to_datetime(value, errors="coerce", dayfirst=True)
    if not pd.isna(parsed):
        return pd.Timestamp(parsed).to_period("M").to_timestamp()
    return parse_period_from_text(value)


def parse_month_from_legacy_period(value: object, financial_year_start: int | None = None) -> Optional[pd.Timestamp]:
    if pd.isna(value):
        return None
    parsed = parse_period_from_text(value)
    if parsed is not None:
        return parsed
    if financial_year_start is None:
        return None
    month_number = None
    for token in re.findall(r"\b[A-Z]+\b", str(value).upper()):
        candidates = [token, token[:4], token[:3]]
        for candidate in candidates:
            if candidate in MONTH_LOOKUP:
                month_number = MONTH_LOOKUP[candidate]
                break
        if month_number is not None:
            break
    if month_number is None:
        return None
    year = int(financial_year_start) if month_number >= 4 else int(financial_year_start) + 1
    return pd.Timestamp(year, month_number, 1)


def numeric_series(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace("nan", "", regex=False)
        .str.replace("*", "", regex=False)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def coalesce_numeric(frame: pd.DataFrame, candidates: list[str]) -> pd.Series:
    result = pd.Series(pd.NA, index=frame.index, dtype="Float64")
    for column in candidates:
        if column in frame.columns:
            result = result.fillna(numeric_series(frame[column]))
    return result.astype(float)


def standardise_part(value: object) -> Optional[str]:
    text = str(value).upper().replace("_", " ")
    match = re.search(r"\bPART\s*([123])\s*([AB]?)\b", text)
    if not match:
        return None
    suffix = match.group(2)
    return f"PART_{match.group(1)}{suffix}" if suffix else f"PART_{match.group(1)}"


def read_rtt_csv_bytes(content: bytes, **kwargs: object) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "cp1252", "latin1"):
        try:
            return pd.read_csv(BytesIO(content), encoding=encoding, **kwargs)
        except Exception as exc:
            last_error = exc
    raise ValueError(f"Could not read RTT CSV bytes with common encodings: {last_error}")


def extract_publication_month_from_path(path: str | Path) -> Optional[pd.Timestamp]:
    return parse_period_from_text(Path(path).stem)
