from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path
from typing import Iterable, Optional, Sequence

import pandas as pd


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nhs_rtt_pipeline.config import COLUMNS, get_paths


PATHS = get_paths(PROJECT_ROOT)
ODS_TRUST_URL = "https://www.odsdatasearchandexport.nhs.uk/api/getReport?report=etr"
POSTCODES_API_URL = "https://api.postcodes.io/postcodes"


def infer_column(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    lookup = {str(column).strip().lower(): str(column) for column in columns}
    for candidate in candidates:
        key = candidate.strip().lower()
        if key in lookup:
            return lookup[key]
    for candidate in candidates:
        key = candidate.strip().lower()
        for column in columns:
            if key in str(column).strip().lower():
                return str(column)
    return None


def normalise_postcode(value: object) -> str:
    return "".join(str(value or "").upper().split())


def chunks(values: Sequence[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(values), size):
        yield list(values[start : start + size])


def download_ods_trusts(raw_path: Path) -> pd.DataFrame:
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if not raw_path.exists():
        urllib.request.urlretrieve(ODS_TRUST_URL, raw_path)
    frame = pd.read_csv(raw_path, dtype=str).fillna("")
    code_col = infer_column(frame.columns, ["OrganisationCode", "Organisation Code"])
    name_col = infer_column(frame.columns, ["OrgName", "Name"])
    postcode_col = infer_column(frame.columns, ["Postcode", "Post Code"])
    end_col = infer_column(frame.columns, ["LegalEndDate", "Close Date", "Legal End Date"])
    region_col = infer_column(frame.columns, ["NHSER_code", "National Grouping", "GOR_code"])
    missing = {
        "trust_code": code_col,
        "trust_name": name_col,
        "postcode": postcode_col,
    }
    absent = [label for label, column in missing.items() if column is None]
    if absent:
        legacy_columns = [
            "OrganisationCode",
            "OrgName",
            "NHSER_code",
            "ICB_code",
            "Address1",
            "Address2",
            "Address3",
            "Town",
            "County",
            "Postcode",
            "LegalStartDate",
            "LegalEndDate",
            "Column13",
            "Column14",
            "Column15",
            "Column16",
            "Column17",
            "TelephoneNumber",
            "Column19",
            "Column20",
            "Column21",
            "AmendedRecordIndicator",
            "Column23",
            "GOR_code",
            "Column25",
            "Column26",
            "Column27",
        ]
        frame = pd.read_csv(raw_path, dtype=str, header=None).fillna("")
        frame.columns = legacy_columns[: len(frame.columns)]
        code_col = "OrganisationCode"
        name_col = "OrgName"
        postcode_col = "Postcode"
        end_col = "LegalEndDate"
        region_col = "NHSER_code"
    keep = [code_col, name_col, postcode_col]
    if end_col:
        keep.append(end_col)
    if region_col:
        keep.append(region_col)
    trusts = frame[keep].copy()
    rename_map = {
        code_col: COLUMNS.trust_code,
        name_col: COLUMNS.trust_name,
        postcode_col: "postcode",
    }
    if end_col:
        rename_map[end_col] = "legal_end_date"
    if region_col:
        rename_map[region_col] = "region_code"
    trusts = trusts.rename(columns=rename_map)
    trusts[COLUMNS.trust_code] = trusts[COLUMNS.trust_code].astype(str).str.strip()
    trusts[COLUMNS.trust_name] = trusts[COLUMNS.trust_name].astype(str).str.strip()
    trusts["postcode"] = trusts["postcode"].map(normalise_postcode)
    if "legal_end_date" in trusts.columns:
        trusts = trusts[trusts["legal_end_date"].astype(str).str.strip().eq("")]
    return trusts.drop_duplicates(COLUMNS.trust_code, keep="first").reset_index(drop=True)


def geocode_postcodes(postcodes: Sequence[str]) -> pd.DataFrame:
    unique_postcodes = sorted({normalise_postcode(postcode) for postcode in postcodes if normalise_postcode(postcode)})
    rows: list[dict[str, object]] = []
    for batch in chunks(unique_postcodes, 100):
        payload = json.dumps({"postcodes": batch}).encode("utf-8")
        request = urllib.request.Request(
            POSTCODES_API_URL,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            result = json.loads(response.read().decode("utf-8"))
        for item in result.get("result", []):
            query = normalise_postcode(item.get("query"))
            data = item.get("result") or {}
            latitude = data.get("latitude")
            longitude = data.get("longitude")
            if latitude is None or longitude is None:
                continue
            rows.append(
                {
                    "postcode": query,
                    "latitude": float(latitude),
                    "longitude": float(longitude),
                    "region": str(data.get("region") or ""),
                    "admin_district": str(data.get("admin_district") or ""),
                    "country": str(data.get("country") or ""),
                }
            )
        time.sleep(0.1)
    return pd.DataFrame(rows).drop_duplicates("postcode", keep="first")


def build_coordinates() -> pd.DataFrame:
    PATHS.dashboard_data_dir.mkdir(parents=True, exist_ok=True)
    raw_reference_dir = PATHS.raw_dir / "reference"
    ods_raw_path = raw_reference_dir / "ods_nhs_trusts_etr.csv"
    trusts = download_ods_trusts(ods_raw_path)
    forecasts = pd.read_parquet(PATHS.future_forecasts)
    forecast_trusts = (
        forecasts[[COLUMNS.trust_code, COLUMNS.trust_name]]
        .drop_duplicates(COLUMNS.trust_code, keep="last")
        .reset_index(drop=True)
    )
    matched = forecast_trusts.merge(
        trusts[[COLUMNS.trust_code, "postcode", "region_code"] if "region_code" in trusts.columns else [COLUMNS.trust_code, "postcode"]],
        on=COLUMNS.trust_code,
        how="inner",
    )
    geocoded = geocode_postcodes(matched["postcode"].tolist())
    coordinates = matched.merge(geocoded, on="postcode", how="inner")
    coordinates = coordinates[
        coordinates["latitude"].between(49.5, 56.5) & coordinates["longitude"].between(-6.5, 2.5)
    ].copy()
    coordinates["coordinate_source"] = "NHS ODS Trust postcode geocoded with postcodes.io"
    output_columns = [
        COLUMNS.trust_code,
        COLUMNS.trust_name,
        "latitude",
        "longitude",
        "region",
        "admin_district",
        "postcode",
        "coordinate_source",
    ]
    coordinates = coordinates[output_columns].sort_values(COLUMNS.trust_name).reset_index(drop=True)
    output_path = PATHS.dashboard_data_dir / "nhs_trust_coordinates.csv"
    coordinates.to_csv(output_path, index=False)
    print(f"Saved Trust coordinates to: {output_path}")
    print(f"Forecast Trusts: {forecast_trusts[COLUMNS.trust_code].nunique():,}")
    print(f"Trusts with coordinates: {coordinates[COLUMNS.trust_code].nunique():,}")
    return coordinates


if __name__ == "__main__":
    build_coordinates()
