from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .config import COLUMNS, require_columns


DATA_START_MONTH = pd.Timestamp("2015-10-01")
STOCK_FORWARD_FILL_COLUMNS = ["waiting_list", "waiting_list_with_dta"]
ACTIVITY_NO_FILL_COLUMNS = ["completed_admitted", "completed_non_admitted", "new_rtt_periods"]


@dataclass(frozen=True)
class DataQualityConfig:
    start_month: str = "2015-10-01"
    min_series_length: int = 36
    large_jump_absolute_threshold: float = 5000.0
    large_jump_relative_threshold: float = 1.0
    part2a_exceeds_total_warning_rate: float = 0.05


def normalise_identifier(value: object) -> str:
    return " ".join(str(value).strip().split())


def normalise_code(value: object) -> str:
    code = normalise_identifier(value).upper()
    if code.startswith("C_") and code[2:].isdigit():
        return code[2:]
    return code


def month_start(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.to_period("M").dt.to_timestamp()


def harmonise_trust_and_specialty_identifiers(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    required = ["trust_code", "trust_name", "specialty_code", "specialty_name"]
    require_columns(working, required, "raw RTT monthly data before identifier harmonisation")
    working["source_trust_code"] = working["trust_code"].map(normalise_identifier)
    working["source_trust_name"] = working["trust_name"].map(normalise_identifier)
    working["source_specialty_code"] = working["specialty_code"].map(normalise_identifier)
    working["source_specialty_name"] = working["specialty_name"].map(normalise_identifier)
    working["trust_code"] = working["source_trust_code"].map(normalise_code)
    working["trust_name"] = working["source_trust_name"].map(normalise_identifier)
    working["specialty_code"] = working["source_specialty_code"].map(normalise_code)
    working["specialty_name"] = working["source_specialty_name"].map(normalise_identifier)
    working["trust_identifier_harmonisation_rule"] = "identity_trim_upper_code_normalise_name_no_organisation_merge"
    working["specialty_identifier_harmonisation_rule"] = "identity_trim_upper_code_normalise_name_no_specialty_merge"
    return working


def _join_unique(values: Iterable[object], max_items: int = 25) -> str:
    unique = []
    seen = set()
    for value in values:
        if pd.isna(value):
            continue
        text = str(value)
        if text and text not in seen:
            unique.append(text)
            seen.add(text)
        if len(unique) >= max_items:
            break
    return ";".join(unique)


def _first_non_empty(values: Iterable[object]) -> str:
    for value in values:
        if pd.notna(value) and str(value).strip():
            return str(value)
    return ""


def _count_negative_invalid(frame: pd.DataFrame, columns: Sequence[str]) -> int:
    total = 0
    for column in columns:
        if column in frame.columns:
            total += int((pd.to_numeric(frame[column], errors="coerce") < 0).sum())
    return total


def source_file_audit(raw_monthly: pd.DataFrame) -> pd.DataFrame:
    if raw_monthly.empty:
        return pd.DataFrame()
    group_columns = ["source_zip", "source_csv", "source_url", "source_publication_month", "source_table_type"]
    for column in group_columns:
        if column not in raw_monthly.columns:
            raw_monthly[column] = ""
    rows = (
        raw_monthly.groupby(group_columns, dropna=False, as_index=False, observed=True)
        .agg(
            row_count=("month", "size"),
            publication_month=("month", "min"),
            distinct_months=("month", "nunique"),
        )
        .reset_index(drop=True)
    )
    rows["audit_section"] = "source_file_loaded"
    rows["issue_type"] = "source_file_loaded"
    rows["severity"] = "info"
    rows["metric"] = "row_count"
    rows["value"] = rows["row_count"]
    rows["details"] = "Rows parsed from the source publication file after table extraction."
    return rows


def duplicate_group_audit(
    raw_monthly: pd.DataFrame,
    value_columns: Sequence[str],
    key_columns: Sequence[str] = ("month", "trust_code", "specialty_code"),
) -> pd.DataFrame:
    require_columns(raw_monthly, key_columns, "raw RTT monthly data before duplicate audit")
    if raw_monthly.empty:
        return pd.DataFrame()
    working = raw_monthly.copy()
    working["month"] = month_start(working["month"])
    duplicated = working[working.duplicated(list(key_columns), keep=False)].copy()
    if duplicated.empty:
        return pd.DataFrame(
            [
                {
                    "audit_section": "duplicates",
                    "issue_type": "duplicate_trust_specialty_month_rows",
                    "severity": "info",
                    "metric": "duplicate_rows_before_aggregation",
                    "value": 0,
                    "details": "No duplicate trust-specialty-month rows found before aggregation.",
                }
            ]
        )

    rows: list[dict[str, Any]] = []
    for group_key, group in duplicated.groupby(list(key_columns), observed=True):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        key_payload = dict(zip(key_columns, group_key))
        conflicting_features = []
        for column in value_columns:
            if column not in group.columns:
                continue
            values = pd.to_numeric(group[column], errors="coerce").dropna().unique()
            if len(values) > 1:
                conflicting_features.append(column)
        rows.append(
            {
                "audit_section": "duplicates",
                "issue_type": "duplicate_trust_specialty_month_rows",
                "severity": "warning" if conflicting_features else "info",
                "metric": "duplicate_rows_before_aggregation",
                "value": int(len(group)),
                "month": key_payload.get("month"),
                COLUMNS.trust_code: key_payload.get(COLUMNS.trust_code),
                COLUMNS.specialty_code: key_payload.get(COLUMNS.specialty_code),
                "conflicting_value_columns": ";".join(conflicting_features),
                "source_files": _join_unique(group.get("source_csv", pd.Series(dtype=object))),
                "details": (
                    "Conflicting duplicate values logged before aggregation."
                    if conflicting_features
                    else "Duplicate rows had no conflicting non-missing values."
                ),
            }
        )
    return pd.DataFrame(rows)


def aggregate_monthly_records(
    raw_monthly: pd.DataFrame,
    value_columns: Sequence[str],
    availability_columns: Sequence[str],
) -> pd.DataFrame:
    key_columns = ["month", COLUMNS.trust_code, COLUMNS.specialty_code]
    require_columns(raw_monthly, key_columns, "raw RTT monthly data before aggregation")
    working = raw_monthly.copy()
    working["month"] = month_start(working["month"])
    for column in value_columns:
        if column not in working.columns:
            working[column] = np.nan
        working[column] = pd.to_numeric(working[column], errors="coerce")
    for column in availability_columns:
        if column not in working.columns:
            working[column] = 0
        working[column] = pd.to_numeric(working[column], errors="coerce").fillna(0).astype(int)

    aggregations: dict[str, Any] = {
        COLUMNS.trust_name: (COLUMNS.trust_name, _first_non_empty),
        COLUMNS.specialty_name: (COLUMNS.specialty_name, _first_non_empty),
        "source_trust_code": ("source_trust_code", _first_non_empty),
        "source_trust_name": ("source_trust_name", _first_non_empty),
        "source_specialty_code": ("source_specialty_code", _first_non_empty),
        "source_specialty_name": ("source_specialty_name", _first_non_empty),
        "trust_identifier_harmonisation_rule": ("trust_identifier_harmonisation_rule", _first_non_empty),
        "specialty_identifier_harmonisation_rule": ("specialty_identifier_harmonisation_rule", _first_non_empty),
        "source_file_count": ("source_csv", "nunique"),
        "source_row_count": ("source_csv", "size"),
        "source_zips": ("source_zip", _join_unique),
        "source_csvs": ("source_csv", _join_unique),
        "source_urls": ("source_url", _join_unique),
        "source_publication_months": ("source_publication_month", _join_unique),
    }
    for column in value_columns:
        aggregations[column] = (column, lambda values: values.max(skipna=True) if values.notna().any() else np.nan)
    for column in availability_columns:
        aggregations[column] = (column, "max")

    monthly = (
        working.groupby(key_columns, as_index=False, observed=True)
        .agg(**aggregations)
        .reset_index(drop=True)
    )
    monthly["observed_month"] = 1
    return monthly


def max_consecutive_true(values: Sequence[bool]) -> int:
    best = 0
    current = 0
    for value in values:
        if bool(value):
            current += 1
            best = max(best, current)
        else:
            current = 0
    return int(best)


def missingness_by_series_report(
    completed_before_filter: pd.DataFrame,
    min_series_length: int,
) -> pd.DataFrame:
    rows = []
    for series_id, group in completed_before_filter.groupby(COLUMNS.series_id, observed=True, sort=False):
        ordered = group.sort_values("month")
        missing = ordered["observed_month"].astype(int).eq(0)
        total_months = int(len(ordered))
        missing_months = int(missing.sum())
        rows.append(
            {
                COLUMNS.series_id: series_id,
                COLUMNS.trust_code: str(ordered[COLUMNS.trust_code].iloc[0]),
                COLUMNS.trust_name: str(ordered[COLUMNS.trust_name].iloc[0]),
                COLUMNS.specialty_code: str(ordered[COLUMNS.specialty_code].iloc[0]),
                COLUMNS.specialty_name: str(ordered[COLUMNS.specialty_name].iloc[0]),
                "first_month": pd.Timestamp(ordered["month"].min()).date().isoformat(),
                "last_month": pd.Timestamp(ordered["month"].max()).date().isoformat(),
                "total_months": total_months,
                "observed_months": int(total_months - missing_months),
                "missing_months": missing_months,
                "missing_pct": float(100.0 * missing_months / total_months) if total_months else 0.0,
                "max_consecutive_missing_months": max_consecutive_true(missing.tolist()),
                "excluded_for_insufficient_history": bool(total_months < int(min_series_length)),
                "discontinued_series": False,
            }
        )
    return pd.DataFrame(rows)


def trust_identifier_changes_report(raw_monthly: pd.DataFrame) -> pd.DataFrame:
    rows = []
    columns = [
        "change_type",
        COLUMNS.trust_code,
        COLUMNS.trust_name,
        COLUMNS.specialty_code,
        COLUMNS.specialty_name,
        "first_month",
        "last_month",
        "distinct_values",
        "details",
    ]
    if raw_monthly.empty:
        return pd.DataFrame(rows, columns=columns)
    working = raw_monthly.copy()
    working["month"] = month_start(working["month"])
    for trust_code, group in working.groupby(COLUMNS.trust_code, observed=True, sort=False):
        names = sorted({str(value) for value in group[COLUMNS.trust_name].dropna().unique() if str(value).strip()})
        if len(names) > 1:
            rows.append(
                {
                    "change_type": "trust_name_changes_for_code",
                    COLUMNS.trust_code: trust_code,
                    COLUMNS.trust_name: _join_unique(names),
                    "first_month": pd.Timestamp(group["month"].min()).date().isoformat(),
                    "last_month": pd.Timestamp(group["month"].max()).date().isoformat(),
                    "distinct_values": len(names),
                    "details": "Same Trust code appears with multiple Trust names. No automatic organisation merge applied.",
                }
            )
    for trust_name, group in working.groupby(COLUMNS.trust_name, observed=True, sort=False):
        codes = sorted({str(value) for value in group[COLUMNS.trust_code].dropna().unique() if str(value).strip()})
        if len(codes) > 1:
            rows.append(
                {
                    "change_type": "trust_code_changes_for_name",
                    COLUMNS.trust_code: _join_unique(codes),
                    COLUMNS.trust_name: trust_name,
                    "first_month": pd.Timestamp(group["month"].min()).date().isoformat(),
                    "last_month": pd.Timestamp(group["month"].max()).date().isoformat(),
                    "distinct_values": len(codes),
                    "details": "Same Trust name appears with multiple Trust codes. No automatic organisation merge applied.",
                }
            )
    for specialty_name, group in working.groupby(COLUMNS.specialty_name, observed=True, sort=False):
        codes = sorted({str(value) for value in group[COLUMNS.specialty_code].dropna().unique() if str(value).strip()})
        if len(codes) > 1:
            rows.append(
                {
                    "change_type": "specialty_code_changes_for_name",
                    COLUMNS.specialty_code: _join_unique(codes),
                    COLUMNS.specialty_name: specialty_name,
                    "first_month": pd.Timestamp(group["month"].min()).date().isoformat(),
                    "last_month": pd.Timestamp(group["month"].max()).date().isoformat(),
                    "distinct_values": len(codes),
                    "details": "Same specialty name appears with multiple treatment-function codes. No automatic specialty merge applied.",
                }
            )
    return pd.DataFrame(rows, columns=columns)


def large_jump_report(
    clean: pd.DataFrame,
    value_column: str = "incomplete_total",
    absolute_threshold: float = 5000.0,
    relative_threshold: float = 1.0,
) -> pd.DataFrame:
    rows = []
    if value_column not in clean.columns:
        return pd.DataFrame(rows)
    working = clean.sort_values([COLUMNS.series_id, "month"]).copy()
    working["previous_value"] = working.groupby(COLUMNS.series_id, observed=True)[value_column].shift(1)
    working["absolute_change"] = pd.to_numeric(working[value_column], errors="coerce") - pd.to_numeric(
        working["previous_value"], errors="coerce"
    )
    denominator = pd.to_numeric(working["previous_value"], errors="coerce").abs().replace(0, np.nan)
    working["relative_change"] = working["absolute_change"].abs() / denominator
    suspicious = working[
        working["absolute_change"].abs().ge(float(absolute_threshold))
        | working["relative_change"].ge(float(relative_threshold))
    ].copy()
    for row in suspicious.sort_values("absolute_change", key=lambda s: s.abs(), ascending=False).head(200).itertuples(index=False):
        rows.append(
            {
                "audit_section": "large_jumps",
                "issue_type": "unexpectedly_large_jump",
                "severity": "warning",
                "metric": value_column,
                "value": float(abs(getattr(row, "absolute_change"))),
                "month": getattr(row, "month"),
                COLUMNS.trust_code: getattr(row, COLUMNS.trust_code),
                COLUMNS.trust_name: getattr(row, COLUMNS.trust_name),
                COLUMNS.specialty_code: getattr(row, COLUMNS.specialty_code),
                COLUMNS.specialty_name: getattr(row, COLUMNS.specialty_name),
                "details": (
                    f"Month-to-month absolute change {float(getattr(row, 'absolute_change')):.2f}; "
                    f"relative change {float(getattr(row, 'relative_change')):.4f}."
                ),
            }
        )
    return pd.DataFrame(rows)


def data_quality_report(
    *,
    manifest: pd.DataFrame,
    raw_monthly: pd.DataFrame,
    monthly: pd.DataFrame,
    clean: pd.DataFrame,
    duplicate_audit: pd.DataFrame,
    missingness: pd.DataFrame,
    identifier_changes: pd.DataFrame,
    config: DataQualityConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rows.append(
        {
            "audit_section": "configuration",
            "issue_type": "date_filter",
            "severity": "info",
            "metric": "explicit_start_month",
            "value": str(config.start_month),
            "details": "Rows are explicitly filtered after source loading with month >= start_month.",
        }
    )
    rows.append(
        {
            "audit_section": "source_files",
            "issue_type": "source_files_loaded",
            "severity": "info",
            "metric": "source_file_count",
            "value": int(len(manifest)),
            "details": "Number of manifest ZIP publications downloaded or reused locally.",
        }
    )
    rows.append(
        {
            "audit_section": "row_counts",
            "issue_type": "row_count",
            "severity": "info",
            "metric": "raw_monthly_rows_after_date_filter",
            "value": int(len(raw_monthly)),
            "details": "Rows after source parsing, identifier harmonisation, and explicit date filter.",
        }
    )
    rows.append(
        {
            "audit_section": "row_counts",
            "issue_type": "row_count",
            "severity": "info",
            "metric": "monthly_rows_after_aggregation",
            "value": int(len(monthly)),
            "details": "Rows after trust-specialty-month aggregation with provenance retained.",
        }
    )
    rows.append(
        {
            "audit_section": "row_counts",
            "issue_type": "row_count",
            "severity": "info",
            "metric": "clean_rows_retained",
            "value": int(len(clean)),
            "details": "Rows retained in the processed modelling panel.",
        }
    )

    count_columns = [
        "waiting_list",
        "waiting_list_with_dta",
        "completed_admitted",
        "completed_non_admitted",
        "new_rtt_periods",
    ]
    rows.append(
        {
            "audit_section": "invalid_values",
            "issue_type": "negative_invalid_counts",
            "severity": "warning" if _count_negative_invalid(raw_monthly, count_columns) else "info",
            "metric": "negative_invalid_counts_before_cleaning",
            "value": _count_negative_invalid(raw_monthly, count_columns),
            "details": "Negative values in source count fields before non-negative cleaning.",
        }
    )
    missing_values = int(raw_monthly[count_columns].isna().sum().sum()) if set(count_columns).issubset(raw_monthly.columns) else 0
    rows.append(
        {
            "audit_section": "missing_values",
            "issue_type": "missing_values",
            "severity": "info",
            "metric": "source_count_missing_values",
            "value": missing_values,
            "details": "Missing values across primary source count fields.",
        }
    )

    if not duplicate_audit.empty:
        rows.extend(duplicate_audit.to_dict(orient="records"))
    if not identifier_changes.empty:
        for row in identifier_changes.to_dict(orient="records"):
            rows.append(
                {
                    "audit_section": "identifier_changes",
                    "issue_type": row.get("change_type"),
                    "severity": "warning",
                    "metric": "distinct_values",
                    "value": row.get("distinct_values"),
                    "details": row.get("details"),
                    **row,
                }
            )
    if not missingness.empty:
        rows.append(
            {
                "audit_section": "missing_months",
                "issue_type": "missing_months",
                "severity": "info",
                "metric": "total_missing_months",
                "value": int(missingness["missing_months"].sum()),
                "details": "Total reindexed missing months across Trust-specialty series.",
            }
        )
        rows.append(
            {
                "audit_section": "missing_months",
                "issue_type": "series_excluded_for_insufficient_history",
                "severity": "warning" if missingness["excluded_for_insufficient_history"].any() else "info",
                "metric": "series_excluded_for_insufficient_history",
                "value": int(missingness["excluded_for_insufficient_history"].sum()),
                "details": "Series excluded because completed monthly history is shorter than min_series_length.",
            }
        )

    comparable = clean[
        clean["incomplete_total"].notna()
        & clean["incomplete_decision_to_admit"].notna()
        & pd.to_numeric(clean.get("incomplete_decision_to_admit_source_available", 0), errors="coerce").fillna(0).astype(int).eq(1)
    ].copy()
    if not comparable.empty:
        exceeds = pd.to_numeric(comparable["incomplete_decision_to_admit"], errors="coerce") > pd.to_numeric(
            comparable["incomplete_total"], errors="coerce"
        )
        rows.append(
            {
                "audit_section": "part2a_consistency",
                "issue_type": "part2a_exceeds_total_incomplete",
                "severity": "warning" if exceeds.any() else "info",
                "metric": "part2a_exceeds_total_rate",
                "value": float(exceeds.mean()),
                "details": "Share of comparable rows where Part 2A exceeds total incomplete pathways.",
            }
        )

    jump_rows = large_jump_report(
        clean,
        value_column="incomplete_total",
        absolute_threshold=config.large_jump_absolute_threshold,
        relative_threshold=config.large_jump_relative_threshold,
    )
    if not jump_rows.empty:
        rows.extend(jump_rows.to_dict(orient="records"))

    for column in STOCK_FORWARD_FILL_COLUMNS + ACTIVITY_NO_FILL_COLUMNS:
        flag_column = f"{column}_imputed"
        if flag_column in clean.columns:
            rows.append(
                {
                    "audit_section": "imputation",
                    "issue_type": "variable_specific_missing_data_rule",
                    "severity": "info",
                    "metric": flag_column,
                    "value": int(pd.to_numeric(clean[flag_column], errors="coerce").fillna(0).sum()),
                    "details": (
                        "Stock variable forward-filled only for missing publication months."
                        if column in STOCK_FORWARD_FILL_COLUMNS
                        else "Activity flow variable not forward-filled; missing values retained with missingness indicators."
                    ),
                }
            )

    report = pd.DataFrame(rows)
    if not report.empty and "check" not in report.columns:
        report.insert(
            0,
            "check",
            (
                report.get("audit_section", "").astype(str)
                + "."
                + report.get("issue_type", "").astype(str)
                + "."
                + report.get("metric", "").astype(str)
            ).str.strip("."),
        )
    for column in ["month"]:
        if column in report.columns:
            report[column] = pd.to_datetime(report[column], errors="coerce").dt.date.astype(str)
    return report


def build_data_quality_summary_text(
    *,
    clean: pd.DataFrame,
    missingness: pd.DataFrame,
    report: pd.DataFrame,
    config: DataQualityConfig,
) -> str:
    trusts = int(clean[COLUMNS.trust_code].nunique()) if COLUMNS.trust_code in clean.columns else 0
    specialties = int(clean[COLUMNS.specialty_code].nunique()) if COLUMNS.specialty_code in clean.columns else 0
    months = int(clean["month"].nunique()) if "month" in clean.columns else 0
    series = int(clean[COLUMNS.series_id].nunique()) if COLUMNS.series_id in clean.columns else 0
    first_month = pd.Timestamp(clean["month"].min()).date().isoformat() if not clean.empty else "n/a"
    last_month = pd.Timestamp(clean["month"].max()).date().isoformat() if not clean.empty else "n/a"
    excluded = int(missingness["excluded_for_insufficient_history"].sum()) if not missingness.empty else 0
    missing_months = int(missingness["missing_months"].sum()) if not missingness.empty else 0
    warning_count = int(report["severity"].astype(str).eq("warning").sum()) if not report.empty and "severity" in report.columns else 0
    lines = [
        "# Data Quality Summary",
        "",
        f"Explicit date filter: `month >= {config.start_month}`.",
        f"Processed coverage: {first_month} to {last_month}.",
        f"Retained Trusts/providers: {trusts}.",
        f"Retained specialties: {specialties}.",
        f"Retained months: {months}.",
        f"Retained Trust-specialty series: {series}.",
        f"Series excluded for insufficient history: {excluded}.",
        f"Missing months inserted during continuity completion: {missing_months}.",
        f"Warning-level audit rows: {warning_count}.",
        "",
        "Identifier harmonisation trims codes and normalises names only. It does not automatically merge organisations.",
        "Stock waiting-list variables may be forward-filled only for missing publication months and are flagged. Activity flow variables are not forward-filled.",
        "Processed rows retain source publication provenance through source ZIP, CSV, URL, publication month, and source row-count audit columns.",
    ]
    return "\n".join(lines)


def assert_no_duplicate_monthly_rows(frame: pd.DataFrame) -> None:
    duplicates = frame.duplicated(["month", COLUMNS.trust_code, COLUMNS.specialty_code], keep=False)
    if duplicates.any():
        raise AssertionError(
            f"Found {int(duplicates.sum())} duplicate trust-specialty-month rows after aggregation."
        )


def assert_months_ordered(frame: pd.DataFrame) -> None:
    for series_id, group in frame.groupby(COLUMNS.series_id, observed=True, sort=False):
        months = pd.to_datetime(group["month"], errors="coerce")
        if months.isna().any() or not months.is_monotonic_increasing:
            raise AssertionError(f"Months are not ordered for series {series_id}.")


def assert_clean_data_quality_contract(
    clean: pd.DataFrame,
    model_features: Sequence[str],
    config: DataQualityConfig,
) -> None:
    require_columns(clean, ["month", COLUMNS.trust_code, COLUMNS.specialty_code, COLUMNS.series_id], "clean RTT data")
    assert_no_duplicate_monthly_rows(clean)
    assert_months_ordered(clean.sort_values([COLUMNS.series_id, "month"]))
    for column in ["waiting_list", "incomplete_total", "waiting_list_with_dta", "incomplete_decision_to_admit"]:
        if column in clean.columns and (pd.to_numeric(clean[column], errors="coerce").dropna() < 0).any():
            raise AssertionError(f"{column} contains negative values after cleaning.")
    missing_features = [column for column in model_features if column not in clean.columns]
    if missing_features:
        raise AssertionError(f"Clean data is missing required model features: {missing_features}")
    comparable = clean[
        clean["incomplete_total"].notna()
        & clean["incomplete_decision_to_admit"].notna()
        & pd.to_numeric(clean.get("incomplete_decision_to_admit_source_available", 0), errors="coerce").fillna(0).astype(int).eq(1)
    ]
    if not comparable.empty:
        exceeds = pd.to_numeric(comparable["incomplete_decision_to_admit"], errors="coerce") > pd.to_numeric(
            comparable["incomplete_total"], errors="coerce"
        )
        if float(exceeds.mean()) > float(config.part2a_exceeds_total_warning_rate):
            raise AssertionError(
                "Part 2A exceeds total incomplete pathways more often than the configured tolerance."
            )
