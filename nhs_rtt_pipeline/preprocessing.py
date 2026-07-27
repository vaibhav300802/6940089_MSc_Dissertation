from __future__ import annotations

from typing import Any, Dict, Iterable, Sequence

import numpy as np
import pandas as pd


NON_NEGATIVE_OPERATIONAL_FEATURES = [
    "waiting_list",
    "incomplete_total",
    "closing_waiting_list",
    "opening_waiting_list",
    "completed_admitted",
    "completed_non_admitted",
    "waiting_list_with_dta",
    "incomplete_decision_to_admit",
    "new_rtt_periods",
    "completed_total",
]

FLOW_COMPONENT_FEATURES = [
    "new_rtt_periods",
    "completed_admitted",
    "completed_non_admitted",
]

NET_INFLOW_COMPONENT_FEATURES = FLOW_COMPONENT_FEATURES
FLOW_LAG_PERIODS = (1, 3, 6)
FLOW_LAG_BASE_FEATURES = [
    "new_rtt_periods",
    "completed_total",
    "reported_net_inflow",
    "unreported_removals",
]
FLOW_LAGGED_FEATURES = [
    f"{feature}_lag{lag}" for feature in FLOW_LAG_BASE_FEATURES for lag in FLOW_LAG_PERIODS
]
FLOW_MISSINGNESS_FEATURES = [
    "new_rtt_periods_missing",
    "completed_admitted_missing",
    "completed_non_admitted_missing",
    "completed_total_missing",
    "opening_waiting_list_missing",
    "closing_waiting_list_missing",
    "reported_net_inflow_missing",
    "unreported_removals_missing",
    "flow_components_missing",
    *[f"{feature}_lag{lag}_missing" for feature in FLOW_LAG_BASE_FEATURES for lag in FLOW_LAG_PERIODS],
]
SIGNED_OPERATIONAL_FEATURES = [
    "net_inflow",
    "reported_net_inflow",
    "unreported_removals",
    "reconciliation_error",
    "reconciliation_error_abs",
    *[f"reported_net_inflow_lag{lag}" for lag in FLOW_LAG_PERIODS],
    *[f"unreported_removals_lag{lag}" for lag in FLOW_LAG_PERIODS],
]
LOG1P_NON_NEGATIVE_FEATURES = [
    "waiting_list",
    "closing_waiting_list",
    "opening_waiting_list",
    "completed_admitted",
    "completed_non_admitted",
    "waiting_list_with_dta",
    "new_rtt_periods",
    "completed_total",
    *[f"new_rtt_periods_lag{lag}" for lag in FLOW_LAG_PERIODS],
    *[f"completed_total_lag{lag}" for lag in FLOW_LAG_PERIODS],
]

FEATURE_GROUPS: Dict[str, str] = {
    "waiting_list": "waiting-list history",
    "incomplete_total": "waiting-list history",
    "closing_waiting_list": "waiting-list history",
    "opening_waiting_list": "waiting-list history",
    "waiting_list_with_dta": "waiting-list history",
    "incomplete_decision_to_admit": "waiting-list history",
    "new_rtt_periods": "referral or clock-start pressure",
    "reported_net_inflow": "referral or clock-start pressure",
    "net_inflow": "referral or clock-start pressure",
    "completed_admitted": "completed-pathway throughput",
    "completed_non_admitted": "completed-pathway throughput",
    "completed_total": "completed-pathway throughput",
    "unreported_removals": "unreported removals",
    "is_imputed_month": "data availability",
    "missing_month": "data availability",
    "observed_month": "data availability",
    "time_idx": "calendar effects",
    "calendar_month": "calendar effects",
    "month_sin": "calendar effects",
    "month_cos": "calendar effects",
}

for _feature in [
    "waiting_list_imputed",
    "waiting_list_with_dta_imputed",
    "completed_admitted_imputed",
    "completed_non_admitted_imputed",
    "new_rtt_periods_imputed",
]:
    FEATURE_GROUPS[_feature] = "data availability"

for _lag in FLOW_LAG_PERIODS:
    FEATURE_GROUPS[f"new_rtt_periods_lag{_lag}"] = "referral or clock-start pressure"
    FEATURE_GROUPS[f"reported_net_inflow_lag{_lag}"] = "referral or clock-start pressure"
    FEATURE_GROUPS[f"completed_total_lag{_lag}"] = "completed-pathway throughput"
    FEATURE_GROUPS[f"unreported_removals_lag{_lag}"] = "unreported removals"

for _feature in FLOW_MISSINGNESS_FEATURES:
    if _feature.startswith("unreported_removals"):
        FEATURE_GROUPS[_feature] = "unreported removals"
    elif _feature.startswith("completed"):
        FEATURE_GROUPS[_feature] = "completed-pathway throughput"
    elif _feature.startswith("new_rtt") or _feature.startswith("reported_net"):
        FEATURE_GROUPS[_feature] = "referral or clock-start pressure"
    elif _feature.startswith("opening") or _feature.startswith("closing"):
        FEATURE_GROUPS[_feature] = "waiting-list history"
    else:
        FEATURE_GROUPS[_feature] = "data availability"

DATA_DICTIONARY: Dict[str, Dict[str, Any]] = {
    "opening_waiting_list": {
        "feature_group": "waiting-list history",
        "description": "Previous month's reported closing RTT waiting-list size within the same Trust-specialty series.",
        "signed": False,
        "missing_allowed": True,
    },
    "incomplete_total": {
        "feature_group": "waiting-list history",
        "description": "Reported total incomplete RTT pathways from Part 2.",
        "signed": False,
        "missing_allowed": True,
    },
    "incomplete_decision_to_admit": {
        "feature_group": "waiting-list history",
        "description": "Reported incomplete RTT pathways with a decision to admit from Part 2A.",
        "signed": False,
        "missing_allowed": True,
    },
    "closing_waiting_list": {
        "feature_group": "waiting-list history",
        "description": "Current month's reported RTT waiting-list size for the Trust-specialty series.",
        "signed": False,
        "missing_allowed": True,
    },
    "new_rtt_periods": {
        "feature_group": "referral or clock-start pressure",
        "description": "Reported new RTT periods or clock starts in the month.",
        "signed": False,
        "missing_allowed": True,
    },
    "completed_total": {
        "feature_group": "completed-pathway throughput",
        "description": "Reported admitted plus non-admitted completed RTT pathways in the month.",
        "signed": False,
        "missing_allowed": True,
    },
    "reported_net_inflow": {
        "feature_group": "referral or clock-start pressure",
        "description": "Reported new RTT periods minus reported completed RTT pathways for the month.",
        "signed": True,
        "missing_allowed": True,
    },
    "net_inflow": {
        "feature_group": "referral or clock-start pressure",
        "description": "Backward-compatible alias of reported_net_inflow.",
        "signed": True,
        "missing_allowed": True,
    },
    "unreported_removals": {
        "feature_group": "unreported removals",
        "description": "Residual accounting term from opening waiting list plus new RTT periods minus completed pathways minus closing waiting list.",
        "signed": True,
        "missing_allowed": True,
    },
    "missing_month": {
        "feature_group": "data availability",
        "description": "Indicator equal to 1 when a Trust-specialty month was inserted during continuity completion because no source publication row was present.",
        "signed": False,
        "missing_allowed": False,
    },
    "source_file_count": {
        "feature_group": "data provenance",
        "description": "Number of distinct source CSV files contributing to the processed Trust-specialty-month row.",
        "signed": False,
        "missing_allowed": False,
    },
    "source_row_count": {
        "feature_group": "data provenance",
        "description": "Number of parsed source rows contributing to the processed Trust-specialty-month row before aggregation.",
        "signed": False,
        "missing_allowed": False,
    },
    "reconciliation_error": {
        "feature_group": "data quality",
        "description": "Difference between the reconstructed closing waiting list and the reported closing waiting list after applying the residual accounting term.",
        "signed": True,
        "missing_allowed": True,
    },
}

for _feature in [
    "waiting_list_imputed",
    "waiting_list_with_dta_imputed",
    "completed_admitted_imputed",
    "completed_non_admitted_imputed",
    "new_rtt_periods_imputed",
]:
    DATA_DICTIONARY[_feature] = {
        "feature_group": "data availability",
        "description": f"Indicator equal to 1 when {_feature.removesuffix('_imputed')} was imputed by the variable-specific missing-data rule.",
        "signed": False,
        "missing_allowed": False,
    }

for _feature in FLOW_MISSINGNESS_FEATURES:
    DATA_DICTIONARY[_feature] = {
        "feature_group": FEATURE_GROUPS.get(_feature, "data availability"),
        "description": f"Indicator equal to 1 when {_feature.removesuffix('_missing')} is unavailable for that row.",
        "signed": False,
        "missing_allowed": False,
    }

for _feature in FLOW_LAGGED_FEATURES:
    base = _feature.rsplit("_lag", 1)[0]
    DATA_DICTIONARY[_feature] = {
        "feature_group": FEATURE_GROUPS.get(_feature, FEATURE_GROUPS.get(base, "data quality")),
        "description": f"Lagged reported accounting component: {base} from an earlier month in the same Trust-specialty series.",
        "signed": base in {"reported_net_inflow", "unreported_removals"},
        "missing_allowed": True,
    }


def _to_numeric_preserve_missing(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("float64")


def _clip_non_negative_preserve_missing(series: pd.Series) -> pd.Series:
    values = _to_numeric_preserve_missing(series)
    return values.mask(values < 0, 0.0)


def _series_sum_preserve_missing(frame: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    return frame[list(columns)].sum(axis=1, min_count=len(columns)).astype("float64")


def clean_rtt_operational_features(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.copy()
    for column in FLOW_COMPONENT_FEATURES + ["waiting_list", "waiting_list_with_dta"]:
        if column not in cleaned.columns:
            cleaned[column] = np.nan
        cleaned[column] = _clip_non_negative_preserve_missing(cleaned[column])

    cleaned["incomplete_total"] = cleaned["waiting_list"]
    cleaned["incomplete_decision_to_admit"] = cleaned["waiting_list_with_dta"]
    if "waiting_list_source_available" in cleaned.columns:
        cleaned["incomplete_total_source_available"] = pd.to_numeric(
            cleaned["waiting_list_source_available"],
            errors="coerce",
        ).fillna(0).astype(int)
    else:
        cleaned["incomplete_total_source_available"] = cleaned["incomplete_total"].notna().astype(int)
    if "waiting_list_with_dta_source_available" in cleaned.columns:
        cleaned["incomplete_decision_to_admit_source_available"] = pd.to_numeric(
            cleaned["waiting_list_with_dta_source_available"],
            errors="coerce",
        ).fillna(0).astype(int)
    else:
        cleaned["incomplete_decision_to_admit_source_available"] = cleaned["incomplete_decision_to_admit"].notna().astype(int)

    cleaned = add_waiting_list_flow_reconciliation(cleaned)
    assert_net_inflow_integrity(cleaned)
    assert_flow_reconciliation_integrity(cleaned)
    return cleaned


def add_waiting_list_flow_reconciliation(
    frame: pd.DataFrame,
    lag_periods: Iterable[int] = FLOW_LAG_PERIODS,
) -> pd.DataFrame:
    cleaned = frame.copy()
    required_series_columns = ["trust_code", "specialty_code", "month"]
    missing_required = [column for column in required_series_columns if column not in cleaned.columns]
    if missing_required:
        raise ValueError(f"Missing required reconciliation columns: {missing_required}")

    cleaned["month"] = pd.to_datetime(cleaned["month"])
    cleaned = cleaned.sort_values(["trust_code", "specialty_code", "month"]).reset_index(drop=True)

    if "waiting_list" not in cleaned.columns:
        cleaned["waiting_list"] = np.nan
    cleaned["waiting_list"] = _clip_non_negative_preserve_missing(cleaned["waiting_list"])
    cleaned["closing_waiting_list"] = cleaned["waiting_list"]
    if "waiting_list_source_available" in cleaned.columns:
        cleaned["closing_waiting_list_source_available"] = (
            pd.to_numeric(cleaned["waiting_list_source_available"], errors="coerce").fillna(0).astype(int)
        )
    else:
        cleaned["closing_waiting_list_source_available"] = cleaned["closing_waiting_list"].notna().astype(int)
    cleaned["opening_waiting_list"] = cleaned.groupby(["trust_code", "specialty_code"], sort=False, observed=True)[
        "closing_waiting_list"
    ].shift(1)
    cleaned["opening_waiting_list_source_available"] = cleaned.groupby(
        ["trust_code", "specialty_code"],
        sort=False,
        observed=True,
    )["closing_waiting_list_source_available"].shift(1)

    for column in ["new_rtt_periods", "completed_admitted", "completed_non_admitted", "waiting_list_with_dta"]:
        if column not in cleaned.columns:
            cleaned[column] = np.nan
        cleaned[column] = _clip_non_negative_preserve_missing(cleaned[column])
        availability_column = f"{column}_source_available"
        if availability_column not in cleaned.columns:
            cleaned[availability_column] = cleaned[column].notna().astype(int)
        cleaned[availability_column] = pd.to_numeric(cleaned[availability_column], errors="coerce").fillna(0).astype(int)

    cleaned["completed_total"] = _series_sum_preserve_missing(
        cleaned,
        ["completed_admitted", "completed_non_admitted"],
    )
    cleaned["completed_total_source_available"] = (
        cleaned["completed_admitted_source_available"].eq(1)
        & cleaned["completed_non_admitted_source_available"].eq(1)
        & cleaned["completed_total"].notna()
    ).astype(int)
    cleaned["reported_net_inflow"] = cleaned["new_rtt_periods"] - cleaned["completed_total"]
    cleaned["net_inflow"] = cleaned["reported_net_inflow"]

    component_availability = {
        "new_rtt_periods_missing": ("new_rtt_periods", "new_rtt_periods_source_available"),
        "completed_admitted_missing": ("completed_admitted", "completed_admitted_source_available"),
        "completed_non_admitted_missing": ("completed_non_admitted", "completed_non_admitted_source_available"),
        "completed_total_missing": ("completed_total", "completed_total_source_available"),
        "opening_waiting_list_missing": ("opening_waiting_list", "opening_waiting_list_source_available"),
        "closing_waiting_list_missing": ("closing_waiting_list", "closing_waiting_list_source_available"),
    }
    for indicator, (source_column, availability_column) in component_availability.items():
        cleaned[indicator] = (
            cleaned[source_column].isna()
            | pd.to_numeric(cleaned[availability_column], errors="coerce").fillna(0).astype(int).eq(0)
        ).astype("int8")
    cleaned["reported_net_inflow_missing"] = (
        cleaned["reported_net_inflow"].isna()
        | cleaned["new_rtt_periods_missing"].eq(1)
        | cleaned["completed_total_missing"].eq(1)
    ).astype("int8")

    reconciliation_components = [
        "opening_waiting_list",
        "new_rtt_periods",
        "completed_total",
        "closing_waiting_list",
    ]
    cleaned["flow_components_missing"] = (
        cleaned[reconciliation_components].isna().any(axis=1)
        | cleaned["opening_waiting_list_missing"].eq(1)
        | cleaned["new_rtt_periods_missing"].eq(1)
        | cleaned["completed_total_missing"].eq(1)
        | cleaned["closing_waiting_list_missing"].eq(1)
    ).astype("int8")
    reconciled_mask = cleaned["flow_components_missing"].eq(0)
    cleaned["unreported_removals"] = np.nan
    cleaned.loc[reconciled_mask, "unreported_removals"] = (
        cleaned.loc[reconciled_mask, "opening_waiting_list"]
        + cleaned.loc[reconciled_mask, "new_rtt_periods"]
        - cleaned.loc[reconciled_mask, "completed_total"]
        - cleaned.loc[reconciled_mask, "closing_waiting_list"]
    )
    cleaned["unreported_removals_missing"] = cleaned["unreported_removals"].isna().astype("int8")
    cleaned["reconciliation_error"] = (
        cleaned["opening_waiting_list"]
        + cleaned["new_rtt_periods"]
        - cleaned["completed_total"]
        - cleaned["unreported_removals"]
        - cleaned["closing_waiting_list"]
    )
    cleaned["reconciliation_error_abs"] = cleaned["reconciliation_error"].abs()

    lag_periods = tuple(int(lag) for lag in lag_periods)
    grouped = cleaned.groupby(["trust_code", "specialty_code"], sort=False, observed=True)
    for feature in FLOW_LAG_BASE_FEATURES:
        for lag in lag_periods:
            lag_column = f"{feature}_lag{lag}"
            missing_column = f"{lag_column}_missing"
            cleaned[lag_column] = grouped[feature].shift(lag)
            cleaned[missing_column] = cleaned[lag_column].isna().astype("int8")

    for indicator in FLOW_MISSINGNESS_FEATURES:
        if indicator not in cleaned.columns:
            cleaned[indicator] = 1
        cleaned[indicator] = pd.to_numeric(cleaned[indicator], errors="coerce").fillna(1).astype("int8")

    return cleaned


def assert_net_inflow_integrity(
    frame: pd.DataFrame,
    non_negative_columns: Sequence[str] = NON_NEGATIVE_OPERATIONAL_FEATURES,
) -> None:
    missing = [column for column in ["new_rtt_periods", "completed_total", "net_inflow"] if column not in frame.columns]
    if missing:
        raise AssertionError(f"Missing net inflow validation columns: {missing}")

    expected = pd.to_numeric(frame["new_rtt_periods"], errors="coerce") - pd.to_numeric(
        frame["completed_total"],
        errors="coerce",
    )
    actual = pd.to_numeric(frame["net_inflow"], errors="coerce")
    comparable = expected.notna() & actual.notna()
    if not np.allclose(
        actual[comparable].to_numpy(dtype=float),
        expected[comparable].to_numpy(dtype=float),
        rtol=0.0,
        atol=1.0e-6,
    ):
        raise AssertionError("net_inflow must equal new_rtt_periods - completed_total.")
    if expected.notna().any() and actual[expected.notna()].isna().any():
        raise AssertionError("net_inflow is missing where new_rtt_periods and completed_total are available.")

    for column in non_negative_columns:
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        non_missing = values.dropna()
        if (non_missing < 0).any():
            min_value = float(values.min())
            raise AssertionError(f"{column} must be non-negative, but minimum value is {min_value}.")

    negative_supported = expected < 0
    if negative_supported.any() and not (actual[negative_supported] < 0).all():
        raise AssertionError("Negative net_inflow values were not preserved when completions exceeded new RTT periods.")


def assert_flow_reconciliation_integrity(frame: pd.DataFrame, tolerance: float = 1.0e-6) -> None:
    required_columns = [
        "opening_waiting_list",
        "new_rtt_periods",
        "completed_total",
        "unreported_removals",
        "closing_waiting_list",
        "reconciliation_error",
    ]
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise AssertionError(f"Missing flow reconciliation columns: {missing}")

    components = frame[["opening_waiting_list", "new_rtt_periods", "completed_total", "closing_waiting_list"]]
    can_reconcile = components.notna().all(axis=1)
    if "flow_components_missing" in frame.columns:
        can_reconcile = can_reconcile & pd.to_numeric(
            frame["flow_components_missing"],
            errors="coerce",
        ).fillna(1).eq(0)
    removals = pd.to_numeric(frame["unreported_removals"], errors="coerce")
    if removals[can_reconcile].isna().any():
        raise AssertionError("unreported_removals is missing where all flow components are available.")

    reconstructed = (
        pd.to_numeric(frame["opening_waiting_list"], errors="coerce")
        + pd.to_numeric(frame["new_rtt_periods"], errors="coerce")
        - pd.to_numeric(frame["completed_total"], errors="coerce")
        - removals
    )
    closing = pd.to_numeric(frame["closing_waiting_list"], errors="coerce")
    error = reconstructed - closing
    comparable = can_reconcile & error.notna()
    if not np.allclose(error[comparable].to_numpy(dtype=float), 0.0, rtol=0.0, atol=tolerance):
        max_error = float(error[comparable].abs().max())
        raise AssertionError(f"RTT flow reconciliation error exceeds tolerance; max absolute error is {max_error}.")

    if (removals.dropna() < 0).any() and (pd.to_numeric(frame["unreported_removals"], errors="coerce").dropna() < 0).sum() == 0:
        raise AssertionError("Negative unreported_removals values were not preserved.")


def net_inflow_quality_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if "net_inflow" not in frame.columns:
        raise ValueError("net_inflow column is required for the data-quality summary.")
    values = pd.to_numeric(frame["net_inflow"], errors="coerce")
    non_missing = values.dropna()

    total = int(len(values))
    missing_count = int(values.isna().sum())
    negative_count = int((non_missing < 0).sum())
    zero_count = int((non_missing == 0).sum())
    positive_count = int((non_missing > 0).sum())

    def pct(count: int) -> float:
        return float(100.0 * count / total) if total else 0.0

    return pd.DataFrame(
        [
            {
                "min_net_inflow": float(non_missing.min()) if len(non_missing) else np.nan,
                "max_net_inflow": float(non_missing.max()) if len(non_missing) else np.nan,
                "missing_observations": missing_count,
                "missing_observations_pct": pct(missing_count),
                "negative_observations": negative_count,
                "negative_observations_pct": pct(negative_count),
                "zero_observations": zero_count,
                "zero_observations_pct": pct(zero_count),
                "positive_observations": positive_count,
                "positive_observations_pct": pct(positive_count),
                "total_observations": total,
            }
        ]
    )


def flow_reconciliation_quality_report(frame: pd.DataFrame, top_n: int = 25) -> Dict[str, Any]:
    required = [
        "trust_code",
        "trust_name",
        "specialty_code",
        "specialty_name",
        "month",
        "opening_waiting_list",
        "new_rtt_periods",
        "completed_total",
        "closing_waiting_list",
        "unreported_removals",
        "flow_components_missing",
        "reconciliation_error",
        "reconciliation_error_abs",
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Cannot build flow reconciliation report; missing columns: {missing}")

    working = frame.copy()
    working["month"] = pd.to_datetime(working["month"], errors="coerce")
    reconciled = working["flow_components_missing"].eq(0) & working["unreported_removals"].notna()
    unreconciled = ~reconciled
    removals = pd.to_numeric(working.loc[reconciled, "unreported_removals"], errors="coerce")
    errors = pd.to_numeric(working.loc[reconciled, "reconciliation_error_abs"], errors="coerce")

    distribution: Dict[str, float | int | None] = {"count": int(removals.notna().sum())}
    if removals.notna().any():
        quantiles = removals.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
        distribution.update(
            {
                "mean": float(removals.mean()),
                "std": float(removals.std(ddof=0)),
                "min": float(removals.min()),
                "p01": float(quantiles.loc[0.01]),
                "p05": float(quantiles.loc[0.05]),
                "p25": float(quantiles.loc[0.25]),
                "median": float(quantiles.loc[0.5]),
                "p75": float(quantiles.loc[0.75]),
                "p95": float(quantiles.loc[0.95]),
                "p99": float(quantiles.loc[0.99]),
                "max": float(removals.max()),
                "negative_observations": int((removals < 0).sum()),
                "zero_observations": int((removals == 0).sum()),
                "positive_observations": int((removals > 0).sum()),
            }
        )

    largest = (
        working.loc[
            reconciled,
            [
                "month",
                "trust_code",
                "trust_name",
                "specialty_code",
                "specialty_name",
                "opening_waiting_list",
                "new_rtt_periods",
                "completed_total",
                "closing_waiting_list",
                "unreported_removals",
                "reconciliation_error",
                "reconciliation_error_abs",
            ],
        ]
        .sort_values("reconciliation_error_abs", ascending=False)
        .head(top_n)
        .copy()
    )
    if not largest.empty:
        largest["month"] = largest["month"].dt.strftime("%Y-%m-%d")

    missing_reasons = {
        column: int(pd.to_numeric(working[column], errors="coerce").fillna(0).sum())
        for column in FLOW_MISSINGNESS_FEATURES
        if column in working.columns
    }
    total_rows = int(len(working))
    return {
        "summary": {
            "total_rows": total_rows,
            "rows_successfully_reconciled": int(reconciled.sum()),
            "rows_could_not_be_reconciled": int(unreconciled.sum()),
            "rows_successfully_reconciled_pct": float(100.0 * reconciled.sum() / total_rows) if total_rows else 0.0,
            "rows_could_not_be_reconciled_pct": float(100.0 * unreconciled.sum() / total_rows) if total_rows else 0.0,
            "max_absolute_reconciliation_error": float(errors.max()) if errors.notna().any() else None,
            "mean_absolute_reconciliation_error": float(errors.mean()) if errors.notna().any() else None,
        },
        "missingness_indicators": missing_reasons,
        "unreported_removals_distribution": distribution,
        "largest_absolute_reconciliation_discrepancies": largest.to_dict(orient="records"),
        "notes": [
            "unreported_removals is a residual accounting feature and may be negative.",
            "Rows with unavailable opening waiting list, closing waiting list, new RTT periods, or completed pathways are not reconciled.",
        ],
    }


def data_dictionary_frame(extra_features: Iterable[str] | None = None) -> pd.DataFrame:
    rows = []
    features = list(DATA_DICTIONARY)
    if extra_features is not None:
        for feature in extra_features:
            if feature not in features:
                features.append(feature)
    for feature in features:
        entry = DATA_DICTIONARY.get(
            feature,
            {
                "feature_group": FEATURE_GROUPS.get(feature, "data quality"),
                "description": "Model feature derived from the reported RTT monthly panel.",
                "signed": feature in SIGNED_OPERATIONAL_FEATURES,
                "missing_allowed": feature not in FLOW_MISSINGNESS_FEATURES,
            },
        )
        rows.append(
            {
                "feature": feature,
                "feature_group": entry["feature_group"],
                "description": entry["description"],
                "signed": bool(entry["signed"]),
                "missing_allowed": bool(entry["missing_allowed"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["feature_group", "feature"]).reset_index(drop=True)


def feature_group_for_column(column: str) -> str:
    raw = str(column).removesuffix("_model")
    if raw in FEATURE_GROUPS:
        return FEATURE_GROUPS[raw]
    for lag in FLOW_LAG_PERIODS:
        suffix = f"_lag{lag}"
        if raw.endswith(suffix):
            return FEATURE_GROUPS.get(raw[: -len(suffix)], "data quality")
        missing_suffix = f"_lag{lag}_missing"
        if raw.endswith(missing_suffix):
            return FEATURE_GROUPS.get(raw[: -len(missing_suffix)], "data availability")
    if raw.endswith("_missing"):
        return FEATURE_GROUPS.get(raw, "data availability")
    return "data quality"
