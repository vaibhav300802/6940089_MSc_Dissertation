from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pandas as pd


@dataclass(frozen=True)
class CovidShockSplitBoundaries:
    train_end: pd.Timestamp
    covid_start: pd.Timestamp
    covid_end: pd.Timestamp
    recovery_start: pd.Timestamp
    validation_start: pd.Timestamp
    core_train_end: pd.Timestamp


def compute_covid_shock_split_boundaries(
    train_end: str,
    covid_test_start: str,
    covid_test_end: str,
    recovery_start: str,
    validation_months: int,
    date_coverage: Mapping[str, object],
) -> CovidShockSplitBoundaries:
    """Compute and validate the pre-COVID/COVID-shock experiment split boundaries.

    Raises ``ValueError`` (rather than a bare ``assert``, which is silently
    stripped under ``python -O``) if the configured boundaries would leak
    COVID-period information into the pre-COVID training/validation split, or
    if the processed dataset does not actually cover the configured periods.
    """
    train_end_ts = pd.Timestamp(train_end)
    covid_start_ts = pd.Timestamp(covid_test_start)
    covid_end_ts = pd.Timestamp(covid_test_end)
    recovery_start_ts = pd.Timestamp(recovery_start)
    validation_start_ts = train_end_ts - pd.DateOffset(months=validation_months - 1)
    core_train_end_ts = validation_start_ts - pd.DateOffset(months=1)

    if not train_end_ts < covid_start_ts:
        raise ValueError("Pre-COVID training end must be earlier than COVID test start.")
    if pd.Timestamp(date_coverage["maximum_month"]) < covid_start_ts:
        raise ValueError("The processed dataset ends before the configured COVID shock test period starts.")
    if pd.Timestamp(date_coverage["minimum_month"]) > core_train_end_ts:
        raise ValueError("The processed dataset starts too late for the configured pre-COVID validation split.")

    return CovidShockSplitBoundaries(
        train_end=train_end_ts,
        covid_start=covid_start_ts,
        covid_end=covid_end_ts,
        recovery_start=recovery_start_ts,
        validation_start=validation_start_ts,
        core_train_end=core_train_end_ts,
    )
