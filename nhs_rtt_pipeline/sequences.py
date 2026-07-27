from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd


@dataclass(frozen=True)
class SequenceWindow:
    series_key: tuple[object, ...]
    encoder_start: int
    encoder_end: int
    target_start: int
    target_end: int


def make_sliding_window_indices(
    frame: pd.DataFrame,
    group_columns: Sequence[str],
    time_column: str,
    encoder_length: int,
    prediction_length: int,
) -> list[SequenceWindow]:
    if encoder_length <= 0:
        raise ValueError("encoder_length must be positive.")
    if prediction_length <= 0:
        raise ValueError("prediction_length must be positive.")
    missing = [column for column in [*group_columns, time_column] if column not in frame.columns]
    if missing:
        raise ValueError(f"Cannot build sequence windows; missing columns: {missing}")

    windows: list[SequenceWindow] = []
    sorted_frame = frame.sort_values([*group_columns, time_column]).reset_index(drop=True)
    for key, group in sorted_frame.groupby(list(group_columns), sort=False, observed=True):
        if not isinstance(key, tuple):
            key = (key,)
        positions = list(group.index)
        if len(positions) < encoder_length + prediction_length:
            continue
        times = pd.to_numeric(group[time_column], errors="raise").to_numpy()
        expected = list(range(int(times.min()), int(times.min()) + len(times)))
        if list(times.astype(int)) != expected:
            raise ValueError(f"Series {key} has non-contiguous or unordered time values.")
        for local_encoder_start in range(0, len(positions) - encoder_length - prediction_length + 1):
            local_encoder_end = local_encoder_start + encoder_length - 1
            local_target_start = local_encoder_end + 1
            local_target_end = local_target_start + prediction_length - 1
            windows.append(
                SequenceWindow(
                    series_key=key,
                    encoder_start=int(positions[local_encoder_start]),
                    encoder_end=int(positions[local_encoder_end]),
                    target_start=int(positions[local_target_start]),
                    target_end=int(positions[local_target_end]),
                )
            )
    return windows
