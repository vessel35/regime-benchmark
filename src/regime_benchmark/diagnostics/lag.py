"""Lag / confirm-bar diagnostics — requirements.md §6.5.1 / design §10.

Computes:
  confirm_bar_j   = first bar after start_j where |p_t - p_start_j| >= theta_dc
  lag_bars_j      = confirm_bar_j - start_j
  lag_move_j      = |p_{confirm_bar_j} - p_{start_j}|
  capturable_amplitude_j = max(A_j - lag_move_j, 0)
  capturable_ratio_j     = capturable_amplitude_j / A_j  (0 if A_j == 0)

[RISK] All diagnostics are post-hoc only. Using as forward-looking features
causes lookahead leakage (design §10.3 S1-S2).
"""

from __future__ import annotations

import numpy as np
import polars as pl

from regime_benchmark.direction.segments import Segment


def compute_lag_diagnostics_segments(
    segments: list[Segment],
    log_prices: np.ndarray,
    theta_dc: float,
) -> list[Segment]:
    """Add lag diagnostic fields to each confirmed Segment in-place.

    For tail segments, all lag fields remain None.

    The confirm_bar used here is the DC engine's confirm_bar, which is already
    stored in Segment.confirm_bar.  lag_bars = confirm_bar - start_bar.

    Args:
        segments: List of Segment objects (from build_segments).
        log_prices: Full log-price array p_t (float64).
        theta_dc: DC threshold for this timeframe.

    Returns:
        The same list with lag fields updated on confirmed segments.
    """
    for seg in segments:
        if seg.is_tail_unconfirmed or seg.confirm_bar is None:
            seg.lag_bars = None
            seg.lag_move = None
            seg.capturable_amplitude = None
            seg.capturable_ratio = None
        else:
            start = seg.start_bar
            confirm = seg.confirm_bar
            p_start = log_prices[start]
            p_confirm = log_prices[confirm]

            lag_b = confirm - start
            lag_m = abs(p_confirm - p_start)
            cap_amp = max(seg.amplitude - lag_m, 0.0)
            cap_ratio = cap_amp / seg.amplitude if seg.amplitude > 0 else 0.0

            seg.lag_bars = lag_b
            seg.lag_move = lag_m
            seg.capturable_amplitude = cap_amp
            seg.capturable_ratio = cap_ratio
    return segments


def compute_lag_diagnostics(
    segments: pl.DataFrame,
    log_prices: pl.Series,
    theta_dc: float,
) -> pl.DataFrame:
    """Add lag diagnostic columns to the segments DataFrame.

    Legacy polars API.

    Args:
        segments: Segment DataFrame with start_bar, end_bar, confirm_bar,
            amplitude, is_tail_unconfirmed columns.
        log_prices: Full log-price series p_t (float64).
        theta_dc: Directional Change threshold used for this timeframe.

    Returns:
        segments DataFrame with columns added:
        lag_bars, lag_move, capturable_amplitude, capturable_ratio.
    """
    p_arr = np.array(log_prices.to_list(), dtype=np.float64)

    lag_bars_col: list[int | None] = []
    lag_move_col: list[float | None] = []
    cap_amp_col: list[float | None] = []
    cap_ratio_col: list[float | None] = []

    for row in segments.iter_rows(named=True):
        is_tail = row.get("is_tail_unconfirmed", False)
        confirm = row.get("confirm_bar")
        if is_tail or confirm is None:
            lag_bars_col.append(None)
            lag_move_col.append(None)
            cap_amp_col.append(None)
            cap_ratio_col.append(None)
        else:
            start = row["start_bar"]
            p_start = p_arr[start]
            p_confirm = p_arr[int(confirm)]
            lag_b = int(confirm) - start
            lag_m = abs(p_confirm - p_start)
            amplitude = row["amplitude"]
            cap_amp = max(amplitude - lag_m, 0.0)
            cap_ratio = cap_amp / amplitude if amplitude > 0 else 0.0
            lag_bars_col.append(lag_b)
            lag_move_col.append(lag_m)
            cap_amp_col.append(cap_amp)
            cap_ratio_col.append(cap_ratio)

    return segments.with_columns(
        pl.Series("lag_bars", lag_bars_col, dtype=pl.Int64),
        pl.Series("lag_move", lag_move_col, dtype=pl.Float64),
        pl.Series("capturable_amplitude", cap_amp_col, dtype=pl.Float64),
        pl.Series("capturable_ratio", cap_ratio_col, dtype=pl.Float64),
    )
