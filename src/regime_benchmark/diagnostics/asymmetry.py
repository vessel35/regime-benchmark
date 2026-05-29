"""Asymmetric-volatility diagnostics — requirements.md §7.3.3 / design §10.

Computes:
  rv_plus_j           = sqrt(sum(d_t^2 * I(d_t > 0)))
  rv_minus_j          = sqrt(sum(d_t^2 * I(d_t < 0)))
  downside_vol_share_j = sum(d_t^2 * I(d_t < 0)) / sum(d_t^2)  (0 if sum == 0)

[BOUNDARY] Leverage-effect literature (Bekaert & Wu 2000) applies to equities.
Do NOT assume the same effect holds for ETHUSDT without separate validation (§7.3.3).
These are diagnostic values only; they do not affect final_label.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl

from regime_benchmark.direction.segments import Segment


def compute_asymmetry_diagnostics_segments(
    segments: list[Segment],
    d: np.ndarray,
) -> list[Segment]:
    """Add asymmetric-volatility diagnostic fields to each Segment in-place.

    Args:
        segments: List of Segment objects.
        d: Full log-return array (float64; d[0] is NaN).

    Returns:
        The same list with rv_plus, rv_minus, downside_vol_share updated.
    """
    for seg in segments:
        start = seg.start_bar
        end = seg.end_bar
        d_slice = d[start + 1 : end + 1]
        d_clean = np.where(np.isnan(d_slice), 0.0, d_slice)
        sq_d = d_clean**2
        sum_sq = float(np.sum(sq_d))

        sum_sq_pos = float(np.sum(sq_d * (d_clean > 0)))
        sum_sq_neg = float(np.sum(sq_d * (d_clean < 0)))

        seg.rv_plus = math.sqrt(sum_sq_pos)
        seg.rv_minus = math.sqrt(sum_sq_neg)
        seg.downside_vol_share = sum_sq_neg / sum_sq if sum_sq > 0 else 0.0

    return segments


def compute_asymmetry_diagnostics(
    segments: pl.DataFrame,
    log_returns: pl.Series,
) -> pl.DataFrame:
    """Add asymmetric-volatility diagnostic columns to the segments DataFrame.

    Legacy polars API.

    Args:
        segments: Segment DataFrame with start_bar, end_bar columns.
        log_returns: Full d_t series (float64).

    Returns:
        segments DataFrame with columns added:
        rv_plus, rv_minus, downside_vol_share.
    """
    d_arr = np.array(log_returns.to_list(), dtype=np.float64)

    rv_plus_col = []
    rv_minus_col = []
    downside_col = []

    for row in segments.iter_rows(named=True):
        start = row["start_bar"]
        end = row["end_bar"]
        d_slice = d_arr[start + 1 : end + 1]
        d_clean = np.where(np.isnan(d_slice), 0.0, d_slice)
        sq_d = d_clean**2
        sum_sq = float(np.sum(sq_d))

        sum_sq_pos = float(np.sum(sq_d * (d_clean > 0)))
        sum_sq_neg = float(np.sum(sq_d * (d_clean < 0)))

        rv_plus_col.append(math.sqrt(sum_sq_pos))
        rv_minus_col.append(math.sqrt(sum_sq_neg))
        downside_col.append(sum_sq_neg / sum_sq if sum_sq > 0 else 0.0)

    return segments.with_columns(
        pl.Series("rv_plus", rv_plus_col, dtype=pl.Float64),
        pl.Series("rv_minus", rv_minus_col, dtype=pl.Float64),
        pl.Series("downside_vol_share", downside_col, dtype=pl.Float64),
    )
