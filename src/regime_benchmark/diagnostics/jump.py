"""Jump / bipower-variation diagnostics — requirements.md §7.3.1 / design §10.

Computes:
  max_abs_d_j        = max(|d_t|) for t in S_j
  max_jump_share_j   = max(d_t^2) / sum(d_t^2)   (0 if sum == 0)
  BV_j               = (pi/2) * sum(|d_t| * |d_{t-1}|)  for t in (start+2..end)
  jump_component_j   = max(RV_j^2 - BV_j, 0)
  jump_share_bv_j    = jump_component_j / RV_j^2   (0 if RV_j == 0)

These values are post-hoc diagnostics only; they do not affect final_label.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl

from regime_benchmark.direction.segments import Segment


def compute_jump_diagnostics_segments(
    segments: list[Segment],
    d: np.ndarray,
) -> list[Segment]:
    """Add jump diagnostic fields to each Segment in-place.

    Args:
        segments: List of Segment objects.
        d: Full log-return array (float64; d[0] is NaN).

    Returns:
        The same list with jump diagnostic fields updated.
    """
    for seg in segments:
        start = seg.start_bar
        end = seg.end_bar
        # d slice: start+1 .. end inclusive (same as RV computation)
        d_slice = d[start + 1 : end + 1]
        d_clean = np.where(np.isnan(d_slice), 0.0, d_slice)

        abs_d = np.abs(d_clean)
        sq_d = d_clean**2
        sum_sq = float(np.sum(sq_d))

        # max_abs_d
        max_abs = float(np.max(abs_d)) if len(abs_d) > 0 else 0.0
        seg.max_abs_d = max_abs

        # max_jump_share
        max_sq = float(np.max(sq_d)) if len(sq_d) > 0 else 0.0
        seg.max_jump_share = max_sq / sum_sq if sum_sq > 0 else 0.0

        # Bipower variation: (pi/2) * sum(|d_t| * |d_{t-1}|) for t in (start+2..end)
        # i.e. consecutive pairs within d_slice
        pi_half = math.pi / 2.0
        if len(abs_d) >= 2:
            bv = pi_half * float(np.sum(abs_d[1:] * abs_d[:-1]))
        else:
            bv = 0.0
        seg.bipower_variation = bv

        # RV^2 for jump component
        rv_sq = seg.realized_volatility**2

        # jump_component = max(RV^2 - BV, 0)
        jump_comp = max(rv_sq - bv, 0.0)
        seg.jump_component = jump_comp

        # jump_share_bv = jump_component / RV^2  (0 if RV == 0)
        seg.jump_share_bv = jump_comp / rv_sq if rv_sq > 0 else 0.0

    return segments


def compute_jump_diagnostics(
    segments: pl.DataFrame,
    log_returns: pl.Series,
) -> pl.DataFrame:
    """Add jump and bipower-variation diagnostic columns to the segments DataFrame.

    Legacy polars API.

    Args:
        segments: Segment DataFrame with start_bar, end_bar,
                  realized_volatility columns.
        log_returns: Full d_t series (float64).

    Returns:
        segments DataFrame with columns added:
        max_abs_d, max_jump_share, bipower_variation,
        jump_component, jump_share_bv.
    """
    d_arr = np.array(log_returns.to_list(), dtype=np.float64)
    pi_half = math.pi / 2.0

    max_abs_d_col = []
    max_jump_share_col = []
    bv_col = []
    jump_comp_col = []
    jump_share_bv_col = []

    for row in segments.iter_rows(named=True):
        start = row["start_bar"]
        end = row["end_bar"]
        d_slice = d_arr[start + 1 : end + 1]
        d_clean = np.where(np.isnan(d_slice), 0.0, d_slice)
        abs_d = np.abs(d_clean)
        sq_d = d_clean**2
        sum_sq = float(np.sum(sq_d))

        max_abs = float(np.max(abs_d)) if len(abs_d) > 0 else 0.0
        max_sq = float(np.max(sq_d)) if len(sq_d) > 0 else 0.0
        max_jump_sh = max_sq / sum_sq if sum_sq > 0 else 0.0

        bv = pi_half * float(np.sum(abs_d[1:] * abs_d[:-1])) if len(abs_d) >= 2 else 0.0

        rv_sq = row["realized_volatility"] ** 2
        jump_comp = max(rv_sq - bv, 0.0)
        jump_share_bv = jump_comp / rv_sq if rv_sq > 0 else 0.0

        max_abs_d_col.append(max_abs)
        max_jump_share_col.append(max_jump_sh)
        bv_col.append(bv)
        jump_comp_col.append(jump_comp)
        jump_share_bv_col.append(jump_share_bv)

    return segments.with_columns(
        pl.Series("max_abs_d", max_abs_d_col, dtype=pl.Float64),
        pl.Series("max_jump_share", max_jump_share_col, dtype=pl.Float64),
        pl.Series("bipower_variation", bv_col, dtype=pl.Float64),
        pl.Series("jump_component", jump_comp_col, dtype=pl.Float64),
        pl.Series("jump_share_bv", jump_share_bv_col, dtype=pl.Float64),
    )
