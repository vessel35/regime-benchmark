"""Directional Change state machine — requirements.md §6.2 / design.md §8.2.

Lookahead-safe turning-point detection:
- Bootstrap: mode=SEEK_UP, p_ext=p_0, t_ext=0 (first bar treated as provisional trough).
- Confirms turning points only from past extrema when subsequent price movement
  exceeds theta_dc. No future information is consumed.

Output columns on segment DataFrame:
  segment_id, start_bar, end_bar, confirm_bar, is_tail_unconfirmed

[RISK §7] p_t is derived from hlc3, which is not an executable fill price.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl


@dataclass
class TurningPoint:
    """A confirmed or provisional turning point from the DC state machine."""

    bar_index: int
    log_price: float
    point_type: str  # 'trough' or 'peak'
    confirm_bar: int  # bar index at which this TP was confirmed


def compute_theta_dc(abs_log_returns: pl.Series, q_dc: float, k_dc: float) -> float:
    """Compute Directional Change threshold: Quantile(|d_t|, q_dc) * k_dc.

    NOTE (M2): quantile is computed over the provided slice.
    Frozen-calibration split semantics (separate calibration window vs labeling
    window) are formalized in M5.  At that point callers should pre-compute theta_dc
    on the calibration split and pass the fixed value into run_dc_engine directly.

    Args:
        abs_log_returns: Series of |d_t| values (float64).  NaN/null entries
            are dropped before the quantile calculation.
        q_dc: Quantile level, e.g. 0.80.
        k_dc: Scale factor.

    Returns:
        theta_dc as float64.  Always positive.

    Raises:
        ValueError: If abs_log_returns is empty after dropping nulls, or if
            the resulting theta_dc is not positive.
    """
    clean = abs_log_returns.drop_nulls().drop_nans()
    if len(clean) == 0:
        raise ValueError("abs_log_returns is empty after dropping nulls/NaNs")
    q_result = clean.quantile(q_dc, interpolation="linear")
    if q_result is None:
        raise ValueError("quantile returned None (empty series)")
    q_val = float(q_result)
    theta = q_val * k_dc
    if theta <= 0:
        raise ValueError(
            f"compute_theta_dc produced non-positive theta={theta!r} "
            f"(q={q_dc}, k={k_dc}, quantile={q_val!r})"
        )
    return theta


def run_dc_engine(
    log_prices: np.ndarray,
    theta_dc: float,
) -> list[TurningPoint]:
    """Run the DC turning-point state machine on a log-price series.

    Bootstrap: mode=SEEK_UP, p_ext=p_0, t_ext=0.
    Last unconfirmed extremum is NOT emitted (becomes the tail segment).

    Design §8.2 state machine:

        bootstrap: mode=SEEK_UP, p_ext=p[0], t_ext=0
        for t in 1..N-1:
          SEEK_UP:
            if p[t] < p_ext:  update trough
            elif p[t] - p_ext >= theta_dc:
                emit TurningPoint(trough at t_ext, confirm=t)
                mode = SEEK_DOWN; p_ext, t_ext = p[t], t
          SEEK_DOWN:
            if p[t] > p_ext:  update peak
            elif p_ext - p[t] >= theta_dc:
                emit TurningPoint(peak at t_ext, confirm=t)
                mode = SEEK_UP; p_ext, t_ext = p[t], t

    The last in-progress extreme after the loop is not emitted.

    Args:
        log_prices: Array of log prices p_t (float64, length N >= 1).
        theta_dc: Directional Change threshold (strictly positive float64).

    Returns:
        List of TurningPoint in chronological order (peak/trough alternating).
        May be empty if no turning point is confirmed.

    Raises:
        ValueError: If log_prices is empty or theta_dc <= 0.
    """
    if len(log_prices) == 0:
        raise ValueError("log_prices must be non-empty")
    if theta_dc <= 0:
        raise ValueError(f"theta_dc must be positive, got {theta_dc!r}")

    p = np.asarray(log_prices, dtype=np.float64)
    n = len(p)

    turning_points: list[TurningPoint] = []

    # Bootstrap: treat bar 0 as provisional trough, seek upward move
    mode_seek_up = True  # True = SEEK_UP, False = SEEK_DOWN
    p_ext = p[0]
    t_ext = 0

    for t in range(1, n):
        pt = p[t]
        if mode_seek_up:
            if pt < p_ext:
                # Update the provisional trough
                p_ext = pt
                t_ext = t
            elif pt - p_ext >= theta_dc:
                # Confirm trough at t_ext, transition to SEEK_DOWN
                turning_points.append(
                    TurningPoint(
                        bar_index=t_ext,
                        log_price=p_ext,
                        point_type="trough",
                        confirm_bar=t,
                    )
                )
                mode_seek_up = False
                p_ext = pt
                t_ext = t
        else:  # SEEK_DOWN
            if pt > p_ext:
                # Update the provisional peak
                p_ext = pt
                t_ext = t
            elif p_ext - pt >= theta_dc:
                # Confirm peak at t_ext, transition to SEEK_UP
                turning_points.append(
                    TurningPoint(
                        bar_index=t_ext,
                        log_price=p_ext,
                        point_type="peak",
                        confirm_bar=t,
                    )
                )
                mode_seek_up = True
                p_ext = pt
                t_ext = t

    # The last in-progress extremum (t_ext, p_ext) is NOT emitted as confirmed.
    # It forms the start of the tail segment (is_tail_unconfirmed=True).

    return turning_points
