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

    Args:
        abs_log_returns: Series of |d_t| values (float64).
        q_dc: Quantile level, e.g. 0.80.
        k_dc: Scale factor.

    Returns:
        theta_dc as float64.

    Raises:
        NotImplementedError: Implementation deferred to Milestone M3.
    """
    raise NotImplementedError("compute_theta_dc is implemented in Milestone M3")


def run_dc_engine(
    log_prices: np.ndarray,
    theta_dc: float,
) -> list[TurningPoint]:
    """Run the DC turning-point state machine on a log-price series.

    Bootstrap: mode=SEEK_UP, p_ext=p_0, t_ext=0.
    Last unconfirmed extremum is flagged as is_tail_unconfirmed=True.

    Args:
        log_prices: Array of log prices p_t (float64, length N).
        theta_dc: Directional Change threshold (positive float64).

    Returns:
        List of TurningPoint in chronological order (peak/trough alternating).

    Raises:
        NotImplementedError: Implementation deferred to Milestone M3.
    """
    raise NotImplementedError("run_dc_engine is implemented in Milestone M3")
