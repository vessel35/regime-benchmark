"""Lag / confirm-bar diagnostics — requirements.md §6.5.1.

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

import polars as pl


def compute_lag_diagnostics(
    segments: pl.DataFrame,
    log_prices: pl.Series,
    theta_dc: float,
) -> pl.DataFrame:
    """Add lag diagnostic columns to the segments DataFrame.

    Args:
        segments: Segment DataFrame with start_bar, end_bar, amplitude columns.
        log_prices: Full log-price series p_t (float64).
        theta_dc: Directional Change threshold used for this timeframe.

    Returns:
        segments DataFrame with columns added:
        confirm_bar, lag_bars, lag_move, capturable_amplitude, capturable_ratio.

    Raises:
        NotImplementedError: Implementation deferred to Milestone M5.
    """
    raise NotImplementedError("compute_lag_diagnostics is implemented in Milestone M5")
