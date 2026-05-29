"""Asymmetric-volatility diagnostics — requirements.md §7.3.3.

Computes:
  rv_plus_j           = sqrt(sum(d_t^2 * I(d_t > 0)))
  rv_minus_j          = sqrt(sum(d_t^2 * I(d_t < 0)))
  downside_vol_share_j = sum(d_t^2 * I(d_t < 0)) / sum(d_t^2)  (0 if sum == 0)

[BOUNDARY] Leverage-effect literature (Bekaert & Wu 2000) applies to equities.
Do NOT assume the same effect holds for ETHUSDT without separate validation (§7.3.3).
These are diagnostic values only; they do not affect final_label.
"""

from __future__ import annotations

import polars as pl


def compute_asymmetry_diagnostics(
    segments: pl.DataFrame,
    log_returns: pl.Series,
) -> pl.DataFrame:
    """Add asymmetric-volatility diagnostic columns to the segments DataFrame.

    Args:
        segments: Segment DataFrame with start_bar, end_bar columns.
        log_returns: Full d_t series (float64).

    Returns:
        segments DataFrame with columns added:
        rv_plus, rv_minus, downside_vol_share.

    Raises:
        NotImplementedError: Implementation deferred to Milestone M5.
    """
    raise NotImplementedError(
        "compute_asymmetry_diagnostics is implemented in Milestone M5"
    )
