"""Segment realized volatility computation — requirements.md §7.

RV_j        = sqrt(sum(d_t^2))   for t in (start_j+1 .. end_j)
RV_per_bar_j = RV_j / sqrt(N_j)

NOTE: denominator is sqrt(N_j), NOT sqrt(N_j-1). This matches spec §7.1 exactly.
Do NOT correct this to the sample-std denominator — unit tests pin this formula.

Volatility labels (timeframe-specific quantiles, never cross-timeframe):
  LOW_VOL  if RV_per_bar_j <= Q_low_tau
  MID_VOL  if Q_low_tau < RV_per_bar_j <= Q_high_tau
  HIGH_VOL if RV_per_bar_j > Q_high_tau
"""

from __future__ import annotations

import polars as pl


def compute_realized_volatility(
    segments: pl.DataFrame,
    log_returns: pl.Series,
) -> pl.DataFrame:
    """Add RV_j and RV_per_bar_j columns to the segments DataFrame.

    Args:
        segments: Segment DataFrame with start_bar, end_bar, N_j columns.
        log_returns: Full d_t series (float64); first value may be NaN.

    Returns:
        segments DataFrame with 'realized_volatility' (RV_j) and
        'realized_volatility_per_bar' (RV_per_bar_j) columns added.

    Raises:
        NotImplementedError: Implementation deferred to Milestone M4.
    """
    raise NotImplementedError("compute_realized_volatility is implemented in Milestone M4")


def compute_volatility_quantiles(
    rv_per_bar: pl.Series,
    q_low: float,
    q_high: float,
) -> tuple[float, float]:
    """Compute Q_low and Q_high threshold values from the RV_per_bar distribution.

    Must be called per timeframe; never mix 1m and 5m series (§7.2).

    Args:
        rv_per_bar: Series of RV_per_bar_j values for one timeframe (float64).
        q_low: Lower quantile level, e.g. 0.33.
        q_high: Upper quantile level, e.g. 0.66.

    Returns:
        Tuple (Q_low_tau, Q_high_tau) as float64 values.

    Raises:
        NotImplementedError: Implementation deferred to Milestone M4.
    """
    raise NotImplementedError(
        "compute_volatility_quantiles is implemented in Milestone M4"
    )


def assign_volatility_labels(
    segments: pl.DataFrame,
    q_low_value: float,
    q_high_value: float,
) -> pl.DataFrame:
    """Add volatility_label column ('LOW_VOL', 'MID_VOL', 'HIGH_VOL').

    Args:
        segments: Segment DataFrame with 'realized_volatility_per_bar' column.
        q_low_value: Computed Q_low threshold value (score boundary).
        q_high_value: Computed Q_high threshold value (score boundary).

    Returns:
        segments DataFrame with 'volatility_label' column added.

    Raises:
        NotImplementedError: Implementation deferred to Milestone M4.
    """
    raise NotImplementedError("assign_volatility_labels is implemented in Milestone M4")
