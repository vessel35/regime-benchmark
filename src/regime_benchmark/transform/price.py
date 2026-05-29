"""Price transformations — requirements.md §5.

Computes:
  P_t = (high_t + low_t + close_t) / 3     # hlc3
  p_t = ln(P_t)                              # log price
  d_t = p_t - p_{t-1}                        # log return (first bar d_t = NaN)

[RISK] hlc3 is a bar-summary value, not an executable fill price (§5.1).
All arithmetic uses float64 — this is post-hoc labeling, not order execution.
"""

from __future__ import annotations

import polars as pl


def compute_hlc3(df: pl.DataFrame) -> pl.DataFrame:
    """Add hlc3 column: (high + low + close) / 3.

    Args:
        df: Kline DataFrame with 'high', 'low', 'close' columns (float64).

    Returns:
        DataFrame with additional 'hlc3' column (float64).

    Raises:
        NotImplementedError: Implementation deferred to Milestone M2.
    """
    raise NotImplementedError("compute_hlc3 is implemented in Milestone M2")


def compute_log_price(df: pl.DataFrame) -> pl.DataFrame:
    """Add log-price column p_t = ln(hlc3_t).

    Args:
        df: DataFrame containing 'hlc3' column (float64, positive).

    Returns:
        DataFrame with additional 'log_price' column (float64).

    Raises:
        NotImplementedError: Implementation deferred to Milestone M2.
    """
    raise NotImplementedError("compute_log_price is implemented in Milestone M2")


def compute_log_returns(df: pl.DataFrame) -> pl.DataFrame:
    """Add log-return column d_t = p_t - p_{t-1}.

    First bar's d_t is NaN and is excluded from segment calculations.

    Args:
        df: DataFrame containing 'log_price' column (float64).

    Returns:
        DataFrame with additional 'log_return' column (float64, first value NaN).

    Raises:
        NotImplementedError: Implementation deferred to Milestone M2.
    """
    raise NotImplementedError("compute_log_returns is implemented in Milestone M2")
