"""Price transformations — requirements.md §5 / design §7.

Computes:
  P_t = (high_t + low_t + close_t) / 3     # hlc3
  p_t = ln(P_t)                              # log price
  d_t = p_t - p_{t-1}                        # log return (first bar d_t = null/NaN)

[RISK] hlc3 is a bar-summary value, not an executable fill price (§5.1).
All arithmetic uses float64 — this is post-hoc labeling, not order execution.
"""

from __future__ import annotations

import polars as pl


def add_price_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Add hlc3, log_price (p), and log_return (d) columns.

    Columns added:
      - hlc3       : (high + low + close) / 3  (float64)
      - log_price  : ln(hlc3)                   (float64)
      - log_return : log_price - lag(log_price) (float64, first row = null)

    Args:
        df: Kline DataFrame with 'high', 'low', 'close' columns (float64).
            DataFrame should be sorted ascending by open_time before calling.

    Returns:
        DataFrame with hlc3, log_price, log_return columns appended.
        All new columns are float64.  The first row's log_return is null.
    """
    return df.with_columns(
        ((pl.col("high") + pl.col("low") + pl.col("close")) / 3.0)
        .cast(pl.Float64)
        .alias("hlc3"),
    ).with_columns(
        pl.col("hlc3").log(base=2.718281828459045).cast(pl.Float64).alias("log_price"),
    ).with_columns(
        (pl.col("log_price") - pl.col("log_price").shift(1))
        .cast(pl.Float64)
        .alias("log_return"),
    )


# ---------------------------------------------------------------------------
# Legacy single-step helpers (kept for compatibility with tests that call them
# individually; add_price_columns is the canonical M2 entrypoint).
# ---------------------------------------------------------------------------


def compute_hlc3(df: pl.DataFrame) -> pl.DataFrame:
    """Add hlc3 column: (high + low + close) / 3.

    Args:
        df: Kline DataFrame with 'high', 'low', 'close' columns (float64).

    Returns:
        DataFrame with additional 'hlc3' column (float64).
    """
    return df.with_columns(
        ((pl.col("high") + pl.col("low") + pl.col("close")) / 3.0)
        .cast(pl.Float64)
        .alias("hlc3"),
    )


def compute_log_price(df: pl.DataFrame) -> pl.DataFrame:
    """Add log-price column p_t = ln(hlc3_t).

    Args:
        df: DataFrame containing 'hlc3' column (float64, positive).

    Returns:
        DataFrame with additional 'log_price' column (float64).
    """
    return df.with_columns(
        pl.col("hlc3").log(base=2.718281828459045).cast(pl.Float64).alias("log_price"),
    )


def compute_log_returns(df: pl.DataFrame) -> pl.DataFrame:
    """Add log-return column d_t = p_t - p_{t-1}.

    First bar's d_t is null and is excluded from segment calculations.

    Args:
        df: DataFrame containing 'log_price' column (float64).

    Returns:
        DataFrame with additional 'log_return' column (float64, first value null).
    """
    return df.with_columns(
        (pl.col("log_price") - pl.col("log_price").shift(1))
        .cast(pl.Float64)
        .alias("log_return"),
    )
