"""Data quality checks — requirements.md §4.3 and §4.4.

Validates duplicate open_time, missing candles, OHLC relationships,
period coverage, and optional 5m resampling cross-check.
"""

from __future__ import annotations

import polars as pl


def check_kline_quality(df: pl.DataFrame, timeframe: str) -> None:
    """Run all §4.3 quality checks on a Kline DataFrame.

    Checks:
    - No duplicate open_time within timeframe.
    - No missing candles (identifies gaps vs expected grid).
    - OHLC relationship: high >= max(open, close), low <= min(open, close), high >= low.
    - Period coverage meets requirements.

    Args:
        df: Kline DataFrame with standardised columns.
        timeframe: Timeframe string ('1m' or '5m').

    Raises:
        ValueError: If any critical quality check fails.
        NotImplementedError: Implementation deferred to Milestone M2.
    """
    raise NotImplementedError("check_kline_quality is implemented in Milestone M2")


def verify_5m_resampling(df_1m: pl.DataFrame, df_5m: pl.DataFrame) -> pl.DataFrame:
    """Optional §4.4 cross-check: resample 1m to 5m and compare with Binance 5m.

    Args:
        df_1m: 1-minute Kline DataFrame.
        df_5m: 5-minute Kline DataFrame from Binance.

    Returns:
        DataFrame of discrepancies (empty if all match).

    Raises:
        NotImplementedError: Implementation deferred to Milestone M2.
    """
    raise NotImplementedError("verify_5m_resampling is implemented in Milestone M2")
