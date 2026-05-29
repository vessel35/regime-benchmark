"""Binance Kline ingestion: monthly zip download, CSV parsing, period trim.

Design §6.1 — source: https://data.binance.vision/data/futures/um/monthly/klines/ETHUSDT/{1m,5m}/
"""

from __future__ import annotations

import polars as pl


def load_klines(
    source_path: str,
    timeframe: str,
    start_utc: str,
    end_utc: str,
) -> pl.DataFrame:
    """Download or read Binance monthly Kline zips, parse, and trim to period.

    Args:
        source_path: Base path or URL prefix for monthly zip files.
        timeframe: Timeframe string, e.g. '1m' or '5m'.
        start_utc: Period start in 'YYYY-MM-DD HH:MM:SS' UTC format.
        end_utc: Period end in 'YYYY-MM-DD HH:MM:SS' UTC format.

    Returns:
        DataFrame with standardised Kline columns (open_time, open, high, low,
        close, volume, close_time, quote_asset_volume, number_of_trades,
        taker_buy_base_volume, taker_buy_quote_volume, ignore).
        Only confirmed candles within [start_utc, end_utc] are included.

    Raises:
        NotImplementedError: Implementation deferred to Milestone M2.
    """
    raise NotImplementedError("load_klines is implemented in Milestone M2")
