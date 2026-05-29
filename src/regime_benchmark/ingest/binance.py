"""Binance Kline ingestion: monthly zip download, CSV parsing, period trim.

Design §6.1 — source: https://data.binance.vision/data/futures/um/monthly/klines/ETHUSDT/{1m,5m}/

M2: local CSV loader + synthetic generator. Real download deferred to M5.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl

# Standard Binance Kline column names (requirements.md §4.2)
_KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "ignore",
]

_TIMEFRAME_SECONDS: dict[str, int] = {"1m": 60, "5m": 300}


def load_klines(
    source: str | Path,
    timeframe: Literal["1m", "5m"],
) -> pl.DataFrame:
    """Read a local CSV with the 12 standard Binance Kline columns.

    Parses open_time as UTC datetime, sorts ascending, validates no duplicates.

    Args:
        source: Path to the local CSV file (M2; real download deferred to M5).
        timeframe: '1m' or '5m' (used only for documentation / future checks).

    Returns:
        DataFrame with the 12 standard Kline columns.  open_time is
        Datetime(time_unit='us', time_zone='UTC'), float64 price columns.

    Raises:
        ValueError: If duplicate open_time values are detected.
        FileNotFoundError: If the source path does not exist.
    """
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Kline CSV not found: {path}")

    df = pl.read_csv(
        path,
        has_header=False,
        new_columns=_KLINE_COLUMNS,
        schema_overrides={
            "open_time": pl.Int64,
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Float64,
            "close_time": pl.Int64,
            "quote_asset_volume": pl.Float64,
            "number_of_trades": pl.Int64,
            "taker_buy_base_volume": pl.Float64,
            "taker_buy_quote_volume": pl.Float64,
            "ignore": pl.Utf8,
        },
    )

    # Convert millisecond epoch → UTC datetime
    df = df.with_columns(
        pl.from_epoch(pl.col("open_time"), time_unit="ms")
        .dt.replace_time_zone("UTC")
        .alias("open_time"),
        pl.from_epoch(pl.col("close_time"), time_unit="ms")
        .dt.replace_time_zone("UTC")
        .alias("close_time"),
    )

    # Sort ascending by open_time
    df = df.sort("open_time")

    # Validate no duplicates
    n_total = len(df)
    n_unique = df.select(pl.col("open_time").n_unique()).item()
    if n_unique != n_total:
        raise ValueError(
            f"Duplicate open_time detected in {path}: "
            f"{n_total} rows, {n_unique} unique timestamps"
        )

    return df


def make_synthetic_klines(
    timeframe: Literal["1m", "5m"],
    start_utc: datetime,
    periods: int,
    seed: int = 42,
) -> pl.DataFrame:
    """Generate a deterministic synthetic OHLCV DataFrame for testing.

    Uses a GBM-ish process (sinusoid trend + Gaussian noise) to produce
    price series that honour OHLC invariants:
      - high >= max(open, close)
      - low  <= min(open, close)
      - high >= low

    The open_time spacing matches the timeframe (60s for 1m, 300s for 5m).

    Args:
        timeframe: '1m' or '5m'.
        start_utc: UTC datetime for the first bar's open_time.  Must be
            timezone-aware.  If naive, treated as UTC.
        periods: Number of bars to generate.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with the 12 standard Kline columns.  open_time is
        UTC-aware.  No duplicate open_times.

    Raises:
        ValueError: If periods < 2 or timeframe is unknown.
    """
    if periods < 2:
        raise ValueError(f"periods must be >= 2, got {periods}")
    if timeframe not in _TIMEFRAME_SECONDS:
        raise ValueError(f"Unknown timeframe {timeframe!r}; expected '1m' or '5m'")

    rng = np.random.default_rng(seed)
    bar_secs = _TIMEFRAME_SECONDS[timeframe]

    # --- Price series via GBM-ish: sinusoid trend + normal noise
    base_price = 2000.0  # ETH-like price
    sigma = 0.0015       # per-bar log-return std
    t = np.arange(periods, dtype=np.float64)
    # Low-frequency sinusoidal drift to create segments
    drift = 0.10 * np.sin(2.0 * math.pi * t / (periods / 3))
    noise = rng.standard_normal(periods) * sigma
    log_returns = drift / periods + noise
    log_prices = np.cumsum(log_returns)
    close_prices = base_price * np.exp(log_prices)

    # Open = previous close (no gap)
    open_prices = np.empty(periods, dtype=np.float64)
    open_prices[0] = close_prices[0] * (1.0 + rng.standard_normal() * sigma * 0.5)
    open_prices[1:] = close_prices[:-1]

    # High/low: expand around max/min of open and close
    half_spread = np.abs(rng.standard_normal(periods)) * sigma * 0.8
    true_high = np.maximum(open_prices, close_prices) * (1.0 + half_spread)
    true_low = np.minimum(open_prices, close_prices) * (1.0 - half_spread)

    # Enforce OHLC invariants strictly
    high_prices = true_high
    low_prices = true_low
    # Ensure high >= max(open,close) and low <= min(open,close)
    high_prices = np.maximum(high_prices, np.maximum(open_prices, close_prices))
    low_prices = np.minimum(low_prices, np.minimum(open_prices, close_prices))
    # Ensure high >= low
    high_prices = np.maximum(high_prices, low_prices)

    volume = np.abs(rng.standard_normal(periods)) * 1000.0 + 500.0
    quote_asset_volume = volume * close_prices
    number_of_trades = (np.abs(rng.integers(50, 500, size=periods))).tolist()
    taker_buy_frac = rng.uniform(0.3, 0.7, periods)
    taker_buy_base = volume * taker_buy_frac
    taker_buy_quote = taker_buy_base * close_prices

    # Ensure start_utc is UTC-aware
    if start_utc.tzinfo is None:
        start_utc = start_utc.replace(tzinfo=timezone.utc)

    start_epoch_ms = int(start_utc.timestamp() * 1000)
    open_times_ms = [start_epoch_ms + i * bar_secs * 1000 for i in range(periods)]
    close_times_ms = [ts + (bar_secs - 1) * 1000 for ts in open_times_ms]

    open_times_dt = [
        datetime.fromtimestamp(ts / 1000, tz=timezone.utc) for ts in open_times_ms
    ]
    close_times_dt = [
        datetime.fromtimestamp(ts / 1000, tz=timezone.utc) for ts in close_times_ms
    ]

    df = pl.DataFrame(
        {
            "open_time": open_times_dt,
            "open": open_prices.tolist(),
            "high": high_prices.tolist(),
            "low": low_prices.tolist(),
            "close": close_prices.tolist(),
            "volume": volume.tolist(),
            "close_time": close_times_dt,
            "quote_asset_volume": quote_asset_volume.tolist(),
            "number_of_trades": number_of_trades,
            "taker_buy_base_volume": taker_buy_base.tolist(),
            "taker_buy_quote_volume": taker_buy_quote.tolist(),
            "ignore": ["0"] * periods,
        },
        schema={
            "open_time": pl.Datetime(time_unit="us", time_zone="UTC"),
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Float64,
            "close_time": pl.Datetime(time_unit="us", time_zone="UTC"),
            "quote_asset_volume": pl.Float64,
            "number_of_trades": pl.Int64,
            "taker_buy_base_volume": pl.Float64,
            "taker_buy_quote_volume": pl.Float64,
            "ignore": pl.Utf8,
        },
    )

    return df
