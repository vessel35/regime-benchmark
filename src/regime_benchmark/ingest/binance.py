"""Binance Kline ingestion: monthly zip download, CSV parsing, period trim.

Design §6.1 — source: https://data.binance.vision/data/futures/um/monthly/klines/ETHUSDT/{1m,5m}/

M2: local CSV loader + synthetic generator.
M5: real monthly-zip download (download_monthly_klines, parse_monthly_zip, load_period).
"""

from __future__ import annotations

import io
import math
import urllib.request
import zipfile
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

# Base URL for Binance public data (USDⓈ-M Futures monthly klines)
_BINANCE_BASE_URL = "https://data.binance.vision/data/futures/um/monthly/klines"


def download_monthly_klines(
    symbol: str,
    timeframe: str,
    year: int,
    month: int,
    dest_dir: Path,
) -> Path:
    """Download one month of Binance UM monthly klines zip to dest_dir.

    URL pattern (design §6.1):
      https://data.binance.vision/data/futures/um/monthly/klines/{symbol}/{timeframe}/
        {symbol}-{timeframe}-{year}-{month:02d}.zip

    Idempotent: skips the download if the file already exists and is a valid zip.
    Does NOT download a .CHECKSUM file (optional in design); instead verifies
    the downloaded file is a non-empty, openable zip.

    Uses only stdlib urllib — no new dependencies.

    Args:
        symbol: Binance symbol, e.g. 'ETHUSDT'.
        timeframe: Kline interval, e.g. '1m' or '5m'.
        year: 4-digit year.
        month: 1-based month number.
        dest_dir: Local directory to save the zip into (created if missing).

    Returns:
        Path to the local zip file.

    Raises:
        urllib.error.URLError: If the download fails.
        ValueError: If the downloaded file is empty or not a valid zip.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{symbol}-{timeframe}-{year}-{month:02d}.zip"
    zip_path = dest_dir / filename

    # Idempotent cache: skip only if the cached zip is FULLY valid.
    # SF1: testzip() validates member CRCs (a truncated download can keep an
    # intact central directory yet corrupt data) — namelist() alone is not enough.
    if zip_path.exists():
        try:
            with zipfile.ZipFile(zip_path) as zf:
                if len(zf.namelist()) > 0 and zf.testzip() is None:
                    return zip_path
        except zipfile.BadZipFile:
            pass  # Corrupt file — re-download below

    url = f"{_BINANCE_BASE_URL}/{symbol}/{timeframe}/{filename}"
    # SF1: download to a temp path and atomic-rename on success so a partial
    # download never poisons the cache under the final filename.
    tmp_path = zip_path.with_suffix(".zip.part")
    urllib.request.urlretrieve(url, tmp_path)  # noqa: S310

    if not tmp_path.exists() or tmp_path.stat().st_size == 0:
        raise ValueError(f"Downloaded file is empty: {url}")
    try:
        with zipfile.ZipFile(tmp_path) as zf:
            if len(zf.namelist()) == 0:
                raise ValueError(f"Downloaded zip has no entries: {url}")
            if zf.testzip() is not None:
                raise ValueError(f"Downloaded zip failed CRC check: {url}")
    except zipfile.BadZipFile as exc:
        tmp_path.unlink(missing_ok=True)
        raise ValueError(f"Downloaded file is not a valid zip: {url}") from exc

    tmp_path.replace(zip_path)  # atomic
    return zip_path


def parse_monthly_zip(zip_path: Path) -> pl.DataFrame:
    """Parse a Binance UM monthly klines zip into a standard Polars DataFrame.

    The CSV inside the zip has NO header row (Binance format). Columns are
    mapped in order to the 12 standard kline fields (requirements.md §4.2):
      open_time, open, high, low, close, volume, close_time,
      quote_asset_volume, number_of_trades, taker_buy_base_volume,
      taker_buy_quote_volume, ignore

    open_time and close_time are epoch-milliseconds → converted to UTC datetime.
    Numeric columns are float64; number_of_trades is Int64.

    Args:
        zip_path: Path to the downloaded monthly zip.

    Returns:
        DataFrame with the 12 standard Kline columns, sorted ascending by open_time.

    Raises:
        FileNotFoundError: If zip_path does not exist.
        ValueError: If the zip contains no CSV file or parsing fails.
    """
    zip_path = Path(zip_path)
    if not zip_path.exists():
        raise FileNotFoundError(f"Zip file not found: {zip_path}")

    with zipfile.ZipFile(zip_path) as zf:
        csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csv_names:
            raise ValueError(f"No CSV file found inside zip: {zip_path}")
        # Use the first (and typically only) CSV entry
        csv_bytes = zf.read(csv_names[0])

    # Strip a UTF-8 BOM if present (else the digit test below misfires and the
    # first real candle is silently dropped — review B1).
    csv_bytes = csv_bytes.lstrip(b"\xef\xbb\xbf")

    # Detect whether the CSV has a header row (some Binance zip files include one).
    # A data row's first field is the epoch-ms open_time (int-parseable); a header
    # row's first field is the text "open_time".
    first_field = csv_bytes.split(b"\n", 1)[0].split(b",", 1)[0].strip()
    try:
        int(first_field)
        skip_rows = 0
    except ValueError:
        skip_rows = 1

    df = pl.read_csv(
        io.BytesIO(csv_bytes),
        has_header=False,
        skip_rows=skip_rows,
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

    return df.sort("open_time")


def load_period(
    symbol: str,
    timeframe: str,
    start_utc: datetime,
    end_utc: datetime,
    dest_dir: Path,
) -> pl.DataFrame:
    """Download and assemble a contiguous kline DataFrame for [start_utc, end_utc].

    Iterates each calendar month that overlaps the requested range, downloads
    + parses its zip (idempotent cache), then:
      1. Concatenates all monthly DataFrames.
      2. Sorts ascending by open_time.
      3. Trims to candles whose open_time is in [start_utc, end_utc].
      4. Raises ValueError on duplicate open_time after sort.
      5. Excludes unfinished candles: any bar whose close_time > end_utc is
         dropped (design §4.3 — only finalized candles). For historical months
         this is purely a boundary trim at the end of the last month.

    Args:
        symbol: Binance symbol, e.g. 'ETHUSDT'.
        timeframe: '1m' or '5m'.
        start_utc: Period start (inclusive). If naive, treated as UTC.
        end_utc: Period end (inclusive, open_time ≤ end_utc). If naive, UTC.
        dest_dir: Local directory for cached zip files.

    Returns:
        Cleaned, trimmed, sorted DataFrame covering [start_utc, end_utc].

    Raises:
        ValueError: On duplicate open_time or unknown timeframe.
        urllib.error.URLError: If a required monthly zip cannot be downloaded.
    """
    if timeframe not in _TIMEFRAME_SECONDS:
        known = list(_TIMEFRAME_SECONDS)
        raise ValueError(f"Unknown timeframe {timeframe!r}; expected one of {known}")

    # Normalise to UTC-aware datetimes
    if start_utc.tzinfo is None:
        start_utc = start_utc.replace(tzinfo=timezone.utc)
    if end_utc.tzinfo is None:
        end_utc = end_utc.replace(tzinfo=timezone.utc)

    # Enumerate all (year, month) pairs covering [start_utc, end_utc]
    months: list[tuple[int, int]] = []
    y, m = start_utc.year, start_utc.month
    end_y, end_m = end_utc.year, end_utc.month
    while (y, m) <= (end_y, end_m):
        months.append((y, m))
        m += 1
        if m > 12:
            m = 1
            y += 1

    frames: list[pl.DataFrame] = []
    for year, month in months:
        zip_path = download_monthly_klines(symbol, timeframe, year, month, dest_dir)
        frames.append(parse_monthly_zip(zip_path))

    if not frames:
        # Empty period — return empty schema-correct DataFrame
        return pl.DataFrame(
            schema={c: pl.Utf8 for c in _KLINE_COLUMNS}
        ).cast(
            {
                "open_time": pl.Datetime(time_unit="us", time_zone="UTC"),
                "close_time": pl.Datetime(time_unit="us", time_zone="UTC"),
            }
        )

    df = pl.concat(frames).sort("open_time")

    # Trim by OPEN_TIME (the label timestamp, §4.2) to [start_utc, end_utc].
    # B2: do NOT filter on close_time — the 23:59 candle's close_time is
    # 23:59:59.999 which would (wrongly) exceed a second-precision end_utc and
    # drop a finalized candle. Monthly zips are published only AFTER the month
    # completes, so every candle in them is already finalized (§4.3); the
    # unfinished-candle concern applies only to the live REST path (not used here).
    df = df.filter(
        (pl.col("open_time") >= pl.lit(start_utc))
        & (pl.col("open_time") <= pl.lit(end_utc))
    )

    # Check for duplicates after sort+trim
    n_total = len(df)
    n_unique = df.select(pl.col("open_time").n_unique()).item()
    if n_unique != n_total:
        raise ValueError(
            f"Duplicate open_time after concat+trim for {symbol}/{timeframe}: "
            f"{n_total} rows, {n_unique} unique timestamps"
        )

    return df


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
