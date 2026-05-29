"""Data quality checks — requirements.md §4.3 and §4.4 / design §6.2 and §6.3.

Validates duplicate open_time, OHLC relationships, missing candles (gap detection),
and optional 5m resampling cross-check.

Contract for check_kline_quality():
  - Raises ValueError on HARD violations: duplicate open_time, unsorted timestamps,
    OHLC relationship violations.
  - Returns a KlineQualityReport dataclass with gap information (gaps are identified
    but NOT fatal — design §4.3 "누락 구간은 라벨 계산 전 명시").

Contract for verify_5m_resampling():
  - Returns a DataFrame of mismatches (empty if consistent).
  - Does NOT raise; mismatch detection is the caller's responsibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import polars as pl

_TIMEFRAME_SECONDS: dict[str, int] = {"1m": 60, "5m": 300}

# Float comparison tolerance for resampling cross-check (§4.4)
_RESAMPLE_TOL: float = 1e-6


@dataclass
class KlineQualityReport:
    """Result returned by check_kline_quality().

    Attributes:
        n_rows: Total row count in the DataFrame.
        n_gaps: Number of missing candle slots (gaps in the expected grid).
        gaps: List of (gap_start, gap_end, missing_count) tuples where
            gap_start and gap_end are UTC datetimes bounding the gap.
        ohlc_violation_count: Number of rows that violate OHLC relationships.
            (These rows raise a ValueError before this field is populated.)
    """

    n_rows: int
    n_gaps: int
    gaps: list[tuple[datetime, datetime, int]] = field(default_factory=list)
    ohlc_violation_count: int = 0


def check_kline_quality(df: pl.DataFrame, timeframe: str) -> KlineQualityReport:
    """Run all §4.3 quality checks on a Kline DataFrame.

    Hard violations (raise ValueError immediately):
      - Duplicate open_time within the timeframe.
      - open_time not sorted ascending.
      - OHLC relationship: high < max(open, close), low > min(open, close),
        or high < low.

    Soft findings (reported, not fatal):
      - Missing candles: gaps in the expected evenly-spaced grid.
        Returned as KlineQualityReport.gaps (list of gap boundaries + count).

    Args:
        df: Kline DataFrame with standardised columns (open_time as UTC datetime).
        timeframe: '1m' or '5m' (determines expected candle interval).

    Returns:
        KlineQualityReport with gap information.

    Raises:
        ValueError: On hard violations (duplicates, unsorted, OHLC).
        ValueError: If timeframe is unknown.
    """
    if timeframe not in _TIMEFRAME_SECONDS:
        raise ValueError(
            f"Unknown timeframe {timeframe!r}; expected one of {list(_TIMEFRAME_SECONDS)}"
        )
    interval_s = _TIMEFRAME_SECONDS[timeframe]

    if len(df) == 0:
        return KlineQualityReport(n_rows=0, n_gaps=0)

    # --- 1. Duplicate open_time check ---
    n_total = len(df)
    n_unique = df.select(pl.col("open_time").n_unique()).item()
    if n_unique != n_total:
        raise ValueError(
            f"Duplicate open_time: {n_total} rows but {n_unique} unique timestamps"
        )

    # --- 2. Sorted ascending ---
    open_times = df["open_time"].to_list()
    for i in range(1, len(open_times)):
        if open_times[i] < open_times[i - 1]:
            raise ValueError(
                f"open_time not sorted ascending: index {i - 1} "
                f"({open_times[i - 1]}) > index {i} ({open_times[i]})"
            )

    # --- 3. OHLC relationship check ---
    violation_mask = (
        (pl.col("high") < pl.max_horizontal("open", "close"))
        | (pl.col("low") > pl.min_horizontal("open", "close"))
        | (pl.col("high") < pl.col("low"))
    )
    n_violations = df.filter(violation_mask).height
    if n_violations > 0:
        raise ValueError(
            f"OHLC relationship violated in {n_violations} row(s): "
            "expected high >= max(open,close), low <= min(open,close), high >= low"
        )

    # --- 4. Missing candle detection (gap identification, not fatal) ---
    interval_td = timedelta(seconds=interval_s)
    gaps: list[tuple[datetime, datetime, int]] = []
    total_gaps = 0

    for i in range(1, len(open_times)):
        t_prev = open_times[i - 1]
        t_curr = open_times[i]
        diff_s = (t_curr - t_prev).total_seconds()
        if diff_s > interval_s:
            n_missing = int(round(diff_s / interval_s)) - 1
            if n_missing > 0:
                # gap_start is the first missing slot (t_prev + 1 interval)
                gap_start = t_prev + interval_td
                # gap_end is the last missing slot (t_curr - 1 interval)
                gap_end = t_curr - interval_td
                gaps.append((gap_start, gap_end, n_missing))
                total_gaps += n_missing

    return KlineQualityReport(
        n_rows=n_total,
        n_gaps=total_gaps,
        gaps=gaps,
        ohlc_violation_count=0,
    )


def verify_5m_resampling(
    df_1m: pl.DataFrame,
    df_5m: pl.DataFrame,
) -> pl.DataFrame:
    """Resample 1m klines to 5m and compare against downloaded 5m klines.

    Resampling rules (design §6.3 / requirements §4.4):
      open_5m  = first 1m open in the 5-bar bucket
      high_5m  = max of 5 1m highs
      low_5m   = min of 5 1m lows
      close_5m = last 1m close in the 5-bar bucket
      volume_5m = sum of 5 1m volumes

    The 5-bar bucket is aligned to open_time of the 5m bar: each 1m bar belongs to
    the 5m bucket whose open_time equals floor(open_time_1m, 5min).

    Float comparison uses a tolerance of 1e-6 (relative or absolute).

    Args:
        df_1m: 1-minute Kline DataFrame sorted ascending by open_time.
        df_5m: 5-minute Kline DataFrame from Binance, sorted ascending.

    Returns:
        DataFrame with columns:
            open_time_5m, field, value_resampled, value_binance
        One row per (5m bucket, field) where the values differ beyond tolerance.
        Empty DataFrame (0 rows) means perfect match.
    """
    if len(df_1m) == 0 or len(df_5m) == 0:
        return pl.DataFrame(
            schema={
                "open_time_5m": pl.Datetime(time_unit="us", time_zone="UTC"),
                "field": pl.Utf8,
                "value_resampled": pl.Float64,
                "value_binance": pl.Float64,
            }
        )

    # Compute the 5m bucket key for each 1m bar by truncating to 5-minute boundary
    # SF2: ensure first()/last() reflect true bar order — sort before bucketing.
    df_1m_with_bucket = df_1m.sort("open_time").with_columns(
        pl.col("open_time")
        .dt.truncate("5m")
        .alias("bucket_5m")
    )

    # Resample: group by 5m bucket
    resampled = df_1m_with_bucket.group_by("bucket_5m").agg(
        pl.col("open").first().alias("open_r"),
        pl.col("high").max().alias("high_r"),
        pl.col("low").min().alias("low_r"),
        pl.col("close").last().alias("close_r"),
        pl.col("volume").sum().alias("volume_r"),
    ).sort("bucket_5m")

    # B3: FULL outer join — a 5m bucket present on one side but missing on the
    # other is a real discrepancy (partial month / misaligned series), NOT
    # something to silently drop as an inner join would.
    joined = resampled.join(
        df_5m.select(["open_time", "open", "high", "low", "close", "volume"]),
        left_on="bucket_5m",
        right_on="open_time",
        how="full",
        coalesce=True,
    )

    mismatches: list[dict[str, object]] = []
    fields = [
        ("open", "open_r"),
        ("high", "high_r"),
        ("low", "low_r"),
        ("close", "close_r"),
        ("volume", "volume_r"),
    ]

    for row in joined.iter_rows(named=True):
        ts = row["bucket_5m"]
        # B3: an unmatched bucket (one side NULL) is a coverage mismatch.
        if row.get("open_r") is None or row.get("open") is None:
            mismatches.append(
                {
                    "open_time_5m": ts,
                    "field": "MISSING_BUCKET",
                    "value_resampled": float("nan") if row.get("open_r") is None else 1.0,
                    "value_binance": float("nan") if row.get("open") is None else 1.0,
                }
            )
            continue
        for binance_col, resample_col in fields:
            v_r = float(row[resample_col])
            v_b = float(row[binance_col])
            # Absolute tolerance check
            if abs(v_r - v_b) > _RESAMPLE_TOL:
                mismatches.append(
                    {
                        "open_time_5m": ts,
                        "field": binance_col,
                        "value_resampled": v_r,
                        "value_binance": v_b,
                    }
                )

    if not mismatches:
        return pl.DataFrame(
            schema={
                "open_time_5m": pl.Datetime(time_unit="us", time_zone="UTC"),
                "field": pl.Utf8,
                "value_resampled": pl.Float64,
                "value_binance": pl.Float64,
            }
        )

    return pl.DataFrame(
        mismatches,
        schema={
            "open_time_5m": pl.Datetime(time_unit="us", time_zone="UTC"),
            "field": pl.Utf8,
            "value_resampled": pl.Float64,
            "value_binance": pl.Float64,
        },
    )
