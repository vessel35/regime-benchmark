"""Tests for quality/checks.py — M5 deliverables.

All tests are network-independent (pure in-memory DataFrames).
"""

from __future__ import annotations

from datetime import datetime, timezone

import polars as pl
import pytest

from regime_benchmark.quality.checks import (
    KlineQualityReport,
    check_kline_quality,
    verify_5m_resampling,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_UTC = timezone.utc


def _dt(h: int = 0, m: int = 0, s: int = 0) -> datetime:
    """Create a 2024-01-01 UTC datetime."""
    return datetime(2024, 1, 1, h, m, s, tzinfo=_UTC)


def _make_kline_df(
    rows: list[tuple[datetime, float, float, float, float]],
    timeframe: str = "1m",
) -> pl.DataFrame:
    """Build a minimal Kline DataFrame.

    Each tuple is (open_time, open, high, low, close).
    close_time = open_time + interval - 1s.
    Other columns filled with dummy values.
    """
    interval_s = {"1m": 60, "5m": 300}[timeframe]
    data = {
        "open_time": [],
        "open": [],
        "high": [],
        "low": [],
        "close": [],
        "volume": [],
        "close_time": [],
        "quote_asset_volume": [],
        "number_of_trades": [],
        "taker_buy_base_volume": [],
        "taker_buy_quote_volume": [],
        "ignore": [],
    }
    from datetime import timedelta
    for ot, o, h, lo, c in rows:
        ct = ot + timedelta(seconds=interval_s - 1)
        data["open_time"].append(ot)
        data["open"].append(float(o))
        data["high"].append(float(h))
        data["low"].append(float(lo))
        data["close"].append(float(c))
        data["volume"].append(100.0)
        data["close_time"].append(ct)
        data["quote_asset_volume"].append(100.0 * c)
        data["number_of_trades"].append(10)
        data["taker_buy_base_volume"].append(50.0)
        data["taker_buy_quote_volume"].append(50.0 * c)
        data["ignore"].append("0")

    return pl.DataFrame(
        data,
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


def _clean_bars_1m(n: int = 5, start_minute: int = 0) -> pl.DataFrame:
    """n clean 1m bars starting at start_minute (2024-01-01 UTC)."""
    rows = []
    for i in range(n):
        ot = _dt(m=start_minute + i)
        o = 2000.0 + i
        h = o + 5.0
        lo = o - 5.0
        c = o + 1.0
        rows.append((ot, o, h, lo, c))
    return _make_kline_df(rows, "1m")


# ---------------------------------------------------------------------------
# check_kline_quality: clean frame
# ---------------------------------------------------------------------------


class TestCheckKlineQualityClean:
    def test_clean_frame_passes(self) -> None:
        df = _clean_bars_1m(10)
        result = check_kline_quality(df, "1m")
        assert isinstance(result, KlineQualityReport)
        assert result.n_rows == 10
        assert result.n_gaps == 0
        assert result.gaps == []

    def test_returns_report_type(self) -> None:
        df = _clean_bars_1m(3)
        result = check_kline_quality(df, "1m")
        assert isinstance(result, KlineQualityReport)

    def test_empty_frame_passes(self) -> None:
        df = _make_kline_df([], "1m")
        result = check_kline_quality(df, "1m")
        assert result.n_rows == 0
        assert result.n_gaps == 0

    def test_5m_clean_frame(self) -> None:
        rows = [
            (_dt(m=0), 2000.0, 2010.0, 1990.0, 2005.0),
            (_dt(m=5), 2005.0, 2015.0, 1995.0, 2010.0),
            (_dt(m=10), 2010.0, 2020.0, 2000.0, 2012.0),
        ]
        df = _make_kline_df(rows, "5m")
        result = check_kline_quality(df, "5m")
        assert result.n_rows == 3
        assert result.n_gaps == 0

    def test_unknown_timeframe_raises(self) -> None:
        df = _clean_bars_1m(3)
        with pytest.raises(ValueError, match="Unknown timeframe"):
            check_kline_quality(df, "3m")


# ---------------------------------------------------------------------------
# check_kline_quality: duplicate open_time
# ---------------------------------------------------------------------------


class TestCheckKlineQualityDuplicates:
    def test_duplicate_raises(self) -> None:
        t = _dt(m=0)
        rows = [
            (t, 2000.0, 2010.0, 1990.0, 2005.0),
            (t, 2001.0, 2011.0, 1991.0, 2006.0),  # duplicate open_time
        ]
        df = _make_kline_df(rows, "1m")
        with pytest.raises(ValueError, match="Duplicate open_time"):
            check_kline_quality(df, "1m")

    def test_three_duplicates_raises(self) -> None:
        t = _dt(h=1, m=0)
        rows = [(t, 2000.0, 2010.0, 1990.0, 2005.0)] * 3
        df = _make_kline_df(rows, "1m")
        with pytest.raises(ValueError, match="Duplicate open_time"):
            check_kline_quality(df, "1m")

    def test_no_false_positive_near_duplicates(self) -> None:
        """Different timestamps separated by exactly 1 interval = no duplicate."""
        rows = [
            (_dt(m=0), 2000.0, 2010.0, 1990.0, 2005.0),
            (_dt(m=1), 2005.0, 2015.0, 1995.0, 2010.0),
        ]
        df = _make_kline_df(rows, "1m")
        result = check_kline_quality(df, "1m")
        assert result.n_rows == 2


# ---------------------------------------------------------------------------
# check_kline_quality: OHLC violations
# ---------------------------------------------------------------------------


class TestCheckKlineQualityOHLC:
    def test_high_less_than_close_raises(self) -> None:
        rows = [
            (_dt(m=0), 2000.0, 1999.0, 1990.0, 2005.0),  # high < close
        ]
        df = _make_kline_df(rows, "1m")
        with pytest.raises(ValueError, match="OHLC"):
            check_kline_quality(df, "1m")

    def test_low_greater_than_open_raises(self) -> None:
        rows = [
            (_dt(m=0), 2000.0, 2010.0, 2003.0, 2005.0),  # low > open
        ]
        df = _make_kline_df(rows, "1m")
        with pytest.raises(ValueError, match="OHLC"):
            check_kline_quality(df, "1m")

    def test_high_less_than_low_raises(self) -> None:
        rows = [
            (_dt(m=0), 2000.0, 1990.0, 1995.0, 2000.0),  # high < low
        ]
        df = _make_kline_df(rows, "1m")
        with pytest.raises(ValueError, match="OHLC"):
            check_kline_quality(df, "1m")

    def test_high_exactly_equal_close_passes(self) -> None:
        """high == close is valid (boundary)."""
        rows = [
            (_dt(m=0), 2000.0, 2005.0, 1995.0, 2005.0),  # high == close
        ]
        df = _make_kline_df(rows, "1m")
        result = check_kline_quality(df, "1m")
        assert isinstance(result, KlineQualityReport)

    def test_mixed_clean_and_violation_raises(self) -> None:
        """Even one bad row among clean rows should raise."""
        rows = [
            (_dt(m=0), 2000.0, 2010.0, 1990.0, 2005.0),  # clean
            (_dt(m=1), 2005.0, 2004.0, 1990.0, 2005.0),  # high < close ✗
            (_dt(m=2), 2005.0, 2015.0, 1995.0, 2010.0),  # clean
        ]
        df = _make_kline_df(rows, "1m")
        with pytest.raises(ValueError, match="OHLC"):
            check_kline_quality(df, "1m")


# ---------------------------------------------------------------------------
# check_kline_quality: missing candle detection
# ---------------------------------------------------------------------------


class TestCheckKlineQualityGaps:
    def test_no_gap_no_report(self) -> None:
        df = _clean_bars_1m(5)
        result = check_kline_quality(df, "1m")
        assert result.n_gaps == 0
        assert result.gaps == []

    def test_one_missing_candle_detected(self) -> None:
        """Skip minute 2 → gap of 1 candle between minute 1 and minute 3."""
        rows = [
            (_dt(m=0), 2000.0, 2010.0, 1990.0, 2005.0),
            (_dt(m=1), 2005.0, 2015.0, 1995.0, 2010.0),
            # minute 2 is missing
            (_dt(m=3), 2010.0, 2020.0, 2000.0, 2015.0),
            (_dt(m=4), 2015.0, 2025.0, 2005.0, 2020.0),
        ]
        df = _make_kline_df(rows, "1m")
        result = check_kline_quality(df, "1m")
        assert result.n_gaps == 1
        assert len(result.gaps) == 1
        gap_start, gap_end, count = result.gaps[0]
        assert count == 1
        assert gap_start == _dt(m=2)
        assert gap_end == _dt(m=2)

    def test_multiple_missing_candles_in_one_gap(self) -> None:
        """Skip minutes 2,3,4 → gap of 3."""
        rows = [
            (_dt(m=0), 2000.0, 2010.0, 1990.0, 2005.0),
            (_dt(m=1), 2005.0, 2015.0, 1995.0, 2010.0),
            # minutes 2,3,4 missing
            (_dt(m=5), 2010.0, 2020.0, 2000.0, 2015.0),
        ]
        df = _make_kline_df(rows, "1m")
        result = check_kline_quality(df, "1m")
        assert result.n_gaps == 3
        assert len(result.gaps) == 1
        _, _, count = result.gaps[0]
        assert count == 3

    def test_two_separate_gaps(self) -> None:
        """Two non-adjacent gaps are reported separately."""
        rows = [
            (_dt(m=0), 2000.0, 2010.0, 1990.0, 2005.0),
            # minute 1 missing
            (_dt(m=2), 2005.0, 2015.0, 1995.0, 2010.0),
            # minute 3 missing
            (_dt(m=4), 2010.0, 2020.0, 2000.0, 2015.0),
        ]
        df = _make_kline_df(rows, "1m")
        result = check_kline_quality(df, "1m")
        assert result.n_gaps == 2
        assert len(result.gaps) == 2

    def test_gap_not_fatal(self) -> None:
        """Gaps return a report but do NOT raise — design §4.3."""
        rows = [
            (_dt(m=0), 2000.0, 2010.0, 1990.0, 2005.0),
            # 5 minutes gap
            (_dt(m=6), 2010.0, 2020.0, 2000.0, 2015.0),
        ]
        df = _make_kline_df(rows, "1m")
        # Should NOT raise
        result = check_kline_quality(df, "1m")
        assert result.n_gaps == 5

    def test_5m_gap_detection(self) -> None:
        """5m timeframe gap is 300s; skip one 5m bar."""
        rows = [
            (_dt(m=0), 2000.0, 2010.0, 1990.0, 2005.0),
            (_dt(m=5), 2005.0, 2015.0, 1995.0, 2010.0),
            # 5m bar at minute 10 missing
            (_dt(m=15), 2010.0, 2020.0, 2000.0, 2015.0),
        ]
        df = _make_kline_df(rows, "5m")
        result = check_kline_quality(df, "5m")
        assert result.n_gaps == 1

    def test_gap_start_and_end_bounds(self) -> None:
        """gap_start = first missing slot, gap_end = last missing slot."""
        rows = [
            (_dt(m=0), 2000.0, 2010.0, 1990.0, 2005.0),
            (_dt(m=4), 2010.0, 2020.0, 2000.0, 2015.0),  # 3 missing: 1,2,3
        ]
        df = _make_kline_df(rows, "1m")
        result = check_kline_quality(df, "1m")
        assert result.n_gaps == 3
        gap_start, gap_end, count = result.gaps[0]
        assert gap_start == _dt(m=1)
        assert gap_end == _dt(m=3)
        assert count == 3


# ---------------------------------------------------------------------------
# verify_5m_resampling
# ---------------------------------------------------------------------------


def _make_1m_for_5m(
    n_five_min_bars: int, start_minute: int = 0
) -> pl.DataFrame:
    """Create n_five_min_bars * 5 = N rows of 1m data forming clean 5m buckets."""
    rows = []
    for b in range(n_five_min_bars):
        base_open = 2000.0 + b * 10
        for i in range(5):
            minute = start_minute + b * 5 + i
            ot = _dt(m=minute)
            o = base_open + i
            h = o + 3.0
            lo = o - 3.0
            c = o + 1.0
            rows.append((ot, o, h, lo, c))
    return _make_kline_df(rows, "1m")


def _make_5m_from_1m(df_1m: pl.DataFrame) -> pl.DataFrame:
    """Produce the expected 5m Binance data by resampling the 1m frame."""
    df_bucketed = df_1m.with_columns(
        pl.col("open_time").dt.truncate("5m").alias("bucket_5m")
    )
    agg = df_bucketed.group_by("bucket_5m").agg(
        pl.col("open").first().alias("open"),
        pl.col("high").max().alias("high"),
        pl.col("low").min().alias("low"),
        pl.col("close").last().alias("close"),
        pl.col("volume").sum().alias("volume"),
    ).sort("bucket_5m").rename({"bucket_5m": "open_time"})

    # Add dummy columns for a proper kline df
    n = len(agg)
    from datetime import timedelta
    close_times = [t + timedelta(seconds=299) for t in agg["open_time"].to_list()]
    agg = agg.with_columns(
        pl.Series("close_time", close_times, dtype=pl.Datetime(time_unit="us", time_zone="UTC")),
        pl.Series("quote_asset_volume", [0.0] * n, dtype=pl.Float64),
        pl.Series("number_of_trades", [0] * n, dtype=pl.Int64),
        pl.Series("taker_buy_base_volume", [0.0] * n, dtype=pl.Float64),
        pl.Series("taker_buy_quote_volume", [0.0] * n, dtype=pl.Float64),
        pl.Series("ignore", ["0"] * n, dtype=pl.Utf8),
    )
    return agg


class TestVerify5mResampling:
    def test_perfect_match_returns_empty(self) -> None:
        """Consistent 1m↔5m pair → empty mismatch DataFrame."""
        df_1m = _make_1m_for_5m(4)
        df_5m = _make_5m_from_1m(df_1m)
        mismatches = verify_5m_resampling(df_1m, df_5m)
        assert len(mismatches) == 0
        # Schema should still have 4 columns
        assert "open_time_5m" in mismatches.columns
        assert "field" in mismatches.columns
        assert "value_resampled" in mismatches.columns
        assert "value_binance" in mismatches.columns

    def test_injected_discrepancy_is_detected(self) -> None:
        """Modify one 5m high value → that field should appear in mismatches."""
        df_1m = _make_1m_for_5m(4)
        df_5m = _make_5m_from_1m(df_1m)

        # Inject a wrong high value in bucket 1 (second bucket = minute 5)
        # Bucket at index 1: open_time = _dt(m=5)
        # Change its high by +100 to force mismatch
        highs = df_5m["high"].to_list()
        highs[1] += 100.0
        df_5m_bad = df_5m.with_columns(
            pl.Series("high", highs, dtype=pl.Float64)
        )
        mismatches = verify_5m_resampling(df_1m, df_5m_bad)
        assert len(mismatches) >= 1
        mismatch_fields = mismatches["field"].to_list()
        assert "high" in mismatch_fields

    def test_injected_low_discrepancy_detected(self) -> None:
        df_1m = _make_1m_for_5m(3)
        df_5m = _make_5m_from_1m(df_1m)
        lows = df_5m["low"].to_list()
        lows[0] -= 50.0  # Make 5m low 50 units lower than resampled
        df_5m_bad = df_5m.with_columns(
            pl.Series("low", lows, dtype=pl.Float64)
        )
        mismatches = verify_5m_resampling(df_1m, df_5m_bad)
        assert len(mismatches) >= 1
        assert "low" in mismatches["field"].to_list()

    def test_injected_volume_discrepancy_detected(self) -> None:
        df_1m = _make_1m_for_5m(3)
        df_5m = _make_5m_from_1m(df_1m)
        vols = df_5m["volume"].to_list()
        vols[2] += 999.9  # Large deviation
        df_5m_bad = df_5m.with_columns(
            pl.Series("volume", vols, dtype=pl.Float64)
        )
        mismatches = verify_5m_resampling(df_1m, df_5m_bad)
        assert len(mismatches) >= 1
        assert "volume" in mismatches["field"].to_list()

    def test_empty_1m_returns_empty_mismatch(self) -> None:
        df_1m = _make_kline_df([], "1m")
        df_5m = _make_1m_for_5m(2)
        mismatches = verify_5m_resampling(df_1m, df_5m)
        assert len(mismatches) == 0

    def test_empty_5m_returns_empty_mismatch(self) -> None:
        df_1m = _make_1m_for_5m(2)
        df_5m = _make_kline_df([], "5m")
        mismatches = verify_5m_resampling(df_1m, df_5m)
        assert len(mismatches) == 0

    def test_within_tolerance_no_mismatch(self) -> None:
        """Difference below 1e-6 tolerance should not be reported."""
        df_1m = _make_1m_for_5m(2)
        df_5m = _make_5m_from_1m(df_1m)
        opens = df_5m["open"].to_list()
        # Perturb by 1e-10 (well below 1e-6)
        opens[0] += 1e-10
        df_5m_tiny = df_5m.with_columns(
            pl.Series("open", opens, dtype=pl.Float64)
        )
        mismatches = verify_5m_resampling(df_1m, df_5m_tiny)
        assert len(mismatches) == 0

    def test_return_schema(self) -> None:
        """Return DataFrame always has the 4 expected columns."""
        df_1m = _make_1m_for_5m(2)
        df_5m = _make_5m_from_1m(df_1m)
        mismatches = verify_5m_resampling(df_1m, df_5m)
        assert set(mismatches.columns) == {
            "open_time_5m", "field", "value_resampled", "value_binance"
        }

    def test_multiple_fields_in_one_bucket_reported_separately(self) -> None:
        """Two bad fields in the same bucket appear as two rows."""
        df_1m = _make_1m_for_5m(3)
        df_5m = _make_5m_from_1m(df_1m)
        opens = df_5m["open"].to_list()
        highs = df_5m["high"].to_list()
        opens[0] += 500.0
        highs[0] += 500.0
        df_5m_bad = df_5m.with_columns(
            pl.Series("open", opens, dtype=pl.Float64),
            pl.Series("high", highs, dtype=pl.Float64),
        )
        mismatches = verify_5m_resampling(df_1m, df_5m_bad)
        # At least open and high mismatch
        fields = set(mismatches["field"].to_list())
        assert "open" in fields
        assert "high" in fields
