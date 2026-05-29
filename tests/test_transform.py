"""Tests for transform/price.py — hlc3/p/d correctness."""

from __future__ import annotations

import math

import polars as pl

from regime_benchmark.transform.price import add_price_columns


def _make_tiny_klines(
    opens: list[float],
    highs: list[float],
    lows: list[float],
    closes: list[float],
) -> pl.DataFrame:
    """Build a minimal kline DataFrame with the OHLC columns."""
    from datetime import datetime, timezone

    n = len(opens)
    open_times = [
        datetime(2024, 1, 1, 0, i, 0, tzinfo=timezone.utc) for i in range(n)
    ]
    return pl.DataFrame(
        {
            "open_time": open_times,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
        },
        schema={
            "open_time": pl.Datetime(time_unit="us", time_zone="UTC"),
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
        },
    )


def test_hlc3_formula() -> None:
    """hlc3 = (high + low + close) / 3 exactly."""
    df = _make_tiny_klines(
        opens=[100.0, 101.0, 102.0],
        highs=[104.0, 105.0, 106.0],
        lows=[98.0, 99.0, 100.0],
        closes=[103.0, 104.0, 105.0],
    )
    result = add_price_columns(df)
    assert "hlc3" in result.columns

    expected = [
        (104.0 + 98.0 + 103.0) / 3.0,
        (105.0 + 99.0 + 104.0) / 3.0,
        (106.0 + 100.0 + 105.0) / 3.0,
    ]
    for i, exp in enumerate(expected):
        assert abs(result["hlc3"][i] - exp) < 1e-12, (
            f"hlc3[{i}] = {result['hlc3'][i]}, expected {exp}"
        )


def test_log_price_is_ln_hlc3() -> None:
    """log_price = ln(hlc3) for each bar."""
    df = _make_tiny_klines(
        opens=[100.0, 200.0],
        highs=[110.0, 210.0],
        lows=[90.0, 190.0],
        closes=[105.0, 205.0],
    )
    result = add_price_columns(df)
    assert "log_price" in result.columns

    for i in range(len(result)):
        hlc3 = result["hlc3"][i]
        lp = result["log_price"][i]
        assert abs(lp - math.log(hlc3)) < 1e-10, (
            f"log_price[{i}] = {lp}, expected ln({hlc3}) = {math.log(hlc3)}"
        )


def test_log_return_first_row_is_null() -> None:
    """First row log_return must be null (no previous bar)."""
    df = _make_tiny_klines(
        opens=[100.0, 101.0, 102.0],
        highs=[104.0, 105.0, 106.0],
        lows=[98.0, 99.0, 100.0],
        closes=[103.0, 104.0, 105.0],
    )
    result = add_price_columns(df)
    assert "log_return" in result.columns
    assert result["log_return"][0] is None, (
        f"First log_return should be null, got {result['log_return'][0]}"
    )


def test_log_return_is_p_diff() -> None:
    """log_return[t] = log_price[t] - log_price[t-1] for t >= 1."""
    df = _make_tiny_klines(
        opens=[100.0, 101.0, 102.0, 103.0],
        highs=[104.0, 105.0, 106.0, 107.0],
        lows=[98.0, 99.0, 100.0, 101.0],
        closes=[103.0, 104.0, 105.0, 106.0],
    )
    result = add_price_columns(df)
    lp = result["log_price"].to_list()
    lr = result["log_return"].to_list()

    for t in range(1, len(lr)):
        expected = lp[t] - lp[t - 1]
        assert abs(lr[t] - expected) < 1e-12, (
            f"log_return[{t}] = {lr[t]}, expected {expected}"
        )


def test_all_columns_float64() -> None:
    """hlc3, log_price, log_return must be Float64."""
    df = _make_tiny_klines(
        opens=[100.0],
        highs=[104.0],
        lows=[98.0],
        closes=[103.0],
    )
    result = add_price_columns(df)
    assert result["hlc3"].dtype == pl.Float64
    assert result["log_price"].dtype == pl.Float64
    assert result["log_return"].dtype == pl.Float64


def test_single_bar_no_log_return() -> None:
    """Single-bar DataFrame produces one null log_return."""
    df = _make_tiny_klines(
        opens=[2000.0],
        highs=[2100.0],
        lows=[1950.0],
        closes=[2050.0],
    )
    result = add_price_columns(df)
    assert len(result) == 1
    assert result["log_return"][0] is None
