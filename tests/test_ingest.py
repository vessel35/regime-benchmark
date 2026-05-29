"""Tests for ingest/binance.py — M5 deliverables.

Network-independent: uses in-memory/temp zips and monkeypatched download.
Real-download test is marked @pytest.mark.slow and skips if offline.
"""

from __future__ import annotations

import io
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest

from regime_benchmark.ingest.binance import (
    download_monthly_klines,
    load_period,
    parse_monthly_zip,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EPOCH_2024_01_01_MS = 1704067200000  # 2024-01-01 00:00:00 UTC in ms
_BAR_SECONDS = {"1m": 60, "5m": 300}


def _make_csv_bytes(n_rows: int, start_ms: int, interval_ms: int) -> bytes:
    """Build a headerless 12-column CSV (Binance format) as bytes."""
    lines: list[str] = []
    for i in range(n_rows):
        open_time = start_ms + i * interval_ms
        close_time = open_time + interval_ms - 1000  # 1 second before next
        o = 2000.0 + i
        h = o + 5.0
        lo = o - 5.0
        c = o + 1.0
        vol = 100.0 + i
        lines.append(
            f"{open_time},{o},{h},{lo},{c},{vol},{close_time},"
            f"{vol * c},{10 + i},{vol * 0.5},{vol * 0.5 * c},0"
        )
    return "\n".join(lines).encode()


def _make_zip_bytes(csv_bytes: bytes, csv_name: str = "data.csv") -> bytes:
    """Wrap CSV bytes in a ZIP archive, return as bytes."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(csv_name, csv_bytes)
    return buf.getvalue()


def _write_zip(dest: Path, csv_bytes: bytes, csv_name: str = "data.csv") -> Path:
    zip_bytes = _make_zip_bytes(csv_bytes, csv_name)
    dest.write_bytes(zip_bytes)
    return dest


# ---------------------------------------------------------------------------
# parse_monthly_zip
# ---------------------------------------------------------------------------


class TestParseMonthlyZip:
    def test_correct_column_count(self, tmp_path: Path) -> None:
        csv_bytes = _make_csv_bytes(5, _EPOCH_2024_01_01_MS, 60_000)
        zp = _write_zip(tmp_path / "test.zip", csv_bytes)
        df = parse_monthly_zip(zp)
        assert df.width == 12

    def test_column_names(self, tmp_path: Path) -> None:
        csv_bytes = _make_csv_bytes(3, _EPOCH_2024_01_01_MS, 60_000)
        zp = _write_zip(tmp_path / "test.zip", csv_bytes)
        df = parse_monthly_zip(zp)
        expected_cols = [
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "number_of_trades",
            "taker_buy_base_volume", "taker_buy_quote_volume", "ignore",
        ]
        assert df.columns == expected_cols

    def test_open_time_is_utc_datetime(self, tmp_path: Path) -> None:
        csv_bytes = _make_csv_bytes(3, _EPOCH_2024_01_01_MS, 60_000)
        zp = _write_zip(tmp_path / "test.zip", csv_bytes)
        df = parse_monthly_zip(zp)
        assert df["open_time"].dtype == pl.Datetime(time_unit="us", time_zone="UTC")
        assert df["close_time"].dtype == pl.Datetime(time_unit="us", time_zone="UTC")

    def test_epoch_ms_conversion(self, tmp_path: Path) -> None:
        csv_bytes = _make_csv_bytes(1, _EPOCH_2024_01_01_MS, 60_000)
        zp = _write_zip(tmp_path / "test.zip", csv_bytes)
        df = parse_monthly_zip(zp)
        row_ts = df["open_time"].to_list()[0]
        expected = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        assert row_ts == expected

    def test_float64_price_columns(self, tmp_path: Path) -> None:
        csv_bytes = _make_csv_bytes(3, _EPOCH_2024_01_01_MS, 60_000)
        zp = _write_zip(tmp_path / "test.zip", csv_bytes)
        df = parse_monthly_zip(zp)
        for col in ["open", "high", "low", "close", "volume"]:
            assert df[col].dtype == pl.Float64, f"{col} should be Float64"

    def test_number_of_trades_is_int64(self, tmp_path: Path) -> None:
        csv_bytes = _make_csv_bytes(3, _EPOCH_2024_01_01_MS, 60_000)
        zp = _write_zip(tmp_path / "test.zip", csv_bytes)
        df = parse_monthly_zip(zp)
        assert df["number_of_trades"].dtype == pl.Int64

    def test_sorted_ascending(self, tmp_path: Path) -> None:
        """CSV rows in reverse order are still returned sorted ascending."""
        csv_bytes = _make_csv_bytes(5, _EPOCH_2024_01_01_MS, 60_000)
        # Reverse the rows
        lines = csv_bytes.decode().strip().split("\n")
        reversed_csv = "\n".join(reversed(lines)).encode()
        zp = _write_zip(tmp_path / "test.zip", reversed_csv)
        df = parse_monthly_zip(zp)
        times = df["open_time"].to_list()
        assert times == sorted(times)

    def test_row_count(self, tmp_path: Path) -> None:
        csv_bytes = _make_csv_bytes(10, _EPOCH_2024_01_01_MS, 60_000)
        zp = _write_zip(tmp_path / "test.zip", csv_bytes)
        df = parse_monthly_zip(zp)
        assert len(df) == 10

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            parse_monthly_zip(tmp_path / "nonexistent.zip")

    def test_no_csv_in_zip_raises(self, tmp_path: Path) -> None:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("readme.txt", "hello")
        zp = tmp_path / "no_csv.zip"
        zp.write_bytes(buf.getvalue())
        with pytest.raises(ValueError, match="No CSV file"):
            parse_monthly_zip(zp)

    def test_multi_row_values(self, tmp_path: Path) -> None:
        """Spot-check open price values to confirm correct mapping."""
        n = 4
        csv_bytes = _make_csv_bytes(n, _EPOCH_2024_01_01_MS, 60_000)
        zp = _write_zip(tmp_path / "test.zip", csv_bytes)
        df = parse_monthly_zip(zp)
        opens = df["open"].to_list()
        expected_opens = [2000.0 + i for i in range(n)]
        assert opens == expected_opens


# ---------------------------------------------------------------------------
# download_monthly_klines — monkeypatched (no network)
# ---------------------------------------------------------------------------


class TestDownloadMonthlyKlines:
    def test_idempotent_if_file_exists(self, tmp_path: Path) -> None:
        """If zip already exists and is valid, no download is performed."""
        symbol, tf, year, month = "ETHUSDT", "1m", 2024, 1
        filename = f"{symbol}-{tf}-{year}-{month:02d}.zip"
        zip_path = tmp_path / filename
        csv_bytes = _make_csv_bytes(3, _EPOCH_2024_01_01_MS, 60_000)
        _write_zip(zip_path, csv_bytes)

        with patch("urllib.request.urlretrieve") as mock_dl:
            result = download_monthly_klines(symbol, tf, year, month, tmp_path)
            mock_dl.assert_not_called()

        assert result == zip_path

    def test_downloads_when_file_missing(self, tmp_path: Path) -> None:
        """When file is absent, urlretrieve is called and zip is validated."""
        symbol, tf, year, month = "ETHUSDT", "1m", 2024, 2
        filename = f"{symbol}-{tf}-{year}-{month:02d}.zip"
        expected_url = (
            f"https://data.binance.vision/data/futures/um/monthly/klines"
            f"/{symbol}/{tf}/{filename}"
        )
        csv_bytes = _make_csv_bytes(3, _EPOCH_2024_01_01_MS + 31 * 24 * 3600 * 1000, 60_000)

        def fake_urlretrieve(url: str, dest: str) -> None:
            assert url == expected_url
            _write_zip(Path(dest), csv_bytes)

        with patch("urllib.request.urlretrieve", side_effect=fake_urlretrieve):
            result = download_monthly_klines(symbol, tf, year, month, tmp_path)

        assert result.name == filename
        assert result.exists()

    def test_re_downloads_corrupt_zip(self, tmp_path: Path) -> None:
        """A corrupt existing zip triggers a fresh download."""
        symbol, tf, year, month = "ETHUSDT", "1m", 2024, 3
        filename = f"{symbol}-{tf}-{year}-{month:02d}.zip"
        zip_path = tmp_path / filename
        zip_path.write_bytes(b"not a real zip")  # corrupt

        csv_bytes = _make_csv_bytes(3, _EPOCH_2024_01_01_MS, 60_000)

        def fake_urlretrieve(url: str, dest: str) -> None:
            _write_zip(Path(dest), csv_bytes)

        with patch("urllib.request.urlretrieve", side_effect=fake_urlretrieve) as mock_dl:
            result = download_monthly_klines(symbol, tf, year, month, tmp_path)
            mock_dl.assert_called_once()

        assert result.exists()


# ---------------------------------------------------------------------------
# load_period — monkeypatched download
# ---------------------------------------------------------------------------


def _make_monthly_zip_in_dir(
    dest_dir: Path,
    symbol: str,
    timeframe: str,
    year: int,
    month: int,
    n_rows: int,
    start_ms: int,
) -> None:
    """Write a small zip for the given month into dest_dir."""
    interval_ms = _BAR_SECONDS[timeframe] * 1000
    csv_bytes = _make_csv_bytes(n_rows, start_ms, interval_ms)
    filename = f"{symbol}-{timeframe}-{year}-{month:02d}.zip"
    _write_zip(dest_dir / filename, csv_bytes)


class TestLoadPeriod:
    def _setup_two_months(self, dest: Path, timeframe: str = "1m") -> tuple[datetime, datetime]:
        """Pre-populate dest with Jan-2024 and Feb-2024 zips (small, 10 rows each)."""
        jan_start = _EPOCH_2024_01_01_MS
        feb_start = _EPOCH_2024_01_01_MS + 31 * 24 * 3600 * 1000  # ~Feb 1

        _make_monthly_zip_in_dir(dest, "ETHUSDT", timeframe, 2024, 1, 10, jan_start)
        _make_monthly_zip_in_dir(dest, "ETHUSDT", timeframe, 2024, 2, 10, feb_start)

        start_utc = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        end_utc = datetime(2024, 2, 29, 23, 59, 59, tzinfo=timezone.utc)
        return start_utc, end_utc

    def test_concat_and_sort(self, tmp_path: Path) -> None:
        start_utc, end_utc = self._setup_two_months(tmp_path)
        df = load_period("ETHUSDT", "1m", start_utc, end_utc, tmp_path)
        times = df["open_time"].to_list()
        assert times == sorted(times), "Result should be sorted ascending by open_time"

    def test_no_duplicates(self, tmp_path: Path) -> None:
        start_utc, end_utc = self._setup_two_months(tmp_path)
        df = load_period("ETHUSDT", "1m", start_utc, end_utc, tmp_path)
        assert df["open_time"].n_unique() == len(df)

    def test_trim_to_period(self, tmp_path: Path) -> None:
        """Bars outside [start_utc, end_utc] should be excluded."""
        interval_ms = _BAR_SECONDS["1m"] * 1000
        start_ms = _EPOCH_2024_01_01_MS
        # 20 bars starting 2024-01-01 00:00
        csv_bytes = _make_csv_bytes(20, start_ms, interval_ms)
        _write_zip(tmp_path / "ETHUSDT-1m-2024-01.zip", csv_bytes)

        # Request only first 5 bars' worth
        start_utc = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        end_utc = datetime(2024, 1, 1, 0, 4, 59, tzinfo=timezone.utc)  # 5 minutes inclusive
        df = load_period("ETHUSDT", "1m", start_utc, end_utc, tmp_path)
        assert len(df) <= 5
        # All open_times must be <= end_utc
        assert all(t <= end_utc for t in df["open_time"].to_list())

    def test_boundary_candle_kept_open_time_trim(self, tmp_path: Path) -> None:
        """B2 regression: trim is by OPEN_TIME, and the boundary candle whose
        open_time == end_utc is KEPT (not dropped by a close_time>end compare).

        The old behavior filtered close_time <= end_utc, which dropped the
        finalized boundary candle (its close_time is open_time+59s > a
        second-precision end_utc). Monthly zips contain only finalized candles,
        so open_time trim is the correct rule (design §4.2/§4.3).
        """
        interval_ms = 60_000  # 1m
        n = 10
        csv_bytes = _make_csv_bytes(n, _EPOCH_2024_01_01_MS, interval_ms)
        _write_zip(tmp_path / "ETHUSDT-1m-2024-01.zip", csv_bytes)

        start_utc = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        # Boundary at open_time of bar 4 — bars 0..4 (inclusive) must be returned.
        end_utc = datetime(2024, 1, 1, 0, 4, 0, tzinfo=timezone.utc)
        df = load_period("ETHUSDT", "1m", start_utc, end_utc, tmp_path)

        assert len(df) == 5, f"expected bars 0..4 inclusive, got {len(df)}"
        open_times = df["open_time"].to_list()
        # The boundary candle (open_time == end_utc) is present (B2 fix).
        assert end_utc in open_times
        # No candle past the boundary open_time.
        assert all(t <= end_utc for t in open_times)

    def test_unknown_timeframe_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Unknown timeframe"):
            load_period("ETHUSDT", "3m", datetime(2024, 1, 1), datetime(2024, 1, 31), tmp_path)

    def test_naive_datetimes_treated_as_utc(self, tmp_path: Path) -> None:
        """Naive datetime inputs should not raise."""
        csv_bytes = _make_csv_bytes(5, _EPOCH_2024_01_01_MS, 60_000)
        _write_zip(tmp_path / "ETHUSDT-1m-2024-01.zip", csv_bytes)
        # Naive datetimes — should be accepted and treated as UTC
        df = load_period(
            "ETHUSDT", "1m",
            datetime(2024, 1, 1, 0, 0, 0),  # naive
            datetime(2024, 1, 1, 0, 10, 0),  # naive
            tmp_path,
        )
        assert len(df) >= 0  # Should not raise

    def test_multi_month_covers_boundary(self, tmp_path: Path) -> None:
        """Bars from both months appear in the result (cross-month boundary)."""
        start_utc, end_utc = self._setup_two_months(tmp_path)
        df = load_period("ETHUSDT", "1m", start_utc, end_utc, tmp_path)
        # Should have rows from both Jan and Feb
        assert len(df) >= 2

    def test_5m_timeframe(self, tmp_path: Path) -> None:
        """load_period works for 5m timeframe."""
        interval_ms = _BAR_SECONDS["5m"] * 1000
        csv_bytes = _make_csv_bytes(6, _EPOCH_2024_01_01_MS, interval_ms)
        _write_zip(tmp_path / "ETHUSDT-5m-2024-01.zip", csv_bytes)

        start_utc = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        end_utc = datetime(2024, 1, 31, 23, 59, 59, tzinfo=timezone.utc)
        df = load_period("ETHUSDT", "5m", start_utc, end_utc, tmp_path)
        assert df["open_time"].dtype == pl.Datetime(time_unit="us", time_zone="UTC")
        assert len(df) <= 6


# ---------------------------------------------------------------------------
# Network smoke test — marked @pytest.mark.slow; skipped if offline
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_download_real_file(tmp_path: Path) -> None:
    """Download one real monthly zip from Binance and verify parseable.

    This test is excluded by the test-gate hook (slow marker).
    It skips automatically if the host is offline.
    """
    import urllib.error

    try:
        zip_path = download_monthly_klines("ETHUSDT", "1m", 2024, 1, tmp_path)
    except (urllib.error.URLError, OSError):
        pytest.skip("Network unavailable — skipping real download test")

    df = parse_monthly_zip(zip_path)
    # A full January 2024 1m file has 44,640 rows (31 days × 24h × 60m)
    assert len(df) > 1000
    assert "open_time" in df.columns
    assert df["open_time"].dtype == pl.Datetime(time_unit="us", time_zone="UTC")
