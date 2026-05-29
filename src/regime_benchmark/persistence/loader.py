"""Bulk loader for segment_labels and bar_labels — design §13.7.

M2 uses psycopg3 ``executemany`` for inserts. Real ``COPY ... FROM STDIN``
(10-100x faster for the ~1.26M 1m bars of a full-period run) is deferred to M5
where the full-scale load actually demands it; for the M2 thin slice executemany
is functionally correct and adequate.

**Atomicity (design §13.2 / review B1)**: register_run / register_params /
copy_segments / copy_bars do NOT commit individually. The whole run is one
transaction — only ``finalize_run`` commits. On any failure the caller MUST
``conn.rollback()``, which un-does register_run + params + segments + bars
atomically, leaving no orphan 'loading' run and no partial children.

Loading order (FK constraint): segment_labels first, then bar_labels.

Write target: dedicated 'regime_benchmark' DB only.
MCP read-only DBs (crypto_data, signal, wallet_db) are excluded.

PostgreSQL pitfalls checklist (postgresql/pitfalls.md):
- #8:  bar_labels PK is composite (run_id, timeframe, open_time) — OK
- #13: DOUBLE PRECISION allows NaN/Inf; _require_finite() rejects non-finite before insert
- #14: nullable BETWEEN uses IS NULL OR ... pattern — handled by DB CHECK
- #15: ENUM cast via ::text used in DB CHECK — applied at INSERT time
- #19: timestamptz literal with +00 suffix — yes in DDL
- #20: date_bin in joined view (not loader concern)
- #21: partition bound TIMESTAMPTZ has +00 — DDL already correct
"""

from __future__ import annotations

import math
import os
from datetime import datetime, timezone
from typing import Any, Sequence

import polars as pl
import psycopg

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_TIMEFRAMES = {"1m", "5m"}


def connect() -> psycopg.Connection[Any]:
    """Open a psycopg3 connection from REGIME_BENCHMARK_DB_URL env var.

    Application WRITE connection (not read-only, not MCP).

    Returns:
        Open psycopg.Connection.

    Raises:
        KeyError: If REGIME_BENCHMARK_DB_URL is not set.
        psycopg.OperationalError: If the connection fails.
    """
    dsn = os.environ["REGIME_BENCHMARK_DB_URL"]
    return psycopg.connect(dsn)


def register_run(
    conn: psycopg.Connection[Any],
    config: Any,  # LabelingConfig
    period_start: datetime,
    period_end: datetime,
    git_commit: str | None = None,
) -> int:
    """Insert a new labeling_runs row with run_status='loading', return run_id.

    Args:
        conn: Open psycopg3 connection (autocommit=False).
        config: LabelingConfig instance.
        period_start: UTC datetime for period_start_utc.
        period_end: UTC datetime for period_end_utc.
        git_commit: Optional short git hash for reproducibility.

    Returns:
        Newly created run_id (int).
    """
    # Ensure UTC-aware
    if period_start.tzinfo is None:
        period_start = period_start.replace(tzinfo=timezone.utc)
    if period_end.tzinfo is None:
        period_end = period_end.replace(tzinfo=timezone.utc)

    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO labeling_runs
                (method_version, symbol, market, period_start_utc, period_end_utc,
                 price_field, git_commit, run_status)
            VALUES (%s, %s, %s, %s, %s, %s, %s, 'loading')
            RETURNING id
            """,
            (
                config.method_version,
                config.data.symbol,
                config.data.market,
                period_start,
                period_end,
                config.price_field,
                git_commit,
            ),
        )
        row = cur.fetchone()
    # No commit here — whole run is one transaction; finalize_run commits (B1).
    assert row is not None
    return int(row[0])


def register_params(
    conn: psycopg.Connection[Any],
    run_id: int,
    timeframe: str,
    theta_dc: float,
    theta_amp: float,
    q_dc: float,
    k_dc: float,
    min_segment_bars: int,
    q_low: float,
    q_high: float,
    taker_fee_rate: float | None = None,
    slippage_rate_estimate: float | None = None,
) -> None:
    """Insert a labeling_run_params row for one timeframe.

    Args:
        conn: Open psycopg3 connection.
        run_id: labeling_runs.id for this run.
        timeframe: '1m' or '5m'.
        theta_dc: Computed DC threshold (> 0).
        theta_amp: Amplitude threshold (> 0).
        q_dc: Quantile level used for theta_dc (e.g. 0.80).
        k_dc: Scale factor used for theta_dc.
        min_segment_bars: Minimum segment bar count.
        q_low: Lower volatility quantile level (e.g. 0.33).
        q_high: Upper volatility quantile level (e.g. 0.66).
        taker_fee_rate: Optional taker fee rate for cost diagnostics.
        slippage_rate_estimate: Optional slippage estimate.
    """
    if timeframe not in _VALID_TIMEFRAMES:
        raise ValueError(f"Invalid timeframe {timeframe!r}")

    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO labeling_run_params
                (run_id, timeframe, q_dc, k_dc, min_segment_bars,
                 theta_dc, theta_amp, q_low, q_high,
                 taker_fee_rate, slippage_rate_estimate)
            VALUES (%s, %s::timeframe_enum, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                run_id,
                timeframe,
                q_dc,
                k_dc,
                min_segment_bars,
                theta_dc,
                theta_amp,
                q_low,
                q_high,
                taker_fee_rate,
                slippage_rate_estimate,
            ),
        )
    # No commit here — finalize_run commits the whole run (B1).


def copy_segments(
    conn: psycopg.Connection[Any],
    run_id: int,
    timeframe: str,
    segments: list[Any],  # list[Segment]
    method_version: str,
    symbol: str = "ETHUSDT",
    market: str = "BINANCE_USDM_FUTURES",
    bars_df: pl.DataFrame | None = None,
) -> None:
    """COPY segment_labels rows from Segment objects.

    Tail segments get direction_label/volatility_label/final_label as NULL
    and confirm fields as NULL (design §11.1 B3, DB ck_segment_labels_confirm).

    Non-tail segments get all fields NOT NULL.

    FK load order: must be called BEFORE copy_bars.

    Args:
        conn: Open psycopg3 connection.
        run_id: labeling_runs.id.
        timeframe: '1m' or '5m'.
        segments: List of Segment objects (from pipeline).
        method_version: e.g. 'regime_label_9axis_v1.1'.
        symbol: Default 'ETHUSDT'.
        market: Default 'BINANCE_USDM_FUTURES'.
        bars_df: Kline DataFrame (for open_time lookup by bar index).
    """
    if timeframe not in _VALID_TIMEFRAMES:
        raise ValueError(f"Invalid timeframe {timeframe!r}")

    # Build list of bar timestamps indexed by bar position for start/end/confirm
    ts_by_idx: list[datetime] = []
    if bars_df is not None:
        ts_col = bars_df["open_time"].to_list()
        ts_by_idx = [
            t if t.tzinfo else t.replace(tzinfo=timezone.utc)
            for t in ts_col
        ]

    def _get_ts(idx: int) -> datetime | None:
        if idx < 0 or idx >= len(ts_by_idx):
            return None
        return ts_by_idx[idx]

    rows: list[Sequence[Any]] = []
    for seg in segments:
        is_tail: bool = seg.is_tail_unconfirmed
        start_ts = _get_ts(seg.start_bar)
        end_ts = _get_ts(seg.end_bar)
        confirm_ts = _get_ts(seg.confirm_bar) if seg.confirm_bar is not None else None

        rows.append(
            (
                run_id,
                symbol,
                market,
                method_version,
                timeframe,
                seg.segment_id,
                start_ts,
                end_ts,
                is_tail,
                confirm_ts,                    # confirm_timestamp (NULL if tail)
                seg.lag_bars,                  # NULL if tail (int)
                _fin(seg.lag_move, "lag_move"),                       # NULL if tail
                _fin(seg.capturable_amplitude, "capturable_amplitude"),  # NULL if tail
                _clamp01(_fin(seg.capturable_ratio, "capturable_ratio")),  # NULL if tail
                _fin(seg.start_price_hlc3, "start_price_hlc3"),
                _fin(seg.end_price_hlc3, "end_price_hlc3"),
                _fin(seg.log_move, "log_move"),
                _fin(seg.amplitude, "amplitude"),
                _fin(seg.path_length, "path_length"),
                _clamp01(_fin(seg.efficiency_ratio, "efficiency_ratio")),
                _fin(seg.realized_volatility, "realized_volatility"),
                _fin(seg.realized_volatility_per_bar, "realized_volatility_per_bar"),
                _fin(seg.max_abs_d, "max_abs_d"),
                _clamp01(_fin(seg.max_jump_share, "max_jump_share")),
                _fin(seg.bipower_variation, "bipower_variation"),
                _fin(seg.jump_component, "jump_component"),
                _fin(seg.jump_share_bv, "jump_share_bv"),
                _fin(seg.rv_plus, "rv_plus"),
                _fin(seg.rv_minus, "rv_minus"),
                _clamp01(_fin(seg.downside_vol_share, "downside_vol_share")),
                _fin(seg.amplitude_to_cost_ratio, "amplitude_to_cost_ratio"),
                seg.low_tradeability_segment_flag,
                seg.pullback_within_parent_trend_flag,
                seg.direction_label,           # NULL if tail
                seg.volatility_label,          # NULL if tail
                seg.final_label,               # NULL if tail
                seg.n_bars,
            )
        )

    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO segment_labels (
                run_id, symbol, market, method_version, timeframe,
                segment_id,
                start_timestamp, end_timestamp,
                is_tail_unconfirmed,
                confirm_timestamp, lag_bars, lag_move,
                capturable_amplitude, capturable_ratio,
                start_price_hlc3, end_price_hlc3,
                log_move, amplitude, path_length, efficiency_ratio,
                realized_volatility, realized_volatility_per_bar,
                max_abs_d, max_jump_share, bipower_variation,
                jump_component, jump_share_bv,
                rv_plus, rv_minus, downside_vol_share,
                amplitude_to_cost_ratio,
                low_tradeability_segment_flag, pullback_within_parent_trend_flag,
                direction_label, volatility_label, final_label,
                segment_bar_count
            ) VALUES (
                %s, %s, %s, %s, %s::timeframe_enum,
                %s,
                %s, %s,
                %s,
                %s, %s, %s,
                %s, %s,
                %s, %s,
                %s, %s, %s, %s,
                %s, %s,
                %s, %s, %s,
                %s, %s,
                %s, %s, %s,
                %s,
                %s, %s,
                %s::direction_label, %s::volatility_label, %s::final_label,
                %s
            )
            """,
            rows,
        )
    # No commit here — finalize_run commits the whole run (B1).


def copy_bars(
    conn: psycopg.Connection[Any],
    run_id: int,
    timeframe: str,
    bars_df: pl.DataFrame,
    method_version: str,
    symbol: str = "ETHUSDT",
    market: str = "BINANCE_USDM_FUTURES",
) -> None:
    """COPY bar_labels rows from a bar-level DataFrame.

    Must be called AFTER copy_segments (FK fk_bar_labels_segment).

    Args:
        conn: Open psycopg3 connection.
        run_id: labeling_runs.id.
        timeframe: '1m' or '5m'.
        bars_df: Bar-level DataFrame with columns:
            open_time, open, high, low, close, hlc3,
            segment_id, direction_label, volatility_label, final_label.
        method_version: e.g. 'regime_label_9axis_v1.1'.
        symbol: Default 'ETHUSDT'.
        market: Default 'BINANCE_USDM_FUTURES'.
    """
    if timeframe not in _VALID_TIMEFRAMES:
        raise ValueError(f"Invalid timeframe {timeframe!r}")

    if len(bars_df) == 0:
        return

    rows: list[Sequence[Any]] = []
    for row in bars_df.iter_rows(named=True):
        open_time = row["open_time"]
        if hasattr(open_time, "tzinfo") and open_time.tzinfo is None:
            open_time = open_time.replace(tzinfo=timezone.utc)
        rows.append(
            (
                run_id,
                symbol,
                market,
                method_version,
                timeframe,
                open_time,
                _fin(float(row["open"]), "open"),
                _fin(float(row["high"]), "high"),
                _fin(float(row["low"]), "low"),
                _fin(float(row["close"]), "close"),
                _fin(float(row["hlc3"]), "hlc3"),
                str(row["segment_id"]),
                str(row["direction_label"]),
                str(row["volatility_label"]),
                str(row["final_label"]),
            )
        )

    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO bar_labels (
                run_id, symbol, market, method_version, timeframe,
                open_time,
                open, high, low, close, hlc3,
                segment_id,
                direction_label, volatility_label, final_label
            ) VALUES (
                %s, %s, %s, %s, %s::timeframe_enum,
                %s,
                %s, %s, %s, %s, %s,
                %s,
                %s::direction_label, %s::volatility_label, %s::final_label
            )
            """,
            rows,
        )
    # No commit here — finalize_run commits the whole run (B1).


def finalize_run(
    conn: psycopg.Connection[Any],
    run_id: int,
) -> None:
    """Set run_status='completed', completed_at=NOW().

    Only call after all COPY operations succeed.

    Args:
        conn: Open psycopg3 connection.
        run_id: labeling_runs.id to finalize.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE labeling_runs
               SET run_status = 'completed',
                   completed_at = NOW()
             WHERE id = %s
            """,
            (run_id,),
        )
    conn.commit()


# ---------------------------------------------------------------------------
# Legacy single-function API (kept for M1 stub compatibility)
# ---------------------------------------------------------------------------


def load_segment_labels(
    dsn: str,
    run_id: int,
    segments: pl.DataFrame,
    timeframe: str,
) -> None:
    """Legacy API: bulk-load segment-level labels into segment_labels via COPY.

    This function is superseded by copy_segments() in the M2 pipeline.
    """
    raise NotImplementedError(
        "load_segment_labels legacy API: use copy_segments() instead (M2)"
    )


def load_bar_labels(
    dsn: str,
    run_id: int,
    bar_labels: pl.DataFrame,
    timeframe: str,
) -> None:
    """Legacy API: bulk-load bar-level labels into bar_labels (partitioned) via COPY.

    This function is superseded by copy_bars() in the M2 pipeline.
    """
    raise NotImplementedError(
        "load_bar_labels legacy API: use copy_bars() instead (M2)"
    )


def register_run_legacy(
    dsn: str,
    method_version: str,
    symbol: str,
    market: str,
    period_start_utc: str,
    period_end_utc: str,
    price_field: str,
    git_commit: str,
) -> int:
    """Legacy API: insert a new labeling_runs row.

    Superseded by register_run() in the M2 pipeline.
    """
    raise NotImplementedError(
        "register_run legacy API: use register_run() instead (M2)"
    )


def complete_run(dsn: str, run_id: int) -> None:
    """Legacy API: set run_status='completed'.

    Superseded by finalize_run() in the M2 pipeline.
    """
    raise NotImplementedError(
        "complete_run legacy API: use finalize_run() instead (M2)"
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _fin(v: float | None, field: str) -> float | None:
    """Reject non-finite floats before insert (B2; pitfalls #13).

    DOUBLE PRECISION silently accepts NaN/Inf, which would corrupt downstream
    aggregations. A non-finite metric signals a computation bug → fail loudly.
    None passes through (nullable tail/confirm fields).
    """
    if v is None:
        return None
    if not math.isfinite(v):
        raise ValueError(f"non-finite value in {field}: {v!r}")
    return v


def _clamp01(v: float | None) -> float | None:
    """Clamp float to [0, 1] to satisfy DB range checks. None passes through."""
    if v is None:
        return None
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v
