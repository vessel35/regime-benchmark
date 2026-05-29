"""Bulk COPY loader for segment_labels and bar_labels — design §13.7.

Uses psycopg3 COPY ... FROM STDIN for high-throughput loading of ~1.26M 1m bars.

Loading order (FK constraint): segment_labels first, then bar_labels.
After both timeframes are loaded, caller sets run_status='completed'.

Write target: dedicated 'regime_benchmark' DB only.
MCP read-only DBs (crypto_data, signal, wallet_db) are excluded.
"""

from __future__ import annotations

import polars as pl


def load_segment_labels(
    dsn: str,
    run_id: int,
    segments: pl.DataFrame,
    timeframe: str,
) -> None:
    """Bulk-load segment-level labels into segment_labels via COPY.

    Args:
        dsn: PostgreSQL DSN string (from env var, never stored).
        run_id: labeling_runs.id for this pipeline run.
        segments: Fully assembled segment DataFrame (all columns required).
        timeframe: '1m' or '5m'.

    Raises:
        NotImplementedError: Implementation deferred to Milestone M7.
    """
    raise NotImplementedError("load_segment_labels is implemented in Milestone M7")


def load_bar_labels(
    dsn: str,
    run_id: int,
    bar_labels: pl.DataFrame,
    timeframe: str,
) -> None:
    """Bulk-load bar-level labels into bar_labels (partitioned) via COPY.

    Args:
        dsn: PostgreSQL DSN string.
        run_id: labeling_runs.id for this pipeline run.
        bar_labels: Bar-level label DataFrame (all columns required).
        timeframe: '1m' or '5m'.

    Raises:
        NotImplementedError: Implementation deferred to Milestone M7.
    """
    raise NotImplementedError("load_bar_labels is implemented in Milestone M7")


def register_run(
    dsn: str,
    method_version: str,
    symbol: str,
    market: str,
    period_start_utc: str,
    period_end_utc: str,
    price_field: str,
    git_commit: str,
) -> int:
    """Insert a new labeling_runs row with run_status='loading' and return run_id.

    Args:
        dsn: PostgreSQL DSN string.
        method_version: e.g. 'regime_label_9axis_v1.1'.
        symbol: e.g. 'ETHUSDT'.
        market: e.g. 'BINANCE_USDM_FUTURES'.
        period_start_utc: ISO timestamp string.
        period_end_utc: ISO timestamp string.
        price_field: e.g. 'hlc3'.
        git_commit: Short git commit hash for reproducibility.

    Returns:
        Newly created run_id (BIGINT).

    Raises:
        NotImplementedError: Implementation deferred to Milestone M7.
    """
    raise NotImplementedError("register_run is implemented in Milestone M7")


def complete_run(dsn: str, run_id: int) -> None:
    """Set run_status='completed' and completed_at=NOW() for the given run.

    Args:
        dsn: PostgreSQL DSN string.
        run_id: labeling_runs.id to mark as completed.

    Raises:
        NotImplementedError: Implementation deferred to Milestone M7.
    """
    raise NotImplementedError("complete_run is implemented in Milestone M7")
