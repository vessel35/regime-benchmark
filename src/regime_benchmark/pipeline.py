"""Top-level pipeline orchestration — design §15 / requirements.md §16.

Executes the 9-label labeling pipeline for both 1m and 5m timeframes:
  1-3.  Ingest & trim (ingest/binance.py)
  4-6.  Quality checks (quality/checks.py) — skipped in M2 synthetic mode
  7-8.  Price transform (transform/price.py)
  9-10. DC threshold + turning points (direction/dc_engine.py)
  11.   Direction labels (direction/segments.py)
  12-14. Realized volatility + quantiles + labels (volatility/realized.py)
  15.   Final 9-label assembly (labeling/assemble.py)
  16.   Auxiliary diagnostics (diagnostics/)
  17.   Bar-level expansion (labeling/assemble.py)
  18.   1m-5m join is a DB VIEW; no computation here
  19.   Validation report (validation/report.py) — deferred to M9

Loading order: segment_labels -> bar_labels -> labeling_reports.
Run status is set to 'completed' only after all steps pass.

Frozen calibration (design §8.1):
  Pass `calibration` to run_pipeline to supply pre-decided per-timeframe
  FrozenParams (chosen on a calibration split, then locked).  When absent the
  pipeline falls back to candidate[0] for backward compatibility.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import polars as pl

from regime_benchmark.calibration import FrozenParams
from regime_benchmark.config import LabelingConfig
from regime_benchmark.diagnostics.asymmetry import compute_asymmetry_diagnostics_segments
from regime_benchmark.diagnostics.cost import compute_cost_diagnostics_segments
from regime_benchmark.diagnostics.jump import compute_jump_diagnostics_segments
from regime_benchmark.diagnostics.lag import compute_lag_diagnostics_segments
from regime_benchmark.direction.dc_engine import compute_theta_dc, run_dc_engine
from regime_benchmark.direction.segments import assign_direction, build_segments
from regime_benchmark.ingest.binance import make_synthetic_klines
from regime_benchmark.labeling.assemble import assign_final_labels, expand_to_bars
from regime_benchmark.persistence.loader import (
    connect,
    copy_bars,
    copy_segments,
    finalize_run,
    register_params,
    register_run,
)
from regime_benchmark.transform.price import add_price_columns
from regime_benchmark.volatility.realized import (
    assign_volatility_labels,
    compute_segment_rv,
    compute_volatility_quantiles,
)

# Cost-diagnostic defaults (the only module defaults consumed; k_dc /
# min_segment_bars / quantiles come from config candidate[0] or FrozenParams).
_DEFAULT_TAKER_FEE = 0.0004
_DEFAULT_SLIPPAGE = 0.0002

# Synthetic data window (M2 thin-run / tests): ~1 month per timeframe.
_BAR_SECONDS = {"1m": 60, "5m": 300}
_SYNTHETIC_PERIODS = {"1m": 43200, "5m": 8640}
_SYNTHETIC_START = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)


def run_pipeline(
    config: LabelingConfig,
    source_map: dict[str, str | Path] | None = None,
    synthetic: bool = False,
    calibration: dict[str, FrozenParams] | None = None,
    period_start: datetime | None = None,
    period_end: datetime | None = None,
) -> int:
    """Execute the full 9-label pipeline and return the run_id.

    Each timeframe tau in {1m, 5m} is processed independently.
    Threshold and quantile calculations are never shared across timeframes.

    Args:
        config: Validated LabelingConfig instance.
        source_map: Dict mapping timeframe to CSV file path (e.g.
            {'1m': 'data/1m.csv', '5m': 'data/5m.csv'}).
            Ignored when synthetic=True.
        synthetic: If True, generate synthetic OHLCV data instead of loading
            from source_map.  Useful for tests and M2 thin-run.
        calibration: Optional per-timeframe FrozenParams chosen on a calibration
            split and locked (design §8.1).  When a timeframe key is present the
            frozen k_dc / q_dc / min_segment_bars / q_low / q_high override the
            config candidate[0] defaults.  When absent (None or missing key) the
            existing candidate[0] fallback is used — backward compatible with all
            M2/M3 tests.
        period_start: Actual start of the loaded data slice, recorded in
            labeling_runs.period_start_utc. Defaults to config.data.start_utc.
            Pass the real slice bounds (e.g. a 6-month dev window) so the run
            metadata reflects what was loaded, not the full config span.
        period_end: Actual end of the loaded data slice. Defaults to
            config.data.end_utc.

    Returns:
        run_id: The labeling_runs.id for this pipeline execution.

    Raises:
        KeyError: If REGIME_BENCHMARK_DB_URL is not set.
        ValueError: If synthetic=False and source_map is None or missing keys.
    """
    # SF3: period_start/period_end must be both-or-neither (a partial override
    # would silently mix the caller's start with config's end → wrong metadata).
    if (period_start is None) != (period_end is None):
        raise ValueError("period_start and period_end must both be provided or both be None")

    # SF2: a partial calibration dict would silently mix frozen params for some
    # timeframes with candidate[0] for others — harmful for reproducibility.
    if calibration is not None:
        missing = set(config.timeframes) - set(calibration.keys())
        if missing:
            raise ValueError(
                f"calibration must cover all timeframes {config.timeframes}; "
                f"missing frozen params for {sorted(missing)}"
            )

    conn = connect()

    try:
        # Determine period bounds. Explicit period_start/period_end (the ACTUAL
        # loaded slice) take precedence so labeling_runs records the real data
        # range, not the full config span (e.g. a 6-month dev slice). Falls back
        # to config span, or the synthetic window when synthetic=True.
        run_period_start = period_start if period_start is not None else config.data.start_utc
        run_period_end = period_end if period_end is not None else config.data.end_utc
        if synthetic and period_start is None and period_end is None:
            run_period_start = _SYNTHETIC_START
            # Derive the actual synthetic end = latest last-bar across timeframes
            # (avoids the old hard-coded date overstating the loaded span).
            run_period_end = max(
                _SYNTHETIC_START
                + timedelta(seconds=(_SYNTHETIC_PERIODS[tf] - 1) * _BAR_SECONDS[tf])
                for tf in config.timeframes
            )

        run_id = register_run(conn, config, run_period_start, run_period_end, git_commit=None)

        # Process each timeframe independently (§9.1: never pool across timeframes)
        for tf in config.timeframes:
            _run_timeframe(
                conn=conn,
                run_id=run_id,
                timeframe=tf,
                config=config,
                source_map=source_map,
                synthetic=synthetic,
                frozen=calibration.get(tf) if calibration is not None else None,
            )

        finalize_run(conn, run_id)

    except Exception:
        # B1: the whole run is one transaction (loader helpers don't commit;
        # finalize_run commits). On any failure, roll back — this un-does
        # register_run + params + segments + bars atomically, leaving NO orphan
        # 'loading' run and NO partial children. (The old code tried to UPDATE
        # on an already-aborted connection, which silently failed and left the
        # observed run_id=1 'loading' orphan with 2110 partial segments.)
        try:
            conn.rollback()
        except Exception:
            pass
        raise
    finally:
        conn.close()

    return run_id


def _run_timeframe(
    conn: object,
    run_id: int,
    timeframe: str,
    config: LabelingConfig,
    source_map: dict[str, str | Path] | None,
    synthetic: bool,
    frozen: FrozenParams | None = None,
) -> None:
    """Run the full pipeline for one timeframe and load into DB.

    Args:
        conn: Open psycopg3 connection (autocommit=False).
        run_id: labeling_runs.id for this pipeline execution.
        timeframe: '1m' or '5m'.
        config: Validated LabelingConfig instance.
        source_map: Dict mapping timeframe to CSV/parquet path (None if synthetic).
        synthetic: If True, use make_synthetic_klines instead of source_map.
        frozen: Pre-decided FrozenParams for this timeframe (design §8.1).
            When None, falls back to config candidate[0] defaults.
    """
    # --- Step 1-3: Ingest
    if synthetic:
        klines = make_synthetic_klines(
            timeframe=timeframe,  # type: ignore[arg-type]
            start_utc=_SYNTHETIC_START,
            periods=_SYNTHETIC_PERIODS[timeframe],
            seed=42,
        )
    else:
        from regime_benchmark.ingest.binance import load_klines
        if source_map is None or timeframe not in source_map:
            raise ValueError(f"source_map missing key {timeframe!r}")
        klines = load_klines(source=source_map[timeframe], timeframe=timeframe)  # type: ignore[arg-type]

    # --- Steps 7-8: Price transform
    klines = add_price_columns(klines)

    p_arr = klines["log_price"].to_numpy().astype(np.float64)
    d_arr = klines["log_return"].to_numpy().astype(np.float64)
    # S1: keep the first-bar NaN (d[0] is undefined); compute_theta_dc drops
    # NaN/null itself. Replacing NaN with 0.0 here would inflate the zero-return
    # mass and bias the quantile downward.
    abs_d_series = pl.Series("abs_d", np.abs(d_arr))

    # --- Step 9: DC threshold — resolve params from frozen calibration or candidate[0]
    tf_params = config.direction_method.params[timeframe]  # type: ignore[index]
    if frozen is not None:
        # Frozen calibration path (design §8.1): use pre-decided k/q, override candidate[0]
        k_dc = float(frozen.k_dc)
        q_dc = float(frozen.q_dc)
        min_segment_bars = int(frozen.min_segment_bars)
        q_low = float(frozen.q_low)
        q_high = float(frozen.q_high)
    else:
        # Backward-compatible candidate[0] fallback
        k_dc = float(tf_params.k_dc_candidates[0])
        q_dc = float(tf_params.q_dc)
        min_segment_bars = int(tf_params.min_segment_bars_candidates[0])
        q_low = float(config.volatility_method.quantiles.low)
        q_high = float(config.volatility_method.quantiles.high)

    theta_dc = compute_theta_dc(abs_d_series, q_dc, k_dc)
    theta_amp = theta_dc  # same_as_theta_dc policy (§8.4)

    # --- Step 10: Turning points + segments
    turning_points = run_dc_engine(p_arr, theta_dc)
    segments = build_segments(turning_points, p_arr, d_arr)

    if len(segments) == 0:
        # Degenerate case: nothing to label; still record the used params
        register_params(
            conn,  # type: ignore[arg-type]
            run_id,
            timeframe,
            theta_dc,
            theta_amp,
            q_dc,
            k_dc,
            min_segment_bars,
            q_low,
            q_high,
        )
        return

    # --- Step 11: Direction labels
    for seg in segments:
        seg.direction_label = assign_direction(seg, min_segment_bars, theta_amp)

    # --- Steps 12-13: Realized volatility + quantiles
    segments = compute_segment_rv(segments, d_arr)
    confirmed_rv = [
        s.realized_volatility_per_bar
        for s in segments
        if not s.is_tail_unconfirmed
    ]
    if confirmed_rv:
        q_low_val, q_high_val = compute_volatility_quantiles(confirmed_rv, q_low, q_high)
    else:
        q_low_val, q_high_val = 0.0, 0.0

    # --- Step 14: Volatility labels
    segments = assign_volatility_labels(segments, q_low_val, q_high_val)

    # --- Step 15: Final 9-labels
    segments = assign_final_labels(segments)

    # --- Step 16: Diagnostics
    compute_lag_diagnostics_segments(segments, p_arr, theta_dc)
    compute_jump_diagnostics_segments(segments, d_arr)
    compute_asymmetry_diagnostics_segments(segments, d_arr)
    compute_cost_diagnostics_segments(segments, _DEFAULT_TAKER_FEE, _DEFAULT_SLIPPAGE)

    # --- Step 17: Bar-level expansion
    bars_df = expand_to_bars(segments, klines)

    # --- Persist: record the actual used values (frozen or candidate[0]) into labeling_run_params
    register_params(
        conn,  # type: ignore[arg-type]
        run_id,
        timeframe,
        theta_dc,
        theta_amp,
        q_dc,
        k_dc,
        min_segment_bars,
        q_low,
        q_high,
        taker_fee_rate=_DEFAULT_TAKER_FEE,
        slippage_rate_estimate=_DEFAULT_SLIPPAGE,
    )

    copy_segments(
        conn,  # type: ignore[arg-type]
        run_id,
        timeframe,
        segments,
        method_version=config.method_version,
        symbol=config.data.symbol,
        market=config.data.market,
        bars_df=klines,
    )

    copy_bars(
        conn,  # type: ignore[arg-type]
        run_id,
        timeframe,
        bars_df,
        method_version=config.method_version,
        symbol=config.data.symbol,
        market=config.data.market,
    )
