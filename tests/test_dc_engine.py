"""Tests for direction/dc_engine.py — M3 deliverable.

Covers:
  - Bootstrap/TP_1 determinism (design §8.2)
  - Peak/trough strict alternation (req §11.2)
  - Lookahead non-reference (stability under right-extension)
  - theta_dc monotonicity (larger => fewer or equal TPs)
  - compute_theta_dc correctness, empty/NaN error handling
  - Tail: last in-progress extreme NOT in turning_points
  - build_segments: tail marked is_tail_unconfirmed
  - Edge cases: N=1, flat series, monotonic series
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from regime_benchmark.direction.dc_engine import (
    compute_theta_dc,
    run_dc_engine,
)
from regime_benchmark.direction.segments import build_segments

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log_prices(prices: list[float]) -> np.ndarray:
    """Convert raw prices to log prices."""
    return np.array([math.log(p) for p in prices], dtype=np.float64)


def _log_returns(p: np.ndarray) -> np.ndarray:
    """Build d array: d[0]=NaN, d[t]=p[t]-p[t-1]."""
    d = np.empty(len(p), dtype=np.float64)
    d[0] = np.nan
    d[1:] = p[1:] - p[:-1]
    return d


# ---------------------------------------------------------------------------
# 1. Bootstrap / TP_1 determinism (design §8.2)
# ---------------------------------------------------------------------------

# Fixed sequence used to lock the bootstrap TP_1 value.
#
# prices = [100.0, 99.0, 98.0, 99.5, 100.5, 100.3, 99.8, 99.0]
# theta_dc = 0.015
#
# Bootstrap: mode=SEEK_UP, p_ext=ln(100), t_ext=0
# t=1: ln(99) < ln(100) → trough candidate moves to t=1
# t=2: ln(98) < ln(99) → trough candidate moves to t=2
# t=3: ln(99.5) - ln(98) = 0.01519 >= 0.015 → CONFIRM trough at bar_index=2, confirm_bar=3
#
# Locked values:
#   TP_1.bar_index = 2
#   TP_1.point_type = 'trough'
#   TP_1.confirm_bar = 3
#   TP_1.log_price = ln(98)

_TP1_PRICES = [100.0, 99.0, 98.0, 99.5, 100.5, 100.3, 99.8, 99.0]
_TP1_THETA = 0.015


def test_tp1_bar_index() -> None:
    """TP_1 bar_index must be 2 (locked by bootstrap design §8.2)."""
    p = _log_prices(_TP1_PRICES)
    tps = run_dc_engine(p, _TP1_THETA)
    assert len(tps) >= 1, "Expected at least one turning point"
    assert tps[0].bar_index == 2, f"TP_1 bar_index={tps[0].bar_index!r}, expected 2"


def test_tp1_point_type() -> None:
    """TP_1 must be a trough (bootstrap starts in SEEK_UP mode)."""
    p = _log_prices(_TP1_PRICES)
    tps = run_dc_engine(p, _TP1_THETA)
    assert tps[0].point_type == "trough", f"TP_1 type={tps[0].point_type!r}, expected 'trough'"


def test_tp1_confirm_bar() -> None:
    """TP_1 confirm_bar must be 3 (first bar that exceeds theta from trough at bar 2)."""
    p = _log_prices(_TP1_PRICES)
    tps = run_dc_engine(p, _TP1_THETA)
    assert tps[0].confirm_bar == 3, f"TP_1 confirm_bar={tps[0].confirm_bar!r}, expected 3"


def test_tp1_log_price() -> None:
    """TP_1 log_price must equal ln(98) (the price at bar 2)."""
    p = _log_prices(_TP1_PRICES)
    tps = run_dc_engine(p, _TP1_THETA)
    expected = math.log(98.0)
    assert abs(tps[0].log_price - expected) < 1e-10, (
        f"TP_1 log_price={tps[0].log_price!r}, expected ln(98)={expected!r}"
    )


# ---------------------------------------------------------------------------
# 2. Peak/trough strict alternation (req §11.2)
# ---------------------------------------------------------------------------

def test_alternation_short_sequence() -> None:
    """Turning points must strictly alternate trough/peak/trough/…"""
    # trough@2 confirm@3, peak@4 confirm@7  (two TPs guaranteed)
    prices = _TP1_PRICES
    p = _log_prices(prices)
    tps = run_dc_engine(p, _TP1_THETA)
    assert len(tps) >= 2, "Expected at least two TPs for alternation test"
    for i in range(1, len(tps)):
        assert tps[i].point_type != tps[i - 1].point_type, (
            f"TP[{i - 1}] and TP[{i}] are both {tps[i].point_type!r} — no alternation"
        )


def test_alternation_zigzag_sequence() -> None:
    """Multiple zigzag turns must all alternate strictly."""
    # Long zigzag: peaks and troughs alternate multiple times
    prices = [
        100.0, 102.0, 98.0, 103.0, 96.0, 104.0, 95.0, 105.0
    ]
    theta = 0.01
    p = _log_prices(prices)
    tps = run_dc_engine(p, theta)
    # With theta=0.01 there should be multiple TPs
    assert len(tps) >= 2, f"Expected multiple TPs, got {len(tps)}"
    types = [tp.point_type for tp in tps]
    for i in range(1, len(types)):
        assert types[i] != types[i - 1], (
            f"Alternation violated at index {i}: {types[i - 1]!r} then {types[i]!r}"
        )


def test_alternation_random_seeded_sequences() -> None:
    """Alternation holds on multiple seeded random walk sequences."""
    rng = np.random.default_rng(seed=42)
    for trial in range(10):
        n = 60
        steps = rng.normal(0, 0.005, n)
        prices = 100.0 * np.exp(np.cumsum(steps))
        p = np.log(prices)
        theta = 0.008
        tps = run_dc_engine(p, theta)
        for i in range(1, len(tps)):
            assert tps[i].point_type != tps[i - 1].point_type, (
                f"Trial {trial}: alternation violated at index {i}"
            )


# ---------------------------------------------------------------------------
# 3. Lookahead non-reference (stability under right-extension)
# ---------------------------------------------------------------------------

def test_lookahead_truncation_same_result() -> None:
    """Feeding p[:confirm_bar+1] must confirm the same TP_1 as the full series.

    Design §11.1: confirmation uses only bars up to confirm_bar.
    Truncating at confirm_bar must yield the exact same (bar_index, point_type, confirm_bar).
    """
    p_full = _log_prices(_TP1_PRICES)
    tps_full = run_dc_engine(p_full, _TP1_THETA)
    assert len(tps_full) >= 1

    tp1 = tps_full[0]
    # Feed only up to and including confirm_bar
    p_prefix = p_full[: tp1.confirm_bar + 1]
    tps_prefix = run_dc_engine(p_prefix, _TP1_THETA)

    assert len(tps_prefix) >= 1, "Prefix run produced no TPs"
    assert tps_prefix[0].bar_index == tp1.bar_index, (
        f"bar_index changed after right-extension: prefix={tps_prefix[0].bar_index!r} "
        f"full={tp1.bar_index!r}"
    )
    assert tps_prefix[0].point_type == tp1.point_type, (
        f"point_type changed: prefix={tps_prefix[0].point_type!r} full={tp1.point_type!r}"
    )
    assert tps_prefix[0].confirm_bar == tp1.confirm_bar, (
        f"confirm_bar changed: prefix={tps_prefix[0].confirm_bar!r} full={tp1.confirm_bar!r}"
    )


def test_prefix_tps_stable_under_right_extension() -> None:
    """Lookahead non-reference proof: a TP confirmed at bar c depends ONLY on bars <= c.

    For every cut point, the prefix run p[:cut] must produce EXACTLY the set of TPs
    that the full run confirms before `cut` — same count, same order, same fields.
    This is two-sided:
      - soundness: every prefix TP appears in the full run (no spurious early TP), and
      - completeness: every full-run TP with confirm_bar < cut appears in the prefix
        (a buggy engine that SUPPRESSES an early TP using future bars would fail here).
    theta_dc is a fixed scalar (not data-derived) so prefix vs full are not confounded
    by a changing quantile.
    """
    prices = [
        100.0, 99.0, 98.0, 99.5, 100.5, 100.3, 99.8,
        99.0, 98.5, 99.8, 101.5, 100.0, 99.2, 100.8,
    ]
    theta = 0.015  # fixed — NOT recomputed from the data
    p_full = _log_prices(prices)
    tps_full = run_dc_engine(p_full, theta)

    if len(tps_full) < 2:
        pytest.skip("Not enough TPs for stability test on this sequence")

    # Parameterize across multiple cut points — one cut can mask a leak.
    for cut in range(2, len(prices) + 1):
        tps_prefix = run_dc_engine(p_full[:cut], theta)
        # The full run, restricted to TPs already confirmed strictly before `cut`,
        # is exactly what a prefix of length `cut` could legitimately know.
        expected = [tp for tp in tps_full if tp.confirm_bar < cut]
        assert len(tps_prefix) == len(expected), (
            f"cut={cut}: prefix has {len(tps_prefix)} TPs, "
            f"full-run-before-cut has {len(expected)} "
            f"(completeness+soundness mismatch → possible lookahead leak)"
        )
        for i, (tp_p, tp_f) in enumerate(zip(tps_prefix, expected)):
            assert tp_p.bar_index == tp_f.bar_index, (
                f"cut={cut} TP[{i}] bar_index: prefix={tp_p.bar_index!r} vs full={tp_f.bar_index!r}"
            )
            assert tp_p.point_type == tp_f.point_type, (
                f"cut={cut} TP[{i}] type: prefix={tp_p.point_type!r} vs full={tp_f.point_type!r}"
            )
            assert tp_p.confirm_bar == tp_f.confirm_bar, (
                f"cut={cut} TP[{i}] confirm_bar: "
                f"prefix={tp_p.confirm_bar!r} vs full={tp_f.confirm_bar!r}"
            )


# ---------------------------------------------------------------------------
# 4. theta_dc monotonicity
# ---------------------------------------------------------------------------

def test_theta_dc_monotonicity_zigzag() -> None:
    """Larger theta_dc produces fewer or equal TPs on the same series."""
    prices = [100.0, 102.0, 98.0, 103.0, 96.0, 104.0, 95.0, 105.0, 94.0, 106.0]
    p = _log_prices(prices)
    thetas = [0.005, 0.01, 0.02, 0.04, 0.08]
    counts = [len(run_dc_engine(p, t)) for t in thetas]
    for i in range(1, len(counts)):
        assert counts[i] <= counts[i - 1], (
            f"theta={thetas[i]} produced {counts[i]} TPs > {counts[i - 1]} TPs "
            f"at theta={thetas[i - 1]} — monotonicity violated"
        )


# ---------------------------------------------------------------------------
# 5. compute_theta_dc correctness
# ---------------------------------------------------------------------------

def test_compute_theta_dc_quantile_k() -> None:
    """theta = quantile(|d|, q) * k — verify with exact values."""
    # 10 values; quantile(0.80) of [0.1..1.0] == 0.82 (linear interp on sorted array)
    vals = pl.Series("abs_d", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    q = vals.quantile(0.80, interpolation="linear")
    assert q is not None
    k = 3.0
    expected = float(q) * k
    result = compute_theta_dc(vals, 0.80, k)
    assert abs(result - expected) < 1e-12, f"theta={result!r} expected={expected!r}"


def test_compute_theta_dc_positive() -> None:
    """compute_theta_dc must always return a strictly positive value."""
    vals = pl.Series("abs_d", [0.001, 0.002, 0.005, 0.010, 0.020])
    theta = compute_theta_dc(vals, 0.80, 4.0)
    assert theta > 0.0, f"theta={theta!r} is not positive"


def test_compute_theta_dc_empty_raises() -> None:
    """Empty Series must raise ValueError."""
    with pytest.raises(ValueError, match="empty"):
        compute_theta_dc(pl.Series("x", [], dtype=pl.Float64), 0.80, 3.0)


def test_compute_theta_dc_all_nan_raises() -> None:
    """All-NaN Series must raise ValueError after drop_nulls/drop_nans."""
    with pytest.raises(ValueError):
        compute_theta_dc(pl.Series("x", [float("nan"), float("nan")]), 0.80, 3.0)


def test_compute_theta_dc_with_nans_ignores_them() -> None:
    """NaN values are dropped before quantile; result uses clean values only."""
    clean = pl.Series("abs_d", [0.001, 0.002, 0.005, 0.010, 0.020])
    dirty = pl.Series("abs_d_nan", [float("nan"), 0.001, 0.002, float("nan"), 0.005, 0.010, 0.020])
    t_clean = compute_theta_dc(clean, 0.80, 3.0)
    t_dirty = compute_theta_dc(dirty, 0.80, 3.0)
    assert abs(t_clean - t_dirty) < 1e-12, (
        f"NaN contamination: clean={t_clean!r} dirty={t_dirty!r}"
    )


# ---------------------------------------------------------------------------
# 6. Tail: last in-progress extreme NOT in turning_points
# ---------------------------------------------------------------------------

def test_tail_not_in_turning_points() -> None:
    """The last in-progress extremum must NOT appear in turning_points."""
    # Using the standard TP_1 sequence: trough@2, peak@4 confirmed by bar 7.
    # After bar 7 the engine is in SEEK_UP again tracking bar 7 as new trough candidate.
    # That trough candidate is NOT emitted.
    p = _log_prices(_TP1_PRICES)
    tps = run_dc_engine(p, _TP1_THETA)
    # All TPs must have confirm_bar < len(p)
    n = len(p)
    for tp in tps:
        assert tp.confirm_bar < n
    # The last element of the series (bar 7) is in seek_up mode
    # and is the last tracked trough — NOT confirmed
    last_bar = n - 1
    tp_bars = {tp.bar_index for tp in tps}
    # The final in-progress extremum must NOT be a confirmed TP — no escape hatch.
    # (Confirmation requires a subsequent bar moving theta_dc away, which cannot
    #  exist for the last bar; so the last bar can never be a legitimate TP.)
    assert last_bar not in tp_bars, (
        f"Last bar {last_bar} emitted as a confirmed TP — tail not suppressed"
    )


def test_build_segments_tail_marked() -> None:
    """build_segments must produce exactly one trailing is_tail_unconfirmed segment."""
    p = _log_prices(_TP1_PRICES)
    d = _log_returns(p)
    tps = run_dc_engine(p, _TP1_THETA)
    segs = build_segments(tps, p, d)

    tail_segs = [s for s in segs if s.is_tail_unconfirmed]
    assert len(tail_segs) == 1, f"Expected 1 tail segment, got {len(tail_segs)}"

    confirmed_segs = [s for s in segs if not s.is_tail_unconfirmed]
    # Confirmed segments must come before the tail
    if confirmed_segs and tail_segs:
        assert confirmed_segs[-1].end_bar == tail_segs[0].start_bar, (
            "Tail segment must start at the last confirmed TP"
        )


# ---------------------------------------------------------------------------
# 7. Edge cases
# ---------------------------------------------------------------------------

def test_single_bar_no_tps() -> None:
    """N=1 → no turning points confirmed."""
    p = _log_prices([100.0])
    tps = run_dc_engine(p, 0.01)
    assert tps == [], f"Expected no TPs for N=1, got {tps}"


def test_single_bar_whole_series_is_tail() -> None:
    """N=1 → build_segments returns one tail segment spanning [0, 0]."""
    p = _log_prices([100.0])
    d = _log_returns(p)
    tps = run_dc_engine(p, 0.01)
    segs = build_segments(tps, p, d)
    assert len(segs) == 1, f"Expected 1 segment, got {len(segs)}"
    assert segs[0].is_tail_unconfirmed is True
    assert segs[0].start_bar == 0
    assert segs[0].end_bar == 0


def test_flat_series_no_tps() -> None:
    """Flat series (all equal p) → no TPs confirmed."""
    prices = [100.0] * 10
    p = _log_prices(prices)
    tps = run_dc_engine(p, 0.01)
    assert tps == [], f"Expected no TPs for flat series, got {tps!r}"


def test_monotonic_ascending_bootstrap_trough() -> None:
    """Strict monotonic ascending → only TP_1 (bootstrap trough) once threshold met."""
    # p[0] is bootstrap trough; price rises monotonically.
    # When p[t] - p[0] >= theta, TP_1 is confirmed.
    # Then SEEK_DOWN is entered but price never drops -> no more TPs.
    prices = [98.0, 99.0, 100.0, 101.0, 102.0, 103.0, 104.0]
    theta = 0.015
    p = _log_prices(prices)
    tps = run_dc_engine(p, theta)
    # May get TP_1 (trough@0) but never a second TP (no drop >= theta)
    assert len(tps) <= 1, f"Expected at most 1 TP for monotonic ascending, got {len(tps)}: {tps}"
    if len(tps) == 1:
        assert tps[0].point_type == "trough"
        assert tps[0].bar_index == 0


def test_monotonic_descending_no_tps() -> None:
    """Strictly monotonic descending → no TPs (bootstrap SEEK_UP, trough keeps moving down)."""
    prices = [104.0, 103.0, 102.0, 101.0, 100.0, 99.0, 98.0]
    theta = 0.015
    p = _log_prices(prices)
    tps = run_dc_engine(p, theta)
    # In SEEK_UP, price keeps falling so trough keeps updating. No upward move
    # of theta means no TP confirmed.
    assert tps == [], f"Expected no TPs for monotonic descending, got {tps!r}"


def test_two_bar_series_no_tp_if_below_threshold() -> None:
    """Two-bar series where rise < theta → no TP."""
    prices = [100.0, 100.005]
    theta = 0.01  # 0.005% rise, far below 1%
    p = _log_prices(prices)
    tps = run_dc_engine(p, theta)
    assert tps == []


def test_two_bar_series_tp_if_exceeds_threshold() -> None:
    """Two-bar series where rise >= theta → TP_1 confirmed at bar 1."""
    prices = [100.0, 102.0]  # ln(102)-ln(100) ≈ 0.0198
    theta = 0.015
    p = _log_prices(prices)
    tps = run_dc_engine(p, theta)
    assert len(tps) == 1
    assert tps[0].bar_index == 0
    assert tps[0].point_type == "trough"
    assert tps[0].confirm_bar == 1


def test_run_dc_engine_empty_raises() -> None:
    """Empty log_prices must raise ValueError."""
    with pytest.raises(ValueError, match="non-empty"):
        run_dc_engine(np.array([], dtype=np.float64), 0.01)


def test_run_dc_engine_nonpositive_theta_raises() -> None:
    """theta_dc <= 0 must raise ValueError."""
    p = _log_prices([100.0, 101.0])
    with pytest.raises(ValueError, match="positive"):
        run_dc_engine(p, 0.0)
    with pytest.raises(ValueError, match="positive"):
        run_dc_engine(p, -0.01)
