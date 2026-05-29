"""Property / invariant tests — design §11 / req §11.2-§11.4.

Tests in this file use fixed-seed synthetic sequences to verify structural
and numerical invariants of the DC engine + segment pipeline.

All tests are DB-free (pure-function, no PostgreSQL).

Invariants verified:
  1. Segment non-overlap + full coverage of labelable window (half-open partition)
  2. Turning point alternation holds on fixed-seed random sequences
  3. Range invariants: 0 <= ER <= 1, RV >= 0, RV_per_bar >= 0, N_j >= 1, A_j >= 0,
     0 <= capturable_ratio <= 1
  4. min_segment_bars / theta_amp gate → NON_DIRECTIONAL (never UP/DOWN)
  5. RV denominator is sqrt(N_j): rv_per_bar(rv, n) == rv / sqrt(n)
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from regime_benchmark.diagnostics.lag import compute_lag_diagnostics_segments
from regime_benchmark.direction.dc_engine import run_dc_engine
from regime_benchmark.direction.segments import Segment, assign_direction, build_segments
from regime_benchmark.labeling.assemble import assign_final_labels
from regime_benchmark.volatility.realized import (
    assign_volatility_labels,
    compute_segment_rv,
    rv_per_bar,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log_prices(prices: list[float] | np.ndarray) -> np.ndarray:
    if isinstance(prices, np.ndarray):
        return np.log(prices)
    return np.array([math.log(p) for p in prices], dtype=np.float64)


def _log_returns(p: np.ndarray) -> np.ndarray:
    d = np.empty(len(p), dtype=np.float64)
    d[0] = np.nan
    d[1:] = p[1:] - p[:-1]
    return d


def _full_pipeline(
    prices: list[float] | np.ndarray,
    theta_dc: float,
    min_segment_bars: int = 2,
    q_low_boundary: float = 0.0,
    q_high_boundary: float = 1e9,
) -> list[Segment]:
    """Run the complete labeling pipeline and return all segments."""
    p = _log_prices(prices)
    d = _log_returns(p)
    tps = run_dc_engine(p, theta_dc)
    segs = build_segments(tps, p, d)
    theta_amp = theta_dc
    for seg in segs:
        seg.direction_label = assign_direction(seg, min_segment_bars, theta_amp)
    segs = compute_segment_rv(segs, d)
    segs = assign_volatility_labels(segs, q_low_boundary, q_high_boundary)
    segs = assign_final_labels(segs)
    compute_lag_diagnostics_segments(segs, p, theta_dc)
    return segs


def _confirmed(segs: list[Segment]) -> list[Segment]:
    return [s for s in segs if not s.is_tail_unconfirmed]


# ---------------------------------------------------------------------------
# 1. Segment non-overlap + full coverage (half-open partition)
# ---------------------------------------------------------------------------

def test_no_duplicate_start_bars() -> None:
    """No two segments may start at the same bar (non-overlap)."""
    rng = np.random.default_rng(seed=7)
    steps = rng.normal(0, 0.008, 80)
    prices = 100.0 * np.exp(np.cumsum(steps))
    segs = _full_pipeline(prices, theta_dc=0.015)
    start_bars = [s.start_bar for s in segs]
    assert len(start_bars) == len(set(start_bars)), (
        f"Duplicate start_bar found: {start_bars!r}"
    )


def test_labelable_window_full_coverage() -> None:
    """Bars in [first_confirmed_TP, last_confirmed_TP] appear exactly once.

    expand_to_bars uses a half-open convention: each segment owns [start, end)
    except the last confirmed segment which owns [start, end].
    This test verifies the coverage at the segment (not bar) level:
    the union of all confirmed segment ranges covers [TP_1.bar, TP_n.bar]
    without gaps or overlaps.
    """
    rng = np.random.default_rng(seed=13)
    steps = rng.normal(0, 0.010, 120)
    prices = 100.0 * np.exp(np.cumsum(steps))
    segs = _full_pipeline(prices, theta_dc=0.020)
    confirmed = _confirmed(segs)

    if len(confirmed) < 2:
        pytest.skip("Not enough confirmed segments for coverage test")

    first_start = confirmed[0].start_bar
    last_end = confirmed[-1].end_bar

    # Build a coverage count for bar-ownership (half-open except last)
    coverage: dict[int, int] = {}
    for i, seg in enumerate(confirmed):
        is_last = (i == len(confirmed) - 1)
        stop = seg.end_bar + 1 if is_last else seg.end_bar
        for bar in range(seg.start_bar, stop):
            coverage[bar] = coverage.get(bar, 0) + 1

    # Every bar in [first_start, last_end] must appear exactly once
    for bar in range(first_start, last_end + 1):
        count = coverage.get(bar, 0)
        assert count == 1, (
            f"Bar {bar} appears {count} times in half-open partition "
            f"(expected 1)"
        )


def test_no_gap_between_consecutive_segments() -> None:
    """Consecutive confirmed segments must be adjacent (no gap bars)."""
    rng = np.random.default_rng(seed=17)
    steps = rng.normal(0, 0.010, 100)
    prices = 100.0 * np.exp(np.cumsum(steps))
    segs = _full_pipeline(prices, theta_dc=0.015)
    confirmed = _confirmed(segs)
    for i in range(1, len(confirmed)):
        prev_end = confirmed[i - 1].end_bar
        curr_start = confirmed[i].start_bar
        assert curr_start == prev_end, (
            f"Gap between seg[{i - 1}].end_bar={prev_end} and "
            f"seg[{i}].start_bar={curr_start}"
        )


# ---------------------------------------------------------------------------
# 2. Turning point alternation (seeded random sequences)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed,n,theta", [
    (0, 50, 0.010),
    (1, 80, 0.012),
    (2, 100, 0.015),
    (3, 60, 0.008),
    (4, 120, 0.020),
])
def test_alternation_seeded(seed: int, n: int, theta: float) -> None:
    """Turning points must strictly alternate trough/peak/trough/…

    Uses fixed seeds to ensure reproducibility (not Math.random).
    """
    rng = np.random.default_rng(seed=seed)
    steps = rng.normal(0, 0.008, n)
    prices = 100.0 * np.exp(np.cumsum(steps))
    p = _log_prices(prices)
    tps = run_dc_engine(p, theta)
    if len(tps) < 2:
        return  # Not enough TPs to test alternation; not a failure
    for i in range(1, len(tps)):
        assert tps[i].point_type != tps[i - 1].point_type, (
            f"seed={seed}: TP[{i - 1}].type={tps[i - 1].point_type!r} and "
            f"TP[{i}].type={tps[i].point_type!r} violate alternation"
        )


# ---------------------------------------------------------------------------
# 3. Range invariants on segment metrics
# ---------------------------------------------------------------------------

def _make_invariant_segs(seed: int, n: int = 120, theta: float = 0.015) -> list[Segment]:
    rng = np.random.default_rng(seed=seed)
    steps = rng.normal(0, 0.008, n)
    prices = 100.0 * np.exp(np.cumsum(steps))
    return _full_pipeline(prices, theta_dc=theta, min_segment_bars=2)


@pytest.mark.parametrize("seed", [5, 6, 7, 8, 9])
def test_efficiency_ratio_range(seed: int) -> None:
    """0 <= ER_j <= 1 for all segments."""
    segs = _make_invariant_segs(seed)
    for s in segs:
        assert 0.0 <= s.efficiency_ratio <= 1.0 + 1e-10, (
            f"ER={s.efficiency_ratio!r} out of [0,1] for {s.segment_id}"
        )


@pytest.mark.parametrize("seed", [5, 6, 7, 8, 9])
def test_realized_volatility_non_negative(seed: int) -> None:
    """RV_j >= 0 for all segments."""
    segs = _make_invariant_segs(seed)
    for s in segs:
        assert s.realized_volatility >= 0.0, (
            f"RV={s.realized_volatility!r} < 0 for {s.segment_id}"
        )


@pytest.mark.parametrize("seed", [5, 6, 7, 8, 9])
def test_rv_per_bar_non_negative(seed: int) -> None:
    """RV_per_bar_j >= 0 for all segments."""
    segs = _make_invariant_segs(seed)
    for s in segs:
        assert s.realized_volatility_per_bar >= 0.0, (
            f"RV_per_bar={s.realized_volatility_per_bar!r} < 0 for {s.segment_id}"
        )


@pytest.mark.parametrize("seed", [5, 6, 7, 8, 9])
def test_n_bars_at_least_one(seed: int) -> None:
    """N_j >= 1 for all segments."""
    segs = _make_invariant_segs(seed)
    for s in segs:
        assert s.n_bars >= 1, f"N_j={s.n_bars} < 1 for {s.segment_id}"


@pytest.mark.parametrize("seed", [5, 6, 7, 8, 9])
def test_amplitude_non_negative(seed: int) -> None:
    """A_j >= 0 for all segments."""
    segs = _make_invariant_segs(seed)
    for s in segs:
        assert s.amplitude >= 0.0, f"A_j={s.amplitude!r} < 0 for {s.segment_id}"


@pytest.mark.parametrize("seed", [5, 6, 7, 8, 9])
def test_capturable_ratio_range(seed: int) -> None:
    """0 <= capturable_ratio <= 1 for confirmed segments."""
    segs = _make_invariant_segs(seed)
    confirmed = _confirmed(segs)
    for s in confirmed:
        if s.capturable_ratio is not None:
            assert 0.0 <= s.capturable_ratio <= 1.0 + 1e-10, (
                f"capturable_ratio={s.capturable_ratio!r} out of [0,1] for {s.segment_id}"
            )


@pytest.mark.parametrize("seed", [5, 6, 7, 8, 9])
def test_lag_bars_non_negative(seed: int) -> None:
    """lag_bars >= 0 for confirmed segments."""
    segs = _make_invariant_segs(seed)
    confirmed = _confirmed(segs)
    for s in confirmed:
        if s.lag_bars is not None:
            assert s.lag_bars >= 0, (
                f"lag_bars={s.lag_bars!r} < 0 for {s.segment_id}"
            )


# ---------------------------------------------------------------------------
# 4. min_segment_bars / theta_amp gate → NON_DIRECTIONAL
# ---------------------------------------------------------------------------

def test_short_segment_is_non_directional() -> None:
    """Segment with n_bars < min_segment_bars must be NON_DIRECTIONAL."""
    # Craft a 2-bar segment (N_j = 2) with a large amplitude.
    # Set min_segment_bars = 3 → forced NON_DIRECTIONAL.
    seg = Segment(
        segment_id="test",
        start_bar=0,
        end_bar=1,       # N_j = 2
        confirm_bar=1,
        is_tail_unconfirmed=False,
        n_bars=2,
        log_move=0.5,    # large positive move → would be UP if not for N gate
        amplitude=0.5,
        path_length=0.5,
        efficiency_ratio=1.0,
    )
    label = assign_direction(seg, min_segment_bars=3, theta_amp=0.01)
    assert label == "NON_DIRECTIONAL", (
        f"Short segment (n=2, min=3) must be NON_DIRECTIONAL, got {label!r}"
    )


def test_small_amplitude_segment_is_non_directional() -> None:
    """Segment with A_j < theta_amp must be NON_DIRECTIONAL."""
    seg = Segment(
        segment_id="test",
        start_bar=0,
        end_bar=9,       # N_j = 10 >= min_segment_bars
        confirm_bar=9,
        is_tail_unconfirmed=False,
        n_bars=10,
        log_move=0.001,  # tiny positive move
        amplitude=0.001, # < theta_amp = 0.01
        path_length=0.01,
        efficiency_ratio=0.1,
    )
    label = assign_direction(seg, min_segment_bars=2, theta_amp=0.01)
    assert label == "NON_DIRECTIONAL", (
        f"Small amplitude (A=0.001, theta_amp=0.01) must be NON_DIRECTIONAL, got {label!r}"
    )


def test_both_gates_satisfied_down() -> None:
    """Segment satisfying both gates with M_j < 0 must be DOWN."""
    seg = Segment(
        segment_id="test",
        start_bar=0,
        end_bar=4,
        confirm_bar=4,
        is_tail_unconfirmed=False,
        n_bars=5,
        log_move=-0.05,  # negative → DOWN
        amplitude=0.05,
        path_length=0.05,
        efficiency_ratio=1.0,
    )
    label = assign_direction(seg, min_segment_bars=3, theta_amp=0.03)
    assert label == "DOWN", f"Expected DOWN, got {label!r}"


def test_both_gates_satisfied_up() -> None:
    """Segment satisfying both gates with M_j > 0 must be UP."""
    seg = Segment(
        segment_id="test",
        start_bar=0,
        end_bar=4,
        confirm_bar=4,
        is_tail_unconfirmed=False,
        n_bars=5,
        log_move=0.05,   # positive → UP
        amplitude=0.05,
        path_length=0.08,
        efficiency_ratio=0.625,
    )
    label = assign_direction(seg, min_segment_bars=3, theta_amp=0.03)
    assert label == "UP", f"Expected UP, got {label!r}"


def test_tail_segment_direction_is_none() -> None:
    """Tail segment (is_tail_unconfirmed=True) must return None direction."""
    seg = Segment(
        segment_id="tail",
        start_bar=5,
        end_bar=10,
        confirm_bar=None,
        is_tail_unconfirmed=True,
        n_bars=6,
        log_move=0.1,
        amplitude=0.1,
        path_length=0.1,
        efficiency_ratio=1.0,
    )
    label = assign_direction(seg, min_segment_bars=2, theta_amp=0.01)
    assert label is None, f"Tail segment direction must be None, got {label!r}"


# ---------------------------------------------------------------------------
# 5. RV denominator is sqrt(N_j)
# ---------------------------------------------------------------------------

def test_rv_per_bar_formula_matches_sqrt_n() -> None:
    """rv_per_bar(rv, n) == rv / sqrt(n) — pins spec §7.1 denominator.

    The denominator is sqrt(N_j), not sqrt(N_j-1).
    """
    cases = [
        (0.01, 1),
        (0.05, 4),
        (0.10, 9),
        (0.20, 16),
        (0.50, 100),
        (1.00, 400),
    ]
    for rv_val, n_val in cases:
        expected = rv_val / math.sqrt(n_val)
        result = rv_per_bar(rv_val, n_val)
        assert abs(result - expected) < 1e-12, (
            f"rv_per_bar({rv_val}, {n_val}) = {result!r}, expected {expected!r}"
        )


def test_rv_per_bar_not_equal_sqrt_n_minus_1() -> None:
    """rv_per_bar must use sqrt(N_j), not sqrt(N_j-1).

    If rv = 1.0 and n = 10:
      spec value  = 1.0 / sqrt(10)  ≈ 0.31623
      wrong value = 1.0 / sqrt(9)   = 1/3      ≈ 0.33333
    """
    rv_val = 1.0
    n = 10
    spec_value = rv_val / math.sqrt(n)           # 0.31623...
    wrong_value = rv_val / math.sqrt(n - 1)      # 0.33333...
    result = rv_per_bar(rv_val, n)
    assert abs(result - spec_value) < 1e-10, (
        f"rv_per_bar uses wrong denominator: got {result!r}, expected {spec_value!r}"
    )
    assert abs(result - wrong_value) > 1e-6, (
        "rv_per_bar appears to use sqrt(N-1) denominator — spec requires sqrt(N)"
    )


# ---------------------------------------------------------------------------
# 6. Turning point bar_index ordering (TPs are chronological)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [10, 11, 12])
def test_turning_points_chronological(seed: int) -> None:
    """Turning point bar_indices must be strictly increasing."""
    rng = np.random.default_rng(seed=seed)
    steps = rng.normal(0, 0.009, 100)
    prices = 100.0 * np.exp(np.cumsum(steps))
    p = _log_prices(prices)
    tps = run_dc_engine(p, theta_dc=0.012)
    for i in range(1, len(tps)):
        assert tps[i].bar_index > tps[i - 1].bar_index, (
            f"seed={seed}: TP bar_indices not strictly increasing: "
            f"{tps[i - 1].bar_index} then {tps[i].bar_index}"
        )


# ---------------------------------------------------------------------------
# 7. Segment end_bar >= start_bar
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [20, 21, 22])
def test_segment_end_bar_ge_start_bar(seed: int) -> None:
    """end_bar >= start_bar for every segment."""
    rng = np.random.default_rng(seed=seed)
    steps = rng.normal(0, 0.008, 90)
    prices = 100.0 * np.exp(np.cumsum(steps))
    segs = _full_pipeline(prices, theta_dc=0.012)
    for s in segs:
        assert s.end_bar >= s.start_bar, (
            f"end_bar={s.end_bar} < start_bar={s.start_bar} for {s.segment_id}"
        )


# ---------------------------------------------------------------------------
# 8. Final labels are one of the 9 canonical labels (no extras)
# ---------------------------------------------------------------------------

_CANONICAL_LABELS = frozenset({
    "UP_LOW_VOL", "UP_MID_VOL", "UP_HIGH_VOL",
    "DOWN_LOW_VOL", "DOWN_MID_VOL", "DOWN_HIGH_VOL",
    "NON_DIRECTIONAL_LOW_VOL", "NON_DIRECTIONAL_MID_VOL", "NON_DIRECTIONAL_HIGH_VOL",
})


@pytest.mark.parametrize("seed", [30, 31, 32])
def test_final_labels_are_canonical(seed: int) -> None:
    """All confirmed final_labels must be one of the 9 canonical labels."""
    rng = np.random.default_rng(seed=seed)
    steps = rng.normal(0, 0.009, 100)
    prices = 100.0 * np.exp(np.cumsum(steps))
    segs = _full_pipeline(prices, theta_dc=0.012, q_low_boundary=0.001, q_high_boundary=0.01)
    confirmed = _confirmed(segs)
    for s in confirmed:
        assert s.final_label in _CANONICAL_LABELS, (
            f"seed={seed}: final_label={s.final_label!r} is not canonical"
        )
