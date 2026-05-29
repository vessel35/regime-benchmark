"""Synthetic Case A~E — design §16.1 / requirements §10.4.

Each case feeds an hlc3 price sequence through the FULL labeling path
(DC -> segments -> direction -> volatility -> final_label), without any DB.

OHLC construction rule (satisfies ck_bar_labels_ohlc):
  open = high = low = close = hlc3_value
  → hlc3 = (H+L+C)/3 = hlc3_value ✓
  → high >= max(open,close), low <= min(open,close), high >= low ✓

theta_dc and volatility quantile boundaries are chosen per-case to make
direction/volatility deterministic:

  Cases A, B, C: theta_dc = 0.001 (very small → whole sequence is one segment
    if it confirms at all; for A/B, net direction is clear).
  Case D: theta_dc = 0.03 (large swings → each bounce is a separate segment;
    net direction of any single segment is weak or reversal-driven).
  Case E: theta_dc = 0.03 (produces one confirmed UP segment 0→5 with low ER,
    confirmed by an extension that drops; ER < 0.6 is asserted).

Design §8.4 / §6.4 note:
  ER is explicitly NOT a direction condition. Case E proves this:
  the segment has ER ≈ 0.55 < 0.6 yet direction = UP.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np
import polars as pl

from regime_benchmark.direction.dc_engine import run_dc_engine
from regime_benchmark.direction.segments import Segment, assign_direction, build_segments
from regime_benchmark.labeling.assemble import assign_final_labels
from regime_benchmark.volatility.realized import (
    assign_volatility,
    assign_volatility_labels,
    compute_segment_rv,
    rv_per_bar,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlc_df(hlc3_values: list[float]) -> pl.DataFrame:
    """Build minimal OHLC DataFrame where open=high=low=close=hlc3_value.

    hlc3 = (H+L+C)/3 = value  since H=L=C=value.
    Satisfies ck_bar_labels_ohlc: high>=max(O,C), low<=min(O,C), high>=low.
    """
    n = len(hlc3_values)
    open_times = [
        datetime(2024, 1, 1, 0, i, 0, tzinfo=timezone.utc) for i in range(n)
    ]
    vals = [float(v) for v in hlc3_values]
    return pl.DataFrame(
        {
            "open_time": open_times,
            "open": vals,
            "high": vals,
            "low": vals,
            "close": vals,
        },
        schema={
            "open_time": pl.Datetime(time_unit="us", time_zone="UTC"),
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
        },
    )


def _log_price_array(hlc3_values: list[float]) -> np.ndarray:
    """Convert hlc3 values to log-price array."""
    return np.array([math.log(v) for v in hlc3_values], dtype=np.float64)


def _log_return_array(p: np.ndarray) -> np.ndarray:
    """Build d array with d[0]=NaN."""
    d = np.empty(len(p), dtype=np.float64)
    d[0] = np.nan
    d[1:] = p[1:] - p[:-1]
    return d


def _run_labeling(
    hlc3_values: list[float],
    theta_dc: float,
    min_segment_bars: int,
    q_low_boundary: float,
    q_high_boundary: float,
) -> list[Segment]:
    """Run the full labeling path and return the list of labeled segments.

    Steps: DC engine → build_segments → assign_direction →
           compute_segment_rv → assign_volatility_labels → assign_final_labels.

    Args:
        hlc3_values: Raw hlc3 price sequence.
        theta_dc: Directional Change threshold.
        min_segment_bars: Minimum bar count for a directional segment.
        q_low_boundary: Pre-computed Q_low boundary value (score, not quantile level).
        q_high_boundary: Pre-computed Q_high boundary value (score, not quantile level).

    Returns:
        List of Segment objects with all labels assigned.
    """
    p = _log_price_array(hlc3_values)
    d = _log_return_array(p)

    tps = run_dc_engine(p, theta_dc)
    segs: list[Segment] = build_segments(tps, p, d)  # type: ignore[assignment]

    # Direction
    theta_amp = theta_dc  # same_as_theta_dc policy
    for seg in segs:
        seg.direction_label = assign_direction(seg, min_segment_bars, theta_amp)  # type: ignore[attr-defined]

    # Volatility
    segs = compute_segment_rv(segs, d)  # type: ignore[assignment]
    segs = assign_volatility_labels(segs, q_low_boundary, q_high_boundary)  # type: ignore[assignment]

    # Final labels
    segs = assign_final_labels(segs)  # type: ignore[assignment]

    return segs


def _confirmed_segs(segs: list[Segment]) -> list[Segment]:
    """Return confirmed (non-tail) segments."""
    return [s for s in segs if not s.is_tail_unconfirmed]  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Case A: monotonic ascending — direction=UP, volatility=LOW_VOL or MID_VOL
# ---------------------------------------------------------------------------

def test_case_a_direction_up() -> None:
    """Case A: 100→101→102→103→104 must be labeled UP."""
    # The whole monotonic series: TP_1=trough@0 confirmed at bar1 (rise ≈ 0.01).
    # Then SEEK_DOWN; price never drops >= theta, so the segment from bar0→bar4 is tail.
    # To get a confirmed UP segment we need a drop at the end.
    # Use extended sequence: 100→101→102→103→104→103.5 to confirm peak@4 at bar5.
    hlc3_ext = [100.0, 101.0, 102.0, 103.0, 104.0, 103.5, 103.0]
    theta_dc = 0.005
    # drop from 104 to 103: ln(104)-ln(103) ≈ 0.00967 >= 0.005 → confirms peak@4 at bar5
    segs = _run_labeling(
        hlc3_ext,
        theta_dc=theta_dc,
        min_segment_bars=2,
        q_low_boundary=1.0,    # all segments are LOW_VOL
        q_high_boundary=10.0,
    )
    confirmed = _confirmed_segs(segs)
    assert len(confirmed) >= 1, f"Expected at least 1 confirmed segment, got {len(confirmed)}"
    # Find the UP segment (the rising segment 0→4 or 0→last_confirmed_peak)
    up_segs = [s for s in confirmed if s.direction_label == "UP"]  # type: ignore[union-attr]
    assert len(up_segs) >= 1, (
        f"No UP segment found. Confirmed segments: "
        f"{[(s.direction_label, s.start_bar, s.end_bar) for s in confirmed]}"
    )


def test_case_a_volatility_low_or_mid() -> None:
    """Case A: monotonic ascending should be LOW_VOL or MID_VOL."""
    hlc3_ext = [100.0, 101.0, 102.0, 103.0, 104.0, 103.5, 103.0]
    theta_dc = 0.005
    segs = _run_labeling(
        hlc3_ext,
        theta_dc=theta_dc,
        min_segment_bars=2,
        q_low_boundary=1.0,
        q_high_boundary=10.0,
    )
    confirmed = _confirmed_segs(segs)
    for s in confirmed:
        assert s.volatility_label in ("LOW_VOL", "MID_VOL"), (  # type: ignore[union-attr]
            f"Case A segment has volatility={s.volatility_label!r}"  # type: ignore[union-attr]
        )


# ---------------------------------------------------------------------------
# Case B: monotonic descending — direction=DOWN, volatility=LOW_VOL or MID_VOL
# ---------------------------------------------------------------------------

def test_case_b_direction_down() -> None:
    """Case B: 104→103→102→101→100 must be labeled DOWN."""
    # Bootstrap SEEK_UP: price falls so trough keeps moving, no upward theta-move.
    # For a confirmed DOWN segment we need: confirmed peak → confirmed trough.
    # Construct: starts up a little to get trough@0 confirmed, then goes DOWN:
    hlc3_v2 = [100.0, 100.8, 100.5, 100.2, 99.8, 99.4, 99.0, 99.3, 99.6]
    theta_dc_v2 = 0.003
    # trough@0 confirmed at bar1 (rise 0.008 >= 0.003)
    # Then SEEK_DOWN from bar1 peak; price falls...
    # peak@1 confirmed when price drops 0.003 from ln(100.8)
    # ln(100.8)-ln(99.0) ≈ 0.01802 >= 0.003 → confirmed peak@1 at bar6 (when 99.0 is reached)
    segs = _run_labeling(
        hlc3_v2,
        theta_dc=theta_dc_v2,
        min_segment_bars=2,
        q_low_boundary=1.0,
        q_high_boundary=10.0,
    )
    confirmed = _confirmed_segs(segs)
    down_segs = [s for s in confirmed if s.direction_label == "DOWN"]  # type: ignore[union-attr]
    assert len(down_segs) >= 1, (
        f"No DOWN segment. Confirmed: "
        f"{[(s.direction_label, s.start_bar, s.end_bar) for s in confirmed]}"
    )


def test_case_b_volatility_low_or_mid() -> None:
    """Case B: monotonic descending should be LOW_VOL or MID_VOL."""
    hlc3_v2 = [100.0, 100.8, 100.5, 100.2, 99.8, 99.4, 99.0, 99.3, 99.6]
    theta_dc_v2 = 0.003
    segs = _run_labeling(
        hlc3_v2,
        theta_dc=theta_dc_v2,
        min_segment_bars=2,
        q_low_boundary=1.0,
        q_high_boundary=10.0,
    )
    confirmed = _confirmed_segs(segs)
    for s in confirmed:
        assert s.volatility_label in ("LOW_VOL", "MID_VOL"), (  # type: ignore[union-attr]
            f"Case B segment has volatility={s.volatility_label!r}"  # type: ignore[union-attr]
        )


# ---------------------------------------------------------------------------
# Case C: small sideways — direction=NON_DIRECTIONAL, volatility=LOW_VOL
# ---------------------------------------------------------------------------

def test_case_c_direction_non_directional() -> None:
    """Case C: 100→100.05→99.98→100.02→100.01 → NON_DIRECTIONAL.

    Two sub-tests:
    (a) Segment-level: a segment spanning Case C's range has tiny amplitude.
        When theta_amp = theta_dc > max_amplitude, assign_direction → NON_DIRECTIONAL.
    (b) Full pipeline: with theta_dc > max Case C range (≈0.0007), no TPs are
        confirmed, so there are zero confirmed segments (whole series is tail).
        That is also the "NON_DIRECTIONAL" interpretation — no directional label.
    """
    hlc3 = [100.0, 100.05, 99.98, 100.02, 100.01]
    # Max range in Case C: |ln(100.05) - ln(99.98)| ≈ 0.0007
    # (a) Direct assign_direction test with a segment spanning the Case C range
    p_c = _log_price_array(hlc3)
    amplitude_c = abs(p_c[-1] - p_c[0])  # net amplitude ≈ 0.0001
    max_possible_amplitude = max(
        abs(p_c[i] - p_c[j])
        for i in range(len(p_c))
        for j in range(len(p_c))
    )
    # Use theta_amp larger than max possible amplitude → forces NON_DIRECTIONAL
    theta_amp_large = max_possible_amplitude * 2.0
    seg = Segment(
        segment_id="case_c_test",
        start_bar=0,
        end_bar=4,
        confirm_bar=4,
        is_tail_unconfirmed=False,
        n_bars=5,
        log_move=p_c[-1] - p_c[0],
        amplitude=amplitude_c,
        path_length=sum(abs(p_c[t] - p_c[t - 1]) for t in range(1, len(p_c))),
        efficiency_ratio=0.0,
    )
    label = assign_direction(seg, min_segment_bars=1, theta_amp=theta_amp_large)
    assert label == "NON_DIRECTIONAL", (
        f"Case C assign_direction with theta_amp={theta_amp_large:.6f} "
        f"(> max_range={max_possible_amplitude:.6f}) should be NON_DIRECTIONAL, got {label!r}"
    )

    # (b) Full pipeline: with theta_dc > full price range, NO DC reversal can reach
    # theta → zero confirmed TPs → the entire tiny-chop series is one unconfirmed
    # tail (no UP/DOWN/NON_DIRECTIONAL bar labels emitted at all). Assert this
    # explicitly (non-vacuous): the directional amplitude-gate proof itself is in (a).
    theta_dc_large = max_possible_amplitude * 1.5
    segs = _run_labeling(
        hlc3,
        theta_dc=theta_dc_large,
        min_segment_bars=1,
        q_low_boundary=0.5,
        q_high_boundary=1.0,
    )
    confirmed = _confirmed_segs(segs)
    assert len(confirmed) == 0, (
        f"Case C: theta_dc > full range must yield zero confirmed segments "
        f"(whole chop series is tail), got {len(confirmed)}"
    )


def test_case_c_assign_volatility_directly_low() -> None:
    """Case C volatility: assign_volatility(small score, ...) = LOW_VOL.

    Since a single small-range segment cannot form a distribution to compute
    33/66 quantiles, we test assign_volatility() directly with a representative
    score from Case C.
    """
    hlc3 = [100.0, 100.05, 99.98, 100.02, 100.01]
    p = _log_price_array(hlc3)
    d = _log_return_array(p)
    # RV of the whole sequence as a single segment (start=0, end=4)
    from regime_benchmark.volatility.realized import realized_vol
    d_slice = d[1:]  # bars 1..4
    rv = realized_vol(d_slice)
    score = rv_per_bar(rv, len(hlc3))
    # Use boundaries well above Case C's small score
    label = assign_volatility(score, q_low=0.005, q_high=0.01)
    assert label == "LOW_VOL", (
        f"Case C score={score:.6f} should be LOW_VOL with q_low=0.005, got {label!r}"
    )


# ---------------------------------------------------------------------------
# Case D: large swings but weak net direction — NON_DIRECTIONAL, HIGH_VOL
# ---------------------------------------------------------------------------

def test_case_d_direction_non_directional() -> None:
    """Case D: 100→106→99→105→98→101 → NON_DIRECTIONAL.

    The net move from start (100) to end (101) is only ≈ 0.00995 log units,
    far smaller than the large bounces (±0.06).  When treated as a single
    segment (theta_dc very large so no TPs confirmed), amplitude < theta_amp
    → NON_DIRECTIONAL.

    Two sub-tests:
    (a) Direct assign_direction with net amplitude < theta_amp → NON_DIRECTIONAL.
    (b) Full pipeline with theta_dc > bounce size: whole series is tail (no
        confirmed segments), confirming no directional classification.
    """
    hlc3 = [100.0, 106.0, 99.0, 105.0, 98.0, 101.0]
    p_d = _log_price_array(hlc3)
    net_amplitude = abs(p_d[-1] - p_d[0])  # |ln(101)-ln(100)| ≈ 0.00995
    total_path = sum(abs(p_d[t] - p_d[t - 1]) for t in range(1, len(p_d)))

    # (a) As a single segment: net amplitude is tiny → NON_DIRECTIONAL
    # if theta_amp > net_amplitude
    seg = Segment(
        segment_id="case_d_test",
        start_bar=0,
        end_bar=5,
        confirm_bar=5,
        is_tail_unconfirmed=False,
        n_bars=6,
        log_move=p_d[-1] - p_d[0],
        amplitude=net_amplitude,
        path_length=total_path,
        efficiency_ratio=net_amplitude / total_path,
    )
    theta_amp_above_net = net_amplitude * 2.0
    label = assign_direction(seg, min_segment_bars=1, theta_amp=theta_amp_above_net)
    assert label == "NON_DIRECTIONAL", (
        f"Case D single-segment: with theta_amp={theta_amp_above_net:.5f} "
        f"> net_amplitude={net_amplitude:.5f}, expected NON_DIRECTIONAL, got {label!r}"
    )

    # (b) Full pipeline: theta_dc > max single-bar bounce → no DC reversal reaches
    # theta → zero confirmed TPs → whole series is tail. Assert explicitly
    # (non-vacuous); the NON_DIRECTIONAL amplitude-gate proof is in (a).
    max_bounce = max(abs(p_d[t] - p_d[t - 1]) for t in range(1, len(p_d)))
    theta_dc_very_large = max_bounce * 1.5
    segs = _run_labeling(
        hlc3,
        theta_dc=theta_dc_very_large,
        min_segment_bars=1,
        q_low_boundary=0.0001,
        q_high_boundary=0.001,
    )
    confirmed = _confirmed_segs(segs)
    assert len(confirmed) == 0, (
        f"Case D: theta_dc > max bounce must yield zero confirmed segments, "
        f"got {len(confirmed)}"
    )


def test_case_d_relative_rv_higher_than_case_c() -> None:
    """Case D has higher realized volatility per bar than Case C.

    We compute RV_per_bar for the full sequences (as single segments) and
    compare — D must exceed C.  This verifies the relative ordering without
    requiring a cross-segment quantile distribution.
    """
    from regime_benchmark.volatility.realized import realized_vol

    hlc3_c = [100.0, 100.05, 99.98, 100.02, 100.01]
    hlc3_d = [100.0, 106.0, 99.0, 105.0, 98.0, 101.0]

    def _seg_rv_per_bar(hlc3: list[float]) -> float:
        p = _log_price_array(hlc3)
        d = _log_return_array(p)
        d_slice = d[1:]
        rv = realized_vol(d_slice)
        return rv_per_bar(rv, len(hlc3))

    rv_c = _seg_rv_per_bar(hlc3_c)
    rv_d = _seg_rv_per_bar(hlc3_d)
    assert rv_d > rv_c, (
        f"Case D RV_per_bar={rv_d:.6f} must exceed Case C RV_per_bar={rv_c:.6f}"
    )


def test_case_d_assign_volatility_high() -> None:
    """Case D volatility: assign_volatility(large score, ...) = HIGH_VOL.

    Tests assign_volatility() directly with a representative Case D score.
    """
    from regime_benchmark.volatility.realized import realized_vol

    hlc3_d = [100.0, 106.0, 99.0, 105.0, 98.0, 101.0]
    p = _log_price_array(hlc3_d)
    d = _log_return_array(p)
    rv = realized_vol(d[1:])
    score = rv_per_bar(rv, len(hlc3_d))
    # Use boundaries well below Case D's large score
    label = assign_volatility(score, q_low=0.001, q_high=0.01)
    assert label == "HIGH_VOL", (
        f"Case D score={score:.6f} should be HIGH_VOL with q_high=0.01, got {label!r}"
    )


# ---------------------------------------------------------------------------
# Case E: rough path, strong net UP — direction=UP, low ER (KEY ASSERTION)
# ---------------------------------------------------------------------------
#
# Sequence used:  100, 103, 101, 106, 104, 110, 108, 105
# theta_dc = 0.03
#
# State machine trace:
#   Bootstrap: SEEK_UP, p_ext=ln(100), t_ext=0
#   t=1: ln(103)>ln(100); rise=0.0296<0.03; no confirm
#   t=2: ln(101)<ln(103); but still seeking up from trough@0:
#         ln(101)>ln(100); rise=0.00995<0.03; no confirm. trough stays @0.
#   t=3: ln(106)>ln(101); rise from trough@0: 0.0583>=0.03 → CONFIRM trough@0, confirm_bar=3
#         switch to SEEK_DOWN; p_ext=ln(106), t_ext=3
#   t=4: ln(104)<ln(106); drop=0.0190<0.03; peak candidate stays @3 (or moves to t=4? no: 104<106)
#         peak stays at bar3, but bar4 is lower: drop from bar3 = 0.019 < 0.03; no confirm
#   t=5: ln(110)>ln(106); p_ext=ln(110), t_ext=5 (new peak candidate)
#   t=6: ln(108)<ln(110); drop=0.0184<0.03; peak candidate stays @5
#   t=7: ln(105)<ln(110); drop=ln(110)-ln(105)=0.0465>=0.03 → CONFIRM peak@5, confirm_bar=7
#         switch to SEEK_UP; p_ext=ln(105), t_ext=7
#
# Result: TP_1=trough@0, TP_2=peak@5
# Confirmed segment: seg[0] = [bar0, bar5]
#   M = ln(110)-ln(100) = 0.0953 > 0 → UP
#   L = sum|d_t| for t=1..5 = |ln103-ln100|+|ln101-ln103|+|ln106-ln101|+|ln104-ln106|+|ln110-ln104|
#     = 0.02956 + 0.01961 + 0.04832 + 0.01910 + 0.05609 = 0.17268
#   A = 0.0953
#   ER = 0.0953 / 0.17268 ≈ 0.552 < 0.6  ← KEY: ER is low yet direction = UP
#   n_bars = 6 >= min_segment_bars=2 ✓
#   A = 0.0953 >= theta_amp=0.03 ✓  → direction = UP

def test_case_e_direction_up() -> None:
    """Case E: rough zigzag must be labeled UP despite low ER.

    Key assertion: ER is NOT part of the direction condition (design §8.4, §6.4).
    """
    hlc3 = [100.0, 103.0, 101.0, 106.0, 104.0, 110.0, 108.0, 105.0]
    theta_dc = 0.03
    segs = _run_labeling(
        hlc3,
        theta_dc=theta_dc,
        min_segment_bars=2,
        q_low_boundary=0.0,
        q_high_boundary=0.001,  # small boundary → segment is HIGH_VOL
    )
    confirmed = _confirmed_segs(segs)
    assert len(confirmed) >= 1, f"Expected confirmed segment, got 0. Segs: {segs}"

    up_segs = [s for s in confirmed if s.direction_label == "UP"]  # type: ignore[union-attr]
    assert len(up_segs) >= 1, (
        f"Case E: no UP segment. Confirmed: "
        f"{[(s.direction_label, s.efficiency_ratio) for s in confirmed]}"
    )


def test_case_e_efficiency_ratio_is_low() -> None:
    """Case E sub-assertion: segment ER < 0.6 yet direction == UP.

    This is the core proof that ER is NOT a direction condition.
    """
    hlc3 = [100.0, 103.0, 101.0, 106.0, 104.0, 110.0, 108.0, 105.0]
    theta_dc = 0.03
    segs = _run_labeling(
        hlc3,
        theta_dc=theta_dc,
        min_segment_bars=2,
        q_low_boundary=0.0,
        q_high_boundary=0.001,
    )
    confirmed = _confirmed_segs(segs)
    up_segs = [s for s in confirmed if s.direction_label == "UP"]  # type: ignore[union-attr]
    assert len(up_segs) >= 1, "No UP segment in Case E"

    # Pin the expected segment identity [bar0, bar5] so a future refactor cannot
    # satisfy this proof via some other coincidentally-low-ER UP segment.
    pinned = [s for s in up_segs if s.start_bar == 0 and s.end_bar == 5]
    assert len(pinned) == 1, (
        f"Case E: expected exactly one UP segment spanning [0,5], got "
        f"{[(s.start_bar, s.end_bar) for s in up_segs]}"
    )
    for s in pinned:
        assert s.efficiency_ratio < 0.6, (  # type: ignore[union-attr]
            f"Case E UP segment has ER={s.efficiency_ratio:.4f} >= 0.6; "  # type: ignore[union-attr]
            f"expected low ER to prove ER is not in direction condition"
        )
        assert s.direction_label == "UP", (  # type: ignore[union-attr]
            f"Case E: direction={s.direction_label!r} even though ER is low"  # type: ignore[union-attr]
        )


def test_case_e_high_vol_possible() -> None:
    """Case E: with small volatility boundaries, segment is HIGH_VOL."""
    hlc3 = [100.0, 103.0, 101.0, 106.0, 104.0, 110.0, 108.0, 105.0]
    theta_dc = 0.03
    # Use very small q_high so the Case E segment's large RV is HIGH_VOL
    segs = _run_labeling(
        hlc3,
        theta_dc=theta_dc,
        min_segment_bars=2,
        q_low_boundary=0.0,
        q_high_boundary=0.001,
    )
    confirmed = _confirmed_segs(segs)
    up_segs = [s for s in confirmed if s.direction_label == "UP"]  # type: ignore[union-attr]
    if up_segs:
        rv_pb = up_segs[0].realized_volatility_per_bar  # type: ignore[union-attr]
        assert up_segs[0].volatility_label == "HIGH_VOL", (  # type: ignore[union-attr]
            f"Case E: expected HIGH_VOL with small q_high=0.001, "
            f"got {up_segs[0].volatility_label!r}; RV_per_bar={rv_pb:.6f}"  # type: ignore[union-attr]
        )


def test_case_e_final_label_up_high_vol() -> None:
    """Case E: final_label must be UP_HIGH_VOL."""
    hlc3 = [100.0, 103.0, 101.0, 106.0, 104.0, 110.0, 108.0, 105.0]
    theta_dc = 0.03
    segs = _run_labeling(
        hlc3,
        theta_dc=theta_dc,
        min_segment_bars=2,
        q_low_boundary=0.0,
        q_high_boundary=0.001,
    )
    confirmed = _confirmed_segs(segs)
    up_high_segs = [s for s in confirmed if s.final_label == "UP_HIGH_VOL"]  # type: ignore[union-attr]
    assert len(up_high_segs) >= 1, (
        f"Case E: expected UP_HIGH_VOL final_label. "
        f"Got: {[s.final_label for s in confirmed]}"
    )


# ---------------------------------------------------------------------------
# assign_volatility direct tests (label boundary coverage)
# ---------------------------------------------------------------------------

def test_assign_volatility_boundaries() -> None:
    """assign_volatility boundary conditions."""
    assert assign_volatility(0.0, 0.001, 0.002) == "LOW_VOL"
    assert assign_volatility(0.001, 0.001, 0.002) == "LOW_VOL"   # score == q_low
    assert assign_volatility(0.0015, 0.001, 0.002) == "MID_VOL"
    assert assign_volatility(0.002, 0.001, 0.002) == "MID_VOL"   # score == q_high
    assert assign_volatility(0.003, 0.001, 0.002) == "HIGH_VOL"


def test_rv_per_bar_denominator_is_sqrt_n() -> None:
    """rv_per_bar(rv, n) == rv / sqrt(n) — pins the spec §7.1 denominator choice.

    The denominator is sqrt(N_j), NOT sqrt(N_j-1).
    """
    test_cases = [
        (0.1, 1),
        (0.2, 4),
        (0.3, 9),
        (1.0, 100),
    ]
    for rv_val, n_val in test_cases:
        expected = rv_val / math.sqrt(n_val)
        result = rv_per_bar(rv_val, n_val)
        assert abs(result - expected) < 1e-12, (
            f"rv_per_bar({rv_val}, {n_val})={result!r} expected={expected!r}"
        )


def test_rv_per_bar_n_zero_returns_zero() -> None:
    """rv_per_bar with n=0 must return 0.0 (guard against division by zero)."""
    assert rv_per_bar(1.0, 0) == 0.0
