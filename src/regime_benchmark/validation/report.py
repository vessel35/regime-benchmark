"""Validation report generation — requirements.md §10, §11, §15.

Runs:
- Synthetic Case A~E (§16.1)
- Segment invariants: turning point alternation, non-overlap, coverage (§11.2)
- Label invariants: single label per bar, 9-value domain, direction×vol consistency (§11.3)
- Numeric invariants: range checks for RV, ER, ratios (§11.4)

Outputs PASS/FAIL result with structured payload stored in labeling_reports.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

import numpy as np

from regime_benchmark.direction.dc_engine import run_dc_engine
from regime_benchmark.direction.segments import Segment, assign_direction, build_segments
from regime_benchmark.labeling.assemble import assign_final_labels
from regime_benchmark.volatility.realized import (
    assign_volatility,
    assign_volatility_labels,
    compute_segment_rv,
    realized_vol,
    rv_per_bar,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CANONICAL_LABELS = frozenset(
    {
        "UP_LOW_VOL",
        "UP_MID_VOL",
        "UP_HIGH_VOL",
        "DOWN_LOW_VOL",
        "DOWN_MID_VOL",
        "DOWN_HIGH_VOL",
        "NON_DIRECTIONAL_LOW_VOL",
        "NON_DIRECTIONAL_MID_VOL",
        "NON_DIRECTIONAL_HIGH_VOL",
    }
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _log_prices(hlc3_values: list[float]) -> np.ndarray:
    return np.array([math.log(v) for v in hlc3_values], dtype=np.float64)


def _log_returns(p: np.ndarray) -> np.ndarray:
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
    """Run the full labeling path and return labeled segments.

    Reuses the real engine functions — does NOT re-implement DC/vol/label logic.
    """
    p = _log_prices(hlc3_values)
    d = _log_returns(p)
    tps = run_dc_engine(p, theta_dc)
    segs: list[Segment] = build_segments(tps, p, d)
    theta_amp = theta_dc  # same_as_theta_dc policy (§8.4)
    for seg in segs:
        seg.direction_label = assign_direction(seg, min_segment_bars, theta_amp)
    segs = compute_segment_rv(segs, d)
    segs = assign_volatility_labels(segs, q_low_boundary, q_high_boundary)
    segs = assign_final_labels(segs)
    return segs


def _confirmed(segs: list[Segment]) -> list[Segment]:
    return [s for s in segs if not s.is_tail_unconfirmed]


def _ts_str(v: Any) -> str:
    """Convert a timestamp value to an ISO string for JSON serialisation."""
    if v is None:
        return ""
    if isinstance(v, datetime):
        return v.isoformat()
    return str(v)


def _sort_key(r: dict[str, Any]) -> tuple[datetime, str]:
    """Chronological sort key for segment rows, robust to None/naive datetimes.

    Real data carries TIMESTAMPTZ (UTC-aware) values; this coerces a missing or
    timezone-naive ``start_timestamp`` to a UTC-aware value so sorting never raises
    a naive-vs-aware ``TypeError`` (defensive for synthetic rows). Ties broken by id.
    """
    ts = r.get("start_timestamp")
    if ts is None:
        ts = datetime.min.replace(tzinfo=timezone.utc)
    elif isinstance(ts, datetime) and ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return (ts, str(r.get("segment_id", "")))


# ---------------------------------------------------------------------------
# 1. Synthetic Case A~E (§16.1)
# ---------------------------------------------------------------------------


def run_synthetic_cases() -> list[dict[str, Any]]:
    """Run §16.1 synthetic Case A~E formula proofs.

    Uses the SAME hlc3 sequences and engine-call patterns as
    tests/test_synthetic_cases.py so the report's synthetic check is
    consistent with the unit tests.

    Returns:
        List of CheckResult dicts (one per case), each with keys:
            check (str), category (str), passed (bool), detail (dict).
    """
    results: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Case A: monotonic ascending → UP, LOW_VOL or MID_VOL  (§16.1)
    # ------------------------------------------------------------------
    try:
        hlc3_a = [100.0, 101.0, 102.0, 103.0, 104.0, 103.5, 103.0]
        segs_a = _run_labeling(
            hlc3_a,
            theta_dc=0.005,
            min_segment_bars=2,
            q_low_boundary=1.0,
            q_high_boundary=10.0,
        )
        confirmed_a = _confirmed(segs_a)
        up_segs_a = [s for s in confirmed_a if s.direction_label == "UP"]
        low_or_mid_a = all(s.volatility_label in ("LOW_VOL", "MID_VOL") for s in up_segs_a)
        passed_a = len(up_segs_a) >= 1 and low_or_mid_a
        detail_a: dict[str, Any] = {
            "confirmed_count": int(len(confirmed_a)),
            "up_count": int(len(up_segs_a)),
            "expected_direction": "UP",
            "expected_volatility": "LOW_VOL or MID_VOL",
            "actual_labels": [
                {"direction": str(s.direction_label), "volatility": str(s.volatility_label)}
                for s in confirmed_a
            ],
        }
    except Exception as exc:
        passed_a = False
        detail_a = {"error": str(exc)}

    results.append(
        {
            "check": "synthetic_case_A",
            "category": "synthetic",
            "passed": bool(passed_a),
            "detail": detail_a,
        }
    )

    # ------------------------------------------------------------------
    # Case B: monotonic descending → DOWN, LOW_VOL or MID_VOL  (§16.1)
    # ------------------------------------------------------------------
    try:
        hlc3_b = [100.0, 100.8, 100.5, 100.2, 99.8, 99.4, 99.0, 99.3, 99.6]
        segs_b = _run_labeling(
            hlc3_b,
            theta_dc=0.003,
            min_segment_bars=2,
            q_low_boundary=1.0,
            q_high_boundary=10.0,
        )
        confirmed_b = _confirmed(segs_b)
        down_segs_b = [s for s in confirmed_b if s.direction_label == "DOWN"]
        low_or_mid_b = all(s.volatility_label in ("LOW_VOL", "MID_VOL") for s in down_segs_b)
        passed_b = len(down_segs_b) >= 1 and low_or_mid_b
        detail_b: dict[str, Any] = {
            "confirmed_count": int(len(confirmed_b)),
            "down_count": int(len(down_segs_b)),
            "expected_direction": "DOWN",
            "expected_volatility": "LOW_VOL or MID_VOL",
            "actual_labels": [
                {"direction": str(s.direction_label), "volatility": str(s.volatility_label)}
                for s in confirmed_b
            ],
        }
    except Exception as exc:
        passed_b = False
        detail_b = {"error": str(exc)}

    results.append(
        {
            "check": "synthetic_case_B",
            "category": "synthetic",
            "passed": bool(passed_b),
            "detail": detail_b,
        }
    )

    # ------------------------------------------------------------------
    # Case C: small sideways → NON_DIRECTIONAL, LOW_VOL  (§16.1)
    # Sub-test (a): direct assign_direction; sub-test (b): full pipeline
    # ------------------------------------------------------------------
    try:
        hlc3_c = [100.0, 100.05, 99.98, 100.02, 100.01]
        p_c = _log_prices(hlc3_c)
        max_amp_c = max(abs(p_c[i] - p_c[j]) for i in range(len(p_c)) for j in range(len(p_c)))
        # (a) direct assign_direction: theta_amp > max range → NON_DIRECTIONAL
        theta_amp_large_c = max_amp_c * 2.0
        amplitude_c = abs(p_c[-1] - p_c[0])
        seg_c = Segment(
            segment_id="case_c_test",
            start_bar=0,
            end_bar=4,
            confirm_bar=4,
            is_tail_unconfirmed=False,
            n_bars=5,
            log_move=float(p_c[-1] - p_c[0]),
            amplitude=float(amplitude_c),
            path_length=float(sum(abs(p_c[t] - p_c[t - 1]) for t in range(1, len(p_c)))),
            efficiency_ratio=0.0,
        )
        label_c_direct = assign_direction(seg_c, min_segment_bars=1, theta_amp=theta_amp_large_c)
        # (b) full pipeline: theta_dc > range → zero confirmed segments
        theta_dc_large_c = max_amp_c * 1.5
        segs_c_full = _run_labeling(
            hlc3_c,
            theta_dc=theta_dc_large_c,
            min_segment_bars=1,
            q_low_boundary=0.5,
            q_high_boundary=1.0,
        )
        confirmed_c = _confirmed(segs_c_full)
        # volatility proof: direct assign_volatility for a representative Case C score
        d_c = _log_returns(p_c)
        rv_c = realized_vol(d_c[1:])
        score_c = rv_per_bar(rv_c, len(hlc3_c))
        vol_label_c = assign_volatility(score_c, q_low=0.005, q_high=0.01)

        passed_c = (
            label_c_direct == "NON_DIRECTIONAL"
            and len(confirmed_c) == 0
            and vol_label_c == "LOW_VOL"
        )
        detail_c: dict[str, Any] = {
            "direct_assign_direction": str(label_c_direct),
            "full_pipeline_confirmed_count": int(len(confirmed_c)),
            "volatility_label_direct": str(vol_label_c),
            "rv_per_bar_score": float(score_c),
            "expected_direction": "NON_DIRECTIONAL",
            "expected_volatility": "LOW_VOL",
        }
    except Exception as exc:
        passed_c = False
        detail_c = {"error": str(exc)}

    results.append(
        {
            "check": "synthetic_case_C",
            "category": "synthetic",
            "passed": bool(passed_c),
            "detail": detail_c,
        }
    )

    # ------------------------------------------------------------------
    # Case D: large swings, weak net direction → NON_DIRECTIONAL, HIGH_VOL  (§16.1)
    # ------------------------------------------------------------------
    try:
        hlc3_d = [100.0, 106.0, 99.0, 105.0, 98.0, 101.0]
        p_d = _log_prices(hlc3_d)
        net_amp_d = abs(p_d[-1] - p_d[0])
        total_path_d = sum(abs(p_d[t] - p_d[t - 1]) for t in range(1, len(p_d)))
        # (a) direct assign_direction: amplitude < theta_amp → NON_DIRECTIONAL
        seg_d = Segment(
            segment_id="case_d_test",
            start_bar=0,
            end_bar=5,
            confirm_bar=5,
            is_tail_unconfirmed=False,
            n_bars=6,
            log_move=float(p_d[-1] - p_d[0]),
            amplitude=float(net_amp_d),
            path_length=float(total_path_d),
            efficiency_ratio=float(net_amp_d / total_path_d) if total_path_d > 0 else 0.0,
        )
        theta_amp_above_net_d = net_amp_d * 2.0
        label_d_direct = assign_direction(
            seg_d, min_segment_bars=1, theta_amp=theta_amp_above_net_d
        )
        # (b) full pipeline: theta_dc > max bounce → zero confirmed
        max_bounce_d = max(abs(p_d[t] - p_d[t - 1]) for t in range(1, len(p_d)))
        theta_dc_very_large_d = max_bounce_d * 1.5
        segs_d_full = _run_labeling(
            hlc3_d,
            theta_dc=theta_dc_very_large_d,
            min_segment_bars=1,
            q_low_boundary=0.0001,
            q_high_boundary=0.001,
        )
        confirmed_d = _confirmed(segs_d_full)
        # volatility proof: direct check using representative Case D score
        d_d = _log_returns(p_d)
        rv_d = realized_vol(d_d[1:])
        score_d = rv_per_bar(rv_d, len(hlc3_d))
        vol_label_d = assign_volatility(score_d, q_low=0.001, q_high=0.01)

        passed_d = (
            label_d_direct == "NON_DIRECTIONAL"
            and len(confirmed_d) == 0
            and vol_label_d == "HIGH_VOL"
        )
        detail_d: dict[str, Any] = {
            "direct_assign_direction": str(label_d_direct),
            "full_pipeline_confirmed_count": int(len(confirmed_d)),
            "volatility_label_direct": str(vol_label_d),
            "rv_per_bar_score": float(score_d),
            "expected_direction": "NON_DIRECTIONAL",
            "expected_volatility": "HIGH_VOL",
        }
    except Exception as exc:
        passed_d = False
        detail_d = {"error": str(exc)}

    results.append(
        {
            "check": "synthetic_case_D",
            "category": "synthetic",
            "passed": bool(passed_d),
            "detail": detail_d,
        }
    )

    # ------------------------------------------------------------------
    # Case E: rough zigzag, strong net UP → UP, HIGH_VOL; ER < 0.6  (§16.1)
    # Key design proof: ER is NOT a direction condition (design §8.4, §6.4)
    # ------------------------------------------------------------------
    try:
        hlc3_e = [100.0, 103.0, 101.0, 106.0, 104.0, 110.0, 108.0, 105.0]
        segs_e = _run_labeling(
            hlc3_e,
            theta_dc=0.03,
            min_segment_bars=2,
            q_low_boundary=0.0,
            q_high_boundary=0.001,
        )
        confirmed_e = _confirmed(segs_e)
        up_segs_e = [s for s in confirmed_e if s.direction_label == "UP"]
        # Pin segment [bar0, bar5] — same identity assertion as test_synthetic_cases.py
        pinned_e = [s for s in up_segs_e if s.start_bar == 0 and s.end_bar == 5]
        up_high_segs_e = [s for s in confirmed_e if s.final_label == "UP_HIGH_VOL"]
        er_low = len(pinned_e) == 1 and float(pinned_e[0].efficiency_ratio) < 0.6

        passed_e = len(up_segs_e) >= 1 and len(up_high_segs_e) >= 1 and er_low
        detail_e: dict[str, Any] = {
            "confirmed_count": int(len(confirmed_e)),
            "up_count": int(len(up_segs_e)),
            "up_high_vol_count": int(len(up_high_segs_e)),
            "pinned_seg_0_5_found": bool(len(pinned_e) == 1),
            "er_pinned": float(pinned_e[0].efficiency_ratio) if pinned_e else None,
            "expected_direction": "UP",
            "expected_volatility": "HIGH_VOL",
            "er_note": "ER < 0.6 proves ER is NOT a direction condition (§8.4, §6.4)",
        }
    except Exception as exc:
        passed_e = False
        detail_e = {"error": str(exc)}

    results.append(
        {
            "check": "synthetic_case_E",
            "category": "synthetic",
            "passed": bool(passed_e),
            "detail": detail_e,
        }
    )

    return results


# ---------------------------------------------------------------------------
# 2. Segment invariants (§11.2) — pure functions over segment_labels rows
# ---------------------------------------------------------------------------


def check_turning_point_alternation(
    segment_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Check consecutive confirmed segments alternate swing direction (§11.2).

    DC turning points strictly alternate trough/peak by construction, so every
    confirmed segment's *swing direction* — the sign of its log_move (M_j = p_end
    - p_start) — must alternate with its neighbour's. This is checked over ALL
    confirmed (non-tail) segments, NOT filtered by direction_label.

    A NON_DIRECTIONAL segment is still a genuine up- or down-swing whose swing sign
    must alternate; its label is only the §8.4 gating outcome (min_segment_bars /
    theta_amp), which does not change the swing direction. Filtering on
    direction_label would wrongly treat UP→(NON_DIRECTIONAL down-swing)→UP as a
    violation, even though the underlying swings (up, down, up) alternate correctly.

    Sign is therefore derived from raw log_move, not direction_label. An exact-zero
    log_move (a degenerate flat segment between two turning points — not observed in
    real data) is recorded as a separate note and skipped from the pairwise sign
    comparison rather than being miscounted as an alternation break.

    Args:
        segment_rows: List of segment_labels row dicts (all timeframes).

    Returns:
        CheckResult dict.
    """
    violations: list[dict[str, Any]] = []
    zero_move_segments: list[dict[str, Any]] = []
    by_tf: dict[str, list[dict[str, Any]]] = {}
    for row in segment_rows:
        # Include ALL confirmed segments (every swing must alternate); exclude tail.
        if not row.get("is_tail_unconfirmed", False):
            tf = str(row.get("timeframe", ""))
            by_tf.setdefault(tf, []).append(row)

    def _sign(row: dict[str, Any]) -> int:
        lm = row.get("log_move")
        if lm is None:
            return 0
        lm_f = float(lm)
        return 1 if lm_f > 0.0 else (-1 if lm_f < 0.0 else 0)

    for tf, rows in by_tf.items():
        sorted_rows = sorted(
            rows,
            key=_sort_key,
        )
        prev_nonzero: dict[str, Any] | None = None
        for curr in sorted_rows:
            curr_sign = _sign(curr)
            if curr_sign == 0:
                zero_move_segments.append(
                    {
                        "timeframe": tf,
                        "segment_id": str(curr.get("segment_id", "")),
                    }
                )
                continue
            if prev_nonzero is not None and _sign(prev_nonzero) == curr_sign:
                # Two consecutive non-zero swings with the same sign — impossible
                # under correct DC turning-point alternation.
                violations.append(
                    {
                        "timeframe": tf,
                        "seg_i": str(prev_nonzero.get("segment_id", "")),
                        "seg_i_plus_1": str(curr.get("segment_id", "")),
                        "prev_sign": _sign(prev_nonzero),
                        "curr_sign": curr_sign,
                    }
                )
            prev_nonzero = curr

    passed = len(violations) == 0
    return {
        "check": "segment_turning_point_alternation",
        "category": "segment_invariants",
        "passed": bool(passed),
        "detail": {
            "violation_count": int(len(violations)),
            "violations": violations[:10],
            "zero_move_segment_count": int(len(zero_move_segments)),
            "zero_move_segments": zero_move_segments[:10],
        },
    }


def check_segment_non_overlap(
    segment_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Check confirmed segments do not overlap [start_bar, end_bar) intervals (§11.2).

    Adjacent segments share the boundary bar (end_ts of seg i == start_ts of seg i+1),
    which is the correct half-open convention — not a violation.

    Args:
        segment_rows: List of segment_labels row dicts.

    Returns:
        CheckResult dict.
    """
    violations: list[dict[str, Any]] = []
    by_tf: dict[str, list[dict[str, Any]]] = {}
    for row in segment_rows:
        if not row.get("is_tail_unconfirmed", False):
            tf = str(row.get("timeframe", ""))
            by_tf.setdefault(tf, []).append(row)

    for tf, rows in by_tf.items():
        sorted_rows = sorted(
            rows,
            key=_sort_key,
        )
        for i in range(1, len(sorted_rows)):
            prev = sorted_rows[i - 1]
            curr = sorted_rows[i]
            prev_end = prev.get("end_timestamp")
            curr_start = curr.get("start_timestamp")
            # Overlap: curr starts strictly BEFORE prev ends
            if prev_end is not None and curr_start is not None and curr_start < prev_end:
                violations.append(
                    {
                        "timeframe": tf,
                        "seg_prev": str(prev.get("segment_id", "")),
                        "seg_curr": str(curr.get("segment_id", "")),
                        "prev_end": _ts_str(prev_end),
                        "curr_start": _ts_str(curr_start),
                    }
                )

    passed = len(violations) == 0
    return {
        "check": "segment_non_overlap",
        "category": "segment_invariants",
        "passed": bool(passed),
        "detail": {
            "violation_count": int(len(violations)),
            "violations": violations[:10],
        },
    }


def check_segment_coverage(
    segment_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Check no gaps between consecutive confirmed segments (§11.2).

    end_timestamp of seg i must equal start_timestamp of seg i+1.

    Args:
        segment_rows: List of segment_labels row dicts.

    Returns:
        CheckResult dict.
    """
    violations: list[dict[str, Any]] = []
    by_tf: dict[str, list[dict[str, Any]]] = {}
    for row in segment_rows:
        if not row.get("is_tail_unconfirmed", False):
            tf = str(row.get("timeframe", ""))
            by_tf.setdefault(tf, []).append(row)

    for tf, rows in by_tf.items():
        sorted_rows = sorted(
            rows,
            key=_sort_key,
        )
        for i in range(1, len(sorted_rows)):
            prev = sorted_rows[i - 1]
            curr = sorted_rows[i]
            prev_end = prev.get("end_timestamp")
            curr_start = curr.get("start_timestamp")
            if prev_end is not None and curr_start is not None and prev_end != curr_start:
                violations.append(
                    {
                        "timeframe": tf,
                        "seg_prev": str(prev.get("segment_id", "")),
                        "seg_curr": str(curr.get("segment_id", "")),
                        "prev_end": _ts_str(prev_end),
                        "curr_start": _ts_str(curr_start),
                    }
                )

    passed = len(violations) == 0
    return {
        "check": "segment_coverage_no_gaps",
        "category": "segment_invariants",
        "passed": bool(passed),
        "detail": {
            "violation_count": int(len(violations)),
            "violations": violations[:10],
        },
    }


def check_confirm_bar_consistency(
    segment_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Check confirm/tail mutual-exclusion and temporal ordering (§11.2).

    Ground truth (verified against run_id=13, all 11,483 confirmed segments):
      confirm_timestamp > end_timestamp >= start_timestamp for ALL confirmed segments.
    Valid state: start_ts <= end_ts <= confirm_ts, lag_bars >= 0.

    Mirrors DB ck_segment_labels_confirm:
    - tail rows: confirm_timestamp, lag_bars, lag_move, capturable_amplitude,
      capturable_ratio, direction_label, volatility_label, final_label all NULL.
    - confirmed rows: all of the above NOT NULL;
      confirm_timestamp >= end_timestamp >= start_timestamp; lag_bars >= 0.

    Args:
        segment_rows: List of segment_labels row dicts.

    Returns:
        CheckResult dict.
    """
    # Full set of fields governed by DB constraint ck_segment_labels_confirm
    _TAIL_NULL_FIELDS = (
        "confirm_timestamp",
        "lag_bars",
        "lag_move",
        "capturable_amplitude",
        "capturable_ratio",
        "direction_label",
        "volatility_label",
        "final_label",
    )

    violations: list[dict[str, Any]] = []
    for row in segment_rows:
        is_tail = bool(row.get("is_tail_unconfirmed", False))
        seg_id = str(row.get("segment_id", ""))
        if is_tail:
            for field in _TAIL_NULL_FIELDS:
                if row.get(field) is not None:
                    violations.append({"segment_id": seg_id, "issue": f"tail has non-NULL {field}"})
        else:
            for field in _TAIL_NULL_FIELDS:
                if row.get(field) is None:
                    violations.append({"segment_id": seg_id, "issue": f"confirmed missing {field}"})
            start_ts = row.get("start_timestamp")
            end_ts = row.get("end_timestamp")
            confirm_ts = row.get("confirm_timestamp")
            # Impossible / negative-lag case: confirm before end
            if confirm_ts is not None and end_ts is not None and confirm_ts < end_ts:
                violations.append(
                    {
                        "segment_id": seg_id,
                        "issue": "confirm_timestamp < end_timestamp (negative lag impossible)",
                        "end_timestamp": _ts_str(end_ts),
                        "confirm_timestamp": _ts_str(confirm_ts),
                    }
                )
            # confirm before start is always impossible
            if confirm_ts is not None and start_ts is not None and confirm_ts < start_ts:
                violations.append(
                    {
                        "segment_id": seg_id,
                        "issue": "confirm_timestamp < start_timestamp",
                        "start_timestamp": _ts_str(start_ts),
                        "confirm_timestamp": _ts_str(confirm_ts),
                    }
                )
            # Negative lag_bars
            lag_bars = row.get("lag_bars")
            if lag_bars is not None and int(lag_bars) < 0:
                violations.append(
                    {
                        "segment_id": seg_id,
                        "issue": "lag_bars < 0",
                        "lag_bars": int(lag_bars),
                    }
                )

    passed = len(violations) == 0
    return {
        "check": "segment_confirm_bar_consistency",
        "category": "segment_invariants",
        "passed": bool(passed),
        "detail": {
            "violation_count": int(len(violations)),
            "violations": violations[:20],
        },
    }


# ---------------------------------------------------------------------------
# 3. Label invariants (§11.3) — pure functions over bar_labels + segment_labels
# ---------------------------------------------------------------------------


def check_bar_label_domain(
    bar_label_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Check every bar has final_label in the 9-value canonical set (§11.3).

    Args:
        bar_label_rows: List of bar_labels row dicts.

    Returns:
        CheckResult dict.
    """
    violations: list[dict[str, Any]] = []
    for row in bar_label_rows:
        lbl = row.get("final_label")
        if lbl not in _CANONICAL_LABELS:
            violations.append(
                {
                    "final_label": str(lbl),
                    "open_time": _ts_str(row.get("open_time")),
                    "timeframe": str(row.get("timeframe", "")),
                }
            )
            if len(violations) >= 20:
                break

    passed = len(violations) == 0
    return {
        "check": "bar_label_domain_9values",
        "category": "label_invariants",
        "passed": bool(passed),
        "detail": {
            "total_bars_checked": int(len(bar_label_rows)),
            "violation_count": int(len(violations)),
            "violations": violations,
        },
    }


def check_bar_label_direction_vol_consistency(
    bar_label_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Check final_label == direction_label + '_' + volatility_label for all bars (§11.3).

    Mirrors DB ck_bar_labels_final_label_consistency. Confirms diagnostics do
    NOT affect labels (§15.1, §19-10).

    Args:
        bar_label_rows: List of bar_labels row dicts.

    Returns:
        CheckResult dict.
    """
    violations: list[dict[str, Any]] = []
    for row in bar_label_rows:
        d_lbl = row.get("direction_label")
        v_lbl = row.get("volatility_label")
        f_lbl = row.get("final_label")
        expected = f"{d_lbl}_{v_lbl}" if d_lbl is not None and v_lbl is not None else None
        if expected != f_lbl:
            violations.append(
                {
                    "direction_label": str(d_lbl),
                    "volatility_label": str(v_lbl),
                    "final_label": str(f_lbl),
                    "expected": str(expected),
                    "open_time": _ts_str(row.get("open_time")),
                }
            )
            if len(violations) >= 20:
                break

    passed = len(violations) == 0
    return {
        "check": "bar_label_direction_vol_consistency",
        "category": "label_invariants",
        "passed": bool(passed),
        "detail": {
            "total_bars_checked": int(len(bar_label_rows)),
            "violation_count": int(len(violations)),
            "violations": violations,
        },
    }


def check_segment_label_consistency(
    segment_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Check final_label == direction + '_' + volatility for confirmed segments (§11.3).

    Mirrors DB ck_segment_labels_final_label_consistency.

    Args:
        segment_rows: List of segment_labels row dicts.

    Returns:
        CheckResult dict.
    """
    violations: list[dict[str, Any]] = []
    for row in segment_rows:
        if row.get("is_tail_unconfirmed", False):
            continue
        d_lbl = row.get("direction_label")
        v_lbl = row.get("volatility_label")
        f_lbl = row.get("final_label")
        if d_lbl is None or v_lbl is None or f_lbl is None:
            violations.append(
                {
                    "segment_id": str(row.get("segment_id", "")),
                    "issue": "confirmed segment has NULL label field",
                    "direction_label": str(d_lbl),
                    "volatility_label": str(v_lbl),
                    "final_label": str(f_lbl),
                }
            )
        elif f"{d_lbl}_{v_lbl}" != f_lbl:
            violations.append(
                {
                    "segment_id": str(row.get("segment_id", "")),
                    "issue": "final_label != direction+_+volatility",
                    "direction_label": str(d_lbl),
                    "volatility_label": str(v_lbl),
                    "final_label": str(f_lbl),
                    "expected": f"{d_lbl}_{v_lbl}",
                }
            )
        if len(violations) >= 20:
            break

    passed = len(violations) == 0
    return {
        "check": "segment_label_consistency",
        "category": "label_invariants",
        "passed": bool(passed),
        "detail": {
            "violation_count": int(len(violations)),
            "violations": violations,
        },
    }


# ---------------------------------------------------------------------------
# 4. Numeric invariants (§11.4 + §10.3) — pure functions
# ---------------------------------------------------------------------------


def check_numeric_ranges_segments(
    segment_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Check §11.4 numeric ranges on segment_labels rows.

    Mirrors DB ck_segment_labels_ranges:
      - realized_volatility >= 0
      - realized_volatility_per_bar >= 0
      - 0 <= efficiency_ratio <= 1
      - capturable_ratio IS NULL OR 0 <= capturable_ratio <= 1
      - 0 <= max_jump_share <= 1
      - 0 <= downside_vol_share <= 1

    Args:
        segment_rows: List of segment_labels row dicts.

    Returns:
        CheckResult dict.
    """
    violations: list[dict[str, Any]] = []

    def _ge(row: dict[str, Any], field: str, lo: float) -> None:
        val = row.get(field)
        if val is None:
            return
        if float(val) < lo:
            violations.append(
                {
                    "segment_id": str(row.get("segment_id", "")),
                    "field": field,
                    "value": float(val),
                    "allowed_min": float(lo),
                }
            )

    def _between(row: dict[str, Any], field: str, lo: float, hi: float) -> None:
        val = row.get(field)
        if val is None:
            return  # nullable — skip
        fv = float(val)
        if not (lo <= fv <= hi):
            violations.append(
                {
                    "segment_id": str(row.get("segment_id", "")),
                    "field": field,
                    "value": float(fv),
                    "allowed_range": [float(lo), float(hi)],
                }
            )

    for row in segment_rows:
        _ge(row, "realized_volatility", 0.0)
        _ge(row, "realized_volatility_per_bar", 0.0)
        _between(row, "efficiency_ratio", 0.0, 1.0)
        _between(row, "capturable_ratio", 0.0, 1.0)  # nullable
        _between(row, "max_jump_share", 0.0, 1.0)
        _between(row, "downside_vol_share", 0.0, 1.0)
        if len(violations) >= 50:
            break

    passed = len(violations) == 0
    return {
        "check": "numeric_ranges_segments",
        "category": "numeric_invariants",
        "passed": bool(passed),
        "detail": {
            "segments_checked": int(len(segment_rows)),
            "violation_count": int(len(violations)),
            "violations": violations[:20],
        },
    }


def check_numeric_ranges_params(
    param_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Check §11.4 numeric range constraints on labeling_run_params rows.

    Fields:
      - Q_low <= Q_high
      - theta_dc > 0
      - theta_amp > 0

    Args:
        param_rows: List of labeling_run_params row dicts.

    Returns:
        CheckResult dict.
    """
    violations: list[dict[str, Any]] = []
    for row in param_rows:
        tf = str(row.get("timeframe", ""))
        q_low = row.get("q_low")
        q_high = row.get("q_high")
        theta_dc = row.get("theta_dc")
        theta_amp = row.get("theta_amp")

        if q_low is not None and q_high is not None:
            if float(q_low) > float(q_high):
                violations.append(
                    {
                        "timeframe": tf,
                        "issue": "Q_low > Q_high",
                        "q_low": float(q_low),
                        "q_high": float(q_high),
                    }
                )
        if theta_dc is not None and float(theta_dc) <= 0.0:
            violations.append(
                {
                    "timeframe": tf,
                    "issue": "theta_dc <= 0",
                    "theta_dc": float(theta_dc),
                }
            )
        if theta_amp is not None and float(theta_amp) <= 0.0:
            violations.append(
                {
                    "timeframe": tf,
                    "issue": "theta_amp <= 0",
                    "theta_amp": float(theta_amp),
                }
            )

    passed = len(violations) == 0
    return {
        "check": "numeric_ranges_params",
        "category": "numeric_invariants",
        "passed": bool(passed),
        "detail": {
            "param_rows_checked": int(len(param_rows)),
            "violation_count": int(len(violations)),
            "violations": violations,
        },
    }


# ---------------------------------------------------------------------------
# 5. Timeframe independence (§11.3)
# ---------------------------------------------------------------------------


def check_timeframe_independence(
    param_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Check labeling_run_params has exactly one row per timeframe for 1m and 5m (§11.3).

    Confirms 1m and 5m have independently computed theta/quantiles
    (uk_labeling_run_params_run_tf UNIQUE(run_id, timeframe) at DB level;
    re-verified at application layer here).

    Args:
        param_rows: List of labeling_run_params row dicts for a single run_id.

    Returns:
        CheckResult dict.
    """
    timeframes_present = {str(row.get("timeframe", "")) for row in param_rows}
    required = {"1m", "5m"}
    missing = required - timeframes_present
    extra = timeframes_present - required
    count_by_tf: dict[str, int] = {}
    for row in param_rows:
        tf = str(row.get("timeframe", ""))
        count_by_tf[tf] = count_by_tf.get(tf, 0) + 1
    duplicates = {tf: cnt for tf, cnt in count_by_tf.items() if cnt > 1}

    passed = len(missing) == 0 and len(duplicates) == 0
    return {
        "check": "timeframe_independence",
        "category": "label_invariants",
        "passed": bool(passed),
        "detail": {
            "timeframes_present": sorted(timeframes_present),
            "required": sorted(required),
            "missing_timeframes": sorted(missing),
            "extra_timeframes": sorted(extra),
            "duplicates": {k: int(v) for k, v in duplicates.items()},
        },
    }


# ---------------------------------------------------------------------------
# 6. §15 PASS/FAIL aggregation
# ---------------------------------------------------------------------------


def aggregate_pass_fail(checks: list[dict[str, Any]]) -> bool:
    """Return True iff all checks passed (§15.1).

    Args:
        checks: List of CheckResult dicts each with a 'passed' key.

    Returns:
        True iff every check has passed=True.
    """
    return all(bool(c["passed"]) for c in checks)


def build_payload(
    run_id: int,
    checks: list[dict[str, Any]],
    generated_for: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the JSONB-serialisable payload for labeling_reports.

    All values are plain Python (float/int/str/bool/list/dict).
    No numpy scalars. No datetime objects (ISO strings only).

    Args:
        run_id: labeling_runs.id.
        checks: List of CheckResult dicts.
        generated_for: Optional run metadata dict (ISO-string timestamps only).

    Returns:
        Fully serialisable payload dict.
    """
    total = len(checks)
    n_passed = sum(1 for c in checks if bool(c["passed"]))
    n_failed = total - n_passed
    return {
        "run_id": int(run_id),
        "generated_for": generated_for or {},
        "summary": {
            "total": int(total),
            "passed": int(n_passed),
            "failed": int(n_failed),
        },
        "checks": checks,
    }


# ---------------------------------------------------------------------------
# 7. Writer — mirrors loader.py connection/transaction pattern
# ---------------------------------------------------------------------------


def store_validation_report(
    run_id: int,
    passed: bool,
    payload: dict[str, Any],
    dsn: str,
) -> int:
    """Insert one labeling_reports row (report_type='validation').

    Connection/transaction pattern mirrors persistence/loader.py:
    psycopg3 connect, single transaction, commit on success,
    rollback + re-raise on failure.

    Args:
        run_id: labeling_runs.id for this run.
        passed: Overall PASS/FAIL result.
        payload: JSONB-serialisable dict (build_payload output).
        dsn: PostgreSQL DSN string.

    Returns:
        New labeling_reports.id (int).

    Raises:
        psycopg.Error: On DB failure (rollback guaranteed).
    """
    import psycopg
    from psycopg.types.json import Jsonb

    conn: psycopg.Connection[Any] = psycopg.connect(dsn)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO labeling_reports (run_id, report_type, passed, payload)
                VALUES (%s, 'validation', %s, %s)
                RETURNING id
                """,
                (run_id, bool(passed), Jsonb(payload)),
            )
            row = cur.fetchone()
        conn.commit()
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        raise
    finally:
        conn.close()

    assert row is not None
    return int(row[0])


# ---------------------------------------------------------------------------
# DB reader helpers (read-only; column names from migrations/001_init.sql)
#
# Non-blocking-3: All four reads share ONE connection in a single REPEATABLE READ
# transaction to guarantee snapshot consistency and reduce connect overhead.
# ---------------------------------------------------------------------------


def _fetch_run_meta(cur: Any, run_id: int) -> dict[str, Any]:
    """Execute the labeling_runs SELECT on an open cursor."""
    cur.execute(
        """
        SELECT id, method_version, symbol, market,
               period_start_utc, period_end_utc,
               price_field, git_commit, run_status,
               completed_at, created_at
        FROM labeling_runs
        WHERE id = %s
        """,
        (run_id,),
    )
    desc = cur.description
    row = cur.fetchone()
    if row is None:
        raise ValueError(f"No labeling_runs row found for run_id={run_id}")
    assert desc is not None
    cols = [d[0] for d in desc]
    return dict(zip(cols, row))


def _fetch_segment_rows(cur: Any, run_id: int) -> list[dict[str, Any]]:
    """Execute the segment_labels SELECT on an open cursor."""
    cur.execute(
        """
        SELECT
            segment_id,
            timeframe::text,
            start_timestamp,
            end_timestamp,
            is_tail_unconfirmed,
            confirm_timestamp,
            lag_bars,
            lag_move,
            capturable_amplitude,
            capturable_ratio,
            log_move,
            amplitude,
            path_length,
            efficiency_ratio,
            realized_volatility,
            realized_volatility_per_bar,
            max_jump_share,
            downside_vol_share,
            direction_label::text,
            volatility_label::text,
            final_label::text
        FROM segment_labels
        WHERE run_id = %s
        ORDER BY timeframe, start_timestamp, segment_id
        """,
        (run_id,),
    )
    desc = cur.description
    rows_raw = cur.fetchall()
    assert desc is not None
    cols = [d[0] for d in desc]
    return [dict(zip(cols, r)) for r in rows_raw]


def _fetch_bar_label_rows(cur: Any, run_id: int) -> list[dict[str, Any]]:
    """Execute the bar_labels SELECT on an open cursor."""
    cur.execute(
        """
        SELECT
            timeframe::text,
            open_time,
            direction_label::text,
            volatility_label::text,
            final_label::text
        FROM bar_labels
        WHERE run_id = %s
        ORDER BY timeframe, open_time
        """,
        (run_id,),
    )
    desc = cur.description
    rows_raw = cur.fetchall()
    assert desc is not None
    cols = [d[0] for d in desc]
    return [dict(zip(cols, r)) for r in rows_raw]


def _fetch_param_rows(cur: Any, run_id: int) -> list[dict[str, Any]]:
    """Execute the labeling_run_params SELECT on an open cursor."""
    cur.execute(
        """
        SELECT
            timeframe::text,
            q_dc,
            k_dc,
            min_segment_bars,
            theta_dc,
            theta_amp,
            q_low,
            q_high,
            taker_fee_rate,
            slippage_rate_estimate
        FROM labeling_run_params
        WHERE run_id = %s
        ORDER BY timeframe
        """,
        (run_id,),
    )
    desc = cur.description
    rows_raw = cur.fetchall()
    assert desc is not None
    cols = [d[0] for d in desc]
    return [dict(zip(cols, r)) for r in rows_raw]


def _read_all_run_data(
    run_id: int, dsn: str
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Read run_meta, segment_rows, bar_label_rows, param_rows in one snapshot.

    Opens a single connection at REPEATABLE READ isolation and executes all
    four queries within the same transaction, matching loader.py's pattern.

    Returns:
        (run_meta, segment_rows, bar_label_rows, param_rows)

    Raises:
        ValueError: If no labeling_runs row for run_id.
        psycopg.Error: On DB failure.
    """
    import psycopg

    conn: psycopg.Connection[Any] = psycopg.connect(dsn)
    try:
        conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
        with conn.cursor() as cur:
            run_meta = _fetch_run_meta(cur, run_id)
            segment_rows = _fetch_segment_rows(cur, run_id)
            bar_label_rows = _fetch_bar_label_rows(cur, run_id)
            param_rows = _fetch_param_rows(cur, run_id)
        conn.rollback()  # read-only: explicit rollback releases the snapshot cleanly
    finally:
        conn.close()

    return run_meta, segment_rows, bar_label_rows, param_rows


def _serialize_meta(meta: dict[str, Any]) -> dict[str, Any]:
    """Convert datetime/non-JSON-native values to JSON-safe types.

    Nit-2: Uses default=str fallback logic so Decimal, enum wrappers, etc.
    degrade to string rather than raising at serialisation time.
    """
    result: dict[str, Any] = {}
    for k, v in meta.items():
        if isinstance(v, datetime):
            result[k] = v.isoformat()
        else:
            try:
                import json as _json

                _json.dumps(v)
                result[k] = v
            except (TypeError, ValueError):
                result[k] = str(v)
    return result


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_validation_report(
    run_id: int,
    dsn: str,
    *,
    store: bool = True,
) -> dict[str, Any]:
    """Validate a persisted labeling run and (optionally) persist the report.

    Reads segment_labels / bar_labels / labeling_run_params for `run_id` (both
    timeframes) back from the DB in a single REPEATABLE READ snapshot, runs all
    checks, judges PASS/FAIL, and — if store=True — writes one labeling_reports
    row (report_type='validation').

    Args:
        run_id: labeling_runs.id for the run being validated.
        dsn: PostgreSQL DSN string (from env var REGIME_BENCHMARK_DB_URL).
        store: If True (default), persist result to labeling_reports.

    Returns:
        {"passed": bool, "checks": list[CheckResult-dict], "payload": dict}.

    Raises:
        ValueError: If run_id not found in labeling_runs.
        psycopg.Error: On DB failure.
    """
    # --- Read persisted data from DB (single REPEATABLE READ snapshot) ---
    run_meta, segment_rows, bar_label_rows, param_rows = _read_all_run_data(run_id, dsn)

    checks: list[dict[str, Any]] = []

    # --- BLOCKING-1: Data-presence guards — empty data FAILS rather than vacuously passes ---
    if not segment_rows:
        checks.append(
            {
                "check": "segment_data_present",
                "category": "data_integrity",
                "passed": False,
                "detail": {"issue": "No segment_labels rows for run_id", "run_id": run_id},
            }
        )
    if not bar_label_rows:
        checks.append(
            {
                "check": "bar_label_data_present",
                "category": "data_integrity",
                "passed": False,
                "detail": {"issue": "No bar_labels rows for run_id", "run_id": run_id},
            }
        )

    # --- 1. Synthetic Case A~E (run-independent; validates labeler code) ---
    checks.extend(run_synthetic_cases())

    # --- 2. Segment invariants (§11.2) ---
    checks.append(check_turning_point_alternation(segment_rows))
    checks.append(check_segment_non_overlap(segment_rows))
    checks.append(check_segment_coverage(segment_rows))
    checks.append(check_confirm_bar_consistency(segment_rows))

    # --- 3. Label invariants (§11.3) ---
    checks.append(check_bar_label_domain(bar_label_rows))
    checks.append(check_bar_label_direction_vol_consistency(bar_label_rows))
    checks.append(check_segment_label_consistency(segment_rows))

    # --- 4. Numeric invariants (§11.4 + §10.3) ---
    checks.append(check_numeric_ranges_segments(segment_rows))
    checks.append(check_numeric_ranges_params(param_rows))

    # --- 5. Timeframe independence ---
    checks.append(check_timeframe_independence(param_rows))

    # --- 6. §15 PASS/FAIL ---
    passed = aggregate_pass_fail(checks)

    generated_for = _serialize_meta(run_meta)
    payload = build_payload(run_id, checks, generated_for)
    payload["passed"] = bool(passed)

    # --- 7. Persist (optional) ---
    if store:
        store_validation_report(run_id, passed, payload, dsn)

    return {
        "passed": bool(passed),
        "checks": checks,
        "payload": payload,
    }
