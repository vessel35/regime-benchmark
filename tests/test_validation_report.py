"""Tests for validation/report.py — M6 PASS/FAIL report.

Unit tests (no DB):
  - run_synthetic_cases() all-pass on canonical fixtures
  - Each invariant pure-function returns passed=False on crafted violating input
    and passed=True on clean input
  - PASS/FAIL aggregation correctness

DB integration tests (skipped if REGIME_BENCHMARK_DB_URL unset):
  - run_validation_report(run_id, dsn, store=True) against a small synthetic run
  - joined_labels_1m_5m 5-minute bucket alignment check
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg
import pytest

from regime_benchmark.validation.report import (
    aggregate_pass_fail,
    build_payload,
    check_bar_label_direction_vol_consistency,
    check_bar_label_domain,
    check_confirm_bar_consistency,
    check_numeric_ranges_params,
    check_numeric_ranges_segments,
    check_segment_coverage,
    check_segment_label_consistency,
    check_segment_non_overlap,
    check_timeframe_independence,
    check_turning_point_alternation,
    run_synthetic_cases,
    run_validation_report,
)

# ---------------------------------------------------------------------------
# DB skip guard (same pattern as test_persistence_roundtrip.py)
# ---------------------------------------------------------------------------

_DB_URL = os.environ.get("REGIME_BENCHMARK_DB_URL", "")
if not _DB_URL:
    _env_paths = [
        Path(__file__).parent.parent / ".env",
        Path(__file__).parent.parent.parent.parent.parent / ".env",
    ]
    for _env_path in _env_paths:
        if _env_path.exists():
            for line in _env_path.read_text().splitlines():
                line = line.strip()
                if line.startswith("REGIME_BENCHMARK_DB_URL=") and not line.startswith("#"):
                    _DB_URL = line.split("=", 1)[1].strip()
                    os.environ["REGIME_BENCHMARK_DB_URL"] = _DB_URL
                    break
        if _DB_URL:
            break


def _db_reachable() -> bool:
    """Return True if _DB_URL is set AND the DB is reachable."""
    if not _DB_URL:
        return False
    try:
        conn = psycopg.connect(_DB_URL)
        conn.close()
        return True
    except Exception:
        return False


_HAS_DB = _db_reachable()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_T0 = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
_T1 = datetime(2024, 1, 1, 0, 1, 0, tzinfo=timezone.utc)
_T2 = datetime(2024, 1, 1, 0, 2, 0, tzinfo=timezone.utc)
_T3 = datetime(2024, 1, 1, 0, 3, 0, tzinfo=timezone.utc)
_T4 = datetime(2024, 1, 1, 0, 4, 0, tzinfo=timezone.utc)
_T5 = datetime(2024, 1, 1, 0, 5, 0, tzinfo=timezone.utc)


def _make_confirmed_seg(
    segment_id: str,
    start_ts: datetime,
    end_ts: datetime,
    confirm_ts: datetime,
    log_move: float = 0.01,
    timeframe: str = "1m",
    direction_label: str = "UP",
    volatility_label: str = "LOW_VOL",
    final_label: str = "UP_LOW_VOL",
    efficiency_ratio: float = 0.8,
    realized_volatility: float = 0.001,
    realized_volatility_per_bar: float = 0.0005,
    max_jump_share: float = 0.1,
    downside_vol_share: float = 0.3,
    capturable_ratio: float | None = 0.9,
) -> dict[str, Any]:
    return {
        "segment_id": segment_id,
        "timeframe": timeframe,
        "start_timestamp": start_ts,
        "end_timestamp": end_ts,
        "is_tail_unconfirmed": False,
        "confirm_timestamp": confirm_ts,
        "lag_bars": 1,
        "lag_move": 0.001,
        "capturable_amplitude": 0.01,
        "capturable_ratio": capturable_ratio,
        "log_move": log_move,
        "amplitude": abs(log_move),
        "path_length": abs(log_move) * 1.2,
        "efficiency_ratio": efficiency_ratio,
        "realized_volatility": realized_volatility,
        "realized_volatility_per_bar": realized_volatility_per_bar,
        "max_jump_share": max_jump_share,
        "downside_vol_share": downside_vol_share,
        "direction_label": direction_label,
        "volatility_label": volatility_label,
        "final_label": final_label,
    }


def _make_tail_seg(
    segment_id: str,
    start_ts: datetime,
    end_ts: datetime,
    timeframe: str = "1m",
) -> dict[str, Any]:
    return {
        "segment_id": segment_id,
        "timeframe": timeframe,
        "start_timestamp": start_ts,
        "end_timestamp": end_ts,
        "is_tail_unconfirmed": True,
        "confirm_timestamp": None,
        "lag_bars": None,
        "lag_move": None,
        "capturable_amplitude": None,
        "capturable_ratio": None,
        "log_move": 0.001,
        "amplitude": 0.001,
        "path_length": 0.001,
        "efficiency_ratio": 1.0,
        "realized_volatility": 0.001,
        "realized_volatility_per_bar": 0.0005,
        "max_jump_share": 0.05,
        "downside_vol_share": 0.2,
        "direction_label": None,
        "volatility_label": None,
        "final_label": None,
    }


def _make_bar(
    open_time: datetime,
    direction_label: str = "UP",
    volatility_label: str = "LOW_VOL",
    final_label: str = "UP_LOW_VOL",
    timeframe: str = "1m",
) -> dict[str, Any]:
    return {
        "timeframe": timeframe,
        "open_time": open_time,
        "direction_label": direction_label,
        "volatility_label": volatility_label,
        "final_label": final_label,
    }


def _make_params(
    timeframe: str = "1m",
    q_low: float = 0.33,
    q_high: float = 0.66,
    theta_dc: float = 0.01,
    theta_amp: float = 0.01,
) -> dict[str, Any]:
    return {
        "timeframe": timeframe,
        "q_dc": 0.80,
        "k_dc": 2.0,
        "min_segment_bars": 2,
        "theta_dc": theta_dc,
        "theta_amp": theta_amp,
        "q_low": q_low,
        "q_high": q_high,
        "taker_fee_rate": 0.0004,
        "slippage_rate_estimate": 0.0002,
    }


# ===========================================================================
# Unit tests — no DB
# ===========================================================================

# ---------------------------------------------------------------------------
# 1. run_synthetic_cases — all five cases must pass on canonical fixtures
# ---------------------------------------------------------------------------


class TestRunSyntheticCases:
    def test_returns_five_checks(self) -> None:
        results = run_synthetic_cases()
        assert len(results) == 5

    def test_all_pass(self) -> None:
        results = run_synthetic_cases()
        failed = [r for r in results if not r["passed"]]
        assert failed == [], f"Failing synthetic cases: {[r['check'] for r in failed]}"

    def test_case_a_present_and_passed(self) -> None:
        results = run_synthetic_cases()
        case_a = next((r for r in results if r["check"] == "synthetic_case_A"), None)
        assert case_a is not None
        assert case_a["passed"] is True

    def test_case_b_present_and_passed(self) -> None:
        results = run_synthetic_cases()
        case_b = next((r for r in results if r["check"] == "synthetic_case_B"), None)
        assert case_b is not None
        assert case_b["passed"] is True

    def test_case_c_present_and_passed(self) -> None:
        results = run_synthetic_cases()
        case_c = next((r for r in results if r["check"] == "synthetic_case_C"), None)
        assert case_c is not None
        assert case_c["passed"] is True

    def test_case_d_present_and_passed(self) -> None:
        results = run_synthetic_cases()
        case_d = next((r for r in results if r["check"] == "synthetic_case_D"), None)
        assert case_d is not None
        assert case_d["passed"] is True

    def test_case_e_present_and_passed(self) -> None:
        results = run_synthetic_cases()
        case_e = next((r for r in results if r["check"] == "synthetic_case_E"), None)
        assert case_e is not None
        assert case_e["passed"] is True

    def test_categories_are_synthetic(self) -> None:
        results = run_synthetic_cases()
        for r in results:
            assert r["category"] == "synthetic", f"{r['check']} has wrong category"

    def test_result_has_required_keys(self) -> None:
        results = run_synthetic_cases()
        for r in results:
            assert "check" in r
            assert "category" in r
            assert "passed" in r
            assert "detail" in r

    def test_passed_values_are_python_bool(self) -> None:
        """Ensure no numpy bools leak into the result."""
        results = run_synthetic_cases()
        for r in results:
            assert type(r["passed"]) is bool, (
                f"{r['check']}: passed={r['passed']!r} is not python bool"
            )


# ---------------------------------------------------------------------------
# 2. check_turning_point_alternation
# ---------------------------------------------------------------------------


class TestTurningPointAlternation:
    def _clean_segs(self) -> list[dict[str, Any]]:
        """Three confirmed segments with alternating direction_labels (UP/DOWN/UP)."""
        return [
            _make_confirmed_seg(
                "s1", _T0, _T1, _T2, log_move=0.02, direction_label="UP", final_label="UP_LOW_VOL"
            ),
            _make_confirmed_seg(
                "s2",
                _T1,
                _T2,
                _T3,
                log_move=-0.015,
                direction_label="DOWN",
                final_label="DOWN_LOW_VOL",
            ),
            _make_confirmed_seg(
                "s3", _T2, _T3, _T4, log_move=0.01, direction_label="UP", final_label="UP_LOW_VOL"
            ),
        ]

    def _violating_segs(self) -> list[dict[str, Any]]:
        """Two consecutive DIRECTIONAL segments with the same direction — violation."""
        return [
            _make_confirmed_seg(
                "s1", _T0, _T1, _T2, log_move=0.02, direction_label="UP", final_label="UP_LOW_VOL"
            ),
            _make_confirmed_seg(
                "s2", _T1, _T2, _T3, log_move=0.015, direction_label="UP", final_label="UP_LOW_VOL"
            ),  # same direction → violation
        ]

    def test_clean_passes(self) -> None:
        result = check_turning_point_alternation(self._clean_segs())
        assert result["passed"] is True
        assert result["detail"]["violation_count"] == 0

    def test_violation_fails(self) -> None:
        result = check_turning_point_alternation(self._violating_segs())
        assert result["passed"] is False
        assert result["detail"]["violation_count"] >= 1

    def test_tail_segments_ignored(self) -> None:
        """After tail exclusion, s1(UP) and s3(UP) are adjacent — same direction is a violation."""
        segs = [
            _make_confirmed_seg(
                "s1", _T0, _T1, _T2, log_move=0.02, direction_label="UP", final_label="UP_LOW_VOL"
            ),
            _make_tail_seg("tail", _T1, _T2),
            _make_confirmed_seg(
                "s3", _T2, _T3, _T4, log_move=0.01, direction_label="UP", final_label="UP_LOW_VOL"
            ),
        ]
        result = check_turning_point_alternation(segs)
        assert result["passed"] is False

    def test_non_directional_swing_participates_in_alternation(self) -> None:
        """Regression (run_id=13): UP → NON_DIRECTIONAL *down-swing* → UP must PASS.

        A NON_DIRECTIONAL segment is still a genuine swing; its sign alternates with
        its neighbours. The labels are UP/ND/UP but the raw swing signs are +,-,+ —
        a valid alternation. Filtering on direction_label (the old buggy behaviour)
        wrongly treated this as UP→UP and false-failed ~1121 real segments.
        """
        segs = [
            _make_confirmed_seg(
                "s1", _T0, _T1, _T2, log_move=0.02, direction_label="UP", final_label="UP_LOW_VOL"
            ),
            _make_confirmed_seg(
                "nd",
                _T1,
                _T2,
                _T3,
                log_move=-0.003,  # genuine down-swing
                direction_label="NON_DIRECTIONAL",
                volatility_label="LOW_VOL",
                final_label="NON_DIRECTIONAL_LOW_VOL",
            ),
            _make_confirmed_seg(
                "s3", _T2, _T3, _T4, log_move=0.01, direction_label="UP", final_label="UP_LOW_VOL"
            ),
        ]
        result = check_turning_point_alternation(segs)
        assert result["passed"] is True
        assert result["detail"]["violation_count"] == 0

    def test_same_sign_across_nd_label_fails(self) -> None:
        """Two consecutive same-sign swings (regardless of label) is a real violation.

        +0.02, +0.001 (ND), ... — two adjacent positive swings cannot occur under
        correct DC turning-point alternation, so this must FAIL even though the
        middle segment is labelled NON_DIRECTIONAL.
        """
        segs = [
            _make_confirmed_seg(
                "s1", _T0, _T1, _T2, log_move=0.02, direction_label="UP", final_label="UP_LOW_VOL"
            ),
            _make_confirmed_seg(
                "nd",
                _T1,
                _T2,
                _T3,
                log_move=0.001,  # same (+) sign → impossible
                direction_label="NON_DIRECTIONAL",
                volatility_label="LOW_VOL",
                final_label="NON_DIRECTIONAL_LOW_VOL",
            ),
        ]
        result = check_turning_point_alternation(segs)
        assert result["passed"] is False
        assert result["detail"]["violation_count"] >= 1

    def test_zero_move_segment_skipped_not_miscounted(self) -> None:
        """An exact-zero log_move segment is recorded and skipped, not flagged as a break.

        +0.02, 0.0, -0.01 — the zero segment is skipped, the surrounding +/- pair
        still alternates, so no violation; the zero is reported in the detail.
        """
        segs = [
            _make_confirmed_seg(
                "s1", _T0, _T1, _T2, log_move=0.02, direction_label="UP", final_label="UP_LOW_VOL"
            ),
            _make_confirmed_seg(
                "z",
                _T1,
                _T2,
                _T3,
                log_move=0.0,
                direction_label="NON_DIRECTIONAL",
                volatility_label="LOW_VOL",
                final_label="NON_DIRECTIONAL_LOW_VOL",
            ),
            _make_confirmed_seg(
                "s3",
                _T2,
                _T3,
                _T4,
                log_move=-0.01,
                direction_label="DOWN",
                final_label="DOWN_LOW_VOL",
            ),
        ]
        result = check_turning_point_alternation(segs)
        assert result["passed"] is True
        assert result["detail"]["zero_move_segment_count"] == 1

    def test_zero_carry_through_violation(self) -> None:
        """+0.02, 0.0, +0.01 — the zero is skipped but prev_nonzero carries the first +,
        so the two positive swings are still compared and must FAIL (locks the
        prev_nonzero pointer contract)."""
        segs = [
            _make_confirmed_seg("s1", _T0, _T1, _T2, log_move=0.02, direction_label="UP",
                                final_label="UP_LOW_VOL"),
            _make_confirmed_seg("z", _T1, _T2, _T3, log_move=0.0,
                                direction_label="NON_DIRECTIONAL",
                                volatility_label="LOW_VOL",
                                final_label="NON_DIRECTIONAL_LOW_VOL"),
            _make_confirmed_seg("s3", _T2, _T3, _T4, log_move=0.01, direction_label="UP",
                                final_label="UP_LOW_VOL"),
        ]
        result = check_turning_point_alternation(segs)
        assert result["passed"] is False
        assert result["detail"]["violation_count"] >= 1
        assert result["detail"]["zero_move_segment_count"] == 1

    def test_empty_segments(self) -> None:
        result = check_turning_point_alternation([])
        assert result["passed"] is True

    def test_check_key(self) -> None:
        result = check_turning_point_alternation([])
        assert result["check"] == "segment_turning_point_alternation"


# ---------------------------------------------------------------------------
# 3. check_segment_non_overlap
# ---------------------------------------------------------------------------


class TestSegmentNonOverlap:
    def _clean_segs(self) -> list[dict[str, Any]]:
        return [
            _make_confirmed_seg("s1", _T0, _T1, _T1),
            _make_confirmed_seg("s2", _T1, _T2, _T2),  # start == prev.end: correct
        ]

    def _overlapping_segs(self) -> list[dict[str, Any]]:
        segs = self._clean_segs()
        # Force s2 to start before s1 ends
        segs[1] = {**segs[1], "start_timestamp": _T0}  # _T0 < _T1
        return segs

    def test_clean_passes(self) -> None:
        result = check_segment_non_overlap(self._clean_segs())
        assert result["passed"] is True

    def test_overlap_fails(self) -> None:
        result = check_segment_non_overlap(self._overlapping_segs())
        assert result["passed"] is False
        assert result["detail"]["violation_count"] >= 1

    def test_tail_excluded(self) -> None:
        segs = [
            _make_confirmed_seg("s1", _T0, _T1, _T1),
            _make_tail_seg("tail", _T0, _T2),  # overlaps s1, but tail so ignored
        ]
        result = check_segment_non_overlap(segs)
        assert result["passed"] is True

    def test_check_key(self) -> None:
        result = check_segment_non_overlap([])
        assert result["check"] == "segment_non_overlap"


# ---------------------------------------------------------------------------
# 4. check_segment_coverage
# ---------------------------------------------------------------------------


class TestSegmentCoverage:
    def _clean_segs(self) -> list[dict[str, Any]]:
        return [
            _make_confirmed_seg("s1", _T0, _T1, _T1),
            _make_confirmed_seg("s2", _T1, _T2, _T2),
        ]

    def _gap_segs(self) -> list[dict[str, Any]]:
        return [
            _make_confirmed_seg("s1", _T0, _T1, _T1),
            _make_confirmed_seg("s2", _T2, _T3, _T3),  # gap: T1 != T2
        ]

    def test_clean_passes(self) -> None:
        result = check_segment_coverage(self._clean_segs())
        assert result["passed"] is True

    def test_gap_fails(self) -> None:
        result = check_segment_coverage(self._gap_segs())
        assert result["passed"] is False
        assert result["detail"]["violation_count"] == 1

    def test_check_key(self) -> None:
        result = check_segment_coverage([])
        assert result["check"] == "segment_coverage_no_gaps"


# ---------------------------------------------------------------------------
# 5. check_confirm_bar_consistency
# ---------------------------------------------------------------------------


class TestConfirmBarConsistency:
    def _clean_segs(self) -> list[dict[str, Any]]:
        # Ground-truth ordering: start <= end <= confirm (confirm always AFTER end).
        # start=T0, end=T1, confirm=T2 mirrors what dc_engine/build_segments produce.
        return [
            _make_confirmed_seg("s1", _T0, _T1, _T2),
            _make_tail_seg("tail", _T1, _T2),
        ]

    def test_clean_passes(self) -> None:
        result = check_confirm_bar_consistency(self._clean_segs())
        assert result["passed"] is True

    def test_tail_with_confirm_ts_fails(self) -> None:
        segs = [_make_tail_seg("tail", _T0, _T1)]
        segs[0] = {**segs[0], "confirm_timestamp": _T1}  # tail should NOT have this
        result = check_confirm_bar_consistency(segs)
        assert result["passed"] is False

    def test_tail_with_direction_label_fails(self) -> None:
        segs = [_make_tail_seg("tail", _T0, _T1)]
        segs[0] = {**segs[0], "direction_label": "UP"}
        result = check_confirm_bar_consistency(segs)
        assert result["passed"] is False

    def test_confirmed_missing_confirm_ts_fails(self) -> None:
        segs = [_make_confirmed_seg("s1", _T0, _T1, _T2)]
        segs[0] = {**segs[0], "confirm_timestamp": None}
        result = check_confirm_bar_consistency(segs)
        assert result["passed"] is False

    def test_confirmed_missing_final_label_fails(self) -> None:
        segs = [_make_confirmed_seg("s1", _T0, _T1, _T2)]
        segs[0] = {**segs[0], "final_label": None}
        result = check_confirm_bar_consistency(segs)
        assert result["passed"] is False

    def test_confirm_before_end_fails(self) -> None:
        # confirm_timestamp < end_timestamp → negative lag, impossible by construction.
        segs = [_make_confirmed_seg("s1", _T0, _T2, _T1)]  # end=T2, confirm=T1 → violation
        result = check_confirm_bar_consistency(segs)
        assert result["passed"] is False

    def test_negative_lag_bars_fails(self) -> None:
        segs = [_make_confirmed_seg("s1", _T0, _T1, _T2)]
        segs[0] = {**segs[0], "lag_bars": -1}
        result = check_confirm_bar_consistency(segs)
        assert result["passed"] is False

    def test_missing_lag_move_fails(self) -> None:
        """lag_move is in the full ck_segment_labels_confirm field set."""
        segs = [_make_confirmed_seg("s1", _T0, _T1, _T2)]
        segs[0] = {**segs[0], "lag_move": None}
        result = check_confirm_bar_consistency(segs)
        assert result["passed"] is False

    def test_missing_capturable_amplitude_fails(self) -> None:
        segs = [_make_confirmed_seg("s1", _T0, _T1, _T2)]
        segs[0] = {**segs[0], "capturable_amplitude": None}
        result = check_confirm_bar_consistency(segs)
        assert result["passed"] is False

    def test_check_key(self) -> None:
        result = check_confirm_bar_consistency([])
        assert result["check"] == "segment_confirm_bar_consistency"


# ---------------------------------------------------------------------------
# 6. check_bar_label_domain
# ---------------------------------------------------------------------------


class TestBarLabelDomain:
    def test_canonical_labels_pass(self) -> None:
        bars = [
            _make_bar(_T0, "UP", "LOW_VOL", "UP_LOW_VOL"),
            _make_bar(_T1, "DOWN", "HIGH_VOL", "DOWN_HIGH_VOL"),
            _make_bar(_T2, "NON_DIRECTIONAL", "MID_VOL", "NON_DIRECTIONAL_MID_VOL"),
        ]
        result = check_bar_label_domain(bars)
        assert result["passed"] is True
        assert result["detail"]["total_bars_checked"] == 3

    def test_invalid_label_fails(self) -> None:
        bars = [_make_bar(_T0, "UP", "LOW_VOL", "INVALID_LABEL")]
        result = check_bar_label_domain(bars)
        assert result["passed"] is False
        assert result["detail"]["violation_count"] >= 1

    def test_empty_bars_passes(self) -> None:
        result = check_bar_label_domain([])
        assert result["passed"] is True

    def test_check_key(self) -> None:
        result = check_bar_label_domain([])
        assert result["check"] == "bar_label_domain_9values"


# ---------------------------------------------------------------------------
# 7. check_bar_label_direction_vol_consistency
# ---------------------------------------------------------------------------


class TestBarLabelDirectionVolConsistency:
    def test_consistent_passes(self) -> None:
        bars = [
            _make_bar(_T0, "UP", "LOW_VOL", "UP_LOW_VOL"),
            _make_bar(_T1, "DOWN", "HIGH_VOL", "DOWN_HIGH_VOL"),
        ]
        result = check_bar_label_direction_vol_consistency(bars)
        assert result["passed"] is True

    def test_inconsistent_fails(self) -> None:
        bars = [_make_bar(_T0, "UP", "LOW_VOL", "DOWN_LOW_VOL")]  # wrong direction in final
        result = check_bar_label_direction_vol_consistency(bars)
        assert result["passed"] is False

    def test_all_nine_labels_consistent(self) -> None:
        canonical = [
            ("UP", "LOW_VOL", "UP_LOW_VOL"),
            ("UP", "MID_VOL", "UP_MID_VOL"),
            ("UP", "HIGH_VOL", "UP_HIGH_VOL"),
            ("DOWN", "LOW_VOL", "DOWN_LOW_VOL"),
            ("DOWN", "MID_VOL", "DOWN_MID_VOL"),
            ("DOWN", "HIGH_VOL", "DOWN_HIGH_VOL"),
            ("NON_DIRECTIONAL", "LOW_VOL", "NON_DIRECTIONAL_LOW_VOL"),
            ("NON_DIRECTIONAL", "MID_VOL", "NON_DIRECTIONAL_MID_VOL"),
            ("NON_DIRECTIONAL", "HIGH_VOL", "NON_DIRECTIONAL_HIGH_VOL"),
        ]
        ts = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        bars = [_make_bar(ts, d, v, f) for d, v, f in canonical]
        result = check_bar_label_direction_vol_consistency(bars)
        assert result["passed"] is True

    def test_check_key(self) -> None:
        result = check_bar_label_direction_vol_consistency([])
        assert result["check"] == "bar_label_direction_vol_consistency"


# ---------------------------------------------------------------------------
# 8. check_segment_label_consistency
# ---------------------------------------------------------------------------


class TestSegmentLabelConsistency:
    def test_consistent_confirmed_passes(self) -> None:
        segs = [_make_confirmed_seg("s1", _T0, _T1, _T1)]
        result = check_segment_label_consistency(segs)
        assert result["passed"] is True

    def test_inconsistent_final_label_fails(self) -> None:
        segs = [_make_confirmed_seg("s1", _T0, _T1, _T1, final_label="DOWN_LOW_VOL")]
        # default direction_label="UP", volatility_label="LOW_VOL" → expected UP_LOW_VOL
        result = check_segment_label_consistency(segs)
        assert result["passed"] is False

    def test_tail_segs_skip(self) -> None:
        segs = [_make_tail_seg("tail", _T0, _T1)]
        result = check_segment_label_consistency(segs)
        assert result["passed"] is True

    def test_check_key(self) -> None:
        result = check_segment_label_consistency([])
        assert result["check"] == "segment_label_consistency"


# ---------------------------------------------------------------------------
# 9. check_numeric_ranges_segments
# ---------------------------------------------------------------------------


class TestNumericRangesSegments:
    def _clean_segs(self) -> list[dict[str, Any]]:
        return [_make_confirmed_seg("s1", _T0, _T1, _T1)]

    def test_clean_passes(self) -> None:
        result = check_numeric_ranges_segments(self._clean_segs())
        assert result["passed"] is True

    def test_negative_rv_fails(self) -> None:
        segs = [_make_confirmed_seg("s1", _T0, _T1, _T1, realized_volatility=-0.001)]
        result = check_numeric_ranges_segments(segs)
        assert result["passed"] is False
        assert any(v["field"] == "realized_volatility" for v in result["detail"]["violations"])

    def test_er_above_1_fails(self) -> None:
        segs = [_make_confirmed_seg("s1", _T0, _T1, _T1, efficiency_ratio=1.5)]
        result = check_numeric_ranges_segments(segs)
        assert result["passed"] is False
        assert any(v["field"] == "efficiency_ratio" for v in result["detail"]["violations"])

    def test_max_jump_share_above_1_fails(self) -> None:
        segs = [_make_confirmed_seg("s1", _T0, _T1, _T1, max_jump_share=1.5)]
        result = check_numeric_ranges_segments(segs)
        assert result["passed"] is False

    def test_downside_vol_share_negative_fails(self) -> None:
        segs = [_make_confirmed_seg("s1", _T0, _T1, _T1, downside_vol_share=-0.1)]
        result = check_numeric_ranges_segments(segs)
        assert result["passed"] is False

    def test_capturable_ratio_above_1_fails(self) -> None:
        segs = [_make_confirmed_seg("s1", _T0, _T1, _T1, capturable_ratio=1.5)]
        result = check_numeric_ranges_segments(segs)
        assert result["passed"] is False

    def test_capturable_ratio_none_ok(self) -> None:
        segs = [_make_confirmed_seg("s1", _T0, _T1, _T1, capturable_ratio=None)]
        result = check_numeric_ranges_segments(segs)
        assert result["passed"] is True

    def test_check_key(self) -> None:
        result = check_numeric_ranges_segments([])
        assert result["check"] == "numeric_ranges_segments"


# ---------------------------------------------------------------------------
# 10. check_numeric_ranges_params
# ---------------------------------------------------------------------------


class TestNumericRangesParams:
    def _clean_params(self) -> list[dict[str, Any]]:
        return [_make_params("1m"), _make_params("5m")]

    def test_clean_passes(self) -> None:
        result = check_numeric_ranges_params(self._clean_params())
        assert result["passed"] is True

    def test_q_low_greater_than_q_high_fails(self) -> None:
        params = [_make_params("1m", q_low=0.66, q_high=0.33)]
        result = check_numeric_ranges_params(params)
        assert result["passed"] is False
        assert any("Q_low > Q_high" in v["issue"] for v in result["detail"]["violations"])

    def test_theta_dc_zero_fails(self) -> None:
        params = [_make_params("1m", theta_dc=0.0)]
        result = check_numeric_ranges_params(params)
        assert result["passed"] is False

    def test_theta_amp_negative_fails(self) -> None:
        params = [_make_params("1m", theta_amp=-0.001)]
        result = check_numeric_ranges_params(params)
        assert result["passed"] is False

    def test_check_key(self) -> None:
        result = check_numeric_ranges_params([])
        assert result["check"] == "numeric_ranges_params"


# ---------------------------------------------------------------------------
# 11. check_timeframe_independence
# ---------------------------------------------------------------------------


class TestTimeframeIndependence:
    def test_both_timeframes_passes(self) -> None:
        params = [_make_params("1m"), _make_params("5m")]
        result = check_timeframe_independence(params)
        assert result["passed"] is True

    def test_missing_5m_fails(self) -> None:
        params = [_make_params("1m")]
        result = check_timeframe_independence(params)
        assert result["passed"] is False
        assert "5m" in result["detail"]["missing_timeframes"]

    def test_missing_1m_fails(self) -> None:
        params = [_make_params("5m")]
        result = check_timeframe_independence(params)
        assert result["passed"] is False
        assert "1m" in result["detail"]["missing_timeframes"]

    def test_duplicate_timeframe_fails(self) -> None:
        params = [_make_params("1m"), _make_params("1m"), _make_params("5m")]
        result = check_timeframe_independence(params)
        assert result["passed"] is False
        assert "1m" in result["detail"]["duplicates"]

    def test_empty_fails(self) -> None:
        result = check_timeframe_independence([])
        assert result["passed"] is False

    def test_check_key(self) -> None:
        result = check_timeframe_independence([])
        assert result["check"] == "timeframe_independence"


# ---------------------------------------------------------------------------
# 12. aggregate_pass_fail + build_payload
# ---------------------------------------------------------------------------


class TestAggregatePassFail:
    def test_all_pass(self) -> None:
        checks = [
            {"check": "a", "passed": True},
            {"check": "b", "passed": True},
        ]
        assert aggregate_pass_fail(checks) is True

    def test_one_fail(self) -> None:
        checks = [
            {"check": "a", "passed": True},
            {"check": "b", "passed": False},
        ]
        assert aggregate_pass_fail(checks) is False

    def test_all_fail(self) -> None:
        checks = [{"check": "a", "passed": False}]
        assert aggregate_pass_fail(checks) is False

    def test_empty_passes(self) -> None:
        assert aggregate_pass_fail([]) is True


class TestBuildPayload:
    def test_structure(self) -> None:
        checks = [
            {"check": "a", "category": "x", "passed": True, "detail": {}},
            {"check": "b", "category": "y", "passed": False, "detail": {}},
        ]
        payload = build_payload(42, checks, {"run_id": 42, "run_status": "completed"})
        assert payload["run_id"] == 42
        assert payload["summary"]["total"] == 2
        assert payload["summary"]["passed"] == 1
        assert payload["summary"]["failed"] == 1
        assert isinstance(payload["generated_for"], dict)
        assert isinstance(payload["checks"], list)

    def test_all_python_types(self) -> None:
        """No numpy scalars — all plain python ints/bools."""
        checks = [{"check": "x", "passed": True, "category": "c", "detail": {}}]
        payload = build_payload(1, checks)
        assert type(payload["run_id"]) is int
        assert type(payload["summary"]["total"]) is int
        assert type(payload["summary"]["passed"]) is int
        assert type(payload["summary"]["failed"]) is int


# ===========================================================================
# DB integration tests (skip if no DB)
# ===========================================================================

_SKIP_DB = pytest.mark.skipif(
    not _HAS_DB,
    reason="REGIME_BENCHMARK_DB_URL not set or DB unreachable — skipping DB integration test",
)


@_SKIP_DB
class TestRunValidationReportDB:
    """Integration: small synthetic run → DB → run_validation_report → assert PASS."""

    def test_validation_report_passes_and_stored(self) -> None:
        """Load a synthetic run, validate it, check passed=True and labeling_reports row."""
        from pathlib import Path as P

        from regime_benchmark.config import LabelingConfig
        from regime_benchmark.pipeline import run_pipeline

        conn: psycopg.Connection[Any] | None = None
        run_id: int | None = None

        try:
            # Connect
            conn = psycopg.connect(_DB_URL)

            # Load a small synthetic run using the same pattern as test_persistence_roundtrip.py
            config_path = P(__file__).parent.parent / "config" / "labeling_config.yaml"
            config = LabelingConfig.from_yaml(config_path)
            run_id = run_pipeline(config=config, synthetic=True)

            # Run the validation report
            result = run_validation_report(run_id, _DB_URL, store=True)

            # Check overall pass
            assert result["passed"] is True, (
                f"Validation FAILED. Failed checks: "
                f"{[c['check'] for c in result['checks'] if not c['passed']]}"
            )
            assert isinstance(result["payload"], dict)
            assert isinstance(result["checks"], list)
            assert len(result["checks"]) > 0

            # Verify labeling_reports row was written
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, passed, report_type
                    FROM labeling_reports
                    WHERE run_id = %s AND report_type = 'validation'
                    ORDER BY id DESC
                    LIMIT 1
                    """,
                    (run_id,),
                )
                row = cur.fetchone()

            assert row is not None, f"No labeling_reports row found for run_id={run_id}"
            report_id, db_passed, report_type = row[0], row[1], row[2]
            assert db_passed is True, f"labeling_reports.passed={db_passed}, expected True"
            assert report_type == "validation"
            assert isinstance(report_id, int)

        finally:
            # Teardown: delete test run (CASCADE handles child rows)
            if run_id is not None and conn is not None:
                try:
                    with conn.cursor() as cur:
                        cur.execute("DELETE FROM labeling_runs WHERE id = %s", (run_id,))
                    conn.commit()
                except Exception as e:
                    print(f"[teardown] WARNING: failed to delete run_id={run_id}: {e}")
            if conn is not None:
                conn.close()

    def test_store_false_does_not_write(self) -> None:
        """With store=False, no labeling_reports row should be written."""
        from pathlib import Path as P

        from regime_benchmark.config import LabelingConfig
        from regime_benchmark.pipeline import run_pipeline

        conn: psycopg.Connection[Any] | None = None
        run_id: int | None = None

        try:
            conn = psycopg.connect(_DB_URL)
            config_path = P(__file__).parent.parent / "config" / "labeling_config.yaml"
            config = LabelingConfig.from_yaml(config_path)
            run_id = run_pipeline(config=config, synthetic=True)

            # Run WITHOUT storing
            result = run_validation_report(run_id, _DB_URL, store=False)
            assert "passed" in result

            # No labeling_reports row should exist
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT COUNT(*) FROM labeling_reports
                    WHERE run_id = %s AND report_type = 'validation'
                    """,
                    (run_id,),
                )
                count = cur.fetchone()[0]  # type: ignore[index]
            assert count == 0, f"Expected 0 labeling_reports rows, got {count}"

        finally:
            if run_id is not None and conn is not None:
                try:
                    with conn.cursor() as cur:
                        cur.execute("DELETE FROM labeling_runs WHERE id = %s", (run_id,))
                    conn.commit()
                except Exception as e:
                    print(f"[store_false teardown] WARNING: {e}")
            if conn is not None:
                conn.close()


@_SKIP_DB
class TestJoinedLabelsView:
    """Integration: verify joined_labels_1m_5m 5-minute bucket alignment."""

    def test_5m_bucket_alignment(self) -> None:
        """Each 1m bar maps to the 5m label of its containing 5-minute bucket.

        For rows where both 1m and 5m labels exist (label_5m IS NOT NULL),
        the 5m bar's open_time_5m_bucket must equal
        date_bin('5 minutes', open_time_1m, epoch) — verified at DB level.
        """
        from pathlib import Path as P

        from regime_benchmark.config import LabelingConfig
        from regime_benchmark.pipeline import run_pipeline

        conn: psycopg.Connection[Any] | None = None
        run_id: int | None = None

        try:
            conn = psycopg.connect(_DB_URL)
            config_path = P(__file__).parent.parent / "config" / "labeling_config.yaml"
            config = LabelingConfig.from_yaml(config_path)
            run_id = run_pipeline(config=config, synthetic=True)

            # Query the VIEW for rows where 5m label is present
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        open_time_1m,
                        label_1m,
                        open_time_5m_bucket,
                        label_5m,
                        date_bin(
                            '5 minutes'::interval,
                            open_time_1m,
                            TIMESTAMPTZ '1970-01-01 00:00:00+00'
                        ) AS computed_5m_bucket
                    FROM joined_labels_1m_5m
                    WHERE run_id = %s
                      AND label_5m IS NOT NULL
                    LIMIT 200
                    """,
                    (run_id,),
                )
                rows = cur.fetchall()

            assert len(rows) > 0, f"No joined rows with label_5m IS NOT NULL for run_id={run_id}"

            # Verify each row: open_time_5m_bucket == date_bin(...)
            mismatches: list[tuple[Any, Any, Any]] = []
            for row in rows:
                open_time_1m, label_1m, open_time_5m_bucket, label_5m, computed_5m = row
                if open_time_5m_bucket != computed_5m:
                    mismatches.append((open_time_1m, open_time_5m_bucket, computed_5m))

            assert len(mismatches) == 0, (
                f"5m bucket mismatch in joined_labels_1m_5m: "
                f"{len(mismatches)} rows misaligned. First: {mismatches[0]}"
            )

            # Also verify label_1m and label_5m are in the canonical set
            valid_labels = {
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
            for row in rows:
                _open_time_1m, label_1m, _open_time_5m_bucket, label_5m, _computed_5m = row
                assert label_1m in valid_labels, f"Invalid label_1m: {label_1m!r}"
                assert label_5m in valid_labels, f"Invalid label_5m: {label_5m!r}"

        finally:
            if run_id is not None and conn is not None:
                try:
                    with conn.cursor() as cur:
                        cur.execute("DELETE FROM labeling_runs WHERE id = %s", (run_id,))
                    conn.commit()
                except Exception as e:
                    print(f"[5m_bucket teardown] WARNING: {e}")
            if conn is not None:
                conn.close()
