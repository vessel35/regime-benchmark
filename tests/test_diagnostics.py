"""Formula-level unit tests for all diagnostic modules — M4 deliverable.

Tests each diagnostic formula against the spec with hand-computed expected
values.  DB-free: no persistence layer involved.

Spec references:
  lag       — requirements.md §6.5.1  / design §10
  cost      — requirements.md §6.5.2  / design §10.1
  jump      — requirements.md §7.3.1  / design §10
  asymmetry — requirements.md §7.3.3  / design §10
  er_corr   — requirements.md §7.3.2  / design §10.2
  volatility boundaries — design §9
"""

from __future__ import annotations

import math

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Imports under test
# ---------------------------------------------------------------------------
from regime_benchmark.diagnostics.asymmetry import compute_asymmetry_diagnostics_segments
from regime_benchmark.diagnostics.cost import compute_cost_diagnostics_segments
from regime_benchmark.diagnostics.er_corr import (
    classify_er_vol_overlap,
    compute_er_vol_correlation,
)
from regime_benchmark.diagnostics.jump import compute_jump_diagnostics_segments
from regime_benchmark.diagnostics.lag import compute_lag_diagnostics_segments
from regime_benchmark.direction.segments import Segment
from regime_benchmark.labeling.assemble import assign_final_labels
from regime_benchmark.volatility.realized import assign_volatility, compute_volatility_quantiles

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_segment(
    start: int = 0,
    end: int = 5,
    confirm: int | None = 2,
    amplitude: float = 0.05,
    log_move: float = 0.05,
    realized_volatility: float = 0.0,
    is_tail: bool = False,
) -> Segment:
    """Construct a minimal Segment for diagnostic tests."""
    n = end - start + 1
    seg = Segment(
        segment_id="seg_test",
        start_bar=start,
        end_bar=end,
        confirm_bar=confirm,
        is_tail_unconfirmed=is_tail,
        n_bars=n,
        log_move=log_move,
        amplitude=amplitude,
        path_length=0.0,
        efficiency_ratio=0.0,
        start_price_hlc3=100.0,
        end_price_hlc3=100.0 * math.exp(log_move),
        realized_volatility=realized_volatility,
        realized_volatility_per_bar=0.0,
    )
    return seg


# ===========================================================================
# 1. LAG DIAGNOSTICS (§6.5.1)
# ===========================================================================

class TestLagDiagnostics:
    """Formula: confirm_bar, lag_bars, lag_move, capturable_amplitude, capturable_ratio."""

    def _build_prices_returns(
        self, d_values: list[float], start: int = 0
    ) -> np.ndarray:
        """Build log-price array from d_values starting at p[start]=0."""
        n = start + len(d_values) + 1
        p = np.zeros(n, dtype=np.float64)
        for i, dv in enumerate(d_values):
            p[start + i + 1] = p[start + i] + dv
        return p

    def test_lag_bars_formula(self) -> None:
        """lag_bars = confirm_bar - start_bar; hand-computed example."""
        # d from bar 1..5: cumulative price moves +0.01, +0.03, +0.025, +0.055, +0.07
        # theta_dc = 0.025 → confirm_bar = bar 2 (cumsum reaches 0.03 ≥ 0.025 at bar 2)
        d_values = [0.01, 0.02, -0.005, 0.03, 0.015]
        # p[0]=0, p[1]=0.01, p[2]=0.03, p[3]=0.025, p[4]=0.055, p[5]=0.07
        p = self._build_prices_returns(d_values, start=0)

        # amplitude = |p[5] - p[0]| = 0.07, theta_dc = 0.025
        seg = _make_segment(start=0, end=5, confirm=None, amplitude=0.07)
        # Confirm bar logic: first bar where |p[t] - p[0]| >= theta_dc
        # bar 1: 0.01 < 0.025 → no
        # bar 2: 0.03 >= 0.025 → yes → confirm_bar = 2
        seg.confirm_bar = 2  # inject known confirm_bar

        segments = compute_lag_diagnostics_segments([seg], p, theta_dc=0.025)
        s = segments[0]

        assert s.lag_bars == 2, f"Expected lag_bars=2, got {s.lag_bars}"
        assert abs(s.lag_move - 0.03) < 1e-12, f"lag_move expected 0.03, got {s.lag_move}"
        # capturable_amplitude = max(0.07 - 0.03, 0) = 0.04
        assert abs(s.capturable_amplitude - 0.04) < 1e-12
        # capturable_ratio = 0.04 / 0.07
        expected_ratio = 0.04 / 0.07
        assert abs(s.capturable_ratio - expected_ratio) < 1e-10

    def test_lag_bars_non_negative(self) -> None:
        """lag_bars must be >= 0 (invariant §6.5.1)."""
        d_values = [0.01, 0.02, 0.03, 0.04, 0.05]
        p = self._build_prices_returns(d_values, start=0)
        seg = _make_segment(start=0, end=5, confirm=3, amplitude=0.10)
        compute_lag_diagnostics_segments([seg], p, theta_dc=0.025)
        assert seg.lag_bars is not None and seg.lag_bars >= 0

    def test_capturable_ratio_in_unit_interval(self) -> None:
        """capturable_ratio must be in [0, 1] (invariant §6.5.1)."""
        d_values = [0.01, 0.02, 0.03, 0.04, 0.05]
        p = self._build_prices_returns(d_values, start=0)
        seg = _make_segment(start=0, end=5, confirm=1, amplitude=0.10)
        compute_lag_diagnostics_segments([seg], p, theta_dc=0.01)
        assert seg.capturable_ratio is not None
        assert 0.0 <= seg.capturable_ratio <= 1.0

    def test_zero_amplitude_capturable_ratio_is_zero(self) -> None:
        """If A_j = 0, capturable_ratio = 0 (§6.5.1 edge case)."""
        p = np.zeros(6, dtype=np.float64)
        seg = _make_segment(start=0, end=5, confirm=2, amplitude=0.0)
        compute_lag_diagnostics_segments([seg], p, theta_dc=0.01)
        assert seg.capturable_ratio == 0.0

    def test_tail_segment_lag_fields_are_none(self) -> None:
        """Tail (is_tail_unconfirmed=True) segments get None lag fields."""
        p = np.zeros(6, dtype=np.float64)
        seg = _make_segment(start=0, end=5, confirm=None, is_tail=True)
        compute_lag_diagnostics_segments([seg], p, theta_dc=0.01)
        assert seg.lag_bars is None
        assert seg.lag_move is None
        assert seg.capturable_amplitude is None
        assert seg.capturable_ratio is None

    def test_capturable_ratio_exact_values(self) -> None:
        """Hand-verify capturable_ratio = 0.04/0.07 for the canonical example."""
        d_values = [0.01, 0.02, -0.005, 0.03, 0.015]
        p = self._build_prices_returns(d_values, start=0)
        # p[2] = 0.03, amplitude = 0.07, confirm_bar = 2
        seg = _make_segment(start=0, end=5, confirm=2, amplitude=0.07)
        compute_lag_diagnostics_segments([seg], p, theta_dc=0.025)
        expected = 0.04 / 0.07
        assert abs(seg.capturable_ratio - expected) < 1e-10


# ===========================================================================
# 2. COST DIAGNOSTICS (§6.5.2)
# ===========================================================================

class TestCostDiagnostics:
    """Formula: estimated_round_trip_cost_log, amplitude_to_cost_ratio, flag."""

    def _run_cost(self, amplitude: float) -> Segment:
        """Helper: run cost diagnostics with fixed fee=0.0004, slip=0.0002."""
        seg = _make_segment(amplitude=amplitude)
        compute_cost_diagnostics_segments(
            [seg], taker_fee_rate=0.0004, slippage_rate_estimate=0.0002
        )
        return seg

    def test_cost_formula_exact(self) -> None:
        """estimated_round_trip_cost_log = 2*taker_fee + slippage."""
        # cost = 2*0.0004 + 0.0002 = 0.001
        # A = 0.004, ratio = 0.004/0.001 = 4.0, flag = False
        seg = self._run_cost(0.004)
        assert abs(seg.amplitude_to_cost_ratio - 4.0) < 1e-12
        assert seg.low_tradeability_segment_flag is False

    def test_ratio_below_one(self) -> None:
        """amplitude < round-trip cost → ratio < 1, flag=True."""
        # cost = 0.001, A = 0.0005, ratio = 0.5
        seg = self._run_cost(0.0005)
        assert abs(seg.amplitude_to_cost_ratio - 0.5) < 1e-12
        assert seg.low_tradeability_segment_flag is True

    def test_ratio_between_one_and_three(self) -> None:
        """1 <= ratio < 3 → flag=True (§6.5.2)."""
        # cost = 0.001, A = 0.002, ratio = 2.0
        seg = self._run_cost(0.002)
        assert abs(seg.amplitude_to_cost_ratio - 2.0) < 1e-12
        assert seg.low_tradeability_segment_flag is True

    def test_ratio_exactly_three_is_not_flagged(self) -> None:
        """ratio == 3 → flag=False (boundary: flag = ratio < 3)."""
        # cost = 0.001, A = 0.003, ratio = 3.0
        seg = self._run_cost(0.003)
        assert abs(seg.amplitude_to_cost_ratio - 3.0) < 1e-12
        assert seg.low_tradeability_segment_flag is False

    def test_ratio_below_three_boundary(self) -> None:
        """ratio = 2.9 (just below 3) → flag=True."""
        # cost = 0.001, A = 0.0029, ratio = 2.9
        seg = self._run_cost(0.0029)
        assert abs(seg.amplitude_to_cost_ratio - 2.9) < 1e-10
        assert seg.low_tradeability_segment_flag is True

    def test_ratio_non_negative(self) -> None:
        """amplitude_to_cost_ratio must be >= 0 (invariant §11.4)."""
        seg = self._run_cost(0.001)
        assert seg.amplitude_to_cost_ratio >= 0.0


# ===========================================================================
# 3. JUMP DIAGNOSTICS (§7.3.1)
# ===========================================================================

class TestJumpDiagnostics:
    """Formula: max_abs_d, max_jump_share, BV, jump_component, jump_share_bv."""

    def _make_d_array(self, start: int, d_values: list[float]) -> np.ndarray:
        """Build full d array; d[0..start] = 0/NaN, d[start+1..] = d_values."""
        n = start + len(d_values) + 1
        d = np.zeros(n, dtype=np.float64)
        d[0] = np.nan
        for i, v in enumerate(d_values):
            d[start + 1 + i] = v
        return d

    def test_max_abs_d(self) -> None:
        """max_abs_d = max(|d_t|) over segment bars."""
        # d_values for bars 1..5 of segment starting at bar 0, ending at bar 5
        d_values = [0.01, 0.02, 0.03, -0.15, 0.02]
        d = self._make_d_array(start=0, d_values=d_values)
        rv = math.sqrt(sum(x**2 for x in d_values))
        seg = _make_segment(start=0, end=5, realized_volatility=rv)
        compute_jump_diagnostics_segments([seg], d)
        assert abs(seg.max_abs_d - 0.15) < 1e-12

    def test_max_jump_share_formula(self) -> None:
        """max_jump_share = max(d_t^2) / sum(d_t^2).

        Hand-computed:
          d = [0.01, 0.02, 0.03, -0.15, 0.02]
          sum d^2 = 0.0001+0.0004+0.0009+0.0225+0.0004 = 0.0243
          max d^2 = 0.0225
          max_jump_share = 0.0225/0.0243
        """
        d_values = [0.01, 0.02, 0.03, -0.15, 0.02]
        d = self._make_d_array(start=0, d_values=d_values)
        sum_sq = sum(x**2 for x in d_values)
        rv = math.sqrt(sum_sq)
        seg = _make_segment(start=0, end=5, realized_volatility=rv)
        compute_jump_diagnostics_segments([seg], d)

        expected_share = 0.0225 / 0.0243
        assert abs(seg.max_jump_share - expected_share) < 1e-10

    def test_max_jump_share_range(self) -> None:
        """max_jump_share must be in [0, 1] (invariant §11.4)."""
        d_values = [0.01, 0.02, 0.03, -0.15, 0.02]
        d = self._make_d_array(start=0, d_values=d_values)
        rv = math.sqrt(sum(x**2 for x in d_values))
        seg = _make_segment(start=0, end=5, realized_volatility=rv)
        compute_jump_diagnostics_segments([seg], d)
        assert 0.0 <= seg.max_jump_share <= 1.0

    def test_bipower_variation_formula(self) -> None:
        """BV = (pi/2) * sum_{consecutive pairs} |d_t| * |d_{t-1}|.

        Hand-computed for d = [0.01, 0.02, 0.03, -0.15, 0.02]:
          pairs: (0.01,0.02)=0.0002, (0.02,0.03)=0.0006,
                 (0.03,0.15)=0.0045, (0.15,0.02)=0.003
          sum_products = 0.0083
          BV = (pi/2) * 0.0083 = 0.013037609512397639
        Hard-coded literal (NOT recomputed with the impl formula) so a wrong
        pairing in the implementation would be caught (review SF1).
        """
        d_values = [0.01, 0.02, 0.03, -0.15, 0.02]
        d = self._make_d_array(start=0, d_values=d_values)
        expected_bv = 0.013037609512397639
        rv = math.sqrt(sum(x**2 for x in d_values))
        seg = _make_segment(start=0, end=5, realized_volatility=rv)
        compute_jump_diagnostics_segments([seg], d)
        assert abs(seg.bipower_variation - expected_bv) < 1e-12

    def test_jump_component_non_negative(self) -> None:
        """jump_component = max(RV^2 - BV, 0) must be >= 0."""
        d_values = [0.01, 0.02, 0.03, -0.15, 0.02]
        d = self._make_d_array(start=0, d_values=d_values)
        rv = math.sqrt(sum(x**2 for x in d_values))
        seg = _make_segment(start=0, end=5, realized_volatility=rv)
        compute_jump_diagnostics_segments([seg], d)
        assert seg.jump_component >= 0.0

    def test_jump_share_bv_formula(self) -> None:
        """jump_share_bv = jump_component / RV^2 hand-verified.

        With d = [0.01, 0.02, 0.03, -0.15, 0.02]:
          sum_sq = 0.0243, BV = (pi/2)*0.0083 = 0.013037609512397639
          jump_component = max(0.0243 - 0.013037609512397639, 0) = 0.01126239048760236
          jump_share_bv = 0.01126239048760236 / 0.0243 = 0.4634728595721136
        Hard-coded literal (NOT recomputed with the impl formula) — review SF2.
        """
        d_values = [0.01, 0.02, 0.03, -0.15, 0.02]
        d = self._make_d_array(start=0, d_values=d_values)
        rv = math.sqrt(sum(x**2 for x in d_values))
        expected_share = 0.4634728595721136

        seg = _make_segment(start=0, end=5, realized_volatility=rv)
        compute_jump_diagnostics_segments([seg], d)
        assert abs(seg.jump_share_bv - expected_share) < 1e-10

    def test_jump_share_bv_non_negative(self) -> None:
        """jump_share_bv must be >= 0 (spec §10.3 note: may exceed 1.0)."""
        d_values = [0.01, 0.02, 0.03, -0.15, 0.02]
        d = self._make_d_array(start=0, d_values=d_values)
        rv = math.sqrt(sum(x**2 for x in d_values))
        seg = _make_segment(start=0, end=5, realized_volatility=rv)
        compute_jump_diagnostics_segments([seg], d)
        assert seg.jump_share_bv >= 0.0

    def test_zero_rv_jump_fields_are_zero(self) -> None:
        """When RV=0 (all d=0), max_jump_share=0 and jump_share_bv=0."""
        d_values = [0.0, 0.0, 0.0, 0.0, 0.0]
        d = self._make_d_array(start=0, d_values=d_values)
        seg = _make_segment(start=0, end=5, realized_volatility=0.0)
        compute_jump_diagnostics_segments([seg], d)
        assert seg.max_jump_share == 0.0
        assert seg.jump_share_bv == 0.0

    def test_single_bar_segment(self) -> None:
        """Single d value: BV=0 (no pairs), jump_component=max(RV^2-0,0)=RV^2."""
        d_values = [0.05]
        d = self._make_d_array(start=0, d_values=d_values)
        rv = 0.05
        seg = _make_segment(start=0, end=1, realized_volatility=rv)
        compute_jump_diagnostics_segments([seg], d)
        assert seg.bipower_variation == 0.0
        # jump_component = max(0.0025 - 0.0, 0) = 0.0025
        assert abs(seg.jump_component - 0.0025) < 1e-12
        # jump_share_bv = 0.0025 / 0.0025 = 1.0
        assert abs(seg.jump_share_bv - 1.0) < 1e-12


# ===========================================================================
# 4. ASYMMETRY DIAGNOSTICS (§7.3.3)
# ===========================================================================

class TestAsymmetryDiagnostics:
    """Formula: rv_plus, rv_minus, downside_vol_share."""

    def _make_d_array(self, start: int, d_values: list[float]) -> np.ndarray:
        n = start + len(d_values) + 1
        d = np.zeros(n, dtype=np.float64)
        d[0] = np.nan
        for i, v in enumerate(d_values):
            d[start + 1 + i] = v
        return d

    def test_rv_plus_formula(self) -> None:
        """rv_plus = sqrt(sum(d^2 * I(d>0))) hand-verified."""
        # d = [0.01, 0.02, -0.10, 0.005, 0.003]
        d_values = [0.01, 0.02, -0.10, 0.005, 0.003]
        d = self._make_d_array(start=0, d_values=d_values)
        pos_sq = 0.01**2 + 0.02**2 + 0.005**2 + 0.003**2
        expected_rv_plus = math.sqrt(pos_sq)
        seg = _make_segment(start=0, end=5)
        compute_asymmetry_diagnostics_segments([seg], d)
        assert abs(seg.rv_plus - expected_rv_plus) < 1e-12

    def test_rv_minus_formula(self) -> None:
        """rv_minus = sqrt(sum(d^2 * I(d<0))) hand-verified."""
        d_values = [0.01, 0.02, -0.10, 0.005, 0.003]
        d = self._make_d_array(start=0, d_values=d_values)
        neg_sq = 0.10**2
        expected_rv_minus = math.sqrt(neg_sq)  # = 0.10
        seg = _make_segment(start=0, end=5)
        compute_asymmetry_diagnostics_segments([seg], d)
        assert abs(seg.rv_minus - expected_rv_minus) < 1e-12

    def test_downside_vol_share_formula(self) -> None:
        """downside_vol_share = sum(d^2*I(d<0)) / sum(d^2) hand-verified."""
        d_values = [0.01, 0.02, -0.10, 0.005, 0.003]
        d = self._make_d_array(start=0, d_values=d_values)
        total_sq = sum(x**2 for x in d_values)
        neg_sq = 0.10**2
        expected_share = neg_sq / total_sq
        seg = _make_segment(start=0, end=5)
        compute_asymmetry_diagnostics_segments([seg], d)
        assert abs(seg.downside_vol_share - expected_share) < 1e-12

    def test_downside_vol_share_range(self) -> None:
        """downside_vol_share must be in [0, 1] (invariant §11.4)."""
        d_values = [0.01, 0.02, -0.10, 0.005, 0.003]
        d = self._make_d_array(start=0, d_values=d_values)
        seg = _make_segment(start=0, end=5)
        compute_asymmetry_diagnostics_segments([seg], d)
        assert 0.0 <= seg.downside_vol_share <= 1.0

    def test_big_down_move_dominates(self) -> None:
        """One large negative move → downside_vol_share > 0.5."""
        d_values = [0.01, 0.02, -0.10, 0.005, 0.003]
        d = self._make_d_array(start=0, d_values=d_values)
        seg = _make_segment(start=0, end=5)
        compute_asymmetry_diagnostics_segments([seg], d)
        assert seg.downside_vol_share > 0.5, (
            f"Expected downside_vol_share > 0.5, got {seg.downside_vol_share}"
        )

    def test_rv_plus_rv_minus_non_negative(self) -> None:
        """rv_plus and rv_minus must be >= 0 (invariant §11.4)."""
        d_values = [0.01, 0.02, -0.10, 0.005, 0.003]
        d = self._make_d_array(start=0, d_values=d_values)
        seg = _make_segment(start=0, end=5)
        compute_asymmetry_diagnostics_segments([seg], d)
        assert seg.rv_plus >= 0.0
        assert seg.rv_minus >= 0.0

    def test_all_positive_moves_zero_downside(self) -> None:
        """All positive d → rv_minus = 0, downside_vol_share = 0."""
        d_values = [0.01, 0.02, 0.03, 0.01, 0.02]
        d = self._make_d_array(start=0, d_values=d_values)
        seg = _make_segment(start=0, end=5)
        compute_asymmetry_diagnostics_segments([seg], d)
        assert seg.rv_minus == 0.0
        assert seg.downside_vol_share == 0.0

    def test_all_negative_moves_full_downside(self) -> None:
        """All negative d → rv_plus = 0, downside_vol_share = 1."""
        d_values = [-0.01, -0.02, -0.03]
        d = self._make_d_array(start=0, d_values=d_values)
        seg = _make_segment(start=0, end=3)
        compute_asymmetry_diagnostics_segments([seg], d)
        assert seg.rv_plus == 0.0
        assert abs(seg.downside_vol_share - 1.0) < 1e-12

    def test_zero_sum_sq_downside_is_zero(self) -> None:
        """All d=0 → downside_vol_share = 0 (edge case §7.3.3)."""
        d_values = [0.0, 0.0, 0.0]
        d = self._make_d_array(start=0, d_values=d_values)
        seg = _make_segment(start=0, end=3)
        compute_asymmetry_diagnostics_segments([seg], d)
        assert seg.downside_vol_share == 0.0


# ===========================================================================
# 5. ER-CORR DIAGNOSTICS (§7.3.2)
# ===========================================================================

class TestErCorrDiagnostics:
    """Spearman rho: perfectly correlated, anti-correlated, independent, edge cases."""

    def test_perfectly_correlated(self) -> None:
        """Identical orderings → rho = 1.0."""
        er = [0.1, 0.3, 0.5, 0.7, 0.9]
        rv = [0.01, 0.03, 0.05, 0.07, 0.09]
        rho = compute_er_vol_correlation(er, rv)
        assert abs(rho - 1.0) < 1e-10, f"Expected rho≈1.0, got {rho}"

    def test_perfectly_anti_correlated(self) -> None:
        """Reversed orderings → rho = -1.0."""
        er = [0.1, 0.3, 0.5, 0.7, 0.9]
        rv = [0.09, 0.07, 0.05, 0.03, 0.01]
        rho = compute_er_vol_correlation(er, rv)
        assert abs(rho - (-1.0)) < 1e-10, f"Expected rho≈-1.0, got {rho}"

    def test_independent_near_zero(self) -> None:
        """Shuffled/random-looking orderings → rho near 0.

        Hand-computed:
          er = [0.1, 0.9, 0.3, 0.5, 0.7]  → ranks [1, 5, 2, 3, 4]
          rv = [0.5, 0.3, 0.1, 0.9, 0.7]  → ranks [3, 2, 1, 5, 4]
          er_dev = [-2, 2, -1, 0, 1],  rv_dev = [0, -1, -2, 2, 1]
          numerator = 0 + (-2) + 2 + 0 + 1 = 1
          denom = sqrt(4+4+1+0+1)*sqrt(0+1+4+4+1) = sqrt(10)*sqrt(10) = 10
          rho = 1/10 = 0.1
        """
        er = [0.1, 0.9, 0.3, 0.5, 0.7]
        rv = [0.5, 0.3, 0.1, 0.9, 0.7]
        rho = compute_er_vol_correlation(er, rv)
        assert abs(rho) < 0.5, f"Expected near-zero rho, got {rho}"
        # Verify exact value: rho should be 0.1
        assert abs(rho - 0.1) < 1e-10, f"Expected rho=0.1, got {rho}"

    def test_fewer_than_two_points_returns_zero(self) -> None:
        """< 2 observations → return 0.0 (edge case)."""
        assert compute_er_vol_correlation([], []) == 0.0
        assert compute_er_vol_correlation([0.5], [0.05]) == 0.0

    def test_constant_er_returns_zero(self) -> None:
        """Constant ER (zero variance) → rho = 0.0 (undefined Spearman)."""
        er = [0.5, 0.5, 0.5, 0.5]
        rv = [0.01, 0.02, 0.03, 0.04]
        rho = compute_er_vol_correlation(er, rv)
        assert rho == 0.0, f"Expected 0.0 for constant er, got {rho}"

    def test_constant_rv_returns_zero(self) -> None:
        """Constant RV (zero variance) → rho = 0.0."""
        er = [0.1, 0.2, 0.3, 0.4]
        rv = [0.05, 0.05, 0.05, 0.05]
        rho = compute_er_vol_correlation(er, rv)
        assert rho == 0.0, f"Expected 0.0 for constant rv, got {rho}"

    def test_result_in_minus_one_to_one(self) -> None:
        """rho must be in [-1, 1] for any valid input."""
        er = [0.1, 0.4, 0.2, 0.8, 0.6]
        rv = [0.05, 0.02, 0.08, 0.01, 0.04]
        rho = compute_er_vol_correlation(er, rv)
        assert -1.0 <= rho <= 1.0

    def test_mismatched_lengths_raises(self) -> None:
        """Mismatched-length inputs must raise ValueError."""
        with pytest.raises(ValueError, match="length"):
            compute_er_vol_correlation([0.1, 0.2], [0.01])

    def test_tie_handling(self) -> None:
        """Ties handled with average ranks; verify known tied case."""
        # er = [0.5, 0.5, 0.9], rv = [0.01, 0.05, 0.09]
        # ranks of er: 1.5, 1.5, 3 (average rank for tied 0.5)
        # ranks of rv: 1, 2, 3
        # Pearson on ranks:
        #   er_ranks = [1.5, 1.5, 3.0], mean = 2.0
        #   rv_ranks = [1.0, 2.0, 3.0], mean = 2.0
        #   er_dev = [-0.5, -0.5, 1.0], rv_dev = [-1, 0, 1]
        #   numerator = 0.5 + 0.0 + 1.0 = 1.5
        #   denom = sqrt(0.25+0.25+1.0) * sqrt(1+0+1) = sqrt(1.5)*sqrt(2) = sqrt(3)
        #   rho = 1.5 / sqrt(3) ≈ 0.8660
        er = [0.5, 0.5, 0.9]
        rv = [0.01, 0.05, 0.09]
        rho = compute_er_vol_correlation(er, rv)
        expected = 1.5 / math.sqrt(3.0)
        assert abs(rho - expected) < 1e-10, f"Expected {expected}, got {rho}"

    def test_numpy_array_input(self) -> None:
        """Accepts numpy arrays as well as lists."""
        er = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        rv = np.array([0.01, 0.03, 0.05, 0.07, 0.09])
        rho = compute_er_vol_correlation(er, rv)
        assert abs(rho - 1.0) < 1e-10


class TestClassifyErVolOverlap:
    """Boundary tests for classify_er_vol_overlap."""

    def test_low_below_0_60(self) -> None:
        assert classify_er_vol_overlap(0.59) == "low"
        assert classify_er_vol_overlap(-0.59) == "low"
        assert classify_er_vol_overlap(0.0) == "low"

    def test_possible_at_0_60_boundary(self) -> None:
        """0.60 exactly → 'possible' (inclusive lower bound)."""
        assert classify_er_vol_overlap(0.60) == "possible"
        assert classify_er_vol_overlap(-0.60) == "possible"

    def test_possible_below_0_80(self) -> None:
        assert classify_er_vol_overlap(0.79) == "possible"
        assert classify_er_vol_overlap(-0.79) == "possible"

    def test_strong_at_0_80_boundary(self) -> None:
        """0.80 exactly → 'strong' (inclusive lower bound)."""
        assert classify_er_vol_overlap(0.80) == "strong"
        assert classify_er_vol_overlap(-0.80) == "strong"

    def test_strong_above_0_80(self) -> None:
        assert classify_er_vol_overlap(0.95) == "strong"
        assert classify_er_vol_overlap(-1.0) == "strong"

    def test_boundary_0_59_is_low(self) -> None:
        """Just below 0.60 → 'low'."""
        assert classify_er_vol_overlap(0.599) == "low"

    def test_boundary_negative_0_80_is_strong(self) -> None:
        """rho = -0.80 → |rho|=0.80 → 'strong'."""
        assert classify_er_vol_overlap(-0.80) == "strong"


# ===========================================================================
# 6. VOLATILITY BOUNDARY SEMANTICS (design §9)
# ===========================================================================

class TestVolatilityBoundaries:
    """Verify assign_volatility boundary semantics: inclusive lower, exclusive upper."""

    def test_score_below_q_low_is_low_vol(self) -> None:
        """score < q_low → LOW_VOL."""
        assert assign_volatility(0.004, q_low=0.005, q_high=0.010) == "LOW_VOL"

    def test_score_exactly_q_low_is_low_vol(self) -> None:
        """score == q_low → LOW_VOL (inclusive per design §9)."""
        assert assign_volatility(0.005, q_low=0.005, q_high=0.010) == "LOW_VOL"

    def test_score_just_above_q_low_is_mid_vol(self) -> None:
        """score just above q_low → MID_VOL."""
        assert assign_volatility(0.0051, q_low=0.005, q_high=0.010) == "MID_VOL"

    def test_score_exactly_q_high_is_mid_vol(self) -> None:
        """score == q_high → MID_VOL (inclusive per design §9: MID if q_low < score <= q_high)."""
        assert assign_volatility(0.010, q_low=0.005, q_high=0.010) == "MID_VOL"

    def test_score_just_above_q_high_is_high_vol(self) -> None:
        """score just above q_high → HIGH_VOL."""
        assert assign_volatility(0.0101, q_low=0.005, q_high=0.010) == "HIGH_VOL"

    def test_score_well_above_q_high_is_high_vol(self) -> None:
        """score >> q_high → HIGH_VOL."""
        assert assign_volatility(0.05, q_low=0.005, q_high=0.010) == "HIGH_VOL"

    def test_quantile_fn_1m_only_population(self) -> None:
        """Quantile computed on 1m-only differs from a pooled population."""
        rv_1m = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006]
        rv_5m = [0.010, 0.020, 0.030, 0.040, 0.050, 0.060]

        # Per-timeframe quantiles
        q_low_1m, q_high_1m = compute_volatility_quantiles(rv_1m, 0.33, 0.66)
        q_low_5m, q_high_5m = compute_volatility_quantiles(rv_5m, 0.33, 0.66)

        # Pooled
        rv_pooled = rv_1m + rv_5m
        q_low_pool, q_high_pool = compute_volatility_quantiles(rv_pooled, 0.33, 0.66)

        # Per-tf boundaries must differ from pooled
        assert q_low_1m != q_low_pool or q_high_1m != q_high_pool, (
            "1m-only quantiles should differ from pooled quantiles"
        )
        assert q_low_5m != q_low_pool or q_high_5m != q_high_pool, (
            "5m-only quantiles should differ from pooled quantiles"
        )

    def test_quantile_fn_per_tf_not_mixed(self) -> None:
        """1m and 5m quantile boundaries are independent (design §7.2, §9.1)."""
        # 1m: small values; 5m: 10x larger
        rv_1m = [0.001, 0.002, 0.003]
        rv_5m = [0.010, 0.020, 0.030]
        q_low_1m, q_high_1m = compute_volatility_quantiles(rv_1m, 0.33, 0.66)
        q_low_5m, q_high_5m = compute_volatility_quantiles(rv_5m, 0.33, 0.66)
        # Both should be valid (Q_low <= Q_high)
        assert q_low_1m <= q_high_1m
        assert q_low_5m <= q_high_5m
        # 5m boundaries should be much larger than 1m
        assert q_low_5m > q_high_1m


# ===========================================================================
# 7. DIAGNOSTIC NON-LEAKAGE (design §10 invariant, §19-10)
# ===========================================================================

class TestDiagnosticNonLeakage:
    """Diagnostics must NOT influence final_label (§19-10).

    final_label = direction × volatility only.
    Two segments with identical (direction, volatility) but wildly different
    diagnostic values must produce identical final_label.
    """

    def _make_labeled_segment(
        self,
        direction: str,
        volatility: str,
        amplitude: float,
        rv: float,
        # Diagnostic values that should NOT affect final_label
        jump_share: float = 0.0,
        downside_vol_share: float = 0.0,
        er: float = 0.5,
        cost_ratio: float = 5.0,
    ) -> Segment:
        """Build a segment with direction/volatility set + injected diagnostics.

        Crucially does NOT set final_label here — the real assign_final_labels()
        derives it, so the non-leakage tests exercise actual pipeline code rather
        than asserting a manually-set string (review B1).
        """
        seg = _make_segment(amplitude=amplitude, realized_volatility=rv)
        seg.direction_label = direction
        seg.volatility_label = volatility
        seg.final_label = None  # derived by assign_final_labels, not hand-set
        # Inject wildly different diagnostic values
        seg.max_jump_share = jump_share
        seg.downside_vol_share = downside_vol_share
        seg.efficiency_ratio = er
        seg.amplitude_to_cost_ratio = cost_ratio
        return seg

    def test_same_final_label_despite_different_diagnostics(self) -> None:
        """Identical (direction, volatility) → identical final_label regardless of diagnostics."""
        seg_a = self._make_labeled_segment(
            direction="UP", volatility="HIGH_VOL",
            amplitude=0.10, rv=0.05,
            jump_share=0.95,         # near-single-bar jump
            downside_vol_share=0.8,  # mostly downside
            er=0.1,                  # very inefficient path
            cost_ratio=0.5,          # not tradeable
        )
        seg_b = self._make_labeled_segment(
            direction="UP", volatility="HIGH_VOL",
            amplitude=0.10, rv=0.05,
            jump_share=0.01,         # smooth continuous vol
            downside_vol_share=0.1,  # mostly upside
            er=0.9,                  # very efficient path
            cost_ratio=50.0,         # very tradeable
        )
        # Run the REAL labeling function — proves diagnostics cannot bleed in.
        assign_final_labels([seg_a, seg_b])
        assert seg_a.final_label == seg_b.final_label
        assert seg_a.final_label == "UP_HIGH_VOL"

    def test_final_label_formula_is_direction_times_volatility(self) -> None:
        """final_label = direction_label + '_' + volatility_label (§8)."""
        for direction in ("UP", "DOWN", "NON_DIRECTIONAL"):
            for volatility in ("LOW_VOL", "MID_VOL", "HIGH_VOL"):
                expected = f"{direction}_{volatility}"
                seg = self._make_labeled_segment(
                    direction=direction,
                    volatility=volatility,
                    amplitude=0.05,
                    rv=0.02,
                )
                assign_final_labels([seg])
                assert seg.final_label == expected

    def test_different_diagnostics_same_label(self) -> None:
        """Exhaustive check: diagnostics vary, final_label constant."""
        base_direction = "DOWN"
        base_volatility = "MID_VOL"
        expected_label = "DOWN_MID_VOL"

        diagnostic_combos = [
            {"jump_share": 0.0, "downside_vol_share": 0.0, "er": 1.0, "cost_ratio": 100.0},
            {"jump_share": 1.0, "downside_vol_share": 1.0, "er": 0.0, "cost_ratio": 0.1},
            {"jump_share": 0.5, "downside_vol_share": 0.5, "er": 0.5, "cost_ratio": 3.0},
            {"jump_share": 0.99, "downside_vol_share": 0.01, "er": 0.01, "cost_ratio": 0.0},
        ]

        for combo in diagnostic_combos:
            seg = self._make_labeled_segment(
                direction=base_direction,
                volatility=base_volatility,
                amplitude=0.04,
                rv=0.02,
                **combo,  # type: ignore[arg-type]
            )
            assign_final_labels([seg])
            assert seg.final_label == expected_label, (
                f"final_label changed with diagnostics {combo}: got {seg.final_label}"
            )
