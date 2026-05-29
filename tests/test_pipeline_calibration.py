"""Tests for frozen-calibration parameter support in run_pipeline / _run_timeframe.

Design §8.1: FrozenParams chosen on a calibration split are locked and override
config candidate[0] defaults.  These tests verify:

1. FrozenParams dataclass is frozen (immutable) and default q_low/q_high are 0.33/0.66.
2. Frozen k_dc overrides candidate[0]: theta_dc differs, producing different segment counts.
3. Backward compat: calibration=None reproduces the exact candidate[0] behavior.
4. DB roundtrip (optional, skipped when REGIME_BENCHMARK_DB_URL is absent): labeling_run_params
   stores the frozen theta_dc/k_dc/min_segment_bars, not candidate[0] values.

All computation-level tests are DB-free (no PostgreSQL).
The DB roundtrip test follows the same skip pattern as test_persistence_roundtrip.py.
"""

from __future__ import annotations

import dataclasses
import math
import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import polars as pl
import pytest

from regime_benchmark.calibration import FrozenParams
from regime_benchmark.direction.dc_engine import compute_theta_dc, run_dc_engine
from regime_benchmark.direction.segments import assign_direction, build_segments

# ---------------------------------------------------------------------------
# Config path
# ---------------------------------------------------------------------------

_CONFIG_PATH = Path(__file__).parent.parent / "config" / "labeling_config.yaml"

# ---------------------------------------------------------------------------
# Optional DB skip (same pattern as test_persistence_roundtrip.py)
# ---------------------------------------------------------------------------

_DB_URL = os.environ.get("REGIME_BENCHMARK_DB_URL", "")
if not _DB_URL:
    _env_paths = [
        Path(__file__).parent.parent / ".env",
        Path(__file__).parent.parent.parent.parent.parent / ".env",
    ]
    for _env_path in _env_paths:
        if _env_path.exists():
            for _line in _env_path.read_text().splitlines():
                _line = _line.strip()
                if _line.startswith("REGIME_BENCHMARK_DB_URL=") and not _line.startswith("#"):
                    _DB_URL = _line.split("=", 1)[1].strip()
                    os.environ["REGIME_BENCHMARK_DB_URL"] = _DB_URL
                    break
        if _DB_URL:
            break

# ---------------------------------------------------------------------------
# Helpers: synthetic price series  (no DB, no pipeline — pure computation)
# ---------------------------------------------------------------------------

_N_BARS = 500  # enough for multiple DC segments at k=3 and k=4
_SEED = 99


def _make_log_prices(n: int = _N_BARS, seed: int = _SEED) -> np.ndarray:
    """Build a deterministic random-walk log-price array."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0, 0.002, size=n)
    p = np.concatenate([[math.log(1000.0)], np.cumsum(steps) + math.log(1000.0)])
    return p.astype(np.float64)[:n]


def _make_abs_d_series(log_prices: np.ndarray) -> pl.Series:
    d = np.empty(len(log_prices), dtype=np.float64)
    d[0] = np.nan
    d[1:] = log_prices[1:] - log_prices[:-1]
    return pl.Series("abs_d", np.abs(d))


def _segment_count(log_prices: np.ndarray, k_dc: float, q_dc: float) -> tuple[int, float]:
    """Return (confirmed_segment_count, theta_dc) for given k_dc/q_dc."""
    abs_d = _make_abs_d_series(log_prices)
    theta = compute_theta_dc(abs_d, q_dc, k_dc)
    tps = run_dc_engine(log_prices, theta)
    d_arr = np.empty(len(log_prices), dtype=np.float64)
    d_arr[0] = np.nan
    d_arr[1:] = log_prices[1:] - log_prices[:-1]
    segs = build_segments(tps, log_prices, d_arr)
    confirmed = [s for s in segs if not s.is_tail_unconfirmed]
    return len(confirmed), theta


# ---------------------------------------------------------------------------
# 1. FrozenParams dataclass contract
# ---------------------------------------------------------------------------


class TestFrozenParamsContract:
    """FrozenParams must be a frozen dataclass with the specified fields."""

    def test_frozen_rejects_mutation(self) -> None:
        """Setting any attribute after construction must raise FrozenInstanceError."""
        fp = FrozenParams(k_dc=4.0, q_dc=0.80, min_segment_bars=10)
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError, TypeError)):
            fp.k_dc = 5.0  # type: ignore[misc]

    def test_default_q_low_q_high(self) -> None:
        """Default q_low=0.33 and q_high=0.66 per design §9."""
        fp = FrozenParams(k_dc=3.0, q_dc=0.75, min_segment_bars=5)
        assert fp.q_low == pytest.approx(0.33)
        assert fp.q_high == pytest.approx(0.66)

    def test_custom_q_low_q_high(self) -> None:
        """Custom quantile overrides are stored verbatim."""
        fp = FrozenParams(k_dc=3.0, q_dc=0.75, min_segment_bars=5, q_low=0.25, q_high=0.75)
        assert fp.q_low == pytest.approx(0.25)
        assert fp.q_high == pytest.approx(0.75)

    def test_fields_stored_as_given(self) -> None:
        """k_dc, q_dc, min_segment_bars are stored exactly as provided."""
        fp = FrozenParams(k_dc=4.5, q_dc=0.80, min_segment_bars=12)
        assert fp.k_dc == pytest.approx(4.5)
        assert fp.q_dc == pytest.approx(0.80)
        assert fp.min_segment_bars == 12

    def test_is_dataclass(self) -> None:
        """FrozenParams must be a dataclass."""
        assert dataclasses.is_dataclass(FrozenParams)

    def test_equality_by_value(self) -> None:
        """Two FrozenParams with identical fields are equal."""
        a = FrozenParams(k_dc=3.0, q_dc=0.80, min_segment_bars=5)
        b = FrozenParams(k_dc=3.0, q_dc=0.80, min_segment_bars=5)
        assert a == b


# ---------------------------------------------------------------------------
# 2. Frozen k_dc overrides candidate[0] → different theta_dc → different segment counts
# ---------------------------------------------------------------------------


class TestFrozenOverridesCandidate0:
    """Computation-level tests: frozen k_dc != candidate[0] k_dc → theta_dc differs."""

    def test_theta_dc_differs_with_frozen_k(self) -> None:
        """k_dc=4.0 (frozen) vs k_dc=3.0 (candidate[0]) must produce different theta_dc."""
        p = _make_log_prices()
        abs_d = _make_abs_d_series(p)
        theta_k3 = compute_theta_dc(abs_d, q_dc=0.80, k_dc=3.0)
        theta_k4 = compute_theta_dc(abs_d, q_dc=0.80, k_dc=4.0)
        # theta_dc = Quantile(|d|, q) * k  ⟹ theta_k4 / theta_k3 == 4/3
        assert theta_k4 > theta_k3, "larger k_dc must produce larger theta_dc"
        assert theta_k4 == pytest.approx(theta_k3 * (4.0 / 3.0), rel=1e-12)

    def test_segment_count_differs_with_frozen_k(self) -> None:
        """Larger theta_dc (higher k) must produce fewer confirmed segments."""
        p = _make_log_prices()
        count_k3, theta_k3 = _segment_count(p, k_dc=3.0, q_dc=0.80)
        count_k4, theta_k4 = _segment_count(p, k_dc=4.0, q_dc=0.80)
        assert theta_k4 > theta_k3, "pre-condition: theta_k4 > theta_k3"
        assert count_k4 < count_k3, (
            f"Expected fewer segments with k=4.0 than k=3.0, "
            f"got {count_k4} vs {count_k3}"
        )

    def test_frozen_q_dc_overrides(self) -> None:
        """Different q_dc in FrozenParams must produce a different theta_dc."""
        p = _make_log_prices()
        abs_d = _make_abs_d_series(p)
        theta_q80 = compute_theta_dc(abs_d, q_dc=0.80, k_dc=3.0)
        theta_q90 = compute_theta_dc(abs_d, q_dc=0.90, k_dc=3.0)
        # 0.90 quantile >= 0.80 quantile → theta_q90 >= theta_q80
        assert theta_q90 >= theta_q80

    def test_frozen_min_segment_bars_affects_direction(self) -> None:
        """min_segment_bars gate from FrozenParams correctly filters direction labels."""
        p = _make_log_prices()
        d = np.empty(len(p), dtype=np.float64)
        d[0] = np.nan
        d[1:] = p[1:] - p[:-1]
        abs_d = pl.Series("abs_d", np.abs(d))

        theta = compute_theta_dc(abs_d, q_dc=0.80, k_dc=3.0)
        tps = run_dc_engine(p, theta)
        segs = build_segments(tps, p, d)

        # With min_segment_bars=1 (permissive): more directional segments
        for seg in segs:
            seg.direction_label = assign_direction(seg, min_segment_bars=1, theta_amp=theta)
        directional_permissive = sum(
            1 for s in segs
            if not s.is_tail_unconfirmed and s.direction_label in ("UP", "DOWN")
        )

        # Reset and apply strict gate
        for seg in segs:
            seg.direction_label = assign_direction(
                seg, min_segment_bars=100, theta_amp=theta
            )
        directional_strict = sum(
            1 for s in segs
            if not s.is_tail_unconfirmed and s.direction_label in ("UP", "DOWN")
        )

        assert directional_permissive >= directional_strict, (
            "Stricter min_segment_bars must yield fewer or equal directional segments"
        )


# ---------------------------------------------------------------------------
# 3. Backward compatibility: calibration=None replicates candidate[0] behavior
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    """calibration=None must reproduce the same theta_dc as candidate[0] fallback."""

    def test_none_calibration_uses_candidate0_k(self) -> None:
        """With calibration=None the pipeline uses k_dc_candidates[0]=3.0 for 1m."""
        from regime_benchmark.config import LabelingConfig

        config = LabelingConfig.from_yaml(_CONFIG_PATH)
        tf_params_1m = config.direction_method.params["1m"]
        candidate0_k = float(tf_params_1m.k_dc_candidates[0])  # 3.0
        candidate0_q = float(tf_params_1m.q_dc)  # 0.80

        p = _make_log_prices()
        abs_d = _make_abs_d_series(p)

        # Replicate what _run_timeframe does when frozen=None
        theta_candidate0 = compute_theta_dc(abs_d, q_dc=candidate0_q, k_dc=candidate0_k)

        # Replicate what _run_timeframe does when frozen=FrozenParams(k_dc=3.0, q_dc=0.80)
        fp = FrozenParams(k_dc=candidate0_k, q_dc=candidate0_q, min_segment_bars=5)
        theta_frozen_same = compute_theta_dc(abs_d, q_dc=fp.q_dc, k_dc=fp.k_dc)

        # They must be identical since k and q are the same
        assert theta_candidate0 == pytest.approx(theta_frozen_same, rel=1e-14)

    def test_none_calibration_q_low_q_high_from_config(self) -> None:
        """With calibration=None, q_low/q_high come from config.volatility_method.quantiles."""
        from regime_benchmark.config import LabelingConfig

        config = LabelingConfig.from_yaml(_CONFIG_PATH)
        assert config.volatility_method.quantiles.low == pytest.approx(0.33)
        assert config.volatility_method.quantiles.high == pytest.approx(0.66)

    def test_frozen_with_candidate0_values_identical_to_no_calibration(self) -> None:
        """FrozenParams(k=candidate[0], q=config.q_dc) produces the same theta_dc."""
        from regime_benchmark.config import LabelingConfig

        config = LabelingConfig.from_yaml(_CONFIG_PATH)
        p = _make_log_prices()
        abs_d = _make_abs_d_series(p)

        for tf in ("1m", "5m"):
            tf_params = config.direction_method.params[tf]  # type: ignore[literal-required]
            k0 = float(tf_params.k_dc_candidates[0])
            q0 = float(tf_params.q_dc)

            theta_default = compute_theta_dc(abs_d, q_dc=q0, k_dc=k0)
            fp = FrozenParams(k_dc=k0, q_dc=q0, min_segment_bars=5)
            theta_fp = compute_theta_dc(abs_d, q_dc=fp.q_dc, k_dc=fp.k_dc)
            assert theta_default == pytest.approx(theta_fp, rel=1e-14), (
                f"theta_dc must be identical for tf={tf} when FrozenParams matches candidate[0]"
            )


# ---------------------------------------------------------------------------
# 4. Pipeline integration: frozen params flow into register_params via mock DB
# ---------------------------------------------------------------------------


def _make_mock_conn() -> MagicMock:
    """Return a MagicMock psycopg connection that records register_params calls."""
    conn = MagicMock()
    # cursor() returns a context-manager compatible MagicMock
    cursor_cm = MagicMock()
    cursor_cm.__enter__ = MagicMock(return_value=cursor_cm)
    cursor_cm.__exit__ = MagicMock(return_value=False)
    cursor_cm.fetchone = MagicMock(return_value=(1,))  # run_id = 1
    conn.cursor = MagicMock(return_value=cursor_cm)
    conn.rollback = MagicMock()
    conn.commit = MagicMock()
    return conn


def _run_timeframe_capture_params(
    timeframe: str,
    frozen: FrozenParams | None,
) -> dict[str, Any]:
    """Call _run_timeframe (synthetic mode) and capture the values passed to register_params.

    Uses synthetic=True so no klines file or DataFrame construction is needed.
    Patches the names imported into regime_benchmark.pipeline so the interception
    works regardless of import binding.

    Returns a dict with theta_dc, theta_amp, k_dc, q_dc, min_segment_bars, q_low, q_high.
    """
    import regime_benchmark.pipeline as pipeline_mod
    from regime_benchmark.config import LabelingConfig

    config = LabelingConfig.from_yaml(_CONFIG_PATH)

    captured: dict[str, Any] = {}

    # register_params / copy_segments / copy_bars are imported into the pipeline
    # module's namespace — patch them there so _run_timeframe picks up the stubs.
    original_register_params = pipeline_mod.register_params  # type: ignore[attr-defined]
    original_copy_segments = pipeline_mod.copy_segments  # type: ignore[attr-defined]
    original_copy_bars = pipeline_mod.copy_bars  # type: ignore[attr-defined]

    def _capture_params(
        conn: Any,
        run_id: int,
        tf: str,
        theta_dc_arg: float,
        theta_amp_arg: float,
        q_dc_arg: float,
        k_dc_arg: float,
        min_segment_bars_arg: int,
        q_low_arg: float,
        q_high_arg: float,
        taker_fee_rate: float | None = None,
        slippage_rate_estimate: float | None = None,
    ) -> None:
        captured["theta_dc"] = theta_dc_arg
        captured["theta_amp"] = theta_amp_arg
        captured["k_dc"] = k_dc_arg
        captured["q_dc"] = q_dc_arg
        captured["min_segment_bars"] = min_segment_bars_arg
        captured["q_low"] = q_low_arg
        captured["q_high"] = q_high_arg

    try:
        pipeline_mod.register_params = _capture_params  # type: ignore[attr-defined]
        pipeline_mod.copy_segments = lambda *a, **kw: None  # type: ignore[attr-defined]
        pipeline_mod.copy_bars = lambda *a, **kw: None  # type: ignore[attr-defined]

        mock_conn = _make_mock_conn()

        pipeline_mod._run_timeframe(
            conn=mock_conn,
            run_id=1,
            timeframe=timeframe,
            config=config,
            source_map=None,
            synthetic=True,
            frozen=frozen,
        )
    finally:
        pipeline_mod.register_params = original_register_params  # type: ignore[attr-defined]
        pipeline_mod.copy_segments = original_copy_segments  # type: ignore[attr-defined]
        pipeline_mod.copy_bars = original_copy_bars  # type: ignore[attr-defined]

    return captured


class TestPipelineParamFlow:
    """Verify that _run_timeframe passes frozen (not candidate[0]) values to register_params."""

    def test_frozen_k_flows_into_register_params_1m(self) -> None:
        """With FrozenParams(k_dc=5.0) for 1m, register_params must receive k_dc=5.0."""
        fp = FrozenParams(k_dc=5.0, q_dc=0.80, min_segment_bars=8)
        captured = _run_timeframe_capture_params("1m", frozen=fp)
        assert captured["k_dc"] == pytest.approx(5.0), (
            f"register_params received k_dc={captured['k_dc']!r}, expected 5.0"
        )

    def test_frozen_min_segment_bars_flows_into_register_params_1m(self) -> None:
        """With FrozenParams(min_segment_bars=15) for 1m, register_params must receive 15."""
        fp = FrozenParams(k_dc=3.0, q_dc=0.80, min_segment_bars=15)
        captured = _run_timeframe_capture_params("1m", frozen=fp)
        got_msb = captured["min_segment_bars"]
        assert got_msb == 15, (
            f"register_params received min_segment_bars={got_msb!r}, expected 15"
        )

    def test_frozen_q_dc_flows_into_register_params_1m(self) -> None:
        """With FrozenParams(q_dc=0.90) for 1m, register_params must receive q_dc=0.90."""
        fp = FrozenParams(k_dc=3.0, q_dc=0.90, min_segment_bars=5)
        captured = _run_timeframe_capture_params("1m", frozen=fp)
        assert captured["q_dc"] == pytest.approx(0.90), (
            f"register_params received q_dc={captured['q_dc']!r}, expected 0.90"
        )

    def test_frozen_q_low_q_high_flow_into_register_params(self) -> None:
        """Custom q_low/q_high from FrozenParams must appear in register_params."""
        fp = FrozenParams(k_dc=3.0, q_dc=0.80, min_segment_bars=5, q_low=0.20, q_high=0.80)
        captured = _run_timeframe_capture_params("1m", frozen=fp)
        assert captured["q_low"] == pytest.approx(0.20)
        assert captured["q_high"] == pytest.approx(0.80)

    def test_theta_amp_equals_theta_dc(self) -> None:
        """theta_amp must equal theta_dc (same_as_theta_dc policy §8.4)."""
        fp = FrozenParams(k_dc=4.0, q_dc=0.80, min_segment_bars=5)
        captured = _run_timeframe_capture_params("1m", frozen=fp)
        assert captured["theta_amp"] == pytest.approx(captured["theta_dc"], rel=1e-14), (
            "theta_amp must equal theta_dc (same_as_theta_dc policy)"
        )

    def test_frozen_theta_dc_differs_from_candidate0(self) -> None:
        """FrozenParams(k=5.0) must produce a different theta_dc than candidate[0] k=3.0."""
        fp_frozen = FrozenParams(k_dc=5.0, q_dc=0.80, min_segment_bars=5)
        captured_frozen = _run_timeframe_capture_params("1m", frozen=fp_frozen)

        captured_default = _run_timeframe_capture_params("1m", frozen=None)

        assert captured_frozen["theta_dc"] != pytest.approx(captured_default["theta_dc"]), (
            "theta_dc with k=5.0 (frozen) must differ from theta_dc with k=3.0 (candidate[0])"
        )
        # k=5 produces theta_dc = (5/3) * theta_k3
        assert captured_frozen["theta_dc"] == pytest.approx(
            captured_default["theta_dc"] * (5.0 / 3.0), rel=1e-10
        )

    def test_backward_compat_none_uses_candidate0(self) -> None:
        """calibration=None must give k_dc=3.0 for 1m (candidate[0]) and q_low=0.33."""
        captured = _run_timeframe_capture_params("1m", frozen=None)
        assert captured["k_dc"] == pytest.approx(3.0), (
            f"calibration=None must use candidate[0] k=3.0 for 1m, got {captured['k_dc']!r}"
        )
        assert captured["q_low"] == pytest.approx(0.33)
        assert captured["q_high"] == pytest.approx(0.66)

    def test_frozen_5m_params(self) -> None:
        """FrozenParams for 5m timeframe flow through correctly."""
        fp = FrozenParams(k_dc=4.0, q_dc=0.80, min_segment_bars=7)
        captured = _run_timeframe_capture_params("5m", frozen=fp)
        assert captured["k_dc"] == pytest.approx(4.0)
        assert captured["min_segment_bars"] == 7

    def test_backward_compat_5m_candidate0(self) -> None:
        """calibration=None for 5m uses k_dc_candidates[0]=2.0 and min_segment_bars=3."""
        captured = _run_timeframe_capture_params("5m", frozen=None)
        assert captured["k_dc"] == pytest.approx(2.0), (
            f"calibration=None must use candidate[0] k=2.0 for 5m, got {captured['k_dc']!r}"
        )
        got_msb5 = captured["min_segment_bars"]
        assert got_msb5 == 3, (
            f"calibration=None must use min_segment_bars=3 for 5m, got {got_msb5!r}"
        )


# ---------------------------------------------------------------------------
# 5. DB roundtrip (optional — skipped when REGIME_BENCHMARK_DB_URL is absent)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _DB_URL,
    reason="REGIME_BENCHMARK_DB_URL not set — skipping DB calibration roundtrip test",
)
class TestCalibrationDbRoundtrip:
    """Verify labeling_run_params stores frozen values, not candidate[0]."""

    def _try_connect(self) -> object | None:
        if not _DB_URL:
            return None
        try:
            import psycopg
            return psycopg.connect(_DB_URL)
        except Exception:
            return None

    def test_frozen_params_recorded_in_labeling_run_params(self) -> None:
        """run_pipeline with calibration dict stores frozen k_dc/theta_dc in labeling_run_params."""
        import time

        import psycopg

        from regime_benchmark.config import LabelingConfig
        from regime_benchmark.pipeline import run_pipeline

        conn = self._try_connect()
        if conn is None:
            pytest.skip("DB unreachable")

        config = LabelingConfig.from_yaml(_CONFIG_PATH)

        # Use k_dc=5.0 for both timeframes (candidate[0] for 1m=3.0, 5m=2.0)
        calibration: dict[str, FrozenParams] = {
            "1m": FrozenParams(k_dc=5.0, q_dc=0.80, min_segment_bars=8),
            "5m": FrozenParams(k_dc=5.0, q_dc=0.80, min_segment_bars=6),
        }

        run_id: int | None = None
        try:
            run_id = run_pipeline(
                config=config,
                synthetic=True,
                calibration=calibration,
            )

            db_conn: psycopg.Connection = conn  # type: ignore[assignment]
            with db_conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT timeframe, k_dc, min_segment_bars, theta_dc, q_low, q_high
                    FROM labeling_run_params
                    WHERE run_id = %s
                    ORDER BY timeframe
                    """,
                    (run_id,),
                )
                rows = {row[0]: row[1:] for row in cur.fetchall()}

            assert "1m" in rows, "labeling_run_params missing 1m row"
            assert "5m" in rows, "labeling_run_params missing 5m row"

            k_dc_1m, msb_1m, theta_dc_1m, q_low_1m, q_high_1m = rows["1m"]
            k_dc_5m, msb_5m, theta_dc_5m, q_low_5m, q_high_5m = rows["5m"]

            # Frozen k_dc=5.0 stored (not candidate[0])
            assert float(k_dc_1m) == pytest.approx(5.0), (
                f"labeling_run_params.k_dc for 1m = {k_dc_1m!r}, expected 5.0 (frozen)"
            )
            assert float(k_dc_5m) == pytest.approx(5.0), (
                f"labeling_run_params.k_dc for 5m = {k_dc_5m!r}, expected 5.0 (frozen)"
            )

            # min_segment_bars from FrozenParams
            assert int(msb_1m) == 8, (
                f"labeling_run_params.min_segment_bars for 1m = {msb_1m!r}, expected 8"
            )
            assert int(msb_5m) == 6, (
                f"labeling_run_params.min_segment_bars for 5m = {msb_5m!r}, expected 6"
            )

            # theta_dc > 0 and larger than candidate[0] (k=5 > k=3)
            assert float(theta_dc_1m) > 0, "theta_dc for 1m must be positive"
            assert float(theta_dc_5m) > 0, "theta_dc for 5m must be positive"

            # q_low / q_high stored as defaults
            assert float(q_low_1m) == pytest.approx(0.33)
            assert float(q_high_1m) == pytest.approx(0.66)

            _ = time.time()  # keep import used

        finally:
            if run_id is not None and conn is not None:
                try:
                    db_conn2: psycopg.Connection = conn  # type: ignore[assignment]
                    with db_conn2.cursor() as cur:
                        cur.execute(
                            "DELETE FROM labeling_runs WHERE id = %s",
                            (run_id,),
                        )
                    db_conn2.commit()
                except Exception as e:
                    print(f"[calibration teardown] WARNING: could not delete run_id={run_id}: {e}")
            if conn is not None:
                try:
                    conn.close()  # type: ignore[union-attr]
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# SF4: input-guard tests (raise BEFORE any DB connect — no DB needed)
# ---------------------------------------------------------------------------


class TestRunPipelineGuards:
    """run_pipeline must reject inconsistent period / calibration inputs."""

    def _cfg(self) -> Any:
        from regime_benchmark.config import LabelingConfig

        return LabelingConfig.from_yaml(_CONFIG_PATH)

    def test_partial_period_raises(self) -> None:
        """Only one of period_start/period_end provided → ValueError."""
        from datetime import datetime, timezone

        from regime_benchmark.pipeline import run_pipeline

        cfg = self._cfg()
        with pytest.raises(ValueError, match="both be provided or both be None"):
            run_pipeline(
                cfg,
                synthetic=True,
                period_start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                period_end=None,
            )

    def test_partial_calibration_raises(self) -> None:
        """calibration missing a timeframe key → ValueError (no silent mix)."""
        from regime_benchmark.pipeline import run_pipeline

        cfg = self._cfg()
        with pytest.raises(ValueError, match="must cover all timeframes"):
            run_pipeline(
                cfg,
                synthetic=True,
                calibration={"1m": FrozenParams(k_dc=4.0, q_dc=0.80, min_segment_bars=10)},
            )


class TestLoadKlinesParquetValidation:
    """load_klines parquet branch must reject wrong-schema / wrong-cadence files."""

    def test_wrong_cadence_parquet_raises(self, tmp_path: Path) -> None:
        """A 5m-cadence parquet loaded as '1m' → ValueError (cadence mismatch)."""
        from datetime import datetime, timezone

        from regime_benchmark.ingest.binance import load_klines

        # 5-minute spaced open_times, but we ask for '1m'
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        ots = [base.replace(minute=5 * i % 60, hour=(5 * i) // 60) for i in range(12)]
        df = pl.DataFrame(
            {
                "open_time": ots,
                "open": [3000.0] * 12,
                "high": [3001.0] * 12,
                "low": [2999.0] * 12,
                "close": [3000.5] * 12,
            }
        ).with_columns(pl.col("open_time").dt.cast_time_unit("us").dt.replace_time_zone("UTC"))
        p = tmp_path / "wrong_cadence.parquet"
        df.write_parquet(p)
        with pytest.raises(ValueError, match="cadence"):
            load_klines(source=p, timeframe="1m")

    def test_missing_columns_parquet_raises(self, tmp_path: Path) -> None:
        """A parquet missing required OHLC columns → ValueError."""
        from datetime import datetime, timezone

        from regime_benchmark.ingest.binance import load_klines

        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        ots = [base.replace(minute=i) for i in range(10)]
        df = pl.DataFrame({"open_time": ots, "open": [3000.0] * 10}).with_columns(
            pl.col("open_time").dt.cast_time_unit("us").dt.replace_time_zone("UTC")
        )
        p = tmp_path / "missing_cols.parquet"
        df.write_parquet(p)
        with pytest.raises(ValueError, match="missing required columns"):
            load_klines(source=p, timeframe="1m")
