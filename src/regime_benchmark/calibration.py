"""Frozen calibration parameters for the Regime 9-Label Benchmark pipeline.

Design §8.1 (frozen calibration policy):
  The calibration SCALE FACTORS (k_dc, q_dc) are chosen on a pre-designated
  calibration split and then locked — NOT the computed theta_dc value itself.
  theta_dc is re-derived from each run's own |d| distribution using the frozen
  (k_dc, q_dc), satisfying causality (no future leak) while keeping the
  sensitivity pre-decided and reproducible.

FrozenParams holds the per-timeframe calibration values chosen on the calibration
split.  The pipeline stores these verbatim into labeling_run_params so downstream
consumers can reconstruct the exact threshold.

theta_amp = theta_dc (same_as_theta_dc policy, §8.4) — derived inside the pipeline,
not stored as a separate frozen field.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FrozenParams:
    """Pre-decided per-timeframe calibration parameters (frozen after calibration split).

    Attributes:
        k_dc: Scale factor for theta_dc = Quantile(|d|, q_dc) * k_dc.
        q_dc: Quantile level for theta_dc (e.g. 0.80).
        min_segment_bars: Minimum bar count for a directional segment (§8.4).
        q_low: Lower volatility quantile position (default 0.33, §9).
        q_high: Upper volatility quantile position (default 0.66, §9).

    Notes:
        - k_dc and q_dc together determine theta_dc when applied to the run's |d|.
        - theta_amp is NOT stored here; the pipeline derives it as theta_amp = theta_dc
          per the same_as_theta_dc policy (§8.4 / design §13.2).
        - q_low and q_high are quantile *positions* (not boundary values); the actual
          RV_per_bar boundaries are computed from the run's segments per the frozen-
          calibration semantics of §9.
    """

    k_dc: float
    q_dc: float
    min_segment_bars: int
    q_low: float = 0.33
    q_high: float = 0.66
