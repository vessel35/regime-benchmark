"""Segment metric computation and direction labeling — requirements.md §6.3-6.4.

For each segment S_j = [TP_j, TP_{j+1}]:
  N_j  = end_j - start_j + 1
  M_j  = p_end - p_start
  A_j  = |M_j|
  L_j  = sum(|d_t|) over (start_j+1 .. end_j)
  ER_j = A_j / L_j  (0 if L_j == 0)

Direction label:
  UP   if M_j > 0 and N_j >= min_segment_bars and A_j >= theta_amp
  DOWN if M_j < 0 and N_j >= min_segment_bars and A_j >= theta_amp
  NON_DIRECTIONAL otherwise

Note: ER_j is NOT a direction condition — see design §8.4 / Case E.
"""

from __future__ import annotations

import polars as pl

from regime_benchmark.direction.dc_engine import TurningPoint


def compute_segment_metrics(
    turning_points: list[TurningPoint],
    log_prices: list[float],
) -> pl.DataFrame:
    """Compute N_j, M_j, A_j, L_j, ER_j for each confirmed segment.

    Args:
        turning_points: Chronological list of turning points from dc_engine.
        log_prices: Full log-price series p_t indexed by bar position.

    Returns:
        DataFrame with one row per segment and columns:
        segment_id, start_bar, end_bar, confirm_bar, is_tail_unconfirmed,
        N_j, log_move (M_j), amplitude (A_j), path_length (L_j),
        efficiency_ratio (ER_j).

    Raises:
        NotImplementedError: Implementation deferred to Milestone M3.
    """
    raise NotImplementedError("compute_segment_metrics is implemented in Milestone M3")


def assign_direction_labels(
    segments: pl.DataFrame,
    min_segment_bars: int,
    theta_amp: float,
) -> pl.DataFrame:
    """Add direction_label column to segment DataFrame.

    Args:
        segments: Output of compute_segment_metrics.
        min_segment_bars: Minimum bar count for a directional segment.
        theta_amp: Minimum amplitude for a directional segment.

    Returns:
        segments DataFrame with 'direction_label' column added
        ('UP', 'DOWN', or 'NON_DIRECTIONAL').

    Raises:
        NotImplementedError: Implementation deferred to Milestone M3.
    """
    raise NotImplementedError("assign_direction_labels is implemented in Milestone M3")
