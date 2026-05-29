"""Tradeability / cost diagnostics — requirements.md §6.5.2 / design §10.1.

Computes:
  estimated_round_trip_cost_log  = 2 * taker_fee_rate + slippage_rate_estimate
  amplitude_to_cost_ratio_j      = A_j / estimated_round_trip_cost_log
  low_tradeability_segment_flag  = (amplitude_to_cost_ratio_j < 3)

[RISK] These are descriptive diagnostics, NOT expected-profitability estimates.
Size, liquidity, fill timing, and order book depth are not modelled (§10.3 S4-S5).
Do NOT use as live signal or model feature without explicit causal gating.
"""

from __future__ import annotations

import polars as pl

from regime_benchmark.direction.segments import Segment


def compute_cost_diagnostics_segments(
    segments: list[Segment],
    taker_fee_rate: float,
    slippage_rate_estimate: float,
) -> list[Segment]:
    """Add cost diagnostic fields to each Segment in-place.

    Args:
        segments: List of Segment objects with amplitude field set.
        taker_fee_rate: Taker fee as a log-scale rate (e.g. 0.0004).
        slippage_rate_estimate: Estimated slippage as a log-scale rate.

    Returns:
        The same list with amplitude_to_cost_ratio and
        low_tradeability_segment_flag updated.
    """
    cost = 2.0 * taker_fee_rate + slippage_rate_estimate
    for seg in segments:
        if cost > 0:
            ratio = seg.amplitude / cost
        else:
            ratio = float("inf")
        seg.amplitude_to_cost_ratio = ratio
        seg.low_tradeability_segment_flag = ratio < 3.0
    return segments


def compute_cost_diagnostics(
    segments: pl.DataFrame,
    taker_fee_rate: float,
    slippage_rate_estimate: float,
) -> pl.DataFrame:
    """Add cost / tradeability diagnostic columns to the segments DataFrame.

    Legacy polars API.

    Args:
        segments: Segment DataFrame with 'amplitude' column (A_j, float64).
        taker_fee_rate: Taker fee as a log-scale rate (e.g. 0.0004).
        slippage_rate_estimate: Estimated slippage as a log-scale rate.

    Returns:
        segments DataFrame with columns added:
        amplitude_to_cost_ratio, low_tradeability_segment_flag.
    """
    cost = 2.0 * taker_fee_rate + slippage_rate_estimate

    if cost > 0:
        df = segments.with_columns(
            (pl.col("amplitude") / cost).alias("amplitude_to_cost_ratio"),
        )
    else:
        df = segments.with_columns(
            pl.lit(float("inf")).alias("amplitude_to_cost_ratio"),
        )

    df = df.with_columns(
        (pl.col("amplitude_to_cost_ratio") < 3.0).alias("low_tradeability_segment_flag"),
    )
    return df
