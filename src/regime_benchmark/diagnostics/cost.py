"""Tradeability / cost diagnostics — requirements.md §6.5.2.

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


def compute_cost_diagnostics(
    segments: pl.DataFrame,
    taker_fee_rate: float,
    slippage_rate_estimate: float,
) -> pl.DataFrame:
    """Add cost / tradeability diagnostic columns to the segments DataFrame.

    Args:
        segments: Segment DataFrame with 'amplitude' column (A_j, float64).
        taker_fee_rate: Taker fee as a log-scale rate (e.g. 0.0004).
        slippage_rate_estimate: Estimated slippage as a log-scale rate.

    Returns:
        segments DataFrame with columns added:
        estimated_round_trip_cost_log, amplitude_to_cost_ratio,
        low_tradeability_segment_flag.

    Raises:
        NotImplementedError: Implementation deferred to Milestone M5.
    """
    raise NotImplementedError("compute_cost_diagnostics is implemented in Milestone M5")
