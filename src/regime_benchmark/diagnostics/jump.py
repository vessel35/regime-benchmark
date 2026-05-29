"""Jump / bipower-variation diagnostics — requirements.md §7.3.1.

Computes:
  max_abs_d_j        = max(|d_t|) for t in S_j
  max_jump_share_j   = max(d_t^2) / sum(d_t^2)   (0 if sum == 0)
  BV_j               = (pi/2) * sum(|d_t| * |d_{t-1}|)  for t in (start+2..end)
  jump_component_j   = max(RV_j^2 - BV_j, 0)
  jump_share_bv_j    = jump_component_j / RV_j^2   (0 if RV_j == 0)

These values are post-hoc diagnostics only; they do not affect final_label.
"""

from __future__ import annotations

import polars as pl


def compute_jump_diagnostics(
    segments: pl.DataFrame,
    log_returns: pl.Series,
) -> pl.DataFrame:
    """Add jump and bipower-variation diagnostic columns to the segments DataFrame.

    Args:
        segments: Segment DataFrame with start_bar, end_bar,
                  realized_volatility columns.
        log_returns: Full d_t series (float64).

    Returns:
        segments DataFrame with columns added:
        max_abs_d, max_jump_share, bipower_variation,
        jump_component, jump_share_bv.

    Raises:
        NotImplementedError: Implementation deferred to Milestone M5.
    """
    raise NotImplementedError("compute_jump_diagnostics is implemented in Milestone M5")
