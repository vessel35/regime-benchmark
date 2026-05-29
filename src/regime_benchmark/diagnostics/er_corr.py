"""ER–RV correlation diagnostic — requirements.md §7.3.2.

Computes:
  rho_er_vol_tau = SpearmanCorr(ER_j, RV_per_bar_j)  per timeframe tau

Interpretation thresholds:
  |rho| < 0.60  -> low redundancy, ER usable as feature
  0.60 <= |rho| < 0.80 -> possible redundancy, monitor
  |rho| >= 0.80 -> high redundancy, ER limited to diagnostic use only

Must be called per timeframe; never mix 1m and 5m segments.
"""

from __future__ import annotations

import polars as pl


def compute_er_vol_correlation(
    segments: pl.DataFrame,
) -> float:
    """Compute Spearman correlation between ER_j and RV_per_bar_j.

    Args:
        segments: Segment DataFrame with 'efficiency_ratio' and
                  'realized_volatility_per_bar' columns (float64).
                  Must contain only one timeframe's segments.

    Returns:
        rho_er_vol_tau as float64 in [-1, 1].

    Raises:
        NotImplementedError: Implementation deferred to Milestone M5.
    """
    raise NotImplementedError(
        "compute_er_vol_correlation is implemented in Milestone M5"
    )
