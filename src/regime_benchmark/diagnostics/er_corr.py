"""ER–RV correlation diagnostic — requirements.md §7.3.2 / design §10.2.

Computes:
  rho_er_vol_tau = SpearmanCorr(ER_j, RV_per_bar_j)  per timeframe tau

Interpretation thresholds (design §10.2):
  |rho| < 0.60       -> low redundancy, ER usable as feature
  0.60 <= |rho| < 0.80  -> possible redundancy, monitor
  |rho| >= 0.80      -> high redundancy, ER limited to diagnostic use only

POST-HOC ONLY (design §10.3): This diagnostic is computed after all segments
are finalised.  It must NOT be used as a forward-looking model feature or live
signal — segment ER values become known only after the segment closes.

TIMEFRAME INDEPENDENCE (design §9.1, §10.2): Always call per timeframe.
Never pass a mixed 1m+5m population.  rho_er_vol_1m and rho_er_vol_5m are
separate quantities and must be stored / interpreted independently.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _average_rank(arr: np.ndarray) -> np.ndarray:
    """Compute average ranks for a 1-D float64 array, handling ties.

    Standard scipy.stats.rankdata behaviour, implemented without scipy.

    Args:
        arr: 1-D numpy float64 array (no NaN).

    Returns:
        Float64 array of average ranks (1-based).
    """
    n = len(arr)
    sorter = np.argsort(arr, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    ranks[sorter] = np.arange(1, n + 1, dtype=np.float64)

    # Resolve ties: for each group of equal values, replace rank with average
    sorted_arr = arr[sorter]
    # Identify tie-group boundaries
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_arr[j] == sorted_arr[i]:
            j += 1
        if j > i + 1:  # tie group [i, j)
            avg = float(np.mean(ranks[sorter[i:j]]))
            for k in range(i, j):
                ranks[sorter[k]] = avg
        i = j
    return ranks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_er_vol_correlation(
    efficiency_ratios: Sequence[float],
    rv_per_bars: Sequence[float],
) -> float:
    """Compute Spearman rho between ER_j and RV_per_bar_j for one timeframe.

    Implementation uses average-rank Spearman (no scipy):
      1. Rank both arrays with tie-averaging.
      2. Compute Pearson correlation on the ranks.

    POST-HOC ONLY (design §10.3): Do not use as a forward model feature.
    Call once per timeframe population; never mix 1m and 5m (design §10.2).

    Args:
        efficiency_ratios: Sequence of ER_j values (float64, 0 <= ER <= 1).
            Tail/unconfirmed segments should be excluded by the caller.
        rv_per_bars: Sequence of RV_per_bar_j values (float64, >= 0).
            Must have the same length as efficiency_ratios.

    Returns:
        Spearman rho as float64 in [-1.0, 1.0].
        Returns 0.0 for edge cases:
          - fewer than 2 observations,
          - zero variance in either ranked array (constant input).

    Raises:
        ValueError: If efficiency_ratios and rv_per_bars differ in length.
    """
    er = np.asarray(efficiency_ratios, dtype=np.float64)
    rv = np.asarray(rv_per_bars, dtype=np.float64)

    if len(er) != len(rv):
        raise ValueError(
            f"efficiency_ratios length {len(er)} != rv_per_bars length {len(rv)}"
        )

    n = len(er)
    if n < 2:
        return 0.0

    r_er = _average_rank(er)
    r_rv = _average_rank(rv)

    # Pearson on ranks
    er_mean = float(np.mean(r_er))
    rv_mean = float(np.mean(r_rv))
    er_dev = r_er - er_mean
    rv_dev = r_rv - rv_mean

    numerator = float(np.sum(er_dev * rv_dev))
    denom = float(np.sqrt(np.sum(er_dev**2) * np.sum(rv_dev**2)))

    if denom == 0.0:
        # Zero variance → Spearman undefined → return 0.0 (diagnostic safe)
        return 0.0

    rho = numerator / denom
    # Clamp to [-1, 1] to guard against floating-point overshoot
    return float(np.clip(rho, -1.0, 1.0))


def classify_er_vol_overlap(rho: float) -> str:
    """Classify ER–RV redundancy level from Spearman rho.

    Thresholds per design §10.2:
      |rho| < 0.60       -> "low"
      0.60 <= |rho| < 0.80  -> "possible"
      |rho| >= 0.80      -> "strong"

    POST-HOC ONLY: Result is a diagnostic annotation, not a signal.

    Args:
        rho: Spearman correlation value, typically from
             compute_er_vol_correlation(); float in [-1, 1].

    Returns:
        One of "low", "possible", "strong".
    """
    abs_rho = abs(rho)
    if abs_rho >= 0.80:
        return "strong"
    elif abs_rho >= 0.60:
        return "possible"
    else:
        return "low"
