"""Segment realized volatility computation — requirements.md §7 / design §9.

RV_j        = sqrt(sum(d_t^2))   for t in (start_j+1 .. end_j)
RV_per_bar_j = RV_j / sqrt(N_j)

NOTE: denominator is sqrt(N_j), NOT sqrt(N_j-1). This matches spec §7.1 exactly.
Do NOT correct this to the sample-std denominator — unit tests pin this formula.

Volatility labels (timeframe-specific quantiles, never cross-timeframe):
  LOW_VOL  if RV_per_bar_j <= Q_low_tau
  MID_VOL  if Q_low_tau < RV_per_bar_j <= Q_high_tau
  HIGH_VOL if RV_per_bar_j > Q_high_tau
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl

from regime_benchmark.direction.segments import Segment


def realized_vol(d_segment: np.ndarray) -> float:
    """Compute RV_j = sqrt(sum(d_t^2)) for the given log-return slice.

    NaN values are treated as zero contribution (nansum).

    Args:
        d_segment: Array of log returns d_t for bars (start+1 .. end) inclusive.

    Returns:
        RV_j >= 0.0 (float64).
    """
    return float(math.sqrt(float(np.nansum(d_segment**2))))


def rv_per_bar(rv: float, n: int) -> float:
    """Compute RV_per_bar_j = RV_j / sqrt(N_j).

    Denominator is sqrt(N_j) per spec §7.1.  Do NOT change to sqrt(N_j-1).

    Args:
        rv: Realized volatility RV_j (>= 0).
        n: Segment bar count N_j (>= 1).

    Returns:
        RV_per_bar_j >= 0 (float64).  Returns 0.0 if n == 0.
    """
    if n <= 0:
        return 0.0
    return rv / math.sqrt(n)


def assign_volatility(score: float, q_low: float, q_high: float) -> str:
    """Assign volatility label from RV_per_bar score and quantile boundaries.

    Args:
        score: RV_per_bar_j for the segment.
        q_low: Q_low_tau boundary value (not level).
        q_high: Q_high_tau boundary value (not level).

    Returns:
        'LOW_VOL', 'MID_VOL', or 'HIGH_VOL'.
    """
    if score <= q_low:
        return "LOW_VOL"
    elif score <= q_high:
        return "MID_VOL"
    else:
        return "HIGH_VOL"


def compute_segment_rv(
    segments: list[Segment],
    d: np.ndarray,
) -> list[Segment]:
    """Compute RV_j and RV_per_bar_j for each segment in-place.

    Modifies each Segment's realized_volatility and realized_volatility_per_bar.

    Args:
        segments: List of Segment objects (confirmed + tail).
        d: Full log-return array (float64, length N; d[0] is NaN).

    Returns:
        The same list with RV fields updated.
    """
    for seg in segments:
        start = seg.start_bar
        end = seg.end_bar
        # d_slice: bars start+1 .. end (inclusive)
        d_slice = d[start + 1 : end + 1]
        rv = realized_vol(d_slice)
        seg.realized_volatility = rv
        seg.realized_volatility_per_bar = rv_per_bar(rv, seg.n_bars)
    return segments


def compute_volatility_quantiles(
    rv_per_bar_values: list[float],
    q_low: float,
    q_high: float,
) -> tuple[float, float]:
    """Compute Q_low and Q_high threshold values from the RV_per_bar distribution.

    Must be called per timeframe; never mix 1m and 5m series (§7.2, §9.1).

    Args:
        rv_per_bar_values: List of RV_per_bar_j values for one timeframe (float64).
            Tail segments (is_tail_unconfirmed=True) should be excluded.
        q_low: Lower quantile level, e.g. 0.33.
        q_high: Upper quantile level, e.g. 0.66.

    Returns:
        Tuple (Q_low_tau, Q_high_tau) as float64 score boundaries.

    Raises:
        ValueError: If rv_per_bar_values is empty.
    """
    if not rv_per_bar_values:
        raise ValueError("rv_per_bar_values is empty; cannot compute quantiles")
    arr = np.array(rv_per_bar_values, dtype=np.float64)
    q_low_val = float(np.nanquantile(arr, q_low))
    q_high_val = float(np.nanquantile(arr, q_high))
    return q_low_val, q_high_val


def assign_volatility_labels(
    segments: list[Segment],
    q_low_value: float,
    q_high_value: float,
) -> list[Segment]:
    """Add volatility_label to each confirmed segment in-place.

    Tail segments (is_tail_unconfirmed=True) retain volatility_label=None.

    Args:
        segments: List of Segment objects with realized_volatility_per_bar set.
        q_low_value: Computed Q_low threshold score value.
        q_high_value: Computed Q_high threshold score value.

    Returns:
        The same list with volatility_label fields updated.
    """
    for seg in segments:
        if seg.is_tail_unconfirmed:
            seg.volatility_label = None
        else:
            seg.volatility_label = assign_volatility(
                seg.realized_volatility_per_bar, q_low_value, q_high_value
            )
    return segments


# ---------------------------------------------------------------------------
# Polars-based legacy API (kept for M1 stubs that expected polars DataFrames)
# ---------------------------------------------------------------------------


def compute_realized_volatility(
    segments: pl.DataFrame,
    log_returns: pl.Series,
) -> pl.DataFrame:
    """Add RV_j and RV_per_bar_j columns to the segments DataFrame.

    Legacy polars API.  The primary M2 path uses compute_segment_rv().

    Args:
        segments: Segment DataFrame with start_bar, end_bar, n_bars columns.
        log_returns: Full d_t series (float64); first value may be NaN.

    Returns:
        segments DataFrame with 'realized_volatility' (RV_j) and
        'realized_volatility_per_bar' (RV_per_bar_j) columns added.
    """
    d_arr = np.array(log_returns.to_list(), dtype=np.float64)
    rvs = []
    rv_pb = []
    for row in segments.iter_rows(named=True):
        start = row["start_bar"]
        end = row["end_bar"]
        n = row["n_bars"]
        d_slice = d_arr[start + 1 : end + 1]
        rv = realized_vol(d_slice)
        rvs.append(rv)
        rv_pb.append(rv_per_bar(rv, n))

    return segments.with_columns(
        pl.Series("realized_volatility", rvs, dtype=pl.Float64),
        pl.Series("realized_volatility_per_bar", rv_pb, dtype=pl.Float64),
    )


def compute_volatility_quantiles_series(
    rv_per_bar: pl.Series,
    q_low: float,
    q_high: float,
) -> tuple[float, float]:
    """Compute Q_low and Q_high threshold values from the RV_per_bar distribution.

    Must be called per timeframe; never mix 1m and 5m series (§7.2).

    Args:
        rv_per_bar: Series of RV_per_bar_j values for one timeframe (float64).
        q_low: Lower quantile level, e.g. 0.33.
        q_high: Upper quantile level, e.g. 0.66.

    Returns:
        Tuple (Q_low_tau, Q_high_tau) as float64 values.
    """
    values = rv_per_bar.drop_nulls().drop_nans().to_list()
    return compute_volatility_quantiles(values, q_low, q_high)


def assign_volatility_labels_df(
    segments: pl.DataFrame,
    q_low_value: float,
    q_high_value: float,
) -> pl.DataFrame:
    """Add volatility_label column ('LOW_VOL', 'MID_VOL', 'HIGH_VOL').

    Legacy polars API.

    Args:
        segments: Segment DataFrame with 'realized_volatility_per_bar' and
            'is_tail_unconfirmed' columns.
        q_low_value: Computed Q_low threshold value (score boundary).
        q_high_value: Computed Q_high threshold value (score boundary).

    Returns:
        segments DataFrame with 'volatility_label' column added.
    """
    labels: list[str | None] = []
    for row in segments.iter_rows(named=True):
        if row.get("is_tail_unconfirmed", False):
            labels.append(None)
        else:
            labels.append(
                assign_volatility(row["realized_volatility_per_bar"], q_low_value, q_high_value)
            )
    return segments.with_columns(pl.Series("volatility_label", labels, dtype=pl.Utf8))
