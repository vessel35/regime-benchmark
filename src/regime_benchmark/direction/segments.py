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

import math
from dataclasses import dataclass

import numpy as np

from regime_benchmark.direction.dc_engine import TurningPoint


@dataclass
class Segment:
    """One segment between two consecutive turning points (or the tail)."""

    segment_id: str
    start_bar: int
    end_bar: int
    confirm_bar: int | None  # None for tail segment
    is_tail_unconfirmed: bool

    # Metrics (computed by build_segments)
    n_bars: int = 0           # N_j
    log_move: float = 0.0    # M_j
    amplitude: float = 0.0   # A_j
    path_length: float = 0.0  # L_j
    efficiency_ratio: float = 0.0  # ER_j

    # Prices
    start_price_hlc3: float = 0.0
    end_price_hlc3: float = 0.0

    # Labels (assigned later)
    direction_label: str | None = None
    volatility_label: str | None = None
    final_label: str | None = None

    # Diagnostics (populated by diagnostics modules in later milestones)
    realized_volatility: float = 0.0
    realized_volatility_per_bar: float = 0.0
    confirm_timestamp_index: int | None = None  # bar index of confirm

    # These are filled by diagnostics modules; defaults satisfy DB NOT NULL
    max_abs_d: float = 0.0
    max_jump_share: float = 0.0
    bipower_variation: float = 0.0
    jump_component: float = 0.0
    jump_share_bv: float = 0.0
    rv_plus: float = 0.0
    rv_minus: float = 0.0
    downside_vol_share: float = 0.0
    amplitude_to_cost_ratio: float = 0.0
    low_tradeability_segment_flag: bool = False
    pullback_within_parent_trend_flag: bool = False

    # Lag diagnostics (None for tail, float for confirmed)
    lag_bars: int | None = None
    lag_move: float | None = None
    capturable_amplitude: float | None = None
    capturable_ratio: float | None = None


def build_segments(
    turning_points: list[TurningPoint],
    p: np.ndarray,
    d: np.ndarray,
) -> list[Segment]:
    """Build segments from consecutive turning point pairs.

    Creates one Segment per adjacent TP pair.  Appends a final tail segment
    from the last TP to the end of the price series (is_tail_unconfirmed=True).

    If fewer than 2 turning points, returns only the tail segment spanning from
    the last confirmed TP (or bar 0 if no TP) to the end of the series; pre-TP
    bars are not part of any confirmed segment.

    Per segment computes:
      - N_j, M_j, A_j, L_j (sum |d| over start+1..end), ER_j (A/L, 0 if L==0)

    Args:
        turning_points: Chronological list from run_dc_engine.  May be empty.
        p: Full log-price array (float64, length N).
        d: Full log-return array (float64, length N; d[0] is NaN).

    Returns:
        List of Segment objects.  The last one has is_tail_unconfirmed=True.
        If no turning points, returns a single tail segment spanning [0, N-1].
    """
    n = len(p)
    if n == 0:
        return []

    segments: list[Segment] = []

    def _make_segment(
        seg_idx: int,
        start: int,
        end: int,
        confirm: int | None,
        is_tail: bool,
    ) -> Segment:
        n_bars = end - start + 1
        m_j = p[end] - p[start]
        a_j = abs(m_j)
        # L_j = sum |d_t| for t in (start+1 .. end) inclusive
        l_j = float(np.nansum(np.abs(d[start + 1 : end + 1])))
        er_j = a_j / l_j if l_j > 0 else 0.0

        seg = Segment(
            segment_id=f"seg_{seg_idx:06d}",
            start_bar=start,
            end_bar=end,
            confirm_bar=confirm,
            is_tail_unconfirmed=is_tail,
            n_bars=n_bars,
            log_move=m_j,
            amplitude=a_j,
            path_length=l_j,
            efficiency_ratio=er_j,
            start_price_hlc3=float(math.exp(p[start])),
            end_price_hlc3=float(math.exp(p[end])),
        )
        return seg

    if len(turning_points) == 0:
        # No confirmed TPs → single tail segment
        segments.append(_make_segment(0, 0, n - 1, None, is_tail=True))
        return segments

    # Build segments from TP pairs
    for j, tp in enumerate(turning_points):
        if j == 0:
            # First segment: from the start of series to TP_0
            # Design §8.2: The first confirmed TP is the start of the
            # "labelable window". Bars before the first TP confirmation belong
            # to the unconfirmed bootstrap period. We model this as a tail
            # segment from bar 0 to the first TP (inclusive) if the first TP
            # is not at bar 0. However, the first TP's bar_index IS the
            # confirmed trough/peak.
            # According to design §11.1 B3: "Labelable window = first TP
            # confirmed onward". Segments are between adjacent TPs.
            # The segment BEFORE the first TP (bars 0..TP[0].bar_index - 1)
            # is the pre-labelable region — not stored.
            # So the first confirmed segment is TP[0]..TP[1].
            continue

        # Segment j-1: from TP[j-1] to TP[j]
        prev_tp = turning_points[j - 1]
        curr_tp = tp
        seg_idx = j - 1
        seg = _make_segment(
            seg_idx=seg_idx,
            start=prev_tp.bar_index,
            end=curr_tp.bar_index,
            confirm=curr_tp.confirm_bar,
            is_tail=False,
        )
        segments.append(seg)

    # Tail segment: from last TP to end of series
    last_tp = turning_points[-1]
    tail_start = last_tp.bar_index
    tail_end = n - 1
    tail_idx = len(turning_points) - 1

    if tail_start <= tail_end:
        tail_seg = _make_segment(
            seg_idx=tail_idx,
            start=tail_start,
            end=tail_end,
            confirm=None,
            is_tail=True,
        )
        segments.append(tail_seg)

    return segments


def assign_direction(
    segment: Segment,
    min_segment_bars: int,
    theta_amp: float,
) -> str | None:
    """Assign direction label to a single segment.

    Returns None for tail (unconfirmed) segments.
    Returns 'UP', 'DOWN', or 'NON_DIRECTIONAL' for confirmed segments.

    Design §8.4:
      UP   if M_j > 0 and N_j >= min_segment_bars and A_j >= theta_amp
      DOWN if M_j < 0 and N_j >= min_segment_bars and A_j >= theta_amp
      NON_DIRECTIONAL otherwise

    ER is explicitly NOT a condition (Case E).

    Args:
        segment: Segment to label.
        min_segment_bars: Minimum N_j for directional label.
        theta_amp: Minimum A_j for directional label (typically = theta_dc).

    Returns:
        Direction string or None for tail segments.
    """
    if segment.is_tail_unconfirmed:
        return None

    n_ok = segment.n_bars >= min_segment_bars
    a_ok = segment.amplitude >= theta_amp

    if n_ok and a_ok:
        if segment.log_move > 0:
            return "UP"
        elif segment.log_move < 0:
            return "DOWN"
    return "NON_DIRECTIONAL"


# ---------------------------------------------------------------------------
# Legacy polars-based API (kept for backward compatibility with M1 stubs)
# ---------------------------------------------------------------------------

import polars as pl  # noqa: E402


def compute_segment_metrics(
    turning_points: list[TurningPoint],
    log_prices: list[float],
) -> pl.DataFrame:
    """Compute N_j, M_j, A_j, L_j, ER_j for each confirmed segment.

    Legacy API used by tests that expect a polars DataFrame.
    For the M2 pipeline, use build_segments() instead.

    Args:
        turning_points: Chronological list of turning points from dc_engine.
        log_prices: Full log-price series p_t indexed by bar position.

    Returns:
        DataFrame with one row per segment and columns:
        segment_id, start_bar, end_bar, confirm_bar, is_tail_unconfirmed,
        n_bars (N_j), log_move (M_j), amplitude (A_j), path_length (L_j),
        efficiency_ratio (ER_j).
    """
    p_arr = np.array(log_prices, dtype=np.float64)
    n = len(p_arr)
    d_arr = np.empty(n, dtype=np.float64)
    d_arr[0] = np.nan
    d_arr[1:] = p_arr[1:] - p_arr[:-1]

    segs = build_segments(turning_points, p_arr, d_arr)

    rows = []
    for s in segs:
        rows.append(
            {
                "segment_id": s.segment_id,
                "start_bar": s.start_bar,
                "end_bar": s.end_bar,
                "confirm_bar": s.confirm_bar,
                "is_tail_unconfirmed": s.is_tail_unconfirmed,
                "n_bars": s.n_bars,
                "log_move": s.log_move,
                "amplitude": s.amplitude,
                "path_length": s.path_length,
                "efficiency_ratio": s.efficiency_ratio,
            }
        )

    if not rows:
        return pl.DataFrame(
            schema={
                "segment_id": pl.Utf8,
                "start_bar": pl.Int64,
                "end_bar": pl.Int64,
                "confirm_bar": pl.Int64,
                "is_tail_unconfirmed": pl.Boolean,
                "n_bars": pl.Int64,
                "log_move": pl.Float64,
                "amplitude": pl.Float64,
                "path_length": pl.Float64,
                "efficiency_ratio": pl.Float64,
            }
        )

    return pl.DataFrame(rows).with_columns(
        pl.col("confirm_bar").cast(pl.Int64),
        pl.col("n_bars").cast(pl.Int64),
        pl.col("start_bar").cast(pl.Int64),
        pl.col("end_bar").cast(pl.Int64),
    )


def assign_direction_labels(
    segments: pl.DataFrame,
    min_segment_bars: int,
    theta_amp: float,
) -> pl.DataFrame:
    """Add direction_label column to segment DataFrame.

    Legacy polars API kept for compatibility.

    Args:
        segments: Output of compute_segment_metrics.
        min_segment_bars: Minimum bar count for a directional segment.
        theta_amp: Minimum amplitude for a directional segment.

    Returns:
        segments DataFrame with 'direction_label' column added
        ('UP', 'DOWN', or 'NON_DIRECTIONAL', null for tail).
    """
    def _label(log_move: float | None, n_bars: int, amplitude: float, is_tail: bool) -> str | None:
        if is_tail:
            return None
        if n_bars >= min_segment_bars and amplitude >= theta_amp:
            if log_move is not None and log_move > 0:
                return "UP"
            elif log_move is not None and log_move < 0:
                return "DOWN"
        return "NON_DIRECTIONAL"

    labels = []
    for row in segments.iter_rows(named=True):
        labels.append(
            _label(
                row["log_move"],
                row["n_bars"],
                row["amplitude"],
                row["is_tail_unconfirmed"],
            )
        )

    return segments.with_columns(pl.Series("direction_label", labels, dtype=pl.Utf8))
