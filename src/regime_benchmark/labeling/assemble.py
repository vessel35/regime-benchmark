"""Final 9-label assembly and bar-level expansion — requirements.md §8, §9.

final_label_j = direction_j + '_' + volatility_j

Bar inheritance:
  label_t = final_label_j  for all t in S_j

1m and 5m are labeled independently. The joined view is a DB VIEW, not a
label-computation step. Tail segments (is_tail_unconfirmed=True) receive
NULL labels and their bars are excluded from bar_labels (design §11.1 B3).
"""

from __future__ import annotations

import polars as pl

from regime_benchmark.direction.segments import Segment


def final_label(direction: str, volatility: str) -> str:
    """Combine direction and volatility into a final 9-label string.

    Args:
        direction: 'UP', 'DOWN', or 'NON_DIRECTIONAL'.
        volatility: 'LOW_VOL', 'MID_VOL', or 'HIGH_VOL'.

    Returns:
        f"{direction}_{volatility}" — one of the 9 canonical labels.
    """
    return f"{direction}_{volatility}"


def assign_final_labels(segments: list[Segment]) -> list[Segment]:
    """Assign final_label to each confirmed segment.

    Tail segments (is_tail_unconfirmed=True) retain final_label=None.

    Args:
        segments: List of Segment objects with direction_label and
            volatility_label set on confirmed segments.

    Returns:
        The same list with final_label updated.

    Raises:
        ValueError: If a confirmed segment has None direction_label or
            volatility_label (indicates pipeline order error).
    """
    for seg in segments:
        if seg.is_tail_unconfirmed:
            seg.final_label = None
        else:
            if seg.direction_label is None or seg.volatility_label is None:
                raise ValueError(
                    f"Confirmed segment {seg.segment_id} has None "
                    f"direction_label={seg.direction_label!r} or "
                    f"volatility_label={seg.volatility_label!r}"
                )
            seg.final_label = final_label(seg.direction_label, seg.volatility_label)
    return segments


def expand_to_bars(
    segments: list[Segment],
    bars_df: pl.DataFrame,
) -> pl.DataFrame:
    """Inherit each segment's label to the bars it OWNS (exactly-one-label partition).

    Bar-ownership rule (strategy-architect decision; requirements §11.3 non-overlap):
    adjacent segments share the turning-point bar (closed metric interval
    [TP_j, TP_{j+1}]), so for LABEL ownership we partition with a **half-open**
    convention — every confirmed segment owns ``[start_bar, end_bar)`` (start
    inclusive, end exclusive) EXCEPT the last confirmed segment which owns the
    closed ``[start_bar, end_bar]`` so the final confirmed TP bar is labeled once.
    The shared TP bar is therefore owned by the segment that STARTS at it (its
    forward-looking regime), never double-counted.

    Note: segment METRICS (N_j, M_j, A_j, L_j, RV) remain on the closed interval
    per §6.3/§8.3 — only bar-label ownership is partitioned here.

    Excludes:
    - Tail-unconfirmed segment bars (design §11.1 B3 — NOT persisted)
    - Bars before the first confirmed TP (pre-labelable window)

    Args:
        segments: List of Segment objects with all labels assigned.
        bars_df: Kline DataFrame with open_time, open, high, low, close, hlc3
            columns.  Row order must match bar index (0-based).

    Returns:
        Bar-level DataFrame with columns:
            open_time, open, high, low, close, hlc3,
            segment_id, direction_label, volatility_label, final_label.
        Only confirmed segment bars are included, each exactly once.
    """
    rows = []
    bars_list = bars_df.to_dicts()

    # Index of the last confirmed (non-tail, labeled) segment — it alone owns
    # its end_bar (closed interval); all others are half-open [start, end).
    last_confirmed_idx = None
    for i, seg in enumerate(segments):
        if not seg.is_tail_unconfirmed and seg.final_label is not None:
            last_confirmed_idx = i

    for i, seg in enumerate(segments):
        if seg.is_tail_unconfirmed:
            # Design §11.1 B3: tail bars are NOT persisted in bar_labels
            continue
        if seg.final_label is None:
            # A non-tail segment must always carry a label by this point.
            # Fail loud rather than silently dropping bars (N1).
            raise ValueError(
                f"confirmed segment {seg.segment_id} has no final_label "
                f"(pipeline ordering error)"
            )

        # Half-open [start, end); the last confirmed segment is closed [start, end].
        stop = seg.end_bar + 1 if i == last_confirmed_idx else seg.end_bar
        for bar_idx in range(seg.start_bar, stop):
            if bar_idx >= len(bars_list):
                break
            bar = bars_list[bar_idx]
            rows.append(
                {
                    "open_time": bar["open_time"],
                    "open": bar["open"],
                    "high": bar["high"],
                    "low": bar["low"],
                    "close": bar["close"],
                    "hlc3": bar["hlc3"],
                    "segment_id": seg.segment_id,
                    "direction_label": seg.direction_label,
                    "volatility_label": seg.volatility_label,
                    "final_label": seg.final_label,
                }
            )

    if not rows:
        return pl.DataFrame(
            schema={
                "open_time": pl.Datetime(time_unit="us", time_zone="UTC"),
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "hlc3": pl.Float64,
                "segment_id": pl.Utf8,
                "direction_label": pl.Utf8,
                "volatility_label": pl.Utf8,
                "final_label": pl.Utf8,
            }
        )

    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# Legacy polars-based API (kept for backward compatibility with M1 stubs)
# ---------------------------------------------------------------------------


def assemble_final_labels(segments: pl.DataFrame) -> pl.DataFrame:
    """Combine direction_label and volatility_label into final_label.

    Legacy polars API.

    Args:
        segments: Segment DataFrame with 'direction_label' and
                  'volatility_label' columns.
                  Tail segments (is_tail_unconfirmed=True) must have
                  NULL direction_label and volatility_label.

    Returns:
        segments DataFrame with 'final_label' column added.
        final_label is NULL for tail segments.
    """
    labels: list[str | None] = []
    for row in segments.iter_rows(named=True):
        is_tail = row.get("is_tail_unconfirmed", False)
        d = row.get("direction_label")
        v = row.get("volatility_label")
        if is_tail or d is None or v is None:
            labels.append(None)
        else:
            labels.append(f"{d}_{v}")
    return segments.with_columns(pl.Series("final_label", labels, dtype=pl.Utf8))


def expand_labels_to_bars(
    segments: pl.DataFrame,
    klines: pl.DataFrame,
) -> pl.DataFrame:
    """Inherit segment labels onto each bar in the segment range.

    Legacy polars API.  Bars in tail unconfirmed segments are excluded.

    Args:
        segments: Segment DataFrame with final_label, start_bar, end_bar,
                  segment_id, is_tail_unconfirmed columns.
        klines: Full Kline DataFrame with open_time, open, high, low, close,
                hlc3 columns.

    Returns:
        Bar-level DataFrame with columns: open_time, open, high, low, close,
        hlc3, segment_id, direction_label, volatility_label, final_label.
    """
    klines_list = klines.to_dicts()
    rows = []

    for row in segments.iter_rows(named=True):
        if row.get("is_tail_unconfirmed", False):
            continue
        final = row.get("final_label")
        if final is None:
            continue
        for bar_idx in range(row["start_bar"], row["end_bar"] + 1):
            if bar_idx >= len(klines_list):
                break
            bar = klines_list[bar_idx]
            rows.append(
                {
                    "open_time": bar["open_time"],
                    "open": bar["open"],
                    "high": bar["high"],
                    "low": bar["low"],
                    "close": bar["close"],
                    "hlc3": bar.get("hlc3", (bar["high"] + bar["low"] + bar["close"]) / 3.0),
                    "segment_id": row["segment_id"],
                    "direction_label": row.get("direction_label"),
                    "volatility_label": row.get("volatility_label"),
                    "final_label": final,
                }
            )

    if not rows:
        return pl.DataFrame(
            schema={
                "open_time": pl.Datetime(time_unit="us", time_zone="UTC"),
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "hlc3": pl.Float64,
                "segment_id": pl.Utf8,
                "direction_label": pl.Utf8,
                "volatility_label": pl.Utf8,
                "final_label": pl.Utf8,
            }
        )

    return pl.DataFrame(rows)
