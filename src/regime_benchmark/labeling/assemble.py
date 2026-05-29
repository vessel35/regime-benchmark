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


def assemble_final_labels(segments: pl.DataFrame) -> pl.DataFrame:
    """Combine direction_label and volatility_label into final_label.

    Args:
        segments: Segment DataFrame with 'direction_label' and
                  'volatility_label' columns.
                  Tail segments (is_tail_unconfirmed=True) must have
                  NULL direction_label and volatility_label.

    Returns:
        segments DataFrame with 'final_label' column added.
        final_label is NULL for tail segments.

    Raises:
        NotImplementedError: Implementation deferred to Milestone M6.
    """
    raise NotImplementedError("assemble_final_labels is implemented in Milestone M6")


def expand_labels_to_bars(
    segments: pl.DataFrame,
    klines: pl.DataFrame,
) -> pl.DataFrame:
    """Inherit segment labels onto each bar in the segment range.

    Bars in tail unconfirmed segments are excluded from output (design §11.1 B3).

    Args:
        segments: Segment DataFrame with final_label, start_bar, end_bar,
                  segment_id, is_tail_unconfirmed columns.
        klines: Full Kline DataFrame with open_time, open, high, low, close,
                hlc3 columns.

    Returns:
        Bar-level DataFrame with columns: open_time, open, high, low, close,
        hlc3, segment_id, direction_label, volatility_label, final_label.

    Raises:
        NotImplementedError: Implementation deferred to Milestone M6.
    """
    raise NotImplementedError("expand_labels_to_bars is implemented in Milestone M6")
