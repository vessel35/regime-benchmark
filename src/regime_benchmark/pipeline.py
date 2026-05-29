"""Top-level pipeline orchestration — design §15 / requirements.md §16.

Executes the 19-step labeling pipeline for both 1m and 5m timeframes:
  1-3.  Ingest & trim (ingest/binance.py)
  4-6.  Quality checks (quality/checks.py)
  7-8.  Price transform (transform/price.py)
  9-10. DC threshold + turning points (direction/dc_engine.py)
  11.   Direction labels (direction/segments.py)
  12-14. Realized volatility + quantiles + labels (volatility/realized.py)
  15.   Final 9-label assembly (labeling/assemble.py)
  16.   Auxiliary diagnostics (diagnostics/)
  17.   Bar-level expansion (labeling/assemble.py)
  18.   1m-5m join is a DB VIEW; no computation here
  19.   Validation report (validation/report.py)

Loading order: segment_labels -> bar_labels -> labeling_reports.
Run status is set to 'completed' only after all steps pass.
"""

from __future__ import annotations

from regime_benchmark.config import LabelingConfig


def run_pipeline(config: LabelingConfig) -> int:
    """Execute the full 9-label pipeline and return the run_id.

    Each timeframe tau in {1m, 5m} is processed independently.
    Threshold and quantile calculations are never shared across timeframes.

    Args:
        config: Validated LabelingConfig instance.

    Returns:
        run_id: The labeling_runs.id for this pipeline execution (BIGINT).

    Raises:
        NotImplementedError: Implementation deferred to Milestone M9.
    """
    raise NotImplementedError("run_pipeline is implemented in Milestone M9")
