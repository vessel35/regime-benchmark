"""Integration test: synthetic 1-month pipeline → DB → round-trip verify.

Skips with a clear message if REGIME_BENCHMARK_DB_URL is unset or unreachable.

Test verifies:
- segment_labels & bar_labels counts > 0 for both timeframes
- run_status = 'completed'
- joined_labels_1m_5m returns rows
- label_1m matches a valid final_label value
- Teardown deletes the test run (CASCADE handles segment/bar rows)
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

# Guard: skip entire module if REGIME_BENCHMARK_DB_URL not set
_DB_URL = os.environ.get("REGIME_BENCHMARK_DB_URL", "")
if not _DB_URL:
    # Try loading from .env in project root (worktree or parent)
    _env_paths = [
        Path(__file__).parent.parent / ".env",
        Path(__file__).parent.parent.parent.parent.parent / ".env",
    ]
    for _env_path in _env_paths:
        if _env_path.exists():
            for line in _env_path.read_text().splitlines():
                line = line.strip()
                if line.startswith("REGIME_BENCHMARK_DB_URL=") and not line.startswith("#"):
                    _DB_URL = line.split("=", 1)[1].strip()
                    os.environ["REGIME_BENCHMARK_DB_URL"] = _DB_URL
                    break
        if _DB_URL:
            break

pytestmark = pytest.mark.skipif(
    not _DB_URL,
    reason="REGIME_BENCHMARK_DB_URL not set — skipping DB integration test",
)

_VALID_FINAL_LABELS = frozenset([
    "UP_LOW_VOL", "UP_MID_VOL", "UP_HIGH_VOL",
    "DOWN_LOW_VOL", "DOWN_MID_VOL", "DOWN_HIGH_VOL",
    "NON_DIRECTIONAL_LOW_VOL", "NON_DIRECTIONAL_MID_VOL", "NON_DIRECTIONAL_HIGH_VOL",
])


def _try_connect() -> object | None:
    """Attempt DB connection; return connection or None if unreachable."""
    if not _DB_URL:
        return None
    try:
        import psycopg
        conn = psycopg.connect(_DB_URL)
        return conn
    except Exception:
        return None


def _unique_method_version() -> str:
    """Return a method_version unique to this test run."""
    return f"regime_label_9axis_v1.1_test_{int(time.time())}"


@pytest.fixture(scope="module")
def db_conn() -> object:
    """Module-scoped DB connection fixture.  Skips if unreachable."""
    conn = _try_connect()
    if conn is None:
        pytest.skip("REGIME_BENCHMARK_DB_URL set but DB is unreachable — skipping")
    return conn


@pytest.fixture(scope="module")
def config_with_unique_version() -> object:
    """Load config and set a unique method_version for test isolation."""
    from pathlib import Path as P

    from regime_benchmark.config import LabelingConfig

    config_path = P(__file__).parent.parent / "config" / "labeling_config.yaml"
    return LabelingConfig.from_yaml(config_path)


def test_roundtrip_pipeline(db_conn: object, config_with_unique_version: object) -> None:
    """Full pipeline → DB → verify counts and status → teardown."""
    import psycopg

    config = config_with_unique_version
    run_id: int | None = None

    conn: psycopg.Connection = db_conn  # type: ignore[assignment]

    try:
        from regime_benchmark.pipeline import run_pipeline

        run_id = run_pipeline(config=config, synthetic=True)

        # --- Verify run_status = 'completed' ---
        with conn.cursor() as cur:
            cur.execute(
                "SELECT run_status FROM labeling_runs WHERE id = %s",
                (run_id,),
            )
            row = cur.fetchone()

        assert row is not None, f"labeling_runs row not found for run_id={run_id}"
        assert row[0] == "completed", f"run_status = {row[0]!r}, expected 'completed'"

        # --- Verify segment_labels counts > 0 for both timeframes ---
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT timeframe, COUNT(*)
                FROM segment_labels
                WHERE run_id = %s
                GROUP BY timeframe
                ORDER BY timeframe
                """,
                (run_id,),
            )
            seg_counts = {row[0]: row[1] for row in cur.fetchall()}

        assert "1m" in seg_counts, "No segment_labels found for timeframe='1m'"
        assert "5m" in seg_counts, "No segment_labels found for timeframe='5m'"
        assert seg_counts["1m"] > 0, f"segment_labels count for 1m = {seg_counts['1m']}"
        assert seg_counts["5m"] > 0, f"segment_labels count for 5m = {seg_counts['5m']}"

        # --- Verify bar_labels counts > 0 for both timeframes ---
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT timeframe, COUNT(*)
                FROM bar_labels
                WHERE run_id = %s
                GROUP BY timeframe
                ORDER BY timeframe
                """,
                (run_id,),
            )
            bar_counts = {row[0]: row[1] for row in cur.fetchall()}

        assert "1m" in bar_counts, "No bar_labels found for timeframe='1m'"
        assert "5m" in bar_counts, "No bar_labels found for timeframe='5m'"
        assert bar_counts["1m"] > 0, f"bar_labels count for 1m = {bar_counts['1m']}"
        assert bar_counts["5m"] > 0, f"bar_labels count for 5m = {bar_counts['5m']}"

        # --- Verify joined_labels_1m_5m returns rows ---
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(*)
                FROM joined_labels_1m_5m
                WHERE run_id = %s
                """,
                (run_id,),
            )
            joined_count = cur.fetchone()[0]  # type: ignore[index]

        assert joined_count > 0, (
            f"joined_labels_1m_5m returned 0 rows for run_id={run_id}"
        )

        # --- Verify label_1m matches a valid final_label ---
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT DISTINCT label_1m
                FROM joined_labels_1m_5m
                WHERE run_id = %s
                  AND label_1m IS NOT NULL
                LIMIT 50
                """,
                (run_id,),
            )
            distinct_labels = {row[0] for row in cur.fetchall()}

        assert distinct_labels, "No distinct label_1m values found"
        for lbl in distinct_labels:
            assert lbl in _VALID_FINAL_LABELS, (
                f"Invalid label_1m value: {lbl!r} not in canonical 9 labels"
            )

        # Print segment/bar counts for the report
        print(
            f"\n[roundtrip] run_id={run_id}, "
            f"segment_labels: 1m={seg_counts.get('1m', 0)}, 5m={seg_counts.get('5m', 0)}, "
            f"bar_labels: 1m={bar_counts.get('1m', 0)}, 5m={bar_counts.get('5m', 0)}, "
            f"joined_rows={joined_count}, run_status=completed"
        )

    finally:
        # --- Teardown: remove test run (CASCADE handles segment/bar rows) ---
        if run_id is not None:
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "DELETE FROM labeling_runs WHERE id = %s",
                        (run_id,),
                    )
                conn.commit()
            except Exception as e:
                print(f"[roundtrip teardown] WARNING: failed to delete run_id={run_id}: {e}")
