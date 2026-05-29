"""Tombstone — schema.py removed in M5 legacy cleanup.

Schema/migrations are applied via ``scripts/bootstrap_db.sh`` +
``migrations/001_init.sql`` (psql), NOT Python. Nothing should import this
module; importing it fails loudly so any stale reference surfaces immediately.
"""

raise ImportError(
    "regime_benchmark.persistence.schema was removed in M5 — apply migrations via "
    "scripts/bootstrap_db.sh + migrations/001_init.sql, not Python."
)
