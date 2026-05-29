"""Schema migration management — design §13.

Applies migrations/001_init.sql to the regime_benchmark PostgreSQL database.
Tracks applied migrations in schema_migrations table.
DSN is read from the env var named in PersistenceConfig.dsn_env — never
hardcoded or written to disk.
"""

from __future__ import annotations

from pathlib import Path


def apply_migration(dsn: str, migration_file: str | Path) -> None:
    """Apply a SQL migration file inside a BEGIN/COMMIT transaction.

    Idempotent: records the migration in schema_migrations; skips if already applied.

    Args:
        dsn: PostgreSQL DSN string (read from env var at call site, never stored).
        migration_file: Path to the .sql migration file.

    Raises:
        NotImplementedError: Implementation deferred to Milestone M7.
    """
    raise NotImplementedError("apply_migration is implemented in Milestone M7")


def ensure_schema(dsn: str, migrations_dir: str | Path) -> None:
    """Apply all pending migration files in order from migrations_dir.

    Args:
        dsn: PostgreSQL DSN string.
        migrations_dir: Directory containing *.sql migration files.

    Raises:
        NotImplementedError: Implementation deferred to Milestone M7.
    """
    raise NotImplementedError("ensure_schema is implemented in Milestone M7")
