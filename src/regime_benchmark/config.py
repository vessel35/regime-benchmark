"""Pydantic v2 configuration models for the Regime 9-Label Benchmark Labeler.

Maps requirements.md §14 (labeling_config) exactly. All models are frozen and
use extra='forbid' for reproducibility and strict validation.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

# ---------------------------------------------------------------------------
# Canonical label set (used by validator below)
# ---------------------------------------------------------------------------

_DIRECTIONS = ("UP", "DOWN", "NON_DIRECTIONAL")
_VOLATILITIES = ("LOW_VOL", "MID_VOL", "HIGH_VOL")

CANONICAL_FINAL_LABELS: frozenset[str] = frozenset(
    f"{d}_{v}" for d in _DIRECTIONS for v in _VOLATILITIES
)

# Valid uppercase env-var identifier (whitelist for dsn_env — blocks DSN leakage)
_ENV_VAR_RE = re.compile(r"^[A-Z_][A-Z0-9_]*$")


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class DataConfig(BaseModel):
    """Source data parameters — requirements.md §14 data block."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    exchange: str
    market: str
    symbol: str
    display_pair: str
    start_utc: datetime
    end_utc: datetime
    source_1m: str
    source_5m: str

    @field_validator("start_utc", "end_utc", mode="before")
    @classmethod
    def parse_utc_string(cls, v: Any) -> datetime:
        """Parse 'YYYY-MM-DD HH:MM:SS' strings as timezone-aware UTC datetimes."""
        if isinstance(v, datetime):
            if v.tzinfo is None:
                return v.replace(tzinfo=timezone.utc)
            return v.astimezone(timezone.utc)
        if isinstance(v, str):
            dt = datetime.strptime(v, "%Y-%m-%d %H:%M:%S")
            return dt.replace(tzinfo=timezone.utc)
        raise ValueError(f"Cannot parse datetime from {v!r}")


class TimeframeParams(BaseModel):
    """Per-timeframe Directional Change calibration parameters."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    q_dc: float
    k_dc_candidates: list[float]
    min_segment_bars_candidates: list[int]
    theta_amp_policy: Literal["same_as_theta_dc"]
    pullback_max_bars_candidates: list[int]


class DirectionMethod(BaseModel):
    """Directional Change method specification."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: str
    threshold_policy: str
    params: dict[Literal["1m", "5m"], TimeframeParams]


class QuantileRange(BaseModel):
    """Volatility quantile boundaries — validates low <= high and both in [0, 1]."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    low: float
    high: float

    @model_validator(mode="after")
    def validate_quantile_order(self) -> "QuantileRange":
        """Enforce 0 <= low <= high <= 1."""
        if not (0.0 <= self.low <= 1.0):
            raise ValueError(f"quantile low={self.low} must be in [0, 1]")
        if not (0.0 <= self.high <= 1.0):
            raise ValueError(f"quantile high={self.high} must be in [0, 1]")
        if self.low > self.high:
            raise ValueError(
                f"quantile low={self.low} must be <= high={self.high}"
            )
        return self


class VolatilityMethod(BaseModel):
    """Realized-volatility method specification."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: str
    formula: str
    quantiles: QuantileRange


class EfficiencyRatioMetrics(BaseModel):
    """Efficiency-ratio diagnostic config."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool
    formula: str
    not_part_of_final_9_labels: bool


class LagDiagnosticsMetrics(BaseModel):
    """Lag / confirm-bar diagnostic config."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool
    metrics: list[str]


class TradeabilityDiagnosticsMetrics(BaseModel):
    """Amplitude-to-cost tradeability diagnostic config."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool
    metrics: list[str]
    not_part_of_final_9_labels: bool


class JumpDiagnosticsMetrics(BaseModel):
    """Jump / bipower-variation diagnostic config."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool
    metrics: list[str]
    not_part_of_final_9_labels: bool


class AsymmetricVolatilityMetrics(BaseModel):
    """Downside-volatility / asymmetric-vol diagnostic config."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool
    metrics: list[str]
    not_part_of_final_9_labels: bool


class AuxiliaryMetrics(BaseModel):
    """Auxiliary diagnostic metric configuration block."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    efficiency_ratio: EfficiencyRatioMetrics
    lag_diagnostics: LagDiagnosticsMetrics
    tradeability_diagnostics: TradeabilityDiagnosticsMetrics
    jump_diagnostics: JumpDiagnosticsMetrics
    asymmetric_volatility: AsymmetricVolatilityMetrics


class BarLabelsPartition(BaseModel):
    """Bar-labels table partitioning specification."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    partition_strategy: str
    partition_column: str
    partition_range: dict[str, str]


class EnumsConfig(BaseModel):
    """PostgreSQL ENUM type value lists."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    timeframe: list[str]
    direction: list[str]
    volatility: list[str]
    final_label_count: int

    @model_validator(mode="after")
    def validate_enum_values(self) -> "EnumsConfig":
        """Reject typos / drift in the persisted ENUM value lists (must match DDL)."""
        if set(self.direction) != set(_DIRECTIONS):
            raise ValueError(f"direction must be exactly {_DIRECTIONS}, got {self.direction}")
        if set(self.volatility) != set(_VOLATILITIES):
            raise ValueError(f"volatility must be exactly {_VOLATILITIES}, got {self.volatility}")
        if set(self.timeframe) != {"1m", "5m"}:
            raise ValueError(f"enums.timeframe must be exactly ['1m','5m'], got {self.timeframe}")
        if self.final_label_count != 9:
            raise ValueError(f"final_label_count must be 9, got {self.final_label_count}")
        return self


class PersistenceConfig(BaseModel):
    """PostgreSQL persistence layer configuration — requirements.md §14 persistence block."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    engine: str
    version: str
    database: str
    # ENV VAR NAME only — never a real DSN/postgres:// URL
    dsn_env: str
    migration_file: str
    write_target: str
    read_only_mcp_dbs_excluded: bool
    bulk_load: str
    versioning: str
    bar_labels: BarLabelsPartition
    enums: EnumsConfig
    numeric_type: str
    views: list[str]

    @field_validator("dsn_env")
    @classmethod
    def dsn_env_must_be_var_name(cls, v: str) -> str:
        """Guard: dsn_env must be an uppercase env-var name, never a DSN/secret.

        Whitelist (not blacklist) — also rejects libpq key-value DSNs like
        'host=localhost user=owner password=secret' which contain no '://'.
        """
        if not _ENV_VAR_RE.match(v):
            raise ValueError(
                f"dsn_env must be an uppercase env-var name "
                f"(e.g. REGIME_BENCHMARK_DB_URL). Got: {v!r}"
            )
        return v


# ---------------------------------------------------------------------------
# Root config model
# ---------------------------------------------------------------------------


class LabelingConfig(BaseModel):
    """Root labeling configuration — requirements.md §14 labeling_config block."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    method_version: str
    data: DataConfig
    price_field: Literal["hlc3"]
    price_formula: str
    timeframes: list[Literal["1m", "5m"]]
    direction_method: DirectionMethod
    volatility_method: VolatilityMethod
    auxiliary_metrics: AuxiliaryMetrics
    final_labels: list[str]
    persistence: PersistenceConfig

    @field_validator("final_labels")
    @classmethod
    def validate_nine_canonical_labels(cls, v: list[str]) -> list[str]:
        """Reject unless final_labels equals exactly the 9 canonical direction x volatility values.

        Validates that the set matches UP/DOWN/NON_DIRECTIONAL x LOW/MID/HIGH_VOL.
        """
        if len(v) != 9 or len(set(v)) != 9:
            raise ValueError(
                f"final_labels must contain exactly 9 unique values, "
                f"got {len(v)} ({len(set(v))} unique)"
            )
        given = set(v)
        if given != CANONICAL_FINAL_LABELS:
            missing = CANONICAL_FINAL_LABELS - given
            extra = given - CANONICAL_FINAL_LABELS
            raise ValueError(
                f"final_labels must be exactly the 9 canonical labels. "
                f"Missing: {sorted(missing)}, Extra: {sorted(extra)}"
            )
        return v

    @model_validator(mode="after")
    def cross_validate_timeframes(self) -> "LabelingConfig":
        """timeframes / direction params / persistence enums must agree (no drift)."""
        tf_set = set(self.timeframes)
        if tf_set != {"1m", "5m"} or len(self.timeframes) != 2:
            raise ValueError("timeframes must be exactly ['1m', '5m'] (unique, both present)")
        if set(self.direction_method.params.keys()) != tf_set:
            raise ValueError("direction_method.params keys must match timeframes")
        if set(self.persistence.enums.timeframe) != tf_set:
            raise ValueError("persistence.enums.timeframe must match timeframes")
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> "LabelingConfig":
        """Load and validate a LabelingConfig from a YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            Validated LabelingConfig instance.

        Raises:
            FileNotFoundError: If the YAML file does not exist.
            ValidationError: If the YAML content fails Pydantic validation.
        """
        with open(path, "r", encoding="utf-8") as fh:
            raw: dict[str, Any] = yaml.safe_load(fh)
        # The YAML has a top-level 'labeling_config' key wrapping the fields
        if "labeling_config" in raw:
            raw = raw["labeling_config"]
        return cls.model_validate(raw)
