"""Tests for LabelingConfig pydantic model — M1 deliverable.

Covers:
- YAML loading succeeds
- Exactly 9 canonical final labels
- Quantile low <= high validator
- TimeframeParams present for both 1m and 5m
- dsn_env stores ENV VAR NAME, not a URL (secret-leakage guard)
- Frozen model rejects mutation
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from pydantic import ValidationError  # noqa: F401 — used in test_frozen + other tests

from regime_benchmark.config import CANONICAL_FINAL_LABELS, LabelingConfig

# Path to the config file relative to the project root
_CONFIG_PATH = Path(__file__).parent.parent / "config" / "labeling_config.yaml"


def test_loads_yaml() -> None:
    """LabelingConfig.from_yaml must load without raising errors."""
    config = LabelingConfig.from_yaml(_CONFIG_PATH)
    assert config is not None
    assert config.method_version == "regime_label_9axis_v1.1"


def test_nine_final_labels() -> None:
    """final_labels must contain exactly 9 canonical cross-product labels."""
    config = LabelingConfig.from_yaml(_CONFIG_PATH)
    assert len(config.final_labels) == 9
    assert set(config.final_labels) == CANONICAL_FINAL_LABELS


def test_quantile_validator_valid() -> None:
    """VolatilityMethod with low <= high must validate successfully."""
    config = LabelingConfig.from_yaml(_CONFIG_PATH)
    q = config.volatility_method.quantiles
    assert q.low <= q.high
    assert 0.0 <= q.low <= 1.0
    assert 0.0 <= q.high <= 1.0


def test_quantile_validator_low_greater_than_high_raises() -> None:
    """Constructing QuantileRange with low > high must raise ValidationError."""
    from regime_benchmark.config import QuantileRange

    with pytest.raises(ValidationError):
        QuantileRange(low=0.66, high=0.33)


def test_quantile_validator_out_of_range_raises() -> None:
    """Constructing QuantileRange with values outside [0,1] must raise ValidationError."""
    from regime_benchmark.config import QuantileRange

    with pytest.raises(ValidationError):
        QuantileRange(low=-0.1, high=0.5)

    with pytest.raises(ValidationError):
        QuantileRange(low=0.3, high=1.5)


def test_timeframe_params_present() -> None:
    """Both '1m' and '5m' timeframe params must be present with correct candidate lengths."""
    config = LabelingConfig.from_yaml(_CONFIG_PATH)
    params = config.direction_method.params

    assert "1m" in params
    assert "5m" in params

    # 1m: k_dc_candidates=[3,4,5], min_segment_bars_candidates=[5,10,15], pullback=[10,20,30]
    p1m = params["1m"]
    assert len(p1m.k_dc_candidates) == 3
    assert len(p1m.min_segment_bars_candidates) == 3
    assert len(p1m.pullback_max_bars_candidates) == 3

    # 5m: k_dc_candidates=[2,3,4], min_segment_bars_candidates=[3,5,8], pullback=[3,5,10]
    p5m = params["5m"]
    assert len(p5m.k_dc_candidates) == 3
    assert len(p5m.min_segment_bars_candidates) == 3
    assert len(p5m.pullback_max_bars_candidates) == 3


def test_dsn_env_is_name_not_value() -> None:
    """dsn_env must be the ENV VAR NAME 'REGIME_BENCHMARK_DB_URL', not a postgres:// URL."""
    config = LabelingConfig.from_yaml(_CONFIG_PATH)
    dsn_env = config.persistence.dsn_env
    assert dsn_env == "REGIME_BENCHMARK_DB_URL"
    # Guard: must be a valid uppercase env-var identifier (whitelist)
    assert re.match(r"^[A-Z_][A-Z0-9_]*$", dsn_env)


def test_dsn_env_url_value_raises() -> None:
    """PersistenceConfig must reject a real DSN URL in dsn_env."""
    import yaml

    from regime_benchmark.config import PersistenceConfig

    raw = yaml.safe_load(_CONFIG_PATH.read_text())
    persistence_raw = raw["labeling_config"]["persistence"].copy()
    persistence_raw["dsn_env"] = "postgresql://regime_owner:secret@localhost/regime_benchmark"

    with pytest.raises(ValidationError):
        PersistenceConfig.model_validate(persistence_raw)


def test_frozen() -> None:
    """Mutating any field on a frozen LabelingConfig must raise an error.

    Pydantic v2 raises ValidationError (frozen_instance) on mutation attempts.
    """
    config = LabelingConfig.from_yaml(_CONFIG_PATH)
    with pytest.raises((TypeError, AttributeError, ValidationError)):
        config.method_version = "tampered"  # type: ignore[misc]


def test_data_config_utc_aware() -> None:
    """DataConfig datetime fields must be timezone-aware UTC."""
    from datetime import timezone

    config = LabelingConfig.from_yaml(_CONFIG_PATH)
    assert config.data.start_utc.tzinfo == timezone.utc
    assert config.data.end_utc.tzinfo == timezone.utc


def test_invalid_final_labels_raises() -> None:
    """LabelingConfig with wrong final_labels must raise ValidationError."""
    import yaml

    raw = yaml.safe_load(_CONFIG_PATH.read_text())
    raw["labeling_config"]["final_labels"] = ["UP_LOW_VOL", "FAKE_LABEL"]

    with pytest.raises(ValidationError):
        LabelingConfig.model_validate(raw["labeling_config"])


def test_final_labels_duplicate_padded_raises() -> None:
    """10-element list = canonical 9 + duplicate must raise ValidationError (B1)."""
    import yaml

    raw = yaml.safe_load(_CONFIG_PATH.read_text())
    raw["labeling_config"]["final_labels"] = sorted(CANONICAL_FINAL_LABELS) + ["UP_LOW_VOL"]

    with pytest.raises(ValidationError):
        LabelingConfig.model_validate(raw["labeling_config"])


def test_dsn_env_libpq_key_value_raises() -> None:
    """libpq key-value DSN in dsn_env must raise ValidationError (B2)."""
    import yaml

    from regime_benchmark.config import PersistenceConfig

    raw = yaml.safe_load(_CONFIG_PATH.read_text())
    persistence_raw = raw["labeling_config"]["persistence"].copy()
    persistence_raw["dsn_env"] = "host=localhost user=owner password=secret"

    with pytest.raises(ValidationError):
        PersistenceConfig.model_validate(persistence_raw)


def test_non_utc_datetime_normalized_to_utc() -> None:
    """A +09:00 tz-aware datetime in start_utc must be normalized to UTC (S1)."""
    from datetime import datetime, timedelta, timezone

    from regime_benchmark.config import DataConfig

    tz_kst = timezone(timedelta(hours=9))
    dc = DataConfig(
        exchange="Binance",
        market="USDM Futures",
        symbol="ETHUSDT",
        display_pair="ETH/USDT",
        start_utc=datetime(2024, 1, 1, 9, 0, 0, tzinfo=tz_kst),
        end_utc=datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
        source_1m="data/1m/",
        source_5m="data/5m/",
    )
    assert dc.start_utc.tzinfo == timezone.utc
    assert dc.start_utc.hour == 0  # 09:00 KST == 00:00 UTC


def test_single_timeframe_raises() -> None:
    """LabelingConfig with timeframes=['1m'] only must raise ValidationError (S2)."""
    import yaml

    raw = yaml.safe_load(_CONFIG_PATH.read_text())
    raw["labeling_config"]["timeframes"] = ["1m"]

    with pytest.raises(ValidationError):
        LabelingConfig.model_validate(raw["labeling_config"])
