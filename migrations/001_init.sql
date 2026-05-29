-- migrations/001_init.sql
-- Regime 9-Label Benchmark Labeler — initial schema
-- Target: PostgreSQL 16+ (postgis/postgis:16-3.4 image; PostGIS unused here)
-- 대응: docs/design.md §13, docs/requirements.md §13/§14/§17
-- 출력 DB: regime_benchmark (별도 쓰기 가능 DB; MCP read-only 연결과 분리)

BEGIN;

-- search_path 명시 (세션 변조 대비)
SET LOCAL search_path = public, pg_catalog;

-- =========================================================================
-- schema_migrations (skill convention)
-- =========================================================================
CREATE TABLE IF NOT EXISTS schema_migrations (
    version    INTEGER PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- =========================================================================
-- ENUM types (final 9 labels + axes)
-- =========================================================================
CREATE TYPE timeframe_enum AS ENUM ('1m', '5m');

CREATE TYPE direction_label AS ENUM ('UP', 'DOWN', 'NON_DIRECTIONAL');

CREATE TYPE volatility_label AS ENUM ('LOW_VOL', 'MID_VOL', 'HIGH_VOL');

CREATE TYPE final_label AS ENUM (
    'UP_LOW_VOL', 'UP_MID_VOL', 'UP_HIGH_VOL',
    'DOWN_LOW_VOL', 'DOWN_MID_VOL', 'DOWN_HIGH_VOL',
    'NON_DIRECTIONAL_LOW_VOL', 'NON_DIRECTIONAL_MID_VOL', 'NON_DIRECTIONAL_HIGH_VOL'
);

-- =========================================================================
-- labeling_runs : 파이프라인 1회 실행(1m·5m 모두 포함) = 재현 단위
-- 동일 run_id 안에서 1m·5m join이 이뤄지도록 timeframe를 run에서 분리하지 않음
-- =========================================================================
CREATE TABLE labeling_runs (
    id                BIGSERIAL PRIMARY KEY,
    method_version    VARCHAR(64)  NOT NULL,
    symbol            VARCHAR(32)  NOT NULL DEFAULT 'ETHUSDT',
    market            VARCHAR(32)  NOT NULL DEFAULT 'BINANCE_USDM_FUTURES',
    period_start_utc  TIMESTAMPTZ  NOT NULL,
    period_end_utc    TIMESTAMPTZ  NOT NULL,
    price_field       VARCHAR(16)  NOT NULL DEFAULT 'hlc3',
    git_commit        VARCHAR(40),
    run_status        VARCHAR(16)  NOT NULL DEFAULT 'loading',
    completed_at      TIMESTAMPTZ,
    created_at        TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    -- append-only: 동일 (method_version, period) 재실행 시 새 run_id 생성. UNIQUE 미강제.
    CONSTRAINT ck_labeling_runs_status CHECK (run_status IN ('loading','completed','failed')),
    CONSTRAINT ck_labeling_runs_completed_at CHECK (
        (run_status = 'completed' AND completed_at IS NOT NULL)
        OR (run_status <> 'completed' AND completed_at IS NULL)
    ),
    CONSTRAINT ck_labeling_runs_period CHECK (period_end_utc >= period_start_utc)
);

CREATE INDEX idx_labeling_runs_lookup        ON labeling_runs (method_version, period_start_utc, period_end_utc);
CREATE INDEX idx_labeling_runs_status        ON labeling_runs (run_status, completed_at);

COMMENT ON COLUMN labeling_runs.run_status   IS 'loading → completed/failed. 소비자는 completed만 신뢰';
COMMENT ON COLUMN labeling_runs.completed_at IS 'run_status=completed 일 때만 설정';

COMMENT ON TABLE  labeling_runs IS '라벨링 파이프라인 1회 실행. 1m·5m 모두 포함, append-only 버전 관리';
COMMENT ON COLUMN labeling_runs.method_version IS '예: regime_label_9axis_v1.1';
COMMENT ON COLUMN labeling_runs.price_field    IS '기준 가격 필드 (기본 hlc3)';

-- =========================================================================
-- labeling_run_params : timeframe별 캘리브레이션 파라미터
-- =========================================================================
CREATE TABLE labeling_run_params (
    id                       BIGSERIAL PRIMARY KEY,
    run_id                   BIGINT          NOT NULL REFERENCES labeling_runs(id) ON DELETE CASCADE,
    timeframe                timeframe_enum  NOT NULL,
    q_dc                     DOUBLE PRECISION NOT NULL,
    k_dc                     DOUBLE PRECISION NOT NULL,
    min_segment_bars         INTEGER          NOT NULL,
    theta_dc                 DOUBLE PRECISION NOT NULL,
    theta_amp                DOUBLE PRECISION NOT NULL,
    q_low                    DOUBLE PRECISION NOT NULL DEFAULT 0.33,
    q_high                   DOUBLE PRECISION NOT NULL DEFAULT 0.66,
    taker_fee_rate           DOUBLE PRECISION,
    slippage_rate_estimate   DOUBLE PRECISION,
    params_extra             JSONB,
    CONSTRAINT uk_labeling_run_params_run_tf UNIQUE (run_id, timeframe),
    CONSTRAINT ck_labeling_run_params_quantiles CHECK (q_low <= q_high),
    CONSTRAINT ck_labeling_run_params_theta_dc  CHECK (theta_dc  > 0),
    CONSTRAINT ck_labeling_run_params_theta_amp CHECK (theta_amp > 0),
    CONSTRAINT ck_labeling_run_params_ranges    CHECK (
        q_dc             BETWEEN 0 AND 1
        AND q_low        BETWEEN 0 AND 1
        AND q_high       BETWEEN 0 AND 1
        AND k_dc                 >  0
        AND min_segment_bars     >= 1
        AND (taker_fee_rate         IS NULL OR taker_fee_rate         >= 0)
        AND (slippage_rate_estimate IS NULL OR slippage_rate_estimate >= 0)
    )
);

CREATE INDEX idx_labeling_run_params_run ON labeling_run_params (run_id);

-- =========================================================================
-- segment_labels : segment-level 라벨 + 진단값 (requirements §13.1)
-- 수천~수만 행 규모 → 파티셔닝 불필요
-- =========================================================================
CREATE TABLE segment_labels (
    id                                 BIGSERIAL PRIMARY KEY,
    run_id                             BIGINT           NOT NULL REFERENCES labeling_runs(id) ON DELETE CASCADE,
    -- §13.1 필수 메타 (run에서 결정되지만 출력 스키마는 명시 요구 → denormalize)
    symbol                             VARCHAR(32)      NOT NULL DEFAULT 'ETHUSDT',
    market                             VARCHAR(32)      NOT NULL DEFAULT 'BINANCE_USDM_FUTURES',
    method_version                     VARCHAR(64)      NOT NULL,
    timeframe                          timeframe_enum   NOT NULL,
    segment_id                         VARCHAR(64)      NOT NULL,
    start_timestamp                    TIMESTAMPTZ      NOT NULL,
    end_timestamp                      TIMESTAMPTZ      NOT NULL,
    -- 미확정 꼬리 segment 표시 (§11.2 "확인 불가 segment 별도 표시")
    is_tail_unconfirmed                BOOLEAN          NOT NULL DEFAULT false,
    -- 확정 시점 진단 (꼬리 segment 외에는 NOT NULL — ck_segment_labels_confirm 참조)
    confirm_timestamp                  TIMESTAMPTZ,
    lag_bars                           INTEGER,
    lag_move                           DOUBLE PRECISION,
    capturable_amplitude               DOUBLE PRECISION,
    capturable_ratio                   DOUBLE PRECISION,
    -- 가격·경로 메트릭 (모든 segment 계산 가능 → NOT NULL)
    start_price_hlc3                   DOUBLE PRECISION NOT NULL,
    end_price_hlc3                     DOUBLE PRECISION NOT NULL,
    log_move                           DOUBLE PRECISION NOT NULL,
    amplitude                          DOUBLE PRECISION NOT NULL,
    path_length                        DOUBLE PRECISION NOT NULL,
    efficiency_ratio                   DOUBLE PRECISION NOT NULL,
    realized_volatility                DOUBLE PRECISION NOT NULL,
    realized_volatility_per_bar        DOUBLE PRECISION NOT NULL,
    -- 진단값 (모든 segment 계산 가능 → NOT NULL; §15.1, §19-10에 의해 최종 라벨에 영향 없음)
    max_abs_d                          DOUBLE PRECISION NOT NULL,
    max_jump_share                     DOUBLE PRECISION NOT NULL,
    bipower_variation                  DOUBLE PRECISION NOT NULL,
    jump_component                     DOUBLE PRECISION NOT NULL,
    jump_share_bv                      DOUBLE PRECISION NOT NULL,
    rv_plus                            DOUBLE PRECISION NOT NULL,
    rv_minus                           DOUBLE PRECISION NOT NULL,
    downside_vol_share                 DOUBLE PRECISION NOT NULL,
    amplitude_to_cost_ratio            DOUBLE PRECISION NOT NULL,
    low_tradeability_segment_flag      BOOLEAN          NOT NULL DEFAULT false,
    pullback_within_parent_trend_flag  BOOLEAN          NOT NULL DEFAULT false,
    -- 라벨 (꼬리 미확정 segment 는 NULL — B3 retrofit, lookahead 차단)
    direction_label                    direction_label,
    volatility_label                   volatility_label,
    final_label                        final_label,
    segment_bar_count                  INTEGER          NOT NULL,
    CONSTRAINT uk_segment_labels_run_tf_seg UNIQUE (run_id, timeframe, segment_id),
    -- 꼬리 미확정 segment 는 확정 5필드 + 라벨 3필드 전부 NULL, 그 외엔 전부 NOT NULL (상호 배타)
    -- B3: 라벨도 NULL 강제 — consumer 가 미확정 라벨을 사용하는 lookahead 차단
    CONSTRAINT ck_segment_labels_confirm CHECK (
        (is_tail_unconfirmed = true
         AND confirm_timestamp    IS NULL
         AND lag_bars             IS NULL
         AND lag_move             IS NULL
         AND capturable_amplitude IS NULL
         AND capturable_ratio     IS NULL
         AND direction_label      IS NULL
         AND volatility_label     IS NULL
         AND final_label          IS NULL)
        OR
        (is_tail_unconfirmed = false
         AND confirm_timestamp    IS NOT NULL
         AND lag_bars             IS NOT NULL
         AND lag_move             IS NOT NULL
         AND capturable_amplitude IS NOT NULL
         AND capturable_ratio     IS NOT NULL
         AND direction_label      IS NOT NULL
         AND volatility_label     IS NOT NULL
         AND final_label          IS NOT NULL)
    ),
    -- §15.1, §19-10: final_label 은 direction × volatility 로 deterministic.
    -- 꼬리 segment (is_tail_unconfirmed=true) 는 라벨 NULL 이므로 검사에서 제외.
    CONSTRAINT ck_segment_labels_final_label_consistency CHECK (
        is_tail_unconfirmed
        OR (direction_label::text || '_' || volatility_label::text) = final_label::text
    ),
    -- §10.3, §11.4 범위 검증 (DB 레벨 강제)
    CONSTRAINT ck_segment_labels_ranges CHECK (
        efficiency_ratio    BETWEEN 0 AND 1
        AND realized_volatility         >= 0
        AND realized_volatility_per_bar >= 0
        AND (capturable_ratio    IS NULL OR capturable_ratio    BETWEEN 0 AND 1)
        AND max_jump_share      BETWEEN 0 AND 1
        AND downside_vol_share  BETWEEN 0 AND 1
        AND (lag_bars            IS NULL OR lag_bars            >= 0)
        AND (lag_move            IS NULL OR lag_move            >= 0)
        AND (capturable_amplitude IS NULL OR capturable_amplitude >= 0)
        AND max_abs_d           >= 0
        AND bipower_variation   >= 0
        AND jump_component      >= 0
        AND jump_share_bv       >= 0
        AND rv_plus             >= 0
        AND rv_minus            >= 0
        AND amplitude_to_cost_ratio >= 0
        AND segment_bar_count   >= 1
        AND end_timestamp >= start_timestamp
    )
);

CREATE INDEX idx_segment_labels_run_tf       ON segment_labels (run_id, timeframe);
CREATE INDEX idx_segment_labels_run_tf_final ON segment_labels (run_id, timeframe, final_label);
CREATE INDEX idx_segment_labels_time_range   ON segment_labels (run_id, timeframe, start_timestamp);

COMMENT ON TABLE segment_labels IS 'segment-level 9라벨 + 진단값 (requirements §13.1)';

-- =========================================================================
-- bar_labels : bar-level 라벨 (requirements §13.2)
-- 1m 약 126만 행, 5m 약 25만 행 → 월 단위 RANGE 파티셔닝
-- PK에 파티션키(open_time) 포함 필수 → BIGSERIAL 단독 PK 컨벤션 예외
-- =========================================================================
CREATE TABLE bar_labels (
    run_id            BIGINT           NOT NULL REFERENCES labeling_runs(id) ON DELETE CASCADE,
    -- §13.2 필수 메타 (denormalize from labeling_runs)
    symbol            VARCHAR(32)      NOT NULL DEFAULT 'ETHUSDT',
    market            VARCHAR(32)      NOT NULL DEFAULT 'BINANCE_USDM_FUTURES',
    method_version    VARCHAR(64)      NOT NULL,
    timeframe         timeframe_enum   NOT NULL,
    open_time         TIMESTAMPTZ      NOT NULL,
    open              DOUBLE PRECISION NOT NULL,
    high              DOUBLE PRECISION NOT NULL,
    low               DOUBLE PRECISION NOT NULL,
    close             DOUBLE PRECISION NOT NULL,
    hlc3              DOUBLE PRECISION NOT NULL,
    segment_id        VARCHAR(64)      NOT NULL,
    direction_label   direction_label  NOT NULL,
    volatility_label  volatility_label NOT NULL,
    final_label       final_label      NOT NULL,
    CONSTRAINT pk_bar_labels PRIMARY KEY (run_id, timeframe, open_time),
    CONSTRAINT ck_bar_labels_ohlc CHECK (
        high >= GREATEST(open, close)
        AND low <= LEAST(open, close)
        AND high >= low
    ),
    -- §15.1, §19-10: final_label은 direction × volatility로 deterministic
    CONSTRAINT ck_bar_labels_final_label_consistency CHECK (
        (direction_label::text || '_' || volatility_label::text) = final_label::text
    ),
    -- 참조 무결성: segment_labels에 존재하는 (run_id, timeframe, segment_id) 만 허용
    -- → orphan segment_id 차단. 적재 순서: segment_labels 먼저, bar_labels 다음.
    CONSTRAINT fk_bar_labels_segment FOREIGN KEY (run_id, timeframe, segment_id)
        REFERENCES segment_labels (run_id, timeframe, segment_id)
        ON DELETE CASCADE
) PARTITION BY RANGE (open_time);

CREATE INDEX idx_bar_labels_run_tf_final ON bar_labels (run_id, timeframe, final_label);
-- FK 가 (run_id, timeframe, segment_id) 이므로 인덱스도 동일 컬럼 순서로 — FK 검사 최적화.
CREATE INDEX idx_bar_labels_segment      ON bar_labels (run_id, timeframe, segment_id);

COMMENT ON TABLE bar_labels IS 'bar-level 9라벨 (requirements §13.2). 월 단위 RANGE 파티셔닝.';

-- 월 파티션 : 2024-01 ~ 2026-06 (요구 기간 2024-01-01 ~ 2026-05-25 + 1개월 버퍼)
CREATE TABLE bar_labels_2024_01 PARTITION OF bar_labels FOR VALUES FROM ('2024-01-01 00:00:00+00') TO ('2024-02-01 00:00:00+00');
CREATE TABLE bar_labels_2024_02 PARTITION OF bar_labels FOR VALUES FROM ('2024-02-01 00:00:00+00') TO ('2024-03-01 00:00:00+00');
CREATE TABLE bar_labels_2024_03 PARTITION OF bar_labels FOR VALUES FROM ('2024-03-01 00:00:00+00') TO ('2024-04-01 00:00:00+00');
CREATE TABLE bar_labels_2024_04 PARTITION OF bar_labels FOR VALUES FROM ('2024-04-01 00:00:00+00') TO ('2024-05-01 00:00:00+00');
CREATE TABLE bar_labels_2024_05 PARTITION OF bar_labels FOR VALUES FROM ('2024-05-01 00:00:00+00') TO ('2024-06-01 00:00:00+00');
CREATE TABLE bar_labels_2024_06 PARTITION OF bar_labels FOR VALUES FROM ('2024-06-01 00:00:00+00') TO ('2024-07-01 00:00:00+00');
CREATE TABLE bar_labels_2024_07 PARTITION OF bar_labels FOR VALUES FROM ('2024-07-01 00:00:00+00') TO ('2024-08-01 00:00:00+00');
CREATE TABLE bar_labels_2024_08 PARTITION OF bar_labels FOR VALUES FROM ('2024-08-01 00:00:00+00') TO ('2024-09-01 00:00:00+00');
CREATE TABLE bar_labels_2024_09 PARTITION OF bar_labels FOR VALUES FROM ('2024-09-01 00:00:00+00') TO ('2024-10-01 00:00:00+00');
CREATE TABLE bar_labels_2024_10 PARTITION OF bar_labels FOR VALUES FROM ('2024-10-01 00:00:00+00') TO ('2024-11-01 00:00:00+00');
CREATE TABLE bar_labels_2024_11 PARTITION OF bar_labels FOR VALUES FROM ('2024-11-01 00:00:00+00') TO ('2024-12-01 00:00:00+00');
CREATE TABLE bar_labels_2024_12 PARTITION OF bar_labels FOR VALUES FROM ('2024-12-01 00:00:00+00') TO ('2025-01-01 00:00:00+00');
CREATE TABLE bar_labels_2025_01 PARTITION OF bar_labels FOR VALUES FROM ('2025-01-01 00:00:00+00') TO ('2025-02-01 00:00:00+00');
CREATE TABLE bar_labels_2025_02 PARTITION OF bar_labels FOR VALUES FROM ('2025-02-01 00:00:00+00') TO ('2025-03-01 00:00:00+00');
CREATE TABLE bar_labels_2025_03 PARTITION OF bar_labels FOR VALUES FROM ('2025-03-01 00:00:00+00') TO ('2025-04-01 00:00:00+00');
CREATE TABLE bar_labels_2025_04 PARTITION OF bar_labels FOR VALUES FROM ('2025-04-01 00:00:00+00') TO ('2025-05-01 00:00:00+00');
CREATE TABLE bar_labels_2025_05 PARTITION OF bar_labels FOR VALUES FROM ('2025-05-01 00:00:00+00') TO ('2025-06-01 00:00:00+00');
CREATE TABLE bar_labels_2025_06 PARTITION OF bar_labels FOR VALUES FROM ('2025-06-01 00:00:00+00') TO ('2025-07-01 00:00:00+00');
CREATE TABLE bar_labels_2025_07 PARTITION OF bar_labels FOR VALUES FROM ('2025-07-01 00:00:00+00') TO ('2025-08-01 00:00:00+00');
CREATE TABLE bar_labels_2025_08 PARTITION OF bar_labels FOR VALUES FROM ('2025-08-01 00:00:00+00') TO ('2025-09-01 00:00:00+00');
CREATE TABLE bar_labels_2025_09 PARTITION OF bar_labels FOR VALUES FROM ('2025-09-01 00:00:00+00') TO ('2025-10-01 00:00:00+00');
CREATE TABLE bar_labels_2025_10 PARTITION OF bar_labels FOR VALUES FROM ('2025-10-01 00:00:00+00') TO ('2025-11-01 00:00:00+00');
CREATE TABLE bar_labels_2025_11 PARTITION OF bar_labels FOR VALUES FROM ('2025-11-01 00:00:00+00') TO ('2025-12-01 00:00:00+00');
CREATE TABLE bar_labels_2025_12 PARTITION OF bar_labels FOR VALUES FROM ('2025-12-01 00:00:00+00') TO ('2026-01-01 00:00:00+00');
CREATE TABLE bar_labels_2026_01 PARTITION OF bar_labels FOR VALUES FROM ('2026-01-01 00:00:00+00') TO ('2026-02-01 00:00:00+00');
CREATE TABLE bar_labels_2026_02 PARTITION OF bar_labels FOR VALUES FROM ('2026-02-01 00:00:00+00') TO ('2026-03-01 00:00:00+00');
CREATE TABLE bar_labels_2026_03 PARTITION OF bar_labels FOR VALUES FROM ('2026-03-01 00:00:00+00') TO ('2026-04-01 00:00:00+00');
CREATE TABLE bar_labels_2026_04 PARTITION OF bar_labels FOR VALUES FROM ('2026-04-01 00:00:00+00') TO ('2026-05-01 00:00:00+00');
CREATE TABLE bar_labels_2026_05 PARTITION OF bar_labels FOR VALUES FROM ('2026-05-01 00:00:00+00') TO ('2026-06-01 00:00:00+00');
-- 사양 데이터 종료(2026-05-25)는 2026-05 파티션에 포함. 아래는 방어용 1개월 버퍼 (정상 운영 시 비어 있음).
CREATE TABLE bar_labels_2026_06 PARTITION OF bar_labels FOR VALUES FROM ('2026-06-01 00:00:00+00') TO ('2026-07-01 00:00:00+00');

-- DEFAULT 파티션: 범위 밖 (예: 사양 기간 이전/이후) INSERT 가 silent fail 하지 않도록 흡수.
-- 정상 운영 시 비어 있어야 함 — 채워졌다면 적재 검증 누락.
CREATE TABLE bar_labels_default PARTITION OF bar_labels DEFAULT;

-- =========================================================================
-- VIEW : requirements §17 산출물명과 1:1 노출
-- =========================================================================
-- WITH LOCAL CHECK OPTION: VIEW 통한 INSERT/UPDATE 시 timeframe 필터를 우회하는 행 거부.
CREATE VIEW bar_labels_1m     AS SELECT * FROM bar_labels     WHERE timeframe = '1m' WITH LOCAL CHECK OPTION;
CREATE VIEW bar_labels_5m     AS SELECT * FROM bar_labels     WHERE timeframe = '5m' WITH LOCAL CHECK OPTION;
CREATE VIEW segment_labels_1m AS SELECT * FROM segment_labels WHERE timeframe = '1m' WITH LOCAL CHECK OPTION;
CREATE VIEW segment_labels_5m AS SELECT * FROM segment_labels WHERE timeframe = '5m' WITH LOCAL CHECK OPTION;

-- joined_labels_1m_5m : 동일 run_id 내 1m bar ↔ 그 분을 덮는 5m bar(5분 floor 버킷).
-- LEFT JOIN: 5m 라벨이 부분 적재여도 1m 행은 보존되며 label_5m / segment_id_5m이 NULL로 노출된다.
-- (§9.2 "5분봉 라벨을 1분 단위 테이블에 붙일 수 있다" — additive 매핑, 필터링 아님)
CREATE VIEW joined_labels_1m_5m AS
SELECT
    b1.run_id,
    b1.symbol,
    b1.market,
    b1.open_time            AS open_time_1m,
    b1.final_label          AS label_1m,
    b1.segment_id           AS segment_id_1m,
    b5.open_time            AS open_time_5m_bucket,
    b5.final_label          AS label_5m,
    b5.segment_id           AS segment_id_5m
FROM bar_labels b1
LEFT JOIN bar_labels b5
  ON b1.run_id    = b5.run_id
 AND b5.timeframe = '5m'
 -- date_bin: SARGable + 파티션 가지치기 가능 (b5.open_time 자체 비교).
 AND b5.open_time = date_bin('5 minutes'::interval, b1.open_time, TIMESTAMPTZ '1970-01-01 00:00:00+00')
WHERE b1.timeframe = '1m';

-- =========================================================================
-- labeling_reports : §15 PASS/FAIL + §13 진단 리포트
-- =========================================================================
CREATE TABLE labeling_reports (
    id           BIGSERIAL PRIMARY KEY,
    run_id       BIGINT       NOT NULL REFERENCES labeling_runs(id) ON DELETE CASCADE,
    report_type  VARCHAR(32)  NOT NULL,
    passed       BOOLEAN,
    payload      JSONB        NOT NULL,
    created_at   TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    CONSTRAINT ck_labeling_reports_type CHECK (report_type IN ('validation', 'diagnostic'))
);

CREATE INDEX idx_labeling_reports_run_type ON labeling_reports (run_id, report_type);

COMMENT ON TABLE labeling_reports IS '검증/진단 리포트 (requirements §13, §15)';

-- =========================================================================
-- 마이그레이션 기록
-- =========================================================================
INSERT INTO schema_migrations (version) VALUES (1);

COMMIT;
