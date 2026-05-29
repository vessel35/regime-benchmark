# ETH/USDT 9-Label Regime Benchmark Labeler — 설계 문서 v1.0

> 대응 사양: [`requirements.md`](./requirements.md) (v1.1)
> 본 문서는 사양(*WHAT*)을 구현 가능한 아키텍처(*HOW*)로 번역하고, 사양에 명시되지 않은
> **PostgreSQL 영속화 계층**을 추가 정의한다.

---

## 1. Context & Goals

`requirements.md`는 Binance USDⓈ-M Futures ETHUSDT 1m·5m Kline에 대해 **방향성 3 × 변동성 3 =
9개 라벨**을 사후(post-hoc) 부여하는 deterministic labeling specification이다. 공식·검증 기준은
완비되어 있으나 아키텍처·모듈·기술 스택·영속화는 없다.

**목표**
- 사양의 19단계 파이프라인(§16)을 구현 가능한 모듈 구조로 설계한다.
- 최종 라벨과 진단값을 **PostgreSQL**에 저장하여 예측 모델 학습·비교에 재사용 가능하게 한다.
- 재현성(동일 입력+설정→동일 라벨)과 라벨 버전 관리를 보장한다.

**비목표**
- 라이브 매매·주문·체결·지갑 쓰기 (별도 preset).
- 전략 손익/포지션 성과 (사양 §0·§19에서 명시적으로 라벨 생성에서 배제).

---

## 2. Scope & 비범위

전체 파이프라인: **다운로드 → 품질검증 → 가격변환 → DC 방향성 → 변동성 → 진단 → 9라벨 →
bar 확장 → 1m·5m join → DB 적재 → 검증 리포트**.

| 범위 | 비범위 |
|---|---|
| Binance public monthly zip 수집·검증 | 거래소 실시간 스트림 |
| 사후 라벨 생성·진단·DB 영속화 | 주문/체결/슬리피지 실측, 전략 PnL |
| 재현 가능한 run 버전 관리 | 읽기전용 MCP DB(`crypto_data`/`signal`/`wallet_db`) 쓰기 |

> **하네스 제약**: 출력은 별도 **쓰기 가능 DB `regime_benchmark`** (예: 로컬 Docker PG16)에만
> 저장한다. MCP read-only 연결에는 절대 쓰지 않는다.

---

## 3. High-level Architecture

```text
                 ┌──────────── per timeframe τ ∈ {1m, 5m} (독립 처리) ────────────┐
binance zip ──▶ ingest ──▶ quality ──▶ transform ──▶ direction(DC) ──▶ volatility
                                                          │                  │
                                                          ▼                  ▼
                                                     diagnostics ──▶ labeling(9) ──▶ bar expand
                                                                              │
            ┌─────────────────────────────────────────────────────────────────┘
            ▼
     join(1m↔5m) ──▶ persistence(COPY→PG) ──▶ validation/report
```

핵심 불변식: **1m과 5m은 threshold·분위수를 각각 독립 계산**(§7.2, §15.1). 1m 경계값을 5m에
적용하지 않는다. join은 저장 후 DB VIEW로 수행하며 라벨 계산에 영향을 주지 않는다.

---

## 4. Tech Stack & Dependencies

| 영역 | 선택 | 근거 |
|---|---|---|
| 언어 | Python 3.12 | 하네스 표준 |
| 데이터 처리 | **polars** + numpy | 126만 행(1m) 벡터 연산·groupby 분위수에 빠르고 메모리 효율적 |
| 수치 타입 | **float64** | hlc3는 체결 가능 가격이 아니라 bar 요약값(§5.1 [RISK]). 라벨 math는 로그수익률 기반이므로 Decimal 불필요 |
| DB 드라이버 | **psycopg 3** + `COPY` | 126만 행 bulk 적재에 최적. ORM 미사용 |
| 설정 | pydantic v2 + YAML | §14 config 검증·타입 안정 |
| 테스트 | pytest | synthetic case + property + integration |

> `decimal-arithmetic-discipline`는 **체결가/수수료/포지션** 계산에 적용된다. 본 파이프라인은
> 사후 라벨 전용이고 비용(taker_fee/slippage)은 진단값으로만 쓰이므로 float64 일관 사용.
> design.md §11의 `amplitude_to_cost_ratio`도 진단 목적이라 float64로 충분.

---

## 5. Module / Package Layout

```text
regime_benchmark/
  pyproject.toml                    # py3.12; polars, numpy, psycopg[binary], pydantic, pyyaml, pytest
  config/labeling_config.yaml       # requirements §14 초기값
  migrations/001_init.sql           # §13 DDL (이 문서 별도 산출)
  src/regime_benchmark/
    config.py                       # pydantic 설정 모델 (LabelingConfig, TimeframeParams, PersistenceConfig)
    ingest/binance.py               # monthly zip 다운로드·파싱·기간 trim
    quality/checks.py               # §4.3 정합성, §4.4 5m resampling 검산
    transform/price.py              # hlc3, log price p_t, returns d_t
    direction/dc_engine.py          # turning-point 상태기계 (lookahead-safe)
    direction/segments.py           # segment 메트릭(M,A,L,ER) + 방향성 라벨
    volatility/realized.py          # RV, RV_per_bar, 분위수 라벨
    diagnostics/lag.py cost.py jump.py asymmetry.py er_corr.py
    labeling/assemble.py            # final 9-label, bar 상속 확장
    persistence/schema.py loader.py # 마이그레이션 적용 + COPY 적재
    validation/report.py            # synthetic case + 불변식 + PASS/FAIL
    pipeline.py                     # 단계 오케스트레이션(τ별), run 등록
  tests/
    test_synthetic_cases.py         # §10.4 Case A~E
    test_invariants.py              # §11 property 검증
    test_dc_engine.py               # turning point 교대성·lookahead
    test_persistence_roundtrip.py   # slice → DB → 재조회
```

---

## 6. Data Ingestion & Quality

### 6.1 수집 (`ingest/binance.py`)
- 소스: `https://data.binance.vision/data/futures/um/monthly/klines/ETHUSDT/{1m,5m}/`.
- 월별 zip 다운로드 → CSV 파싱 → §4.2 표준 컬럼 매핑(`open_time`,`open`,`high`,`low`,`close`,`volume`,`close_time`,`quote_asset_volume`,`number_of_trades`,`taker_buy_base_volume`,`taker_buy_quote_volume`,`ignore`).
- 기간 trim: `2024-01-01 00:00:00 UTC ~ 2026-05-25 23:59:59 UTC`. 라벨 기준 timestamp = `open_time`(§4.2).
- 미완성 candle 제외(§4.3): 다운로드 종료 후 확정 candle만.

### 6.2 품질 검증 (`quality/checks.py`) — §4.3
| 검사 | 기준 | 실패 처리 |
|---|---|---|
| 중복 `open_time` | timeframe 내 0건 | 중단(§19-2) |
| 누락 candle | 기대 그리드 대비 gap 식별·리포트 | gap 많으면 구간 제외/중단(§19-1) |
| OHLC 관계 | `high≥max(open,close)`, `low≤min(open,close)`, `high≥low` | 위반 행 격리·리포트 |
| 기간 커버리지 | 요구 기간 완전 포함 | 미달 시 중단(§19-1) |

### 6.3 5m 검산 (선택, §4.4)
1m → 5m resampling 후 Binance 5m과 비교. 반복 불일치 시 timestamp 정렬·누락·경계 점검.

```text
open_5m_check   = 첫 번째 1분봉 open
high_5m_check   = 5개 1분봉 high 의 max
low_5m_check    = 5개 1분봉 low 의 min
close_5m_check  = 다섯 번째 1분봉 close
volume_5m_check = 5개 1분봉 volume 의 sum
```

---

## 7. Price Transform (`transform/price.py`) — §5

```text
P_t = (high_t + low_t + close_t) / 3      # hlc3
p_t = ln(P_t)
d_t = p_t - p_{t-1}                        # 첫 bar의 d_t = NaN → segment 계산에서 제외
```
polars expression으로 벡터화. [RISK] hlc3는 주문 체결가로 해석 금지(§5.1).

---

## 8. Directional Change Engine (핵심, `direction/dc_engine.py`) — §6

### 8.1 Threshold (§6.1)
```text
theta_dc_τ = Quantile(|d_t|, q_dc_τ) × k_dc_τ
```
초기 후보: 1m `q=0.80, k∈{3,4,5}`; 5m `q=0.80, k∈{2,3,4}`. 캘리브레이션은 §15에서.

> **분위수 모집단 (frozen calibration 방식)** — 본 라벨러는 **frozen** 정책:
> `theta_dc_τ`는 사전 지정된 **calibration split**(예: `2024-01-01 ~ 2024-12-31`)의 `|d_t|`
> 분포에서 **사전 산출 후 lock**, 라벨 생성 구간(validation/test) 전체에 동일 값 적용.
> 산출된 값은 `labeling_run_params.theta_dc`에 저장되어 재현성·causality를 보장한다.
> expanding-window causal 변형이 필요하면 별도 method_version 으로 분리 (현 spec 미포함).

### 8.2 Lookahead-safe 상태기계 (§6.2)
turning point는 **과거 극값**이지만 **확정(confirm)** 은 이후 bar에서만 발생 → 미래 정보 미사용.

**부트스트랩 (결정성·재현성)**: 첫 bar `t=0`에서 `mode=SEEK_UP`, `p_ext=p_0`, `t_ext=0`으로
시작. 즉 첫 bar를 잠정 trough로 간주하고 상승 반전을 탐색한다. 가격이 먼저 하락하면 trough가
갱신되고, 먼저 상승하면 TP_1=`(trough at t=0)`이 확정된다. 이 선택은 결정론적이며 §15.1
재현성을 보장한다. 다른 부트스트랩 선택은 TP_1 식별만 바꾸므로 단위테스트(`test_dc_engine.py`)
에서 fixed sequence에 대한 TP_1을 고정값으로 검증한다.

```text
# bootstrap: mode=SEEK_UP, p_ext=p_0, t_ext=0 (첫 bar를 잠정 trough로)
for t in 1..N-1:
  if mode == SEEK_UP:                      # 직전 trough에서 상승 반전 탐색
      if p_t < p_ext: p_ext, t_ext = p_t, t          # trough 갱신
      elif p_t - p_ext >= theta_dc:                  # 상승 확정 (confirm at bar t) — [RISK §7] p_t는 hlc3 기반
          emit TP(trough, t_ext, confirm_bar=t); mode = SEEK_DOWN
          p_ext, t_ext = p_t, t                      # 새 peak 추적 시작
  else:  # SEEK_DOWN                        # 직전 peak에서 하락 반전 탐색
      if p_t > p_ext: p_ext, t_ext = p_t, t          # peak 갱신
      elif p_ext - p_t >= theta_dc:                  # 하락 확정 (confirm at bar t)
          emit TP(peak, t_ext, confirm_bar=t); mode = SEEK_UP
          p_ext, t_ext = p_t, t
# 종료 시점에서 마지막 추적 중인 극값은 미확정 → 꼬리 segment `is_tail_unconfirmed=true`
```
- 출력: 시간순 turning point 열 `TP_1..TP_n` (peak/trough 교대, §11.2 검증).
- 마지막 미확정 구간은 `is_tail_unconfirmed=true`로 별도 표시(§11.2; 확정 진단값은 NULL 허용).

### 8.3 Segment 메트릭 (`direction/segments.py`, §6.3)
인접 TP 쌍 `S_j=[TP_j, TP_{j+1}]`:
```text
N_j = end_j - start_j + 1
M_j = p_end - p_start ; A_j = |M_j|
L_j = Σ |d_t| (start_j+1..end_j) ; ER_j = A_j / L_j   (L_j=0 → ER_j=0)
```

### 8.4 방향성 라벨 (§6.4)
```text
direction_j = UP   if M_j>0 and N_j≥min_segment_bars_τ and A_j≥theta_amp_τ
              DOWN if M_j<0 and N_j≥min_segment_bars_τ and A_j≥theta_amp_τ
              NON_DIRECTIONAL otherwise
```
`theta_amp_τ = theta_dc_τ`(초기). **ER은 방향성 필수 조건에서 제외**(§6.4 주의, Case E §10.4):
경로가 거칠어도 순방향 이동이 충분하면 방향성 인정. `ER_j`는 `segment_labels.efficiency_ratio`에 저장.

### 8.5 경로 효율성 진단 라벨 `path_quality_j` (§6.4)

ER로부터 파생되는 진단 라벨 — **9개 기본 라벨에 포함하지 않으며 컬럼으로도 저장하지 않는다**.
application 단계에서 `efficiency_ratio`로부터 즉시 계산 (DDL 미변경, 캘리브레이션 자유도 확보).

```text
path_quality_j =
  EFFICIENT,   if ER_j >= theta_er_τ
  INEFFICIENT, otherwise
```

| 파라미터 | 초기 후보 |
|---|---|
| `theta_er_τ` | `0.35 ~ 0.50` (1m·5m 공통 초기 grid, 캘리브레이션 §12.1) |

용도: HIGH_VOL 방향성 segment 중 "정연한 추세 vs 거친 추세" 사후 진단.
`final_label`에 영향 없음 — §19-10 `ck_*_final_label_consistency`로 구조적 차단.
ER이 `RV_per_bar`와 강한 상관(§10.2 `|ρ| ≥ 0.80`)이면 본 진단도 진단 전용으로 제한.

---

## 9. Volatility Labeling (`volatility/realized.py`) — §7

```text
RV_j        = sqrt(Σ d_t²)  (start_j+1..end_j)         # N_j-1 개 d 합
RV_per_bar_j = RV_j / sqrt(N_j)                         # 길이 보정, 기본 vol_score
```
> **주의 (사양 일치)**: 분모는 `sqrt(N_j-1)` 이 아니라 **`sqrt(N_j)`** 이다. RV가 `N_j-1`개 returns의
> 제곱합 sqrt임에도 분모를 N_j로 둔 것은 사양 §7.1 그대로다. 구현 시 "정정" 금지 — 단위테스트로 고정.
timeframe별 분위수(혼용 금지, §7.2):
```text
Q_low_τ  = Quantile(RV_per_bar, 0.33) ; Q_high_τ = Quantile(RV_per_bar, 0.66)
volatility_j = LOW_VOL  if score ≤ Q_low_τ
               MID_VOL  if Q_low_τ < score ≤ Q_high_τ
               HIGH_VOL if score > Q_high_τ
```
분위수 후보 25/75·20/80은 캘리브레이션 옵션(§7.2 주의, §18).

> **분위수 모집단 (frozen)** — `Q_low_τ`/`Q_high_τ`는 §8.1과 동일하게 **frozen calibration split의 segment
> `RV_per_bar` 분포**에서 산출 후 lock. validation/test 구간 전체에 동일 경계 적용. 산출값은
> `labeling_run_params.q_low`/`q_high`(분위수 위치)에 저장하되, 실제 적용 경계(score 값)는 application
> 캘리브레이션 단계에서 frozen 후 라벨 생성에 사용.

---

## 10. Auxiliary Diagnostics (`diagnostics/`) — §6.5, §7.3

| 모듈 | 산출 | 사양 |
|---|---|---|
| `lag.py` | `confirm_bar`, `lag_bars`, `lag_move`, `capturable_amplitude/ratio` | §6.5.1 |
| `cost.py` | `estimated_round_trip_cost_log`, `amplitude_to_cost_ratio`, `low_tradeability_segment_flag` | §6.5.2 |
| `jump.py` | `max_abs_d`, `max_jump_share`, `bipower_variation`, `jump_component`, `jump_share_bv` | §7.3.1 |
| `asymmetry.py` | `rv_plus`, `rv_minus`, `downside_vol_share` | §7.3.3 |
| `er_corr.py` | `rho_er_vol_τ = SpearmanCorr(ER, RV_per_bar)` (per τ) | §7.3.2 |

`confirm_bar_j` = `start_j` 이후 `|p_t - p_start| ≥ theta_dc`인 최초 bar → `lag_bars = confirm_bar - start ≥ 0`.
`pullback_within_parent_trend_flag`(§6.5.3): NON_DIRECTIONAL이고 양 옆 segment 동방향 +
`A_{j-1}+A_{j+1} ≥ theta_parent_amp` + `N_j ≤ pullback_max_bars`.

### 10.1 거래가능성 공식 (cost.py, §6.5.2)

```text
estimated_round_trip_cost_log = 2 × taker_fee_rate + slippage_rate_estimate
amplitude_to_cost_ratio_j     = A_j / estimated_round_trip_cost_log
low_tradeability_segment_flag = (amplitude_to_cost_ratio_j < 3)     # 초기 진단 임계
```
| `amplitude_to_cost_ratio` | 해석 |
|---|---|
| `< 1` | segment 진폭 < 왕복 비용 |
| `1 ~ 3` | 비용 대비 여유 작음 (flag=true) |
| `≥ 3` | 비용 대비 여유 있음 (flag=false) |

`taker_fee_rate`, `slippage_rate_estimate`는 `labeling_run_params`에서 timeframe별 주입.
임계값 `3`은 §6.5.2 초기값 — DB CHECK로 강제하지 않음(캘리브레이션 여지). application에서 계산·저장.

### 10.2 ER-RV 중복 진단 임계 (er_corr.py, §7.3.2)

`rho_er_vol_τ = SpearmanCorr(ER_j, RV_per_bar_j)` (timeframe별 산출):

| `|ρ_er_vol_τ|` | 판정 | 대응 |
|---|---|---|
| `< 0.60` | 중복 낮음 | ER 진단값으로 정상 사용 |
| `0.60 ~ 0.80` | 중복 가능 | 보고서에 표시, 모니터링 |
| `≥ 0.80` | 중복 강함 | ER을 모델 feature가 아닌 진단 전용으로 제한 (§2.2) |

> **불변식**: 위 모든 진단값(lag·cost·jump·asymmetry·ER)은 **최종 9라벨 조건에 절대 영향을 주지 않는다**
> (§15.1, §19-10). DB의 `ck_*_final_label_consistency`가 구조적으로 차단. 저장만 한다.

### 10.3 사용 제약 — 사후 진단 only (S1~S5)

§6.5·§7.3의 모든 보조 진단(`lag/capturable`, `cost/tradeability`, `jump/BV`, `asymmetry`, `ER-RV corr`)은
**사후 진단 전용**이다. forward-looking model feature 또는 live signal 로 사용 금지 — segment 가 닫히고
`confirm_bar_j` 시점에서야 확정되기 때문 (§11.1 availability rule 참조).

| 진단 | 가정 / 한계 | 부정확/오용 위험 |
|---|---|---|
| **lag·capturable** (S1·S2) | 가상 entry timing = `confirm_bar_j + 1` bar **open** (next-bar open 모델). post-hoc 통계. | feature로 쓰면 lookahead. fill timing 변경 시 값 변동. |
| **lag_move_j** (S3) | `p_{confirm_bar_j}` = `ln(hlc3_{confirm_bar_j})` 사용 — `[RISK — §7]` 통계 근사. | HLC3는 체결가 아님 → 실 슬리피지 과소평가. |
| **`estimated_round_trip_cost_log = 2·taker_fee + slippage`** (S4) | **size-independent** 가정. 작은 size + 평상시 호가에 한정 유효. | 큰 size, 얇은 호가, 변동성 폭증 시 무효 — impact 모델 필요. |
| **`amplitude_to_cost_ratio`** (S5) | descriptive 진폭-비용 비율. | **expected profitability 아님**. 실 마진은 size·liquidity·체결 timing·book 깊이에 의존. |

> consumer가 본 절 진단값을 model feature로 export 하려면 `label_available_at` (§11.1) 이후 시각에서만
> 참조하고, size·liquidity·fill timing 가정을 명시 검증해야 한다.

---

## 11. Final Label Assembly (`labeling/assemble.py`) — §8, §9

```text
final_label_j = direction_j + "_" + volatility_j     # 9개 중 하나
label_t = final_label_j  for t ∈ S_j                  # bar 상속 확장
```
1m·5m는 각자 계산 후 저장. join(`joined_labels_1m_5m`)은 §13 DB VIEW로 제공 — 5분 버킷
매핑은 저장 데이터에 대한 조회 편의이며 라벨 자체를 변경하지 않는다(§9.2 주의).

> **§9.3 timeframe 라벨 불일치는 정상 현상**. 동일 시각에 1m이 `UP_HIGH_VOL`인데 5m이
> `NON_DIRECTIONAL_MID_VOL`처럼 다르게 나올 수 있다. 1m 단기 흔들림이 5m 기준 방향성으로는
> 약하게 보이는 등 timeframe별 시장 구조가 다르게 측정되기 때문 — **오류가 아니다**.
> 소비자(예측 모델 학습)는 두 timeframe 라벨을 서로 모순으로 다루지 말고 독립 정보로 활용한다.

### 11.1 Causal consumption — assignment vs availability

**용어 구분 (N1)**:
- `label_assignment_time` = 라벨이 DB에 저장된 bar의 `open_time` (post-hoc annotation 시각).
- `label_availability_time` = causal consumer (예측 모델·백테스트·실거래 신호)가 안전하게 사용할 수
  있는 가장 이른 시각. **`assignment ≠ availability`** — 두 시각은 다르며 본 절의 규칙으로 관리.

**Availability rule (B2)** — segment `S_j = [TP_j, TP_{j+1}]`의 `final_label_j`가 모든 bar에 상속되지만,
causal 사용은 다음 조건에서만 허용:

```text
label_t 는 t >= confirm_bar_j + 1 (bar) 부터 사용 가능
```

- bar `t ∈ [start_j, confirm_bar_j - 1]` 는 **post-hoc annotation only** — feature·signal로 쓰면 lookahead leak.
- bar `t ∈ [confirm_bar_j, end_j]` 는 segment 방향이 이미 확정된 후 시각이므로 causal 사용 가능.
- 권장 운영:
  - DB 는 모든 bar 에 라벨 상속 저장 (annotation 가치 보존).
  - consumer 는 조회 시 `WHERE bar.open_time > seg.confirm_timestamp` (또는 `>= seg.confirm_timestamp + bar_interval`) 로 필터.
  - 또는 application 레이어가 `label_available_at = confirm_timestamp + bar_interval` 컬럼을 derive 해서 사용.

**Same-bar HLC3 contract (B4)** — `label_t` 와 `segment_id_t` 의 모든 변종은 **`close_time_t`** 에 결정된다
(hlc3 = (high+low+close)/3, §7 [RISK]; high/low 는 bar 종료 시에만 완전 관측). 따라서:

```text
가장 이른 causal 실행/모델 결정: bar (t+1) 의 open
```

- bar `t` 의 mid-bar 시점에 `label_t` 를 사용하면 **same-bar high/low leakage** (intra-bar 극값이 결정에 흡수).
- 라이브 신호 산출 모델은 항상 `t+1` 부터 시작하도록 시뮬레이션·백테스트를 설계.
- 데이터셋에 `label_available_at = close_time_t` 또는 `next_open_time = open_time_{t+1}` 컬럼을 derive 권장.

**Tail segment 라벨 (B3)** — `is_tail_unconfirmed=true` segment는 confirm 시각이 없으므로
`direction_label / volatility_label / final_label` 모두 NULL이며 (§13.3 CHECK), 그 segment에 속하는
bar는 **`bar_labels`에 적재하지 않는다**. "라벨 가능 구간" = 첫 TP 확정 이후 ~ 마지막 confirm 까지로 정의.

---

## 12. Configuration (`config.py`, `config/labeling_config.yaml`) — §14

pydantic 모델: `LabelingConfig{ method_version, data, price_field, timeframes, direction_method,
volatility_method, auxiliary_metrics, persistence }`. timeframe별 `TimeframeParams{ q_dc, k_dc,
min_segment_bars, theta_amp_policy, pullback_max_bars }`. 캘리브레이션 워크플로: §8.1/§9의 후보
grid별 (segment 수, 평균/중앙 길이, NON_DIRECTIONAL 비율, 라벨 분포, jump share)를 산출해 best 선택(§2.2).

### 12.1 캘리브레이션 변경 기준 (§2.2 운영 정의)

각 파라미터의 best 제안·변경 트리거·변경 방법. **사양 §2.2 그대로** — 변경 시 본 문서와 §2.2 동시 갱신.

| 파라미터 | best 제안 | 변경 트리거 | 변경 방법 |
|---|---|---|---|
| 기준 가격 | `hlc3` | `close` 대비 turning point 과다/과소 생성 | `close`·`hlc3` 라벨 일치율, segment 수, 평균 길이 비교 |
| `theta_dc` | `Quantile(\|d\|, 0.80)·k`, 1m: `k∈{3,4,5}`, 5m: `k∈{2,3,4}` | segment 수 과다, 평균/중앙 길이 극단 | 후보 grid별 segment 수·길이·NON_DIRECTIONAL 비율·라벨 안정성 비교 |
| `min_segment_bars` | 1m: `{5,10,15}`, 5m: `{3,5,8}` | 강한 impulse가 NON_DIRECTIONAL로 밀림 | `A_j ≥ theta_impulse` 예외 검토. 기본 9라벨 유지, `short_impulse_flag`로 우선 기록 |
| 변동성 분위수 | 33/66 | HIGH_VOL 비율 과다 → 극단 의미 약함 | 25/75 또는 20/80 병행 계산, 라벨 분포·지속시간·jump share 비교 |
| NON_DIRECTIONAL 조건 | `N_j < min_segment_bars` 또는 `A_j < theta_amp` | NON_DIRECTIONAL이 동일 방향 큰 segment 사이에 끼는 빈도 | `pullback_within_parent_trend_flag` 추가, parent segment 별도 컬럼 관리. 9라벨 자체는 불변 |
| `ER_j` | 최종 라벨 조건에서 제외, 진단값으로 저장 | ER과 변동성 강한 상관 | `\|ρ(ER, RV_per_bar)\| ≥ 0.80`이면 ER은 진단 전용 (§10.2) |
| 거래비용·슬리피지 | 라벨 생성 미사용, 진단값으로만 | `amplitude_to_cost_ratio` 낮은 segment 다수 | `low_tradeability_segment_flag` 표시, 9라벨 조건 불변 |

---

## 13. PostgreSQL Persistence Layer

> 대상 DB: 별도 쓰기 가능 `regime_benchmark` (PG16, Docker `postgis/postgis:16-3.4`).
> DSN은 env var `${REGIME_BENCHMARK_DB_URL}`. 마이그레이션은 `migrations/001_init.sql`(BEGIN/COMMIT, `schema_migrations` 기록).

### 13.1 ENUM 타입
```sql
CREATE TYPE timeframe_enum   AS ENUM ('1m','5m');
CREATE TYPE direction_label  AS ENUM ('UP','DOWN','NON_DIRECTIONAL');
CREATE TYPE volatility_label AS ENUM ('LOW_VOL','MID_VOL','HIGH_VOL');
CREATE TYPE final_label      AS ENUM (
  'UP_LOW_VOL','UP_MID_VOL','UP_HIGH_VOL',
  'DOWN_LOW_VOL','DOWN_MID_VOL','DOWN_HIGH_VOL',
  'NON_DIRECTIONAL_LOW_VOL','NON_DIRECTIONAL_MID_VOL','NON_DIRECTIONAL_HIGH_VOL');
```

### 13.2 라벨 run / 파라미터 (재현성)
- `labeling_runs` — 파이프라인 1회 실행(1m·5m 모두 포함). `id BIGSERIAL PK`, `method_version`,
  `symbol`, `market`, `period_start_utc`, `period_end_utc`, `price_field`, `git_commit`, `created_at`,
  **`run_status ∈ {loading, completed, failed}` + `completed_at`** (부분/실패 run 식별).
  `ck_labeling_runs_period`로 `period_end_utc >= period_start_utc` 강제.
  1m·5m join이 **동일 `run_id`** 안에서 이뤄지도록 timeframe를 run에서 분리하지 않는다.
- `labeling_run_params` — timeframe별 캘리브레이션 값. `run_id FK`, `timeframe`, `q_dc`, `k_dc`,
  `min_segment_bars`, `theta_dc`, `theta_amp`, `q_low`, `q_high`, `taker_fee_rate`,
  `slippage_rate_estimate`, `params_extra JSONB`, `UNIQUE(run_id, timeframe)`.

> **소비자 규칙**: 라벨 조회는 `WHERE run_status='completed'` 로 필터해 부분 적재된 run을 배제한다.
> 적재 종료 시점에 application이 `run_status='completed'`, `completed_at=NOW()` 로 한 번에 전환.

> **동일 `run_id` 결합의 trade-off (의식적 설계)**: 사양 §9.1은 1m·5m 라벨링이 timeframe 독립임을
> 명시한다. 본 설계는 **동일 run 안에서 두 timeframe을 함께 산출**해 `joined_labels_1m_5m` join을 단순한
> `run_id` 일치로 정의한다. 결과: (a) 1m·5m 라벨 수치는 여전히 timeframe별 분위수·threshold로 독립
> 계산되며 사양 §7.2·§19-5를 위반하지 않는다. (b) **운영 제약**: 1m만(또는 5m만) 재실행하려면 새 run을 만들고
> 짝 timeframe도 다시 산출해야 한다. 비대칭 재실행이 잦으면 `run_pair_mappings(run_1m_id, run_5m_id)`
> 보조 테이블을 추가하는 확장 옵션을 고려할 수 있다(현재 미도입).

### 13.3 `segment_labels` (§13.1) — 수천~수만 행, 파티셔닝 불필요
`id BIGSERIAL PK`, `run_id FK`, **denormalize: `symbol`, `market`, `method_version`** (§13.1 출력 스키마 명시
필드), `timeframe`, `segment_id`, **`is_tail_unconfirmed BOOLEAN NOT NULL DEFAULT false`** (§11.2 확인 불가
segment 표시), + §13.1 전 필드(start/end/confirm ts, lag·capturable, start/end_price_hlc3, log_move,
amplitude, path_length, efficiency_ratio, RV, RV_per_bar, max_abs_d, max_jump_share, bipower_variation,
jump_component, jump_share_bv, rv_plus, rv_minus, downside_vol_share, amplitude_to_cost_ratio,
low_tradeability_segment_flag, pullback_within_parent_trend_flag, direction_label, volatility_label,
final_label, segment_bar_count). 수치 `DOUBLE PRECISION`. `UNIQUE(run_id, timeframe, segment_id)`.

**CHECK 제약** (DB 레벨 사양 강제):
- `ck_segment_labels_confirm`: **상호 배타** — `is_tail_unconfirmed=true` 이면 **확정 5필드 + 라벨 3필드**
  (`confirm_timestamp, lag_bars, lag_move, capturable_amplitude, capturable_ratio,
  direction_label, volatility_label, final_label`) 전부 NULL, 그 외에는 전부 NOT NULL. **꼬리 segment의
  미확정 라벨을 consumer가 사용하는 lookahead 경로를 구조적으로 차단** (B3).
- `ck_segment_labels_final_label_consistency`: `is_tail_unconfirmed=true` 인 행은 제외(라벨 NULL이므로),
  그 외 행은 `final_label = direction_label || '_' || volatility_label` (§19-10 진단값 오염 차단).
- `ck_segment_labels_ranges`: §10.3·§11.4 모든 범위/비음수 조건(`0≤ER≤1`, `0≤capturable_ratio≤1`, `0≤max_jump_share≤1`, `0≤downside_vol_share≤1`, `RV/RV_per_bar/max_abs_d/BV/jump_component/jump_share_bv/rv_plus/rv_minus/amplitude_to_cost_ratio/lag_move/capturable_amplitude ≥ 0`, `lag_bars ≥ 0`, `segment_bar_count ≥ 1`, `end ≥ start`).
- 진단값(`max_*`, `jump_*`, `rv_plus/minus`, `downside_vol_share`, `amplitude_to_cost_ratio`, 두 flag)은 **NOT NULL** — §13.1 출력 스키마가 요구. flag 기본값 `false`.
- 라벨 3필드(`direction_label`, `volatility_label`, `final_label`)는 **NULL 허용** (tail 전용).

### 13.4 `bar_labels` (§13.2) — **월별 RANGE 파티셔닝** (1m ~126만 행)
컬럼: `run_id FK`, **denormalize: `symbol`, `market`, `method_version`** (§13.2 출력 스키마),
`timeframe`, `open_time TIMESTAMPTZ`, `open/high/low/close/hlc3 DOUBLE PRECISION`,
`segment_id NOT NULL`, `direction_label`, `volatility_label`, `final_label`.
```sql
PRIMARY KEY (run_id, timeframe, open_time)   -- 파티션키(open_time) 포함 필수
... PARTITION BY RANGE (open_time);          -- bar_labels_2024_01 … 2026_05 (+ 2026_06 버퍼)
```
> **convention 예외**: 파티션 테이블은 PK에 파티션키를 포함해야 하므로 skill의 `id BIGSERIAL`
> 단독 PK 대신 자연 복합키를 사용한다.

**CHECK / FK 제약**:
- `ck_bar_labels_ohlc`: §4.3 OHLC 관계 강제.
- `ck_bar_labels_final_label_consistency`: §19-10 진단값 오염 차단 (segment_labels와 동일).
- `fk_bar_labels_segment`: composite FK `(run_id, timeframe, segment_id) → segment_labels(run_id, timeframe, segment_id)` — orphan segment_id 차단. **적재 순서 제약**: `segment_labels` 먼저 COPY 적재, 그다음 `bar_labels` (`ON DELETE CASCADE`).

인덱스: `idx_bar_labels_run_tf_final (run_id, timeframe, final_label)`(라벨 분포 조회),
`idx_bar_labels_segment (run_id, segment_id)`(segment 조인 + FK 검사).

> **라벨 가능 구간만 적재**: 첫 TP 확정 이전 bar는 어느 segment에도 속하지 않으므로 적재하지 않는다
> (§11.2 "segment가 전체 라벨링 가능 구간을 덮음"). 꼬리 미확정 segment의 bar는 적재하되
> 해당 segment에 `is_tail_unconfirmed=true` 표시.

### 13.5 VIEW (requirements §17 산출물명 1:1 노출)
```sql
CREATE VIEW bar_labels_1m AS SELECT * FROM bar_labels WHERE timeframe='1m';
CREATE VIEW bar_labels_5m AS SELECT * FROM bar_labels WHERE timeframe='5m';
-- LEFT JOIN: 5m 부분 적재 시 1m 행은 보존, label_5m NULL로 누락 신호 (§9.2 additive 매핑)
-- §13.3 joined_output_1m 스키마에 symbol·market 포함
-- §9.3 label_1m ≠ label_5m 인 행은 정상 (timeframe별 시장 구조 차이) — 오류로 처리 금지
CREATE VIEW joined_labels_1m_5m AS
SELECT b1.run_id, b1.symbol, b1.market,
       b1.open_time AS open_time_1m, b1.final_label AS label_1m,
       b1.segment_id AS segment_id_1m,
       b5.open_time AS open_time_5m_bucket, b5.final_label AS label_5m,
       b5.segment_id AS segment_id_5m
FROM bar_labels b1
LEFT JOIN bar_labels b5
  ON b1.run_id = b5.run_id AND b5.timeframe='5m'
 AND b5.open_time = date_bin('5 minutes'::interval, b1.open_time, TIMESTAMPTZ '1970-01-01 00:00:00+00')
WHERE b1.timeframe='1m';
```

### 13.6 `labeling_reports` (§13, §15)
`id BIGSERIAL PK`, `run_id FK`, `report_type VARCHAR`(`validation`|`diagnostic`), `passed BOOLEAN`,
`payload JSONB`, `created_at`.

### 13.7 적재 전략 (`persistence/loader.py`)
- polars DataFrame → psycopg3 `COPY ... FROM STDIN`. segment·bar 분리 적재.
- **append-only 버전 관리**: 동일 `(method_version, period)` 재실행 시 새 `run_id` 생성 → 과거 라벨 보존.
- 파티션은 마이그레이션에서 사전 생성(2024-01~2026-05) 또는 적재 전 `CREATE TABLE IF NOT EXISTS` 동적 생성.

### 13.8 DB 부트스트랩 (role + database)

| 파일 | 역할 |
|---|---|
| `migrations/000_init_roles.sql` | `regime_owner` role + `regime_benchmark` DB 생성 (superuser 권한, idempotent) |
| `migrations/001_init.sql` | 스키마 생성 (`regime_owner` 권한) |
| `scripts/bootstrap_db.sh` | 위 둘을 순차 적용 |
| `env.example` | env 템플릿 |

**컨텍스트 분리**:
- **`.env` (application 런타임용)** — `REGIME_BENCHMARK_DB_URL` 만 보관. `env.example` 참조.
- **부트스트랩 (1 회용)** — 셸 inline 으로 전달. 자격을 영구 저장하지 않는다.

**부트스트랩 변수** (`./scripts/bootstrap_db.sh`):
- `REGIME_SUPERUSER` — superuser role 이름 (예: `trader`, `postgres`)
- `REGIME_OWNER_PASSWORD` — 새로 만들 `regime_owner` 비밀번호
- `PGPASSWORD` — superuser 비밀번호 (또는 `~/.pgpass` 활용)
- 선택: `PGHOST` (기본 `localhost`), `PGPORT` (기본 `5432`)

**실행 예**:
```bash
REGIME_SUPERUSER=trader \
PGPASSWORD='<trader_pw>' \
REGIME_OWNER_PASSWORD='<신규>' \
./scripts/bootstrap_db.sh
```

`~/.pgpass`를 미리 등록해 두면 `PGPASSWORD` 생략 가능:
```
# ~/.pgpass (chmod 600)
localhost:5432:postgres:trader:<trader_pw>
```

부트스트랩 종료 후 `regime_owner` 비밀번호는 `~/.pgpass` 또는 `REGIME_BENCHMARK_DB_URL` 인라인으로 옮긴다:
```
localhost:5432:regime_benchmark:regime_owner:<owner_pw>
```

**application 사용**:
```bash
cp env.example .env       # 필요 시 DSN 의 비밀번호 인라인
set -a; source .env; set +a
```

**멱등성·재실행**:
- 1단계는 role/DB가 이미 있으면 NOTICE 후 no-op (`CREATE ROLE` DO block + `CREATE DATABASE` \gexec).
- 2단계는 두 번째 실행 시 ENUM/테이블 중복 생성 에러로 BEGIN/COMMIT 트랜잭션 전체 롤백 — 안전 거부. 후속 변경은 `migrations/002_*.sql`.

`regime_owner`가 DB 소유자이므로 별도 GRANT 불필요 (owner 단일 정책). 권한 분리가 필요해지면
별도 마이그레이션 파일(`002_*.sql`)로 `regime_writer/regime_reader`를 추가한다.

**알려진 운영 trade-off** (보안):
- **PostgreSQL 서버 로그**: `EXECUTE format('CREATE ROLE ... PASSWORD %L', pw)`가 `log_statement='all'` 시
  서버 로그에 평문 노출. 부트스트랩 직전 `ALTER SYSTEM SET log_statement='none'; SELECT pg_reload_conf();`
  로 임시 OFF 권장.
- **`pg_stat_activity.query`**에도 일시 노출 가능. 운영자가 다른 세션에서 조회하지 않는 시점에 실행.
- **application = owner**: 단일 role 정책이라 application이 schema 변경 권한 보유. 권한 최소화가 필요하면
  추가 마이그레이션으로 `regime_app` role 분리.
- **default 파티션 `bar_labels_default`**: 정상 운영 시 비어 있어야 함. 채워졌다면 적재 검증 누락 — 모니터링.

---

## 14. Validation & PASS/FAIL Reporting (`validation/report.py`) — §10, §11, §15

- **Synthetic Case A~E**(§10.4): §16 testing strategy 참조.
- **불변식**(§11): 아래 매트릭스 — 각 항목은 *application 검증* 또는 *DB CHECK* 둘 중 하나로 강제.
- 결과를 `labeling_reports`에 JSONB로 저장 + PASS/FAIL 판정(§15).

### 14.1 §11 체크리스트 → 강제 위치 매트릭스

**§11.1 데이터 검증**

| 항목 | 강제 |
|---|---|
| Binance 1m·5m 다운로드 | `ingest/binance.py` |
| 기간 trim (2024-01-01 ~ 2026-05-25 UTC) | `ingest/binance.py` |
| `open_time` 정렬 | `quality/checks.py` |
| 중복 `open_time` 없음 | `quality/checks.py` |
| 누락 candle 식별 | `quality/checks.py` |
| 미완성 candle 제외 | `ingest/binance.py` + `quality/checks.py` |
| OHLC 관계 | `quality/checks.py` **+ `ck_bar_labels_ohlc`** |
| 1m·5m 독립 계산 | 파이프라인 구조(`pipeline.py` τ별 독립) |
| 1m → 5m resampling 비교 (선택) | `quality/checks.py` (§6.3) |

**§11.2 segment 검증**

| 항목 | 강제 |
|---|---|
| turning point 시간순 정렬 | `direction/dc_engine.py` (emit 순서) |
| peak/trough 교대 | `direction/dc_engine.py` (상태기계) **+ `validation/report.py`** 사후 검증 |
| segment 비겹침 | `direction/segments.py` + `validation/report.py` (적재 전 거부) |
| 라벨링 가능 구간 덮음 | `direction/segments.py` |
| `N_j < min_segment_bars` → 방향성 금지 | `direction/segments.py` (§8.4 조건) |
| `A_j < theta_amp` → 방향성 금지 | `direction/segments.py` |
| `confirm_bar_j >= start_j` | `direction/dc_engine.py` **+ `ck_segment_labels_confirm`** (`is_tail_unconfirmed=false`일 때 lag_bars ≥ 0) |
| `end_j >= confirm_bar_j` 또는 확인 불가 별도 표시 | `is_tail_unconfirmed` 컬럼 **+ `ck_segment_labels_confirm`** (mutually exclusive) |

**§11.3 라벨 검증**

| 항목 | 강제 |
|---|---|
| bar당 단일 최종 라벨 | **`pk_bar_labels = (run_id, timeframe, open_time)`** |
| 9개 중 하나 | **`final_label` ENUM (정확히 9값)** |
| UP·DOWN 동시 부여 안 됨 | **`direction_label` ENUM** (단일값) + `ck_*_final_label_consistency` |
| LOW/MID/HIGH 단일 | **`volatility_label` ENUM** (단일값) |
| 1m·5m 분위수 혼용 안 됨 | **`uk_labeling_run_params_run_tf UNIQUE(run_id, timeframe)`** + 파이프라인 timeframe별 독립 계산 |
| `path_quality`, `jump_share_bv`, `downside_vol_share` 9라벨 비포함 | **`ck_*_final_label_consistency`**: `final_label`은 direction × volatility만 의존 |

**§11.4 수치 검증**

| 항목 | 강제 |
|---|---|
| `RV ≥ 0`, `RV_per_bar ≥ 0` | **`ck_segment_labels_ranges`** |
| `0 ≤ ER ≤ 1` | **`ck_segment_labels_ranges`** |
| `0 ≤ capturable_ratio ≤ 1` | **`ck_segment_labels_ranges`** |
| `0 ≤ max_jump_share ≤ 1` | **`ck_segment_labels_ranges`** |
| `0 ≤ downside_vol_share ≤ 1` | **`ck_segment_labels_ranges`** |
| `Q_low ≤ Q_high` | **`ck_labeling_run_params_quantiles`** |
| `theta_dc > 0`, `theta_amp > 0` | **`ck_labeling_run_params_theta_*`** |

---

## 15. Implementation Sequence — §16 매핑

`requirements.md §16`의 19단계 ↔ §5 모듈 1:1 매핑.

| # | §16 단계 | 모듈 / 산출 |
|---:|---|---|
| 1 | Binance ETHUSDT 1m monthly Kline zip 다운로드 | `ingest/binance.py` |
| 2 | Binance ETHUSDT 5m monthly Kline zip 다운로드 | `ingest/binance.py` |
| 3 | 기간 trim (2024-01-01 ~ 2026-05-25 UTC) | `ingest/binance.py` (trim + 미완성 제외) |
| 4 | 1분봉 데이터 품질 검증 | `quality/checks.py` (§4.3) |
| 5 | 5분봉 데이터 품질 검증 | `quality/checks.py` (§4.3) |
| 6 | (선택) 1m → 5m resampling 검산 | `quality/checks.py` (§6.3 공식) |
| 7 | timeframe별 hlc3 생성 | `transform/price.py` |
| 8 | timeframe별 로그 가격·returns 생성 | `transform/price.py` |
| 9 | timeframe별 DC threshold 계산 | `direction/dc_engine.py` (§8.1) |
| 10 | timeframe별 turning point·segment 생성 | `direction/dc_engine.py` (§8.2) + `direction/segments.py` (§8.3) |
| 11 | segment별 방향성 라벨 | `direction/segments.py` (§8.4) |
| 12 | segment별 realized volatility 계산 | `volatility/realized.py` (RV, RV_per_bar) |
| 13 | timeframe별 변동성 분위수 계산 | `volatility/realized.py` (33/66) |
| 14 | segment별 변동성 라벨 | `volatility/realized.py` |
| 15 | 최종 9라벨 생성 | `labeling/assemble.py` |
| 16 | lag·cost·jump·downside·ER-RV 보조 진단 | `diagnostics/{lag,cost,jump,asymmetry,er_corr}.py` (§10) |
| 17 | bar-level 라벨로 확장 | `labeling/assemble.py` (segment_id 상속) |
| 18 | 1m·5m 라벨 join | **DB VIEW `joined_labels_1m_5m`** (적재 후 조회) |
| 19 | 공식 검증 + PASS/FAIL 리포트 | `validation/report.py` → `labeling_reports` |

> 적재 순서: segment_labels → bar_labels (`fk_bar_labels_segment` 강제) → labeling_reports.
> run 완료 시 `labeling_runs.run_status='completed', completed_at=NOW()` 한 번에 전환.

---

## 16. Testing Strategy

| 레벨 | 내용 |
|---|---|
| unit | §10.4 Case A~E 기대 라벨(아래 표); DC 상태기계 turning point 교대성·lookahead 비참조 |
| property | §11 불변식(범위·겹침·교대·단일라벨)을 합성/소표본에 대해 검증 |
| integration | 소규모 slice(예: 1개월) → 실제 PG16(Docker) 마이그레이션·COPY 적재 → 재조회 round-trip, `joined_labels_1m_5m` 5분 버킷 정확성 |

### 16.1 §10.4 Synthetic Case A~E (단위테스트 명세)

`tests/test_synthetic_cases.py`. 각 케이스는 `hlc3` 가격열을 입력으로 받아 단일 segment의
`(direction, volatility, final_label)` + 진단값을 검증한다. theta/분위수는 케이스가
방향성·변동성을 결정적으로 만들도록 충분히 작게/크게 설정한다.

| Case | 입력 hlc3 시퀀스 | 기대 direction | 기대 volatility | 비고 |
|---|---|---|---|---|
| A | `100 → 101 → 102 → 103 → 104` | `UP` | `LOW_VOL` 또는 `MID_VOL` | 단조 상승, ER ≈ 1 |
| B | `104 → 103 → 102 → 101 → 100` | `DOWN` | `LOW_VOL` 또는 `MID_VOL` | 단조 하락, ER ≈ 1 |
| C | `100 → 100.05 → 99.98 → 100.02 → 100.01` | `NON_DIRECTIONAL` | `LOW_VOL` | 작은 횡보 |
| D | `100 → 106 → 99 → 105 → 98 → 101` | `NON_DIRECTIONAL` | `HIGH_VOL` | 큰 흔들림·약한 순방향. `max_jump_share` 또는 `downside_vol_share` 확인 |
| E | `100 → 103 → 101 → 106 → 104 → 110` | `UP` | `HIGH_VOL` 가능 | 큰 흔들림·강한 순상승. ER 낮음 → **ER을 방향성 필수조건에 넣지 않는 근거** |

**Case E 의미**: ER 낮은 거친 경로지만 순방향 이동이 충분 → `UP_HIGH_VOL` 분류. ER을 방향성 조건에
넣으면 `UP_HIGH_VOL` 구간이 과도하게 `NON_DIRECTIONAL`로 잘못 분류된다 (§6.4 주의 사항).

---

## 17. Open Questions / Calibration TODO — §18, §19

- `theta_dc_τ`, `min_segment_bars_τ`, 변동성 분위수(33/66 vs 25/75 vs 20/80): 2024-01~2026-05 분포로 캘리브레이션(§18).
- `hlc3` vs `close` 라벨 안정성 비교(§2.2, §18).
- jump 분리·비대칭 변동성의 ETHUSDT 유효성: 진단값으로만 저장 후 별도 검증(§7.3, §18).
- **진행중단 조건**(§19): 데이터 커버리지 미달·누락·timestamp 모호·고정 threshold·분위수 혼용·
  강제 UP/DOWN·검증 실패·9라벨 외 출력·전략손익 혼입·진단값의 라벨 변경 — 하나라도 해당 시 라벨 미확정.

---

## 18. 하네스 / 운영 노트

- 출력 DB는 쓰기 가능 `regime_benchmark` 전용. MCP `crypto_data`/`signal`/`wallet_db`(read-only)에 쓰지 않음.
- 데이터 조회 검증이 필요하면 `data-agent`(SELECT-only), 백테스트성 실행은 본 preset 범위 밖.
- 구현은 `quant-impl`(worktree 격리), 교차 리뷰는 `review-agent`(Codex/GPT-5.5), 정합성은 `mech-agent`.
