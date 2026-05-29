# ETH/USDT 1분봉·5분봉 사후 시장 방향성 × 변동성 9라벨 분류 요구사항서 v1.1

## 0. 문서 목적

본 문서는 Binance USDⓈ-M Futures의 ETH/USDT 과거 Kline 데이터를 사용하여 각 시점의 시장 상태를 **방향성 3개 × 변동성 3개 = 9개 라벨**로 분류하기 위한 구현 요구사항, 분류 공식, 공식 검증 기준을 정의한다.

이 문서의 산출물은 예측 모델의 성능을 비교하기 위한 **사후 benchmark label**이다. 실시간 매매 판단 로직이 아니다.

---

## 1. 핵심 원칙

| 구분 | 원칙 |
|---|---|
| 데이터 | Binance USDⓈ-M Futures ETHUSDT 완료 Kline 데이터 |
| 기간 | 2024-01-01 00:00:00 UTC ~ 2026-05-25 23:59:59 UTC |
| 대상 timeframe | 1분봉, 5분봉 |
| 목적 | 가격 경로의 사후 방향성·변동성 상태 분류 |
| 제외 | 전략 손익, 포지션 손익, 거래 성과, 수익성 판단 |
| 라벨 수 | 기본 라벨 9개 |
| 보조값 | 경로 효율성, 확인 지연, 점프 비중, 하방 변동성 비중, segment 거래가능성 진단값 |
| 구현 성격 | 문헌 기반 구성요소를 결합한 deterministic labeling specification |

---

## 2. 이론적 근거와 운영 정의의 분리

### 2.1 문헌 기반으로 사용할 수 있는 부분

| 구성요소 | 근거 성격 | 문헌 원문·확인 링크 | 본 문서에서 사용하는 범위 |
|---|---|---|---|
| Peak / trough 기반 방향성 구간 | 문헌 기반 | Bry & Boschan, *Programmed Selection of Cyclical Turning Points*, NBER, 1971. 원문 PDF: https://www.nber.org/system/files/chapters/c2148/c2148.pdf / NBER page: https://www.nber.org/books-and-chapters/cyclical-analysis-time-series-selected-procedures-and-computer-programs/programmed-selection-cyclical-turning-points | 가격 경로에서 turning point를 찾고 인접 turning point 사이를 phase로 보는 사고방식 |
| Bull / bear market dating | 문헌 기반 | Pagan & Sossounov, *A Simple Framework for Analysing Bull and Bear Markets*, Journal of Applied Econometrics, 2003. DOI page: https://onlinelibrary.wiley.com/doi/abs/10.1002/jae.664 / EconPapers: https://econpapers.repec.org/RePEc:jae:japmet:v:18:y:2003:i:1:p:23-46 | 저점→고점 구간을 상승 phase, 고점→저점 구간을 하락 phase로 해석하는 구조 |
| Harding-Pagan turning point dating | 문헌 기반 | Harding & Pagan, *Dissecting the Cycle: A Methodological Investigation*, Journal of Monetary Economics, 2002. DOI page: https://www.sciencedirect.com/science/article/pii/S0304393201001088 / bibliographic page: https://ideas.repec.org/a/eee/moneco/v49y2002i2p365-381.html | turning point event를 정의하고 cycle phase를 구분하는 방식 |
| Directional Change / Intrinsic Time | 문헌 기반 | Aloud et al., *A Directional-Change Event Approach for Studying Financial Time Series*, Economics Discussion Papers, 2012. Page: https://www.econstor.eu/handle/10419/65285 / Directional Change overview: https://arxiv.org/html/2406.07354v1 | 가격이 threshold 이상 반대 방향으로 움직일 때 event를 확정하는 방식 |
| Realized volatility | 문헌 기반 | Andersen, Bollerslev, Diebold & Labys, *Modeling and Forecasting Realized Volatility*, NBER Working Paper 8160, 2001. 원문 PDF: https://www.nber.org/system/files/working_papers/w8160/w8160.pdf / NBER page: https://www.nber.org/papers/w8160 | segment 내부 로그 가격 변화량 제곱합을 사용한 사후 변동성 측정 |
| Jump / continuous volatility 분리 | 문헌 기반 보조 진단 | Barndorff-Nielsen & Shephard, *Power and Bipower Variation with Stochastic Volatility and Jumps*, Journal of Financial Econometrics, 2004. Article page: https://ideas.repec.org/a/oup/jfinec/v2y2004i1p1-37.html / PDF: https://public.econ.duke.edu/~get/browse/courses/883/Spr16/COURSE-MATERIALS/Z_Papers/BNSJFEC2004.pdf | realized volatility가 단일 jump와 지속적 변동을 섞는 문제를 진단하는 보조 지표 |
| 비대칭 변동성 | 문헌 기반 보조 진단 | Bekaert & Wu, *Asymmetric Volatility and Risk in Equity Markets*, Review of Financial Studies, 2000. JSTOR: https://www.jstor.org/stable/2646079 / PDF: https://business.columbia.edu/sites/default/files-efs/citation_file_upload/asymmetric%20volatility.pdf | 상승 방향 가격 변화와 하락 방향 가격 변화의 변동성 기여도를 분리해 저장하는 보조 지표 |
| Binance Kline 데이터 | 공식 데이터 근거 | Binance public data GitHub: https://github.com/binance/binance-public-data / Binance Data: https://data.binance.vision/ / USDⓈ-M Futures Kline API: https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Kline-Candlestick-Data | ETHUSDT 1m·5m Kline 원천 데이터 다운로드 및 검증 |

#### 해석 원칙

- 위 문헌은 **9개 라벨 자체**를 표준 이론으로 제시하지 않는다.
- 본 문서의 9개 라벨은 다음 두 구성요소를 결합한 benchmark specification이다.

```text
방향성: turning point / directional-change 기반 phase labeling
변동성: realized volatility 기반 LOW/MID/HIGH labeling
```

---

### 2.2 운영 정의로 해결할 부분

아래 항목은 문헌에서 단일 최적값을 제공하지 않는다. 따라서 본 문서에서는 **초기 best 설정**을 제안하되, 데이터 검증 결과에 따라 변경할 수 있음을 명시한다.

| 항목 | best 제안 | 제안 이유 | 이유 검증 | 변경 기준 | 변경 방법 |
|---|---|---|---|---|---|
| 기준 가격 | `hlc3` | 고가·저가·종가를 함께 반영하여 단일 close보다 bar 내부 가격 범위 정보를 일부 반영한다. 1분봉·5분봉 모두 동일 공식 적용이 가능하다. | 올바른 이유다. 다만 `hlc3`는 체결 가능한 가격이 아니라 bar 요약값이므로 실거래 가격으로 해석하면 안 된다. 본 문서는 사후 라벨용이므로 허용된다. | `close` 대비 turning point 과다 생성 또는 과소 생성이 발생하는 경우 | `close`와 `hlc3` 라벨 일치율, segment 수, 평균 segment 길이를 비교한다. 라벨 일치율이 낮고 `hlc3`가 과도하게 noise를 줄이면 `close`로 대체 가능하다. |
| 방향성 threshold `theta_dc` | `Quantile(|d_t|, 0.80) × k`, 1m: `k ∈ {3,4,5}`, 5m: `k ∈ {2,3,4}` | 종목·timeframe별 가격 흔들림 규모를 반영한다. 고정 %보다 ETHUSDT 구간별 변동성 차이에 덜 취약하다. | 올바른 이유다. 다만 분위수 기반 threshold도 분석 기간의 변동성 분포에 의존한다. | segment 수가 과도하거나, 평균 segment 길이가 지나치게 짧거나 길 때 | 후보 grid별 segment 수, 평균/중앙 segment 길이, NON_DIRECTIONAL 비율, 라벨 안정성을 비교한다. |
| 최소 segment 길이 `min_segment_bars` | 1m: `{5,10,15}`, 5m: `{3,5,8}` | 1분봉은 micro noise가 많으므로 더 긴 최소 길이가 필요하다. 5분봉은 이미 집계된 데이터이므로 상대적으로 짧게 둔다. | 대체로 올바른 이유다. 단, 강한 급등락은 짧은 segment라도 의미가 있을 수 있으므로 길이 조건만으로 배제하면 안 된다. | 급격한 impulse move가 NON_DIRECTIONAL로 밀리는 경우 | `A_j >= theta_impulse`이면 길이 조건을 완화하는 예외 후보를 검토한다. 단, 기본 9라벨은 유지하고 `short_impulse_flag`로 먼저 기록한다. |
| 변동성 경계 | segment `RV_per_bar`의 33/66 분위수 | LOW/MID/HIGH 3분할을 균형 있게 만들기 쉽고, 라벨 imbalance를 줄인다. | 올바른 이유다. 다만 HIGH_VOL을 극단 변동성으로 정의하려면 66분위수는 낮을 수 있다. | HIGH_VOL 비율이 너무 높아 극단 구간 의미가 약할 때 | 25/75 또는 20/80 분위수 기준을 병행 계산하고 라벨 분포·지속시간·jump share를 비교한다. |
| NON_DIRECTIONAL 조건 | `N_j < min_segment_bars` 또는 `A_j < theta_amp` | 상승·하락으로 보기 어려운 segment를 강제로 UP/DOWN으로 배정하지 않는다. | 올바른 이유다. 단, deep pullback으로 인해 큰 parent trend 내부가 잘릴 수 있다. | NON_DIRECTIONAL segment가 동일 방향 큰 segment 사이에 자주 끼는 경우 | `pullback_within_parent_trend_flag`를 추가하고, parent segment는 별도 보조 컬럼으로 관리한다. 기본 9라벨은 변경하지 않는다. |
| 경로 효율성 `ER_j` | 최종 라벨 조건에서 제외, 진단값으로 저장 | `UP_HIGH_VOL`, `DOWN_HIGH_VOL`은 거칠지만 방향성이 있는 구간이므로 ER을 필수 조건으로 넣으면 오분류가 증가할 수 있다. | 올바른 이유다. ER은 방향성 자체보다 경로 품질에 가깝다. | ER과 volatility가 과도하게 중복되는 경우 | Spearman correlation을 산출한다. `|rho(ER, RV_per_bar)| >= 0.80`이면 ER은 모델 학습 feature가 아니라 진단 전용으로 제한한다. |
| 거래비용·슬리피지 | 라벨 생성에는 미사용, segment 진단값으로만 저장 | 이 문서는 가격 상태 정답지다. 비용을 라벨 조건에 넣으면 시장 상태와 전략 가능성이 섞인다. | 올바른 이유다. 다만 segment가 너무 촘촘하면 실전 추세 추종 benchmark로 오해될 수 있다. | `amplitude_to_cost_ratio`가 낮은 segment가 다수일 때 | `amplitude_to_cost_ratio = A_j / estimated_round_trip_cost_log`를 저장하고 낮은 segment를 `low_tradeability_segment_flag`로 표시한다. |

---

## 3. 최종 라벨 체계

### 3.1 방향성 라벨 3개

| 라벨 | 의미 |
|---|---|
| `UP` | 유효한 상승 방향 segment |
| `DOWN` | 유효한 하락 방향 segment |
| `NON_DIRECTIONAL` | 상승·하락 방향 조건을 충족하지 못한 segment |

### 3.2 변동성 라벨 3개

| 라벨 | 의미 |
|---|---|
| `LOW_VOL` | 해당 timeframe 기준 낮은 변동성 segment |
| `MID_VOL` | 해당 timeframe 기준 중간 변동성 segment |
| `HIGH_VOL` | 해당 timeframe 기준 높은 변동성 segment |

### 3.3 9개 최종 라벨

| 방향성 \ 변동성 | LOW_VOL | MID_VOL | HIGH_VOL |
|---|---|---|---|
| `UP` | `UP_LOW_VOL` | `UP_MID_VOL` | `UP_HIGH_VOL` |
| `DOWN` | `DOWN_LOW_VOL` | `DOWN_MID_VOL` | `DOWN_HIGH_VOL` |
| `NON_DIRECTIONAL` | `NON_DIRECTIONAL_LOW_VOL` | `NON_DIRECTIONAL_MID_VOL` | `NON_DIRECTIONAL_HIGH_VOL` |

---

## 4. 입력 데이터 요구사항

### 4.1 데이터 소스

| 항목 | 요구사항 |
|---|---|
| 거래소 | Binance |
| 시장 | USDⓈ-M Futures |
| 심볼 | `ETHUSDT` |
| 표시명 | ETH/USDT |
| 데이터 종류 | Kline / Candlestick |
| timeframe | `1m`, `5m` 각각 다운로드 |
| 기간 | 2024-01-01 00:00:00 UTC ~ 2026-05-25 23:59:59 UTC |
| 권장 다운로드 위치 | `https://data.binance.vision/?prefix=data/futures/um/monthly/klines/ETHUSDT/` |
| 1분봉 경로 | `data/futures/um/monthly/klines/ETHUSDT/1m/` |
| 5분봉 경로 | `data/futures/um/monthly/klines/ETHUSDT/5m/` |
| 대체 API | `GET /fapi/v1/klines`, 단 대량 다운로드는 public monthly zip 우선 |

### 4.2 필수 컬럼

Binance Kline 파일의 원천 컬럼은 구현 시 아래 표준 컬럼으로 매핑한다.

| 표준 컬럼 | Binance Kline 의미 |
|---|---|
| `open_time` | Kline open time |
| `open` | 시가 |
| `high` | 고가 |
| `low` | 저가 |
| `close` | 종가 |
| `volume` | 거래량 |
| `close_time` | Kline close time |
| `quote_asset_volume` | quote asset volume |
| `number_of_trades` | 거래 횟수 |
| `taker_buy_base_volume` | taker buy base asset volume |
| `taker_buy_quote_volume` | taker buy quote asset volume |
| `ignore` | 미사용 |

라벨링 기준 timestamp는 `open_time`으로 고정한다.

### 4.3 데이터 정합성 요구사항

| 요구사항 | 기준 |
|---|---|
| 중복 `open_time` 없음 | 동일 timeframe 내 중복 금지 |
| 누락 candle 식별 | 누락 구간은 라벨 계산 전 명시 |
| 미완성 candle 제외 | 다운로드 기간 종료 후 확정된 candle만 사용 |
| 가격 값 검증 | `high >= max(open, close)`, `low <= min(open, close)`, `high >= low` |
| timeframe 독립 처리 | 1분봉 라벨과 5분봉 라벨은 각각 다운로드한 데이터로 별도 계산 |
| 5분봉 검산 | 선택적으로 1분봉에서 5분봉을 resampling하여 Binance 5분봉과 비교 |

### 4.4 5분봉 검산용 resampling 규칙

5분봉은 Binance에서 직접 다운로드한 데이터를 사용한다. 아래 resampling은 검산 전용이다.

```text
open_5m_check   = 첫 번째 1분봉 open
high_5m_check   = 5개 1분봉 high의 max
low_5m_check    = 5개 1분봉 low의 min
close_5m_check  = 다섯 번째 1분봉 close
volume_5m_check = 5개 1분봉 volume의 sum
```

검산 결과가 Binance 5분봉과 반복적으로 불일치하면 timestamp 정렬, 누락 candle, 시간 구간 경계를 먼저 점검한다.

---

## 5. 기본 가격 변환 공식

각 timeframe `tau ∈ {1m, 5m}`에 대해 독립적으로 계산한다.

### 5.1 기준 가격

기준 가격은 `hlc3`로 고정한다.

```text
P_t = (high_t + low_t + close_t) / 3
```

[ASSUMPTION] `hlc3`는 사후 가격 상태 라벨링에는 적합한 기준 가격 후보이다.  
[RISK] `hlc3`는 체결 가능한 단일 가격이 아니므로 주문 체결 가격 또는 매매 성과 계산에는 사용하지 않는다.

### 5.2 로그 가격

```text
p_t = ln(P_t)
```

### 5.3 로그 가격 변화량

```text
d_t = p_t - p_{t-1}
```

---

## 6. 방향성 분류 공식

본 문서의 권장 방향성 분류는 **Directional Change 기반 swing segment 분할 후 segment별 방향성 부여** 방식이다.

### 6.1 Directional Change threshold

timeframe별 threshold는 고정값보다 데이터 기반 분위수로 정한다.

```text
abs_d_t = |p_t - p_{t-1}|
```

```text
theta_dc_tau = Quantile(abs_d_t, q_dc_tau) × k_dc_tau
```

권장 초기값:

| timeframe | `q_dc_tau` | `k_dc_tau` | 설명 |
|---|---:|---:|---|
| 1분봉 | 0.80 | 3.0 ~ 5.0 | 작은 흔들림 제거 |
| 5분봉 | 0.80 | 2.0 ~ 4.0 | 1분봉보다 완만한 threshold |

주의: 위 값은 확정값이 아니라 초기 후보 범위다. ETHUSDT 2024-01-01 ~ 2026-05-25 분포를 확인해 calibration한다.

---

### 6.2 Directional Change swing 확정 규칙

#### 상승 swing 후보

마지막 유효 저점 이후 가격이 다음 조건을 만족하면 상승 swing을 인정한다.

```text
p_t - p_trough >= theta_dc_tau
```

#### 하락 swing 후보

마지막 유효 고점 이후 가격이 다음 조건을 만족하면 하락 swing을 인정한다.

```text
p_peak - p_t >= theta_dc_tau
```

#### 고점 확정 규칙

상승 swing 진행 중 관측된 최고 로그 가격을 `p_peak`라고 할 때, 그 뒤 가격이 다음 조건을 만족하면 해당 `p_peak`를 swing high로 확정한다.

```text
p_peak - p_t >= theta_dc_tau
```

#### 저점 확정 규칙

하락 swing 진행 중 관측된 최저 로그 가격을 `p_trough`라고 할 때, 그 뒤 가격이 다음 조건을 만족하면 해당 `p_trough`를 swing low로 확정한다.

```text
p_t - p_trough >= theta_dc_tau
```

---

### 6.3 swing segment 정의

확정된 turning point를 시간순으로 나열한다.

```text
TP_1, TP_2, ..., TP_n
```

인접한 두 turning point 사이를 하나의 segment로 정의한다.

```text
S_j = [TP_j, TP_{j+1}]
```

각 segment에 대해 다음 값을 계산한다.

```text
start_j = TP_j
end_j   = TP_{j+1}
N_j     = end_j - start_j + 1
```

```text
M_j = p_{end_j} - p_{start_j}
```

```text
A_j = |M_j|
```

```text
L_j = Σ_{t=start_j+1}^{end_j} |p_t - p_{t-1}|
```

```text
ER_j = A_j / L_j
```

단, `L_j = 0`이면 `ER_j = 0`으로 둔다.

| 기호 | 의미 |
|---|---|
| `M_j` | segment의 순방향 가격 변화 |
| `A_j` | segment의 절대 이동폭 |
| `L_j` | segment 내부 경로 길이 |
| `ER_j` | 경로 효율성. 0~1 사이 값 |

---

### 6.4 방향성 라벨 부여

방향성 라벨은 segment 단위로 부여하고, segment 내부 모든 bar에 동일하게 할당한다.

#### 최소 조건

```text
N_j >= min_segment_bars_tau
```

```text
A_j >= theta_amp_tau
```

초기 후보:

| timeframe | `min_segment_bars_tau` | `theta_amp_tau` |
|---|---:|---:|
| 1분봉 | 5 ~ 15 | `theta_dc_tau` |
| 5분봉 | 3 ~ 8 | `theta_dc_tau` |

#### 방향성 공식

```text
direction_j =
  UP,   if M_j > 0  and N_j >= min_segment_bars_tau and A_j >= theta_amp_tau
  DOWN, if M_j < 0  and N_j >= min_segment_bars_tau and A_j >= theta_amp_tau
  NON_DIRECTIONAL, otherwise
```

주의: `ER_j`는 기본 방향성 판정의 필수 조건으로 사용하지 않는다.  
이유는 `UP_HIGH_VOL`, `DOWN_HIGH_VOL` 같은 라벨이 존재하므로, 경로가 거칠더라도 순방향 이동이 충분하면 방향성 segment로 인정해야 하기 때문이다.

다만 `ER_j`는 진단용으로 반드시 저장한다.

```text
path_quality_j =
  EFFICIENT, if ER_j >= theta_er_tau
  INEFFICIENT, otherwise
```

초기 후보:

```text
theta_er_tau = 0.35 ~ 0.50
```

`path_quality_j`는 9개 기본 라벨에 포함하지 않는다.

---

### 6.5 방향성 공식에 대한 보완: 지연 시간, 비용, deep pullback

이 절은 기본 9라벨 구조를 변경하지 않고, 방향성 라벨을 해석할 때 필요한 보조 진단값을 추가한다.

#### 6.5.1 추세 확인 지연 시간 진단

[RISK] 사후 segment는 turning point에서 시작하지만, 실전 추세 추종 로직은 turning point를 그 시점에 알 수 없다. 따라서 segment start와 확인 가능 시점 사이의 차이를 저장해야 한다.

```text
confirm_bar_j = segment 방향이 theta_dc_tau 이상 확인된 최초 bar
```

```text
lag_bars_j = confirm_bar_j - start_j
```

```text
lag_move_j = |p_{confirm_bar_j} - p_{start_j}|
```

```text
capturable_amplitude_j = max(A_j - lag_move_j, 0)
```

```text
capturable_ratio_j = capturable_amplitude_j / A_j
```

단, `A_j = 0`이면 `capturable_ratio_j = 0`으로 둔다.

해석:

| 값 | 의미 |
|---|---|
| `lag_bars_j` | 사후 segment 시작점과 확인 가능 지점의 bar 차이 |
| `lag_move_j` | 확인 전 이미 진행된 가격 이동 |
| `capturable_ratio_j` | 확인 이후 남아 있는 segment 이동 비율 |

[PASS/FAIL CRITERIA]

| 조건 | 판단 |
|---|---|
| `lag_bars_j >= 0` | PASS |
| `0 <= capturable_ratio_j <= 1` | PASS |
| `capturable_ratio_j`가 대부분 0에 가까움 | threshold가 너무 작거나 segment가 너무 짧을 가능성 |

#### 6.5.2 segment 거래가능성 진단

[RISK] segment 수가 너무 많고 각 segment의 진폭이 너무 작으면, 가격 경로상 방향은 존재하더라도 실전 추세 추종 대상으로는 부적합할 수 있다.

비용은 라벨 생성 조건에 넣지 않는다. 대신 다음 진단값을 저장한다.

```text
estimated_round_trip_cost_log = 2 × taker_fee_rate + slippage_rate_estimate
```

```text
amplitude_to_cost_ratio_j = A_j / estimated_round_trip_cost_log
```

초기 진단 기준:

```text
low_tradeability_segment_flag_j = true, if amplitude_to_cost_ratio_j < 3
```

해석:

| 항목 | 설명 |
|---|---|
| `amplitude_to_cost_ratio_j < 1` | 가격 이동폭이 왕복 비용보다 작음 |
| `1 <= amplitude_to_cost_ratio_j < 3` | 비용 대비 여유가 작음 |
| `amplitude_to_cost_ratio_j >= 3` | segment 진폭만 보면 비용 대비 여유가 있음 |

주의: 이 값은 전략 성과 판단이 아니라, 라벨 segment가 너무 미세하게 쪼개졌는지 확인하는 진단값이다.

#### 6.5.3 deep pullback에 의한 NON_DIRECTIONAL 오인 진단

[RISK] 큰 추세 내부의 짧고 깊은 되돌림이 segment를 분할하면, parent trend 관점에서는 추세 지속인데 일부 구간이 `NON_DIRECTIONAL`로 표시될 수 있다.

기본 라벨은 유지한다. 대신 다음 flag를 추가한다.

```text
pullback_within_parent_trend_flag_j = true
```

조건 후보:

```text
segment j가 NON_DIRECTIONAL이고,
segment j-1과 segment j+1의 direction이 같고,
A_{j-1} + A_{j+1} >= theta_parent_amp_tau이고,
N_j <= pullback_max_bars_tau
```

초기 후보:

| timeframe | `pullback_max_bars_tau` | `theta_parent_amp_tau` |
|---|---:|---:|
| 1분봉 | 10 ~ 30 | `2 × theta_dc_tau` |
| 5분봉 | 3 ~ 10 | `2 × theta_dc_tau` |

[INFERENCE] 이 보완은 기존 9라벨을 변경하지 않으면서, `NON_DIRECTIONAL` 라벨 중 parent trend 내부 pullback 후보를 분리 검토할 수 있게 한다.

---

## 7. 변동성 분류 공식

### 7.1 segment realized volatility

각 segment `S_j`에 대해 로그 가격 변화량의 제곱합으로 변동성을 계산한다.

```text
RV_j = sqrt(Σ_{t=start_j+1}^{end_j} d_t^2)
```

segment 길이가 다르면 길이 보정값도 함께 저장한다.

```text
RV_per_bar_j = RV_j / sqrt(N_j)
```

권장 기본값:

```text
vol_score_j = RV_per_bar_j
```

이유: segment 길이가 서로 다르면 단순 `RV_j`는 긴 segment에 유리하게 커질 수 있다. 1분봉·5분봉 모두 `RV_per_bar_j`를 기본 변동성 점수로 사용한다.

---

### 7.2 timeframe별 변동성 분위수 계산

각 timeframe `tau`에 대해 segment별 `vol_score_j` 분포를 만든다.

```text
Q_low_tau  = Quantile(vol_score_j, 0.33)
Q_high_tau = Quantile(vol_score_j, 0.66)
```

초기 기본값은 33/66 분위수다.

#### 변동성 라벨 공식

```text
volatility_j =
  LOW_VOL,  if vol_score_j <= Q_low_tau
  MID_VOL,  if Q_low_tau < vol_score_j <= Q_high_tau
  HIGH_VOL, if vol_score_j > Q_high_tau
```

주의: 1분봉의 분위수와 5분봉의 분위수는 각각 별도로 계산한다.  
1분봉 변동성 경계값을 5분봉에 적용하지 않는다.

---

### 7.3 변동성 공식에 대한 보완: jump, ER 중복, 비대칭 변동성

이 절은 기본 변동성 라벨을 변경하지 않고, `HIGH_VOL`의 성격을 해석하기 위한 보조 진단값을 추가한다.

#### 7.3.1 Jump risk와 continuous volatility 분리 진단

[RISK] `RV_j`는 segment 전체의 제곱 변화를 합산하므로 단일 대형 bar와 지속적 지그재그 변동을 모두 `HIGH_VOL`로 분류할 수 있다.

기본 라벨은 `RV_per_bar_j`로 유지한다. 대신 다음 보조값을 저장한다.

```text
max_abs_d_j = max(|d_t|), t ∈ S_j
```

```text
max_jump_share_j = max(d_t^2) / Σ d_t^2
```

단, `Σ d_t^2 = 0`이면 `max_jump_share_j = 0`으로 둔다.

보조적으로 bipower variation을 계산할 수 있다.

```text
BV_j = (π / 2) × Σ_{t=start_j+2}^{end_j} |d_t| × |d_{t-1}|
```

```text
jump_component_j = max(RV_j^2 - BV_j, 0)
```

```text
jump_share_bv_j = jump_component_j / RV_j^2
```

단, `RV_j = 0`이면 `jump_share_bv_j = 0`으로 둔다.

해석:

| 조건 | 해석 |
|---|---|
| `max_jump_share_j` 높음 | segment 변동성이 소수 bar에 집중됨 |
| `jump_share_bv_j` 높음 | continuous volatility보다 jump component 비중이 클 가능성 |
| `RV_per_bar_j` 높고 `jump_share_bv_j` 낮음 | 지속적 고변동 구간 가능성 |

#### 7.3.2 ER과 변동성 정보 중복 진단

[RISK] `ER_j = A_j / L_j`와 `RV_per_bar_j`는 둘 다 경로의 거칠기와 관련될 수 있다. 따라서 강한 음의 상관이 발생할 수 있다.

기본 대응:

```text
ER_j는 최종 9라벨에 포함하지 않는다.
```

검증 공식:

```text
rho_er_vol_tau = SpearmanCorr(ER_j, RV_per_bar_j) within timeframe tau
```

판정 기준:

| 조건 | 판단 |
|---|---|
| `|rho_er_vol_tau| < 0.60` | 중복 낮음 |
| `0.60 <= |rho_er_vol_tau| < 0.80` | 중복 가능성 있음 |
| `|rho_er_vol_tau| >= 0.80` | 중복 강함. ER은 진단값으로만 유지 |

주의: ER은 기본 라벨 축이 아니므로, 상관이 높아도 9라벨 자체의 논리 오류는 아니다.

#### 7.3.3 비대칭 변동성 진단

[RISK] `d_t^2`는 부호를 제거하므로 상승 방향 변동성과 하락 방향 변동성을 구분하지 못한다.

기본 라벨은 유지하되, 다음 보조값을 저장한다.

```text
RV_plus_j = sqrt(Σ d_t^2 × I(d_t > 0))
```

```text
RV_minus_j = sqrt(Σ d_t^2 × I(d_t < 0))
```

```text
downside_vol_share_j = Σ d_t^2 × I(d_t < 0) / Σ d_t^2
```

단, `Σ d_t^2 = 0`이면 `downside_vol_share_j = 0`으로 둔다.

해석:

| 값 | 의미 |
|---|---|
| `downside_vol_share_j > 0.5` | 하락 방향 가격 변화가 변동성의 절반 이상을 차지 |
| `downside_vol_share_j < 0.5` | 상승 방향 가격 변화가 변동성의 절반 이상을 차지 |
| `RV_minus_j >> RV_plus_j` | 패닉성 하락 변동성 가능성 |

[BOUNDARY] 주식시장의 leverage effect 문헌을 ETH/USDT에 그대로 적용하면 안 된다. 본 문서에서는 비대칭 변동성을 **진단값**으로만 저장하고, ETHUSDT에서 실제로 의미가 있는지는 별도 검증한다.

---

## 8. 최종 9라벨 생성 공식

```text
final_label_j = direction_j + "_" + volatility_j
```

단, `direction_j = NON_DIRECTIONAL`이면 다음 형식을 사용한다.

```text
NON_DIRECTIONAL_LOW_VOL
NON_DIRECTIONAL_MID_VOL
NON_DIRECTIONAL_HIGH_VOL
```

각 bar의 라벨은 해당 bar가 속한 segment의 `final_label_j`를 상속한다.

```text
label_t = final_label_j, where t ∈ S_j
```

---

## 9. 1분봉·5분봉 동시 처리 요구사항

### 9.1 독립 라벨링

1분봉과 5분봉은 각각 독립된 라벨 체계를 가진다.

```text
label_1m_t = Labeler(data_1m)
label_5m_t = Labeler(data_5m)
```

### 9.2 5분봉 라벨의 1분봉 매핑

예측 모델 학습 또는 비교를 위해 5분봉 라벨을 1분 단위 테이블에 붙일 수 있다.

```text
label_5m_for_1m_t = label_5m_k
where t ∈ five_minute_bucket_k
```

주의: 이는 join 편의를 위한 매핑이다.  
5분봉 라벨 자체는 5분봉 데이터에서 계산되어야 한다.

### 9.3 timeframe 간 라벨 불일치 처리

동일 시각에 다음과 같은 조합이 나올 수 있다.

| 1분봉 라벨 | 5분봉 라벨 | 해석 |
|---|---|---|
| `UP_HIGH_VOL` | `NON_DIRECTIONAL_MID_VOL` | 1분 단기 상승 흔들림이 있으나 5분 기준 방향성은 약함 |
| `NON_DIRECTIONAL_HIGH_VOL` | `DOWN_MID_VOL` | 1분 내부 혼조가 있지만 5분 기준 하락 segment |
| `DOWN_LOW_VOL` | `DOWN_HIGH_VOL` | 방향은 같으나 변동성 강도가 timeframe별로 다름 |

이 불일치는 오류가 아니다. timeframe별 시장 구조가 다르게 보이는 정상 현상이다.

---

## 10. 공식 검증

### 10.1 경로 효율성 `ER_j` 검증

정의:

```text
ER_j = |p_end - p_start| / Σ |p_t - p_{t-1}|
```

삼각부등식에 의해 다음이 성립한다.

```text
0 <= ER_j <= 1
```

검증 조건:

| 조건 | 기대값 |
|---|---|
| 완전 단조 상승 | `ER_j = 1` |
| 완전 단조 하락 | `ER_j = 1` |
| 시작점과 종료점이 같고 중간 변동이 큼 | `ER_j = 0`에 가까움 |
| 가격 변화 없음 | `L_j = 0`, `ER_j = 0` |

---

### 10.2 realized volatility 검증

정의:

```text
RV_j = sqrt(Σ d_t^2)
```

검증 조건:

| 조건 | 기대값 |
|---|---|
| 모든 `d_t = 0` | `RV_j = 0` |
| 가격 변화량 절대값이 커짐 | `RV_j` 증가 |
| segment 길이가 길어짐 | `RV_j`는 증가 가능, `RV_per_bar_j`로 길이 보정 |
| 음수 변동성 | 발생 불가 |

---

### 10.3 보조 진단 공식 검증

| 지표 | 정상 범위 | 검증 기준 |
|---|---:|---|
| `lag_bars_j` | `>= 0` | 음수이면 구현 오류 |
| `capturable_ratio_j` | `0 ~ 1` | 범위 밖이면 구현 오류 |
| `amplitude_to_cost_ratio_j` | `>= 0` | 음수이면 구현 오류 |
| `max_jump_share_j` | `0 ~ 1` | 범위 밖이면 구현 오류 |
| `jump_share_bv_j` | `0 ~ 1` 권장 | 1 초과가 반복되면 BV 계산 또는 표본 길이 점검 |
| `downside_vol_share_j` | `0 ~ 1` | 범위 밖이면 구현 오류 |
| `rho_er_vol_tau` | `-1 ~ 1` | 범위 밖이면 상관 계산 오류 |

---

### 10.4 방향성 라벨 검증

#### Case A. 명확한 상승

```text
가격: 100 → 101 → 102 → 103 → 104
```

기대 결과:

```text
direction = UP
volatility = LOW_VOL 또는 MID_VOL
```

#### Case B. 명확한 하락

```text
가격: 104 → 103 → 102 → 101 → 100
```

기대 결과:

```text
direction = DOWN
volatility = LOW_VOL 또는 MID_VOL
```

#### Case C. 작은 횡보

```text
가격: 100 → 100.05 → 99.98 → 100.02 → 100.01
```

기대 결과:

```text
direction = NON_DIRECTIONAL
volatility = LOW_VOL
```

#### Case D. 큰 흔들림이 있지만 순방향 이동이 약함

```text
가격: 100 → 106 → 99 → 105 → 98 → 101
```

기대 결과:

```text
direction = NON_DIRECTIONAL
volatility = HIGH_VOL
max_jump_share_j 또는 downside_vol_share_j 확인 필요
```

#### Case E. 큰 흔들림이 있으나 순상승이 명확함

```text
가격: 100 → 103 → 101 → 106 → 104 → 110
```

기대 결과:

```text
direction = UP
volatility = HIGH_VOL 가능
ER_j = 낮을 수 있음
```

이 Case E 때문에 `ER_j`를 기본 방향성 조건에 넣지 않는다.  
경로 효율성을 방향성 조건에 넣으면 `UP_HIGH_VOL` 구간이 과도하게 `NON_DIRECTIONAL`로 분류될 수 있다.

---

## 11. 구현 검증 체크리스트

### 11.1 데이터 검증

- [ ] Binance USDⓈ-M Futures ETHUSDT 1m 데이터를 다운로드했다.
- [ ] Binance USDⓈ-M Futures ETHUSDT 5m 데이터를 다운로드했다.
- [ ] 기간은 2024-01-01 00:00:00 UTC ~ 2026-05-25 23:59:59 UTC로 trim했다.
- [ ] `open_time` 기준 timestamp 정렬 완료
- [ ] 중복 `open_time` 없음
- [ ] 누락 candle 식별
- [ ] 미완성 candle 제외
- [ ] OHLC 관계 검증 완료
- [ ] 1분봉과 5분봉을 독립 계산
- [ ] 선택적으로 1분봉 resampling 5분봉과 Binance 5분봉 비교 완료

### 11.2 segment 검증

- [ ] turning point가 시간순으로 정렬됨
- [ ] peak와 trough가 교대로 등장
- [ ] segment가 서로 겹치지 않음
- [ ] segment가 전체 라벨링 가능 구간을 덮음
- [ ] segment 길이가 `min_segment_bars_tau`보다 작으면 방향성 인정 금지
- [ ] `A_j < theta_amp_tau`이면 방향성 인정 금지
- [ ] `confirm_bar_j >= start_j`
- [ ] `end_j >= confirm_bar_j` 또는 확인 불가 segment로 별도 표시

### 11.3 라벨 검증

- [ ] 모든 bar는 하나의 최종 라벨만 가짐
- [ ] 최종 라벨은 9개 중 하나여야 함
- [ ] `UP`과 `DOWN`이 동시에 부여되지 않음
- [ ] `LOW_VOL`, `MID_VOL`, `HIGH_VOL` 중 하나만 부여됨
- [ ] 1분봉 분위수와 5분봉 분위수를 혼용하지 않음
- [ ] `path_quality_j`, `jump_share_bv_j`, `downside_vol_share_j`는 보조값이며 최종 9라벨에는 포함하지 않음

### 11.4 수치 검증

- [ ] 모든 `RV_j >= 0`
- [ ] 모든 `RV_per_bar_j >= 0`
- [ ] 모든 `0 <= ER_j <= 1`
- [ ] 모든 `0 <= capturable_ratio_j <= 1`
- [ ] 모든 `0 <= max_jump_share_j <= 1`
- [ ] 모든 `0 <= downside_vol_share_j <= 1`
- [ ] `Q_low_tau <= Q_high_tau`
- [ ] `theta_dc_tau > 0`
- [ ] `theta_amp_tau > 0`

---

## 12. 부적합한 방식

| 방식 | 판단 |
|---|---|
| 이동평균 정배열/역배열만으로 9라벨 생성 | 부적합 |
| RSI, MACD 단독 사용 | 부적합 |
| HMM, K-means 결과를 정답지로 직접 사용 | 부적합 |
| 전략 손익으로 방향성 라벨 생성 | 부적합 |
| 1분봉 threshold를 5분봉에 그대로 적용 | 부적합 |
| `ER_j`만으로 방향성 결정 | 부적합 |
| 비용을 최종 시장 상태 라벨 조건에 직접 포함 | 부적합 |
| jump 진단값만으로 LOW/MID/HIGH_VOL을 대체 | 부적합 |

---

## 13. 출력 스키마

### 13.1 segment-level output

```yaml
segment_output:
  symbol: "ETHUSDT"
  market: "BINANCE_USDM_FUTURES"
  timeframe: "1m | 5m"
  segment_id: string
  start_timestamp: datetime
  end_timestamp: datetime
  confirm_timestamp: datetime
  lag_bars: integer
  lag_move: float
  capturable_amplitude: float
  capturable_ratio: float
  start_price_hlc3: float
  end_price_hlc3: float
  log_move: float
  amplitude: float
  path_length: float
  efficiency_ratio: float
  realized_volatility: float
  realized_volatility_per_bar: float
  max_abs_d: float
  max_jump_share: float
  bipower_variation: float
  jump_component: float
  jump_share_bv: float
  rv_plus: float
  rv_minus: float
  downside_vol_share: float
  amplitude_to_cost_ratio: float
  low_tradeability_segment_flag: boolean
  pullback_within_parent_trend_flag: boolean
  direction_label: "UP | DOWN | NON_DIRECTIONAL"
  volatility_label: "LOW_VOL | MID_VOL | HIGH_VOL"
  final_label: string
  segment_bar_count: integer
  method_version: string
```

### 13.2 bar-level output

```yaml
bar_output:
  symbol: "ETHUSDT"
  market: "BINANCE_USDM_FUTURES"
  timeframe: "1m | 5m"
  open_time: datetime
  open: float
  high: float
  low: float
  close: float
  hlc3: float
  segment_id: string
  direction_label: "UP | DOWN | NON_DIRECTIONAL"
  volatility_label: "LOW_VOL | MID_VOL | HIGH_VOL"
  final_label: string
  method_version: string
```

### 13.3 1분봉 테이블에 5분봉 라벨을 함께 붙인 출력

```yaml
joined_output_1m:
  symbol: "ETHUSDT"
  market: "BINANCE_USDM_FUTURES"
  open_time_1m: datetime
  label_1m: string
  segment_id_1m: string
  open_time_5m_bucket: datetime
  label_5m: string
  segment_id_5m: string
```

### 13.4 PostgreSQL 영속화 대상

13.1·13.2·13.3의 출력은 추후 예측 모델 학습·비교에 재사용할 수 있도록 PostgreSQL에 영속화한다.
저장은 **별도의 쓰기 가능한 전용 DB** (`regime_benchmark`, 예: 로컬 Docker `postgis/postgis:16-3.4`)에서 수행하며,
하네스의 읽기 전용 MCP DB (`crypto_data`, `signal`, `wallet_db`)에는 쓰지 않는다.

DSN은 환경변수 `${REGIME_BENCHMARK_DB_URL}`로 주입하고, 스키마는 `migrations/001_init.sql`을 통해 생성한다.
본 DSN은 **`regime_owner` 사용자**로 접속하며, `regime_owner` role과 `regime_benchmark` DB는
`migrations/000_init_roles.sql` + `scripts/bootstrap_db.sh`로 일회 부트스트랩한다 (자세한 절차는
`docs/design.md` §13.8 참조).

| 항목 | 요구사항 |
|---|---|
| 엔진 | PostgreSQL 16+ (네이티브 declarative 파티셔닝) |
| 라벨 수치 타입 | `DOUBLE PRECISION` (float64; hlc3는 체결가 아님) |
| 라벨 축 타입 | PostgreSQL ENUM (`direction_label`, `volatility_label`, `final_label`, `timeframe_enum`) |
| 스키마 구조 | 통합 테이블 `segment_labels`, `bar_labels` + `timeframe` 컬럼. timeframe별 분리 테이블은 사용하지 않는다 |
| `bar_labels` 파티셔닝 | `open_time` 기준 월 단위 RANGE 파티션 (2024-01 ~ 2026-06, 30개월) |
| 재현성 단위 | `labeling_runs(id, method_version, period, run_status, completed_at, ...)` + timeframe별 `labeling_run_params`. 1m·5m가 **동일 run_id**에 속해야 join 일관성이 보장된다 (§9.1 timeframe 독립 계산은 유지 — run은 운영 단위 grouping이며 분위수·threshold는 timeframe별 독립) |
| 버전 관리 | append-only. 동일 (`method_version`, period) 재실행 시 새 `run_id` 생성, 과거 라벨 보존 |
| Run 상태 | `run_status ∈ {loading, completed, failed}` + `completed_at`. 소비자는 `WHERE run_status='completed'` 로 부분/실패 run 배제 |
| 적재 방식 | `psycopg3 COPY ... FROM STDIN` (1m bar 약 126만 행, 대량 적재 최적화) |
| 13.1/13.2/13.3 매핑 | `segment_labels` / `bar_labels` (월 파티션) / VIEW `joined_labels_1m_5m` |
| `_1m` / `_5m` 노출 | VIEW `bar_labels_1m`, `bar_labels_5m`, `segment_labels_1m`, `segment_labels_5m` (timeframe 컬럼 필터) |
| 검증/진단 리포트 | `labeling_reports(run_id, report_type, passed, payload JSONB)` |

5분 버킷 매핑 (`joined_labels_1m_5m` view, **LEFT JOIN — additive 매핑**, 5m 부분 적재 시 `label_5m`이 NULL로 노출):

```sql
SELECT ... FROM bar_labels b1
LEFT JOIN bar_labels b5
  ON b1.run_id = b5.run_id AND b5.timeframe='5m'
 AND extract(epoch FROM b5.open_time) = floor(extract(epoch FROM b1.open_time)/300)*300
WHERE b1.timeframe='1m';
```

PASS/FAIL 정합성 (DB 레벨 구조적 강제):
- bar 당 단일 9라벨 (`pk_bar_labels = (run_id, timeframe, open_time)`).
- segment 비겹침·교대성은 적재 전 검증(`validation/`)에서 확인하고 위반 시 적재하지 않는다.
- **`ck_*_final_label_consistency`**: `final_label = direction_label || '_' || volatility_label` 을 segment·bar 양 테이블에서 CHECK로 강제 → §19-10 "진단값이 최종 9라벨을 변경" 실패 모드를 **구조적으로 차단**.
- `ck_segment_labels_ranges`: §11.4 범위 검증 (`0≤ER≤1`, `0≤capturable_ratio≤1`, `0≤max_jump_share≤1`, `0≤downside_vol_share≤1`, RV·jump·rv±·amplitude_to_cost_ratio 등 비음수, `segment_bar_count≥1`, `end≥start`).
- `ck_segment_labels_confirm`: 꼬리 미확정(`is_tail_unconfirmed=true`)을 제외하면 (`confirm_timestamp`, `lag_bars`, `lag_move`, `capturable_amplitude`, `capturable_ratio`) 전부 NOT NULL.
- `ck_bar_labels_ohlc`로 §4.3 OHLC 관계를 DB 레벨에서 강제한다.
- `symbol`, `market`, `method_version`은 §13.1·§13.2 출력 스키마 명시 필드이므로 두 테이블에 **denormalize NOT NULL** 컬럼으로 보유한다.

> 자세한 DDL은 [`/migrations/001_init.sql`](../migrations/001_init.sql), 모듈 설계는 [`/docs/design.md`](./design.md) §13 참조.

---

## 14. 권장 설정 초기값

아래 값은 확정값이 아니라 실험 시작점이다.

```yaml
labeling_config:
  method_version: "regime_label_9axis_v1.1"

  data:
    exchange: "Binance"
    market: "USDⓈ-M Futures"
    symbol: "ETHUSDT"
    display_pair: "ETH/USDT"
    start_utc: "2024-01-01 00:00:00"
    end_utc: "2026-05-25 23:59:59"
    source_1m: "data/futures/um/monthly/klines/ETHUSDT/1m/"
    source_5m: "data/futures/um/monthly/klines/ETHUSDT/5m/"

  price_field: "hlc3"
  price_formula: "(high + low + close) / 3"

  timeframes:
    - "1m"
    - "5m"

  direction_method:
    type: "directional_change_segment"
    threshold_policy: "abs_log_price_change_quantile_scaled"

    params:
      "1m":
        q_dc: 0.80
        k_dc_candidates: [3.0, 4.0, 5.0]
        min_segment_bars_candidates: [5, 10, 15]
        theta_amp_policy: "same_as_theta_dc"
        pullback_max_bars_candidates: [10, 20, 30]
      "5m":
        q_dc: 0.80
        k_dc_candidates: [2.0, 3.0, 4.0]
        min_segment_bars_candidates: [3, 5, 8]
        theta_amp_policy: "same_as_theta_dc"
        pullback_max_bars_candidates: [3, 5, 10]

  volatility_method:
    type: "segment_realized_volatility_per_bar"
    formula: "sqrt(sum(d_t^2)) / sqrt(N_j)"
    quantiles:
      low: 0.33
      high: 0.66

  auxiliary_metrics:
    efficiency_ratio:
      enabled: true
      formula: "abs(p_end - p_start) / sum(abs(d_t))"
      not_part_of_final_9_labels: true
    lag_diagnostics:
      enabled: true
      metrics: ["confirm_bar", "lag_bars", "lag_move", "capturable_ratio"]
    tradeability_diagnostics:
      enabled: true
      metrics: ["amplitude_to_cost_ratio", "low_tradeability_segment_flag"]
      not_part_of_final_9_labels: true
    jump_diagnostics:
      enabled: true
      metrics: ["max_jump_share", "bipower_variation", "jump_share_bv"]
      not_part_of_final_9_labels: true
    asymmetric_volatility:
      enabled: true
      metrics: ["rv_plus", "rv_minus", "downside_vol_share"]
      not_part_of_final_9_labels: true

  final_labels:
    - "UP_LOW_VOL"
    - "UP_MID_VOL"
    - "UP_HIGH_VOL"
    - "DOWN_LOW_VOL"
    - "DOWN_MID_VOL"
    - "DOWN_HIGH_VOL"
    - "NON_DIRECTIONAL_LOW_VOL"
    - "NON_DIRECTIONAL_MID_VOL"
    - "NON_DIRECTIONAL_HIGH_VOL"

  persistence:
    engine: "postgresql"
    version: ">=16"
    database: "regime_benchmark"
    dsn_env: "REGIME_BENCHMARK_DB_URL"
    migration_file: "migrations/001_init.sql"
    write_target: "dedicated_writable_db_only"
    read_only_mcp_dbs_excluded: true        # crypto_data / signal / wallet_db 미사용
    bulk_load: "psycopg3_copy_from_stdin"
    versioning: "append_only_per_run_id"
    bar_labels:
      partition_strategy: "range_monthly"
      partition_column: "open_time"
      partition_range:
        start: "2024-01"
        end:   "2026-06"
    enums:
      timeframe:   ["1m", "5m"]
      direction:   ["UP", "DOWN", "NON_DIRECTIONAL"]
      volatility:  ["LOW_VOL", "MID_VOL", "HIGH_VOL"]
      final_label_count: 9
    numeric_type: "double_precision"        # float64; hlc3는 체결가 아님 (§5.1)
    views:
      - "bar_labels_1m"
      - "bar_labels_5m"
      - "segment_labels_1m"
      - "segment_labels_5m"
      - "joined_labels_1m_5m"
```

---

## 15. PASS / FAIL 기준

### 15.1 PASS 기준

| 항목 | 기준 |
|---|---|
| 재현성 | 동일 입력 + 동일 설정 → 동일 라벨 |
| 라벨 완전성 | 모든 유효 bar에 9개 중 하나의 라벨 부여 |
| 수치 안정성 | `RV >= 0`, `0 <= ER <= 1`, `0 <= downside_vol_share <= 1` |
| timeframe 독립성 | 1분봉과 5분봉 threshold·분위수 별도 계산 |
| segment 정합성 | segment 겹침 없음, turning point 교대성 유지 |
| 문헌 대응성 | 방향성은 DC/turning point, 변동성은 realized volatility 기반 |
| 전략 손익 배제 | 포지션 성과 또는 거래 성과 미사용 |
| 보조 진단 분리 | lag, cost, jump, downside volatility는 최종 9라벨 조건에 직접 미포함 |

### 15.2 FAIL 기준

| 조건 | 처리 |
|---|---|
| 누락 candle이 많아 segment가 왜곡됨 | 라벨링 중단 또는 구간 제외 |
| 5분봉 데이터가 직접 다운로드되지 않음 | 요구사항 미충족 |
| `Q_low > Q_high` | 계산 오류 |
| `ER < 0` 또는 `ER > 1` | 계산 오류 |
| 9개 외 라벨 생성 | 구현 오류 |
| 1분봉 분위수를 5분봉에 적용 | 구현 오류 |
| 전략 손익을 라벨 생성에 사용 | 목적 위반 |
| lag, cost, jump 진단값이 최종 라벨을 변경 | 스키마 위반 |

---

## 16. 구현 순서

```text
1. Binance ETHUSDT 1m monthly Kline zip 다운로드
2. Binance ETHUSDT 5m monthly Kline zip 다운로드
3. 기간을 2024-01-01 00:00:00 UTC ~ 2026-05-25 23:59:59 UTC로 trim
4. 1분봉 데이터 품질 검증
5. 5분봉 데이터 품질 검증
6. 선택적으로 1분봉 resampling 5분봉과 Binance 5분봉 비교
7. timeframe별 hlc3 생성
8. timeframe별 로그 가격 생성
9. timeframe별 Directional Change threshold 계산
10. timeframe별 turning point 및 segment 생성
11. segment별 방향성 라벨 계산
12. segment별 realized volatility 계산
13. timeframe별 변동성 분위수 계산
14. segment별 변동성 라벨 계산
15. 최종 9라벨 생성
16. lag, cost, jump, downside volatility 보조 진단값 생성
17. bar-level 라벨로 확장
18. 1분봉·5분봉 라벨 join 테이블 생성
19. 공식 검증 및 PASS/FAIL 리포트 생성
```

---

## 17. 핵심 산출물

| 산출물 | 설명 | PostgreSQL 물리 대상 (§13.4) |
|---|---|---|
| `segment_labels_1m` | 1분봉 segment-level 라벨 | VIEW `segment_labels_1m` (← `segment_labels` where `timeframe='1m'`) |
| `bar_labels_1m` | 1분봉 bar-level 라벨 | VIEW `bar_labels_1m` (← 월 파티션 테이블 `bar_labels` where `timeframe='1m'`) |
| `segment_labels_5m` | 5분봉 segment-level 라벨 | VIEW `segment_labels_5m` (← `segment_labels` where `timeframe='5m'`) |
| `bar_labels_5m` | 5분봉 bar-level 라벨 | VIEW `bar_labels_5m` (← `bar_labels` where `timeframe='5m'`) |
| `joined_labels_1m_5m` | 1분 단위 기준으로 1분봉·5분봉 라벨을 함께 제공 | VIEW `joined_labels_1m_5m` (동일 `run_id` 내 5분 floor 버킷 조인) |
| `labeling_validation_report` | 공식 검증, 데이터 검증, PASS/FAIL 결과 | `labeling_reports` (`report_type='validation'`, `passed BOOLEAN`, `payload JSONB`) |
| `diagnostic_report` | lag, tradeability, jump, downside volatility, ER-RV correlation 진단 결과 | `labeling_reports` (`report_type='diagnostic'`, `payload JSONB`) |
| (재현성) | 라벨 run + 파라미터 | `labeling_runs`, `labeling_run_params` |

---

## 18. 남아 있는 불확실성

| 항목 | 불확실성 |
|---|---|
| `theta_dc_tau` | ETHUSDT 2024-01-01 ~ 2026-05-25 분포 기반 calibration 필요 |
| `min_segment_bars_tau` | 너무 짧으면 흔들림 증가, 너무 길면 세부 segment 누락 |
| 변동성 분위수 | 33/66이 기본값이지만 목적에 따라 25/75 또는 20/80 가능 |
| `hlc3` | close 대비 라벨 안정성이 더 좋은지 검증 필요 |
| NON_DIRECTIONAL 비율 | threshold 설정에 크게 영향받음 |
| jump 진단 | 1분봉에서도 단일 bar jump와 연속 변동을 완벽히 분리한다고 볼 수 없음 |
| 비대칭 변동성 | ETHUSDT에서 주식시장형 leverage effect가 동일하게 나타난다고 가정하면 안 됨 |

---

## 19. 진행 중단 조건

다음 중 하나라도 해당하면 라벨링을 확정하지 않는다.

1. Binance ETHUSDT 1m·5m 데이터 다운로드 범위가 요구 기간을 완전히 덮지 못함
2. 데이터 누락·중복·미완성 candle 문제가 해결되지 않음
3. 1분봉과 5분봉의 timestamp 정렬 기준이 불명확함
4. Directional Change threshold가 종목별 가격 분포와 무관하게 고정됨
5. 변동성 분위수를 timeframe별로 따로 계산하지 않음
6. `NON_DIRECTIONAL` 정의 없이 UP/DOWN만 강제 부여함
7. 공식 검증 테스트 케이스를 통과하지 못함
8. 9개 라벨 외의 값이 출력됨
9. 전략 손익 또는 포지션 성과가 라벨 생성에 섞임
10. lag, cost, jump, downside volatility 진단값이 최종 9라벨을 변경함

---

## 20. 다음 단계 입력값

```yaml
next_step_input:
  task: "implement_regime_labeler"
  required_data:
    - "Binance USDⓈ-M Futures ETHUSDT 1m Kline"
    - "Binance USDⓈ-M Futures ETHUSDT 5m Kline"
  analysis_period_utc:
    start: "2024-01-01 00:00:00"
    end: "2026-05-25 23:59:59"
  target_timeframes:
    - "1m"
    - "5m"
  price_field:
    name: "hlc3"
    formula: "(high + low + close) / 3"
  label_schema:
    direction:
      - "UP"
      - "DOWN"
      - "NON_DIRECTIONAL"
    volatility:
      - "LOW_VOL"
      - "MID_VOL"
      - "HIGH_VOL"
    final_label_count: 9
  primary_direction_method:
    name: "Directional Change segment labeling"
    outputs:
      - "turning_points"
      - "segments"
      - "direction_label"
  volatility_method:
    name: "segment realized volatility per bar"
    outputs:
      - "realized_volatility"
      - "realized_volatility_per_bar"
      - "volatility_label"
  auxiliary_diagnostics:
    - "lag_diagnostics"
    - "amplitude_to_cost_ratio"
    - "pullback_within_parent_trend_flag"
    - "jump_share"
    - "bipower_variation"
    - "downside_vol_share"
    - "er_vol_correlation"
  validation:
    required:
      - "data_quality_validation"
      - "segment_invariant_validation"
      - "formula_range_validation"
      - "synthetic_case_validation"
      - "label_domain_validation"
      - "diagnostic_metric_validation"
```

---

## 21. 참고문헌 및 근거 자료

1. Bry, G. and Boschan, C. (1971). *Programmed Selection of Cyclical Turning Points*. NBER.  
   - PDF: https://www.nber.org/system/files/chapters/c2148/c2148.pdf  
   - Page: https://www.nber.org/books-and-chapters/cyclical-analysis-time-series-selected-procedures-and-computer-programs/programmed-selection-cyclical-turning-points

2. Pagan, A. and Sossounov, K. (2003). *A Simple Framework for Analysing Bull and Bear Markets*. Journal of Applied Econometrics.  
   - DOI page: https://onlinelibrary.wiley.com/doi/abs/10.1002/jae.664  
   - EconPapers: https://econpapers.repec.org/RePEc:jae:japmet:v:18:y:2003:i:1:p:23-46

3. Harding, D. and Pagan, A. (2002). *Dissecting the Cycle: A Methodological Investigation*. Journal of Monetary Economics.  
   - DOI page: https://www.sciencedirect.com/science/article/pii/S0304393201001088  
   - Ideas/RePEc: https://ideas.repec.org/a/eee/moneco/v49y2002i2p365-381.html

4. Aloud, M., Tsang, E., Olsen, R., and Dupuis, A. (2012). *A Directional-Change Event Approach for Studying Financial Time Series*.  
   - Page: https://www.econstor.eu/handle/10419/65285

5. Andersen, T. G., Bollerslev, T., Diebold, F. X., and Labys, P. (2001). *Modeling and Forecasting Realized Volatility*. NBER Working Paper 8160.  
   - PDF: https://www.nber.org/system/files/working_papers/w8160/w8160.pdf  
   - Page: https://www.nber.org/papers/w8160

6. Barndorff-Nielsen, O. E. and Shephard, N. (2004). *Power and Bipower Variation with Stochastic Volatility and Jumps*. Journal of Financial Econometrics.  
   - Article page: https://ideas.repec.org/a/oup/jfinec/v2y2004i1p1-37.html  
   - PDF: https://public.econ.duke.edu/~get/browse/courses/883/Spr16/COURSE-MATERIALS/Z_Papers/BNSJFEC2004.pdf

7. Bekaert, G. and Wu, G. (2000). *Asymmetric Volatility and Risk in Equity Markets*. Review of Financial Studies.  
   - JSTOR: https://www.jstor.org/stable/2646079  
   - PDF: https://business.columbia.edu/sites/default/files-efs/citation_file_upload/asymmetric%20volatility.pdf

8. Binance public data and USDⓈ-M Futures Kline references.  
   - Binance public data GitHub: https://github.com/binance/binance-public-data  
   - Binance Data Collection: https://data.binance.vision/  
   - ETHUSDT monthly futures klines: https://data.binance.vision/?prefix=data/futures/um/monthly/klines/ETHUSDT/  
   - USDⓈ-M Futures Kline API: https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Kline-Candlestick-Data
