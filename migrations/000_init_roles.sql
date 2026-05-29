-- migrations/000_init_roles.sql
-- Regime Benchmark — role + database 부트스트랩
-- 실행: scripts/bootstrap_db.sh 가 superuser 권한으로 호출.
--       (수동: REGIME_OWNER_PASSWORD=... psql -U <super> -d postgres -f migrations/000_init_roles.sql)
-- 멱등: 이미 존재하면 no-op.
-- 본 스크립트는 application 스키마를 만들지 않는다 (001_init.sql 책임).
--
-- 비밀번호 처리:
--   psql 의 :'var' 치환은 dollar-quoted ($$...$$) 안에서 작동하지 않으므로
--   \getenv 로 env var 를 psql 변수에 받은 뒤 set_config() (함수 호출, parameter-safe) 로
--   세션 GUC 에 주입 → DO block 안에서 current_setting() 으로 읽는다.

\set ON_ERROR_STOP on

-- env var REGIME_OWNER_PASSWORD → psql 변수 :owner_pw
\getenv owner_pw REGIME_OWNER_PASSWORD

-- 빈 값 또는 미설정 시 즉시 종료 (스크립트 외 단독 실행 대비)
\if :{?owner_pw}
\else
  \warn 'REGIME_OWNER_PASSWORD must be set in the env'
  \quit
\endif

-- 안전 주입 (single-quote 등 escape 는 set_config 가 처리)
-- \g /dev/null : 반환값이 비밀번호 평문이므로 출력 캡처 회피.
SELECT set_config('bootstrap.owner_pw', :'owner_pw', false) \g /dev/null

-- 1. regime_owner role (CREATE ROLE 은 IF NOT EXISTS 미지원 → DO block)
DO $bootstrap$
DECLARE
    pw text := current_setting('bootstrap.owner_pw');
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'regime_owner') THEN
        EXECUTE format('CREATE ROLE regime_owner LOGIN PASSWORD %L', pw);
        RAISE NOTICE 'role regime_owner created';
    ELSE
        RAISE NOTICE 'role regime_owner already exists — skipped';
    END IF;
END
$bootstrap$;

-- 세션 GUC 에서 비밀번호 즉시 제거 (방어). 출력도 캡처 회피.
SELECT set_config('bootstrap.owner_pw', '', false) \g /dev/null

-- 2. regime_benchmark database (CREATE DATABASE 는 트랜잭션 외부 → \gexec)
SELECT 'CREATE DATABASE regime_benchmark OWNER regime_owner ENCODING ''UTF8'' TEMPLATE template0'
WHERE NOT EXISTS (SELECT 1 FROM pg_database WHERE datname = 'regime_benchmark')
\gexec

-- 확인 출력 (owner 와 encoding 까지 검증)
SELECT
    (SELECT count(*) > 0 FROM pg_roles WHERE rolname = 'regime_owner')                       AS role_ready,
    (SELECT count(*) > 0 FROM pg_database WHERE datname = 'regime_benchmark')                AS db_exists,
    (SELECT datdba::regrole::text FROM pg_database WHERE datname = 'regime_benchmark')       AS db_owner,
    (SELECT pg_encoding_to_char(encoding) FROM pg_database WHERE datname = 'regime_benchmark') AS db_encoding;
