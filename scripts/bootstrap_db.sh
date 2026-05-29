#!/usr/bin/env bash
# scripts/bootstrap_db.sh — Regime Benchmark DB 부트스트랩 (1 회용)
#
#   1) regime_owner role + regime_benchmark DB 생성 (superuser 권한)
#   2) 스키마 생성 (regime_owner 권한)
#
# 본 스크립트는 .env 를 source 하지 않는다 (application 컨텍스트와 분리).
# 필수 env 는 셸에서 인라인으로 1 회 전달:
#
#   REGIME_SUPERUSER=trader \
#   PGPASSWORD='***'              \   # superuser 비밀번호 (또는 ~/.pgpass 활용)
#   REGIME_OWNER_PASSWORD='***'   \   # 새로 만들 regime_owner 비밀번호
#   ./scripts/bootstrap_db.sh
#
# 선택:
#   PGHOST (기본 localhost), PGPORT (기본 5432)
#
# 멱등성:
#   - 1 단계: role/database 이미 존재 시 no-op.
#   - 2 단계: 두 번째 실행 시 ENUM/테이블 중복 에러로 트랜잭션 전체 롤백.

set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

: "${REGIME_SUPERUSER:?REGIME_SUPERUSER must be set inline (e.g. REGIME_SUPERUSER=trader ./scripts/bootstrap_db.sh)}"
: "${REGIME_OWNER_PASSWORD:?REGIME_OWNER_PASSWORD must be set inline}"

# 1/2 — role + database (superuser 로 postgres DB 접속)
# superuser 비밀번호: PGPASSWORD 인라인 또는 ~/.pgpass.
# REGIME_OWNER_PASSWORD 는 환경변수로 상속되어 SQL 의 \getenv 가 읽는다.
export REGIME_OWNER_PASSWORD
psql -X -v ON_ERROR_STOP=1 \
     -h "${PGHOST:-localhost}" -p "${PGPORT:-5432}" \
     -U "$REGIME_SUPERUSER" -d postgres \
     -f "$REPO/migrations/000_init_roles.sql"

# 2/2 — schema (regime_owner 로 regime_benchmark 접속)
PGPASSWORD="$REGIME_OWNER_PASSWORD" \
psql -X -v ON_ERROR_STOP=1 \
     -h "${PGHOST:-localhost}" -p "${PGPORT:-5432}" \
     -U regime_owner -d regime_benchmark \
     -f "$REPO/migrations/001_init.sql"

echo "bootstrap complete: regime_benchmark ready."
echo
echo "next: register regime_owner 비밀번호를 ~/.pgpass 또는 REGIME_BENCHMARK_DB_URL 인라인으로 보관."
