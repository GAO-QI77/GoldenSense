#!/usr/bin/env bash

set -euo pipefail

SCRIPT_PATH="$0"
if [[ "$SCRIPT_PATH" != /* ]]; then
  SCRIPT_PATH="$PWD/$SCRIPT_PATH"
fi
ROOT_DIR="$(cd "$(dirname "$SCRIPT_PATH")/.." && pwd)"
STATE_DIR="${TMPDIR:-/tmp}/goldensense-dev-stack"
LOG_DIR="$STATE_DIR/logs"
PID_DIR="$STATE_DIR/pids"
MPL_DIR="$STATE_DIR/mplconfig"

mkdir -p "$LOG_DIR" "$PID_DIR" "$MPL_DIR"

DATABASE_URL="${DATABASE_URL:-postgresql://postgres:postgres@127.0.0.1:5432/postgres}"
REDIS_URL="${REDIS_URL:-redis://127.0.0.1:6379/0}"
FORECAST_URL="${FORECAST_URL:-http://127.0.0.1:8010/api/v1/forecast}"
MEMORY_URL="${MEMORY_URL:-http://127.0.0.1:8012/api/v1/memory/search}"
MARKET_SNAPSHOT_URL="${MARKET_SNAPSHOT_URL:-http://127.0.0.1:8014/api/v1/market/snapshot/latest}"
RECENT_NEWS_URL="${RECENT_NEWS_URL:-http://127.0.0.1:8016/api/v1/news/recent}"
AGENT_PUBLIC_API_KEYS="${AGENT_PUBLIC_API_KEYS:-dev-public-key}"
AGENT_INTERNAL_API_KEYS="${AGENT_INTERNAL_API_KEYS:-dev-internal-key}"
AGENT_GATEWAY_INTERNAL_API_KEY="${AGENT_GATEWAY_INTERNAL_API_KEY:-dev-internal-key}"

start_service() {
  local name="$1"
  shift
  local pid_file="$PID_DIR/$name.pid"
  local log_file="$LOG_DIR/$name.log"
  if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
    echo "$name already running (pid $(cat "$pid_file"))"
    return 0
  fi
  (
    cd "$ROOT_DIR"
    nohup "$@" >"$log_file" 2>&1 < /dev/null &
    echo $! >"$pid_file"
  )
  echo "started $name (pid $(cat "$pid_file"))"
}

stop_service() {
  local name="$1"
  local pid_file="$PID_DIR/$name.pid"
  if [[ ! -f "$pid_file" ]]; then
    echo "$name not running"
    return 0
  fi
  local pid
  pid="$(cat "$pid_file")"
  if kill -0 "$pid" 2>/dev/null; then
    kill "$pid"
    echo "stopped $name (pid $pid)"
  else
    echo "$name already stopped"
  fi
  rm -f "$pid_file"
}

service_status() {
  local name="$1"
  local port="$2"
  local pid_file="$PID_DIR/$name.pid"
  if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
    echo "$name: running (pid $(cat "$pid_file"), port $port)"
  else
    echo "$name: stopped (port $port)"
  fi
}

start_all() {
  start_service inference env python3 -m uvicorn inference_service:app --host 127.0.0.1 --port 8010
  start_service memory env DATABASE_URL="$DATABASE_URL" MPLCONFIGDIR="$MPL_DIR" python3 -m uvicorn memory_service:app --host 127.0.0.1 --port 8012
  start_service market env DATABASE_URL="$DATABASE_URL" REDIS_URL="$REDIS_URL" MARKET_START_BACKGROUND_TASK=0 MARKET_ALLOW_SYNTHETIC_FALLBACK=1 python3 -m uvicorn market_snapshot_service:app --host 127.0.0.1 --port 8014
  start_service news env DATABASE_URL="$DATABASE_URL" REDIS_URL="$REDIS_URL" NEWS_START_BACKGROUND_TASK=0 NEWS_ALLOW_SAMPLE_FALLBACK=1 python3 -m uvicorn news_ingest_service:app --host 127.0.0.1 --port 8016
  start_service agent env DATABASE_URL="$DATABASE_URL" FORECAST_URL="$FORECAST_URL" MEMORY_URL="$MEMORY_URL" MARKET_SNAPSHOT_URL="$MARKET_SNAPSHOT_URL" RECENT_NEWS_URL="$RECENT_NEWS_URL" AGENT_PUBLIC_API_KEYS="$AGENT_PUBLIC_API_KEYS" AGENT_INTERNAL_API_KEYS="$AGENT_INTERNAL_API_KEYS" AGENT_TOOL_TIMEOUT_SECONDS=4.0 AGENT_TOOL_CONNECT_TIMEOUT_SECONDS=1.5 python3 -m uvicorn agent_gateway:app --host 127.0.0.1 --port 8020
}

stop_all() {
  stop_service agent
  stop_service news
  stop_service market
  stop_service memory
  stop_service inference
}

status_all() {
  service_status inference 8010
  service_status memory 8012
  service_status market 8014
  service_status news 8016
  service_status agent 8020
  echo "logs: $LOG_DIR"
}

case "${1:-status}" in
  start)
    start_all
    ;;
  stop)
    stop_all
    ;;
  restart)
    stop_all
    start_all
    ;;
  status)
    status_all
    ;;
  *)
    echo "usage: $0 {start|stop|restart|status}"
    exit 1
    ;;
esac
