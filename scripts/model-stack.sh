#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${MODEL_ENV_FILE:-$ROOT/.env.runtime}"
RUN_DIR="${MODEL_RUN_DIR:-$ROOT/.run}"
LOG_DIR="${MODEL_LOG_DIR:-$ROOT/logs}"
CACHE_DIR="${MODEL_CACHE_DIR:-$ROOT/.cache}"
MODEL_TMP_DIR="${MODEL_TMP_DIR:-$ROOT/.tmp}"

if [[ -f "$ENV_FILE" ]]; then
  if grep -q $'\r' "$ENV_FILE"; then
    echo "[ERROR] $ENV_FILE uses Windows CRLF line endings. Convert it with: sed -i 's/\r$//' '$ENV_FILE'" >&2
    exit 1
  fi
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

: "${APP_ENV:=production}"
: "${MODEL_DEVICE:=gpu:0}"
: "${PADDLE_PDX_MODEL_SOURCE:=BOS}"
: "${INTERNAL_SERVICE_HOST:=127.0.0.1}"
: "${GATEWAY_HOST:=127.0.0.1}"
: "${DEMO_HOST:=127.0.0.1}"
: "${STARTUP_WAIT_SECONDS:=45}"
: "${SHUTDOWN_WAIT_SECONDS:=30}"
: "${READINESS_TIMEOUT_SECONDS:=300}"

: "${LAYOUT_PORT:=8001}"
: "${DET_V5_PORT:=8002}"
: "${DET_V6_PORT:=8003}"
: "${DETECTION_PORT:=$DET_V5_PORT}"
: "${REC_TH_PORT:=8004}"
: "${OCR_CUSTOM_PORT:=8005}"
: "${OCR_PADDLE_PORT:=8006}"
: "${TABLE_WIRED_PORT:=8007}"
: "${TABLE_WIRELESS_PORT:=8008}"
: "${SIGLIP_PORT:=8009}"
: "${LAYOUT_PIPELINE_PORT:=8010}"
: "${TABLE_PIPELINE_PORT:=8011}"
: "${IMAGE_VERIFICATION_PORT:=8012}"
: "${TABLE_V2_PORT:=8013}"
: "${GATEWAY_PORT:=8080}"
: "${DEMO_PORT:=8501}"

: "${API_PYTHON:=$ROOT/.venv-api/bin/python}"
: "${PADDLE_PYTHON:=$ROOT/.venv-paddle/bin/python}"
: "${SIGLIP_PYTHON:=$ROOT/.venv-siglip/bin/python}"
: "${PADDLE_PDX_CACHE_HOME:=$CACHE_DIR/paddlex}"
: "${HF_HOME:=$CACHE_DIR/huggingface}"
: "${TORCH_HOME:=$CACHE_DIR/torch}"
: "${XDG_CACHE_HOME:=$CACHE_DIR/xdg}"
: "${TMPDIR:=$MODEL_TMP_DIR}"

export APP_ENV MODEL_DEVICE PADDLE_PDX_MODEL_SOURCE INTERNAL_SERVICE_HOST GATEWAY_HOST DEMO_HOST
export PADDLE_PDX_CACHE_HOME HF_HOME TORCH_HOME XDG_CACHE_HOME TMPDIR
mkdir -p "$RUN_DIR" "$LOG_DIR" "$CACHE_DIR" "$MODEL_TMP_DIR"
chmod 700 "$RUN_DIR" "$LOG_DIR" "$CACHE_DIR" "$MODEL_TMP_DIR"

ALL_SERVICES=(
  layout detection rec-th table-wired table-wireless table-v2 siglip
  ocr-custom ocr-paddle layout-pipeline table-pipeline image-verification gateway
)
ALL_WITH_DEMO=("${ALL_SERVICES[@]}" demo)
CORE_SERVICES=(
  layout detection rec-th table-wired table-wireless siglip
  ocr-custom layout-pipeline table-pipeline image-verification gateway
)

usage() {
  cat <<'EOF'
Usage:
  scripts/model-stack.sh start <service|profile>
  scripts/model-stack.sh stop <service|profile|all>
  scripts/model-stack.sh restart <service|profile>
  scripts/model-stack.sh check
  scripts/model-stack.sh status [service|profile|all]
  scripts/model-stack.sh readiness [service|profile|all]
  scripts/model-stack.sh logs <service> [lines]

Profiles:
  core-stack        default production stack using split polygon det + rec OCR
  ocr-custom-stack  detection (v5/v6) + rec-th + ocr-custom + gateway
  ocr-paddle-stack  ocr-paddle + gateway
  layout-stack      layout + detection + layout-pipeline + gateway
  table-stack       detection + rec-th + ocr-custom + table models/pipeline + gateway
  table-v2-stack    notebook TableRecognitionPipelineV2 + gateway (no split table models)
  verification-stack siglip + image-verification + gateway
  all               every model/pipeline + gateway (Demo is started separately)

Services:
  layout detection rec-th ocr-custom ocr-paddle table-wired
  table-wireless table-v2 siglip layout-pipeline table-pipeline image-verification
  gateway demo

Legacy det-v5/det-v6 names remain for stop/status/logs migration only.
Use detection for one process serving version=5/6 and model variants.
EOF
}

target_services() {
  case "${1:-all}" in
    core-stack) echo "${CORE_SERVICES[*]}" ;;
    ocr-custom-stack) echo "detection rec-th ocr-custom gateway" ;;
    ocr-paddle-stack) echo "ocr-paddle gateway" ;;
    layout-stack) echo "layout detection layout-pipeline gateway" ;;
    table-stack) echo "detection rec-th ocr-custom table-wired table-wireless table-pipeline gateway" ;;
    table-v2-stack) echo "table-v2 gateway" ;;
    verification-stack) echo "siglip image-verification gateway" ;;
    all) echo "${ALL_WITH_DEMO[*]}" ;;
    layout|detection|det-v5|det-v6|rec-th|ocr-custom|ocr-paddle|table-wired|table-wireless|table-v2|siglip|layout-pipeline|table-pipeline|image-verification|gateway|demo) echo "$1" ;;
    *)
      echo "[ERROR] Unknown service/profile: $1" >&2
      usage >&2
      return 2
      ;;
  esac
}

start_target_services() {
  if [[ "$1" == "all" ]]; then
    echo "${ALL_SERVICES[*]}"
  else
    target_services "$1"
  fi
}

profile_pipelines() {
  case "$1" in
    core-stack) echo "layout,ocr-custom,table,image-verification,text-detection,text-recognition,siglip" ;;
    ocr-custom-stack) echo "ocr-custom,text-detection,text-recognition" ;;
    ocr-paddle-stack) echo "ocr-paddle" ;;
    layout-stack) echo "layout,text-detection" ;;
    table-stack) echo "table,text-detection,text-recognition" ;;
    table-v2-stack) echo "table-model" ;;
    verification-stack) echo "image-verification,siglip" ;;
    all) echo "all" ;;
    *) echo "" ;;
  esac
}

service_port() {
  case "$1" in
    layout) echo "$LAYOUT_PORT" ;;
    detection) echo "$DETECTION_PORT" ;;
    det-v5) echo "$DET_V5_PORT" ;;
    det-v6) echo "$DET_V6_PORT" ;;
    rec-th) echo "$REC_TH_PORT" ;;
    ocr-custom) echo "$OCR_CUSTOM_PORT" ;;
    ocr-paddle) echo "$OCR_PADDLE_PORT" ;;
    table-wired) echo "$TABLE_WIRED_PORT" ;;
    table-wireless) echo "$TABLE_WIRELESS_PORT" ;;
    table-v2) echo "$TABLE_V2_PORT" ;;
    siglip) echo "$SIGLIP_PORT" ;;
    layout-pipeline) echo "$LAYOUT_PIPELINE_PORT" ;;
    table-pipeline) echo "$TABLE_PIPELINE_PORT" ;;
    image-verification) echo "$IMAGE_VERIFICATION_PORT" ;;
    gateway) echo "$GATEWAY_PORT" ;;
    demo) echo "$DEMO_PORT" ;;
  esac
}

service_module() {
  case "$1" in
    layout) echo "services.layout.main:app" ;;
    detection|det-v5|det-v6) echo "services.text_det.main:app" ;;
    rec-th) echo "services.text_rec.main:app" ;;
    ocr-custom) echo "services.ocr_pipeline_custom.main:app" ;;
    ocr-paddle) echo "services.ocr_pipeline_paddle.main:app" ;;
    table-wired|table-wireless) echo "services.table.main:app" ;;
    table-v2) echo "services.table_v2.main:app" ;;
    siglip) echo "services.siglip.main:app" ;;
    layout-pipeline) echo "services.layout_pipeline.main:app" ;;
    table-pipeline) echo "services.table_pipeline.main:app" ;;
    image-verification) echo "services.image_verification_pipeline.main:app" ;;
    gateway) echo "services.gateway.main:app" ;;
    demo) echo "demo/app.py" ;;
  esac
}

service_python() {
  case "$1" in
    layout|detection|det-v5|det-v6|rec-th|ocr-paddle|table-wired|table-wireless|table-v2) echo "$PADDLE_PYTHON" ;;
    siglip) echo "$SIGLIP_PYTHON" ;;
    *) echo "$API_PYTHON" ;;
  esac
}

service_host() {
  case "$1" in
    gateway) echo "$GATEWAY_HOST" ;;
    demo) echo "$DEMO_HOST" ;;
    *) echo "$INTERNAL_SERVICE_HOST" ;;
  esac
}

pid_file() { echo "$RUN_DIR/$1.pid"; }
log_file() { echo "$LOG_DIR/$1.log"; }

read_pid() {
  local file
  file="$(pid_file "$1")"
  [[ -f "$file" ]] || return 1
  local pid
  pid="$(tr -d '[:space:]' < "$file")"
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  echo "$pid"
}

pid_matches_service() {
  local service="$1"
  local pid="$2"
  [[ -r "/proc/$pid/cmdline" ]] || return 1
  local command_line
  command_line="$(tr '\0' ' ' < "/proc/$pid/cmdline")"
  [[ "$command_line" == *"$(service_module "$service")"* ]]
}

service_running() {
  local service="$1"
  local pid
  pid="$(read_pid "$service")" || return 1
  kill -0 "$pid" 2>/dev/null && pid_matches_service "$service" "$pid"
}

port_in_use() {
  local port="$1"
  if command -v ss >/dev/null 2>&1; then
    ss -ltn 2>/dev/null | awk 'NR > 1 {print $4}' | grep -Eq "(^|:)${port}$"
  elif command -v lsof >/dev/null 2>&1; then
    lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1
  else
    return 1
  fi
}

validate_secret() {
  local name="$1"
  local value="${!name:-}"
  if [[ "$APP_ENV" == "production" || "$APP_ENV" == "prod" ]]; then
    if (( ${#value} < 32 )) || [[ "$value" == replace-* || "$value" == change-me* || "$value" == '<'*'>' ]]; then
      echo "[ERROR] $name must be a non-placeholder secret of at least 32 characters in production." >&2
      return 1
    fi
  elif [[ -z "$value" ]]; then
    echo "[WARN] $name is empty; protected APIs will run without authentication in development." >&2
  fi
}

preflight() {
  for command_name in curl nohup; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
      echo "[ERROR] Required command not found: $command_name" >&2
      return 1
    fi
  done
  validate_secret MODEL_GATEWAY_API_KEY
  validate_secret INTERNAL_API_TOKEN
  if [[ -f "$ENV_FILE" ]] && command -v stat >/dev/null 2>&1; then
    local env_mode
    env_mode="$(stat -c '%a' "$ENV_FILE" 2>/dev/null || true)"
    if [[ -n "$env_mode" ]] && (( (8#$env_mode & 077) != 0 )); then
      echo "[ERROR] $ENV_FILE permissions are $env_mode; run: chmod 600 '$ENV_FILE'" >&2
      return 1
    fi
  fi
  if [[ ! -d "$ROOT/weights" ]]; then
    echo "[ERROR] Missing weights directory: $ROOT/weights" >&2
    return 1
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi >/dev/null || {
      echo "[ERROR] nvidia-smi exists but the NVIDIA driver/GPU is unavailable." >&2
      return 1
    }
  elif [[ "$MODEL_DEVICE" == gpu* ]]; then
    echo "[ERROR] MODEL_DEVICE=$MODEL_DEVICE but nvidia-smi was not found." >&2
    return 1
  fi
}

print_check() {
  preflight
  echo "[OK] root=$ROOT"
  echo "[OK] environment=$APP_ENV device=$MODEL_DEVICE"
  echo "[OK] runtime_dir=$RUN_DIR log_dir=$LOG_DIR"
  echo "[OK] cache_dir=$CACHE_DIR temp_dir=$MODEL_TMP_DIR"
  for environment in "$API_PYTHON" "$PADDLE_PYTHON" "$SIGLIP_PYTHON"; do
    if [[ -x "$environment" ]]; then
      echo "[OK] python=$environment ($("$environment" --version 2>&1))"
    else
      echo "[WARN] Python environment is not installed: $environment"
    fi
  done
  local model_file_count
  model_file_count="$(find "$ROOT/weights" -type f ! -name .gitkeep 2>/dev/null | wc -l | tr -d '[:space:]')"
  if [[ "$model_file_count" == "0" ]]; then
    echo "[WARN] weights/ has no model files; first start requires model registry/network access."
  else
    echo "[OK] local_weight_files=$model_file_count"
  fi
}

export_service_environment() {
  local service="$1"
  export LAYOUT_MODEL_DIR="${LAYOUT_MODEL_DIR:-$ROOT/weights/layout}"
  export TEXT_DETECTION_URL="${TEXT_DETECTION_URL:-http://127.0.0.1:$DETECTION_PORT}"
  export DET_SERVICE_URL="$TEXT_DETECTION_URL"
  export REC_SERVICE_URL="${REC_SERVICE_URL:-http://127.0.0.1:$REC_TH_PORT}"
  export TABLE_WIRED_SERVICE_URL="${TABLE_WIRED_SERVICE_URL:-http://127.0.0.1:$TABLE_WIRED_PORT}"
  export TABLE_WIRELESS_SERVICE_URL="${TABLE_WIRELESS_SERVICE_URL:-http://127.0.0.1:$TABLE_WIRELESS_PORT}"
  export OCR_SERVICE_URL="${OCR_SERVICE_URL:-http://127.0.0.1:$OCR_CUSTOM_PORT}"
  export LAYOUT_SERVICE_URL="${LAYOUT_SERVICE_URL:-http://127.0.0.1:$LAYOUT_PORT}"
  export SIGLIP_SERVICE_URL="${SIGLIP_SERVICE_URL:-http://127.0.0.1:$SIGLIP_PORT}"
  export LAYOUT_PIPELINE_URL="${LAYOUT_PIPELINE_URL:-http://127.0.0.1:$LAYOUT_PIPELINE_PORT}"
  export OCR_CUSTOM_URL="${OCR_CUSTOM_URL:-http://127.0.0.1:$OCR_CUSTOM_PORT}"
  export OCR_PADDLE_URL="${OCR_PADDLE_URL:-http://127.0.0.1:$OCR_PADDLE_PORT}"
  export TABLE_PIPELINE_URL="${TABLE_PIPELINE_URL:-http://127.0.0.1:$TABLE_PIPELINE_PORT}"
  export TABLE_MODEL_URL="${TABLE_MODEL_URL:-http://127.0.0.1:$TABLE_V2_PORT}"
  export IMAGE_VERIFICATION_URL="${IMAGE_VERIFICATION_URL:-http://127.0.0.1:$IMAGE_VERIFICATION_PORT}"
  export GATEWAY_URL="${GATEWAY_URL:-http://127.0.0.1:$GATEWAY_PORT}"
  export LAYOUT_URL="${LAYOUT_URL:-http://127.0.0.1:$LAYOUT_PORT}"
  export DET_V5_URL="$TEXT_DETECTION_URL"
  export DET_V6_URL="$TEXT_DETECTION_URL"
  export REC_TH_URL="${REC_TH_URL:-http://127.0.0.1:$REC_TH_PORT}"
  export TABLE_WIRED_URL="${TABLE_WIRED_URL:-http://127.0.0.1:$TABLE_WIRED_PORT}"
  export TABLE_WIRELESS_URL="${TABLE_WIRELESS_URL:-http://127.0.0.1:$TABLE_WIRELESS_PORT}"
  export SIGLIP_URL="${SIGLIP_URL:-http://127.0.0.1:$SIGLIP_PORT}"
  if [[ "$service" == "gateway" ]]; then
    export MAX_CONCURRENT_REQUESTS="${GATEWAY_MAX_CONCURRENT_REQUESTS:-4}"
    if [[ -n "${STACK_GATEWAY_PIPELINES:-}" ]]; then
      export GATEWAY_ENABLED_PIPELINES="$STACK_GATEWAY_PIPELINES"
    fi
  fi

  case "$service" in
    detection)
      # Registry owns model names/directories for BOTH versions. Do not impose
      # the old per-process DET_MODEL_NAME / DET_MODEL_DIR baseline overrides.
      unset DET_MODEL_NAME DET_MODEL_DIR
      export DET_MODEL_VERSION="${DET_MODEL_VERSION:-v5}"
      export MODEL_VARIANTS_CONFIG="${MODEL_VARIANTS_CONFIG:-$ROOT/model_variants.json}"
      ;;
    rec-th)
      export REC_MODEL_NAME=th_PP-OCRv5_mobile_rec
      export REC_MODEL_VERSION=v5
      export REC_MODEL_DIR="${REC_TH_MODEL_DIR:-$ROOT/weights/recognition/rec-v5}"
      ;;
    ocr-paddle)
      export DET_MODEL_NAME=PP-OCRv6_medium_det
      export DET_MODEL_DIR="${DET_V6_MODEL_DIR:-$ROOT/weights/detection/det-v6}"
      export REC_MODEL_NAME=th_PP-OCRv5_mobile_rec
      export REC_MODEL_DIR="${REC_TH_MODEL_DIR:-$ROOT/weights/recognition/rec-v5}"
      ;;
    table-wired)
      export TABLE_MODEL_NAME=SLANeXt_wired
      export TABLE_MODEL_DIR="${TABLE_WIRED_MODEL_DIR:-$ROOT/weights/table-wired}"
      ;;
    table-wireless)
      export TABLE_MODEL_NAME=SLANeXt_wireless
      export TABLE_MODEL_DIR="${TABLE_WIRELESS_MODEL_DIR:-$ROOT/weights/table-wireless}"
      ;;
    siglip)
      export SIGLIP_MODEL_NAME="${SIGLIP_MODEL_NAME:-google/siglip-so400m-patch14-384}"
      export SIGLIP_MODEL_DIR="${SIGLIP_MODEL_DIR:-$ROOT/weights/siglip}"
      ;;
  esac
}

wait_for_health() {
  local service="$1"
  local port
  port="$(service_port "$service")"
  local deadline=$((SECONDS + STARTUP_WAIT_SECONDS))
  local path="/health"
  [[ "$service" == "demo" ]] && path="/"
  while (( SECONDS < deadline )); do
    if curl -fsS --max-time 2 "http://127.0.0.1:$port$path" >/dev/null 2>&1; then
      return 0
    fi
    service_running "$service" || return 1
    sleep 1
  done
  return 1
}

start_service() {
  local service="$1"
  if [[ "$service" == "det-v5" || "$service" == "det-v6" ]]; then
    echo "[ERROR] Use 'start detection'; select version/model in each API request." >&2
    return 1
  fi
  if [[ "$service" == "detection" ]] && { service_running det-v5 || service_running det-v6; }; then
    echo "[ERROR] Stop legacy det-v5 and det-v6 before starting detection." >&2
    return 1
  fi
  if service_running "$service"; then
    echo "[SKIP] $service is already running (PID $(read_pid "$service"))."
    return 0
  fi

  local stale
  stale="$(pid_file "$service")"
  [[ -f "$stale" ]] && rm -f -- "$stale"

  local port python module host
  port="$(service_port "$service")"
  python="$(service_python "$service")"
  module="$(service_module "$service")"
  host="$(service_host "$service")"

  if [[ ! -x "$python" ]]; then
    echo "[ERROR] $service Python environment not found: $python" >&2
    echo "        Run: scripts/setup-linux.sh all" >&2
    return 1
  fi
  if port_in_use "$port"; then
    echo "[ERROR] Port $port for $service is already in use by an unmanaged process." >&2
    return 1
  fi

  echo "[START] $service on $host:$port"
  (
    trap '' HUP
    cd "$ROOT"
    export_service_environment "$service"
    if [[ "$service" == "demo" ]]; then
      exec nohup "$python" -m streamlit run "$module" --server.address "$host" --server.port "$port"
    else
      exec nohup "$python" -m uvicorn "$module" --host "$host" --port "$port" --workers 1
    fi
  ) </dev/null >>"$(log_file "$service")" 2>&1 &
  local pid=$!
  printf '%s\n' "$pid" > "$(pid_file "$service")"

  if wait_for_health "$service"; then
    echo "[OK] $service is live (PID $pid)."
  else
    echo "[ERROR] $service did not become live. Inspect: $(log_file "$service")" >&2
    if ! service_running "$service"; then
      rm -f -- "$(pid_file "$service")"
    fi
    return 1
  fi
}

stop_service() {
  local service="$1"
  local pid
  if ! pid="$(read_pid "$service")"; then
    echo "[SKIP] $service has no PID file."
    return 0
  fi
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "[CLEAN] Removing stale PID for $service."
    rm -f -- "$(pid_file "$service")"
    return 0
  fi
  if ! pid_matches_service "$service" "$pid"; then
    echo "[ERROR] Refusing to kill PID $pid: it no longer matches $service." >&2
    return 1
  fi

  echo "[STOP] $service (PID $pid)"
  kill -TERM "$pid"
  local deadline=$((SECONDS + SHUTDOWN_WAIT_SECONDS))
  while kill -0 "$pid" 2>/dev/null && (( SECONDS < deadline )); do
    sleep 1
  done
  if kill -0 "$pid" 2>/dev/null; then
    echo "[WARN] $service did not stop gracefully; sending SIGKILL." >&2
    kill -KILL "$pid"
  fi
  rm -f -- "$(pid_file "$service")"
  echo "[OK] $service stopped."
}

status_service() {
  local service="$1"
  local port pid="-" state="stopped" health="offline"
  port="$(service_port "$service")"
  if service_running "$service"; then
    pid="$(read_pid "$service")"
    state="running"
    local path="/health"
    [[ "$service" == "demo" ]] && path="/"
    if curl -fsS --max-time 2 "http://127.0.0.1:$port$path" >/dev/null 2>&1; then
      health="live"
    else
      health="starting/unhealthy"
    fi
  elif port_in_use "$port"; then
    state="unmanaged"
    health="port-in-use"
  fi
  printf '%-24s port=%-5s pid=%-8s state=%-10s health=%s\n' "$service" "$port" "$pid" "$state" "$health"
}

readiness_service() {
  local service="$1"
  if [[ "$service" == "demo" ]]; then
    status_service "$service"
    return
  fi
  local port token
  port="$(service_port "$service")"
  token="${INTERNAL_API_TOKEN:-}"
  [[ "$service" == "gateway" ]] && token="${MODEL_GATEWAY_API_KEY:-}"
  echo "--- $service :$port readiness ---"
  curl -sS --max-time "$READINESS_TIMEOUT_SECONDS" \
    -H "Authorization: Bearer $token" \
    -w $'\nHTTP %{http_code}\n' \
    "http://127.0.0.1:$port/api/v1/readiness" || true
}

action="${1:-}"
target="${2:-all}"

case "$action" in
  start)
    preflight
    STACK_GATEWAY_PIPELINES="$(profile_pipelines "$target")"
    export STACK_GATEWAY_PIPELINES
    read -r -a services <<< "$(start_target_services "$target")"
    for service in "${services[@]}"; do start_service "$service"; done
    ;;
  stop)
    read -r -a services <<< "$(target_services "$target")"
    [[ "$target" == "all" ]] && services+=(det-v5 det-v6)
    for ((index=${#services[@]}-1; index>=0; index--)); do stop_service "${services[index]}"; done
    ;;
  restart)
    "$ROOT/scripts/model-stack.sh" stop "$target"
    "$ROOT/scripts/model-stack.sh" start "$target"
    ;;
  check) print_check ;;
  status)
    read -r -a services <<< "$(target_services "$target")"
    [[ "$target" == "all" ]] && services+=(det-v5 det-v6)
    for service in "${services[@]}"; do status_service "$service"; done
    ;;
  readiness)
    read -r -a services <<< "$(target_services "$target")"
    for service in "${services[@]}"; do readiness_service "$service"; done
    ;;
  logs)
    service="${2:-}"
    lines="${3:-100}"
    if [[ -z "$service" ]]; then usage; exit 2; fi
    target_services "$service" >/dev/null
    if [[ ! -f "$(log_file "$service")" ]]; then
      echo "[ERROR] No log file for $service: $(log_file "$service")" >&2
      exit 1
    fi
    exec tail -n "$lines" -F "$(log_file "$service")"
    ;;
  help|-h|--help) usage ;;
  *) usage; exit 2 ;;
esac
