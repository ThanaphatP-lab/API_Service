#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${1:-all}"
PYTHON_BIN="${PYTHON_BIN:-}"
: "${PIP_TIMEOUT_SECONDS:=180}"
: "${PIP_RETRIES:=20}"
: "${PIP_RESUME_RETRIES:=50}"

if [[ -z "$PYTHON_BIN" ]]; then
  for candidate in python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
      PYTHON_BIN="$candidate"
      break
    fi
  done
fi

if [[ -z "$PYTHON_BIN" ]] || ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "[ERROR] Python 3.10-3.12 was not found. Set PYTHON_BIN explicitly." >&2
  exit 1
fi

python_version="$($PYTHON_BIN -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
case "$python_version" in
  3.10|3.11|3.12) ;;
  *)
    echo "[ERROR] Python $python_version is unsupported by the pinned GPU dependencies; use 3.10-3.12." >&2
    exit 1
    ;;
esac

setup_env() {
  local directory="$1"
  local requirements="$2"
  local environment="$ROOT/$directory"

  if [[ ! -x "$environment/bin/python" ]]; then
    echo "[INFO] Creating $directory with $PYTHON_BIN..."
    "$PYTHON_BIN" -m venv "$environment"
  fi

  # A venv may have been created before the OS python-venv package was
  # installed, leaving bin/python present but pip missing. Repair that state
  # instead of treating the partially-created environment as usable.
  if ! "$environment/bin/python" -m pip --version >/dev/null 2>&1; then
    echo "[WARN] pip is missing from $directory; repairing it with ensurepip..."
    if ! "$environment/bin/python" -m ensurepip --upgrade; then
      echo "[ERROR] Could not install pip into $directory." >&2
      echo "        Install the matching OS package (for example python3.12-venv)" >&2
      echo "        or move the broken environment aside and rerun this script." >&2
      return 1
    fi
  fi

  echo "[INFO] Installing $requirements into $directory..."
  "$environment/bin/python" -m pip install --upgrade pip
  if ! "$environment/bin/python" -m pip install \
    --timeout "$PIP_TIMEOUT_SECONDS" \
    --retries "$PIP_RETRIES" \
    --resume-retries "$PIP_RESUME_RETRIES" \
    -r "$ROOT/$requirements"; then
    echo "[ERROR] Dependency download/install failed for $directory." >&2
    echo "        The environment is reusable; do not delete it." >&2
    echo "        Retry with: $0 ${directory#.venv-}" >&2
    return 1
  fi
}

case "$TARGET" in
  all)
    setup_env .venv-paddle requirements-paddle.txt
    setup_env .venv-siglip requirements-siglip.txt
    setup_env .venv-api requirements-api.txt
    ;;
  paddle) setup_env .venv-paddle requirements-paddle.txt ;;
  siglip) setup_env .venv-siglip requirements-siglip.txt ;;
  api) setup_env .venv-api requirements-api.txt ;;
  *)
    echo "Usage: $0 <all|paddle|siglip|api>" >&2
    exit 2
    ;;
esac

echo "[OK] Linux environment setup complete: $TARGET"
