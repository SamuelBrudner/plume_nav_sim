#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<USAGE
Usage: $(basename "$0") [--dev]
  --dev   Install development dependencies
USAGE
}

DEV=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dev)
      DEV=1
      shift
      ;;
    -h|--help)
      show_help
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown option: $1" >&2
      show_help >&2
      exit 1
      ;;
  esac
done

PY_BIN=$(command -v python || true)
PIP_BIN=$(command -v pip || true)

echo "[INFO] Using python: ${PY_BIN:-missing}"
echo "[INFO] Using pip: ${PIP_BIN:-missing}"

if [[ -z "$PY_BIN" ]]; then
  echo "[ERROR] python not found" >&2
  exit 1
fi
if [[ -z "$PIP_BIN" ]]; then
  echo "[ERROR] pip not found" >&2
  exit 1
fi

REQ=requirements.txt
if [[ $DEV -eq 1 ]]; then
  REQ=requirements-dev.txt
fi

if [[ ! -f "$REQ" ]]; then
  echo "[ERROR] $REQ not found" >&2
  exit 1
fi

echo "[INFO] Installing dependencies from $REQ"
$PIP_BIN install -r "$REQ"

echo "[INFO] Installing plume_nav_sim in editable mode"
$PIP_BIN install -e .

echo "[INFO] Environment ready"
$PY_BIN --version
$PIP_BIN --version
