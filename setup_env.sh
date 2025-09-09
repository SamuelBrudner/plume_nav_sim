#!/usr/bin/env bash
set -euo pipefail

trap 'echo "[ERROR] Script failed at line $LINENO" >&2' ERR

OS_NAME="$(uname)"
case "$OS_NAME" in
  Linux|Darwin)
    ;; # supported
  *)
    echo "[ERROR] Unsupported platform: $OS_NAME" >&2
    exit 1
    ;;
esac

dev_mode=0
while getopts ":-:" opt; do
  case "$opt" in
    -)
      case "$OPTARG" in
        dev) dev_mode=1 ;;
        *) echo "[ERROR] Unknown option --$OPTARG" >&2; exit 1 ;;
      esac
      ;;
    *)
      echo "[ERROR] Unknown option -$opt" >&2
      exit 1
      ;;
  esac
done
shift $((OPTIND -1))

req_file="requirements.txt"
if [[ "$dev_mode" -eq 1 ]]; then
  req_file="requirements-dev.txt"
  echo "[INFO] Installing development dependencies"
else
  echo "[INFO] Installing dependencies"
fi

echo "[INFO] Verifying required tools"
for tool in python pip; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "[ERROR] Required tool '$tool' not found in PATH" >&2
    exit 1
  fi
done

if command -v conda >/dev/null 2>&1; then
  echo "[INFO] Conda detected: $(conda --version 2>/dev/null)"
else
  echo "[WARN] Conda not found; proceeding without it"
fi

if [[ ! -f "$req_file" ]]; then
  echo "[ERROR] Requirements file '$req_file' not found" >&2
  exit 1
fi

if [[ "${PLUMENAV_SKIP_INSTALL:-0}" == "1" ]]; then
  echo "[WARN] PLUMENAV_SKIP_INSTALL set; skipping pip install"
else
  echo "[INFO] Installing from $req_file"
  pip install -r "$req_file"
  echo "[INFO] Installation complete"
fi

echo "[INFO] Python version: $(python --version 2>&1)"
echo "[INFO] Pip version: $(pip --version 2>&1)"

echo "[INFO] Dependency versions:"
python <<PY
import importlib.metadata, pathlib, sys
req_file = pathlib.Path('$req_file')
for line in req_file.read_text().splitlines():
    line = line.strip()
    if not line or line.startswith('#'):
        continue
    pkg = line.split('==')[0].split('>=')[0].split('<=')[0].split('[')[0]
    try:
        version = importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError:
        version = 'not installed'
    print(f"[INFO]   {pkg}: {version}")
PY
