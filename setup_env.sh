#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}/src/backend"

ENV_NAME="plume_nav_sim"
PYTHON_VERSION="3.10"
INCLUDE_DEV=0
PERFORM_UPDATE=0

log_info() {
    echo "[INFO] $*"
}

log_error() {
    echo "[ERROR] $*" >&2
}

log_debug() {
    echo "[DEBUG] $*"
}

usage() {
    cat <<USAGE
Usage: $0 [OPTIONS]

Options:
  --name NAME        Name of the conda environment to manage (default: ${ENV_NAME})
  --python VERSION   Python version to install when creating the environment (default: ${PYTHON_VERSION})
  --dev              Install development dependencies in addition to runtime dependencies
  --update           Update an existing environment instead of creating a new one
  -h, --help         Show this help message and exit
USAGE
}

failure() {
    local exit_code=$1
    local line_no=$2
    log_error "setup_env.sh failed at line ${line_no} with exit code ${exit_code}"
    exit "${exit_code}"
}

trap 'failure $? $LINENO' ERR

ensure_conda_available() {
    if ! command -v conda >/dev/null 2>&1; then
        log_error "conda command not found. Please install Miniconda or Anaconda first."
        exit 1
    fi
    log_debug "Found conda executable: $(command -v conda)"
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --name)
                shift
                [[ $# -gt 0 ]] || { log_error "Missing argument for --name"; usage >&2; exit 1; }
                ENV_NAME="$1"
                ;;
            --python)
                shift
                [[ $# -gt 0 ]] || { log_error "Missing argument for --python"; usage >&2; exit 1; }
                PYTHON_VERSION="$1"
                ;;
            --dev)
                INCLUDE_DEV=1
                ;;
            --update)
                PERFORM_UPDATE=1
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                usage >&2
                exit 1
                ;;
        esac
        shift
    done
}

environment_exists() {
    conda env list --json | python - "$ENV_NAME" <<'PY'
import json
import os
import sys

name = sys.argv[1]
try:
    data = json.load(sys.stdin)
except json.JSONDecodeError as exc:  # pragma: no cover - defensive
    raise SystemExit(f"Unable to parse output from conda: {exc}")
for path in data.get("envs", []):
    env_name = os.path.basename(path.rstrip(os.sep))
    if env_name == name:
        raise SystemExit(0)
raise SystemExit(1)
PY
}

assert_project_directory() {
    if [[ ! -d "${PROJECT_DIR}" ]]; then
        log_error "Project directory ${PROJECT_DIR} not found."
        exit 1
    fi
    if [[ ! -f "${PROJECT_DIR}/pyproject.toml" ]]; then
        log_error "pyproject.toml not found in ${PROJECT_DIR}."
        exit 1
    fi
}

install_dependencies() {
    local pip_target="${PROJECT_DIR}"
    if [[ ${INCLUDE_DEV} -eq 1 ]]; then
        pip_target="${pip_target}[dev]"
    fi
    log_info "Installing Python dependencies via pip for environment ${ENV_NAME}"
    conda run -n "${ENV_NAME}" python -m pip install --upgrade pip
    conda run -n "${ENV_NAME}" python -m pip install --upgrade -e "${pip_target}"
}

create_environment() {
    if environment_exists; then
        log_error "Conda environment '${ENV_NAME}' already exists. Use --update to refresh it."
        exit 1
    fi
    log_info "Creating conda environment '${ENV_NAME}' with Python ${PYTHON_VERSION}"
    conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}" pip
    install_dependencies
    log_info "Environment '${ENV_NAME}' created successfully."
}

update_environment() {
    if ! environment_exists; then
        log_error "Conda environment '${ENV_NAME}' does not exist. Run without --update to create it first."
        exit 1
    fi
    log_info "Updating conda environment '${ENV_NAME}'"
    install_dependencies
    log_info "Environment '${ENV_NAME}' updated successfully."
}

main() {
    parse_args "$@"
    ensure_conda_available
    assert_project_directory

    if [[ ${PERFORM_UPDATE} -eq 1 ]]; then
        update_environment
    else
        create_environment
    fi
}

main "$@"
