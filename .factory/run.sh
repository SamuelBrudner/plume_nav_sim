#!/usr/bin/env bash
set -euo pipefail
python3 -m venv .venv || true
. .venv/bin/activate
python -m pip install -U pip setuptools wheel
python -m pip install -e ".[dev]"
pytest tests/utils -q
