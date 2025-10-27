SHELL := /bin/bash
MAKEFLAGS += --warn-undefined-variables

SETUP_SCRIPT := ./setup_env.sh
ENV_NAME ?= plume-nav-sim
PYTHON_VERSION ?= 3.10
DEV ?= 0

ifeq ($(DEV),1)
DEV_FLAG := --dev
else
DEV_FLAG :=
endif

.PHONY: setup setup-dev maintain

setup:
	$(SETUP_SCRIPT) --name $(ENV_NAME) --python $(PYTHON_VERSION) $(DEV_FLAG)

setup-dev:
	$(SETUP_SCRIPT) --name $(ENV_NAME) --python $(PYTHON_VERSION) --dev

maintain:
	$(SETUP_SCRIPT) --name $(ENV_NAME) --python $(PYTHON_VERSION) --update $(DEV_FLAG)

.PHONY: dev-core install-qt debugger

# Editable install of backend/core in the conda env
dev-core:
	conda run -n $(ENV_NAME) pip install -e src/backend

# Install Qt toolkit for the debugger UI (PySide6)
install-qt:
	conda run -n $(ENV_NAME) python -m pip install "PySide6>=6.5"

# Run the debugger from source without packaging (uses PYTHONPATH)
debugger:
	PYTHONPATH=src conda run -n $(ENV_NAME) python -m plume_nav_debugger.app

.PHONY: nb-clean
# Strip outputs/exec counts from notebooks in-place for clean commits
nb-clean:
	python src/backend/scripts/strip_notebook_outputs.py notebooks/*.ipynb
