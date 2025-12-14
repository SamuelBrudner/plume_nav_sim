SHELL := /bin/bash
MAKEFLAGS += --warn-undefined-variables

SETUP_SCRIPT := ./setup_env.sh
ENV_NAME ?= plume_nav_sim
PYTHON_VERSION ?= 3.10
DEV ?= 0

ifeq ($(DEV),1)
DEV_FLAG := --dev
else
DEV_FLAG :=
endif

.PHONY: setup setup-dev maintain lint

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
	conda run -n $(ENV_NAME) python -m pip install "PySide6>=6.7.0"

# Run the debugger from source without packaging (uses PYTHONPATH)
debugger:
	PYTHONPATH=src conda run -n $(ENV_NAME) python -m plume_nav_debugger.app

.PHONY: nb-clean
# Strip outputs/exec counts from notebooks in-place for clean commits
nb-clean:
	python src/backend/scripts/strip_notebook_outputs.py notebooks/*.ipynb

.PHONY: nb-render
# Render the stable capture exploration notebook to docs via nbconvert
nb-render:
	@mkdir -p src/backend/docs/notebooks
	conda run -n $(ENV_NAME) jupyter nbconvert --to html \
		--output-dir src/backend/docs/notebooks \
		notebooks/stable/capture_end_to_end.ipynb

# Lint the library exactly like CI (flake8)
lint:
	@echo "[lint] Running flake8 with CI-equivalent options"
	conda run -n $(ENV_NAME) flake8 src/backend/plume_nav_sim \
		--max-line-length=88 \
		--extend-ignore=E203,W503,E501 \
		--select=E,W,F,C,N \
		--max-complexity=10 \
		--per-file-ignores="src/backend/plume_nav_sim/__init__.py:F401,F403,F405,src/backend/plume_nav_sim/envs/base_env.py:C901,src/backend/plume_nav_sim/core/episode_manager.py:C901"
