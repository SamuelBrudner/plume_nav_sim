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
