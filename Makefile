.DEFAULT_GOAL := help

.PHONY: help install test test-debugger lint demo demo-debugger clean

help: ## Print available targets
	@echo "Available targets:"
	@awk 'BEGIN {FS = ":.*## "}; /^[a-zA-Z0-9_.-]+:.*## / {printf "  %-14s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Editable install with dev extras
	pip install -e src/backend[dev]

test: ## Run backend tests
	cd src/backend && python -m pytest --tb=short -q

test-debugger: ## Run debugger tests
	python -m pytest tests/debugger --tb=short -q

lint: ## Run backend lint checks
	cd src/backend && python -m ruff check .

demo: ## Run a quick 50-step episode
	python -c "import plume_nav_sim as pns; env = pns.make_env(grid_size=(32,32), max_steps=50, render_mode='rgb_array'); obs, info = env.reset(seed=42); done = False; t = 0;\
while not done:\
    obs, r, term, trunc, info = env.step(env.action_space.sample()); t += 1; done = term or trunc;\
print(f'Episode complete: {t} steps'); env.close()"

demo-debugger: ## Launch the debugger GUI
	python -m plume_nav_debugger

clean: ## Remove build and cache artifacts
	rm -rf src/backend/dist src/backend/*.egg-info **/__pycache__
