# Makefile for the plume_nav_sim project

.PHONY: test

test:
	@echo "Running the test suite..."
	@conda run --prefix src/backend/conda_env pytest src/backend/tests/
