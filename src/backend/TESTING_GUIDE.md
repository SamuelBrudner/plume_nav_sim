# Testing Guide

This guide describes the active test surface in `plume_nav_sim`. It is intentionally short and repo-specific. When this file and the code disagree, follow the code and tests.

## Where Tests Live

- `tests/` at the repo root: repo-level smoke tests for Make targets, debugger wiring, and regression guardrails.
- `src/backend/tests/`: the supported backend suite.
- `src/backend/tests/contracts/`: public contracts and deprecation enforcement.
- `src/backend/tests/integration/`: cross-component behavior.
- `src/backend/tests/unit/`: focused behavior for policies, rewards, observations, and actions.
- `src/backend/tests/runner/`: runner utilities and episode execution helpers.
- `src/backend/tests/archived/`: historical suites kept for reference only.

## Canonical Commands

From the repo root:

```bash
make lint
make test
make test-debugger
```

From `src/backend/`:

```bash
python -m pytest tests --tb=short -q
python -m ruff check .
python -m mypy --strict --config-file mypy.ini --follow-imports=skip -p plume_nav_sim.registration
```

Packaging smoke:

```bash
python -m build
python -m twine check dist/*
```

## Markers in Regular Use

The backend pytest config defines many markers, but the high-signal ones for routine work are:

- `unit`
- `integration`
- `performance`
- `slow`
- `reproducibility`
- `api_compliance`

Examples:

```bash
python -m pytest tests -m unit -q
python -m pytest tests -m integration -q
python -m pytest tests -m "not performance and not slow" -q
```

## Test Writing Rules

- Test public behavior, not internal structure.
- Prefer a single clear behavior per test.
- Use integration tests when the value comes from component interaction.
- Keep contract tests close to public APIs and remove stale claims when code changes.
- Add repo-level regression tests for stale docs or workflow drift when cleanup work removes dead references.

## Archived Tests

`src/backend/tests/conftest.py` excludes `src/backend/tests/archived` from normal collection.

Archived files are kept because they sometimes contain useful scenario ideas, not because they remain supported. Direct invocation of archived files should skip explicitly rather than fail during import.

If you need behavior coverage for something only shown in an archived file, extract the scenario into an active unit, integration, or contract test instead of re-enabling the archived suite.
