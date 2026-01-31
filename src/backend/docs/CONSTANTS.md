Constants Guide

Scope

- YAML (conf/constants.yaml): User‑tunable identifiers and targets
  - package: name, version, environment_id
  - performance: tracking_enabled, timing targets (step, render, reset, generation), boundary enforcement
  - testing: default_seeds
- Python (plume_nav_sim/core/constants.py): Code‑level primitives and derived constants
  - Environment defaults: DEFAULT_GRID_SIZE, MIN_GRID_SIZE, DEFAULT_SOURCE_LOCATION, DEFAULT_PLUME_SIGMA, DEFAULT_GOAL_RADIUS, DEFAULT_MAX_STEPS
  - Action space: ACTION_* values, ACTION_SPACE_SIZE, MOVEMENT_VECTORS
  - Data types: FIELD_DTYPE, OBSERVATION_DTYPE, RGB_DTYPE
  - Rendering: marker colors/sizes, SUPPORTED_RENDER_MODES
  - Numeric limits: CONCENTRATION_RANGE, GAUSSIAN_PRECISION, DISTANCE_PRECISION, MIN/MAX_PLUME_SIGMA
  - Seeding bounds, component names, memory limits

Design Principles

- YAML is for deployment/configuration knobs users may tune without code changes.
- Python is for invariants and implementation details used directly by code paths.
- Python loads YAML and exposes typed constants so there is a single place to import from in code.

Mapping

- Mapped from YAML → Python (checked by validator):
  - package.name → PACKAGE_NAME
  - package.version → PACKAGE_VERSION
  - package.environment_id → ENVIRONMENT_ID
  - performance.*→ PERFORMANCE_* constants and BOUNDARY_ENFORCEMENT_PERFORMANCE_TARGET_MS
  - testing.default_seeds → DEFAULT_TEST_SEEDS

Drift Prevention

- Run the drift validator to ensure YAML and Python stay in sync:
  - CLI: plume-nav-constants-check
  - Python: python -m scripts.validate_constants_drift
  - Exits non‑zero if required keys are missing or values diverge.

When to add to YAML vs Python

- Add to YAML when:
  - The value is likely to change across environments or deployments
  - You want to expose a knob for ops/tests without a code change
- Keep in Python when:
  - The value is an implementation detail or invariant required by code
  - It is tightly coupled with types/enums or derived programmatically

Examples

- Good YAML additions: new performance timing thresholds, default seed sets, environment ID changes
- Good Python-only constants: additional Action enum values, dtype changes, new geometry helpers

Notes

- Avoid duplicating the same concept in multiple Python modules. Derive from core/constants when needed (e.g., logging component names).
- If a new YAML field is introduced, wire it into core/constants first and add a validator check.
