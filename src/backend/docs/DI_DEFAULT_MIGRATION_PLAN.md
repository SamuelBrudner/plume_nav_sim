# DI Default Migration Plan and Legacy Deprecation

This document outlines the plan to make the component-based (DI) environment the default for
`ENV_ID`, deprecate the legacy `PlumeSearchEnv`, and remove legacy code without cruft.

## Objective

- Make the component-based environment the default for `ENV_ID`.
- Deprecate and remove the legacy monolithic environment safely.
- Keep migration predictable for users and tests.

## Rationale

- DI provides composability, clearer contracts, and better testing.
- Unit and integration tests validate component behavior more directly.
- Parity test confirms equivalent end behavior for default configuration.

## Current Status (Baseline)

- DI env id is available: `COMPONENT_ENV_ID` resolves to
  `plume_nav_sim.envs.factory:create_component_environment` via the registration module
  at src/backend/plume_nav_sim/registration/register.py:51.
- Registration mapping for DI is implemented: legacy kwargs are converted for the DI factory
  (`source_location` → `goal_location`, `plume_params.sigma` → `plume_sigma`). Unknown/private
  kwargs are dropped. See `_convert_kwargs_for_component_env()` in
  src/backend/plume_nav_sim/registration/register.py:332.
- Helper is available: `ensure_component_env_registered()` registers the DI env id if missing.
- Legacy remains the default entry point for `ENV_ID`. There is no env‑var switch wired into
  the code yet; any `PLUMENAV_DEFAULT=components` mention in examples is aspirational and does
  not affect behavior today.
- Docs and examples recommend DI and include runnable DI examples.
- Tests:
  - Registration suite: green.
  - Unit suites (actions/observations/rewards): green; abstract contract bases are not collected
    directly; concrete tests are collected via `__test__=True`.
  - DI vs legacy parity integration test present at
    src/backend/tests/integration/test_di_legacy_parity.py:1.
  - Legacy dict‑only observation assumptions relaxed where needed.

## Timeline

### Phase 0 (Done)

- Land DI env id + mapping, helper, docs, and DI mapping/permutation tests.
- Add parity test and deprecation warnings.

### Phase 1 (Next minor)

- Keep legacy as default; introduce DeprecationWarning on legacy implicit usage.
- Update examples/docs to recommend DI; optional env‑var opt‑in wiring (if implemented).
- Relax dict‑only assumptions where they cause failures.

### Phase 2 (Major)

- Flip default: `ENV_ID` routes to DI entry point by default.
- (Optional) Add short-lived opt‑back: `PLUMENAV_DEFAULT=legacy` for one transitional release.
- Provide a minimal compatibility shim for import stability (or remove entirely; see “Removal”).

### Phase 3 (Following minor/major)

- Remove legacy defaults and toggles; remove shim if added.
- Clean docs/examples accordingly.

## Actions to Flip Default and Remove Legacy

1. Flip default to DI (new PR):
   - In registration, choose `COMPONENT_ENTRY_POINT` when `effective_env_id == ENV_ID`.
   - Avoid env‑var gates for the permanent flip; keep behavior deterministic.
   - Preserve the dedicated `COMPONENT_ENV_ID`.

2. Update tests to DI explicitly:
   - Replace direct `PlumeSearchEnv` instantiation with DI factory or DI env id usage.
   - Centralize DI environment construction via fixtures in `tests/conftest.py`.
   - Observation semantics: prefer DI arrays (shape, range, finiteness). Only use dict wrappers
     in tests that require them (as a last resort, and in test code only).

3. Remove or shim legacy:
   - Low‑cruft shim: expose `PlumeSearchEnv` that internally builds a DI env with default components
     (classless façade) to preserve imports temporarily.
   - No‑cruft option: remove legacy module and update imports throughout tests and examples to DI.

4. Docs/Examples:
   - Promote DI usage exclusively; maintain a short “Legacy” note for one release, then remove.

## Test Strategy

- Keep the registration suite green as a gate.
- Maintain unit contract tests for all implementations (actions, observations, rewards):
  - Concrete classes inherit the abstract contract suites and set `__test__=True`.
  - Abstract bases set `__test__=False` to avoid direct collection.
- Parity integration test confirms end-behavior match for default configuration.
- Relax dict observation assumptions where needed; prefer array checks.

## Rollout Safeguards

- Optional feature flags (if implemented) can allow opt‑in/out during migration. Until then,
  use explicit `env_id` selection to avoid ambiguity.
- Suppress deprecation warnings in CI if needed (policy‑dependent).

## References

- DI factory: `plume_nav_sim.envs.factory:create_component_environment`
- DI env id: `COMPONENT_ENV_ID`
- Helper: `ensure_component_env_registered()`
- Parity test: `tests/integration/test_di_legacy_parity.py`
- Example: `examples/component_di_usage.py`
