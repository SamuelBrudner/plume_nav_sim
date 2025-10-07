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

- DI env id: `COMPONENT_ENV_ID` → `plume_nav_sim.envs.factory:create_component_environment`.
- Registration mapping for DI: `source_location` → `goal_location`; unknown/private kwargs dropped.
- Opt‑in default toggle: `PLUMENAV_DEFAULT=components` (or `PLUMENAV_USE_COMPONENTS=1`).
- Helper: `ensure_component_env_registered()`.
- Deprecation warnings: legacy entry path and direct `PlumeSearchEnv` instantiation (suppress via
  `PLUMENAV_DEPRECATION_SILENCE=1`).
- Docs/examples updated to recommend DI and provide a runnable DI example.
- Tests updated:
  - Registration suite: green.
  - Unit suites (actions/observations/rewards): green; abstract contract bases are not collected
    directly; concrete tests are collected via `__test__=True`.
  - DI vs legacy parity integration test added.
  - Legacy dict-only observation assumptions relaxed in representative tests.

## Timeline

### Phase 0 (Done)
- Land DI env id + mapping, helper, docs, and DI mapping/permutation tests.
- Add parity test and deprecation warnings.

### Phase 1 (Next minor)
- Keep legacy as default; introduce DeprecationWarning on legacy implicit usage.
- Update examples/docs to recommend DI; provide env-var opt‑in.
- Relax dict-only assumptions where they cause failures.

### Phase 2 (Major)
- Flip default: `ENV_ID` routes to DI entry point by default.
- (Optional) Add short-lived opt‑back: `PLUMENAV_DEFAULT=legacy` for one transitional release.
- Provide a minimal compatibility shim for import stability (or remove entirely; see “Removal”).

### Phase 3 (Following minor/major)
- Remove legacy defaults and toggles; remove shim if added.
- Clean docs/examples accordingly.

## Actions to Flip Default and Remove Legacy

1. Flip default to DI (new PR):
   - `register_env()` uses DI entry point for `ENV_ID` by default (no env-var gate).
   - Preserve `COMPONENT_ENV_ID` behavior.

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

- Feature flags allow opt‑in/out during migration:
  - `PLUMENAV_DEFAULT=components` (opt‑in to DI)
  - (Optional) `PLUMENAV_DEFAULT=legacy` for one release after flipping default.
- Suppress warnings in CI: `PLUMENAV_DEPRECATION_SILENCE=1` while refactoring tests.

## References

- DI factory: `plume_nav_sim.envs.factory:create_component_environment`
- DI env id: `COMPONENT_ENV_ID`
- Helper: `ensure_component_env_registered()`
- Parity test: `tests/integration/test_di_legacy_parity.py`
- Example: `examples/component_di_usage.py`
