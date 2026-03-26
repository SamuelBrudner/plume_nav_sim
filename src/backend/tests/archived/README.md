# Archived Tests

This directory contains tests kept for historical reference after the repo moved away from a larger legacy core and implementation-detail-heavy test style.

## Current Policy

- Normal backend collection ignores `src/backend/tests/archived`.
- Directly targeting these files should skip cleanly rather than fail during import.
- These tests are not part of the supported CI surface.

## Archived Files

### Implementation-detail suites (archived 2025-10-10)
- These exercised private attributes, error-wrapping details, and internal logging structure.
- The active replacement for base environment behavior is [src/backend/tests/plume_nav_sim/envs/test_base_env.py](/Users/samuelbrudner/Documents/GitHub/plume_nav_sim/src/backend/tests/plume_nav_sim/envs/test_base_env.py).
- Logging behavior is now covered indirectly by active integration and runner tests rather than dedicated internal-API assertions.

### Legacy core monolith suites (archived 2026-02-06)
- `test_boundary_enforcer_legacy_core.py`
- `test_reward_calculator_legacy_core.py`
- `test_episode_manager_legacy_core.py`
- `test_config_contracts_legacy_core.py`
- These files target removed modules such as `episode_manager`, `state_manager`, `reward_calculator`, and `boundary_enforcer`.
- Their supported replacements live in the active contract, integration, and unit suites under `src/backend/tests/`.

## Why Keep Them?

- Historical context for refactors already completed.
- A source of edge-case scenarios that can be salvaged into behavior tests later.
- A record of what was intentionally removed from the active surface.

## What to Use Instead

- Public environment behavior: `src/backend/tests/plume_nav_sim/envs/`
- Component contracts and invariants: `src/backend/tests/contracts/`
- Cross-component behavior: `src/backend/tests/integration/`
- Policy, reward, observation, and runner behavior: `src/backend/tests/unit/` and `src/backend/tests/runner/`
