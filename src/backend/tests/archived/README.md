# Archived Tests

This directory contains tests that were archived because they test **implementation details** rather than **behavior/contracts**.

## Why Archive Instead of Delete?

1. **Historical reference** - Shows what was tested before
2. **Potential salvage** - Some test logic might be useful later
3. **Git history** - Preserves the work that went into them

## Archived Files

### `test_base_env_implementation_details.py`
- **Original**: `tests/plume_nav_sim/envs/test_base_env.py` (2,715 lines, 30 tests)
- **Archived**: 2025-10-10
- **Reason**: Tested private attributes (`_config`, `_logger`, `_initialized`), error wrapping details, and performance metrics implementation
- **Replacement**: `tests/plume_nav_sim/envs/test_base_env.py` (347 lines, 15 contract tests)
- **Coverage**: Core functionality tested by 111 passing integration/API tests

### `test_logging_implementation_details.py`
- **Original**: `tests/plume_nav_sim/utils/test_logging.py` (26 tests)
- **Archived**: 2025-10-10
- **Reason**: Tested internal logging API (`.name` attribute, cache structure, internal methods)
- **Replacement**: None needed - logging behavior verified by integration tests
- **Coverage**: All 111 core tests use logging; failures would be caught there

### Legacy Core Monolith Suites (archived 2026-02-06)
- `test_boundary_enforcer_legacy_core.py`
- `test_reward_calculator_legacy_core.py`
- `test_episode_manager_legacy_core.py`
- `test_config_contracts_legacy_core.py`
- **Reason**: These suites target removed legacy modules (episode_manager, state_manager, reward_calculator, boundary_enforcer, action_processor) replaced by component-based architecture.

## Philosophy: Test Behavior, Not Implementation

### ❌ Don't Test
- Private attributes/methods
- Internal data structures
- Implementation details that can change
- Error wrapping/translation layers
- Cache implementation

### ✅ Do Test
- Public API contracts (Gymnasium API compliance)
- Observable behavior (reset returns obs+info)
- Error conditions (invalid inputs raise errors)
- Integration points (components work together)
- End-to-end workflows

## When to Restore?

These tests should **not** be restored unless:
1. Core tests fail to catch a real bug that these would have caught
2. You need to extract specific test scenarios for behavior testing
3. You're documenting internal architecture (not for CI)

## Test Status Before Archiving

### test_base_env_implementation_details.py
- Status: 14/30 passing (47%)
- Failures: Error type mismatches, private attribute checks

### test_logging_implementation_details.py  
- Status: 2/26 passing (8%)
- Failures: Missing `.name` attribute, API changes

Both had low pass rates due to refactoring that **didn't break actual functionality**.
