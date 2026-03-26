# Decision: Archive Implementation Detail Tests

**Date**: 2025-10-10  
**Decision**: Archive `test_base_env.py` and `test_logging.py` implementation tests  
**Status**: ✅ Completed

## Summary

Archived implementation-detail-heavy suites in favor of smaller behavior-oriented coverage. The active replacement surface now lives in:

- [src/backend/tests/plume_nav_sim/envs/test_base_env.py](/Users/samuelbrudner/Documents/GitHub/plume_nav_sim/src/backend/tests/plume_nav_sim/envs/test_base_env.py)
- `src/backend/tests/contracts/`
- `src/backend/tests/integration/`
- `src/backend/tests/unit/`

## Rationale

### Why Archive?

1. **High maintenance cost**: Tests broke during refactoring that didn't break functionality
2. **Testing wrong things**: Checked private attributes (`.name`, `._config`, `._logger`)
3. **False failures**: 40 failures from internal API changes, 0 from actual bugs
4. **Better coverage exists**: current contract, integration, and unit suites verify actual behavior

### What We Kept

Behavior-focused coverage:
- Gymnasium API compliance
- Basic configuration validation
- Observation/reward/info structure checks
- Integration tests for component assembly and runtime behavior

### What We Archived

**Implementation details**:
- ❌ Private attribute checks (`_initialized`, `_config`, `_logger`)
- ❌ Internal method call order
- ❌ Error wrapping/translation layers
- ❌ Cache implementation details
- ❌ Performance metric internal structure
- ❌ Logger `.name` attribute existence

## Impact Analysis

### Risks
- **Low**: core functionality remains covered by active suites
- **Mitigation**: active integration and contract tests catch public regressions

### Benefits
- **Faster CI**: Fewer tests to run
- **Easier refactoring**: Can change internals without breaking tests
- **Better signal**: Failures indicate real problems, not API changes
- **Clearer intent**: Tests document public contracts, not internals

## Current Handling

- `src/backend/tests/conftest.py` ignores the `archived/` directory during normal collection.
- Archived modules now skip explicitly at import time so direct invocation does not fail on removed legacy imports.
- These files remain reference material, not supported CI inputs.
