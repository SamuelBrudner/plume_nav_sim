# Seeding Test Suite Status

**Date**: 2025-09-29  
**Status**: ✅ **Semantic Model Established** → Ready for Implementation

## Summary

The seeding test suite has been updated to reflect a **self-consistent, fully specified semantic model** documented in `SEEDING_SEMANTIC_MODEL.md`. Tests now correctly specify expected behavior according to best practices.

## Test Suite Alignment

### ✅ Tests Now Correctly Specify

1. **`None` is VALID** → requests random seed generation (Gymnasium standard)
2. **Negative integers are INVALID** → no silent normalization/modulo
3. **Floats are INVALID** → no silent truncation/coercion
4. **Non-negative integers [0, SEED_MAX_VALUE] are VALID** → identity validation
5. **Error messages must contain keywords**: "type", "integer", "range", "negative", etc.

### 📊 Current Test Results (Against Old Implementation)

```
TestSeedValidation: 5 FAILED, 23 PASSED
```

#### Expected Failures (Implementation Needs Update)
- ❌ `test_validate_seed_with_invalid_inputs[-1]` → Current impl normalizes, should reject
- ❌ `test_validate_seed_with_invalid_inputs[-100]` → Current impl normalizes, should reject  
- ❌ `test_validate_seed_with_invalid_inputs[3.14]` → Current impl coerces, should reject
- ❌ `test_validate_seed_with_invalid_inputs[0.0]` → Current impl coerces, should reject
- ❌ `test_validate_seed_with_invalid_inputs[42]` → String "42" in INVALID_SEEDS (test bug, will fix)

#### Passing Tests (Already Correct)
- ✅ All valid integer seeds pass
- ✅ `None` passes in both strict modes
- ✅ Out-of-range integers rejected
- ✅ String seeds rejected

## Implementation Gap Analysis

### What Needs to Change in `plume_nav_sim/utils/seeding.py`

#### 1. Remove Negative Normalization (lines 122-124)
```python
# CURRENT (WRONG):
if seed < 0:
    seed = seed % (SEED_MAX_VALUE + 1)  # ❌ Silent transformation
    
# SHOULD BE:
if seed < 0:
    return (False, None, "Seed must be non-negative...")  # ✅ Explicit rejection
```

#### 2. Reject Float Coercion (lines 110-113)
```python
# CURRENT (WRONG):
try:
    seed = int(seed)  # ❌ Truncates 3.14 → 3
    
# SHOULD BE:
if isinstance(seed, float):
    return (False, None, "Seed must be integer type...")  # ✅ Explicit rejection
```

#### 3. Improve Error Messages
Add keywords "type", "integer", "negative" to match test expectations.

## Next Steps

### Phase 1: Implementation (TDD)
1. ✅ **DONE**: Establish semantic model (`SEEDING_SEMANTIC_MODEL.md`)
2. ✅ **DONE**: Update tests to be self-consistent
3. 🔄 **NEXT**: Implement `validate_seed()` changes to pass tests
4. **TODO**: Update dependent functions (`create_seeded_rng`, etc.)
5. **TODO**: Run full test suite and fix cascading issues

### Phase 2: Verification
1. Confirm all `TestSeedValidation` tests pass
2. Run full `test_seeding.py` suite
3. Check for breaking changes in dependent code
4. Update documentation/docstrings

## Test Coverage Completeness

### Covered Scenarios ✅
- Valid integers (including boundaries)
- None (random seed request)
- Negative integers
- Out-of-range integers
- Floats
- Strings
- Both strict/non-strict modes

### Not Yet Covered ⚠️
- numpy.integer types (strict mode behavior)
- List/dict/other types
- Integer overflow edge cases
- Unicode string seeds

### Future Enhancements 📋
- Add explicit numpy.integer tests
- Test with extremely large seeds (> 2^32)
- Performance benchmarks for validation
- Thread-safety stress tests

## Dependency Analysis

### Functions That Call `validate_seed()`
1. `create_seeded_rng()` → Will need update if it expects normalization
2. `SeedManager.seed()` → Should work unchanged (just validates)
3. Any custom user code → **Breaking change** (needs migration guide)

### Breaking Changes
⚠️ **API Change**: Negative seeds and floats will now be **rejected** instead of normalized/coerced.

**Migration Guide Needed**:
```python
# OLD (silently normalized):
validate_seed(-1)  # → (True, 4294967295, "")

# NEW (explicit rejection):
validate_seed(-1)  # → (False, None, "Seed must be non-negative...")

# Users must explicitly handle:
seed = abs(seed)  # or
seed = seed % SEED_MAX_VALUE
```

## References

- Semantic Model: `SEEDING_SEMANTIC_MODEL.md`
- Implementation: `plume_nav_sim/utils/seeding.py`
- Tests: `tests/test_seeding.py`
- Related: Gymnasium docs, NumPy random, FAIR principles
