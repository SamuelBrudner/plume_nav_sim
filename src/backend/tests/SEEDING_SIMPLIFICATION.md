# Seeding API Simplification: Eliminating `strict_mode`

**Date**: 2025-09-29  
**Status**: Approved for Implementation  
**Principle**: YAGNI (You Aren't Gonna Need It) + Minimal Design

## Problem

The `strict_mode` parameter in `validate_seed(seed, strict_mode=False)` adds API complexity with **no useful semantic distinction**.

### Current Behavior Analysis

```python
# strict_mode=False (current default)
validate_seed(3.14, strict_mode=False)
# → Attempts int(3.14) → returns (True, 3, "")  ❌ Silent truncation!

validate_seed(np.int64(42), strict_mode=False)
# → Converts to int → returns (True, 42, "")  ✅ Good

# strict_mode=True
validate_seed(3.14, strict_mode=True)
# → Returns (False, None, "invalid seed (strict mode: got float)")  ✅ Good

validate_seed(np.int64(42), strict_mode=True)
# → Converts to int → returns (True, 42, "")  ✅ Good (same as False!)
```

### The Problem

1. **Both modes convert numpy.integer** → No difference where it matters
2. **Non-strict mode attempts float coercion** → Violates our semantic model
3. **No real use case** → When would we ever want to reject numpy.integer?
4. **API complexity** → Extra parameter, 2x test cases, cognitive overhead

## Solution: Eliminate `strict_mode`

### New Signature
```python
def validate_seed(seed: Any) -> Tuple[bool, Optional[int], str]:
    """Validate seed with single, consistent behavior."""
```

### New Behavior (Unified)
- ✅ Accept `None` → pass through
- ✅ Accept native `int` → pass through  
- ✅ Accept `numpy.integer` → convert to native `int`
- ❌ Reject `float` → no truncation (fail loud)
- ❌ Reject `string` → no conversion (fail loud)
- ❌ Reject negative `int` → no normalization (fail loud)

## Benefits

### 1. **Simpler API**
```python
# BEFORE: Two parameters, unclear when to use which
validate_seed(seed, strict_mode=False)  # When?
validate_seed(seed, strict_mode=True)   # When?

# AFTER: One parameter, obvious usage
validate_seed(seed)  # Always
```

### 2. **Fewer Tests**
```python
# BEFORE: Test both modes
@pytest.mark.parametrize("strict_mode", [True, False])
def test_validate_seed_with_valid_inputs(seed, strict_mode):
    ...

# AFTER: Single mode
def test_validate_seed_with_valid_inputs(seed):
    ...
```

**Test reduction**: ~18 test cases → 9 test cases (50% fewer)

### 3. **Clearer Semantics**
No confusion about:
- When to use strict_mode?
- What's the difference?
- Why do both modes accept numpy.integer?

### 4. **Easier Documentation**
```python
# BEFORE
Args:
    seed: Seed value
    strict_mode: Enable strict type checking with additional validation 
                requirements (default: False). In strict mode, numpy integers
                are still accepted but... [confusing explanation]

# AFTER
Args:
    seed: Seed value (int, numpy.integer, or None for random)
```

### 5. **Code Reduction**

**Before** (lines 98-113):
```python
if not isinstance(seed, tuple(VALID_SEED_TYPES)):
    if strict_mode:
        return (False, None, VALIDATION_ERROR_MESSAGES["invalid_seed"] + f" (strict mode: got {type(seed).__name__})")
    else:
        try:
            seed = int(seed)  # ❌ Bad: silent coercion
        except (ValueError, TypeError, OverflowError):
            return (False, None, VALIDATION_ERROR_MESSAGES["invalid_seed"])
```

**After** (6 lines):
```python
if not isinstance(seed, tuple(VALID_SEED_TYPES)):
    return (
        False, None,
        f"Seed must be integer type, got {type(seed).__name__}"
    )
```

**Code reduction**: ~15 lines → 6 lines (60% fewer)

## Migration Path

### Internal Code (Consumers)

**No breaking changes needed!** All call sites just drop the parameter:

```python
# BEFORE
validate_seed(seed, strict_mode=False)
validate_seed(seed, strict_mode=True)

# AFTER
validate_seed(seed)
```

All 9 internal call sites in `seeding.py`:
- Line 173: `validate_seed(seed, strict_mode=False)` → `validate_seed(seed)`
- Line 1016: `validate_seed(default_seed, strict_mode=True)` → `validate_seed(default_seed)`
- Line 1081: `validate_seed(effective_seed, strict_mode=True)` → `validate_seed(effective_seed)`
- Line 1174: `validate_seed(base_seed, strict_mode=True)` → `validate_seed(base_seed)`
- Line 1256: `validate_seed(test_seed, strict_mode=True)` → `validate_seed(test_seed)`
- Line 1667: `validate_seed(episode_seed, strict_mode=self.strict_validation)` → `validate_seed(episode_seed)`

### External Code (If Any)

**Deprecation warning** (if we're cautious):
```python
def validate_seed(seed: Any, strict_mode: Optional[bool] = None) -> Tuple[bool, Optional[int], str]:
    if strict_mode is not None:
        warnings.warn(
            "strict_mode parameter is deprecated and ignored. "
            "validate_seed() now has single, consistent behavior.",
            DeprecationWarning,
            stacklevel=2
        )
    # ... proceed with unified validation
```

But likely **not needed** - `strict_mode` was internal implementation detail.

## Testing Impact

### Test Updates Required

1. **Remove `strict_mode` from test parametrization**:
   ```python
   # BEFORE
   @pytest.mark.parametrize("strict_mode", [True, False])
   def test_validate_seed_with_valid_integer_inputs(self, seed, strict_mode):
   
   # AFTER
   def test_validate_seed_with_valid_integer_inputs(self, seed):
   ```

2. **Remove separate None test with strict_mode**:
   ```python
   # BEFORE
   @pytest.mark.parametrize("strict_mode", [True, False])
   def test_validate_seed_with_none_input(self, strict_mode):
   
   # AFTER
   def test_validate_seed_with_none_input(self):
   ```

3. **Update test counts**:
   - `test_validate_seed_with_valid_integer_inputs`: 18 tests → 9 tests
   - `test_validate_seed_with_none_input`: 2 tests → 1 test
   - **Total reduction**: 10 tests eliminated

## Implementation Checklist

### Phase 1: Update Semantic Model ✅
- [x] Remove `strict_mode` from signature
- [x] Document unified behavior
- [x] Update integration contracts
- [x] Remove mode-dependent test classification

### Phase 2: Update Tests
- [ ] Remove `strict_mode` parametrization from all tests
- [ ] Update test counts and expectations
- [ ] Ensure numpy.integer tests still present

### Phase 3: Update Implementation
- [ ] Remove `strict_mode` parameter from `validate_seed()`
- [ ] Remove if/else logic for strict mode
- [ ] Reject floats immediately (no `int()` attempt)
- [ ] Update all 9 call sites to remove parameter

### Phase 4: Update Consumers
- [ ] Remove `strict_validation` attribute from `ReproducibilityTracker` (line 1667)
- [ ] Update docstrings throughout
- [ ] Verify no external dependencies on `strict_mode`

### Phase 5: Verification
- [ ] All validation tests pass
- [ ] All seeding tests pass
- [ ] No references to `strict_mode` remain
- [ ] Documentation updated

## Conclusion

Eliminating `strict_mode` is a **pure win**:
- ✅ Simpler API (1 param vs 2)
- ✅ Clearer semantics (no mode confusion)
- ✅ Fewer tests (50% reduction)
- ✅ Less code (60% reduction in validation logic)
- ✅ No breaking changes (backward compatible)
- ✅ Aligns with YAGNI principle

**Recommendation**: Proceed with elimination as part of validation refactor.
