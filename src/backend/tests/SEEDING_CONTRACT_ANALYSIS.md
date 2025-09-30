# Seeding System Contract Analysis

**Date**: 2025-09-29  
**Status**: ⚠️ **Inconsistencies Found** → Needs Specification

## Purpose

Analyze inter-component contracts between seeding validation (`validate_seed`) and its consumers to ensure consistency with the semantic model.

---

## Component Interaction Map

```
validate_seed(seed, strict_mode) → (bool, Optional[int], str)
           ↓
    ┌──────┴──────┬───────────────┬─────────────────┐
    ↓             ↓               ↓                 ↓
create_seeded_rng()  SeedManager.__init__()  SeedManager.seed()  Tests
    ↓             
gymnasium.utils.seeding.np_random()
    ↓
PlumeSearchEnv.reset(seed=...)
```

---

## Contract #1: `create_seeded_rng()` ↔ `validate_seed()`

### Current Implementation

```python
def create_seeded_rng(seed: Optional[int] = None, validate_input: bool = True):
    if validate_input and seed is not None:  # ⚠️ Only validates if seed is not None
        is_valid, normalized_seed, error_message = validate_seed(
            seed, strict_mode=False
        )
        if not is_valid:
            raise ValidationError(...)
        seed = normalized_seed  # ⚠️ Expects normalization
```

### Issues Found

| Issue | Severity | Description |
|-------|----------|-------------|
| ❌ **Assumes normalization** | HIGH | Line 186: `seed = normalized_seed` expects transformation, but semantic model says **validation only** |
| ⚠️ **Skips None validation** | MEDIUM | Line 171: `if seed is not None` skips validation, but should verify `None` is valid |
| ⚠️ **Silent None handling** | LOW | Lines 190-193: Redundant if/else (both branches identical) |

### Contract Specification Needed

**Question**: When `validate_seed(None)` returns `(True, None, "")`, should `create_seeded_rng()`:
1. ✅ **Accept it** and pass `None` to gymnasium (our semantic model)
2. ❌ **Replace it** with random seed first (current unclear behavior)

**Recommendation**: Clarify that `None` is passed through to `gymnasium.utils.seeding.np_random(None)`, which generates random seed.

---

## Contract #2: `SeedManager.__init__()` ↔ `validate_seed()`

### Current Implementation

```python
def __init__(self, default_seed: Optional[int] = None, ...):
    if default_seed is not None:
        is_valid, normalized_seed, error_message = validate_seed(
            default_seed, strict_mode=True
        )
        if not is_valid:
            raise ValidationError(...)
        self.default_seed = normalized_seed  # ⚠️ Expects normalization
    else:
        self.default_seed = default_seed  # Allows None
```

### Issues Found

| Issue | Severity | Description |
|-------|----------|-------------|
| ❌ **Inconsistent None handling** | HIGH | `None` is allowed as default_seed but never validated |
| ❌ **Assumes normalization** | HIGH | Line 1025: `self.default_seed = normalized_seed` expects transformation |
| ⚠️ **Strict mode inconsistency** | MEDIUM | Uses `strict_mode=True` but doesn't document why |

### Ambiguity

**What does `default_seed=None` mean?**
- **Option A**: "No default, always require explicit seed" → But then `seed()` method allows `None`
- **Option B**: "Generate random for each call" → But not documented

**Current behavior**: Lines 1077 shows `effective_seed = seed if seed is not None else self.default_seed`, so `default_seed=None` means "no default, user must provide or gets random".

### Contract Specification Needed

**Clarify**:
1. `default_seed=None` means "no default, use None when no seed provided"
2. When `SeedManager.seed(seed=None, ...)` is called → uses `default_seed` if set, else `None`
3. `None` ultimately triggers random seed generation in `create_seeded_rng()`

---

## Contract #3: `SeedManager.seed()` ↔ `validate_seed()`

### Current Implementation

```python
def seed(self, seed: Optional[int] = None, context_id: Optional[str] = None):
    effective_seed = seed if seed is not None else self.default_seed
    
    if self.enable_validation and effective_seed is not None:  # ⚠️
        is_valid, validated_seed, error_message = validate_seed(
            effective_seed, strict_mode=True
        )
        if not is_valid:
            raise ValidationError(...)
        effective_seed = validated_seed  # ⚠️ Expects normalization
    
    np_random, seed_used = create_seeded_rng(
        effective_seed, validate_input=False  # ⚠️ Skips re-validation
    )
```

### Issues Found

| Issue | Severity | Description |
|-------|----------|-------------|
| ❌ **Assumes normalization** | HIGH | Line 1090: `effective_seed = validated_seed` expects transformation |
| ⚠️ **Skips None validation** | MEDIUM | Line 1079: `if effective_seed is not None` skips `None` validation |
| ✅ **Avoids double validation** | GOOD | Passes `validate_input=False` to avoid redundant check |
| ⚠️ **Inconsistent strict mode** | LOW | Uses `strict_mode=True` while `create_seeded_rng` uses `strict_mode=False` |

### Contract Specification Needed

**Decision Required**: What's the contract between `SeedManager` and `validate_seed`?

1. **Normalization expectation**: Should `SeedManager` expect `validated_seed == effective_seed` (identity)?
2. **None handling**: Should `effective_seed=None` be validated or skip validation?
3. **Strict mode policy**: When to use `strict_mode=True` vs `False`?

---

## Contract #4: `gymnasium.utils.seeding.np_random()` ↔ Our System

### Gymnasium Contract (External)

```python
# From Gymnasium documentation:
def np_random(seed: Optional[int] = None) -> Tuple[Generator, int]:
    """
    Args:
        seed: If None, generates random seed. Otherwise uses provided seed.
    
    Returns:
        (generator, seed_used): Generator and actual seed used (generated if None)
    """
```

### Our Assumptions

| Assumption | Valid? | Notes |
|------------|--------|-------|
| `None` is valid input | ✅ Yes | Gymnasium explicitly supports this |
| Negative seeds accepted | ❓ Unknown | Gymnasium doesn't specify, likely accepts |
| Floats accepted | ❓ Unknown | Gymnasium doesn't specify |
| Return value always int | ✅ Yes | Documented behavior |

### Specification Gap

**We need to decide**: Should our validation be **more strict** than Gymnasium's?
- **Yes (current model)**: Reject negatives/floats **before** passing to Gymnasium
- **No (permissive)**: Pass everything through, let Gymnasium handle it

**Recommendation**: **Yes, be more strict**. Rationale:
1. Scientific reproducibility requires explicit semantics
2. Prevents platform-dependent behavior (Gymnasium may differ)
3. Aligns with "fail loud and fast" principle

---

## Contract #5: Normalization Expectation Throughout

### Critical Inconsistency

**ALL** consumers of `validate_seed()` currently expect:
```python
is_valid, normalized_seed, error_message = validate_seed(seed)
# EXPECT: normalized_seed MAY differ from seed (transformation occurred)
# CURRENT: Lines do `seed = normalized_seed` assuming potential change
```

But our semantic model says:
```python
# SEMANTIC MODEL: normalized_seed == seed (identity, no transformation)
```

### Impact Assessment

**Breaking Change**: If we remove normalization logic:

| Component | Current Expectation | Impact if Changed | Mitigation |
|-----------|---------------------|-------------------|------------|
| `create_seeded_rng()` | May transform | None (still works) | Just returns same value |
| `SeedManager.__init__()` | May transform | None (still works) | Just returns same value |
| `SeedManager.seed()` | May transform | None (still works) | Just returns same value |
| External callers | May transform | ⚠️ **Breaking** | Need migration guide |

**Good news**: Since transformation is one-way assignment (`seed = normalized_seed`), removing transformation is **backward compatible** for internal code!

---

## Contract #6: Error Message Expectations

### Current Contract (Implicit)

Consumers expect error messages but don't specify format:
```python
if not is_valid:
    raise ValidationError(message=f"Invalid seed: {error_message}", ...)
```

### Issues

| Issue | Severity | Description |
|-------|----------|-------------|
| ✅ **Flexible format** | GOOD | Consumers just wrap the message, don't parse it |
| ⚠️ **No structured errors** | MEDIUM | Could benefit from error codes for programmatic handling |
| ✅ **User-friendly** | GOOD | Messages passed through to user |

### Specification Status

✅ **Adequate**: Current contract is loose enough that improving error messages won't break anything.

---

## Summary of Inconsistencies

### 🔴 Critical (Must Fix)

1. **Normalization assumption throughout** - All code assumes `normalized_seed ≠ seed` possible
   - **Fix**: None needed! Identity transformation is backward compatible
   
2. **None validation skipped** - `if seed is not None` bypasses validation
   - **Fix**: Remove guard, always call `validate_seed()` (it handles `None`)

### 🟡 Moderate (Should Clarify)

3. **Strict mode inconsistency** - No clear policy on when to use `strict_mode=True` vs `False`
   - **Clarify**: Document in semantic model

4. **`default_seed=None` semantics** - Unclear what this means
   - **Clarify**: Document "no default, use None when no seed provided"

### 🟢 Minor (Nice to Have)

5. **Double validation prevention** - Some paths validate twice
   - **Optimize**: Already handled with `validate_input=False` flag

---

## Required Specification Additions

### Add to `SEEDING_SEMANTIC_MODEL.md`:

#### Section: Integration Contracts

**`create_seeded_rng(seed, validate_input)`**:
- MUST call `validate_seed(seed, strict_mode=False)` if `validate_input=True`
- MUST handle `seed=None` as "generate random" (pass to Gymnasium)
- MUST NOT assume `validated_seed ≠ seed` (identity transformation)
- Return: `(Generator, int)` where int is **actual** seed used

**`SeedManager.__init__(default_seed, ...)`**:
- MUST validate `default_seed` if not `None`
- `default_seed=None` means "no default, use None when no seed provided"
- MUST use `strict_mode=True` for initialization (one-time check)

**`SeedManager.seed(seed, context_id)`**:
- MUST validate `effective_seed` if `enable_validation=True`
- MUST handle `seed=None` → uses `default_seed` → may still be `None` → random
- MUST use `strict_mode=True` (stricter for explicit operations)
- MUST NOT double-validate (pass `validate_input=False` to `create_seeded_rng`)

#### Section: Strict Mode Policy

**When to use `strict_mode=True`**:
- Explicit user-provided seeds (`SeedManager.seed()`)
- Configuration initialization (`SeedManager.__init__()`)
- API boundaries where type safety matters

**When to use `strict_mode=False`**:
- Internal operations where numpy.integer acceptable
- Backward compatibility layers
- Default behavior for `validate_seed()`

---

## Action Items

### Phase 1: Specification (Now)
- [x] Document normalization = identity transformation
- [ ] Add "Integration Contracts" section to semantic model
- [ ] Add "Strict Mode Policy" section to semantic model
- [ ] Document `None` handling throughout call chain

### Phase 2: Implementation (Next)
- [ ] Remove negative normalization in `validate_seed()`
- [ ] Remove float coercion in `validate_seed()`
- [ ] Add `None` to test suite as valid case
- [ ] Remove `if seed is not None` guards before validation calls
- [ ] Update error messages to include required keywords

### Phase 3: Verification (Final)
- [ ] Run full test suite
- [ ] Check no behavioral changes from identity transformation
- [ ] Verify error messages meet specification
- [ ] Update docstrings with contracts

---

## Conclusion

**Status**: ⚠️ **Contracts are mostly consistent but underspecified**

**Key Findings**:
1. ✅ **Good news**: Identity transformation is backward compatible
2. ✅ **No breaking changes** needed in consumer code
3. ⚠️ **Need specification** for strict mode policy and None handling
4. ⚠️ **Need tests** to verify contract behavior

**Next Step**: Update `SEEDING_SEMANTIC_MODEL.md` with integration contracts before implementing.
