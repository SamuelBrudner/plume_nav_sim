# Refactoring Plan: Contracts-First Approach

**Date:** 2025-09-30  
**Status:** Ready to Execute  
**Based On:** CONTRACTS.md, SEMANTIC_MODEL.md, TESTING_GUIDE.md

---

## 📊 Current State

**Test Suite Status:**
- ✅ **399 passing** (66%)
- ❌ **204 failing** (34%)
- ⚠️ **38 collection errors**
- 🚫 **6 render modules blocked** (import errors)

**Root Causes:**
1. ❌ API drift - `parameter_value` vs `invalid_value` inconsistency
2. ❌ Signature changes - `ComponentError` breaking changes
3. ❌ Missing constants - Moved/renamed without updating imports
4. ❌ Over-parametrization - 177 tests could be 75 (~60% waste)
5. ⚠️ Circular imports - Render modules failing

---

## 🎯 Phase 1: Quick Wins (Immediate - 2 hours)

### Priority: CRITICAL - Fixes ~40 test failures

### Task 1.1: Global Parameter Rename
**Affected:** ~30 test failures

**Files to Fix:**
```bash
# Find all remaining usages:
grep -r "invalid_value=" src/backend/plume_nav_sim/

# Fix these files:
- plume_nav_sim/registration/register.py (lines 702, 731)
- plume_nav_sim/registration/__init__.py (line 452)
- plume_nav_sim/core/boundary_enforcer.py (line 407)
- plume_nav_sim/utils/validation.py (multiple)
- plume_nav_sim/utils/seeding.py (multiple)
```

**Action:**
```bash
# Global find-replace in remaining files
find src/backend/plume_nav_sim -name "*.py" -type f \
  -exec sed -i '' 's/invalid_value=/parameter_value=/g' {} +
```

**Expected Gain:** +30 passing tests

---

### Task 1.2: Fix ComponentError Calls
**Affected:** 2 test failures

**Files:**
- `plume_nav_sim/plume/static_gaussian.py:98`
- `plume_nav_sim/plume/plume_model.py:1660`

**Current (WRONG):**
```python
raise ComponentError("message", severity=ErrorSeverity.HIGH)
```

**Fixed (CORRECT):**
```python
raise ComponentError(
    "message",
    component_name="PlumeModel",
    operation_name="validate",
)
```

**Expected Gain:** +2 passing tests

---

### Task 1.3: Restore Missing Constants
**Affected:** 5 test failures

**Add to `core/constants.py`:**
```python
# Action space
ACTION_SPACE_SIZE = 9  # 8 directions + stay

# Observation space  
CONCENTRATION_RANGE = (0.0, 1.0)  # (min, max)

# Reward defaults (if missing)
DEFAULT_GOAL_RADIUS = 5.0

# Caching
BOUNDARY_VALIDATION_CACHE_SIZE = 1000
```

**Expected Gain:** +5 passing tests

---

**Phase 1 Total:** ~37 tests fixed (2 hours work)

---

## 🔧 Phase 2: API Alignment (Short-term - 4 hours)

### Priority: HIGH - Fixes ~60 test failures

### Task 2.1: Make EpisodeManagerConfig Backward Compatible

**Problem:** Tests expect optional params, code requires them.

**Solution:** Add defaults
```python
@dataclass
class EpisodeManagerConfig:
    # REQUIRED
    max_steps: int
    
    # OPTIONAL (with defaults) - ADD THESE
    enable_performance_monitoring: bool = True
    enable_state_validation: bool = True
```

**Expected Gain:** +15 passing tests

---

### Task 2.2: Fix Function Signature Changes

**Functions to update:**

1. `create_coordinates()` - restore kwargs support
   ```python
   def create_coordinates(x: int, y: int) -> Coordinates:
       # Support both positional and kwargs for compatibility
       return Coordinates(x, y)
   ```

2. Remove `strict_mode` parameter completely
   ```python
   # Remove from:
   - validate_constant_consistency()
   - validate_base_environment_setup()
   ```

3. `EpisodeStatistics` - restore `session_id` parameter
   ```python
   @dataclass
   class EpisodeStatistics:
       ...
       session_id: Optional[str] = None  # Add back
   ```

**Expected Gain:** +20 passing tests

---

### Task 2.3: Fix Method Name Mismatches

**Classes to update:**

1. `PerformanceMetrics`:
   ```python
   # Add alias for backward compatibility
   def get_summary(self):
       return self.get_statistics()
   ```

2. `GridSize`:
   ```python
   def to_dict(self):
       return {"width": self.width, "height": self.height}
   ```

3. `EpisodeResult`:
   ```python
   def get_performance_metrics(self):
       # Return stored metrics
       return self.performance_metrics
   ```

**Expected Gain:** +10 passing tests

---

### Task 2.4: Fix ValidationResult Signature

**File:** `benchmarks/environment_performance.py:235`

**Problem:**
```python
ValidationResult(validated_object=...)  # Wrong parameter name
```

**Fix:** Check actual signature and use correct parameter

**Expected Gain:** +1 passing test

---

**Phase 2 Total:** ~46 tests fixed (4 hours work)

---

## 🎨 Phase 3: Render Module Debug (Medium-term - 3 hours)

### Priority: MEDIUM - Unblocks 30+ tests

### Task 3.1: Investigate Circular Import

**Error:** `ModuleNotFoundError: No module named 'logging'`

**This is bizarre** - `logging` is stdlib, shouldn't fail.

**Likely Cause:** Circular import causing Python's import system to fail

**Investigation Steps:**
1. Check import order in render/__init__.py
2. Look for `import logging as logging` patterns (can cause issues)
3. Check if any render module has variable named `logging`
4. Use `python -v` to trace import sequence

**Files to check:**
- `render/__init__.py`
- `render/base_renderer.py`
- `render/numpy_rgb.py`
- `render/matplotlib_viz.py`

**Expected Gain:** +30 passing tests (entire render suite)

---

## 📝 Phase 4: Test Suite Pruning (Long-term - 6 hours)

### Priority: LOW (but high value) - Reduces maintenance burden

### Task 4.1: Implement YAGNI Recommendations

**Target:** `test_seeding.py` (177 tests → 75 tests)

**Based on:** TEST_SUITE_YAGNI_ANALYSIS.md

**Actions:**
1. Remove parametric explosion in `TestSeedManager` (60 → 12 tests)
2. Consolidate `TestReproducibilityTracker` (40 → 8 tests)
3. Simplify `TestReproducibilityVerification` (30 → 8 tests)
4. Remove redundant hash algorithm tests (15 → 6 tests)

**Methodology:**
- Keep tests that test DIFFERENT behavior
- Remove tests that test SAME logic with different data
- Use property-based tests for parametric coverage

**Expected Gain:** 
- Faster CI (50% time reduction)
- Clearer test intentions
- Easier maintenance

---

### Task 4.2: Add Contract Tests

**Create:** `tests/contracts/`

**Add tests for:**
```python
test_validation_error_signature_stable()
test_component_error_signature_stable()
test_configuration_error_signature_stable()
test_reward_calculator_config_interface()
test_episode_manager_config_interface()
```

**Expected Gain:**
- Prevent future API drift
- Catch breaking changes in CI
- Serve as living documentation

---

## 📊 Success Metrics

### Target Outcomes

**After Phase 1+2 (6 hours):**
- ✅ ~480 passing tests (80%)
- ❌ ~120 failing tests (20%)
- 🚫 Render module still blocked

**After Phase 3 (9 hours):**
- ✅ ~510 passing tests (85%)
- ❌ ~90 failing tests (15%)
- ✅ Render module working

**After Phase 4 (15 hours):**
- ✅ ~450 passing tests (95%+ of retained tests)
- ❌ ~20 failing tests (edge cases)
- 🎯 Suite size: ~475 tests (down from 650)
- ⚡ CI time: 50% faster

---

## 🚀 Execution Order

### Recommended Sequence:

**Day 1 (Morning):**
1. Task 1.1 - Global parameter rename (30 min)
2. Task 1.2 - Fix ComponentError (30 min)
3. Task 1.3 - Restore constants (30 min)
4. Run tests, verify ~37 fixed ✅

**Day 1 (Afternoon):**
5. Task 2.1 - EpisodeManagerConfig (1 hour)
6. Task 2.2 - Function signatures (1.5 hours)
7. Task 2.3 - Method names (1 hour)
8. Task 2.4 - ValidationResult (30 min)
9. Run tests, verify ~83 total fixed ✅

**Day 2 (Morning):**
10. Task 3.1 - Debug render imports (3 hours)
11. Run full suite, verify ~113 total fixed ✅

**Day 2 (Afternoon) + Day 3:**
12. Task 4.1 - Prune test suite (4 hours)
13. Task 4.2 - Add contract tests (2 hours)
14. Final verification run ✅

---

## 🛡️ Safety Measures

### Before Starting:

1. **Create branch:** `git checkout -b fix/api-contracts`
2. **Baseline:** Run tests, save output
3. **Commit often:** After each task
4. **Track progress:** Update this doc with ✅/❌

### During Refactoring:

1. **Run tests after each file change**
2. **Never batch > 5 files without testing**
3. **Document unexpected issues**
4. **Keep CONTRACTS.md updated**

### After Each Phase:

1. **Run full test suite**
2. **Document pass rate**
3. **Commit with message:** `"Phase N complete: X tests fixed"`
4. **Review failures - expected or unexpected?**

---

## 📝 Progress Tracking

### Phase 1: Quick Wins
- [x] Task 1.1: Global parameter rename ✅
- [x] Task 1.2: Fix ComponentError ✅
- [x] Task 1.3: Restore constants ✅
- [ ] Verification run (in progress)
- [ ] **Target:** 399 → 436 passing (+37)

### Phase 2: API Alignment  
- [x] Task 2.1: EpisodeManagerConfig ✅
- [x] Task 2.2: Function signatures ✅
- [x] Task 2.3: Method names ✅
- [ ] Task 2.4: ValidationResult (skipping - not found in test errors)
- [ ] Verification run (in progress)
- [ ] **Target:** 407 → ~450 passing (+43)

### Phase 3: Render Fix
- [ ] Task 3.1: Debug imports
- [ ] Verification run
- [ ] **Target:** 482 → 512 passing (+30)

### Phase 4: Test Pruning
- [ ] Task 4.1: YAGNI implementation
- [ ] Task 4.2: Contract tests
- [ ] Final verification
- [ ] **Target:** ~450 high-quality tests

---

## 🎯 Final Checklist

Before marking complete:

- [ ] All 3 docs (CONTRACTS, SEMANTIC_MODEL, TESTING_GUIDE) accurate
- [ ] CI passing at ≥95% for retained tests
- [ ] No known API drift issues
- [ ] Contract tests in place to prevent regression
- [ ] Test suite pruned to remove YAGNI violations
- [ ] Documentation updated
- [ ] PR ready for review

---

## 📞 Decision Points

**If you encounter:**

1. **Unexpected test failures after fix**
   - Revert last change
   - Investigate root cause
   - Update CONTRACTS.md if assumption was wrong

2. **Render import issue unsolvable in 3 hours**
   - Skip Phase 3
   - File issue for later
   - Continue to Phase 4

3. **Test pruning reveals real bugs**
   - File issues for each bug
   - Fix critical bugs in Phase 5
   - Update SEMANTIC_MODEL.md with new invariants

---

**Ready to execute! Start with Phase 1, Task 1.1.**
