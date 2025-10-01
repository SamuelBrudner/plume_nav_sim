# Testing Guide & Best Practices

**Version:** 1.0.0  
**Date:** 2025-09-30  
**Companion to:** `CONTRACTS.md` and `SEMANTIC_MODEL.md`

> This guide shows HOW to test based on the contracts and semantic model. It replaces guesswork with systematic testing strategies.

---

## 🎯 Testing Philosophy

### Core Principles

1. **Test Behavior, Not Implementation**
   - ✅ Test what function does
   - ❌ Test how it does it

2. **One Behavior Per Test**
   - ✅ `test_seed_manager_generates_deterministic_seeds()`
   - ❌ `test_seed_manager_basic_operations()`

3. **Avoid Parametric Explosion**
   - ✅ Test boundary cases (0, 1, MAX)
   - ❌ Test arbitrary values (42, 123, 789, 456, 999)

4. **Test Contracts, Not Data**
   - ✅ Test that ValidationError has correct parameters
   - ❌ Test ValidationError with 50 different invalid inputs

5. **Tests Are Documentation**
   - Test names should explain behavior
   - Test body should be minimal and clear

---

## 📋 Test Classification

### **Unit Tests** (400-500 total, ~70% of suite)

**Definition:** Test single function/method in isolation.

**Characteristics:**
- No I/O (disk, network, database)
- No sleep/timing dependencies
- Mocked external dependencies
- Fast (< 10ms per test)

**Example:**
```python
def test_validate_seed_accepts_valid_integer():
    """ValidationError should not be raised for valid seed."""
    seed, error = validate_seed(42)
    assert seed == 42
    assert error == ""
```

**Coverage Requirements:**
- All public functions/methods
- All exception paths
- Boundary conditions
- Edge cases

---

### **Integration Tests** (100-150 total, ~20% of suite)

**Definition:** Test interaction between 2+ components.

**Characteristics:**
- Real component instances (no mocks)
- Test interfaces/contracts between modules
- Medium speed (10-100ms per test)

**Example:**
```python
def test_reward_calculator_updates_agent_state():
    """RewardCalculator should correctly update AgentState total_reward."""
    config = RewardCalculatorConfig(...)
    calculator = RewardCalculator(config)
    agent = AgentState(position=Coordinates(0, 0), ...)
    
    result = calculator.calculate_reward(agent.position, source_pos)
    calculator.update_agent_reward(agent, result)
    
    assert agent.total_reward == result.reward
```

**Coverage Requirements:**
- Component interaction patterns
- Data flow between components
- Contract compliance (API boundaries)

---

### **End-to-End Tests** (20-30 total, ~5% of suite)

**Definition:** Test complete workflows from user perspective.

**Characteristics:**
- Full system integration
- Real environment instances
- Slower (100ms - 1s per test)

**Example:**
```python
def test_complete_episode_workflow():
    """Complete episode from reset to termination should work."""
    env = gym.make('PlumeNavigation-v0')
    obs, info = env.reset(seed=42)
    
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    
    env.close()
    assert True  # No exceptions = success
```

**Coverage Requirements:**
- Critical user workflows
- Gymnasium API compliance
- Error recovery scenarios

---

### **Property Tests** (30-50 total, ~5% of suite)

**Definition:** Test invariants that must always hold.

**Characteristics:**
- Use Hypothesis for random inputs
- Test properties, not examples
- Can be slow (generate many cases)

**Example:**
```python
from hypothesis import given
import hypothesis.strategies as st

@given(st.integers(min_value=0, max_value=2**32-1))
def test_seed_roundtrip_property(seed):
    """Any valid seed should survive save/load cycle."""
    saved = save_seed_state(seed)
    loaded = load_seed_state(saved)
    assert loaded == seed
```

**Coverage Requirements:**
- Semantic invariants from SEMANTIC_MODEL.md
- Mathematical properties (symmetry, commutativity, etc.)
- State machine validity

---

## ✂️ Anti-Pattern: Parametric Explosion

### ❌ **Don't Do This:**

```python
@pytest.mark.parametrize("seed", [42, 123, 456, 789, 999])
@pytest.mark.parametrize("episode", [0, 1, 10, 100])
@pytest.mark.parametrize("exp_id", ["exp1", "baseline", None])
def test_episode_seed_generation(seed, episode, exp_id):
    # This creates 5 × 4 × 3 = 60 tests
    # ALL testing the SAME LOGIC with different data
    manager = SeedManager(seed)
    result = manager.generate_episode_seed(episode, exp_id)
    assert isinstance(result, int)
```

**Problems:**
- 60 tests, but only 1 code path
- Doesn't test different behavior
- Slow (60× longer than needed)
- Hard to debug failures

---

### ✅ **Do This Instead:**

```python
def test_episode_seed_is_deterministic():
    """Same inputs should produce same episode seed."""
    manager = SeedManager(42)
    seed1 = manager.generate_episode_seed(0, "exp1")
    seed2 = manager.generate_episode_seed(0, "exp1")
    assert seed1 == seed2

def test_episode_seed_varies_by_episode_number():
    """Different episode numbers should produce different seeds."""
    manager = SeedManager(42)
    seed0 = manager.generate_episode_seed(0, "exp1")
    seed1 = manager.generate_episode_seed(1, "exp1")
    assert seed0 != seed1

def test_episode_seed_varies_by_experiment_id():
    """Different experiment IDs should produce different seeds."""
    manager = SeedManager(42)
    seed_exp1 = manager.generate_episode_seed(0, "exp1")
    seed_exp2 = manager.generate_episode_seed(0, "exp2")
    assert seed_exp1 != seed_exp2

def test_episode_seed_handles_none_experiment_id():
    """None experiment_id should be valid."""
    manager = SeedManager(42)
    seed = manager.generate_episode_seed(0, None)
    assert isinstance(seed, int)
```

**Benefits:**
- 4 tests instead of 60 (93% reduction)
- Each test documents ONE behavior
- Clear failure diagnosis
- Fast execution

---

## 🎯 Testing Contracts (CONTRACTS.md)

### Pattern: **Signature Stability Tests**

**Purpose:** Ensure API contracts don't change silently.

```python
import inspect

def test_validation_error_signature():
    """ValidationError signature must match CONTRACTS.md."""
    sig = inspect.signature(ValidationError.__init__)
    params = list(sig.parameters.keys())
    
    expected = [
        'self', 'message', 'parameter_name', 'parameter_value',
        'expected_format', 'parameter_constraints', 'context'
    ]
    
    assert params == expected, (
        f"ValidationError signature changed!\n"
        f"Expected: {expected}\n"
        f"Got: {params}\n"
        f"Update CONTRACTS.md if intentional."
    )

def test_validation_error_parameter_defaults():
    """Only 'message' is required, others optional."""
    # Should work with just message
    error = ValidationError("test message")
    assert error.message == "test message"
    assert error.parameter_name is None
    assert error.parameter_value is None
```

**When to Use:**
- For all classes/functions in CONTRACTS.md
- Run in CI to catch breaking changes
- Update when intentionally changing contract

---

### Pattern: **Contract Compliance Tests**

**Purpose:** Verify implementations follow documented contracts.

```python
def test_reward_calculator_config_validates_immediately():
    """RewardCalculatorConfig must validate in __post_init__."""
    with pytest.raises(ValidationError) as exc_info:
        RewardCalculatorConfig(
            goal_radius=-1.0,  # Invalid: negative
            reward_goal_reached=1.0,
            reward_default=0.0
        )
    
    assert "goal_radius" in str(exc_info.value)
    assert "non-negative" in str(exc_info.value)

def test_validation_error_uses_parameter_value_not_invalid_value():
    """ValidationError must use parameter_value, not deprecated invalid_value."""
    error = ValidationError(
        "test",
        parameter_name="x",
        parameter_value=42
    )
    
    assert hasattr(error, 'parameter_value')
    assert not hasattr(error, 'invalid_value')  # Deprecated
    assert error.parameter_value == 42
```

---

## 🔍 Testing Semantic Invariants (SEMANTIC_MODEL.md)

### Pattern: **Invariant Tests**

**Purpose:** Verify semantic invariants always hold.

```python
def test_position_invariant_during_step():
    """Agent position must stay within grid bounds."""
    env = create_test_environment(grid_size=GridDimensions(10, 10))
    env.reset()
    
    for _ in range(100):
        action = env.action_space.sample()
        env.step(action)
        
        # Invariant: position always in bounds
        pos = env.agent_state.position
        assert 0 <= pos.x < 10
        assert 0 <= pos.y < 10

def test_step_count_monotonically_increases():
    """Step count must increase by 1 each step."""
    env = create_test_environment()
    env.reset()
    
    initial_count = env.agent_state.step_count
    for i in range(10):
        env.step(0)  # Any valid action
        expected = initial_count + i + 1
        assert env.agent_state.step_count == expected

def test_reward_accumulation_invariant():
    """Total reward must equal sum of step rewards."""
    env = create_test_environment()
    env.reset()
    
    accumulated = 0.0
    for _ in range(20):
        _, reward, _, _, _ = env.step(0)
        accumulated += reward
        assert abs(env.agent_state.total_reward - accumulated) < 1e-10
```

**When to Use:**
- For every invariant listed in SEMANTIC_MODEL.md
- Test under various conditions (random actions, edge cases)
- Use property-based testing when possible

---

### Pattern: **Determinism Tests**

**Purpose:** Verify reproducibility from seeding.

```python
def test_environment_is_deterministic_with_same_seed():
    """Same seed produces identical trajectories."""
    def run_episode(seed):
        env = gym.make('PlumeNavigation-v0')
        trajectory = []
        
        obs, _ = env.reset(seed=seed)
        for _ in range(50):
            action = 2  # Fixed policy
            obs, reward, term, trunc, _ = env.step(action)
            trajectory.append((obs, reward, term, trunc))
            if term or trunc:
                break
        
        env.close()
        return trajectory
    
    traj1 = run_episode(42)
    traj2 = run_episode(42)
    
    assert len(traj1) == len(traj2)
    for t1, t2 in zip(traj1, traj2):
        assert t1 == t2  # Exact equality
```

---

## 🏗️ Testing Patterns by Component

### **Testing Configuration Classes**

```python
class TestRewardCalculatorConfig:
    """Test pattern for configuration dataclasses."""
    
    def test_valid_configuration_succeeds(self):
        """Valid params should create config without error."""
        config = RewardCalculatorConfig(
            goal_radius=5.0,
            reward_goal_reached=1.0,
            reward_default=0.0
        )
        assert config.validate() is True
    
    def test_required_parameters_have_no_defaults(self):
        """Required params should not have defaults."""
        # This should fail (missing required params)
        with pytest.raises(TypeError):
            RewardCalculatorConfig()  # Missing required args
    
    def test_invalid_parameter_raises_validation_error(self):
        """Invalid param should raise ValidationError with details."""
        with pytest.raises(ValidationError) as exc_info:
            RewardCalculatorConfig(
                goal_radius=-1.0,
                reward_goal_reached=1.0,
                reward_default=0.0
            )
        
        error = exc_info.value
        assert error.parameter_name == "goal_radius"
        assert error.parameter_value == -1.0
        assert "non-negative" in error.message.lower()
    
    def test_optional_parameters_use_defaults(self):
        """Optional params should have sensible defaults."""
        config = RewardCalculatorConfig(
            goal_radius=5.0,
            reward_goal_reached=1.0,
            reward_default=0.0
            # Optional params use defaults
        )
        assert config.enable_caching is True
        assert config.enable_performance_monitoring is True
```

---

### **Testing Validation Functions**

```python
class TestValidateSeed:
    """Test pattern for validation functions."""
    
    def test_valid_seed_returns_seed_and_empty_error(self):
        """Valid seed should return (seed, empty_string)."""
        seed, error = validate_seed(42)
        assert seed == 42
        assert error == ""
    
    def test_invalid_seed_returns_none_and_error_message(self):
        """Invalid seed should return (None, error_msg)."""
        seed, error = validate_seed(-1)
        assert seed is None
        assert "non-negative" in error.lower()
    
    def test_none_is_valid_seed(self):
        """None should be valid (means use entropy)."""
        seed, error = validate_seed(None)
        assert seed is None
        assert error == ""
    
    def test_boundary_values(self):
        """Test boundary conditions."""
        # Minimum valid
        seed, error = validate_seed(0)
        assert seed == 0
        assert error == ""
        
        # Maximum valid
        seed, error = validate_seed(2**32 - 1)
        assert seed == 2**32 - 1
        assert error == ""
        
        # Just beyond max
        seed, error = validate_seed(2**32)
        assert seed is None
        assert "too large" in error.lower()
```

---

### **Testing Exception Handlers**

```python
def test_component_error_wraps_underlying_exception():
    """ComponentError should preserve underlying exception."""
    original = ValueError("original error")
    
    try:
        raise ComponentError(
            "Wrapper message",
            component_name="TestComponent",
            operation_name="test_op",
            underlying_error=original
        ) from original
    except ComponentError as e:
        assert e.component_name == "TestComponent"
        assert e.operation_name == "test_op"
        assert e.underlying_error is original
        assert e.__cause__ is original

def test_validation_error_reraises_without_wrapping():
    """ValidationError should not be wrapped in ComponentError."""
    def inner_function():
        raise ValidationError("inner validation failed")
    
    def outer_function():
        try:
            inner_function()
        except ValidationError:
            raise  # Re-raise without wrapping
        except Exception as e:
            raise ComponentError(...) from e
    
    with pytest.raises(ValidationError) as exc_info:
        outer_function()
    
    # Should be ValidationError, not ComponentError
    assert type(exc_info.value) == ValidationError
```

---

## 📊 Test Organization

### Directory Structure

```
tests/
├── unit/                       # Fast, isolated tests
│   ├── core/
│   │   ├── test_reward_calculator.py
│   │   ├── test_episode_manager.py
│   │   └── test_state_manager.py
│   ├── utils/
│   │   ├── test_seeding.py
│   │   ├── test_validation.py
│   │   └── test_exceptions.py
│   └── plume/
│       └── test_plume_models.py
│
├── integration/                # Component interaction tests
│   ├── test_environment_components.py
│   ├── test_reward_state_integration.py
│   └── test_seeding_integration.py
│
├── e2e/                       # End-to-end workflows
│   ├── test_complete_episodes.py
│   ├── test_gymnasium_api.py
│   └── test_error_recovery.py
│
├── property/                  # Property-based tests
│   ├── test_invariants.py
│   ├── test_determinism.py
│   └── test_mathematical_properties.py
│
├── contracts/                 # API contract tests
│   ├── test_exception_signatures.py
│   ├── test_config_interfaces.py
│   └── test_function_contracts.py
│
└── performance/               # Performance benchmarks (optional)
    └── test_latency_targets.py
```

---

### Naming Conventions

**Test Files:**
- `test_<module_name>.py` - mirrors source structure
- One test file per source file (generally)

**Test Classes:**
- `TestComponentName` - for grouping related tests
- Optional - prefer functions for simplicity

**Test Functions:**
- `test_<component>_<behavior>_<condition>`
- Examples:
  - `test_reward_calculator_detects_goal_at_exact_radius()`
  - `test_seed_manager_raises_error_for_negative_seed()`
  - `test_episode_manager_resets_state_after_termination()`

**Parametrize Naming:**
- Keep parameter names semantic
- `@pytest.mark.parametrize("invalid_input", [...])` ✅
- `@pytest.mark.parametrize("x", [...])` ❌

---

## 🎯 Test Selection Guidelines

### When to Write a Test

**✅ Write test if:**
- Public API function/method
- Exception path exists
- Boundary condition (0, 1, MAX, MIN)
- Semantic invariant must hold
- Component interaction contract
- Bug was found (regression test)

**❌ Don't write test if:**
- Private implementation detail
- Same logic tested elsewhere
- Parametric variation of existing test (unless different behavior)
- Performance only (use benchmark suite)
- Obvious pass-through (getter/setter with no logic)

---

### Decision Tree

```
Does this test a PUBLIC API?
├─ No → Skip (test through public API instead)
└─ Yes → Continue

Does it test BEHAVIOR or IMPLEMENTATION?
├─ Implementation → Skip (refactor-sensitive)
└─ Behavior → Continue

Is there ALREADY a test for this behavior?
├─ Yes → Check if different code path
│   ├─ Same path → Skip (redundant)
│   └─ Different path → Continue
└─ No → Continue

Does it test SAME LOGIC with different DATA?
├─ Yes → Consolidate into property test or skip
└─ No → WRITE THE TEST
```

---

## 🔧 Test Fixtures & Helpers

### Standard Fixtures

```python
@pytest.fixture
def valid_reward_config():
    """Standard valid RewardCalculatorConfig."""
    return RewardCalculatorConfig(
        goal_radius=5.0,
        reward_goal_reached=1.0,
        reward_default=0.0
    )

@pytest.fixture
def test_grid_size():
    """Standard test grid size."""
    return GridDimensions(width=100, height=100)

@pytest.fixture
def test_environment(test_grid_size):
    """Fully configured test environment."""
    env = create_plume_search_env(
        grid_size=test_grid_size,
        seed=42
    )
    yield env
    env.close()  # Cleanup
```

### Factory Fixtures

```python
@pytest.fixture
def make_reward_calculator():
    """Factory for creating RewardCalculator instances."""
    def _make(**kwargs):
        defaults = {
            'goal_radius': 5.0,
            'reward_goal_reached': 1.0,
            'reward_default': 0.0
        }
        defaults.update(kwargs)
        config = RewardCalculatorConfig(**defaults)
        return RewardCalculator(config)
    return _make

# Usage in test:
def test_something(make_reward_calculator):
    calc = make_reward_calculator(goal_radius=10.0)
    ...
```

---

## 📈 Coverage Targets

### Minimum Coverage Requirements

| Component Type | Line Coverage | Branch Coverage |
|----------------|---------------|-----------------|
| Core logic (reward, episode, state) | 95% | 90% |
| Utilities (validation, seeding) | 90% | 85% |
| Plume models | 85% | 80% |
| Rendering | 70% | 65% |
| Tests themselves | N/A | N/A |

### What NOT to Count Toward Coverage

- Deprecated code (will be removed)
- Debug-only code
- Type stubs / protocols
- `__repr__` / `__str__` methods (unless complex logic)

---

## 🚀 Running Tests

### Basic Commands

```bash
# All tests
pytest

# Specific file
pytest tests/unit/core/test_reward_calculator.py

# Specific test
pytest tests/unit/core/test_reward_calculator.py::test_config_validation

# With coverage
pytest --cov=plume_nav_sim --cov-report=html

# Fail fast (stop at first failure)
pytest -x

# Verbose
pytest -v

# Only unit tests
pytest tests/unit/

# Only integration tests
pytest tests/integration/
```

### Test Markers

```python
# Mark as slow
@pytest.mark.slow
def test_long_episode():
    ...

# Mark as integration
@pytest.mark.integration
def test_component_interaction():
    ...

# Skip if condition
@pytest.mark.skipif(sys.platform == 'win32', reason="Unix only")
def test_unix_feature():
    ...
```

**Run only marked tests:**
```bash
pytest -m "not slow"  # Skip slow tests
pytest -m integration  # Only integration tests
```

---

## ✅ Test Quality Checklist

Before committing tests, verify:

- [ ] Test name clearly describes behavior
- [ ] One assertion per logical check
- [ ] No commented-out code
- [ ] No print statements (use logging if needed)
- [ ] Fixtures used for common setup
- [ ] Cleanup handled (files, resources)
- [ ] Fast (< 100ms for unit tests)
- [ ] Deterministic (no flaky behavior)
- [ ] Tests contract, not implementation
- [ ] No parametric explosion
- [ ] Clear failure messages

---

## 📚 Example: Complete Test Module

```python
"""
Tests for RewardCalculator following TESTING_GUIDE.md patterns.
"""
import pytest
from plume_nav_sim.core.reward_calculator import (
    RewardCalculator,
    RewardCalculatorConfig,
    RewardResult
)
from plume_nav_sim.core.geometry import Coordinates
from plume_nav_sim.utils.exceptions import ValidationError


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def valid_config():
    """Standard valid configuration."""
    return RewardCalculatorConfig(
        goal_radius=5.0,
        reward_goal_reached=1.0,
        reward_default=0.0
    )

@pytest.fixture
def calculator(valid_config):
    """Standard RewardCalculator instance."""
    return RewardCalculator(valid_config)


# ============================================================================
# Configuration Tests
# ============================================================================

class TestRewardCalculatorConfig:
    """Test configuration validation and contract compliance."""
    
    def test_valid_config_succeeds(self):
        """Valid configuration should not raise error."""
        config = RewardCalculatorConfig(
            goal_radius=5.0,
            reward_goal_reached=1.0,
            reward_default=0.0
        )
        assert config.validate() is True
    
    def test_negative_goal_radius_raises_validation_error(self):
        """Negative goal_radius should raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            RewardCalculatorConfig(
                goal_radius=-1.0,
                reward_goal_reached=1.0,
                reward_default=0.0
            )
        
        assert exc_info.value.parameter_name == "goal_radius"
        assert exc_info.value.parameter_value == -1.0


# ============================================================================
# Behavior Tests
# ============================================================================

class TestRewardCalculation:
    """Test reward calculation behavior."""
    
    def test_goal_reached_when_at_exact_radius(self, calculator):
        """Agent exactly at goal_radius distance should reach goal."""
        source = Coordinates(10, 10)
        # Position at exactly 5.0 units away (3-4-5 triangle)
        agent = Coordinates(13, 14)
        
        result = calculator.calculate_reward(agent, source)
        
        assert result.goal_reached is True
        assert result.reward == 1.0
    
    def test_goal_not_reached_beyond_radius(self, calculator):
        """Agent beyond goal_radius should not reach goal."""
        source = Coordinates(10, 10)
        agent = Coordinates(20, 20)  # Far away
        
        result = calculator.calculate_reward(agent, source)
        
        assert result.goal_reached is False
        assert result.reward == 0.0


# ============================================================================
# Invariant Tests
# ============================================================================

def test_distance_is_symmetric(calculator):
    """distance(a, b) must equal distance(b, a)."""
    pos1 = Coordinates(0, 0)
    pos2 = Coordinates(10, 10)
    
    dist1 = calculator.get_distance_to_goal(pos1, pos2)
    dist2 = calculator.get_distance_to_goal(pos2, pos1)
    
    assert dist1 == dist2


# ============================================================================
# Contract Tests
# ============================================================================

def test_validation_error_uses_parameter_value():
    """ValidationError must use parameter_value per CONTRACTS.md."""
    with pytest.raises(ValidationError) as exc_info:
        RewardCalculatorConfig(
            goal_radius="invalid",  # Wrong type
            reward_goal_reached=1.0,
            reward_default=0.0
        )
    
    error = exc_info.value
    assert hasattr(error, 'parameter_value')
    assert not hasattr(error, 'invalid_value')  # Deprecated


# End of example module
```

---

**END OF TESTING_GUIDE.MD**
