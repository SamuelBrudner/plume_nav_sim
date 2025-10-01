# Test Taxonomy: Formal Verification Strategy

**Date:** 2025-09-30  
**Purpose:** Comprehensive test categories for semantic correctness  
**Approach:** Mathematical properties + Contract enforcement

---

## 🎯 Philosophy

> **"Tests should verify mathematical properties, not implementation details."**

Each test category serves a specific purpose in proving correctness:
- **Property Tests** → Universal quantifiers (∀)
- **Contract Guards** → Precondition/postcondition enforcement
- **Semantic Invariants** → "Always true" statements
- **Schema Compliance** → Type safety and structure validation

---

## 📊 Test Category Reference

### 1. Property Tests (Hypothesis-Based)

**Purpose:** Verify mathematical properties hold for all valid inputs

**Tools:** `hypothesis` library with strategies

**Examples:**

```python
from hypothesis import given, strategies as st

@given(
    x=st.integers(),
    y=st.integers()
)
def test_addition_commutative(x, y):
    """Addition is commutative: x + y = y + x"""
    assert x + y == y + x

@given(
    seed=st.integers(0, 2**31-1),
    actions=st.lists(st.integers(0, 8))
)
def test_determinism_property(seed, actions):
    """Same seed + actions → identical outcomes (∀ seed, actions)"""
    env1 = create_env()
    env2 = create_env()
    
    traj1 = run_episode(env1, seed, actions)
    traj2 = run_episode(env2, seed, actions)
    
    assert traj1 == traj2
```

**Key Properties to Test:**
- **Algebraic:** commutativity, associativity, identity, inverses
- **Order:** reflexivity, transitivity, antisymmetry
- **Bounds:** min/max values, ranges
- **Relationships:** monotonicity, symmetry

---

### 2. Contract Guard Tests

**Purpose:** Enforce design-by-contract (pre/post conditions + invariants)

**Pattern:** Check contracts at API boundaries

**Examples:**

```python
def test_step_requires_reset_precondition():
    """Precondition: cannot step() before reset()"""
    env = PlumeSearchEnv()
    
    with pytest.raises(StateError, match="Cannot step before reset"):
        env.step(0)

def test_step_postcondition_returns_five_tuple():
    """Postcondition: step() returns exactly 5 elements"""
    env = PlumeSearchEnv()
    env.reset(seed=42)
    
    result = env.step(0)
    assert len(result) == 5
    assert isinstance(result[0], np.ndarray)  # obs
    assert isinstance(result[1], float)       # reward
    assert isinstance(result[2], bool)        # terminated
    assert isinstance(result[3], bool)        # truncated
    assert isinstance(result[4], dict)        # info

def test_reward_calculator_invariant_binary_output():
    """Invariant: reward is always 0.0 or 1.0"""
    reward = calculate_reward(pos_a, pos_b, radius)
    assert reward in (0.0, 1.0), "Reward must be binary"
```

**Contract Elements:**
- **Preconditions:** What caller must guarantee
- **Postconditions:** What function guarantees to return
- **Invariants:** Properties that hold before AND after

---

### 3. Semantic Invariant Tests

**Purpose:** Verify domain-specific "always true" statements

**Focus:** Business logic and physics constraints

**Examples:**

```python
def test_position_always_within_bounds_invariant():
    """Semantic: agent position always valid after step"""
    env = PlumeSearchEnv(grid_size=(32, 32))
    env.reset(seed=42)
    
    for _ in range(100):
        obs, _, terminated, truncated, _ = env.step(random.randint(0, 8))
        
        # Extract position from obs
        x, y = obs['position']
        
        # Invariant: always within bounds
        assert 0 <= x < 32, f"X position {x} out of bounds"
        assert 0 <= y < 32, f"Y position {y} out of bounds"
        
        if terminated or truncated:
            break

def test_step_count_monotonic_invariant():
    """Semantic: step count never decreases"""
    env = PlumeSearchEnv()
    env.reset(seed=42)
    
    prev_step = 0
    for _ in range(50):
        obs, _, terminated, truncated, info = env.step(0)
        current_step = info.get('step', obs.get('step_count', 0))
        
        assert current_step > prev_step, "Step count must increase"
        assert current_step == prev_step + 1, "Step count increases by 1"
        
        prev_step = current_step
        
        if terminated or truncated:
            break

def test_concentration_maximum_at_source_invariant():
    """Semantic: plume concentration highest at source"""
    field = create_concentration_field(
        grid_size=(32, 32),
        source_location=(16, 16),
        sigma=2.0
    )
    
    source_concentration = field.sample((16, 16))
    
    # Check all positions
    for x in range(32):
        for y in range(32):
            concentration = field.sample((x, y))
            assert concentration <= source_concentration, \
                f"Concentration at ({x},{y}) exceeds source"
```

**Invariant Categories:**
- **Physical Laws:** concentration decay, distance properties
- **State Consistency:** step count, reward accumulation
- **Safety Properties:** bounds, type safety, resource limits

---

### 4. Schema Compliance Tests

**Purpose:** Validate data structure conformance

**Tools:** `pydantic`, JSON schema, type checkers

**Examples:**

```python
def test_observation_schema_compliance():
    """Observation matches declared schema exactly"""
    env = PlumeSearchEnv()
    obs, info = env.reset(seed=42)
    
    # Required fields
    assert 'concentration' in obs
    assert 'position' in obs
    assert 'step_count' in obs
    assert 'goal_reached' in obs
    
    # Correct types
    assert isinstance(obs['concentration'], (float, np.floating))
    assert isinstance(obs['position'], (tuple, Coordinates))
    assert isinstance(obs['step_count'], (int, np.integer))
    assert isinstance(obs['goal_reached'], (bool, np.bool_))
    
    # Value constraints
    assert 0.0 <= obs['concentration'] <= 1.0
    assert obs['step_count'] >= 0

def test_info_dict_schema_compliance():
    """Info dict contains required keys after step"""
    env = PlumeSearchEnv()
    env.reset(seed=42)
    _, _, terminated, _, info = env.step(0)
    
    # Required keys
    required = {'step', 'episode', 'seed'}
    assert required.issubset(info.keys())
    
    # Conditional keys
    if terminated:
        assert 'termination_reason' in info

def test_coordinates_type_safety():
    """Coordinates must be (int, int), not floats"""
    with pytest.raises(TypeError):
        create_coordinates((3.5, 4.2))  # Floats not allowed
    
    # Valid
    coords = create_coordinates((3, 4))
    assert isinstance(coords.x, int)
    assert isinstance(coords.y, int)
```

**Schema Types:**
- **Input Validation:** Arguments to public methods
- **Output Validation:** Return types and structures
- **Intermediate Validation:** Internal data structures

---

### 5. Idempotency Tests

**Purpose:** Verify f(f(x)) = f(x) for idempotent operations

**Examples:**

```python
def test_close_is_idempotent():
    """Calling close() multiple times is safe"""
    env = PlumeSearchEnv()
    env.reset(seed=42)
    
    env.close()
    env.close()  # Should not error
    env.close()  # Still safe

def test_goal_reached_is_idempotent():
    """Once goal reached, stays reached"""
    agent_state = AgentState(position=(10, 10))
    
    agent_state.mark_goal_reached()
    assert agent_state.goal_reached == True
    
    # Cannot un-reach goal
    agent_state.mark_goal_reached()
    assert agent_state.goal_reached == True

def test_field_generation_idempotent():
    """Generating field twice with same params yields same field"""
    field1 = create_concentration_field(
        grid_size=(32, 32),
        source_location=(16, 16),
        sigma=2.0
    )
    
    field2 = create_concentration_field(
        grid_size=(32, 32),
        source_location=(16, 16),
        sigma=2.0
    )
    
    assert np.allclose(field1.field, field2.field)
```

**Idempotent Operations:**
- Resource cleanup (close, release)
- State flags (goal_reached, initialized)
- Deterministic computations (field generation)

---

### 6. Determinism Tests

**Purpose:** Same inputs → same outputs (no hidden state)

**Critical for:** RL reproducibility, debugging, testing

**Examples:**

```python
def test_reset_determinism():
    """Same seed → identical initial state"""
    env1, env2 = PlumeSearchEnv(), PlumeSearchEnv()
    
    obs1, info1 = env1.reset(seed=42)
    obs2, info2 = env2.reset(seed=42)
    
    assert np.array_equal(obs1, obs2)
    assert info1['seed'] == info2['seed']

def test_episode_determinism():
    """Same seed + actions → identical trajectory"""
    env1, env2 = PlumeSearchEnv(), PlumeSearchEnv()
    
    env1.reset(seed=42)
    env2.reset(seed=42)
    
    actions = [0, 2, 1, 4, 3, 0, 2]
    
    trajectory1 = []
    trajectory2 = []
    
    for action in actions:
        result1 = env1.step(action)
        result2 = env2.step(action)
        
        trajectory1.append(result1)
        trajectory2.append(result2)
    
    # Every step identical
    for t1, t2 in zip(trajectory1, trajectory2):
        assert np.array_equal(t1[0], t2[0])  # obs
        assert t1[1] == t2[1]                # reward
        assert t1[2] == t2[2]                # terminated
        assert t1[3] == t2[3]                # truncated

def test_reward_function_determinism():
    """Pure function: no internal state"""
    pos_a = (10, 10)
    pos_b = (15, 15)
    radius = 5.0
    
    # Call multiple times
    r1 = calculate_reward(pos_a, pos_b, radius)
    r2 = calculate_reward(pos_a, pos_b, radius)
    r3 = calculate_reward(pos_a, pos_b, radius)
    
    assert r1 == r2 == r3, "Pure function must be deterministic"

def test_no_global_state_pollution():
    """Environments don't interfere with each other"""
    env1 = PlumeSearchEnv()
    env2 = PlumeSearchEnv()
    
    env1.reset(seed=42)
    env2.reset(seed=99)
    
    # Stepping env1 shouldn't affect env2
    result1 = env1.step(0)
    result2a = env2.step(0)
    
    # Reset env2 with same seed
    env2.reset(seed=99)
    result2b = env2.step(0)
    
    # env2's second run should match first (no pollution from env1)
    assert np.array_equal(result2a[0], result2b[0])
```

**Determinism Levels:**
- **Strong:** Bit-for-bit identical outputs
- **Weak:** Numerically close (for floating point)
- **Statistical:** Same distribution (for randomness)

---

### 7. Commutativity Tests

**Purpose:** Verify f(a, b) = f(b, a) where applicable

**Examples:**

```python
def test_distance_commutative():
    """distance(a, b) = distance(b, a)"""
    pos_a = Coordinates(10, 10)
    pos_b = Coordinates(20, 15)
    
    dist_ab = pos_a.distance_to(pos_b)
    dist_ba = pos_b.distance_to(pos_a)
    
    assert dist_ab == dist_ba, "Distance must be symmetric"

def test_reward_commutative_in_positions():
    """reward(agent, source) = reward(source, agent)"""
    # If we're measuring distance, order shouldn't matter
    pos_a = (10, 10)
    pos_b = (15, 15)
    radius = 3.0
    
    # This assumes reward is purely distance-based
    dist_ab = calculate_distance(pos_a, pos_b)
    dist_ba = calculate_distance(pos_b, pos_a)
    
    assert dist_ab == dist_ba

def test_set_operations_commutative():
    """Union/intersection are commutative"""
    seeds_a = {1, 2, 3}
    seeds_b = {3, 4, 5}
    
    assert seeds_a | seeds_b == seeds_b | seeds_a  # union
    assert seeds_a & seeds_b == seeds_b & seeds_a  # intersection
```

**Non-Commutative Operations (should test they're NOT):**
```python
def test_step_sequence_not_commutative():
    """Action order matters: [a, b] ≠ [b, a]"""
    env = PlumeSearchEnv()
    env.reset(seed=42)
    
    # Sequence 1: north then east
    env.reset(seed=42)
    env.step(0)  # north
    obs1, _, _, _, _ = env.step(2)  # east
    
    # Sequence 2: east then north
    env.reset(seed=42)
    env.step(2)  # east
    obs2, _, _, _, _ = env.step(0)  # north
    
    # Final positions should differ (actions don't commute)
    assert not np.array_equal(obs1['position'], obs2['position'])
```

---

### 8. Associativity Tests

**Purpose:** Verify (a ∘ b) ∘ c = a ∘ (b ∘ c)

**Examples:**

```python
def test_reward_accumulation_associative():
    """Total reward = sum of step rewards (associative sum)"""
    env = PlumeSearchEnv()
    env.reset(seed=42)
    
    rewards = []
    for _ in range(10):
        _, reward, terminated, truncated, _ = env.step(0)
        rewards.append(reward)
        if terminated or truncated:
            break
    
    # Different groupings should give same total
    total1 = sum(rewards)
    total2 = sum(rewards[:5]) + sum(rewards[5:])
    total3 = (rewards[0] + rewards[1]) + sum(rewards[2:])
    
    assert total1 == total2 == total3

def test_coordinate_translation_associative():
    """Multiple translations can be grouped"""
    pos = Coordinates(10, 10)
    
    # (pos + v1) + v2
    result1 = pos.translate(1, 0).translate(0, 1)
    
    # pos + (v1 + v2)
    result2 = pos.translate(1, 1)
    
    assert result1 == result2, "Translation is associative"

@given(
    x=st.floats(-100, 100),
    y=st.floats(-100, 100),
    z=st.floats(-100, 100)
)
def test_addition_associative(x, y, z):
    """(x + y) + z = x + (y + z)"""
    # Note: may need tolerance for floating point
    assert abs(((x + y) + z) - (x + (y + z))) < 1e-10
```

**Non-Associative (test they're NOT):**
```python
def test_matrix_multiplication_may_not_be_associative():
    """Demonstrate non-associativity where relevant"""
    # e.g., string formatting, function composition with side effects
```

---

## 🗺️ Test Organization Strategy

### Directory Structure

```
tests/
├── contracts/              # Contract guard tests
│   ├── test_preconditions.py
│   ├── test_postconditions.py
│   └── test_state_machines.py
│
├── properties/             # Hypothesis property tests
│   ├── test_determinism.py
│   ├── test_algebraic_properties.py
│   └── test_mathematical_invariants.py
│
├── invariants/             # Semantic invariants
│   ├── test_domain_invariants.py
│   ├── test_safety_invariants.py
│   └── test_physics_invariants.py
│
├── schemas/                # Schema compliance
│   ├── test_type_safety.py
│   ├── test_data_structures.py
│   └── test_api_signatures.py
│
└── integration/            # End-to-end workflows
    ├── test_episode_lifecycle.py
    └── test_component_integration.py
```

---

## 📋 Test Checklist by Component

### Environment (PlumeSearchEnv)

**Contract Guards:**
- [ ] Cannot step before reset (precondition)
- [ ] Cannot use after close (state machine)
- [ ] reset() returns valid (obs, info) (postcondition)
- [ ] step() returns 5-tuple (postcondition)
- [ ] Invalid seed raises ValidationError

**Property Tests:**
- [ ] Determinism: same seed → same trajectory
- [ ] No global state pollution
- [ ] Episode reproducibility

**Semantic Invariants:**
- [ ] State transitions follow state machine
- [ ] Step count monotonically increases
- [ ] Position always valid after step
- [ ] Terminated/truncated mutually exclusive (mostly)

**Schema Compliance:**
- [ ] Observation structure matches spec
- [ ] Info dict has required keys
- [ ] Action space validation

**Idempotency:**
- [ ] close() is idempotent
- [ ] Multiple resets allowed

---

### Reward Calculator

**Contract Guards:**
- [ ] Returns only {0.0, 1.0}
- [ ] Requires valid positions (precondition)
- [ ] goal_radius > 0 (precondition)

**Property Tests:**
- [ ] Pure function (deterministic)
- [ ] Binary output for all inputs
- [ ] Boundary: d = goal_radius → 1.0

**Commutativity:**
- [ ] distance(a, b) = distance(b, a)

**Determinism:**
- [ ] No side effects
- [ ] Same inputs → same output

---

### Concentration Field

**Contract Guards:**
- [ ] field.shape matches grid_size
- [ ] All values in [0, 1] (postcondition)
- [ ] sigma > 0 (precondition)

**Property Tests:**
- [ ] Maximum at source (∀ other positions)
- [ ] Decay with distance (monotonic)
- [ ] Symmetry around source
- [ ] Non-negativity (∀ positions)

**Semantic Invariants:**
- [ ] Physical laws (Gaussian shape)
- [ ] Smoothness (no discontinuities)

**Determinism:**
- [ ] Same params → identical field

---

### AgentState

**Contract Guards:**
- [ ] step_count >= 0 always
- [ ] position valid after update
- [ ] goal_reached write-once

**Property Tests:**
- [ ] Step count monotonic (never decreases)
- [ ] Total reward monotonic (never decreases)

**Idempotency:**
- [ ] mark_goal_reached() idempotent

**Schema Compliance:**
- [ ] position is Coordinates type
- [ ] step_count is int
- [ ] goal_reached is bool

---

## 🎯 Priority Matrix

| Test Category | Priority | Effort | Impact |
|--------------|----------|--------|--------|
| Contract Guards | **HIGH** | Medium | Critical (safety) |
| Determinism Tests | **HIGH** | Low | Critical (RL) |
| Semantic Invariants | **HIGH** | Medium | Critical (correctness) |
| Property Tests (core) | High | High | High (coverage) |
| Schema Compliance | High | Low | Medium (type safety) |
| Idempotency | Medium | Low | Medium (robustness) |
| Commutativity | Medium | Low | Low (nice-to-have) |
| Associativity | Low | Low | Low (mathematical rigor) |

---

## 🚀 Implementation Order

### Phase 1: Safety & Correctness (Week 1)
1. Contract guards (state machines, pre/post conditions)
2. Determinism tests (reproducibility)
3. Semantic invariants (bounds, monotonicity)

### Phase 2: Mathematical Properties (Week 2)
4. Property tests (Hypothesis for universal properties)
5. Schema compliance (type safety)
6. Boundary conditions

### Phase 3: Robustness (Week 3)
7. Idempotency tests
8. Commutativity (where applicable)
9. Edge cases and error paths

---

## 📊 Coverage Goals

- **Contract Guards:** 100% of public API
- **Property Tests:** 50+ properties covering core invariants
- **Semantic Invariants:** All documented in SEMANTIC_MODEL.md
- **Schema Compliance:** 100% of data structures
- **Overall Test Coverage:** 90%+ (excluding render/performance)

---

## ✅ Success Criteria

**Tests are successful when:**
- All guard tests passing (contracts enforced)
- Property tests find no counterexamples (after 1000+ examples)
- Semantic invariants hold across all scenarios
- Schema validation catches type errors at boundaries
- Determinism verified across seeds and action sequences
- Mathematical properties proven via property tests

**System is correct when:**
- All contracts satisfied
- All invariants hold
- All properties verified
- Zero semantic contradictions
