# Semantic Model & Contract Audit

**Date:** 2025-09-30  
**Purpose:** Identify gaps between documented semantics and implementation  
**Status:** DRAFT - Foundation for contract formalization

---

## 🎯 Executive Summary

### Current State
- ✅ **SEMANTIC_MODEL.md**: Well-defined core abstractions
- ✅ **CONTRACTS.md**: API signatures documented
- ⚠️ **Implementation**: Partial alignment with model
- ❌ **Tests**: Mix of correct contracts & outdated assumptions

### Critical Gaps Found
1. **Missing Invariant Enforcement**: Many invariants documented but not enforced
2. **Incomplete State Machines**: State transitions not formally validated
3. **Test Misalignment**: ~161 failing tests, many due to API mismatches
4. **Missing Guard Tests**: Few property-based tests for invariants

---

## 📊 Gap Analysis by Component

### 1. Environment (PlumeSearchEnv)

#### ✅ What's Correct
- Inherits from `gym.Env` (fixed today)
- Gymnasium API compliance (reset/step/render/close)
- Deterministic behavior with seeding

#### ❌ Gaps Found

**Missing State Machine Validation:**
```python
# SEMANTIC_MODEL.md says:
# - Can only step() after reset()
# - Cannot reset() after close()

# Current Implementation: NO ENFORCEMENT
# Should have:
class EnvironmentState(Enum):
    CREATED = "created"
    READY = "ready"      # after reset()
    TERMINATED = "terminated"
    TRUNCATED = "truncated"
    CLOSED = "closed"

def step(self):
    if self.state not in (EnvironmentState.READY, ...):
        raise StateError("Cannot step before reset")
```

**Missing Invariant Checks:**
- ❌ No validation that observations match stated structure
- ❌ No check that reward is in valid range [0, 1]
- ❌ No verification that info dict contains required keys

**Action Items:**
- [ ] Add `_state: EnvironmentState` attribute
- [ ] Validate state transitions in reset/step/close
- [ ] Add post-condition checks on return values
- [ ] Create property tests for determinism invariant

---

### 2. AgentState

#### ✅ What's Correct
- Position tracked correctly
- Step count increments

#### ❌ Gaps Found

**Invariant:** "Position always within grid bounds"
```python
# Currently: BoundaryEnforcer validates, but AgentState can hold invalid positions
# Should: AgentState constructor validates or enforces bounds

@dataclass
class AgentState:
    position: Coordinates
    grid_size: GridSize  # MISSING - needed to validate position
    
    def __post_init__(self):
        if not (0 <= self.position.x < self.grid_size.width):
            raise ValidationError("Position x out of bounds")
```

**Invariant:** "Step count monotonically increases"
```python
# Currently: increment_step() doesn't check
# Should:
def increment_step(self):
    old = self.step_count
    self.step_count += 1
    assert self.step_count == old + 1, "Step count must increase by 1"
```

**Invariant:** "Goal status is idempotent"
```python
# Currently: No enforcement
# Should track state transitions:
def update_goal_status(self, reached: bool):
    if self.goal_reached and not reached:
        raise StateError("Cannot un-reach goal")
    self.goal_reached = reached
```

**Action Items:**
- [ ] Add grid_size to AgentState (or validate externally)
- [ ] Enforce monotonic step count
- [ ] Make goal_reached idempotent (write-once)
- [ ] Add property test: step count never decreases

---

### 3. Plume / ConcentrationField

#### ✅ What's Correct
- Gaussian field generation
- Source location handling

#### ❌ Gaps Found

**Invariant:** "Concentration highest at source, decays with distance"
```python
# Currently: Assumed true, not tested
# Should: Property test

@given(
    source=st.tuples(st.integers(0, 31), st.integers(0, 31)),
    other_pos=st.tuples(st.integers(0, 31), st.integers(0, 31))
)
def test_concentration_decays_with_distance(source, other_pos):
    field = ConcentrationField(...)
    if other_pos != source:
        assert field[other_pos] <= field[source]
```

**Invariant:** "Values in [0.0, 1.0]"
```python
# Currently: Checked in some places, not everywhere
# Should: Always validate after generation

def _generate_field(self):
    field = ... # generation logic
    assert np.all(field >= 0) and np.all(field <= 1), "Field values out of range"
    return field
```

**Action Items:**
- [ ] Add post-condition checks on field generation
- [ ] Property test: concentration decreases with distance from source
- [ ] Property test: all values in [0, 1]
- [ ] Property test: max concentration at source

---

### 4. Reward Calculation

#### ✅ What's Correct
- Sparse reward structure (0 or 1)
- Goal detection via distance

#### ❌ Gaps Found

**Invariant:** "Reward determined solely by distance to goal"
```python
# Currently: Correct, but not formally tested
# Should: Property test

@given(
    agent_pos=coordinates_strategy,
    source_pos=coordinates_strategy,
    goal_radius=st.floats(0.1, 10.0)
)
def test_reward_pure_function_of_distance(agent_pos, source_pos, goal_radius):
    """Reward depends ONLY on distance, nothing else."""
    dist = agent_pos.distance_to(source_pos)
    reward1 = calculate_reward(agent_pos, source_pos, goal_radius)
    reward2 = calculate_reward(agent_pos, source_pos, goal_radius)
    
    # Same inputs → same output (determinism)
    assert reward1 == reward2
    
    # Correct value
    expected = 1.0 if dist <= goal_radius else 0.0
    assert reward1 == expected
```

**Invariant:** "Goal detection is deterministic and consistent"
```python
# Edge case: What happens at exactly distance == goal_radius?
# SEMANTIC_MODEL.md says: "≤, not <"
# Must test this boundary!

def test_goal_boundary_condition():
    # At exactly goal_radius distance, should be goal reached
    assert calculate_reward(distance=5.0, goal_radius=5.0) == 1.0
    # Just outside, should not be goal
    assert calculate_reward(distance=5.001, goal_radius=5.0) == 0.0
```

**Action Items:**
- [ ] Property test: reward is pure function
- [ ] Edge case test: boundary at goal_radius
- [ ] Property test: reward always in {0.0, 1.0}

---

### 5. Episode Management

#### ✅ What's Correct
- Episode lifecycle tracking
- Statistics collection

#### ❌ Gaps Found

**Invariant:** "Each episode has unique seed"
```python
# Currently: Not enforced
# Should: Track used seeds

class EpisodeManager:
    def __init__(self):
        self._used_seeds: Set[int] = set()
    
    def start_episode(self, seed: int):
        if seed in self._used_seeds:
            warnings.warn(f"Seed {seed} reused - not unique!")
        self._used_seeds.add(seed)
```

**Invariant:** "Trajectory length ≤ max_steps"
```python
# Should: Enforce during stepping
def step(self):
    if len(self.trajectory) >= self.max_steps:
        raise StateError("Exceeded max_steps")
```

**Invariant:** "Reproducible: same seed → same episode"
```python
# This is CRITICAL but only tested manually
# Should: Property test

def test_episode_reproducibility():
    """Two episodes with same seed must be identical."""
    seed = 42
    
    env1 = create_env()
    traj1 = run_episode(env1, seed, policy)
    
    env2 = create_env()
    traj2 = run_episode(env2, seed, policy)
    
    # Exact same trajectory
    assert traj1 == traj2
```

**Action Items:**
- [ ] Add seed uniqueness tracking (or document why not needed)
- [ ] Enforce trajectory length limit
- [ ] Property test: determinism/reproducibility
- [ ] Property test: termination conditions mutually exclusive

---

## 🔒 Critical Invariants Needing Guards

### Priority 1: Safety Invariants (MUST enforce)

1. **State Machine Integrity**
   - Cannot step before reset
   - Cannot use after close
   - Terminal states are final

2. **Bounds Checking**
   - Positions within grid
   - Rewards in valid range
   - Step counts non-negative

3. **Type Safety**
   - Coordinates are (int, int)
   - Actions in valid range [0, 8]
   - Seeds are valid integers

### Priority 2: Semantic Correctness

4. **Determinism**
   - Same seed → same sequence
   - Pure functions stay pure
   - No hidden global state

5. **Physical Consistency**
   - Distance metric properties
   - Concentration field properties
   - Reward calculation consistency

### Priority 3: Performance Guarantees

6. **Resource Bounds**
   - Memory usage within limits
   - Step execution time targets
   - Rendering performance goals

---

## 📋 Contract Formalization Plan

### Phase 2: What We Need to Document

For each component, we need:

#### A. **Preconditions** (what caller must guarantee)
```python
def step(self, action: int) -> Tuple[...]:
    """
    Preconditions:
    - reset() has been called at least once
    - action in range [0, 8]
    - environment not closed
    - not already terminated/truncated
    """
```

#### B. **Postconditions** (what function guarantees)
```python
def step(self, action: int) -> Tuple[Obs, float, bool, bool, dict]:
    """
    Postconditions:
    - Returns exactly 5 elements
    - reward in [0.0, 1.0]
    - terminated and truncated are bool
    - if terminated: info['termination_reason'] exists
    - observation matches stated structure
    """
```

#### C. **Invariants** (always true before & after)
```python
class Environment:
    """
    Class Invariants:
    - self._episode_count >= 0
    - if self._state == READY: self._agent_state is not None
    - self._step_count <= self._max_steps
    """
```

#### D. **State Transitions** (valid state changes)
```
CREATED --reset()--> READY
READY --step()--> READY | TERMINATED | TRUNCATED
TERMINATED --reset()--> READY
TRUNCATED --reset()--> READY
* --close()--> CLOSED
CLOSED --X--> (cannot leave)
```

---

## 🧪 Guard Test Strategy

### Phase 3: Tests We Need to Write

#### 1. **Contract Enforcement Tests** (per component)
```python
# tests/contracts/test_environment_contracts.py

class TestEnvironmentStateTransitions:
    def test_cannot_step_before_reset(self):
        env = PlumeSearchEnv()
        with pytest.raises(StateError, match="Cannot step before reset"):
            env.step(0)
    
    def test_cannot_reset_after_close(self):
        env = PlumeSearchEnv()
        env.close()
        with pytest.raises(StateError, match="Cannot reset after close"):
            env.reset()
```

#### 2. **Invariant Property Tests** (using Hypothesis)
```python
@given(
    seed=st.integers(0, 2**31-1),
    actions=st.lists(st.integers(0, 8), min_size=1, max_size=100)
)
def test_determinism_invariant(seed, actions):
    """Same seed and actions → identical outcomes."""
    env1, env2 = PlumeSearchEnv(), PlumeSearchEnv()
    
    obs1, _ = env1.reset(seed=seed)
    obs2, _ = env2.reset(seed=seed)
    assert np.array_equal(obs1, obs2)
    
    for action in actions:
        result1 = env1.step(action)
        result2 = env2.step(action)
        assert result1 == result2
```

#### 3. **Boundary Condition Tests**
```python
def test_goal_radius_boundary():
    """Test exact boundary: distance == goal_radius."""
    env = PlumeSearchEnv(goal_radius=5.0)
    # Position agent exactly goal_radius away
    # Verify reward = 1.0 (not 0.0)
```

#### 4. **Metamorphic Tests**
```python
def test_reward_scaling_invariant():
    """Scaling grid shouldn't change reward structure."""
    # Run on 10x10 grid
    # Run on 100x100 grid (scaled positions)
    # Reward behavior should be equivalent
```

---

## 🎯 Success Criteria

### Phase Completion Checklist

**Phase 2 Complete When:**
- [ ] All components have documented pre/post conditions
- [ ] All invariants explicitly listed in CONTRACTS.md
- [ ] State machines formalized with transition tables
- [ ] Mathematical properties documented (distance, reward, etc.)

**Phase 3 Complete When:**
- [ ] Contract guard tests for all state transitions
- [ ] Property tests for all critical invariants
- [ ] Boundary tests for all edge cases
- [ ] All guard tests passing (100% green)

**Phase 4 Complete When:**
- [ ] All unit tests align with contracts
- [ ] No tests expect deprecated APIs
- [ ] Test names clearly state what contract they verify
- [ ] Remove/fix tests that contradict semantic model

**Phase 5 Complete When:**
- [ ] All contract guard tests passing
- [ ] All aligned unit tests passing
- [ ] Performance benchmarks separated from correctness tests
- [ ] Zero API mismatches or deprecated patterns

---

## 📝 Next Steps

### Immediate Actions (Phase 2)

1. **Formalize Environment Contract** (2-3 hours)
   - Document state machine
   - List all pre/post conditions
   - Define invariants explicitly

2. **Formalize Core Components** (4-6 hours)
   - AgentState, Coordinates, GridSize
   - RewardCalculator
   - BoundaryEnforcer
   - ConcentrationField

3. **Update CONTRACTS.md** (1-2 hours)
   - Add invariant section
   - Add state transition diagrams
   - Add property specifications

### Then Phase 3: Write Guard Tests

Start with highest priority invariants and work down.

---

**REMEMBER:** Don't chase green tests. Build the right contracts first, then make code satisfy them.
