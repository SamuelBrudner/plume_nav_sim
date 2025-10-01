# Environment State Machine Contract

**Component:** `PlumeSearchEnv`  
**Version:** 2.0.0  
**Date:** 2025-10-01  
**Status:** CANONICAL - All implementations MUST conform

---

## 🎯 Purpose

Define the formal state machine for Environment lifecycle, including:
- Valid states
- Transition rules
- Preconditions & postconditions
- Class invariants
- Error conditions

## 📦 Type & Component Dependencies

- `AgentState` (see `core_types.md`) — includes `position` and `orientation`
- `ActionProcessor`, `ObservationModel`, `RewardFunction` (see respective interface contracts)
- `ConcentrationField` (see `concentration_field.md`)
- `ObservationSpace`, `ActionSpace` derived from injected components

---

## 📊 State Definition

```python
class EnvironmentState(Enum):
    """Formal states for Environment lifecycle."""
    
    CREATED = "created"
    # After __init__() completes
    # - Components allocated but not initialized
    # - Cannot step(), must reset() first
    
    READY = "ready"
    # After reset() completes
    # - Episode active, agent positioned
    # - Can step(), can reset() again, can close()
    
    TERMINATED = "terminated"
    # Episode ended (goal reached or failure)
    # - Cannot step() further in same episode
    # - Can reset() to start new episode
    # - Can close()
    
    TRUNCATED = "truncated"
    # Episode timeout (max steps reached)
    # - Cannot step() further in same episode
    # - Can reset() to start new episode
    # - Can close()
    
    CLOSED = "closed"
    # Resources released
    # - Cannot perform any operations
    # - Terminal state (no transitions out)
```

---

## 🔄 State Transition Rules

### Formal Notation

```
Γ ⊢ state : EnvironmentState

Transition rules use inference notation:
  premises
  ──────────  (RULE-NAME)
  conclusion
```

### Transition Table

| From State | Method | To State | Condition | Error If |
|-----------|--------|----------|-----------|----------|
| CREATED | `reset()` | READY | seed valid | Invalid seed |
| READY | `step()` | READY | ¬terminal ∧ steps < max | - |
| READY | `step()` | TERMINATED | goal reached | - |
| READY | `step()` | TRUNCATED | steps ≥ max | - |
| TERMINATED | `reset()` | READY | - | - |
| TRUNCATED | `reset()` | READY | - | - |
| * | `close()` | CLOSED | - | - |
| CLOSED | `reset()` | - | - | StateError |
| CLOSED | `step()` | - | - | StateError |
| CREATED | `step()` | - | - | StateError |

### Formal Transition Rules

```
  state = CREATED ∧ seed_valid(seed)
  ─────────────────────────────────  (RESET-FROM-CREATED)
  reset(seed) → state' = READY


  state = READY ∧ ¬goal_reached ∧ step_count < max_steps
  ──────────────────────────────────────────────────────  (STEP-CONTINUE)
  step(action) → state' = READY


  state = READY ∧ goal_reached
  ─────────────────────────  (STEP-TERMINATE-GOAL)
  step(action) → state' = TERMINATED


  state = READY ∧ step_count ≥ max_steps
  ─────────────────────────────────  (STEP-TRUNCATE)
  step(action) → state' = TRUNCATED


  state ∈ {TERMINATED, TRUNCATED}
  ───────────────────────────────  (RESET-AFTER-EPISODE)
  reset(seed) → state' = READY


  state ≠ CLOSED
  ────────────────  (CLOSE)
  close() → state' = CLOSED


  state = CLOSED
  ─────────────────  (NO-RESET-AFTER-CLOSE)
  reset() → StateError


  state = CLOSED
  ─────────────────  (NO-STEP-AFTER-CLOSE)
  step(action) → StateError


  state = CREATED
  ─────────────────  (NO-STEP-BEFORE-RESET)
  step(action) → StateError
```

---

## 🔒 Class Invariants

These MUST hold at all times (before and after every method call):

```python
I1: Type Safety
    self._state ∈ {CREATED, READY, TERMINATED, TRUNCATED, CLOSED}

I2: Agent State Consistency
    self._state = READY ⇒ (self._agent_state is AgentState
                           ∧ 0 ≤ self._agent_state.orientation < 360)

I3: Closed State is Terminal
    self._state = CLOSED ⇒ ∀ operations (except __del__) raise StateError

I4: Episode Count Non-Negative
    self._episode_count ≥ 0 ∧ monotonically increases

I5: Component Availability
    self._action_proc, self._obs_model, self._reward_fn, self._plume ≠ null

I6: Step Count Non-Negative
    self._step_count ≥ 0 ∧ resets with each episode

I7: Step Count Bound
    self._state = READY ⇒ self._step_count ≤ self._max_steps

I8: Episode Count Monotonic
    episode_count' ≥ episode_count (never decreases)

I9: Seed Validity
    self._seed = null ∨ (0 ≤ self._seed < 2³¹)
```

---

## 📋 Method Contracts

### `__init__()`

```python
def __init__(
    self,
    *,
    plume_model: Optional[BasePlumeModel] = None,
    reward_fn: Optional[RewardFunction] = None,
    observation_model: Optional[ObservationModel] = None,
    action_processor: Optional[ActionProcessor] = None,
    config: Optional[EnvironmentConfig] = None,
    grid_size: Optional[tuple[int, int]] = None,
    source_location: Optional[tuple[int, int]] = None,
    max_steps: Optional[int] = None,
    goal_radius: Optional[float] = None,
    render_mode: Optional[str] = None
) -> None:
    """Initialize environment in CREATED state.
    
    Preconditions:
      P1: If config provided ⇒ factories resolve successfully
      P2: If components provided ⇒ satisfy respective protocols
      P3: grid_size = null ∨ (width > 0 ∧ height > 0)
      P4: max_steps = null ∨ max_steps > 0
      P5: goal_radius = null ∨ goal_radius > 0
      P6: render_mode ∈ {null, "rgb_array", "human"}
    
    Postconditions:
      C1: self._state = CREATED
      C2: self._episode_count = 0
      C3: self._step_count = 0
      C4: Components instantiated (directly or via config)
      C5: self._action_proc.action_space assigned to self.action_space
      C6: self._obs_model.observation_space assigned to self.observation_space
      C7: No episode active (agent_state = None)
    
    Raises:
      ValidationError: If any precondition violated
      ComponentError: If factories or injected components invalid
    
    Modifies:
      All instance attributes
    """
```

### `reset()`

```python
def reset(
    self,
    *,
    seed: Optional[int] = None,
    options: Optional[dict] = None
) -> tuple[Observation, Info]:
    """Begin new episode, transition to READY state.
    
    Preconditions:
      P1: self._state ≠ CLOSED
      P2: seed = null ∨ (seed ∈ ℕ ∧ 0 ≤ seed < 2³¹)
      P3: options = null ∨ isinstance(options, dict)
    
    Postconditions:
      C1: self._state = READY
      C2: self._step_count = 0
      C3: self._episode_count = old(self._episode_count) + 1
      C4: self._agent_state = AgentState(position=start_pos,
                                         orientation=start_orientation,
                                         step_count=0,
                                         total_reward=0.0)
      C5: self._seed = seed (if provided) or auto-generated
      C6: returns (obs, info) where:
          - obs ∈ ObservationSpace
          - obs matches current state exactly
          - info['seed'] = self._seed
          - info['episode'] = self._episode_count
    
    Raises:
      StateError: If state = CLOSED
      ValidationError: If seed invalid
    
    Modifies:
      {_state, _step_count, _episode_count, _agent_state, _seed, _plume_field}
    
    Determinism:
      ∀ env₁, env₂, seed:
        env₁.reset(seed=seed)[0] = env₂.reset(seed=seed)[0]
    """
```

### `step()`

```python
def step(
    self,
    action: ActionType
) -> tuple[Observation, float, bool, bool, Info]:
    """Execute action, advance one timestep.
    
    Preconditions:
      P1: self._state ∈ {READY, TERMINATED, TRUNCATED}
          (note: can call step on terminal states, but should reset)
      P2: self._action_proc.validate_action(action) = True

    Postconditions:
      C1: returns (obs, reward, terminated, truncated, info)
      C2: obs ∈ self._obs_model.observation_space
      C3: reward = self._reward_fn(self._agent_state, action)
      C4: terminated ∈ {True, False}
      C5: truncated ∈ {True, False}
      C6: info ∈ InfoSchema
      C7: self._step_count = old(self._step_count) + 1
      C8: self._state ∈ {READY, TERMINATED, TRUNCATED}
    
    Raises:
      StateError: If state = CREATED or CLOSED
      ValidationError: If action invalid
    
    Modifies:
      {
        _agent_state,
        _step_count,
        _state (possibly),
        _last_observation,
        _last_reward
      }

    Determinism:
      ∀ env₁, env₂, seed, actions:
        env₁.reset(seed) then apply actions →
        env₂.reset(seed) then apply actions →
        identical sequences of (obs, reward, terminated, truncated)
    
    Side Effects:
      None (pure function of state and injected components)
    """
```

### `close()`

```python
def close(self) -> None:
    """Release resources, transition to CLOSED state.
    
    Preconditions:
      None (can call from any state)
    
    Postconditions:
      C1: self._state = CLOSED
      C2: All resources released
      C3: Subsequent operations raise StateError
    
    Raises:
      Never raises (defensive)
    
    Modifies:
      {_state, _closed flag, internal resources}
    
    Idempotency:
      Multiple calls have no effect after first
      close(); close(); close() is safe
    """
```

### `render()`

```python
def render(self) -> Optional[np.ndarray]:
    """Generate visualization of current state.
    
    Preconditions:
      P1: self._state ∈ {READY, TERMINATED, TRUNCATED}
      P2: self.render_mode ≠ null
    
    Postconditions:
      C1: If render_mode = "rgb_array":
          - returns ndarray with shape (H, W, 3)
          - dtype = uint8
          - values in [0, 255]
      C2: If render_mode = "human":
          - displays visualization
          - returns null
    
    Raises:
      StateError: If state = CREATED or CLOSED
      RenderingError: If render fails
    
    Modifies:
      None (read-only operation)
    
    Side Effects:
      May display window (if render_mode = "human")
    """
```

---

## 🧪 Test Requirements

Every transition and invariant MUST have corresponding tests:

### State Transition Tests

```python
# tests/contracts/test_environment_state_transitions.py

def test_initial_state_is_created():
    """After __init__, state = CREATED"""
    
def test_cannot_step_before_reset():
    """CREATED --step()--> StateError"""
    
def test_reset_transitions_to_ready():
    """CREATED --reset()--> READY"""
    
def test_step_keeps_ready_when_not_terminal():
    """READY --step()--> READY (normal case)"""
    
def test_step_transitions_to_terminated_on_goal():
    """READY --step()--> TERMINATED (goal reached)"""
    
def test_step_transitions_to_truncated_on_timeout():
    """READY --step()--> TRUNCATED (max steps)"""
    
def test_can_reset_from_terminated():
    """TERMINATED --reset()--> READY"""
    
def test_can_reset_from_truncated():
    """TRUNCATED --reset()--> READY"""
    
def test_close_from_any_state():
    """* --close()--> CLOSED"""
    
def test_cannot_reset_after_close():
    """CLOSED --reset()--> StateError"""
    
def test_cannot_step_after_close():
    """CLOSED --step()--> StateError"""
    
def test_close_is_idempotent():
    """close(); close(); close() works"""
```

### Invariant Tests

```python
def test_state_always_valid():
    """I1: state ∈ valid states"""
    
def test_agent_state_exists_when_ready():
    """I2: READY ⇒ agent_state ≠ null"""
    
def test_episode_count_monotonic():
    """I4, I7: episode_count increases"""
    
def test_step_count_non_negative():
    """I5: step_count ≥ 0"""
    
def test_step_count_resets_with_episode():
    """I5: reset() sets step_count = 0"""
    
def test_step_count_bounded():
    """I6: step_count ≤ max_steps"""
```

### Determinism Tests

```python
@given(seed=st.integers(0, 2**31-1))
def test_reset_determinism(seed):
    """Same seed → identical initial state"""
    
@given(seed=st.integers(0, 2**31-1), actions=st.lists(st.integers(0, 8)))
def test_episode_determinism(seed, actions):
    """Same seed + actions → identical trajectory"""
```

---

## 📐 Mathematical Properties

### State Reachability

```
From CREATED:
  - Can reach: READY, CLOSED
  - Cannot reach: TERMINATED, TRUNCATED (must go through READY)

From READY:
  - Can reach: READY, TERMINATED, TRUNCATED, CLOSED
  - Cannot reach: CREATED (no going back)

From TERMINATED or TRUNCATED:
  - Can reach: READY, CLOSED
  - Cannot reach: CREATED

From CLOSED:
  - Can reach: (none - terminal state)
```

### Liveness Properties

```
1. Progress: From READY, eventually reach TERMINATED or TRUNCATED
   (given bounded max_steps)

2. Termination: Every episode eventually ends
   (guaranteed by max_steps bound)

3. Restart: Can always reset (except from CLOSED)
```

### Safety Properties

```
1. No stepping before ready:
   ¬(state = CREATED ∧ step() succeeds)

2. No operations after close:
   state = CLOSED ⇒ ∀ ops (except __del__): error

3. Step count never exceeds max:
   step_count ≤ max_steps (enforced by truncation)
```

---

## ⚠️ Common Pitfalls

### ❌ Don't Do This

```python
# Calling step before reset
env = PlumeSearchEnv()
env.step(0)  # StateError!

# Using after close
env.close()
env.reset()  # StateError!

# Modifying state directly
env._state = EnvironmentState.READY  # Breaks invariants!
```

### ✅ Do This

```python
# Proper lifecycle
env = PlumeSearchEnv()
obs, info = env.reset(seed=42)
obs, reward, terminated, truncated, info = env.step(0)
if terminated or truncated:
    obs, info = env.reset(seed=43)
env.close()

# Idempotent close
env.close()
env.close()  # Safe

# Check state before operations
if env._state != EnvironmentState.CLOSED:
    env.reset()
```

---

## 🔗 Related Specifications

- **Core Types:** `contracts/core_types.md` (Coordinates, GridSize, AgentState)
- **Reward:** `contracts/reward_function.md` (Goal detection logic)
- **Observation:** `contracts/observation_schema.md` (obs structure)
- **Info Dict:** `contracts/info_schema.md` (info structure)

---

## 📚 References

- Gymnasium API: https://gymnasium.farama.org/api/env/
- SEMANTIC_MODEL.md: Environment lifecycle section
- CONTRACTS.md: Public API signatures

---

**Last Updated:** 2025-09-30  
**Next Review:** After Phase 3 (guard tests implemented)
