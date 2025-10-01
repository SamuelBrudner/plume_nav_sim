# Gymnasium API Contract

**Component:** Public Environment Interface  
**Version:** 1.0.0  
**Date:** 2025-10-01  
**Status:** CANONICAL - All implementations MUST conform

---

## 🎯 Purpose

Define the formal contract for the Gymnasium-compatible RL environment API. This is the **public interface** that external RL libraries and agents depend on.

**Critical:** Changes to this contract are **breaking changes** for downstream users.

---

## 📋 Gymnasium Standard Compliance

This environment implements the [Gymnasium v0.29+ API](https://gymnasium.farama.org/api/env/).

### Required Interface

```python
class PlumeSearchEnv(gymnasium.Env):
    """Gymnasium-compatible environment for plume search.
    
    Standard Methods:
      - reset(seed, options) → (observation, info)
      - step(action) → (observation, reward, terminated, truncated, info)
      - close() → None
    
    Standard Attributes:
      - action_space: gymnasium.Space
      - observation_space: gymnasium.Space
      - metadata: dict
    """
```

---

## 🎮 Action Space

### Current Implementation

```python
action_space: Discrete(4)
```

**Actions:**
```python
0: UP     → (0, +1)
1: RIGHT  → (+1, 0)
2: DOWN   → (0, -1)
3: LEFT   → (-1, 0)
```

### Type Definition

```python
ActionType = int  # ∈ [0, 3] for current implementation

# Valid actions
∀ action: ActionType:
  action ∈ {0, 1, 2, 3}
```

### Movement Mapping

```python
MOVEMENT_VECTORS: Dict[int, Tuple[int, int]] = {
    0: (0, 1),   # UP
    1: (1, 0),   # RIGHT
    2: (0, -1),  # DOWN
    3: (-1, 0),  # LEFT
}
```

### Validation Contract

```python
def step(action: int) -> tuple:
    """Execute one step.
    
    Preconditions:
      P1: action ∈ [0, 3]
      P2: action is integer
      P3: Environment not closed
      P4: reset() called at least once
    
    Raises:
      ValueError: If action out of bounds
      StateError: If step() called before reset()
      StateError: If step() called after close()
    """
```

### ⚠️ KNOWN ISSUE: Action Space Mismatch

**Documentation says:** 9 actions (8 directions + stay)  
**Implementation has:** 4 actions (cardinal directions only)

**Resolution needed:** See `PHASE3_COMPLETE_SUMMARY.md` Finding #1.

---

## 👁️ Observation Space

### Current Implementation

```python
observation_space: Dict({
    "agent_position": Box(
        low=0, 
        high=grid_size-1, 
        shape=(2,), 
        dtype=np.int32
    ),
    "concentration_field": Box(
        low=0.0,
        high=1.0,
        shape=(height, width),
        dtype=np.float32
    ),
    "source_location": Box(
        low=0,
        high=grid_size-1,
        shape=(2,),
        dtype=np.int32
    )
})
```

### Type Definition

```python
ObservationType = Dict[str, np.ndarray]

# Structure guarantee
observation: ObservationType = {
    "agent_position": np.ndarray[shape=(2,), dtype=np.int32],
    "concentration_field": np.ndarray[shape=(H, W), dtype=np.float32],
    "source_location": np.ndarray[shape=(2,), dtype=np.int32]
}
```

### Observation Contract

```python
def reset(seed=None, options=None) -> Tuple[ObservationType, InfoType]:
    """Reset environment.
    
    Postconditions:
      C1: observation ∈ observation_space
      C2: observation["agent_position"] ∈ [0, grid_size)²
      C3: observation["concentration_field"].shape = (height, width)
      C4: observation["concentration_field"] ∈ [0, 1]
      C5: observation["source_location"] ∈ [0, grid_size)²
      C6: All arrays are valid numpy arrays
      C7: No NaN or Inf values
    """

def step(action) -> Tuple[ObservationType, float, bool, bool, InfoType]:
    """Take action.
    
    Postconditions:
      C1: observation ∈ observation_space
      C2: All observation invariants hold (same as reset)
      C3: observation reflects post-action state
    """
```

### Observation Invariants

```python
# Universal invariants for all observations
∀ obs: ObservationType:
  I1: "agent_position" ∈ obs.keys()
  I2: "concentration_field" ∈ obs.keys()
  I3: "source_location" ∈ obs.keys()
  I4: obs["agent_position"].dtype = np.int32
  I5: obs["concentration_field"].dtype = np.float32
  I6: obs["source_location"].dtype = np.int32
  I7: ¬np.any(np.isnan(obs[k])) ∀ k
  I8: ¬np.any(np.isinf(obs[k])) ∀ k
```

---

## ℹ️ Info Dictionary

### Type Definition

```python
InfoType = Dict[str, Any]

# Required keys (always present)
info: InfoType = {
    "seed": Optional[int],           # Set after reset()
    "step_count": int,               # Current step in episode
    "total_reward": float,           # Cumulative reward
    "goal_reached": bool,            # Whether goal found
    # ... optional keys below
}
```

### Info Contract

```python
def reset(seed=None, options=None) -> Tuple[ObservationType, InfoType]:
    """Reset info structure.
    
    Postconditions:
      C1: "seed" in info
      C2: info["seed"] = seed (or None if not provided)
      C3: No other required keys in reset info
    """

def step(action) -> Tuple[ObservationType, float, bool, bool, InfoType]:
    """Step info structure.
    
    Postconditions:
      C1: "step_count" in info
      C2: "total_reward" in info
      C3: "goal_reached" in info
      C4: info["step_count"] ≥ 0
      C5: info["total_reward"] ≥ 0
      C6: info["goal_reached"] ∈ {True, False}
    """
```

### Info Invariants

```python
# After reset()
∀ info from reset():
  I1: "seed" in info
  I2: isinstance(info["seed"], (int, type(None)))

# After step()
∀ info from step():
  I1: "step_count" in info
  I2: info["step_count"] ≥ 0
  I3: "total_reward" in info
  I4: info["total_reward"] ≥ 0.0
  I5: "goal_reached" in info
  I6: isinstance(info["goal_reached"], bool)
```

### Optional Info Keys

```python
# May be present in step() info
optional_keys = {
    "distance_to_goal": float,        # Current distance
    "concentration_at_agent": float,  # Local concentration
    "episode_time_ms": float,         # Episode duration
    "performance_metrics": dict,      # Timing info
}
```

**Contract:** Optional keys may be added without breaking API.

---

## 🔄 Method Contracts

### `reset()`

```python
def reset(
    self,
    *,
    seed: Optional[int] = None,
    options: Optional[dict] = None
) -> Tuple[ObservationType, InfoType]:
    """Reset environment for new episode.
    
    Preconditions:
      P1: Environment not closed
      P2: seed is None or int ≥ 0
      P3: options is None or dict
    
    Postconditions:
      C1: Returns (observation, info) tuple
      C2: observation ∈ observation_space
      C3: info["seed"] = seed
      C4: Episode step_count = 0
      C5: Environment in READY state
      C6: Can now call step()
    
    Determinism:
      ∀ env₁, env₂, seed:
        env₁.reset(seed=seed) == env₂.reset(seed=seed)
        (Same seed → identical initial states)
    
    Idempotency:
      Can call reset() multiple times safely
    
    Raises:
      StateError: If environment closed
      ValidationError: If seed invalid
    
    Examples:
      obs, info = env.reset(seed=42)
      assert obs in env.observation_space
      assert info["seed"] == 42
    """
```

### `step()`

```python
def step(
    self,
    action: int
) -> Tuple[ObservationType, float, bool, bool, InfoType]:
    """Execute one timestep.
    
    Preconditions:
      P1: reset() called at least once
      P2: Environment not closed
      P3: action ∈ action_space
      P4: Previous episode not terminated (unless reset() called)
    
    Postconditions:
      C1: Returns (observation, reward, terminated, truncated, info)
      C2: observation ∈ observation_space
      C3: reward ∈ ℝ (typically {0.0, 1.0} for current implementation)
      C4: terminated ∈ {True, False}
      C5: truncated ∈ {True, False}
      C6: info is InfoType with required keys
      C7: step_count incremented by 1
      C8: total_reward = prev_total_reward + reward
      C9: If terminated or truncated, must call reset() before next step()
    
    Termination Conditions:
      terminated = True ⟺ goal_reached
      truncated = True ⟺ step_count ≥ max_steps
    
    Exclusivity (Usually):
      terminated ∧ truncated = False (typically)
      (Edge case: can both be True if goal reached on last step)
    
    Raises:
      StateError: If step() before reset()
      StateError: If step() after close()
      ValueError: If action invalid
    
    Examples:
      obs, reward, terminated, truncated, info = env.step(0)
      assert obs in env.observation_space
      assert reward in {0.0, 1.0}
      assert isinstance(terminated, bool)
    """
```

### `close()`

```python
def close(self) -> None:
    """Release resources.
    
    Preconditions:
      None (always safe to call)
    
    Postconditions:
      C1: Environment in CLOSED state
      C2: Cannot call reset() or step() after close()
      C3: Resources released
    
    Idempotency:
      Can call close() multiple times safely
      Second call is no-op
    
    Side Effects:
      - Closes render window (if open)
      - Releases display resources
      - Flushes logs
    
    Raises:
      Never raises (absorbs errors)
    
    Examples:
      env.close()
      env.close()  # Safe, no error
    """
```

---

## 🔒 Invariants

### Global Environment Invariants

```python
# G1: State consistency
∀ env: PlumeSearchEnv:
  env.observation_space defined
  env.action_space defined
  env.metadata defined

# G2: Determinism with seed
∀ env, seed, actions:
  env.reset(seed=seed) then apply(actions) produces identical trajectory

# G3: Episode independence
∀ episode₁, episode₂:
  episode₁ does not affect episode₂ (after reset())

# G4: No global mutable state
  All state encapsulated in env instance
  No module-level mutable variables

# G5: Type stability
  observation always same structure
  info always dict with required keys
  reward always float
```

### Lifecycle Invariants

```python
# L1: Initial state
After __init__(): state = CREATED, cannot step()

# L2: After reset()
state = READY, can step()

# L3: After terminal step()
Can still call step() (but should reset())
Gymnasium allows over-running episodes

# L4: After close()
state = CLOSED, cannot reset() or step()
```

---

## 🎨 Render Mode

### Render Modes

```python
metadata = {
    "render_modes": ["human", "rgb_array"],
    "render_fps": 30
}
```

### Render Contract

```python
def render(self) -> Optional[np.ndarray]:
    """Render environment state.
    
    Preconditions:
      P1: Environment initialized with render_mode
      P2: reset() called at least once
    
    Returns:
      - None if render_mode = "human"
      - np.ndarray[shape=(H,W,3), dtype=uint8] if render_mode = "rgb_array"
    
    Side Effects:
      - If "human": displays window, updates screen
      - If "rgb_array": none
    
    Notes:
      - Render is optional, not required for RL training
      - Can be expensive, avoid in tight loops
    """
```

---

## 🧪 Testing Requirements

### API Compliance Tests

```python
def test_gymnasium_api_compliance():
    """Environment satisfies Gymnasium API."""
    from gymnasium.utils.env_checker import check_env
    
    env = PlumeSearchEnv()
    check_env(env)  # Official Gymnasium checker

def test_observation_space_consistency():
    """All observations match declared space."""
    env = PlumeSearchEnv()
    obs, info = env.reset(seed=42)
    
    assert obs in env.observation_space
    
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs in env.observation_space
        
        if terminated or truncated:
            break

def test_action_space_validation():
    """Invalid actions rejected."""
    env = PlumeSearchEnv()
    env.reset(seed=42)
    
    # Valid actions
    for action in [0, 1, 2, 3]:
        env.reset(seed=42)
        result = env.step(action)
        assert len(result) == 5
    
    # Invalid actions
    for invalid in [-1, 4, 100]:
        env.reset(seed=42)
        with pytest.raises(ValueError):
            env.step(invalid)

def test_info_keys_present():
    """Info dict has required keys."""
    env = PlumeSearchEnv()
    obs, info = env.reset(seed=42)
    
    assert "seed" in info
    
    obs, reward, term, trunc, info = env.step(0)
    assert "step_count" in info
    assert "total_reward" in info
    assert "goal_reached" in info

def test_determinism():
    """Same seed produces same trajectory."""
    env1 = PlumeSearchEnv()
    env2 = PlumeSearchEnv()
    
    obs1, _ = env1.reset(seed=42)
    obs2, _ = env2.reset(seed=42)
    
    np.testing.assert_array_equal(obs1["agent_position"], obs2["agent_position"])
    
    actions = [0, 1, 2, 3, 0]
    for action in actions:
        obs1, r1, t1, tr1, i1 = env1.step(action)
        obs2, r2, t2, tr2, i2 = env2.step(action)
        
        np.testing.assert_array_equal(obs1["agent_position"], obs2["agent_position"])
        assert r1 == r2
        assert t1 == t2
```

---

## 🔗 Related Contracts

- **environment_state_machine.md** - Internal state transitions
- **reward_function.md** - Reward calculation semantics
- **core_types.md** - Data type definitions
- **concentration_field.md** - Observation component

---

## ⚠️ Breaking Changes

Changes that break this contract require major version bump:

1. **Action space size change** (e.g., 4 → 9 actions)
2. **Observation structure change** (add/remove keys)
3. **Required info keys change**
4. **Termination logic change**
5. **Type changes** (e.g., obs from dict to array)

**Non-breaking changes:**
- Add optional info keys
- Extend metadata
- Add render modes
- Internal refactors

---

## 📊 API Stability Guarantee

**Commitment:** This API is **stable** within major versions.

**Semantic Versioning:**
- `MAJOR`: Breaking API changes
- `MINOR`: Backward-compatible additions
- `PATCH`: Bug fixes

**Current:** v1.0.0 (Gymnasium v0.29+ compatible)

---

**Last Updated:** 2025-10-01  
**Next Review:** After resolving action space mismatch (Finding #1)
