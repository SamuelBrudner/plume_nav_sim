# Reward Function Interface Contract

**Component:** Reward Function Abstraction  
**Version:** 2.0.0  
**Date:** 2025-10-01  
**Status:** CANONICAL - All implementations MUST conform

---

## 📦 Type Dependencies

This contract references types defined in other contracts:

- `AgentState`: See `core_types.md` - Contains position and orientation
- `ActionType`: See `action_processor_interface.md` - Discrete `int` or continuous `np.ndarray`
- `ConcentrationField`: See `concentration_field.md` - Plume sampling interface
- `Coordinates`: See `core_types.md` - 2D integer grid position

---

## 🎯 Purpose

Define the **universal interface** for all reward function implementations, separating reward calculation logic from environment implementation to enable:

- Pluggable reward functions (sparse, dense, shaped, multi-objective)
- Research flexibility without environment modification
- Config-as-code for reproducible experiments
- Clean separation of concerns

**CRITICAL DISTINCTION:**

- **This contract:** Universal interface ALL reward functions must implement
- **reward_function.md:** Specification of ONE implementation (sparse binary reward)

---

## 📐 Interface Definition

### Type Signature

```python
RewardFunction: (AgentState, ActionType, AgentState, ConcentrationField) → ℝ

Where:
  - prev_state: AgentState before action
  - action: Action taken (matches ActionProcessor.ActionType: int or np.ndarray)
  - next_state: AgentState after action
  - plume_field: ConcentrationField for context
  - Returns: Scalar reward value

Implementations SHOULD accept the library `ConcentrationField` type and MAY also
handle raw NumPy arrays for backward compatibility in tests or custom pipelines.
```

### Protocol Specification

```python
class RewardFunction(Protocol):
    """Protocol defining reward function interface.
    
    All reward function implementations must conform to this interface
    to be compatible with the environment and config system.
    """
    
    def compute_reward(
        self,
        prev_state: AgentState,
        action: ActionType,
        next_state: AgentState,
        plume_field: ConcentrationField
    ) -> float:
        """Compute reward for state transition.
        
        Preconditions:
          P1: prev_state is valid AgentState
          P2: action is valid Action
          P3: next_state is valid AgentState
          P4: plume_field is valid ConcentrationField
        
        Postconditions:
          C1: result is finite float (not NaN, not inf)
          C2: result is deterministic (same inputs → same output)
          C3: isinstance(result, (float, int, np.floating, np.integer))
        
        Properties:
          1. Determinism: Same (s, a, s') always produces same reward
          2. Purity: No side effects or hidden state
          3. Finite: Result is always finite
        
        Returns:
            Scalar reward value (implementation-specific range)
        """
        ...
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return reward function metadata for logging/reproducibility.
        
        Postconditions:
          C1: Returns dictionary with at least 'type' key
          C2: All values are JSON-serializable
        
        Returns:
            Dictionary containing:
            - 'type': str - Reward function type identifier
            - 'parameters': dict - Configuration parameters
            - Additional implementation-specific metadata
        """
        ...
```

---

## 🌍 Universal Properties

These apply to ALL reward function implementations:

### Property 1: Determinism (UNIVERSAL)

```python
∀ s, a, s', field:
  compute_reward(s, a, s', field) = compute_reward(s, a, s', field)

No dependency on:
  - Time
  - Order of calls
  - Random state
  - External mutable state
```

**Test:**

```python
@given(
    prev_state=agent_state_strategy(),
    action=action_strategy(),
    next_state=agent_state_strategy(),
    plume_field=concentration_field_strategy()
)
def test_reward_deterministic(prev_state, action, next_state, plume_field):
    """Same inputs always produce same reward."""
    reward_fn = create_test_reward_function()
    
    result1 = reward_fn.compute_reward(prev_state, action, next_state, plume_field)
    result2 = reward_fn.compute_reward(prev_state, action, next_state, plume_field)
    
    assert result1 == result2, "Reward must be deterministic"
```

### Property 2: Purity (UNIVERSAL)

```python
∀ inputs: compute_reward(inputs) has no side effects

No modification of:
  - Input arguments
  - Global state
  - File system
  - Network
```

**Test:**

```python
def test_reward_is_pure():
    """Reward computation has no side effects."""
    reward_fn = create_test_reward_function()
    
    prev_state = AgentState(position=Coordinates(10, 10))
    next_state = AgentState(position=Coordinates(11, 10))
    action = Action.RIGHT
    field = create_test_field()
    
    # Store original values
    orig_prev = copy.deepcopy(prev_state)
    orig_next = copy.deepcopy(next_state)
    orig_field = copy.deepcopy(field.field)
    
    # Compute reward
    reward = reward_fn.compute_reward(prev_state, action, next_state, field)
    
    # Verify no mutations
    assert prev_state == orig_prev, "prev_state modified"
    assert next_state == orig_next, "next_state modified"
    assert np.array_equal(field.field, orig_field), "field modified"
```

### Property 3: Finiteness (UNIVERSAL)

```python
∀ s, a, s', field:
  reward = compute_reward(s, a, s', field)
  ⇒ reward ∈ ℝ ∧ |reward| < ∞ ∧ ¬isnan(reward)
```

**Test:**

```python
@given(
    prev_state=agent_state_strategy(),
    action=action_strategy(),
    next_state=agent_state_strategy(),
    plume_field=concentration_field_strategy()
)
def test_reward_is_finite(prev_state, action, next_state, plume_field):
    """Reward is always finite."""
    reward_fn = create_test_reward_function()
    
    reward = reward_fn.compute_reward(prev_state, action, next_state, plume_field)
    
    assert math.isfinite(reward), f"Reward {reward} is not finite"
    assert not math.isnan(reward), f"Reward is NaN"
    assert not math.isinf(reward), f"Reward is infinite"
```

---

## 📊 Implementation-Specific Properties

Different reward types have different properties (NOT universal):

### Sparse Binary Reward (ONE implementation)

- Codomain: {0.0, 1.0} exactly
- Distance-based threshold
- See reward_function.md for full specification

### Step Penalty Reward (ANOTHER implementation - WILL IMPLEMENT)

- Codomain: (-∞, goal_reward] (can be negative!)
- Constant penalty per time step to encourage efficiency
- Formula: reward = goal_reward if at goal, else -step_penalty
- Common in RL for time-limited tasks
- See: Mnih et al. (2015) DQN; Sutton & Barto (2018) Ch. 3
- **Demonstrates rewards need not be in [0,1]**
 - Note: This reward is independent of step_count; it is computed per-step and does not require internal episode counters.

### Dense Distance Reward (ANOTHER implementation - documentation example)

- Codomain: [0.0, 1.0] continuous
- Monotonic with distance
- Smooth gradient

### Shaped Reward (ANOTHER implementation - documentation example)

- Potential-based: reward = φ(s') - φ(s)
- Policy-invariant (Ng et al. 1999)

---

## 🧪 Required Test Suite

Every reward function implementation MUST pass:

### Universal Property Tests

```python
class TestRewardFunctionInterface:
    """Test suite for reward function interface compliance."""
    
    @pytest.fixture
    def reward_function(self):
        """Override in concrete test classes."""
        raise NotImplementedError
    
    @given(state=agent_state_strategy(), ...)
    def test_determinism(self, reward_function, ...):
        """Test P1: Determinism."""
        
    def test_purity(self, reward_function):
        """Test P2: Purity (no side effects)."""
        
    @given(...)
    def test_finiteness(self, reward_function, ...):
        """Test P3: Finiteness."""
        
    def test_return_type(self, reward_function):
        """Test postcondition: returns numeric type."""
        
    def test_metadata_structure(self, reward_function):
        """Test get_metadata() returns valid structure."""
```

### Edge Case Tests (Implementation-Specific)

```python
def test_same_state_reward():
    """Test reward when prev_state = next_state."""
    
def test_maximum_distance_reward():
    """Test reward at maximum possible distance."""
    
def test_boundary_positions():
    """Test reward at grid boundaries."""
```

---

## 💻 Reference Implementation

### Sparse Binary Reward (Existing Logic)

```python
@dataclass
class SparseGoalReward:
    """Sparse binary reward based on goal proximity.
    
    Implements existing reward_function.md specification.
    """
    goal_radius: float
    source_location: Coordinates
    
    def __post_init__(self):
        if self.goal_radius < 0:
            raise ValidationError("goal_radius must be non-negative")
    
    def compute_reward(
        self,
        prev_state: AgentState,
        action: Action,
        next_state: AgentState,
        plume_field: ConcentrationField
    ) -> float:
        """Sparse binary reward: 1.0 if within goal_radius, else 0.0."""
        # Use source_location from initialization or field
        source = self.source_location or plume_field.source_location
        
        # Calculate distance from new position to source
        distance = next_state.position.distance_to(source)
        
        # Binary reward with inclusive boundary
        return 1.0 if distance <= self.goal_radius else 0.0
    
    def get_metadata(self) -> Dict[str, Any]:
        return {
            'type': 'sparse_goal',
            'parameters': {
                'goal_radius': self.goal_radius,
                'source_location': (self.source_location.x, self.source_location.y)
            }
        }


@dataclass
class StepPenaltyReward:
    """Step penalty reward encouraging efficient search.
    
    Applies constant negative penalty per time step to incentivize faster
    goal-reaching. Standard approach in time-limited RL tasks.
    
    Reward structure:
      - At goal: +goal_reward (e.g., +1.0)
      - Not at goal: -step_penalty (e.g., -0.01)
    
    This creates pressure to minimize episode length while still providing
    strong positive signal at goal.
    
    Citation: Mnih et al. (2015), "Human-level control through deep RL"
              Sutton & Barto (2018), Chapter 3 "Finite MDPs"
    """
    goal_radius: float
    source_location: Coordinates
    goal_reward: float = 1.0        # Positive reward at goal
    step_penalty: float = 0.01      # Penalty per step (positive value)
    
    def __post_init__(self):
        if self.goal_radius < 0:
            raise ValidationError("goal_radius must be non-negative")
        if self.step_penalty < 0:
            raise ValidationError("step_penalty must be non-negative")
        # Note: goal_reward can be any finite value (positive, negative, zero)
        if not math.isfinite(self.goal_reward):
            raise ValidationError("goal_reward must be finite")
    
    def compute_reward(
        self,
        prev_state: AgentState,
        action: Action,
        next_state: AgentState,
        plume_field: ConcentrationField
    ) -> float:
        """Compute reward: goal_reward at goal, -step_penalty elsewhere.
        
        Reward structure:
          - At goal: +goal_reward (large positive)
          - Searching: -step_penalty (small negative)
        
        Properties:
          - Encourages minimal-time solutions (negative per step)
          - Clear positive signal at goal (terminal reward)
          - Codomain: (-∞, goal_reward] (can be negative!)
        
        Example with defaults:
          - At goal: +1.0
          - Searching: -0.01 per step
          - Episode of 100 steps: cumulative = +1.0 - 100*0.01 = 0.0
          - Episode of 50 steps: cumulative = +1.0 - 50*0.01 = +0.5 (better!)
        
        Note: Step penalty applied every non-goal step, regardless of step number.
              Does not require step_count - time pressure is constant per step.
        """
        source = self.source_location or plume_field.source_location
        distance = next_state.position.distance_to(source)
        
        # Check if at goal
        if distance <= self.goal_radius:
            return self.goal_reward
        
        # Not at goal: apply constant step penalty
        return -self.step_penalty
    
    def get_metadata(self) -> Dict[str, Any]:
        return {
            'type': 'step_penalty',
            'parameters': {
                'goal_radius': self.goal_radius,
                'source_location': (self.source_location.x, self.source_location.y),
                'goal_reward': self.goal_reward,
                'step_penalty': self.step_penalty
            },
            'codomain': f'(-∞, {self.goal_reward}]',
            'citation': 'Mnih et al. (2015); Sutton & Barto (2018)'
        }

---

## 🔗 Integration Requirements

### Environment Integration

```python
class PlumeEnv:
    def __init__(
        self,
        reward_fn: Optional[RewardFunction] = None,
        **kwargs
    ):
        # Default to existing behavior
        self.reward_fn = reward_fn or SparseGoalReward(
            goal_radius=kwargs.get('goal_radius', 1.0),
            source_location=self.source_location
        )
    
    def step(self, action):
        # Store previous state
        prev_state = copy.copy(self.agent_state)
        
        # Execute action (update agent_state)
        # ...
        
        # Compute reward using injected function
        reward = self.reward_fn.compute_reward(
            prev_state=prev_state,
            action=action,
            next_state=self.agent_state,
            plume_field=self.plume_field
        )
        
        # ...
```

---

## ⚙️ Configuration Management

**IMPORTANT**: Reward function parameters should be **configurable**, not hardcoded.

### Config File Pattern (YAML)

```yaml
# conf/experiment/efficient_search.yaml
environment:
  grid_size: [128, 128]
  max_steps: 1000

reward:
  _target_: plume_nav_sim.rewards.StepPenaltyReward
  goal_radius: 1.0
  source_location: [64, 64]
  goal_reward: 1.0        # Configurable!
  step_penalty: 0.01      # Configurable!

# conf/experiment/slow_search_tolerance.yaml
reward:
  _target_: plume_nav_sim.rewards.StepPenaltyReward
  goal_radius: 2.0
  goal_reward: 10.0       # Larger reward
  step_penalty: 0.001     # Smaller penalty (more tolerant)
```

### Factory Integration

```python
# plume_nav_sim/factories.py
from omegaconf import DictConfig

def create_reward_function(config: RewardConfig) -> RewardFunction:
    """Factory creates reward from config."""
    if config.type == 'sparse':
        return SparseGoalReward(
            goal_radius=config.goal_radius,
            source_location=Coordinates(*config.source_location)
        )
    elif config.type == 'step_penalty':
        return StepPenaltyReward(
            goal_radius=config.goal_radius,
            source_location=Coordinates(*config.source_location),
            goal_reward=config.goal_reward,      # From config!
            step_penalty=config.step_penalty     # From config!
        )
    # ...
```

### Benefits of Config-Based Parameters

1. **Reproducibility**: Config files version-controlled with results
2. **Sweeps**: Easy parameter exploration with Hydra multirun
3. **No Code Changes**: Tune reward without touching source
4. **Documentation**: Config self-documents experimental setup

**Design Rule**: All reward parameters that affect behavior MUST be:

- Accepted as constructor arguments
- Included in `get_metadata()` output
- Exposed in config schemas
- Validated in `__post_init__()`

### Config Integration

```python
@dataclass
class RewardConfig:
    type: str  # 'sparse', 'dense', 'shaped'
    parameters: Dict[str, Any]

def create_reward_function(config: RewardConfig) -> RewardFunction:
    """Factory for reward functions."""
    if config.type == 'sparse':
        return SparseGoalReward(**config.parameters)
    elif config.type == 'dense':
        return DenseDistanceReward(**config.parameters)
    else:
        raise ValueError(f"Unknown reward type: {config.type}")
```

---

## ⚠️ Common Implementation Errors

### ❌ Wrong: Non-deterministic

```python
# WRONG - uses random state
class BadReward:
    def compute_reward(self, ...):
        noise = np.random.randn()  # ❌ Non-deterministic!
        return base_reward + noise
```

### ❌ Wrong: Mutates Inputs

```python
# WRONG - modifies agent state
class BadReward:
    def compute_reward(self, prev_state, action, next_state, field):
        next_state.total_reward += 1.0  # ❌ Side effect!
        return 1.0
```

### ❌ Wrong: Infinite/NaN Values

```python
# WRONG - can return inf
class BadReward:
    def compute_reward(self, ...):
        distance = next_state.position.distance_to(source)
        return 1.0 / distance  # ❌ Can be inf if distance = 0!
```

### ✅ Correct: Pure, Deterministic, Finite

```python
class GoodReward:
    def compute_reward(self, prev_state, action, next_state, field):
        distance = next_state.position.distance_to(self.source)
        # Clamp to prevent division by zero
        safe_distance = max(distance, 1e-6)
        return min(1.0, 1.0 / safe_distance)  # Finite
```

---

## 📊 Verification Checklist

Implementation MUST satisfy:

- [ ] Implements RewardFunction protocol
- [ ] Deterministic (same inputs → same output)
- [ ] Pure (no side effects)
- [ ] Returns finite float
- [ ] Passes all property tests
- [ ] Includes get_metadata() implementation
- [ ] Documents codomain (range of possible rewards)
- [ ] Documents mathematical formula
- [ ] Includes edge case tests

---

**Last Updated:** 2025-10-01  
**Related Contracts:**

- `reward_function.md` - Sparse binary reward specification
- `core_types.md` - AgentState, Action definitions
- `concentration_field.md` - ConcentrationField specification
