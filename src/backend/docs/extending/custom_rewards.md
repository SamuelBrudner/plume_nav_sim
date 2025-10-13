# Creating Custom Reward Functions

**A practical guide to implementing reward functions for plume_nav_sim**

---

## Quick Start

```python
from plume_nav_sim.interfaces import RewardFunction
from plume_nav_sim.core.state import AgentState

class MyReward:
    """Custom reward: goal bonus minus time penalty."""
    
    def __init__(self, goal_position, goal_radius=5.0):
        self.goal_position = goal_position
        self.goal_radius = goal_radius
    
    def compute_reward(self, prev_state, action, next_state, plume_field):
        # Check if agent reached goal
        if self._at_goal(next_state):
            return 10.0  # Big reward at goal
        return -0.01  # Small penalty per step
    
    def get_metadata(self):
        return {
            "type": "time_penalty",
            "parameters": {"goal_radius": self.goal_radius}
        }
    
    def _at_goal(self, state):
        dx = state.position.x - self.goal_position.x
        dy = state.position.y - self.goal_position.y
        distance = (dx*dx + dy*dy) ** 0.5
        return distance <= self.goal_radius
```

**Use it:**

```python
from plume_nav_sim.envs import ComponentBasedEnvironment

env = ComponentBasedEnvironment(
    reward_function=MyReward(goal_position=Coordinates(64, 64)),
    ...
)
```

---

## The RewardFunction Protocol

### Required Methods

```python
def compute_reward(
    self,
    prev_state: AgentState,
    action: int,
    next_state: AgentState,
    plume_field: ConcentrationField
) -> float:
    """Compute reward for transition.
    
    Args:
        prev_state: State before action
        action: Action taken (integer from action_space)
        next_state: State after action
        plume_field: Concentration field for context
    
    Returns:
        Scalar reward value
    """
    ...

def get_metadata(self) -> Dict[str, Any]:
    """Return reward function metadata.
    
    Returns:
        Dictionary with at least 'type' key
    """
    ...
```

### Universal Properties

Your reward function **must** satisfy:

1. **Determinism**: Same inputs always produce same reward
2. **Purity**: No side effects, no mutations
3. **Finiteness**: Result is finite (not NaN, not inf)

---

## Implementation Patterns

### Pattern 1: Goal-Based Rewards

**Sparse (binary):**

```python
class SparseGoal:
    def compute_reward(self, prev_state, action, next_state, plume_field):
        return 1.0 if at_goal(next_state) else 0.0
```

**Dense (distance-based):**

```python
class DenseGoal:
    def compute_reward(self, prev_state, action, next_state, plume_field):
        prev_dist = distance_to_goal(prev_state)
        next_dist = distance_to_goal(next_state)
        return prev_dist - next_dist  # Reward for getting closer
```

### Pattern 2: Concentration-Based Rewards

**Following gradient:**

```python
class ConcentrationGradient:
    def compute_reward(self, prev_state, action, next_state, plume_field):
        prev_conc = plume_field.field_array[prev_state.position.y, prev_state.position.x]
        next_conc = plume_field.field_array[next_state.position.y, next_state.position.x]
        return next_conc - prev_conc  # Reward for higher concentration
```

### Pattern 3: Composite Rewards

**Multiple terms:**

```python
class CompositeReward:
    def __init__(self, goal_position, distance_weight=0.5, conc_weight=0.5):
        self.goal_position = goal_position
        self.distance_weight = distance_weight
        self.conc_weight = conc_weight
    
    def compute_reward(self, prev_state, action, next_state, plume_field):
        # Distance term
        dist_reward = self._distance_reward(prev_state, next_state)
        
        # Concentration term
        conc_reward = self._concentration_reward(prev_state, next_state, plume_field)
        
        # Weighted sum
        return (self.distance_weight * dist_reward + 
                self.conc_weight * conc_reward)
```

### Pattern 4: Infotaxis-Style

**Information gain:**

```python
class InfotaxisReward:
    def __init__(self, goal_position):
        self.goal_position = goal_position
        self.belief_state = ...  # Your belief representation
    
    def compute_reward(self, prev_state, action, next_state, plume_field):
        # Entropy before observation
        prev_entropy = self._compute_entropy(prev_state)
        
        # Update belief with new observation
        obs = plume_field.field_array[next_state.position.y, next_state.position.x]
        self._update_belief(next_state, obs)
        
        # Entropy after observation
        next_entropy = self._compute_entropy(next_state)
        
        # Reward = information gain
        return prev_entropy - next_entropy
```

---

## Complete Examples

### Example 1: Time Penalty Reward

```python
class TimePenaltyReward:
    """Penalizes long episodes to encourage efficiency."""
    
    def __init__(self, goal_position, goal_radius=5.0, 
                 goal_reward=10.0, step_penalty=0.01):
        self.goal_position = goal_position
        self.goal_radius = goal_radius
        self.goal_reward = goal_reward
        self.step_penalty = step_penalty
    
    def compute_reward(self, prev_state, action, next_state, plume_field):
        # Check goal
        distance = self._distance_to_goal(next_state.position)
        if distance <= self.goal_radius:
            return self.goal_reward
        
        # Penalty for each step
        return -self.step_penalty
    
    def get_metadata(self):
        return {
            "type": "time_penalty",
            "parameters": {
                "goal_radius": self.goal_radius,
                "goal_reward": self.goal_reward,
                "step_penalty": self.step_penalty
            }
        }
    
    def _distance_to_goal(self, position):
        dx = position.x - self.goal_position.x
        dy = position.y - self.goal_position.y
        return (dx*dx + dy*dy) ** 0.5
```

### Example 2: Shaped Navigation Reward

```python
import numpy as np

class ShapedNavigationReward:
    """Dense reward combining distance and concentration."""
    
    def __init__(self, goal_position, goal_radius=5.0,
                 distance_weight=0.5, concentration_weight=0.5):
        self.goal_position = goal_position
        self.goal_radius = goal_radius
        self.distance_weight = distance_weight
        self.concentration_weight = concentration_weight
    
    def compute_reward(self, prev_state, action, next_state, plume_field):
        # Goal bonus
        if self._at_goal(next_state):
            return 10.0
        
        # Distance reward (potential-based shaping)
        prev_dist = self._distance_to_goal(prev_state.position)
        next_dist = self._distance_to_goal(next_state.position)
        distance_reward = prev_dist - next_dist
        
        # Concentration reward
        prev_conc = self._get_concentration(prev_state.position, plume_field)
        next_conc = self._get_concentration(next_state.position, plume_field)
        concentration_reward = next_conc - prev_conc
        
        # Weighted combination
        return (self.distance_weight * distance_reward + 
                self.concentration_weight * concentration_reward)
    
    def get_metadata(self):
        return {
            "type": "shaped_navigation",
            "parameters": {
                "goal_radius": self.goal_radius,
                "distance_weight": self.distance_weight,
                "concentration_weight": self.concentration_weight
            }
        }
    
    def _at_goal(self, state):
        return self._distance_to_goal(state.position) <= self.goal_radius
    
    def _distance_to_goal(self, position):
        dx = position.x - self.goal_position.x
        dy = position.y - self.goal_position.y
        return np.sqrt(dx*dx + dy*dy)
    
    def _get_concentration(self, position, plume_field):
        return float(plume_field.field_array[position.y, position.x])
```

---

## Testing Your Reward Function

### 1. Contract Tests (Required)

```python
from tests.contracts.test_reward_function_interface import TestRewardFunctionInterface
import pytest

class TestTimePenaltyReward(TestRewardFunctionInterface):
    """Contract compliance tests."""
    
    @pytest.fixture
    def reward_function(self):
        return TimePenaltyReward(
            goal_position=Coordinates(64, 64),
            goal_radius=5.0
        )
    
    # Inherits all property tests automatically:
    # - test_determinism
    # - test_purity
    # - test_finiteness
    # - test_returns_numeric_type
    # - test_returns_scalar_not_array
```

### 2. Implementation-Specific Tests

```python
class TestTimePenaltyRewardBehavior:
    """Implementation-specific behavior tests."""
    
    def test_goal_reward(self):
        """Test: Returns goal_reward when at goal."""
        reward_fn = TimePenaltyReward(Coordinates(10, 10), goal_radius=1.0, goal_reward=10.0)
        
        prev_state = AgentState(position=Coordinates(5, 5))
        next_state = AgentState(position=Coordinates(10, 10))  # At goal
        
        reward = reward_fn.compute_reward(prev_state, 0, next_state, None)
        assert reward == 10.0
    
    def test_step_penalty(self):
        """Test: Returns negative penalty when searching."""
        reward_fn = TimePenaltyReward(Coordinates(10, 10), step_penalty=0.01)
        
        prev_state = AgentState(position=Coordinates(5, 5))
        next_state = AgentState(position=Coordinates(6, 5))  # Not at goal
        
        reward = reward_fn.compute_reward(prev_state, 0, next_state, None)
        assert reward == -0.01
    
    def test_metadata_complete(self):
        """Test: Metadata includes all parameters."""
        reward_fn = TimePenaltyReward(Coordinates(10, 10))
        metadata = reward_fn.get_metadata()
        
        assert metadata["type"] == "time_penalty"
        assert "goal_radius" in metadata["parameters"]
        assert "step_penalty" in metadata["parameters"]
```

---

## Common Pitfalls

### ❌ Mutating State

```python
# BAD: Mutates input
def compute_reward(self, prev_state, action, next_state, plume_field):
    next_state.total_reward += 1.0  # ❌ Mutation!
    return 1.0
```

✅ **Fix:** Don't modify inputs—they're read-only.

### ❌ Non-Determinism

```python
# BAD: Random reward
def compute_reward(self, prev_state, action, next_state, plume_field):
    return random.random()  # ❌ Different every time!
```

✅ **Fix:** Same inputs must give same output.

### ❌ Returning Non-Finite

```python
# BAD: Can return infinity
def compute_reward(self, prev_state, action, next_state, plume_field):
    distance = calculate_distance(...)
    return 1.0 / distance  # ❌ Infinity when distance=0!
```

✅ **Fix:** Clamp or handle edge cases:

```python
distance = max(distance, 1e-6)  # Avoid division by zero
return 1.0 / distance
```

### ❌ Hidden State

```python
# BAD: Depends on instance state
def __init__(self):
    self.step_count = 0

def compute_reward(self, prev_state, action, next_state, plume_field):
    self.step_count += 1  # ❌ Hidden state!
    return -self.step_count
```

✅ **Fix:** Use `next_state.step_count` instead.

---

## Best Practices

### ✅ Normalize Rewards

```python
# Scale to consistent range
def compute_reward(self, ...):
    raw_reward = ...
    return np.clip(raw_reward, -1.0, 1.0)
```

### ✅ Document Reward Range

```python
def get_metadata(self):
    return {
        "type": "my_reward",
        "reward_range": [-0.01, 10.0],  # [min, max]
        "parameters": {...}
    }
```

### ✅ Make Configurable

```python
@dataclass
class MyRewardConfig:
    goal_radius: float = 5.0
    penalty: float = 0.01

class MyReward:
    def __init__(self, config: MyRewardConfig):
        self.config = config
```

### ✅ Log Reward Components

```python
def compute_reward(self, ...):
    distance_term = ...
    concentration_term = ...
    
    # Store for logging (in metadata or separate)
    self._last_reward_breakdown = {
        "distance": distance_term,
        "concentration": concentration_term
    }
    
    return distance_term + concentration_term
```

---

## Integration with Configs

### Pydantic Config

```python
from plume_nav_sim.config import RewardConfig

config = RewardConfig(
    type="time_penalty",
    goal_radius=5.0,
    parameters={"step_penalty": 0.02}
)
```

### YAML Config

```yaml
# conf/experiment/my_experiment.yaml
reward:
  type: time_penalty
  goal_radius: 5.0
  parameters:
    step_penalty: 0.02
```

### Factory Extension

```python
# plume_nav_sim/config/factories.py
def create_reward_function(config, goal_location):
    if config.type == "time_penalty":
        return TimePenaltyReward(
            goal_position=goal_location,
            goal_radius=config.goal_radius,
            step_penalty=config.parameters.get("step_penalty", 0.01)
        )
    # ... other types
```

---

## Further Reading

- [Protocol Interfaces Guide](protocol_interfaces.md) - Understanding protocols
- [Contract Testing](../testing/contract_tests.md) - Property-based tests
- `contracts/reward_function_interface.md` - Formal specification
- `plume_nav_sim/rewards/` - Built-in implementations

---

## Summary

**To create a custom reward function:**

1. Implement `compute_reward()` and `get_metadata()`
2. Ensure determinism, purity, and finiteness
3. Write contract tests (inherit from `TestRewardFunctionInterface`)
4. Add implementation-specific tests
5. Integrate with config system (optional)
6. Use in `ComponentBasedEnvironment`

**That's it—no inheritance, no registration, just implement and use!**
