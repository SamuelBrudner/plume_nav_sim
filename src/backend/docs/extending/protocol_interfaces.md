# Protocol-Based Component Interfaces

**Understanding the Protocol Pattern in plume_nav_sim**

---

## What Are Protocols?

Protocols are Python's way of defining interfaces using **structural subtyping** (duck typing). Unlike inheritance, you don't need to explicitly inherit from a base class—if your class has the right methods with the right signatures, it "satisfies" the protocol.

```python
from typing import Protocol

class Greeter(Protocol):
    def greet(self, name: str) -> str: ...

# This class satisfies Greeter (no inheritance needed!)
class FriendlyGreeter:
    def greet(self, name: str) -> str:
        return f"Hello, {name}!"

# Type checker accepts this
def greet_user(greeter: Greeter, user: str):
    print(greeter.greet(user))

greet_user(FriendlyGreeter(), "Alice")  # ✅ Works!
```

---

## Why Protocols for Components?

### Traditional Approach (Inheritance)
```python
from abc import ABC, abstractmethod

class RewardFunction(ABC):
    @abstractmethod
    def compute_reward(self, ...): ...

# MUST inherit from RewardFunction
class MyReward(RewardFunction):
    def compute_reward(self, ...):
        return 1.0
```

**Problems:**
- ❌ Tight coupling to base class
- ❌ Hard to extend from external packages
- ❌ Can't retroactively make existing classes conform
- ❌ Forces inheritance hierarchy

### Protocol Approach (Structural)
```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class RewardFunction(Protocol):
    def compute_reward(self, ...): ...

# No inheritance needed!
class MyReward:
    def compute_reward(self, ...):
        return 1.0

# Still works
isinstance(MyReward(), RewardFunction)  # True
```

**Benefits:**
- ✅ No coupling to base class
- ✅ External packages can extend easily
- ✅ Existing classes can satisfy protocols
- ✅ No forced hierarchy

---

## Protocol Interfaces in plume_nav_sim

We define three core protocols:

### 1. **ActionProcessor**
Processes actions and computes new agent states.

**Contract:** `contracts/action_processor_interface.md`

**Required Methods:**
```python
@property
def action_space(self) -> gym.Space: ...

def process_action(
    self, action: int, current_state: AgentState, grid_size: GridSize
) -> AgentState: ...

def validate_action(self, action: int) -> bool: ...

def get_metadata(self) -> Dict[str, Any]: ...
```

### 2. **ObservationModel**
Generates observations from environment state.

**Contract:** `contracts/observation_model_interface.md`

**Required Methods:**
```python
@property
def observation_space(self) -> gym.Space: ...

def get_observation(self, env_state: Dict[str, Any]) -> ObservationType: ...

def get_metadata(self) -> Dict[str, Any]: ...
```

### 3. **RewardFunction**
Computes rewards for state transitions.

**Contract:** `contracts/reward_function_interface.md`

**Required Methods:**
```python
def compute_reward(
    self, 
    prev_state: AgentState,
    action: int,
    next_state: AgentState,
    plume_field: ConcentrationField
) -> float: ...

def get_metadata(self) -> Dict[str, Any]: ...
```

---

## Implementing a Protocol

### Step 1: Import the Protocol
```python
from plume_nav_sim.interfaces import RewardFunction
```

### Step 2: Implement Required Methods
```python
class CustomReward:  # No inheritance!
    def __init__(self, goal_position, penalty=-0.01):
        self.goal_position = goal_position
        self.penalty = penalty
    
    def compute_reward(self, prev_state, action, next_state, plume_field):
        if self._at_goal(next_state):
            return 10.0
        return self.penalty  # Small penalty per step
    
    def get_metadata(self):
        return {
            "type": "custom_penalty",
            "parameters": {"penalty": self.penalty}
        }
    
    def _at_goal(self, state):
        dx = state.position.x - self.goal_position.x
        dy = state.position.y - self.goal_position.y
        return (dx*dx + dy*dy) < 25  # 5.0 radius
```

### Step 3: Verify Protocol Conformance
```python
# Type checking (mypy/IDE)
reward: RewardFunction = CustomReward(...)  # ✅ Type checks

# Runtime checking
from plume_nav_sim.interfaces import RewardFunction
assert isinstance(CustomReward(...), RewardFunction)  # ✅ True
```

### Step 4: Use in Environment
```python
from plume_nav_sim.envs import ComponentBasedEnvironment

env = ComponentBasedEnvironment(
    reward_function=CustomReward(...),  # ✅ Works!
    ...
)
```

---

## Contract Testing

**All implementations should pass contract tests:**

```python
from tests.contracts.test_reward_function_interface import TestRewardFunctionInterface

class TestCustomReward(TestRewardFunctionInterface):
    """Contract compliance tests for CustomReward."""
    
    @pytest.fixture
    def reward_function(self):
        return CustomReward(goal_position=Coordinates(64, 64))
    
    # Inherits all interface tests:
    # - test_determinism
    # - test_purity
    # - test_finiteness
    # - test_returns_numeric_type
    # - etc.
```

Run with:
```bash
pytest tests/contracts/ -v -k CustomReward
```

---

## Protocol Properties

All protocol implementations must satisfy:

### Universal Properties
1. **Determinism**: Same inputs → same outputs
2. **Purity**: No side effects, no hidden state
3. **Type Safety**: Return correct types

### Component-Specific Properties

**ActionProcessor:**
- Boundary safety (stays in grid)
- Returns new AgentState (no mutation)

**ObservationModel:**
- Space containment (observation ∈ observation_space)
- Shape consistency

**RewardFunction:**
- Finiteness (no NaN, no inf)
- Returns scalar (not array)

---

## Best Practices

### ✅ DO
- Implement all required methods
- Follow method signatures exactly
- Return correct types
- Pass contract tests
- Document your implementation

### ❌ DON'T
- Mutate input parameters
- Have side effects
- Return None where value expected
- Depend on hidden state
- Skip contract tests

---

## Type Checking

Protocols work with static type checkers (mypy, pyright):

```python
from plume_nav_sim.interfaces import ActionProcessor

def use_processor(proc: ActionProcessor):
    # IDE autocomplete works
    space = proc.action_space
    
    # Type checking works
    result = proc.process_action(0, state, grid)
    # mypy knows: result is AgentState
```

Enable strict checking:
```bash
mypy --strict plume_nav_sim/
```

---

## External Package Example

You can implement protocols in your own package:

```python
# my_extensions/custom_rewards.py
from plume_nav_sim.interfaces import RewardFunction
from plume_nav_sim.core.state import AgentState

class MyReward:  # In separate package!
    def compute_reward(self, prev_state, action, next_state, plume_field):
        # Your logic
        return 0.5
    
    def get_metadata(self):
        return {"type": "my_custom_reward"}

# Use it
from plume_nav_sim.envs import ComponentBasedEnvironment
from my_extensions.custom_rewards import MyReward

env = ComponentBasedEnvironment(
    reward_function=MyReward(),  # ✅ Works!
    ...
)
```

**No need to modify plume_nav_sim source code!**

---

## Debugging Protocol Issues

### Error: "object does not conform to protocol"

**Cause:** Missing or incorrectly typed method.

**Fix:**
```python
# Check signature matches exactly
from plume_nav_sim.interfaces import RewardFunction
import inspect

print(inspect.signature(RewardFunction.compute_reward))
print(inspect.signature(MyReward.compute_reward))
# Signatures must match
```

### Error: Contract test fails

**Cause:** Implementation violates property (e.g., not deterministic).

**Fix:** Check property test output and fix implementation.

---

## Summary

| Aspect | Inheritance | Protocols |
|--------|-------------|-----------|
| **Coupling** | Tight (must inherit) | Loose (duck typing) |
| **External Extension** | Hard (need access to base) | Easy (just implement) |
| **Existing Classes** | Can't retroactively add | Can satisfy later |
| **Type Checking** | Nominal (by name) | Structural (by shape) |
| **Recommended For** | Internal hierarchy | Plugin architecture |

**plume_nav_sim uses protocols for maximum extensibility.**

---

## Next Steps

- [Custom Rewards Guide](custom_rewards.md) - Implement reward functions
- [Custom Observations Guide](custom_observations.md) - Implement observation models
- [Custom Actions Guide](custom_actions.md) - Implement action processors
- [External Libraries Guide](external_libraries.md) - Build extension packages
