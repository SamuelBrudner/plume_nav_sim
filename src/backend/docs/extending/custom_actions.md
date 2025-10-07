# Creating Custom Action Processors

## Implementing custom movement patterns for plume_nav_sim

---

## Quick Start

```python
from plume_nav_sim.interfaces import ActionProcessor
from plume_nav_sim.core.state import AgentState
from plume_nav_sim.core.geometry import Coordinates, GridSize
import gymnasium as gym
import numpy as np
from dataclasses import replace

class DiagonalActions:
    """8-direction movement including diagonals."""
    
    def __init__(self, step_size=1):
        self.step_size = step_size
        self._action_space = gym.spaces.Discrete(8)
        # 0=N, 1=NE, 2=E, 3=SE, 4=S, 5=SW, 6=W, 7=NW
        self.deltas = [
            (0, step_size),    # N
            (step_size, step_size),   # NE
            (step_size, 0),    # E
            (step_size, -step_size),  # SE
            (0, -step_size),   # S
            (-step_size, -step_size), # SW
            (-step_size, 0),   # W
            (-step_size, step_size),  # NW
        ]
    
    @property
    def action_space(self):
        return self._action_space
    
    def process_action(self, action, current_state, grid_size):
        dx, dy = self.deltas[action]
        
        # Compute new position with boundary clamping
        new_x = max(0, min(grid_size.width - 1, current_state.position.x + dx))
        new_y = max(0, min(grid_size.height - 1, current_state.position.y + dy))
        
        # Return new state (immutable update)
        return replace(
            current_state,
            position=Coordinates(new_x, new_y)
        )
    
    def validate_action(self, action):
        return isinstance(action, (int, np.integer)) and 0 <= action < 8
    
    def get_metadata(self):
        return {
            "type": "diagonal_actions",
            "n_actions": 8,
            "parameters": {"step_size": self.step_size}
        }
```

---

## The ActionProcessor Protocol

```python
@property
def action_space(self) -> gym.Space: ...

def process_action(
    self, action: int, current_state: AgentState, grid_size: GridSize
) -> AgentState: ...

def validate_action(self, action: int) -> bool: ...

def get_metadata(self) -> Dict[str, Any]: ...
```

**Key Properties:**
- **Purity**: Never mutate `current_state` or `grid_size`
- **Boundary Safety**: Always return position within grid
- **Determinism**: Same inputs → same output
- **Returns New State**: Use `dataclasses.replace()` or create new `AgentState`

---

## Implementation Examples

### Example 1: Continuous Actions

```python
class ContinuousActions:
    """Continuous (dx, dy) movement."""
    
    def __init__(self, max_step=2.0):
        self.max_step = max_step
        self._action_space = gym.spaces.Box(
            low=-max_step, high=max_step,
            shape=(2,), dtype=np.float32
        )
    
    @property
    def action_space(self):
        return self._action_space
    
    def process_action(self, action, current_state, grid_size):
        dx, dy = action
        
        # Clip to max step
        dx = np.clip(dx, -self.max_step, self.max_step)
        dy = np.clip(dy, -self.max_step, self.max_step)
        
        # Compute new position
        new_x = current_state.position.x + dx
        new_y = current_state.position.y + dy
        
        # Clamp to grid
        new_x = max(0, min(grid_size.width - 1, int(new_x)))
        new_y = max(0, min(grid_size.height - 1, int(new_y)))
        
        return replace(current_state, position=Coordinates(new_x, new_y))
    
    def validate_action(self, action):
        return (isinstance(action, np.ndarray) and 
                action.shape == (2,) and
                np.all(np.abs(action) <= self.max_step))
    
    def get_metadata(self):
        return {"type": "continuous", "max_step": self.max_step}
```

### Example 2: Altitude + Translation

```python
class AltitudeActions:
    """3D movement: up/down/left/right + altitude change."""
    
    def __init__(self, step_size=1, max_altitude=10):
        self.step_size = step_size
        self.max_altitude = max_altitude
        self._action_space = gym.spaces.MultiDiscrete([5, 3])
        # [0-4]: still, up, right, down, left
        # [0-2]: altitude down, stay, up
    
    @property
    def action_space(self):
        return self._action_space
    
    def process_action(self, action, current_state, grid_size):
        direction, altitude_change = action
        
        # Movement
        deltas = [(0,0), (0,1), (1,0), (0,-1), (-1,0)]
        dx, dy = deltas[direction]
        new_x = max(0, min(grid_size.width - 1, current_state.position.x + dx * self.step_size))
        new_y = max(0, min(grid_size.height - 1, current_state.position.y + dy * self.step_size))
        
        # Altitude (stored in orientation for simplicity)
        altitude_delta = altitude_change - 1  # -1, 0, +1
        new_altitude = max(0, min(self.max_altitude, 
                                   current_state.orientation + altitude_delta))
        
        return replace(
            current_state,
            position=Coordinates(new_x, new_y),
            orientation=new_altitude  # Hijack orientation for altitude
        )
    
    def validate_action(self, action):
        return (len(action) == 2 and
                0 <= action[0] < 5 and
                0 <= action[1] < 3)
    
    def get_metadata(self):
        return {
            "type": "altitude_actions",
            "max_altitude": self.max_altitude
        }
```

---

## Testing

```python
from tests.contracts.test_action_processor_interface import TestActionProcessorInterface

class TestDiagonalActions(TestActionProcessorInterface):
    @pytest.fixture
    def action_processor(self):
        return DiagonalActions(step_size=1)
    
    # Add specific tests
    def test_diagonal_movement(self):
        processor = DiagonalActions()
        state = AgentState(position=Coordinates(10, 10))
        grid = GridSize(128, 128)
        
        # Action 1 = NE
        new_state = processor.process_action(1, state, grid)
        assert new_state.position.x == 11
        assert new_state.position.y == 11
```

---

## Common Patterns

**Variable step sizes:**
```python
class VariableStepActions:
    def __init__(self, step_sizes=[1, 2, 5]):
        self.step_sizes = step_sizes
        # Action = (direction, step_size_index)
        self._action_space = gym.spaces.MultiDiscrete([4, len(step_sizes)])
```

**Momentum:**
```python
class MomentumActions:
    def process_action(self, action, current_state, grid_size):
        # Use movement_history from current_state
        prev_move = self._get_last_move(current_state)
        new_move = self._apply_action(action, prev_move)
        # Apply with momentum...
```

---

## See Also

- [Protocol Interfaces](protocol_interfaces.md)
- `contracts/action_processor_interface.md`
- `plume_nav_sim/actions/` - Built-in implementations
