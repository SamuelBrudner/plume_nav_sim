# Creating Custom Observation Models

**Implementing custom sensors for plume_nav_sim**

---

## Quick Start

```python
from plume_nav_sim.interfaces import ObservationModel
import gymnasium as gym
import numpy as np

class GradientSensor:
    """Observes concentration gradient at agent position."""
    
    def __init__(self):
        # Define observation space
        self._observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32
        )
    
    @property
    def observation_space(self):
        return self._observation_space
    
    def get_observation(self, env_state):
        agent_state = env_state["agent_state"]
        plume_field = env_state["plume_field"]
        
        # Compute gradient
        pos = agent_state.position
        dx = self._gradient_x(plume_field, pos)
        dy = self._gradient_y(plume_field, pos)
        
        return np.array([dx, dy], dtype=np.float32)
    
    def get_metadata(self):
        return {
            "type": "gradient_sensor",
            "modality": "olfactory",
            "parameters": {}
        }
    
    def _gradient_x(self, field, pos):
        if pos.x == 0:
            return float(field[pos.y, pos.x+1] - field[pos.y, pos.x])
        elif pos.x == field.shape[1] - 1:
            return float(field[pos.y, pos.x] - field[pos.y, pos.x-1])
        else:
            return float(field[pos.y, pos.x+1] - field[pos.y, pos.x-1]) / 2.0
    
    def _gradient_y(self, field, pos):
        if pos.y == 0:
            return float(field[pos.y+1, pos.x] - field[pos.y, pos.x])
        elif pos.y == field.shape[0] - 1:
            return float(field[pos.y, pos.x] - field[pos.y-1, pos.x])
        else:
            return float(field[pos.y+1, pos.x] - field[pos.y-1, pos.x]) / 2.0
```

---

## The ObservationModel Protocol

```python
@property
def observation_space(self) -> gym.Space: ...

def get_observation(self, env_state: Dict[str, Any]) -> ObservationType: ...

def get_metadata(self) -> Dict[str, Any]: ...
```

### env_state Dictionary

The `env_state` dict contains:

| Key | Type | Description |
|-----|------|-------------|
| `agent_state` | `AgentState` | Position, orientation, step_count |
| `plume_field` | `np.ndarray` | 2D concentration array (height, width) |
| `concentration_field` | `ConcentrationField` | Full field object |
| `goal_location` | `Coordinates` | Target position |
| `grid_size` | `GridSize` | Environment bounds |
| `step_count` | `int` | Current step number |

---

## Implementation Examples

### Example 1: Multi-Point Sampler

```python
class MultiPointSensor:
    """Sample at multiple fixed offsets."""
    
    def __init__(self, sample_offsets=[(0,0), (1,0), (0,1), (-1,0), (0,-1)]):
        self.offsets = sample_offsets
        self._observation_space = gym.spaces.Box(
            low=0.0, high=1.0, 
            shape=(len(sample_offsets),),
            dtype=np.float32
        )
    
    @property
    def observation_space(self):
        return self._observation_space
    
    def get_observation(self, env_state):
        agent_pos = env_state["agent_state"].position
        plume_field = env_state["plume_field"]
        grid_size = env_state["grid_size"]
        
        observations = []
        for dx, dy in self.offsets:
            x = np.clip(agent_pos.x + dx, 0, grid_size.width - 1)
            y = np.clip(agent_pos.y + dy, 0, grid_size.height - 1)
            conc = float(plume_field[y, x])
            observations.append(conc)
        
        return np.array(observations, dtype=np.float32)
    
    def get_metadata(self):
        return {
            "type": "multi_point",
            "modality": "olfactory",
            "parameters": {"offsets": self.offsets}
        }
```

### Example 2: Temporal Stack

```python
from collections import deque

class TemporalStackSensor:
    """Stack observations over time (e.g., velocity estimation)."""
    
    def __init__(self, stack_size=4):
        self.stack_size = stack_size
        self.history = deque(maxlen=stack_size)
        self._observation_space = gym.spaces.Box(
            low=0.0, high=1.0,
            shape=(stack_size,),
            dtype=np.float32
        )
    
    @property
    def observation_space(self):
        return self._observation_space
    
    def get_observation(self, env_state):
        agent_pos = env_state["agent_state"].position
        plume_field = env_state["plume_field"]
        
        # Current concentration
        current = float(plume_field[agent_pos.y, agent_pos.x])
        
        # Add to history
        self.history.append(current)
        
        # Pad if needed
        obs = list(self.history)
        while len(obs) < self.stack_size:
            obs.insert(0, 0.0)
        
        return np.array(obs, dtype=np.float32)
    
    def get_metadata(self):
        return {
            "type": "temporal_stack",
            "modality": "olfactory",
            "parameters": {"stack_size": self.stack_size},
            "required_state_keys": ["agent_state", "plume_field"]
        }
```

---

## Testing

```python
from tests.contracts.test_observation_model_interface import TestObservationModelInterface

class TestGradientSensor(TestObservationModelInterface):
    @pytest.fixture
    def observation_model(self):
        return GradientSensor()
    
    # Add specific tests
    def test_gradient_at_peak(self):
        model = GradientSensor()
        # Create test field with peak at center
        field = create_test_field()
        env_state = {
            "agent_state": AgentState(position=Coordinates(64, 64)),
            "plume_field": field
        }
        
        obs = model.get_observation(env_state)
        assert obs.shape == (2,)
        # At peak, gradient should be near zero
        assert np.allclose(obs, [0.0, 0.0], atol=0.1)
```

---

## See Also

- [Protocol Interfaces](protocol_interfaces.md)
- `contracts/observation_model_interface.md`
- `plume_nav_sim/observations/` - Built-in implementations
