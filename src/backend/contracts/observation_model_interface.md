# Observation Model Interface Contract

**Component:** Observation Model Abstraction  
**Version:** 1.0.0  
**Date:** 2025-10-01  
**Status:** CANONICAL - All implementations MUST conform

---

## 📦 Type Dependencies

This contract references types defined in other contracts:

- `AgentState`: See `core_types.md` - Contains position and orientation
- `ConcentrationField`: See `concentration_field.md` - Plume sampling interface
- `Coordinates`: See `core_types.md` - 2D integer grid position
- `GridSize`: See `core_types.md` - Grid dimensions
- `ObservationType`: `Union[np.ndarray, Dict[str, np.ndarray], Tuple[np.ndarray, ...]]`

---

## 🎯 Purpose

Define the **universal interface** for observation models, enabling:

- Diverse sensor configurations (single sensor, antenna arrays, temporal buffers)
- Pluggable observation spaces without environment modification
- Research flexibility for different sensing modalities
- Clean separation between environment state and sensor readings

---

## 📐 Interface Definition

### Type Signature

```python
ObservationModel: (EnvironmentState) → ObservationSpace × Observation

Where:
  - Defines: observation_space (any Gymnasium Space: Box, Dict, Tuple, etc.)
  - Computes: observation from environment state dictionary
  - Returns: Observation matching space (np.ndarray, dict, tuple, etc.)
```

### Protocol Specification

```python
class ObservationModel(Protocol):
    """Protocol defining observation model interface.
    
    All observation models must conform to this interface to be
    compatible with the environment and config system.
    
    This abstraction is GENERAL - not specific to olfactory sensing.
    Sensors can observe any aspect of environment state: odor, wind,
    obstacles, time, or custom fields added by users.
    """
    
    @property
    def observation_space(self) -> gym.Space:
        """Gymnasium observation space definition.
        
        Can be any Gymnasium space type:
        - Box: Vector/tensor observations (e.g., concentration, wind)
        - Dict: Named multi-modal observations (composition)
        - Tuple: Ordered multi-modal observations
        - Discrete: Categorical observations
        - MultiDiscrete: Multiple categorical observations
        
        Postconditions:
          C1: Returns valid gym.Space instance
          C2: Space is immutable (same instance every call)
          C3: Space fully defines valid observations
        
        Returns:
            Gymnasium Space defining valid observations
        """
        ...
    
    def get_observation(
        self,
        env_state: Dict[str, Any]
    ) -> ObservationType:
        """Compute observation from environment state.
        
        Preconditions:
          P1: env_state contains required keys for this sensor
          P2: env_state['agent_state'] is valid AgentState
          P3: Agent position is within environment bounds
        
        Postconditions:
          C1: observation ∈ self.observation_space
          C2: observation matches space structure (shape, dtype, keys)
          C3: Result is deterministic (same env_state → same observation)
        
        Properties:
          1. Determinism: Same env_state → same observation
          2. Purity: No side effects, no mutations
          3. Validity: observation_space.contains(observation) = True
        
        Args:
            env_state: Environment state dictionary containing:
                Required:
                - 'agent_state': AgentState (position, orientation, etc.)
                
                Common (depends on sensor needs):
                - 'plume_field': ConcentrationField (for olfactory sensors)
                - 'wind_field': WindField (for mechanosensory)
                - 'obstacle_map': np.ndarray (for vision/proximity)
                - 'time_step': int (for temporal context)
                
                Custom fields can be added by users for novel sensors.
        
        Returns:
            Observation matching observation_space.
            Type depends on space: np.ndarray (Box), dict (Dict), tuple (Tuple)
        """
        ...
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return observation model metadata.
        
        Returns:
            Dictionary containing:
            - 'type': str - Observation model type
            - 'modality': str - Sensory modality (olfactory, visual, etc.)
            - 'parameters': dict - Configuration
            - 'required_state_keys': List[str] - Required env_state keys
        """
        ...
```

---

## 🌍 Universal Properties

### Property 1: Space Containment (UNIVERSAL)

```python
∀ env_state:
  obs = get_observation(env_state)
  ⇒ observation_space.contains(obs) = True
```

**Test:**

```python
@given(env_state=env_state_strategy())
def test_observation_in_space(obs_model, env_state):
    """Observation always satisfies space constraints."""
    observation = obs_model.get_observation(env_state)
    
    assert obs_model.observation_space.contains(observation), \
        f"Observation not in space {obs_model.observation_space}"
```

### Property 2: Determinism (UNIVERSAL)

```python
∀ env_state:
  get_observation(env_state) = get_observation(env_state)
```

**Test:**

```python
@given(env_state=env_state_strategy())
def test_observation_deterministic(obs_model, env_state):
    """Same state produces same observation."""
    obs1 = obs_model.get_observation(env_state)
    obs2 = obs_model.get_observation(env_state)
    
    # Use appropriate comparison based on observation type
    if isinstance(obs1, np.ndarray):
        np.testing.assert_array_equal(obs1, obs2)
    elif isinstance(obs1, dict):
        assert obs1.keys() == obs2.keys()
        for key in obs1.keys():
            np.testing.assert_array_equal(obs1[key], obs2[key])
    else:
        assert obs1 == obs2
```

### Property 3: Purity (UNIVERSAL)

```python
∀ env_state: get_observation(env_state) has no side effects

No modification of:
  - env_state dictionary or its contents
  - agent_state
  - Any fields (plume_field, wind_field, etc.)
  - Global state
```

**Test:**

```python
def test_observation_purity(obs_model):
    """Observation computation has no side effects."""
    env_state = {
        'agent_state': AgentState(position=Coordinates(10, 10)),
        'plume_field': create_test_field(),
        'time_step': 5
    }
    
    # Deep copy to check for mutations
    orig_state = copy.deepcopy(env_state)
    
    obs = obs_model.get_observation(env_state)
    
    # Verify no mutations
    assert env_state == orig_state, "env_state was modified"
```

### Property 4: Shape Consistency (UNIVERSAL)

```python
∀ env_state:
  obs = get_observation(env_state)
  ⇒ obs.shape = observation_space.shape  (for Box spaces)
```

**Test:**

```python
def test_observation_shape(obs_model):
    """Observation shape matches space specification."""
    env_state = {
        'agent_state': create_test_agent_state(),
        'plume_field': create_test_field()
    }
    
    obs = obs_model.get_observation(env_state)
    expected_shape = obs_model.observation_space.shape
    
    assert obs.shape == expected_shape, \
        f"Expected shape {expected_shape}, got {obs.shape}"
```

---

## 📊 Implementation Examples

### 1. Concentration Sensor (Default - Most Common)

**Simple, convenient access to odor samples - the 95% use case.**

```python
class ConcentrationSensor:
    """Single concentration value at agent position.
    
    Default observation model - convenient odor sampling.
    This is what most plume navigation research needs.
    """
    
    @property
    def observation_space(self) -> gym.Space:
        return gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )
    
    def get_observation(self, env_state: Dict[str, Any]) -> np.ndarray:
        """Sample concentration at agent position.
        
        Convenient helper for common case - extract what we need from env_state.
        """
        agent_state = env_state['agent_state']
        plume_field = env_state['plume_field']
        
        concentration = plume_field.sample(agent_state.position)
        return np.array([concentration], dtype=np.float32)
    
    def get_metadata(self) -> Dict[str, Any]:
        return {
            'type': 'concentration_sensor',
            'modality': 'olfactory',
            'parameters': {},
            'required_state_keys': ['agent_state', 'plume_field'],
            'sensor_config': {'num_sensors': 1, 'positions': [(0, 0)]}
        }


# Usage - simple and convenient
env = PlumeEnv(
    observation_model=ConcentrationSensor()  # Just works!
)
```

### 2. Antenna Array (Multi-Sensor Olfactory)

```python
@dataclass
class AntennaeArraySensor:
    """Multiple concentration sensors at relative positions.
    
    Enables spatial gradient sensing like biological chemoreceptors.
    Still olfactory, but spatially distributed.
    """
    antenna_positions: List[Tuple[int, int]]  # Relative to agent
    
    @property
    def observation_space(self) -> gym.Space:
        n_sensors = len(self.antenna_positions)
        return gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(n_sensors,),
            dtype=np.float32
        )
    
    def get_observation(self, env_state: Dict[str, Any]) -> np.ndarray:
        """Sample concentration at each antenna position."""
        agent_state = env_state['agent_state']
        plume_field = env_state['plume_field']
        
        concentrations = []
        for dx, dy in self.antenna_positions:
            sensor_pos = Coordinates(
                x=agent_state.position.x + dx,
                y=agent_state.position.y + dy
            )
            # Boundary handling
            if plume_field.grid_size.contains(sensor_pos):
                c = plume_field.sample(sensor_pos)
            else:
                c = 0.0  # Out of bounds = no odor
            concentrations.append(c)
        
        return np.array(concentrations, dtype=np.float32)
    
    def get_metadata(self) -> Dict[str, Any]:
        return {
            'type': 'antenna_array',
            'modality': 'olfactory',
            'parameters': {'antenna_positions': self.antenna_positions},
            'required_state_keys': ['agent_state', 'plume_field'],
            'sensor_config': {
                'num_sensors': len(self.antenna_positions),
                'positions': self.antenna_positions
            }
        }
```

### 3. Wind Sensor (Non-Olfactory)

**Example of sensing a different environmental field.**

Built-in option: `WindVectorSensor` (config: `observation_type='wind_vector'`, optional `noise_std`).
Environment state integration: `env_state['wind_field']` is optional; sensors must return zeros if missing to keep odor-only simulations intact.

```python
class WindSensor:
    """Wind velocity sensor - mechanosensory modality.
    
    Demonstrates how to sense non-olfactory fields.
    Users can add custom fields to env_state for novel sensors.
    """
    
    @property
    def observation_space(self) -> gym.Space:
        # Wind as 2D vector: (vx, vy) in [-1, 1]
        return gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32
        )
    
    def get_observation(self, env_state: Dict[str, Any]) -> np.ndarray:
        """Sample wind velocity at agent position."""
        agent_state = env_state['agent_state']
        wind_field = env_state.get('wind_field')  # Optional field
        
        if wind_field is None:
            # No wind model in environment
            return np.array([0.0, 0.0], dtype=np.float32)
        
        wind_vector = wind_field.sample(agent_state.position)
        return wind_vector.astype(np.float32)
    
    def get_metadata(self) -> Dict[str, Any]:
        return {
            'type': 'wind_sensor',
            'modality': 'mechanosensory',
            'parameters': {},
            'required_state_keys': ['agent_state'],
            'optional_state_keys': ['wind_field']
        }


# Usage - environment needs to provide wind_field
env = PlumeEnv(
    observation_model=WindSensor(),
    wind_model=ConstantWindModel(direction=45, speed=0.5)  # Adds wind_field to env_state
)
```

### 4. Multi-Modal Composition (Using Gymnasium Dict Space)

**Compose multiple sensors without reinventing composition.**

```python
class MultiModalSensor:
    """Compose multiple sensors using Gymnasium's Dict space.
    
    This is THE RIGHT WAY - leverage Gymnasium, don't reinvent.
    """
    
    def __init__(self, sensors: Dict[str, ObservationModel]):
        self.sensors = sensors
        self._space = gym.spaces.Dict({
            name: sensor.observation_space
            for name, sensor in sensors.items()
        })
    
    @property
    def observation_space(self) -> gym.spaces.Dict:
        return self._space
    
    def get_observation(self, env_state: Dict[str, Any]) -> Dict[str, Any]:
        """Get observations from all sensors."""
        return {
            name: sensor.get_observation(env_state)
            for name, sensor in self.sensors.items()
        }
    
    def get_metadata(self) -> Dict[str, Any]:
        return {
            'type': 'multi_modal',
            'modality': 'composite',
            'sensors': {
                name: sensor.get_metadata()
                for name, sensor in self.sensors.items()
            }
        }


# Usage - odor + wind sensing
env = PlumeEnv(
    observation_model=MultiModalSensor({
        'odor': ConcentrationSensor(),
        'wind': WindSensor(),
        'time': TimeStepSensor()
    }),
    wind_model=TurbulentWindModel()
)

# Observation structure:
# {
#     'odor': array([0.42], dtype=float32),
#     'wind': array([0.3, -0.8], dtype=float32),
#     'time': array([45], dtype=int32)
# }
```

### 5. Flattened Composition (For RL Algorithms)

**When you need a single flat vector (many RL libs prefer this).**

```python
class FlattenedMultiSensor:
    """Flatten multiple sensors into single Box observation.
    
    Useful when RL algorithm expects flat feature vector.
    """
    
    def __init__(self, sensors: List[ObservationModel]):
        self.sensors = sensors
        
        # Calculate total dimension
        self._dims = []
        for sensor in sensors:
            space = sensor.observation_space
            if not isinstance(space, gym.spaces.Box):
                raise ValueError("Can only flatten Box spaces")
            self._dims.append(np.prod(space.shape))
        
        self._total_dim = sum(self._dims)
    
    @property
    def observation_space(self) -> gym.spaces.Box:
        return gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._total_dim,),
            dtype=np.float32
        )
    
    def get_observation(self, env_state: Dict[str, Any]) -> np.ndarray:
        """Concatenate all sensor outputs."""
        observations = [
            sensor.get_observation(env_state).flatten()
            for sensor in self.sensors
        ]
        return np.concatenate(observations).astype(np.float32)
    
    def get_metadata(self) -> Dict[str, Any]:
        return {
            'type': 'flattened_multi',
            'modality': 'composite',
            'dimensions': self._dims,
            'total_dim': self._total_dim,
            'sensors': [s.get_metadata() for s in self.sensors]
        }


# Usage - single flat vector
env = PlumeEnv(
    observation_model=FlattenedMultiSensor([
        ConcentrationSensor(),      # dim 1
        WindSensor(),               # dim 2
        AntennaeArraySensor([(0,0), (1,0)])  # dim 2
    ])
    # Total observation: shape (5,) flat vector
)
```

---

## 🧪 Required Test Suite

```python
class TestObservationModelInterface:
    """Universal test suite for observation models."""
    
    @pytest.fixture
    def observation_model(self):
        """Override in concrete test classes."""
        raise NotImplementedError
    
    @given(env_state=env_state_strategy())
    def test_observation_in_space(self, observation_model, env_state):
        """P1: Observation satisfies space constraints."""
        observation = observation_model.get_observation(env_state)
        assert observation_model.observation_space.contains(observation)
        
    @given(env_state=env_state_strategy())
    def test_determinism(self, observation_model, env_state):
        """P2: Deterministic observation."""
        obs1 = observation_model.get_observation(env_state)
        obs2 = observation_model.get_observation(env_state)
        if isinstance(obs1, np.ndarray):
            np.testing.assert_array_equal(obs1, obs2)
        elif isinstance(obs1, dict):
            assert obs1.keys() == obs2.keys()
            for key in obs1:
                np.testing.assert_array_equal(obs1[key], obs2[key])
        
    def test_purity(self, observation_model):
        """P3: No side effects."""
        env_state = create_test_env_state()
        orig = copy.deepcopy(env_state)
        _ = observation_model.get_observation(env_state)
        assert env_state == orig
        
    def test_shape_consistency(self, observation_model):
        """P4: Shape matches space (for Box spaces)."""
        env_state = create_test_env_state()
        obs = observation_model.get_observation(env_state)
        if isinstance(observation_model.observation_space, gym.spaces.Box):
            assert obs.shape == observation_model.observation_space.shape
        
    def test_dtype_consistency(self, observation_model):
        """Dtype matches space specification."""
        env_state = create_test_env_state()
        obs = observation_model.get_observation(env_state)
        if isinstance(observation_model.observation_space, gym.spaces.Box):
            assert obs.dtype == observation_model.observation_space.dtype
        
    def test_space_immutability(self, observation_model):
        """observation_space returns same instance."""
        space1 = observation_model.observation_space
        space2 = observation_model.observation_space
        assert space1 is space2
```

---

## 🔗 Integration Requirements

### Environment Integration

- Environment state **may** include `wind_field` when wind is configured; sensors must treat it as optional and degrade gracefully (e.g., return zeros) to keep odor-only simulations intact.

- Config-driven environments (e.g., `EnvironmentConfig` → `create_environment_from_config`) automatically inject a constant wind field when `wind` is provided and pass it through `env_state['wind_field']`.

```python
class PlumeEnv:
    def __init__(
        self,
        observation_model: Optional[ObservationModel] = None,
        # Optional environmental features
        wind_model: Optional[WindModel] = None,
        **kwargs
    ):
        # Default to convenient odor sensing
        self.obs_model = observation_model or ConcentrationSensor()

        # Observation space from model (can be Box, Dict, Tuple, etc.)
        self.observation_space = self.obs_model.observation_space

        # Optional environmental features
        self.wind_model = wind_model
    
    def _get_env_state(self) -> Dict[str, Any]:
        """Assemble environment state dictionary for sensors.
        
        This is the extensibility point - users can subclass and add
        custom fields for novel sensors.
        """
        state = {
            'agent_state': self.agent_state,
            'plume_field': self.plume_field,
            'time_step': self.step_count,
            'grid_size': self.grid_size
        }

        # Optional fields (only if environment has them)
        if self.wind_model is not None:
            state['wind_field'] = self.wind_model.get_field()
        
        # Users can extend this in subclasses:
        # if hasattr(self, 'obstacle_map'):
        #     state['obstacle_map'] = self.obstacle_map
        
        return state
    
    def reset(self, **kwargs):
        # Initialize episode state
        self._reset_episode_state()
        
        # Reset observation model if it has state (e.g., history buffer)
        if hasattr(self.obs_model, 'reset'):
            self.obs_model.reset()
        
        # Get initial observation
        env_state = self._get_env_state()
        observation = self.obs_model.get_observation(env_state)
        
        info = self._get_info()
        return observation, info
    
    def step(self, action):
        # Process action, update agent_state
        # ... (action processing, reward calculation, etc.) ...
        
        # Get observation from sensor(s)
        env_state = self._get_env_state()
        observation = self.obs_model.get_observation(env_state)
        
        return observation, reward, terminated, truncated, info
```

---

## ⚠️ Common Implementation Errors

### ❌ Wrong: Observation Outside Space

```python
class BadObservation:
    @property
    def observation_space(self):
        return Box(low=0.0, high=1.0, shape=(1,))
    
    def get_observation(self, state, field):
        return np.array([2.0])  # ❌ Outside [0, 1] bounds!
```

### ❌ Wrong: Shape Mismatch

```python
class BadObservation:
    @property
    def observation_space(self):
        return Box(low=0.0, high=1.0, shape=(3,))
    
    def get_observation(self, state, field):
        return np.array([0.5, 0.7])  # ❌ Shape (2,) != (3,)
```

### ✅ Correct: Always Valid

```python
class GoodObservation:
    @property
    def observation_space(self):
        return Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
    
    def get_observation(self, state, field):
        concentration = field.sample(state.position)
        # Clamp to ensure validity
        clamped = np.clip(concentration, 0.0, 1.0)
        return np.array([clamped], dtype=np.float32)
```

---

## 📊 Verification Checklist

Implementation MUST satisfy:

- [ ] Implements ObservationModel protocol
- [ ] observation_space is valid gym.Space
- [ ] observation_space is immutable (same instance)
- [ ] get_observation() returns valid observations
- [ ] Observations match space shape and dtype
- [ ] Deterministic (same state → same observation)
- [ ] Pure (no side effects)
- [ ] Passes all property tests
- [ ] Handles boundary cases (agent at edge)

---

**Last Updated:** 2025-10-01  
**Related Contracts:**

- `core_types.md` - AgentState definition
- `concentration_field.md` - ConcentrationField specification
