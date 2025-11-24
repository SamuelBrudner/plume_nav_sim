# Component Interfaces Contract

**Component:** Dependency Injection Architecture  
**Version:** 1.0.0  
**Date:** 2025-10-01  
**Status:** CANONICAL - Architectural foundation for extensibility

---

## 📦 Public API for Extensions

### PlumeSearchEnv Public Attributes

**For Wrappers and Subclasses:**

```python
class PlumeSearchEnv(gym.Env):
    """
    Public Attributes (safe to access):
        agent_state (AgentState): Current agent position and orientation
        plume_field (ConcentrationField): Current plume for sampling
        grid_size (GridSize): Environment bounds
        step_count (int): Current episode step number
        
    Public Methods:
        _get_env_state() -> Dict[str, Any]: Extensible state assembly
            Override in subclasses to add custom state fields
    """
```

### Extension Points

**Gymnasium Wrapper Pattern:**

```python
class MyWrapper(gym.Wrapper):
    def step(self, action):
        # Access public attributes
        position = self.env.agent_state.position
        orientation = self.env.agent_state.orientation
        # ... your logic
```

**Subclass Pattern:**

```python
class MyEnv(PlumeSearchEnv):
    def _get_env_state(self):
        state = super()._get_env_state()
        state['my_custom_field'] = self.my_data  # Add custom state
        return state
```

---

## 🎯 Purpose

Define the **complete component interface architecture** enabling:

- Pluggable components via dependency injection
- Config-as-code for reproducible research
- Research flexibility without environment modification
- Clean separation of concerns

This document provides the **high-level architecture** tying together:

- `reward_function_interface.md` - Reward computation
- `observation_model_interface.md` - Sensor models
- `action_processor_interface.md` - Movement models

---

## 📐 Architectural Overview

### Component Dependency Graph

```
┌─────────────────────────────────────┐
│      PlumeSearchEnv                 │
│                                     │
│  Orchestrates components via        │
│  dependency injection               │
└──────────┬──────────────────────────┘
           │
           ├─── plume_model: BasePlumeModel
           │    └─ Provides: ConcentrationField
           │
           ├─── reward_fn: RewardFunction
           │    └─ Computes: reward from state transition
           │
           ├─── observation_model: ObservationModel
           │    └─ Defines: observation_space
           │    └─ Computes: observation from state
           │
           └─── action_processor: ActionProcessor
                └─ Defines: action_space
                └─ Computes: new position from action
```

### Data Flow

```
Environment.step(action):
  1. Store previous state
     prev_state = copy.copy(self.agent_state)
  
  2. Process action to get new state
     new_state = action_processor.process_action(action, self.agent_state, self.grid_size)
     → AgentState with updated position and/or orientation
  
  3. Update environment agent state
     self.agent_state = new_state
  
  4. Assemble environment state dictionary
     env_state = self._get_env_state()  # Contains agent_state, plume_field, grid_size, time_step, etc.
  
  5. Get observation from sensor(s)
     observation = observation_model.get_observation(env_state)
     → Observation matching observation_space
  
  6. Compute reward from transition
     reward = reward_fn.compute_reward(prev_state, action, new_state, plume_field)
     → float reward value
  
  7. Check termination conditions
     → terminated, truncated
  
  8. Return (observation, reward, terminated, truncated, info)
```

### Environment State Dictionary (env_state)

All ObservationModel implementations consume a dictionary `env_state` assembled by the environment. The canonical keys are:

- `agent_state`: AgentState (position, orientation, counters)
- `plume_field`: ConcentrationField or 2D ndarray (depending on sensor needs)
- `grid_size`: GridSize (width, height) for bounds checks
- `time_step`: int (optional) current step index
- `goal_location`: Coordinates (optional) if relevant to sensors
- `wind_field`: WindField (optional) when wind is configured; may be absent for odor-only simulations

Implementations may ignore unused keys. Custom environments or wrappers may extend `env_state` by overriding `_get_env_state()` while preserving these core keys for compatibility.

```

---

## 🔧 Constructor Pattern

### Full Dependency Injection

```python
class PlumeSearchEnv(gym.Env):
    """Plume navigation environment with injectable components.
    
    Supports three initialization patterns:
    1. Component injection (power users)
    2. Config object (reproducible research)
    3. Keyword arguments (backward compatibility)
    """
    
    def __init__(
        self,
        # Pattern 1: Direct component injection
        plume_model: Optional[BasePlumeModel] = None,
        reward_fn: Optional[RewardFunction] = None,
        observation_model: Optional[ObservationModel] = None,
        action_processor: Optional[ActionProcessor] = None,
        
        # Pattern 2: Config object
        config: Optional[EnvironmentConfig] = None,
        
        # Pattern 3: Legacy kwargs (backward compatibility)
        grid_size: Optional[Tuple[int, int]] = None,
        source_location: Optional[Tuple[int, int]] = None,
        max_steps: Optional[int] = None,
        goal_radius: Optional[float] = None,
        **kwargs
    ):
        """Initialize environment with flexible component configuration.
        
        Priority order:
        1. Explicit components (plume_model, reward_fn, etc.)
        2. Config object
        3. Kwargs → create default components
        """
        
        if config is not None:
            # Pattern 2: Build from config
            self._init_from_config(config)
        elif plume_model is not None:
            # Pattern 1: Use injected components
            self.plume = plume_model
            self.reward_fn = reward_fn or SparseGoalReward()
            self.obs_model = observation_model or ConcentrationSensor()
            self.action_proc = action_processor or DiscreteGridActions()
            # Extract other params from kwargs
            self.grid_size = kwargs.get('grid_size', (128, 128))
            self.max_steps = kwargs.get('max_steps', 1000)
        else:
            # Pattern 3: Build from kwargs (backward compatible)
            self._init_from_kwargs(
                grid_size=grid_size,
                source_location=source_location,
                max_steps=max_steps,
                goal_radius=goal_radius,
                **kwargs
            )
        
        # Action and observation spaces from components
        self.action_space = self.action_proc.action_space
        self.observation_space = self.obs_model.observation_space
        
        # Initialize episode state
        self._reset_episode_state()
    
    def _init_from_config(self, config: EnvironmentConfig):
        """Initialize from config object."""
        # Use factories to instantiate components
        self.plume = create_plume_model(config.plume)
        self.reward_fn = create_reward_function(config.reward)
        self.obs_model = create_observation_model(config.observation)
        self.action_proc = create_action_processor(config.action)
        
        # Extract environment params
        self.grid_size = GridSize(*config.grid_size)
        self.max_steps = config.max_steps
    
    def _init_from_kwargs(
        self,
        grid_size: Optional[Tuple[int, int]],
        source_location: Optional[Tuple[int, int]],
        max_steps: Optional[int],
        goal_radius: Optional[float],
        **kwargs
    ):
        """Initialize from kwargs (backward compatible)."""
        # Extract or use defaults
        self.grid_size = GridSize(*(grid_size or (128, 128)))
        self.max_steps = max_steps or 1000
        
        source = source_location or (self.grid_size.width // 2, self.grid_size.height // 2)
        sigma = kwargs.get('sigma', 12.0)
        
        # Create default components matching old behavior
        self.plume = StaticGaussianPlume(
            source_location=Coordinates(*source),
            sigma=sigma,
            grid_size=self.grid_size
        )
        
        self.reward_fn = SparseGoalReward(
            goal_radius=goal_radius or 1.0,
            source_location=self.plume.source_location
        )
        
        self.obs_model = ConcentrationSensor()
        self.action_proc = DiscreteGridActions()
```

---

## 🧪 Integration Tests

### Test All Three Initialization Patterns

```python
class TestComponentIntegration:
    """Test that all initialization patterns work."""
    
    def test_component_injection(self):
        """Pattern 1: Direct component injection."""
        custom_reward = DenseDistanceReward(decay=0.95)
        custom_obs = AntennaeArrayObservation([(0,0), (1,0)])
        
        env = PlumeSearchEnv(
            reward_fn=custom_reward,
            observation_model=custom_obs
        )
        
        # Verify components were used
        assert env.reward_fn is custom_reward
        assert env.obs_model is custom_obs
        
        # Verify environment works
        obs, info = env.reset(seed=42)
        obs, reward, terminated, truncated, info = env.step(0)
        assert not math.isnan(reward)
    
    def test_config_initialization(self):
        """Pattern 2: Config object."""
        config = EnvironmentConfig(
            grid_size=(64, 64),
            max_steps=500,
            plume=PlumeConfig(model_type='static_gaussian', sigma=10.0),
            reward=RewardConfig(type='dense', decay_rate=0.95),
            observation=ObservationConfig(type='single_sensor')
        )
        
        env = PlumeSearchEnv(config=config)
        
        # Verify config was applied
        assert env.grid_size == GridSize(64, 64)
        assert env.max_steps == 500
        assert isinstance(env.reward_fn, DenseDistanceReward)
        
        # Verify environment works
        obs, info = env.reset(seed=42)
        assert obs.shape == env.observation_space.shape
    
    def test_kwargs_backward_compatibility(self):
        """Pattern 3: Legacy kwargs."""
        env = PlumeSearchEnv(
            grid_size=(100, 100),
            source_location=(50, 50),
            max_steps=750,
            goal_radius=2.0,
            sigma=15.0
        )
        
        # Verify defaults were created
        assert isinstance(env.reward_fn, SparseGoalReward)
        assert isinstance(env.obs_model, SingleSensorObservation)
        assert isinstance(env.action_proc, DiscreteGridActions)
        
        # Verify environment works
        obs, info = env.reset(seed=42)
        obs, reward, terminated, truncated, info = env.step(1)
        assert reward in (0.0, 1.0)  # Sparse reward
```

---

## 📋 Factory Functions

### Component Factories

```python
# plume_nav_sim/factories.py

def create_plume_model(config: PlumeConfig) -> BasePlumeModel:
    """Factory for plume models."""
    if config.model_type == 'static_gaussian':
        return StaticGaussianPlume(
            sigma=config.sigma,
            source_location=Coordinates(*config.source_location),
            grid_size=GridSize(*config.grid_size)
        )
    elif config.model_type == 'turbulent':
        return TurbulentPlume(**config.custom_params)
    else:
        # Check registry for custom types
        if config.model_type in _PLUME_MODEL_REGISTRY:
            model_class = _PLUME_MODEL_REGISTRY[config.model_type]
            return model_class(**config.custom_params)
        raise ValueError(f"Unknown plume type: {config.model_type}")

def create_reward_function(config: RewardConfig) -> RewardFunction:
    """Factory for reward functions."""
    if config.type == 'sparse':
        return SparseGoalReward(
            goal_radius=config.goal_radius,
            source_location=config.source_location
        )
    elif config.type == 'dense':
        return DenseDistanceReward(
            goal_radius=config.goal_radius,
            decay_rate=config.decay_rate
        )
    elif config.type == 'shaped':
        return PotentialBasedReward(
            potential_fn=config.potential_fn
        )
    else:
        raise ValueError(f"Unknown reward type: {config.type}")

def create_observation_model(config: ObservationConfig) -> ObservationModel:
    """Factory for observation models."""
    if config.type == 'single_sensor':
        return SingleSensorObservation()
    elif config.type == 'antenna_array':
        return AntennaeArrayObservation(
            antenna_positions=config.antenna_positions
        )
    elif config.type == 'temporal':
        return TemporalObservation(
            history_length=config.history_length
        )
    else:
        raise ValueError(f"Unknown observation type: {config.type}")

def create_action_processor(config: ActionConfig) -> ActionProcessor:
    """Factory for action processors."""
    if config.type == 'discrete_grid':
        return DiscreteGridActions(step_size=config.step_size)
    elif config.type == 'continuous':
        return ContinuousActions(max_speed=config.max_speed)
    elif config.type == 'eight_direction_stay':
        return EightDirectionActions(step_size=config.step_size)
    else:
        raise ValueError(f"Unknown action type: {config.type}")
```

---

## 🔄 Component Lifecycle

### Reset Behavior

Components with internal state must implement `reset()`:

```python
class PlumeSearchEnv:
    def reset(self, *, seed=None, options=None):
        """Reset environment and component state."""
        # Reset RNG
        if seed is not None:
            self._rng, seed = seeding.np_random(seed)
        
        # Reset components with state
        if hasattr(self.obs_model, 'reset'):
            self.obs_model.reset()
        
        if hasattr(self.reward_fn, 'reset'):
            self.reward_fn.reset()
        
        # Initialize episode
        self._reset_episode_state()
        
        # Get initial observation
        env_state = self._get_env_state()
        observation = self.obs_model.get_observation(env_state)
        
        info = self._get_info()
        
        return observation, info
```

---

## ✅ Verification Checklist

Component architecture MUST satisfy:

### Structural Requirements

- [ ] All four component interfaces defined
- [ ] Environment accepts component injection
- [ ] Config object can instantiate all components
- [ ] Backward compatibility maintained
- [ ] Factory functions for all component types

### Behavioral Requirements

- [ ] Components are truly swappable
- [ ] Action/observation spaces from components
- [ ] No hardcoded component logic in environment
- [ ] Components follow their interface contracts
- [ ] All property tests pass

### Integration Requirements

- [ ] Three initialization patterns work
- [ ] Components interact correctly during step()
- [ ] Reset behavior handles stateful components
- [ ] Config serialization/deserialization works
- [ ] Example configs for common scenarios

---

## 📚 Related Contracts

- `reward_function_interface.md` - Reward function protocol
- `observation_model_interface.md` - Observation model protocol
- `action_processor_interface.md` - Action processor protocol
- `environment_state_machine.md` - Environment lifecycle
- `core_types.md` - Shared type definitions

---

**Last Updated:** 2025-10-01  
**Implementation Order:**

1. Create protocol definitions
2. Extract existing logic to concrete classes
3. Update environment constructor
4. Implement factory functions
5. Add config layer
6. Write integration tests
