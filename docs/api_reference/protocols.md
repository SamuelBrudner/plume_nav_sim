# Protocol Interfaces API Reference

## Table of Contents

- [Protocol System Overview](#protocol-system-overview)
- [Core Navigation Protocol](#core-navigation-protocol)
- [Source Management Protocols](#source-management-protocols)
- [Boundary Policy Protocols](#boundary-policy-protocols)
- [Action Interface Protocols](#action-interface-protocols)
- [Agent Initialization Protocols](#agent-initialization-protocols)
- [Data Recording Protocols](#data-recording-protocols)
- [Statistics Aggregation Protocols](#statistics-aggregation-protocols)
- [Environment Modeling Protocols](#environment-modeling-protocols)
- [Sensor Interface Protocols](#sensor-interface-protocols)
- [Space Construction Protocols](#space-construction-protocols)
- [Implementation Guidelines](#implementation-guidelines)
- [Configuration Patterns](#configuration-patterns)
- [Migration and Compatibility](#migration-and-compatibility)

## Protocol System Overview

The plume_nav_sim v1.0 architecture is built on a comprehensive protocol-based system that enables zero-code extensibility through configuration-driven component composition. This design establishes the foundation for pluggable component architecture where different implementations can be seamlessly swapped via Hydra configuration without modifying core simulation logic.

### Pluggable Architecture Explanation

The protocol system enables true pluggability through structural typing via Python's `Protocol` classes from the `typing` module. Each major subsystem in plume_nav_sim is defined by a protocol interface that specifies:

- **Method signatures** with complete type annotations
- **Performance requirements** for real-time simulation compatibility  
- **Behavioral contracts** that implementations must satisfy
- **Integration points** with other system components

Key architectural benefits:

- **Modular Design**: Components can be developed and tested independently
- **Type Safety**: Protocol compliance is enforced at development time via type checking
- **Performance Guarantees**: Protocol specifications include timing requirements
- **Research Flexibility**: New algorithms can be integrated without framework changes

### Protocol-Based Dependency Injection

The protocol system integrates seamlessly with Hydra's dependency injection capabilities, enabling runtime component composition:

```yaml
# Configuration-driven component selection
navigator:
  _target_: plume_nav_sim.core.controllers.SingleAgentController
  position: [10.0, 20.0]
  max_speed: 2.0
  
source:
  _target_: plume_nav_sim.core.sources.PointSource
  position: [50.0, 50.0]
  emission_rate: 1000.0
  
boundary_policy:
  _target_: plume_nav_sim.core.boundaries.TerminatePolicy
  domain_bounds: [100, 100]
```

### Zero-Code Extensibility

The protocol-based design enables researchers to extend functionality without modifying core library code:

1. **Implement Protocol Interface**: Create new component implementing the appropriate protocol
2. **Register Configuration**: Add Hydra configuration targeting your implementation
3. **Use Immediately**: Framework automatically discovers and integrates your component

```python
class CustomSource(SourceProtocol):
    """Custom source implementation following SourceProtocol."""
    
    def get_emission_rate(self) -> float:
        # Your custom emission logic
        return self._compute_dynamic_rate()
    
    def get_position(self) -> Tuple[float, float]:
        # Your custom positioning logic
        return self._current_position
    
    def update_state(self, dt: float = 1.0) -> None:
        # Your custom temporal dynamics
        self._evolve_state(dt)
```

### Hydra Configuration Integration

The protocol system leverages Hydra's advanced configuration management:

- **Structured Configs**: Type-safe configuration with Pydantic validation
- **Config Composition**: Mix and match components via config groups
- **Override Patterns**: Runtime parameter modification via CLI or API
- **Multi-Run Support**: Systematic parameter sweeps for research

### V1.0 Design Principles

The v1.0 protocol architecture embodies several key design principles:

1. **Separation of Concerns**: Each protocol addresses a single system responsibility
2. **Performance First**: All protocols specify timing requirements for real-time simulation
3. **Research Enabling**: Protocols provide extension points for algorithm development
4. **Production Ready**: Full error handling, validation, and monitoring capabilities
5. **Framework Agnostic**: Compatible with multiple RL libraries and research frameworks

## Core Navigation Protocol

### NavigatorProtocol Interface Specification

The `NavigatorProtocol` defines the fundamental contract for navigation controllers in plume_nav_sim v1.0. This protocol prescribes the exact properties and methods that any concrete navigator implementation must provide, ensuring uniform API across single-agent and multi-agent navigation logic.

```python
from typing import Protocol, Optional, Union, Tuple, Any, Dict
import numpy as np
from gymnasium import spaces

@runtime_checkable
class NavigatorProtocol(Protocol):
    """Core protocol interface defining navigation controller contract."""
```

#### Core State Properties

**Agent Positions**
```python
@property
def positions(self) -> np.ndarray:
    """
    Get current agent position(s) as numpy array.
    
    Returns:
        np.ndarray: Agent positions with shape:
            - Single agent: (1, 2) for [x, y] coordinates
            - Multi-agent: (n_agents, 2) for [[x1, y1], [x2, y2], ...]
            
    Performance: O(1) property access with no computation during retrieval.
    """
```

**Agent Orientations**
```python
@property
def orientations(self) -> np.ndarray:
    """
    Get current agent orientation(s) in degrees.
    
    Returns:
        np.ndarray: Agent orientations with shape:
            - Single agent: (1,) for [orientation]
            - Multi-agent: (n_agents,) for [ori1, ori2, ...]
            
    Notes:
        - Orientations in degrees: 0° = right (positive x-axis)
        - 90° = up (negative y-axis) following navigation conventions
        - Values normalized to [0, 360) range
    """
```

**Agent Speeds and Constraints**
```python
@property
def speeds(self) -> np.ndarray:
    """Current agent speed(s) in units per time step."""

@property  
def max_speeds(self) -> np.ndarray:
    """Maximum allowed speed(s) for each agent."""

@property
def angular_velocities(self) -> np.ndarray:
    """Current agent angular velocity/velocities in degrees per second."""

@property
def num_agents(self) -> int:
    """Total number of agents managed by this navigator."""
```

#### Method Signatures

**State Management**
```python
def reset(self, **kwargs: Any) -> None:
    """
    Reset navigator to initial state with optional parameter overrides.
    
    Args:
        **kwargs: Optional parameters to override initial settings:
            - position/positions: New initial position(s)
            - orientation/orientations: New initial orientation(s)
            - speed/speeds: New initial speed(s)
            - max_speed/max_speeds: New maximum speed(s)
            
    Performance: <1ms for single agent, <10ms for 100 agents
    
    Example:
        navigator.reset(position=(10.0, 20.0))  # Single agent
        navigator.reset(positions=[[0,0], [10,10]])  # Multi-agent
    """

def step(self, env_array: np.ndarray, dt: float = 1.0) -> None:
    """
    Execute one simulation time step with environment interaction.
    
    Args:
        env_array: Environment data array (e.g., odor plume frame)
        dt: Time step size in seconds (default: 1.0)
        
    Performance: <1ms for single agent, <10ms for 100 agents
    Must support 30+ fps simulation for real-time visualization
    
    Notes:
        - Position updates: new_pos = pos + speed * dt * [cos(θ), sin(θ)]
        - Orientation updates: new_θ = θ + angular_velocity * dt
        - Automatic constraint enforcement for boundaries and speeds
    """
```

**Sensor Integration**
```python
def sample_odor(self, env_array: np.ndarray) -> Union[float, np.ndarray]:
    """
    Sample odor concentration(s) at current agent position(s).
    
    Args:
        env_array: Environment array containing odor concentration data
        
    Returns:
        Union[float, np.ndarray]: Concentration value(s):
            - Single agent: float value
            - Multi-agent: np.ndarray with shape (n_agents,)
            
    Performance: <100μs per agent for sub-millisecond total sampling
    
    Notes:
        - Uses bilinear interpolation for sub-pixel accuracy
        - Values normalized to [0, 1] range
        - Out-of-bounds positions return 0.0 concentration
    """

def sample_multiple_sensors(
    self, 
    env_array: np.ndarray, 
    sensor_distance: float = 5.0,
    sensor_angle: float = 45.0,
    num_sensors: int = 2,
    layout_name: Optional[str] = None
) -> np.ndarray:
    """
    Sample odor at multiple sensor positions relative to each agent.
    
    Args:
        env_array: Environment array containing odor concentration data
        sensor_distance: Distance from agent center to each sensor
        sensor_angle: Angular separation between sensors in degrees
        num_sensors: Number of sensors per agent
        layout_name: Predefined sensor layout:
            - "LEFT_RIGHT": Two sensors at ±90° from agent orientation
            - "FORWARD_BACK": Sensors at 0° and 180°
            - "TRIANGLE": Three sensors in triangular arrangement
            
    Returns:
        np.ndarray: Sensor readings with shape:
            - Single agent: (num_sensors,)
            - Multi-agent: (n_agents, num_sensors)
            
    Performance: <500μs per agent for efficient multi-sensor sampling
    """
```

#### Extensibility Hooks

The NavigatorProtocol includes extensibility hooks for custom research implementations:

**Custom Observation Extension**
```python
def compute_additional_obs(self, base_obs: dict) -> dict:
    """
    Compute additional observations for custom environment extensions.
    
    Args:
        base_obs: Base observation dict with standard navigation data
        
    Returns:
        dict: Additional observation components to merge with base_obs
        
    Performance: <1ms to maintain environment step time requirements
    
    Example Implementation:
        def compute_additional_obs(self, base_obs: dict) -> dict:
            return {
                "wind_direction": self.sample_wind_direction(),
                "distance_to_wall": self.compute_wall_distance(),
                "energy_level": self.get_energy_remaining()
            }
    """
```

**Custom Reward Shaping**
```python
def compute_extra_reward(self, base_reward: float, info: dict) -> float:
    """
    Compute additional reward components for custom reward shaping.
    
    Args:
        base_reward: Base reward from environment's standard function
        info: Environment info dict containing episode state and metrics
        
    Returns:
        float: Additional reward component (can be positive or negative)
        
    Performance: <0.5ms to maintain environment step time requirements
    
    Example Implementation:
        def compute_extra_reward(self, base_reward: float, info: dict) -> float:
            # Exploration bonus for visiting unexplored areas
            if self.is_novel_position(self.positions[-1]):
                return 0.1
            # Energy conservation penalty
            speed_penalty = -0.01 * np.mean(self.speeds ** 2)
            return speed_penalty
    """
```

**Episode Completion Handling**
```python
def on_episode_end(self, final_info: dict) -> None:
    """
    Handle episode completion events for logging and cleanup.
    
    Args:
        final_info: Final environment info dict with episode summary
        
    Performance: <5ms to avoid blocking episode transitions
    
    Example Implementation:
        def on_episode_end(self, final_info: dict) -> None:
            episode_length = final_info.get('episode_length', 0)
            success_rate = final_info.get('success', False)
            self.logger.info(f"Episode ended: length={episode_length}, success={success_rate}")
            
            # Custom trajectory analysis
            trajectory = final_info.get('trajectory', [])
            path_efficiency = self.analyze_path_efficiency(trajectory)
            self.metrics_tracker.record('path_efficiency', path_efficiency)
    """
```

#### State Management

The NavigatorProtocol provides comprehensive state management capabilities:

**Memory Interface (Optional)**
```python
def load_memory(self, memory_data: Optional[Dict[str, Any]] = None) -> None:
    """
    Load agent memory state from external storage (optional interface).
    
    Args:
        memory_data: Optional dict containing serialized memory state:
            - 'trajectory_history': List of past positions and actions
            - 'odor_concentration_history': Time series of sensor readings
            - 'spatial_map': Learned environment representation
            - 'internal_state': Algorithm-specific data structures
            
    Notes:
        - Default implementation is no-op for memory-less agents
        - Enables same simulator core for reactive and cognitive agents
        - Should validate data consistency and handle corruption gracefully
        
    Performance: <10ms to avoid blocking episode initialization
    """

def save_memory(self) -> Optional[Dict[str, Any]]:
    """
    Save current agent memory state for external storage (optional interface).
    
    Returns:
        Optional[Dict[str, Any]]: Serializable memory state or None
        
    Notes:
        - Default implementation returns None for memory-less agents
        - All returned data must be JSON-serializable for storage compatibility
        - Should include metadata (timestamps, episode info, version tags)
        
    Performance: <10ms to avoid blocking episode transitions
    """
```

#### Implementation Requirements

**Type Safety and Protocol Compliance**
- All methods must maintain exact signature compatibility
- Property access must be O(1) performance
- NumPy arrays must use appropriate dtypes (typically float64 for positions)
- Error handling must be comprehensive with informative exceptions

**Performance Constraints**
- Step execution: <1ms for single agent, <10ms for 100 agents  
- Memory efficiency: <10MB overhead per 100 agents
- Vectorized operations for scalable multi-agent performance
- Support for 30+ fps simulation rates

**Integration Requirements**
- Gymnasium 0.29.x API compatibility with dual format support
- Hydra configuration integration for dependency injection
- Protocol-based extensibility for custom algorithm implementation
- Thread safety for concurrent simulation scenarios

## Source Management Protocols

### SourceProtocol Interface Specification

The `SourceProtocol` enables seamless switching between different odor source implementations, providing flexible odor source modeling through protocol-based composition. This protocol defines the contract for all odor emission sources in the simulation environment.

```python
@runtime_checkable
class SourceProtocol(Protocol):
    """Protocol defining the interface for pluggable odor source implementations."""
```

#### Emission Modeling

**Real-time Emission Queries**
```python
def get_emission_rate(self) -> float:
    """
    Get current odor emission rate from this source.
    
    Returns:
        float: Emission rate in source-specific units (typically molecules/second
            or concentration units/second). Non-negative value representing
            current source strength.
            
    Performance: <0.1ms for real-time simulation compatibility
    
    Notes:
        - Emission rate may vary over time for dynamic sources
        - Rate should remain physically realistic and consistent
        - Multi-source implementations return total aggregate rate
        
    Example:
        rate = source.get_emission_rate()
        assert rate >= 0.0, "Emission rate must be non-negative"
    """
```

#### Position Management

**Spatial Position Access**
```python
def get_position(self) -> Tuple[float, float]:
    """
    Get current source position coordinates.
    
    Returns:
        Tuple[float, float]: Source position as (x, y) coordinates in
            environment coordinate system. Values should be within
            environment domain bounds.
            
    Performance: <0.1ms for minimal spatial query overhead
    
    Notes:
        - Position may be static for fixed sources or dynamic for moving sources
        - Coordinates follow environment convention (typically top-left origin)
        - Multi-source implementations return centroid or primary position
        
    Example:
        x, y = source.get_position()
        assert 0 <= x <= domain_width and 0 <= y <= domain_height
    """
```

#### Temporal Evolution

**State Update Dynamics**
```python
def update_state(self, dt: float = 1.0) -> None:
    """
    Advance source state by specified time delta.
    
    Args:
        dt: Time step size in seconds. Controls temporal resolution of
            source dynamics including emission variations, position changes,
            and internal state evolution.
            
    Performance: <1ms per step for real-time simulation compatibility
    
    Notes:
        Updates source internal state including:
        - Emission rate variations based on temporal patterns
        - Position changes for mobile sources  
        - Internal parameters for complex source dynamics
        - Environmental interactions and external influences
        
    Example:
        source.update_state(dt=1.0)  # Standard time evolution
        
        # High-frequency dynamics
        for _ in range(10):
            source.update_state(dt=0.1)  # 10x higher temporal resolution
    """
```

#### Implementation Examples

**Point Source Implementation Pattern**
```python
class PointSource:
    """Simple stationary point source implementation."""
    
    def __init__(self, position: Tuple[float, float], emission_rate: float):
        self._position = position
        self._emission_rate = emission_rate
    
    def get_emission_rate(self) -> float:
        return self._emission_rate
    
    def get_position(self) -> Tuple[float, float]:
        return self._position
    
    def update_state(self, dt: float = 1.0) -> None:
        pass  # Static source - no state updates needed
```

**Dynamic Source Implementation Pattern**
```python
class DynamicSource:
    """Time-varying source with configurable emission patterns."""
    
    def __init__(self, initial_position: Tuple[float, float], 
                 emission_pattern: str = "sinusoidal", period: float = 60.0):
        self._position = initial_position
        self._pattern = emission_pattern
        self._period = period
        self._time = 0.0
        self._base_rate = 1000.0
    
    def get_emission_rate(self) -> float:
        if self._pattern == "sinusoidal":
            phase = 2 * np.pi * self._time / self._period
            return self._base_rate * (0.5 + 0.5 * np.sin(phase))
        return self._base_rate
    
    def get_position(self) -> Tuple[float, float]:
        return self._position
    
    def update_state(self, dt: float = 1.0) -> None:
        self._time += dt
```

#### Configuration Patterns

**Hydra Configuration Integration**
```yaml
# Point source configuration
source:
  _target_: plume_nav_sim.core.sources.PointSource
  position: [50.0, 50.0]
  emission_rate: 1000.0

# Dynamic source configuration  
source:
  _target_: plume_nav_sim.core.sources.DynamicSource
  initial_position: [25.0, 75.0]
  emission_pattern: "sinusoidal"
  period: 60.0
  base_rate: 800.0

# Multi-source configuration
source:
  _target_: plume_nav_sim.core.sources.MultiSource
  sources:
    - position: [20.0, 20.0]
      emission_rate: 500.0
    - position: [80.0, 80.0] 
      emission_rate: 750.0
```

**Factory Method Integration**
```python
# Configuration-driven source creation
source = NavigatorFactory.create_source({
    'type': 'PointSource',
    'position': (50, 50),
    'emission_rate': 1000.0
})

# Dynamic source creation
source = NavigatorFactory.create_source({
    'type': 'DynamicSource', 
    'initial_position': (25, 75),
    'emission_pattern': 'sinusoidal',
    'period': 60.0
})
```

## Boundary Policy Protocols

### BoundaryPolicyProtocol Interface Specification

The `BoundaryPolicyProtocol` defines configurable boundary handling strategies for domain edge management, enabling flexible boundary behavior implementations through protocol-based composition.

```python
@runtime_checkable  
class BoundaryPolicyProtocol(Protocol):
    """Protocol defining configurable boundary handling strategies."""
```

#### Domain Edge Management

**Policy Application**
```python
def apply_policy(
    self, 
    positions: np.ndarray, 
    velocities: Optional[np.ndarray] = None
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Apply boundary policy to agent positions and optionally velocities.
    
    Args:
        positions: Agent positions as array with shape (n_agents, 2) or (2,)
        velocities: Optional agent velocities with same shape as positions
        
    Returns:
        Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
            - If velocities not provided: corrected positions array
            - If velocities provided: tuple of (corrected_positions, corrected_velocities)
            
    Performance: <1ms for 100 agents with vectorized operations
    
    Notes:
        Policy application depends on boundary type:
        - Terminate: positions returned unchanged (termination handled separately)
        - Bounce: positions and velocities corrected for elastic/inelastic collisions
        - Wrap: positions wrapped to opposite boundary with velocity preservation
        - Clip: positions constrained to domain bounds with velocity zeroing
    """
```

#### Vectorized Operations

**Efficient Violation Detection**
```python
def check_violations(self, positions: np.ndarray) -> np.ndarray:
    """
    Detect boundary violations for given agent positions.
    
    Args:
        positions: Agent positions as array with shape (n_agents, 2) or (2,)
        
    Returns:
        np.ndarray: Boolean array with shape (n_agents,) or scalar bool.
            True indicates boundary violation requiring policy application.
            
    Performance: <0.5ms for boundary detection across 100 agents
    
    Notes:
        - Violation detection consistent across policy types
        - Efficient vectorized implementation for large agent populations
        - May include predictive violation detection for smooth handling
        
    Example:
        positions = np.array([[50, 50], [105, 25], [25, 105]])
        violations = policy.check_violations(positions)
        # Returns [False, True, True] for domain bounds (100, 100)
    """
```

#### Termination Logic

**Episode Management**
```python
def get_termination_status(self) -> str:
    """
    Get episode termination status for boundary policy.
    
    Returns:
        str: Termination status string indicating boundary policy behavior:
            - "oob": Out of bounds termination (TerminatePolicy)
            - "continue": Episode continues with correction (BouncePolicy, WrapPolicy, ClipPolicy)
            - "boundary_contact": Boundary interaction without termination
            
    Performance: <0.1ms for immediate episode management decisions
    
    Example:
        status = policy.get_termination_status()
        episode_done = (status == "oob")
    """
```

#### Policy Implementations

**Terminate Boundary Policy**
```python
class TerminatePolicy:
    """End episode when agent reaches boundary (status = "oob")."""
    
    def __init__(self, domain_bounds: Tuple[float, float]):
        self.bounds = domain_bounds
    
    def apply_policy(self, positions: np.ndarray, 
                    velocities: Optional[np.ndarray] = None):
        # Return positions unchanged - termination handled by status
        if velocities is not None:
            return positions, velocities
        return positions
    
    def check_violations(self, positions: np.ndarray) -> np.ndarray:
        width, height = self.bounds
        x_violations = (positions[..., 0] < 0) | (positions[..., 0] >= width)
        y_violations = (positions[..., 1] < 0) | (positions[..., 1] >= height)
        return x_violations | y_violations
    
    def get_termination_status(self) -> str:
        return "oob"
```

**Bounce Boundary Policy**
```python
class BouncePolicy:
    """Reflect agent trajectory off boundary walls with energy conservation."""
    
    def __init__(self, domain_bounds: Tuple[float, float], energy_loss: float = 0.0):
        self.bounds = domain_bounds
        self.energy_loss = energy_loss  # 0.0 = elastic, 1.0 = inelastic
    
    def apply_policy(self, positions: np.ndarray, 
                    velocities: Optional[np.ndarray] = None):
        width, height = self.bounds
        
        # Reflect positions off boundaries
        corrected_pos = positions.copy()
        corrected_pos[..., 0] = np.clip(corrected_pos[..., 0], 0, width)
        corrected_pos[..., 1] = np.clip(corrected_pos[..., 1], 0, height)
        
        if velocities is not None:
            corrected_vel = velocities.copy()
            # Reverse velocity components at boundaries with energy loss
            x_bounce = (positions[..., 0] < 0) | (positions[..., 0] >= width)
            y_bounce = (positions[..., 1] < 0) | (positions[..., 1] >= height)
            
            corrected_vel[x_bounce, 0] *= -(1.0 - self.energy_loss)
            corrected_vel[y_bounce, 1] *= -(1.0 - self.energy_loss)
            
            return corrected_pos, corrected_vel
        
        return corrected_pos
    
    def get_termination_status(self) -> str:
        return "continue"
```

#### Performance Considerations

**Vectorized Multi-Agent Processing**
```python
# Efficient boundary checking for large agent populations
def vectorized_boundary_check(positions: np.ndarray, bounds: Tuple[float, float]) -> np.ndarray:
    """Optimized violation detection using NumPy broadcasting."""
    width, height = bounds
    
    # Single vectorized operation for all agents
    violations = ((positions[..., 0] < 0) | (positions[..., 0] >= width) |
                 (positions[..., 1] < 0) | (positions[..., 1] >= height))
    
    return violations

# Batch policy application
def vectorized_policy_application(positions: np.ndarray, velocities: np.ndarray,
                                bounds: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
    """Apply boundary corrections to entire agent population simultaneously."""
    width, height = bounds
    
    # Vectorized position clamping
    corrected_positions = np.clip(positions, [0, 0], [width, height])
    
    # Vectorized velocity reflection
    corrected_velocities = velocities.copy()
    boundary_contacts = (positions != corrected_positions)
    corrected_velocities[boundary_contacts] *= -1
    
    return corrected_positions, corrected_velocities
```

## Action Interface Protocols

### ActionInterfaceProtocol Interface Specification

The `ActionInterfaceProtocol` defines standardized action space translation for RL framework integration, enabling unified action handling across different control paradigms while maintaining compatibility with navigation controllers.

```python
@runtime_checkable
class ActionInterfaceProtocol(Protocol):
    """Protocol defining standardized action space translation for RL framework integration."""
```

#### Action Translation

**RL Framework to Navigation Command Conversion**
```python
def translate_action(
    self, 
    action: Union[np.ndarray, int, float, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Translate RL framework action to navigation controller commands.
    
    Args:
        action: RL action in framework-specific format:
            - np.ndarray: Continuous control vectors (velocity, acceleration)
            - int: Discrete action indices for directional commands
            - float: Scalar actions for 1D control
            - Dict[str, Any]: Structured actions for complex control schemes
            
    Returns:
        Dict[str, Any]: Navigation command dictionary with standardized keys:
            - 'linear_velocity': Target linear velocity (float)
            - 'angular_velocity': Target angular velocity (float)
            - 'action_type': Control scheme identifier (str)
            - Additional keys for specialized control modes
            
    Performance: <0.1ms per agent for minimal control overhead
    
    Notes:
        - Translation handles scaling, bounds checking, coordinate transformations
        - Continuous actions typically normalized to [-1, 1] range in RL frameworks
        - Output format consistent across action interface implementations
        
    Example:
        # Continuous action translation
        rl_action = np.array([0.5, -0.2])  # [linear_vel_norm, angular_vel_norm]
        command = action_interface.translate_action(rl_action)
        assert 'linear_velocity' in command and 'angular_velocity' in command
    """
```

#### Space Construction

**Gymnasium Integration**
```python
def get_action_space(self) -> Optional['spaces.Space']:
    """
    Construct Gymnasium action space definition for RL framework integration.
    
    Returns:
        Optional[spaces.Space]: Gymnasium action space defining valid action
            structure and value ranges. Returns None if Gymnasium not available.
            Common space types:
            - spaces.Box: Continuous control with bounded ranges
            - spaces.Discrete: Discrete action indices
            - spaces.Dict: Structured action dictionaries
            - spaces.MultiDiscrete: Multiple discrete action dimensions
            
    Performance: <1ms for space construction (called infrequently)
    
    Notes:
        - Action space automatically reflects interface configuration
        - Space definition consistent with translate_action() and validate_action()
        - Enables proper RL framework integration and training
        
    Example:
        action_space = action_interface.get_action_space()
        sample_action = action_space.sample()
        assert action_interface.validate_action(sample_action) == True
    """
```

#### RL Framework Integration

**Action Validation and Constraint Enforcement**
```python
def validate_action(
    self, 
    action: Union[np.ndarray, int, float, Dict[str, Any]]
) -> bool:
    """
    Validate action compliance with interface constraints and safety limits.
    
    Args:
        action: RL action to validate in same format as translate_action()
        
    Returns:
        bool: True if action is valid and safe to execute, False otherwise
        
    Performance: <0.05ms per action for constraint checking
    
    Notes:
        Validation includes:
        - Bounds checking for continuous actions (within normalized ranges)
        - Index validation for discrete actions (valid action indices)
        - Safety constraints (maximum velocities, acceleration limits)
        - Type checking and format validation
        
    Example:
        # Continuous action validation
        valid_action = np.array([0.5, -0.2])  # Valid normalized action
        assert action_interface.validate_action(valid_action) == True
        
        invalid_action = np.array([2.0, -3.0])  # Out of bounds
        assert action_interface.validate_action(invalid_action) == False
    """
```

#### Validation Patterns

**Comprehensive Input Validation**
```python
class ActionValidationMixin:
    """Mixin providing common action validation patterns."""
    
    def validate_continuous_action(self, action: np.ndarray, 
                                 bounds: Tuple[np.ndarray, np.ndarray]) -> bool:
        """Validate continuous action against bounds."""
        if not isinstance(action, np.ndarray):
            return False
        
        low, high = bounds
        return np.all(action >= low) and np.all(action <= high)
    
    def validate_discrete_action(self, action: int, max_actions: int) -> bool:
        """Validate discrete action index."""
        return isinstance(action, int) and 0 <= action < max_actions
    
    def validate_structured_action(self, action: Dict[str, Any], 
                                 schema: Dict[str, type]) -> bool:
        """Validate structured action dictionary."""
        if not isinstance(action, dict):
            return False
        
        for key, expected_type in schema.items():
            if key not in action or not isinstance(action[key], expected_type):
                return False
        
        return True
```

#### Continuous Discrete Support

**Continuous 2D Action Interface**
```python
class Continuous2DAction:
    """Continuous velocity control with bounded action spaces."""
    
    def __init__(self, max_linear_velocity: float = 2.0, 
                 max_angular_velocity: float = 45.0):
        self.max_linear_vel = max_linear_velocity
        self.max_angular_vel = max_angular_velocity
    
    def translate_action(self, action: np.ndarray) -> Dict[str, Any]:
        """Translate normalized continuous action to velocity commands."""
        if not self.validate_action(action):
            raise ValueError("Invalid continuous action")
        
        linear_vel = action[0] * self.max_linear_vel
        angular_vel = action[1] * self.max_angular_vel
        
        return {
            'linear_velocity': float(linear_vel),
            'angular_velocity': float(angular_vel),
            'action_type': 'continuous_2d'
        }
    
    def validate_action(self, action: np.ndarray) -> bool:
        """Validate continuous action in [-1, 1] range."""
        return (isinstance(action, np.ndarray) and 
                action.shape == (2,) and
                np.all(action >= -1.0) and 
                np.all(action <= 1.0))
    
    def get_action_space(self) -> Optional['spaces.Space']:
        """Create continuous Box action space."""
        if spaces is None:
            return None
        return spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
```

**Cardinal Discrete Action Interface**
```python
class CardinalDiscreteAction:
    """Discrete directional commands (N, S, E, W, NE, NW, SE, SW, Stop)."""
    
    def __init__(self, move_speed: float = 1.0):
        self.move_speed = move_speed
        self.action_map = {
            0: (0.0, 0.0),      # Stop
            1: (1.0, 0.0),      # East
            2: (-1.0, 0.0),     # West  
            3: (0.0, 1.0),      # North
            4: (0.0, -1.0),     # South
            5: (1.0, 1.0),      # Northeast
            6: (-1.0, 1.0),     # Northwest
            7: (1.0, -1.0),     # Southeast
            8: (-1.0, -1.0),    # Southwest
        }
    
    def translate_action(self, action: int) -> Dict[str, Any]:
        """Translate discrete action index to velocity commands."""
        if not self.validate_action(action):
            raise ValueError("Invalid discrete action")
        
        dx, dy = self.action_map[action]
        
        # Convert to velocity and angular velocity
        linear_velocity = self.move_speed * np.sqrt(dx*dx + dy*dy)
        if linear_velocity > 0:
            angular_velocity = np.degrees(np.arctan2(dy, dx))
        else:
            angular_velocity = 0.0
        
        return {
            'linear_velocity': float(linear_velocity),
            'angular_velocity': float(angular_velocity),
            'action_type': 'cardinal_discrete'
        }
    
    def validate_action(self, action: int) -> bool:
        """Validate discrete action index."""
        return isinstance(action, int) and action in self.action_map
    
    def get_action_space(self) -> Optional['spaces.Space']:
        """Create discrete action space."""
        if spaces is None:
            return None
        return spaces.Discrete(len(self.action_map))
```

## Agent Initialization Protocols

### AgentInitializerProtocol Interface Specification

The `AgentInitializerProtocol` defines configurable agent initialization strategies for diverse experimental setups, enabling flexible starting position generation patterns through protocol-based composition.

```python
@runtime_checkable
class AgentInitializerProtocol(Protocol):
    """Protocol defining configurable agent initialization strategies."""
```

#### Position Generation

**Flexible Starting Configuration**
```python
def initialize_positions(
    self, 
    num_agents: int,
    **kwargs: Any
) -> np.ndarray:
    """
    Generate initial agent positions based on configured strategy.
    
    Args:
        num_agents: Number of agent positions to generate (must be positive)
        **kwargs: Additional strategy-specific parameters:
            - exclusion_zones: List of spatial regions to avoid
            - clustering_factor: Spatial clustering strength for grouped initialization
            - minimum_distance: Minimum separation between agent positions
            - boundary_margin: Margin from domain edges for position placement
            
    Returns:
        np.ndarray: Agent positions with shape (num_agents, 2) containing
            [x, y] coordinates. All positions guaranteed within domain bounds
            and satisfying strategy constraints.
            
    Performance: <5ms for 100 agents with spatial distribution algorithms
    
    Notes:
        - Position generation follows strategy-specific algorithms
        - Deterministic behavior based on internal random state
        - Validation ensures domain constraint compliance
        - Multi-agent collision avoidance and spatial optimization
        
    Example:
        positions = initializer.initialize_positions(num_agents=25)
        assert positions.shape == (25, 2)
        assert np.all(positions >= 0)  # Within domain bounds
    """
```

#### Deterministic Seeding

**Reproducible Experiment Setup**
```python
def reset(self, seed: Optional[int] = None, **kwargs: Any) -> None:
    """
    Reset initializer state for deterministic position generation.
    
    Args:
        seed: Optional random seed for deterministic behavior reproduction
        **kwargs: Additional reset parameters for strategy-specific state
        
    Performance: <1ms for strategy state reset with deterministic seeding
    
    Notes:
        - Reset operation reinitializes internal random number generators
        - Strategy-specific state reset for reproducible experiment conditions
        - Deterministic seeding enables exact reproduction of initialization patterns
        
    Example:
        # Deterministic reset
        initializer.reset(seed=42)
        positions_1 = initializer.initialize_positions(num_agents=10)
        initializer.reset(seed=42)
        positions_2 = initializer.initialize_positions(num_agents=10)
        assert np.array_equal(positions_1, positions_2)
    """
```

#### Validation Patterns

**Domain Constraint Verification**
```python
def validate_domain(
    self, 
    positions: np.ndarray,
    domain_bounds: Tuple[float, float]
) -> bool:
    """
    Validate that positions comply with domain constraints and strategy requirements.
    
    Args:
        positions: Agent positions to validate with shape (n_agents, 2)
        domain_bounds: Spatial domain limits as (width, height) tuple
        
    Returns:
        bool: True if all positions are valid and compliant, False otherwise
        
    Performance: <1ms for position constraint checking and validation
    
    Notes:
        Validation includes:
        - Boundary checking for domain compliance
        - Strategy-specific constraint verification
        - Collision detection and minimum distance requirements
        - Exclusion zone compliance and spatial restrictions
        
    Example:
        positions = np.array([[25, 30], [75, 80], [10, 90]])
        domain_bounds = (100, 100)
        is_valid = initializer.validate_domain(positions, domain_bounds)
        assert is_valid == True
    """
```

#### Strategy Implementations

**Uniform Random Initializer**
```python
class UniformRandomInitializer:
    """Random positions with uniform spatial distribution."""
    
    def __init__(self, domain_bounds: Tuple[float, float], 
                 seed: Optional[int] = None, boundary_margin: float = 0.0):
        self.bounds = domain_bounds
        self.margin = boundary_margin
        self.rng = np.random.RandomState(seed)
    
    def initialize_positions(self, num_agents: int, **kwargs) -> np.ndarray:
        """Generate uniformly distributed random positions."""
        width, height = self.bounds
        margin = kwargs.get('boundary_margin', self.margin)
        
        # Effective domain after margin
        effective_width = width - 2 * margin
        effective_height = height - 2 * margin
        
        positions = self.rng.uniform(
            low=[margin, margin],
            high=[margin + effective_width, margin + effective_height],
            size=(num_agents, 2)
        )
        
        return positions
    
    def reset(self, seed: Optional[int] = None, **kwargs) -> None:
        """Reset random number generator state."""
        if seed is not None:
            self.rng = np.random.RandomState(seed)
    
    def validate_domain(self, positions: np.ndarray, 
                       domain_bounds: Tuple[float, float]) -> bool:
        """Validate positions within domain bounds."""
        width, height = domain_bounds
        return (np.all(positions >= 0) and 
                np.all(positions[:, 0] <= width) and
                np.all(positions[:, 1] <= height))
    
    def get_strategy_name(self) -> str:
        return "uniform_random"
```

**Grid Initializer**
```python
class GridInitializer:
    """Regular grid patterns for systematic spatial coverage."""
    
    def __init__(self, domain_bounds: Tuple[float, float], 
                 grid_spacing: float = 10.0, grid_offset: float = 5.0):
        self.bounds = domain_bounds
        self.spacing = grid_spacing
        self.offset = grid_offset
    
    def initialize_positions(self, num_agents: int, **kwargs) -> np.ndarray:
        """Generate positions on regular grid pattern."""
        width, height = self.bounds
        spacing = kwargs.get('grid_spacing', self.spacing)
        offset = kwargs.get('grid_offset', self.offset)
        
        # Calculate grid dimensions
        nx = int((width - 2 * offset) // spacing) + 1
        ny = int((height - 2 * offset) // spacing) + 1
        
        # Generate grid coordinates
        x_coords = np.linspace(offset, width - offset, nx)
        y_coords = np.linspace(offset, height - offset, ny)
        
        # Create all grid positions
        grid_positions = []
        for y in y_coords:
            for x in x_coords:
                grid_positions.append([x, y])
        
        # Select subset for requested number of agents
        if num_agents > len(grid_positions):
            raise ValueError(f"Requested {num_agents} agents but grid only supports {len(grid_positions)}")
        
        selected_positions = grid_positions[:num_agents]
        return np.array(selected_positions)
    
    def get_strategy_name(self) -> str:
        return "grid"
```

#### Multi-Agent Support

**Collision Avoidance and Spatial Distribution**
```python
class CollisionAwareInitializer:
    """Base class for initializers with collision avoidance."""
    
    def __init__(self, minimum_distance: float = 5.0):
        self.min_distance = minimum_distance
    
    def enforce_minimum_distance(self, positions: np.ndarray, 
                                min_distance: float) -> np.ndarray:
        """Adjust positions to maintain minimum separation."""
        n_agents = len(positions)
        adjusted_positions = positions.copy()
        
        for i in range(n_agents):
            for j in range(i + 1, n_agents):
                # Calculate distance between agents
                distance = np.linalg.norm(adjusted_positions[i] - adjusted_positions[j])
                
                if distance < min_distance:
                    # Calculate adjustment vector
                    direction = adjusted_positions[i] - adjusted_positions[j]
                    direction = direction / np.linalg.norm(direction)
                    
                    # Move agents apart
                    adjustment = (min_distance - distance) / 2
                    adjusted_positions[i] += direction * adjustment
                    adjusted_positions[j] -= direction * adjustment
        
        return adjusted_positions
    
    def check_collisions(self, positions: np.ndarray, 
                        min_distance: float) -> np.ndarray:
        """Check for position collisions between agents."""
        n_agents = len(positions)
        distances = np.zeros((n_agents, n_agents))
        
        for i in range(n_agents):
            for j in range(i + 1, n_agents):
                dist = np.linalg.norm(positions[i] - positions[j])
                distances[i, j] = dist
                distances[j, i] = dist
        
        # Find pairs with violations
        violations = distances < min_distance
        violations[np.diag_indices(n_agents)] = False  # Ignore self-comparisons
        
        return violations
```

## Data Recording Protocols

### RecorderProtocol Interface Specification

The `RecorderProtocol` defines comprehensive data recording interfaces for experiment persistence, enabling configurable data collection with multiple storage backends while maintaining minimal performance impact on simulation.

```python
@runtime_checkable
class RecorderProtocol(Protocol):
    """Protocol defining comprehensive data recording interfaces for experiment persistence."""
```

#### Data Persistence

**Step-Level Recording**
```python
def record_step(
    self, 
    step_data: Dict[str, Any], 
    step_number: int,
    episode_id: Optional[int] = None,
    **metadata: Any
) -> None:
    """
    Record simulation state data for a single time step.
    
    Args:
        step_data: Dictionary containing step-level measurements and state:
            - 'position': Agent position coordinates
            - 'velocity': Agent velocity vector
            - 'concentration': Sampled odor concentration
            - 'action': Applied navigation command
            - 'reward': Step reward value
            - Additional domain-specific measurements
        step_number: Sequential step index within current episode (0-based)
        episode_id: Optional episode identifier for data organization
        **metadata: Additional metadata for step context and debugging
        
    Performance: <0.1ms overhead when disabled, <1ms when enabled
    
    Notes:
        - Step recording provides detailed trajectory capture for analysis
        - Data is buffered for performance and flushed based on configuration
        - Recording granularity is configurable (every step vs periodic snapshots)
        
    Example:
        step_data = {
            'position': np.array([45.2, 78.1]),
            'concentration': 0.23,
            'action': {'linear_velocity': 1.5, 'angular_velocity': 0.0}
        }
        recorder.record_step(step_data, step_number=125)
    """
```

**Episode-Level Recording**
```python
def record_episode(
    self, 
    episode_data: Dict[str, Any], 
    episode_id: int,
    **metadata: Any
) -> None:
    """
    Record episode-level summary data and metrics.
    
    Args:
        episode_data: Dictionary containing episode summary information:
            - 'total_steps': Episode length in simulation steps
            - 'success': Boolean success indicator
            - 'final_position': Agent position at episode end
            - 'total_reward': Cumulative episode reward
            - 'path_efficiency': Navigation performance metric
            - 'exploration_coverage': Spatial coverage measure
            - Additional domain-specific episode metrics
        episode_id: Unique episode identifier for data organization
        **metadata: Additional metadata for episode context and configuration
        
    Performance: <10ms for episode finalization and metadata storage
    
    Notes:
        - Episode recording provides summary statistics and metadata
        - Data includes computed metrics and configuration snapshots
        - Episode finalization may trigger buffer flushing and file organization
        
    Example:
        episode_data = {
            'total_steps': 245,
            'success': True,
            'final_position': (87.3, 91.8),
            'total_reward': 12.4,
            'path_efficiency': 0.78
        }
        recorder.record_episode(episode_data, episode_id=42)
    """
```

#### Backend Implementations

**Structured Output Organization**
```python
def export_data(
    self, 
    output_path: str,
    format: str = "parquet",
    compression: Optional[str] = None,
    filter_episodes: Optional[List[int]] = None,
    **export_options: Any
) -> bool:
    """
    Export recorded data to specified file format and location.
    
    Args:
        output_path: File system path for exported data output
        format: Export format specification:
            - "parquet": Columnar format with excellent compression
            - "hdf5": Hierarchical scientific data format
            - "csv": Human-readable comma-separated values
            - "json": JSON format for structured data exchange
        compression: Optional compression method (format-specific):
            - Parquet: "snappy", "gzip", "brotli", "lz4"
            - HDF5: "gzip", "lzf", "szip"
        filter_episodes: Optional list of episode IDs to export (default: all)
        **export_options: Additional format-specific export parameters
        
    Returns:
        bool: True if export completed successfully, False otherwise
        
    Performance: <100ms for typical dataset export with compression
    
    Notes:
        - Export operation consolidates buffered data and generates output files
        - Large datasets may require streaming export to manage memory usage
        - Export includes step-level trajectory data and episode-level summaries
        
    Example:
        # Parquet export with compression
        success = recorder.export_data(
            output_path="./results/experiment_001.parquet",
            format="parquet",
            compression="snappy"
        )
    """
```

#### Performance Optimization

**Buffering Strategies**
```python
class BufferedRecorder:
    """Base class providing efficient buffering for high-frequency recording."""
    
    def __init__(self, buffer_size: int = 1000, auto_flush: bool = True):
        self.buffer_size = buffer_size
        self.auto_flush = auto_flush
        self.step_buffer = []
        self.episode_buffer = []
        self._total_steps_recorded = 0
    
    def _buffer_step_data(self, step_data: Dict[str, Any], 
                         step_number: int, episode_id: Optional[int] = None) -> None:
        """Add step data to buffer with automatic flushing."""
        buffered_record = {
            'step_number': step_number,
            'episode_id': episode_id,
            'timestamp': time.time(),
            **step_data
        }
        
        self.step_buffer.append(buffered_record)
        self._total_steps_recorded += 1
        
        # Auto-flush when buffer is full
        if self.auto_flush and len(self.step_buffer) >= self.buffer_size:
            self._flush_step_buffer()
    
    def _flush_step_buffer(self) -> None:
        """Flush step buffer to persistent storage."""
        if not self.step_buffer:
            return
        
        # Convert to DataFrame for efficient batch processing
        import pandas as pd
        df = pd.DataFrame(self.step_buffer)
        
        # Backend-specific persistence logic
        self._persist_step_dataframe(df)
        
        # Clear buffer
        self.step_buffer.clear()
    
    def get_buffer_status(self) -> Dict[str, int]:
        """Get current buffer status for monitoring."""
        return {
            'step_buffer_size': len(self.step_buffer),
            'episode_buffer_size': len(self.episode_buffer),
            'total_steps_recorded': self._total_steps_recorded,
            'buffer_utilization': len(self.step_buffer) / self.buffer_size
        }
```

**Compression and Memory Management**
```python
class CompressedRecorder:
    """Recorder with compression and memory management capabilities."""
    
    def __init__(self, compression_method: str = "snappy", 
                 memory_limit_mb: int = 100):
        self.compression = compression_method
        self.memory_limit = memory_limit_mb * 1024 * 1024  # Convert to bytes
        self._current_memory_usage = 0
    
    def _estimate_memory_usage(self, data: Dict[str, Any]) -> int:
        """Estimate memory usage of data structures."""
        import sys
        
        total_size = 0
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                total_size += value.nbytes
            elif isinstance(value, str):
                total_size += sys.getsizeof(value)
            else:
                total_size += sys.getsizeof(value)
        
        return total_size
    
    def _check_memory_pressure(self) -> bool:
        """Check if memory usage exceeds limits."""
        return self._current_memory_usage > self.memory_limit
    
    def _compress_and_store(self, data: pd.DataFrame) -> None:
        """Compress data and store to persistent backend."""
        if self.compression == "snappy":
            compressed_data = data.to_parquet(compression='snappy')
        elif self.compression == "gzip":
            compressed_data = data.to_parquet(compression='gzip')
        elif self.compression == "brotli":
            compressed_data = data.to_parquet(compression='brotli')
        else:
            compressed_data = data.to_parquet()
        
        # Store compressed data using backend-specific method
        self._store_compressed_data(compressed_data)
```

#### Backend-Specific Implementations

**Parquet Backend**
```python
class ParquetRecorder:
    """High-performance columnar storage with compression."""
    
    def __init__(self, output_dir: str, compression: str = "snappy", 
                 partition_by_episode: bool = True):
        self.output_dir = Path(output_dir)
        self.compression = compression
        self.partition_by_episode = partition_by_episode
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def record_step(self, step_data: Dict[str, Any], step_number: int,
                   episode_id: Optional[int] = None, **metadata: Any) -> None:
        """Record step data using Parquet format."""
        # Convert numpy arrays to lists for Parquet compatibility
        serializable_data = self._serialize_step_data(step_data)
        
        record = {
            'step_number': step_number,
            'episode_id': episode_id,
            'timestamp': time.time(),
            **serializable_data,
            **metadata
        }
        
        self._buffer_step_data(record, step_number, episode_id)
    
    def _serialize_step_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert complex data types to Parquet-compatible formats."""
        serialized = {}
        
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                # Convert arrays to lists
                serialized[key] = value.tolist()
            elif isinstance(value, dict):
                # Flatten nested dictionaries
                for subkey, subvalue in value.items():
                    serialized[f"{key}_{subkey}"] = subvalue
            else:
                serialized[key] = value
        
        return serialized
    
    def export_data(self, output_path: str, **export_options) -> bool:
        """Export all recorded data to single Parquet file."""
        try:
            # Flush any remaining buffered data
            self._flush_step_buffer()
            
            # Read all step data files
            step_files = list(self.output_dir.glob("steps_*.parquet"))
            if step_files:
                import pandas as pd
                combined_df = pd.concat([pd.read_parquet(f) for f in step_files])
                combined_df.to_parquet(output_path, compression=self.compression)
            
            return True
        except Exception as e:
            print(f"Export failed: {e}")
            return False
```

**HDF5 Backend**
```python
class HDF5Recorder:
    """Hierarchical scientific data format with metadata support."""
    
    def __init__(self, output_dir: str, compression: str = "gzip"):
        self.output_dir = Path(output_dir)
        self.compression = compression
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._current_file = None
        self._current_episode = None
    
    def record_step(self, step_data: Dict[str, Any], step_number: int,
                   episode_id: Optional[int] = None, **metadata: Any) -> None:
        """Record step data using HDF5 hierarchical structure."""
        import h5py
        
        # Create/open HDF5 file for current episode
        if episode_id != self._current_episode:
            self._open_episode_file(episode_id)
        
        # Create step group
        step_group = self._current_file.create_group(f"step_{step_number}")
        
        # Store step data with appropriate data types
        for key, value in step_data.items():
            if isinstance(value, np.ndarray):
                step_group.create_dataset(key, data=value, compression=self.compression)
            else:
                step_group.attrs[key] = value
        
        # Store metadata as attributes
        for key, value in metadata.items():
            step_group.attrs[f"meta_{key}"] = value
    
    def _open_episode_file(self, episode_id: Optional[int]) -> None:
        """Open HDF5 file for specific episode."""
        import h5py
        
        if self._current_file:
            self._current_file.close()
        
        filename = f"episode_{episode_id}.h5" if episode_id else "data.h5"
        filepath = self.output_dir / filename
        
        self._current_file = h5py.File(filepath, 'a')
        self._current_episode = episode_id
```

## Statistics Aggregation Protocols

### StatsAggregatorProtocol Interface Specification

The `StatsAggregatorProtocol` defines automated statistics collection for research-focused metrics, enabling standardized analysis and summary generation while maintaining research reproducibility and comparison standards.

```python
@runtime_checkable
class StatsAggregatorProtocol(Protocol):
    """Protocol defining automated statistics collection for research-focused metrics."""
```

#### Metrics Calculation

**Episode-Level Analysis**
```python
def calculate_episode_stats(
    self, 
    trajectory_data: Dict[str, Any],
    episode_id: int,
    custom_metrics: Optional[Dict[str, callable]] = None
) -> Dict[str, float]:
    """
    Calculate comprehensive statistics for a single episode.
    
    Args:
        trajectory_data: Dictionary containing episode trajectory information:
            - 'positions': Agent position time series
            - 'concentrations': Odor concentration measurements
            - 'actions': Applied navigation commands
            - 'rewards': Step-wise reward values
            - 'timestamps': Temporal information
            - Additional domain-specific trajectory data
        episode_id: Unique episode identifier for metric correlation
        custom_metrics: Optional dictionary of custom metric calculation functions
        
    Returns:
        Dict[str, float]: Dictionary of calculated episode-level metrics:
            - 'path_efficiency': Ratio of direct distance to actual path length
            - 'exploration_coverage': Fraction of domain area explored
            - 'mean_concentration': Average odor concentration encountered
            - 'success_indicator': Binary success metric (1.0 if successful)
            - 'total_reward': Cumulative episode reward
            - 'episode_length': Number of simulation steps
            - Additional computed and custom metrics
            
    Performance: <10ms for episode-level metric computation
    
    Notes:
        - Metric calculation uses standard statistical methods
        - Custom metrics enable specialized analysis for research requirements
        - All metrics computed as floating-point values for consistent analysis
        
    Example:
        trajectory_data = {
            'positions': position_time_series,
            'concentrations': concentration_measurements,
            'actions': action_sequence,
            'rewards': reward_time_series
        }
        metrics = aggregator.calculate_episode_stats(trajectory_data, episode_id=42)
        print(f"Path efficiency: {metrics['path_efficiency']:.3f}")
    """
```

#### Summary Generation

**Research Integration**
```python
def calculate_run_stats(
    self, 
    episode_data_list: List[Dict[str, Any]],
    run_id: str,
    statistical_tests: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Calculate aggregate statistics across multiple episodes for run-level analysis.
    
    Args:
        episode_data_list: List of episode data dictionaries from calculate_episode_stats()
        run_id: Unique run identifier for experimental tracking and comparison
        statistical_tests: Optional list of statistical tests to perform:
            Supported tests: ["t_test", "anova", "ks_test", "wilcoxon"]
            
    Returns:
        Dict[str, float]: Dictionary of run-level aggregate metrics:
            - 'success_rate': Fraction of successful episodes
            - 'mean_path_efficiency': Average path efficiency across episodes
            - 'std_path_efficiency': Standard deviation of path efficiency
            - 'mean_episode_length': Average episode duration
            - 'total_episodes': Number of episodes in run
            - 'confidence_intervals': Statistical confidence bounds (if requested)
            - Additional aggregated metrics and statistical test results
            
    Performance: <100ms for multi-episode aggregation analysis
    
    Notes:
        - Run-level analysis provides statistical summary across episode populations
        - Statistical tests enable rigorous analysis of experimental differences
        - Aggregate metrics include central tendency, variability, and distributions
        
    Example:
        episode_data_list = [episode_metrics_1, episode_metrics_2, ...]
        run_metrics = aggregator.calculate_run_stats(
            episode_data_list, run_id="experiment_001"
        )
        print(f"Success rate: {run_metrics['success_rate']:.2%}")
    """
```

#### Automation Patterns

**Publication Quality Results**
```python
def export_summary(
    self, 
    output_path: str,
    run_data: Optional[Dict[str, Any]] = None,
    include_distributions: bool = False,
    format: str = "json"
) -> bool:
    """
    Generate and export standardized summary report for research publication.
    
    Args:
        output_path: File system path for summary report output
        run_data: Optional run-level data from calculate_run_stats() for inclusion
        include_distributions: Include distribution plots and histograms in summary
        format: Output format specification ("json", "yaml", "markdown", "latex")
        
    Returns:
        bool: True if summary export completed successfully, False otherwise
        
    Performance: <50ms for summary generation and file output
    
    Notes:
        - Summary export generates publication-ready reports with standardized metrics
        - Output format supports research workflows and publication requirements
        - Summary includes experiment configuration, statistical results, and performance metrics
        
    Example:
        # JSON summary export
        success = aggregator.export_summary(
            output_path="./results/experiment_summary.json",
            run_data=run_metrics,
            include_distributions=False
        )
        
        # Markdown report with visualizations
        success = aggregator.export_summary(
            output_path="./results/experiment_report.md",
            run_data=run_metrics,
            include_distributions=True,
            format="markdown"
        )
    """
```

#### Standard Metrics Implementation

**Path Efficiency Calculation**
```python
class PathEfficiencyCalculator:
    """Calculate path efficiency metrics for navigation analysis."""
    
    @staticmethod
    def calculate_path_efficiency(positions: np.ndarray, 
                                target_position: Optional[np.ndarray] = None) -> float:
        """
        Calculate ratio of direct distance to actual path length.
        
        Args:
            positions: Agent position time series with shape (n_steps, 2)
            target_position: Optional target position for goal-directed tasks
            
        Returns:
            float: Path efficiency ratio [0, 1] where 1.0 is perfectly efficient
        """
        if len(positions) < 2:
            return 1.0
        
        # Calculate actual path length
        path_segments = np.diff(positions, axis=0)
        path_distances = np.linalg.norm(path_segments, axis=1)
        total_path_length = np.sum(path_distances)
        
        # Calculate direct distance
        if target_position is not None:
            # Distance from start to target
            direct_distance = np.linalg.norm(positions[0] - target_position)
        else:
            # Distance from start to end
            direct_distance = np.linalg.norm(positions[0] - positions[-1])
        
        if total_path_length == 0:
            return 1.0
        
        return min(direct_distance / total_path_length, 1.0)

    @staticmethod
    def calculate_tortuosity(positions: np.ndarray) -> float:
        """Calculate path tortuosity (inverse of efficiency)."""
        efficiency = PathEfficiencyCalculator.calculate_path_efficiency(positions)
        return 1.0 / efficiency if efficiency > 0 else float('inf')
```

**Exploration Coverage Metrics**
```python
class ExplorationCalculator:
    """Calculate spatial exploration coverage metrics."""
    
    @staticmethod
    def calculate_coverage(positions: np.ndarray, domain_bounds: Tuple[float, float],
                          grid_resolution: float = 5.0) -> float:
        """
        Calculate fraction of domain area explored using grid-based method.
        
        Args:
            positions: Agent position time series with shape (n_steps, 2)
            domain_bounds: Domain size as (width, height) tuple
            grid_resolution: Grid cell size for coverage calculation
            
        Returns:
            float: Coverage fraction [0, 1] representing explored area
        """
        width, height = domain_bounds
        
        # Create coverage grid
        n_x = int(np.ceil(width / grid_resolution))
        n_y = int(np.ceil(height / grid_resolution))
        coverage_grid = np.zeros((n_y, n_x), dtype=bool)
        
        # Mark visited grid cells
        for pos in positions:
            grid_x = int(pos[0] // grid_resolution)
            grid_y = int(pos[1] // grid_resolution)
            
            # Ensure indices are within bounds
            if 0 <= grid_x < n_x and 0 <= grid_y < n_y:
                coverage_grid[grid_y, grid_x] = True
        
        # Calculate coverage fraction
        total_cells = n_x * n_y
        visited_cells = np.sum(coverage_grid)
        
        return visited_cells / total_cells

    @staticmethod
    def calculate_entropy_coverage(positions: np.ndarray, 
                                 domain_bounds: Tuple[float, float],
                                 grid_resolution: float = 5.0) -> float:
        """Calculate exploration entropy as alternative coverage measure."""
        width, height = domain_bounds
        
        # Create visitation count grid
        n_x = int(np.ceil(width / grid_resolution))
        n_y = int(np.ceil(height / grid_resolution))
        visit_counts = np.zeros((n_y, n_x))
        
        # Count visits to each grid cell
        for pos in positions:
            grid_x = int(pos[0] // grid_resolution)
            grid_y = int(pos[1] // grid_resolution)
            
            if 0 <= grid_x < n_x and 0 <= grid_y < n_y:
                visit_counts[grid_y, grid_x] += 1
        
        # Calculate normalized visit probabilities
        total_visits = np.sum(visit_counts)
        if total_visits == 0:
            return 0.0
        
        visit_probs = visit_counts / total_visits
        
        # Calculate entropy (exclude zero probabilities)
        nonzero_probs = visit_probs[visit_probs > 0]
        entropy = -np.sum(nonzero_probs * np.log2(nonzero_probs))
        
        # Normalize by maximum possible entropy
        max_entropy = np.log2(n_x * n_y)
        
        return entropy / max_entropy if max_entropy > 0 else 0.0
```

**Statistical Analysis Integration**
```python
class StatisticalAnalyzer:
    """Comprehensive statistical analysis for experimental results."""
    
    @staticmethod
    def perform_t_test(group1_data: List[float], group2_data: List[float]) -> Dict[str, float]:
        """Perform two-sample t-test for group comparison."""
        from scipy import stats
        
        t_statistic, p_value = stats.ttest_ind(group1_data, group2_data)
        
        return {
            't_statistic': float(t_statistic),
            'p_value': float(p_value),
            'degrees_of_freedom': len(group1_data) + len(group2_data) - 2,
            'effect_size_cohens_d': StatisticalAnalyzer._calculate_cohens_d(group1_data, group2_data)
        }
    
    @staticmethod
    def _calculate_cohens_d(group1: List[float], group2: List[float]) -> float:
        """Calculate Cohen's d effect size."""
        mean1, mean2 = np.mean(group1), np.mean(group2)
        std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
        n1, n2 = len(group1), len(group2)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
        
        return (mean1 - mean2) / pooled_std
    
    @staticmethod
    def calculate_confidence_interval(data: List[float], confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for mean."""
        from scipy import stats
        
        mean = np.mean(data)
        sem = stats.sem(data)  # Standard error of mean
        n = len(data)
        
        # t-distribution critical value
        alpha = 1 - confidence
        t_critical = stats.t.ppf(1 - alpha/2, df=n-1)
        
        margin_error = t_critical * sem
        
        return (mean - margin_error, mean + margin_error)
```

## Environment Modeling Protocols

### PlumeModelProtocol Interface Specification

The `PlumeModelProtocol` defines the interface for pluggable odor plume modeling implementations, enabling seamless switching between different plume simulation approaches while maintaining performance requirements for interactive simulation.

```python
@runtime_checkable
class PlumeModelProtocol(Protocol):
    """Protocol defining the interface for pluggable odor plume modeling implementations."""
```

#### Spatial Sampling

**Concentration Field Access**
```python
def concentration_at(self, positions: np.ndarray) -> np.ndarray:
    """
    Compute odor concentrations at specified spatial locations.
    
    Args:
        positions: Agent positions as array with shape (n_agents, 2) for multiple
            agents or (2,) for single agent. Coordinates in environment units.
            
    Returns:
        np.ndarray: Concentration values with shape (n_agents,) or scalar for
            single agent. Values normalized to [0, 1] range representing
            relative odor intensity.
            
    Performance: <1ms for single query, <10ms for 100 concurrent agents
    
    Notes:
        - Uses spatial interpolation for sub-pixel accuracy when applicable
        - Positions outside plume boundaries return 0.0 concentration
        - Implementation may cache results for performance optimization
        
    Example:
        # Single agent query
        position = np.array([10.5, 20.3])
        concentration = plume_model.concentration_at(position)
        
        # Multi-agent batch query
        positions = np.array([[10, 20], [15, 25], [20, 30]])
        concentrations = plume_model.concentration_at(positions)
    """
```

#### Temporal Evolution

**Environmental Dynamics**
```python
def step(self, dt: float = 1.0) -> None:
    """
    Advance plume state by specified time delta.
    
    Args:
        dt: Time step size in seconds. Controls temporal resolution of
            environmental dynamics including dispersion, transport, and
            source evolution.
            
    Performance: <5ms per step for real-time simulation compatibility
    
    Notes:
        Updates internal plume state including:
        - Dispersion dynamics and spatial evolution
        - Source strength variations (if applicable)
        - WindField integration for transport effects
        - Turbulent mixing and dissipation processes
        
    Example:
        plume_model.step(dt=1.0)  # Standard time step
        
        # High-frequency simulation
        for _ in range(10):
            plume_model.step(dt=0.1)  # 10x higher temporal resolution
    """
```

#### State Management

**Episode Reset and Configuration**
```python
def reset(self, **kwargs: Any) -> None:
    """
    Reset plume state to initial conditions.
    
    Args:
        **kwargs: Optional parameters to override initial settings:
            - source_position: New source location (x, y)
            - source_strength: Initial emission rate
            - wind_conditions: WindField configuration updates
            - boundary_conditions: Spatial domain parameters
            
    Performance: <10ms to avoid blocking episode initialization
    
    Notes:
        - Reinitializes all plume state while preserving model configuration
        - Parameter overrides applied for this episode only unless configured for persistence
        - WindField integration reset to initial conditions with parameter updates
        
    Example:
        plume_model.reset()  # Reset to default initial state
        
        # Reset with new source location
        plume_model.reset(source_position=(25, 75), source_strength=1500)
    """
```

#### Implementation Examples

**Gaussian Plume Model**
```python
class GaussianPlumeModel:
    """Fast analytical dispersion calculations using Gaussian plume equations."""
    
    def __init__(self, source_position: Tuple[float, float], 
                 source_strength: float, wind_velocity: Tuple[float, float] = (1.0, 0.0),
                 dispersion_coeffs: Tuple[float, float] = (1.0, 1.0)):
        self.source_pos = np.array(source_position)
        self.source_strength = source_strength
        self.wind_vel = np.array(wind_velocity)
        self.sigma_x, self.sigma_y = dispersion_coeffs
        self.time = 0.0
    
    def concentration_at(self, positions: np.ndarray) -> np.ndarray:
        """Calculate Gaussian plume concentration using analytical formula."""
        if positions.ndim == 1:
            positions = positions.reshape(1, -1)
        
        # Relative positions from source
        rel_pos = positions - self.source_pos
        
        # Transform to wind coordinate system
        wind_speed = np.linalg.norm(self.wind_vel)
        if wind_speed > 0:
            wind_dir = self.wind_vel / wind_speed
            # Rotate coordinates to align with wind direction
            cos_theta = wind_dir[0]
            sin_theta = wind_dir[1]
            
            x_wind = rel_pos[:, 0] * cos_theta + rel_pos[:, 1] * sin_theta
            y_wind = -rel_pos[:, 0] * sin_theta + rel_pos[:, 1] * cos_theta
        else:
            x_wind = rel_pos[:, 0]
            y_wind = rel_pos[:, 1]
        
        # Gaussian plume equation
        concentrations = np.zeros(len(positions))
        
        # Only calculate for downwind positions
        downwind_mask = x_wind > 0
        
        if np.any(downwind_mask):
            x_down = x_wind[downwind_mask]
            y_down = y_wind[downwind_mask]
            
            # Dispersion parameters as function of distance
            sigma_y_eff = self.sigma_y * np.sqrt(x_down)
            sigma_z_eff = self.sigma_x * np.sqrt(x_down)  # Using sigma_x for vertical
            
            # Gaussian concentration calculation
            concentrations[downwind_mask] = (
                self.source_strength / (2 * np.pi * wind_speed * sigma_y_eff * sigma_z_eff) *
                np.exp(-0.5 * (y_down / sigma_y_eff)**2)
            )
        
        return concentrations if len(positions) > 1 else concentrations[0]
    
    def step(self, dt: float = 1.0) -> None:
        """Update time-dependent parameters."""
        self.time += dt
        # Could add time-varying source strength or wind conditions
    
    def reset(self, **kwargs) -> None:
        """Reset to initial conditions with optional parameter updates."""
        self.time = 0.0
        if 'source_position' in kwargs:
            self.source_pos = np.array(kwargs['source_position'])
        if 'source_strength' in kwargs:
            self.source_strength = kwargs['source_strength']
        if 'wind_velocity' in kwargs:
            self.wind_vel = np.array(kwargs['wind_velocity'])
```

### WindFieldProtocol Interface Specification

The `WindFieldProtocol` defines environmental wind dynamics for realistic plume transport modeling, enabling configurable wind field implementations supporting various levels of environmental realism.

```python
@runtime_checkable
class WindFieldProtocol(Protocol):
    """Protocol defining environmental wind dynamics for realistic plume transport modeling."""
```

#### Spatial Velocity Queries

**Wind Field Sampling**
```python
def velocity_at(self, positions: np.ndarray) -> np.ndarray:
    """
    Compute wind velocity vectors at specified spatial locations.
    
    Args:
        positions: Spatial positions as array with shape (n_positions, 2) for
            multiple locations or (2,) for single position. Coordinates in
            environment units.
            
    Returns:
        np.ndarray: Velocity vectors with shape (n_positions, 2) or (2,) for
            single position. Components represent [u_x, u_y] in environment
            units per time step.
            
    Performance: <0.5ms for single query, <5ms for spatial field evaluation
    
    Notes:
        Velocity components follow standard meteorological conventions:
        - u_x: eastward wind component (positive = eastward)
        - u_y: northward wind component (positive = northward)
        Spatial interpolation provides smooth velocity fields for realistic transport physics.
        
    Example:
        # Single position query
        position = np.array([25.5, 35.2])
        velocity = wind_field.velocity_at(position)
        
        # Spatial field evaluation
        positions = np.array([[x, y] for x in range(0, 100, 10) 
                                        for y in range(0, 100, 10)])
        velocity_field = wind_field.velocity_at(positions)
    """
```

#### Implementation Examples

**Constant Wind Field**
```python
class ConstantWindField:
    """Uniform directional flow with minimal computational overhead."""
    
    def __init__(self, velocity: Tuple[float, float] = (1.0, 0.0)):
        self.velocity = np.array(velocity)
    
    def velocity_at(self, positions: np.ndarray) -> np.ndarray:
        """Return constant velocity for all positions."""
        if positions.ndim == 1:
            return self.velocity.copy()
        else:
            # Broadcast to match position array shape
            return np.broadcast_to(self.velocity, (len(positions), 2)).copy()
    
    def step(self, dt: float = 1.0) -> None:
        """No temporal evolution for constant wind."""
        pass
    
    def reset(self, **kwargs) -> None:
        """Reset with optional velocity update."""
        if 'velocity' in kwargs:
            self.velocity = np.array(kwargs['velocity'])
```

**Turbulent Wind Field**
```python
class TurbulentWindField:
    """Realistic atmospheric boundary layer with gusty conditions."""
    
    def __init__(self, mean_velocity: Tuple[float, float] = (2.0, 0.0),
                 turbulence_intensity: float = 0.2, correlation_length: float = 10.0):
        self.mean_vel = np.array(mean_velocity)
        self.turbulence_intensity = turbulence_intensity
        self.correlation_length = correlation_length
        self.time = 0.0
        self.rng = np.random.RandomState(42)
        
        # Initialize turbulent field
        self._initialize_turbulence_field()
    
    def _initialize_turbulence_field(self) -> None:
        """Initialize spatially correlated turbulent fluctuations."""
        # Create spatial grid for turbulence
        self.grid_size = (20, 20)  # Grid resolution
        self.grid_spacing = 5.0    # Spatial resolution
        
        # Generate correlated random field using spectral method
        nx, ny = self.grid_size
        
        # Frequency domain coordinates
        kx = np.fft.fftfreq(nx, self.grid_spacing)
        ky = np.fft.fftfreq(ny, self.grid_spacing)
        kx_grid, ky_grid = np.meshgrid(kx, ky)
        k_magnitude = np.sqrt(kx_grid**2 + ky_grid**2)
        
        # Power spectrum for spatial correlation
        # Use exponential correlation function
        power_spectrum = np.exp(-k_magnitude * self.correlation_length)
        power_spectrum[0, 0] = 0  # Remove DC component
        
        # Generate turbulent fluctuations
        self.turbulence_u = self._generate_turbulent_component(power_spectrum)
        self.turbulence_v = self._generate_turbulent_component(power_spectrum)
    
    def _generate_turbulent_component(self, power_spectrum: np.ndarray) -> np.ndarray:
        """Generate single component of turbulent field."""
        # Random phases
        phases = self.rng.uniform(0, 2*np.pi, power_spectrum.shape)
        
        # Complex amplitudes
        amplitudes = np.sqrt(power_spectrum) * np.exp(1j * phases)
        
        # Inverse FFT to get spatial field
        turbulent_field = np.fft.ifft2(amplitudes).real
        
        # Scale by turbulence intensity
        turbulent_field *= self.turbulence_intensity * np.linalg.norm(self.mean_vel)
        
        return turbulent_field
    
    def velocity_at(self, positions: np.ndarray) -> np.ndarray:
        """Compute wind velocity including turbulent fluctuations."""
        if positions.ndim == 1:
            positions = positions.reshape(1, -1)
        
        velocities = np.zeros_like(positions)
        
        for i, pos in enumerate(positions):
            # Grid indices for interpolation
            grid_x = pos[0] / self.grid_spacing
            grid_y = pos[1] / self.grid_spacing
            
            # Bilinear interpolation of turbulent components
            turb_u = self._interpolate_turbulence(self.turbulence_u, grid_x, grid_y)
            turb_v = self._interpolate_turbulence(self.turbulence_v, grid_x, grid_y)
            
            # Combine mean and turbulent components
            velocities[i] = self.mean_vel + np.array([turb_u, turb_v])
        
        return velocities if len(positions) > 1 else velocities[0]
    
    def _interpolate_turbulence(self, field: np.ndarray, x: float, y: float) -> float:
        """Bilinear interpolation of turbulent field."""
        ny, nx = field.shape
        
        # Wrap coordinates for periodic boundary conditions
        x = x % nx
        y = y % ny
        
        # Grid indices
        x0, x1 = int(x), int(x + 1) % nx
        y0, y1 = int(y), int(y + 1) % ny
        
        # Interpolation weights
        wx = x - x0
        wy = y - y0
        
        # Bilinear interpolation
        value = ((1 - wx) * (1 - wy) * field[y0, x0] +
                 wx * (1 - wy) * field[y0, x1] +
                 (1 - wx) * wy * field[y1, x0] +
                 wx * wy * field[y1, x1])
        
        return value
    
    def step(self, dt: float = 1.0) -> None:
        """Evolve turbulent fluctuations over time."""
        self.time += dt
        
        # Simple temporal evolution - add random perturbations
        decay_factor = 0.95  # Turbulence decay rate
        noise_strength = 0.1 * self.turbulence_intensity
        
        # Add correlated noise and apply decay
        self.turbulence_u = (decay_factor * self.turbulence_u + 
                            noise_strength * self.rng.normal(0, 1, self.turbulence_u.shape))
        self.turbulence_v = (decay_factor * self.turbulence_v + 
                            noise_strength * self.rng.normal(0, 1, self.turbulence_v.shape))
```

## Sensor Interface Protocols

### SensorProtocol Interface Specification

The `SensorProtocol` defines configurable sensor interfaces for flexible agent perception modeling, enabling diverse sensing modalities without modifying core navigation logic while providing realistic sensing limitations and noise characteristics.

```python
@runtime_checkable
class SensorProtocol(Protocol):
    """Protocol defining configurable sensor interfaces for flexible agent perception modeling."""
```

#### Modular Sensing

**Binary Detection Interface**
```python
def detect(self, plume_state: Any, positions: np.ndarray) -> np.ndarray:
    """
    Perform binary detection at specified agent positions (BinarySensor implementation).
    
    Args:
        plume_state: Current plume model state providing concentration field access.
            Typically a PlumeModel instance or spatial concentration array.
        positions: Agent positions as array with shape (n_agents, 2) or (2,) for
            single agent. Coordinates in environment units.
            
    Returns:
        np.ndarray: Boolean detection results with shape (n_agents,) or scalar
            for single agent. True indicates odor detection above threshold.
            
    Performance: <0.1ms per agent for minimal sensing overhead
    
    Notes:
        - Binary sensors apply configurable thresholds with optional hysteresis
        - Noise modeling includes false positive and false negative rates
        - Non-binary sensors may return detection status based on confidence criteria
        
    Example:
        # Single agent detection
        position = np.array([15, 25])
        detected = sensor.detect(plume_state, position)
        
        # Multi-agent batch detection
        positions = np.array([[10, 20], [15, 25], [20, 30]])
        detections = sensor.detect(plume_state, positions)
    """
```

**Quantitative Measurement Interface**
```python
def measure(self, plume_state: Any, positions: np.ndarray) -> np.ndarray:
    """
    Perform quantitative measurements at specified agent positions (ConcentrationSensor).
    
    Args:
        plume_state: Current plume model state providing concentration field access
        positions: Agent positions as array with shape (n_agents, 2) or (2,)
        
    Returns:
        np.ndarray: Quantitative measurement values with shape (n_agents,) or
            scalar for single agent. Values in sensor-specific units with
            configured dynamic range and resolution.
            
    Performance: <0.1ms per agent for minimal sensing overhead
    
    Notes:
        - Concentration sensors provide calibrated measurements with configurable parameters
        - Temporal filtering and response delays model realistic sensor dynamics
        - Measurements may include saturation effects and calibration drift
        
    Example:
        # Single agent measurement
        position = np.array([15, 25])
        concentration = sensor.measure(plume_state, position)
        
        # Multi-agent batch measurement
        positions = np.array([[10, 20], [15, 25], [20, 30]])
        concentrations = sensor.measure(plume_state, positions)
    """
```

**Gradient Computation Interface**
```python
def compute_gradient(self, plume_state: Any, positions: np.ndarray) -> np.ndarray:
    """
    Compute spatial gradients at specified agent positions (GradientSensor implementation).
    
    Args:
        plume_state: Current plume model state providing concentration field access
        positions: Agent positions as array with shape (n_agents, 2) or (2,)
        
    Returns:
        np.ndarray: Gradient vectors with shape (n_agents, 2) or (2,) for single
            agent. Components represent [∂c/∂x, ∂c/∂y] spatial derivatives in
            concentration units per distance unit.
            
    Performance: <0.2ms per agent due to multi-point sampling requirements
    
    Notes:
        - Gradient sensors use finite difference methods with configurable spatial resolution
        - Multi-point sampling enables accurate gradient estimation with noise suppression
        - Adaptive step sizing and error estimation for robust computation
        
    Example:
        # Single agent gradient
        position = np.array([15, 25])
        gradient = sensor.compute_gradient(plume_state, position)
        
        # Multi-agent batch gradients
        positions = np.array([[10, 20], [15, 25], [20, 30]])
        gradients = sensor.compute_gradient(plume_state, positions)
    """
```

#### Plume Model Integration

**Configuration Interface**
```python
def configure(self, **kwargs: Any) -> None:
    """
    Update sensor configuration parameters during runtime.
    
    Args:
        **kwargs: Sensor-specific configuration parameters:
            - threshold: Detection threshold for binary sensors
            - dynamic_range: Measurement range for concentration sensors
            - spatial_resolution: Finite difference step size for gradient sensors
            - noise_parameters: False positive/negative rates, measurement noise
            - temporal_filtering: Response time constants and history length
            
    Notes:
        - Configuration updates apply immediately to subsequent sensor operations
        - Parameter validation ensures physical consistency and performance requirements
        - Temporal parameters may trigger reset of internal state buffers
        
    Example:
        # Update binary sensor threshold
        sensor.configure(threshold=0.05, false_positive_rate=0.01)
        
        # Adjust concentration sensor range
        sensor.configure(dynamic_range=(0, 2.0), resolution=0.0001)
    """
```

#### Sensor Implementation Examples

**Binary Sensor with Noise Modeling**
```python
class BinarySensor:
    """Threshold-based detection with configurable false positive/negative rates."""
    
    def __init__(self, threshold: float = 0.1, false_positive_rate: float = 0.02,
                 false_negative_rate: float = 0.05, hysteresis: float = 0.01):
        self.threshold = threshold
        self.false_positive_rate = false_positive_rate
        self.false_negative_rate = false_negative_rate
        self.hysteresis = hysteresis
        self.rng = np.random.RandomState(42)
        
        # State for hysteresis
        self._last_detections = {}
    
    def detect(self, plume_state: Any, positions: np.ndarray) -> np.ndarray:
        """Perform binary detection with noise and hysteresis."""
        if positions.ndim == 1:
            positions = positions.reshape(1, -1)
        
        # Sample concentrations from plume model
        if hasattr(plume_state, 'concentration_at'):
            concentrations = plume_state.concentration_at(positions)
        else:
            # Assume plume_state is concentration array
            concentrations = self._sample_concentrations(plume_state, positions)
        
        detections = np.zeros(len(positions), dtype=bool)
        
        for i, (pos, conc) in enumerate(zip(positions, concentrations)):
            # Apply hysteresis using position as key
            pos_key = tuple(pos)
            last_detection = self._last_detections.get(pos_key, False)
            
            if last_detection:
                # Higher threshold for turning off (hysteresis)
                threshold_effective = self.threshold + self.hysteresis
            else:
                # Normal threshold for turning on
                threshold_effective = self.threshold
            
            # Basic detection
            detected = conc > threshold_effective
            
            # Apply noise
            if detected:
                # False negative
                if self.rng.random() < self.false_negative_rate:
                    detected = False
            else:
                # False positive
                if self.rng.random() < self.false_positive_rate:
                    detected = True
            
            detections[i] = detected
            self._last_detections[pos_key] = detected
        
        return detections if len(positions) > 1 else detections[0]
    
    def _sample_concentrations(self, concentration_array: np.ndarray, 
                              positions: np.ndarray) -> np.ndarray:
        """Sample concentrations using bilinear interpolation."""
        height, width = concentration_array.shape
        concentrations = np.zeros(len(positions))
        
        for i, pos in enumerate(positions):
            x, y = pos
            
            # Convert to array indices
            x_idx = np.clip(x, 0, width - 1)
            y_idx = np.clip(y, 0, height - 1)
            
            # Bilinear interpolation
            x0, x1 = int(x_idx), min(int(x_idx) + 1, width - 1)
            y0, y1 = int(y_idx), min(int(y_idx) + 1, height - 1)
            
            wx = x_idx - x0
            wy = y_idx - y0
            
            conc = ((1 - wx) * (1 - wy) * concentration_array[y0, x0] +
                    wx * (1 - wy) * concentration_array[y0, x1] +
                    (1 - wx) * wy * concentration_array[y1, x0] +
                    wx * wy * concentration_array[y1, x1])
            
            concentrations[i] = conc
        
        return concentrations
    
    def configure(self, **kwargs) -> None:
        """Update sensor configuration."""
        if 'threshold' in kwargs:
            self.threshold = kwargs['threshold']
        if 'false_positive_rate' in kwargs:
            self.false_positive_rate = kwargs['false_positive_rate']
        if 'false_negative_rate' in kwargs:
            self.false_negative_rate = kwargs['false_negative_rate']
        if 'hysteresis' in kwargs:
            self.hysteresis = kwargs['hysteresis']
```

**Concentration Sensor with Dynamic Range**
```python
class ConcentrationSensor:
    """Quantitative measurements with dynamic range and noise modeling."""
    
    def __init__(self, dynamic_range: Tuple[float, float] = (0.0, 1.0),
                 resolution: float = 0.001, noise_std: float = 0.01):
        self.range_min, self.range_max = dynamic_range
        self.resolution = resolution
        self.noise_std = noise_std
        self.rng = np.random.RandomState(42)
    
    def measure(self, plume_state: Any, positions: np.ndarray) -> np.ndarray:
        """Perform quantitative concentration measurement."""
        if positions.ndim == 1:
            positions = positions.reshape(1, -1)
        
        # Sample true concentrations
        if hasattr(plume_state, 'concentration_at'):
            true_concentrations = plume_state.concentration_at(positions)
        else:
            true_concentrations = self._sample_concentrations(plume_state, positions)
        
        # Apply sensor characteristics
        measurements = np.zeros(len(positions))
        
        for i, true_conc in enumerate(true_concentrations):
            # Add measurement noise
            noisy_conc = true_conc + self.rng.normal(0, self.noise_std)
            
            # Apply dynamic range clipping
            clipped_conc = np.clip(noisy_conc, self.range_min, self.range_max)
            
            # Apply resolution quantization
            quantized_conc = np.round(clipped_conc / self.resolution) * self.resolution
            
            measurements[i] = quantized_conc
        
        return measurements if len(positions) > 1 else measurements[0]
    
    def configure(self, **kwargs) -> None:
        """Update sensor configuration."""
        if 'dynamic_range' in kwargs:
            self.range_min, self.range_max = kwargs['dynamic_range']
        if 'resolution' in kwargs:
            self.resolution = kwargs['resolution']
        if 'noise_std' in kwargs:
            self.noise_std = kwargs['noise_std']
```

**Gradient Sensor with Finite Difference**
```python
class GradientSensor:
    """Spatial derivative computation for directional navigation cues."""
    
    def __init__(self, spatial_resolution: Tuple[float, float] = (0.5, 0.5),
                 method: str = 'central'):
        self.dx, self.dy = spatial_resolution
        self.method = method
    
    def compute_gradient(self, plume_state: Any, positions: np.ndarray) -> np.ndarray:
        """Compute spatial gradients using finite difference methods."""
        if positions.ndim == 1:
            positions = positions.reshape(1, -1)
        
        gradients = np.zeros_like(positions)
        
        for i, pos in enumerate(positions):
            x, y = pos
            
            if self.method == 'central':
                # Central difference for better accuracy
                pos_x_plus = np.array([[x + self.dx, y]])
                pos_x_minus = np.array([[x - self.dx, y]])
                pos_y_plus = np.array([[x, y + self.dy]])
                pos_y_minus = np.array([[x, y - self.dy]])
                
                if hasattr(plume_state, 'concentration_at'):
                    conc_x_plus = plume_state.concentration_at(pos_x_plus)[0]
                    conc_x_minus = plume_state.concentration_at(pos_x_minus)[0]
                    conc_y_plus = plume_state.concentration_at(pos_y_plus)[0]
                    conc_y_minus = plume_state.concentration_at(pos_y_minus)[0]
                else:
                    conc_x_plus = self._sample_concentrations(plume_state, pos_x_plus)[0]
                    conc_x_minus = self._sample_concentrations(plume_state, pos_x_minus)[0]
                    conc_y_plus = self._sample_concentrations(plume_state, pos_y_plus)[0]
                    conc_y_minus = self._sample_concentrations(plume_state, pos_y_minus)[0]
                
                # Central difference gradients
                grad_x = (conc_x_plus - conc_x_minus) / (2 * self.dx)
                grad_y = (conc_y_plus - conc_y_minus) / (2 * self.dy)
                
            elif self.method == 'forward':
                # Forward difference
                pos_center = pos.reshape(1, -1)
                pos_x_plus = np.array([[x + self.dx, y]])
                pos_y_plus = np.array([[x, y + self.dy]])
                
                if hasattr(plume_state, 'concentration_at'):
                    conc_center = plume_state.concentration_at(pos_center)[0]
                    conc_x_plus = plume_state.concentration_at(pos_x_plus)[0]
                    conc_y_plus = plume_state.concentration_at(pos_y_plus)[0]
                else:
                    conc_center = self._sample_concentrations(plume_state, pos_center)[0]
                    conc_x_plus = self._sample_concentrations(plume_state, pos_x_plus)[0]
                    conc_y_plus = self._sample_concentrations(plume_state, pos_y_plus)[0]
                
                # Forward difference gradients
                grad_x = (conc_x_plus - conc_center) / self.dx
                grad_y = (conc_y_plus - conc_center) / self.dy
            
            gradients[i] = [grad_x, grad_y]
        
        return gradients if len(positions) > 1 else gradients[0]
    
    def configure(self, **kwargs) -> None:
        """Update gradient sensor configuration."""
        if 'spatial_resolution' in kwargs:
            self.dx, self.dy = kwargs['spatial_resolution']
        if 'method' in kwargs:
            self.method = kwargs['method']
```

## Space Construction Protocols

### AgentObservationProtocol Interface Specification

The `AgentObservationProtocol` defines standardized observation structures for agent-environment interaction, enabling flexible observation space construction that automatically adapts to active sensor configurations while maintaining type safety and Gymnasium compatibility.

```python
@runtime_checkable
class AgentObservationProtocol(Protocol):
    """Protocol defining standardized observation structures for agent-environment interaction."""
```

#### Structured Observations

**Observation Dictionary Construction**
```python
def construct_observation(
    self, 
    agent_state: Dict[str, Any], 
    plume_state: Any, 
    **kwargs: Any
) -> Dict[str, Any]:
    """
    Construct structured observation dictionary from agent and environment state.
    
    Args:
        agent_state: Current agent state dictionary containing position, orientation,
            speed, and other navigator-managed properties
        plume_state: Current plume model state for sensor sampling
        **kwargs: Additional observation components (wind data, derived metrics, etc.)
        
    Returns:
        Dict[str, Any]: Structured observation dictionary with standardized keys:
            - 'position': Agent position as (x, y) array
            - 'orientation': Agent orientation in degrees
            - 'speed': Current agent speed
            - 'sensor_readings': Dictionary of sensor-specific observations
            - Additional keys from kwargs and custom observation components
            
    Performance: <0.5ms per agent for real-time step performance
    
    Notes:
        - Observation structure adapts automatically to active sensor configuration
        - Sensor readings organized by sensor type and identifier for clarity
        - Custom observation components can extend base observations
        
    Example:
        obs = obs_protocol.construct_observation(agent_state, plume_state)
        assert 'position' in obs and 'sensor_readings' in obs
        
        # With additional wind data
        obs = obs_protocol.construct_observation(
            agent_state, plume_state, wind_velocity=(2.0, 1.0)
        )
    """
```

#### Dynamic Observation Space Adaptation

**Gymnasium Space Integration**
```python
def get_observation_space(self) -> Optional['spaces.Space']:
    """
    Construct Gymnasium observation space matching constructed observations.
    
    Returns:
        Optional[spaces.Space]: Gymnasium observation space (typically spaces.Dict)
            defining the structure and bounds of observation dictionaries. Returns
            None if Gymnasium is not available.
            
    Notes:
        - Observation space structure automatically adapts to active sensor configuration
        - Space bounds inferred from sensor specifications and agent state constraints
        - Enables proper RL framework integration for training and evaluation
        
    Example:
        obs_space = obs_protocol.get_observation_space()
        assert isinstance(obs_space, gymnasium.spaces.Dict)
        assert 'position' in obs_space.spaces
    """
```

#### Implementation Example

**Multi-Sensor Observation Protocol**
```python
class MultiSensorObservationProtocol:
    """Observation protocol integrating multiple sensor types."""
    
    def __init__(self, sensors: List[SensorProtocol], 
                 domain_bounds: Tuple[float, float] = (100.0, 100.0)):
        self.sensors = sensors
        self.domain_bounds = domain_bounds
    
    def construct_observation(self, agent_state: Dict[str, Any], 
                             plume_state: Any, **kwargs: Any) -> Dict[str, Any]:
        """Construct multi-sensor observation dictionary."""
        # Base agent state observations
        obs = {
            'position': agent_state['position'],
            'orientation': agent_state['orientation'],
            'speed': agent_state['speed']
        }
        
        # Sensor readings
        sensor_readings = {}
        agent_positions = agent_state['position'].reshape(1, -1)
        
        for sensor_id, sensor in enumerate(self.sensors):
            sensor_key = f"sensor_{sensor_id}"
            
            # Use appropriate sensor method based on type
            if hasattr(sensor, 'detect'):
                sensor_readings[sensor_key] = sensor.detect(plume_state, agent_positions)[0]
            elif hasattr(sensor, 'measure'):
                sensor_readings[sensor_key] = sensor.measure(plume_state, agent_positions)[0]
            elif hasattr(sensor, 'compute_gradient'):
                sensor_readings[sensor_key] = sensor.compute_gradient(plume_state, agent_positions)[0]
        
        obs['sensor_readings'] = sensor_readings
        
        # Add additional components from kwargs
        for key, value in kwargs.items():
            obs[key] = value
        
        return obs
    
    def get_observation_space(self) -> Optional['spaces.Space']:
        """Construct observation space for multi-sensor setup."""
        if spaces is None:
            return None
        
        # Agent state spaces
        position_space = spaces.Box(
            low=0.0, high=max(self.domain_bounds), 
            shape=(2,), dtype=np.float32
        )
        orientation_space = spaces.Box(
            low=0.0, high=360.0, shape=(), dtype=np.float32
        )
        speed_space = spaces.Box(
            low=0.0, high=10.0, shape=(), dtype=np.float32  # Assume max speed of 10
        )
        
        # Sensor spaces
        sensor_spaces = {}
        for sensor_id, sensor in enumerate(self.sensors):
            sensor_key = f"sensor_{sensor_id}"
            
            if hasattr(sensor, 'detect'):
                # Binary sensor
                sensor_spaces[sensor_key] = spaces.Discrete(2)
            elif hasattr(sensor, 'measure'):
                # Concentration sensor - assume [0, 1] range
                sensor_spaces[sensor_key] = spaces.Box(
                    low=0.0, high=1.0, shape=(), dtype=np.float32
                )
            elif hasattr(sensor, 'compute_gradient'):
                # Gradient sensor - assume [-1, 1] range for gradients
                sensor_spaces[sensor_key] = spaces.Box(
                    low=-1.0, high=1.0, shape=(2,), dtype=np.float32
                )
        
        # Combine into structured space
        observation_spaces = {
            'position': position_space,
            'orientation': orientation_space,
            'speed': speed_space,
            'sensor_readings': spaces.Dict(sensor_spaces)
        }
        
        return spaces.Dict(observation_spaces)
```

### AgentActionProtocol Interface Specification

The `AgentActionProtocol` defines standardized action structures for agent control interfaces, enabling flexible action space construction supporting various control modalities while maintaining consistency with NavigatorProtocol implementations.

```python
@runtime_checkable
class AgentActionProtocol(Protocol):
    """Protocol defining standardized action structures for agent control interfaces."""
```

#### Structured Actions

**Action Validation and Processing**
```python
def validate_action(self, action: Union[np.ndarray, Dict[str, Any]]) -> Union[np.ndarray, Dict[str, Any]]:
    """
    Validate and constrain action values within physical and safety limits.
    
    Args:
        action: Raw action from RL framework as array or dictionary
        
    Returns:
        Union[np.ndarray, Dict[str, Any]]: Validated action with constraints applied
        
    Performance: <0.05ms per agent for minimal control overhead
    
    Notes:
        Constraint enforcement includes:
        - Physical limits (maximum speeds, accelerations)
        - Safety boundaries (collision avoidance, environment bounds)
        - Numerical stability (avoiding division by zero, NaN values)
        Invalid actions are clipped or modified to nearest valid values
        
    Example:
        action = np.array([2.5, 60.0])  # [linear_vel, angular_vel]
        validated = action_protocol.validate_action(action)
        # Clips to [2.0, 45.0] based on max_speed and max_angular_velocity
    """

def process_action(self, action: Union[np.ndarray, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Process raw action into navigator-compatible control parameters.
    
    Args:
        action: Validated action from RL framework as array or dictionary
        
    Returns:
        Dict[str, Any]: Processed action dictionary with keys matching
            NavigatorProtocol control interface:
            - 'target_speed': Desired linear velocity
            - 'target_angular_velocity': Desired angular velocity
            - 'control_mode': Control method specification
            - Additional control parameters as needed
            
    Performance: <0.1ms per agent for minimal control overhead
    
    Notes:
        - Action processing handles conversion between different control modalities
        - May include coordinate transformations, control law evaluation
        - Processed actions immediately applicable to navigator step() methods
        
    Example:
        action = np.array([1.5, 20.0])
        processed = action_protocol.process_action(action)
        assert 'target_speed' in processed
    """
```

#### Action Space Integration

**Gymnasium Action Space Construction**
```python
def get_action_space(self) -> Optional['spaces.Space']:
    """
    Construct Gymnasium action space matching expected action format.
    
    Returns:
        Optional[spaces.Space]: Gymnasium action space defining valid action
            structure and value ranges. Returns None if Gymnasium not available.
            
    Notes:
        - Action space automatically reflects control modality constraints
        - Space bounds match validation constraints for consistency
        - Enables proper RL framework integration and training
        
    Example:
        action_space = action_protocol.get_action_space()
        sample_action = action_space.sample()
        assert action_protocol.validate_action(sample_action)
    """
```

## Implementation Guidelines

### Protocol Compliance Requirements

All protocol implementations in plume_nav_sim v1.0 must satisfy the following compliance requirements to ensure consistent behavior and seamless integration:

#### Type Annotations

**Complete Method Signatures**
- All protocol methods must include comprehensive type annotations
- Parameter types must specify exact expected formats (Union types for flexibility)
- Return types must be precisely defined with shape information for arrays
- Optional parameters must be explicitly marked with `Optional[]` or default values

```python
# Correct type annotation example
def apply_policy(
    self, 
    positions: np.ndarray,  # Shape: (n_agents, 2) or (2,)
    velocities: Optional[np.ndarray] = None  # Same shape as positions
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Return type depends on whether velocities provided."""
```

#### Method Implementation Patterns

**Error Handling Standards**
- All methods must include comprehensive input validation
- Invalid inputs should raise informative exceptions with specific error messages
- Performance-critical methods should use assert statements for debug builds
- Production code should gracefully handle edge cases without crashing

```python
# Standard error handling pattern
def initialize_positions(self, num_agents: int, **kwargs) -> np.ndarray:
    if num_agents <= 0:
        raise ValueError(f"num_agents must be positive, got {num_agents}")
    
    if not isinstance(num_agents, int):
        raise TypeError(f"num_agents must be int, got {type(num_agents)}")
    
    # Implementation...
    
    # Validate output
    result = self._generate_positions(num_agents, **kwargs)
    assert result.shape == (num_agents, 2), "Internal error: invalid shape"
    
    return result
```

#### Performance Considerations

**Timing Requirements**
- All protocol methods must meet specified performance requirements
- Critical path methods should be profiled and optimized
- Memory allocation should be minimized in high-frequency operations
- Vectorized NumPy operations preferred over Python loops

```python
# Performance-optimized implementation pattern
def check_violations(self, positions: np.ndarray) -> np.ndarray:
    """Must execute in <0.5ms for 100 agents."""
    # Vectorized boundary checking - single operation for all agents
    width, height = self.bounds
    
    # Use NumPy broadcasting for efficiency
    violations = ((positions[..., 0] < 0) | (positions[..., 0] >= width) |
                 (positions[..., 1] < 0) | (positions[..., 1] >= height))
    
    return violations
```

### Error Handling

**Exception Hierarchy**
```python
class PlumeNavSimError(Exception):
    """Base exception for plume_nav_sim errors."""
    pass

class ProtocolComplianceError(PlumeNavSimError):
    """Raised when a component doesn't implement required protocol methods."""
    pass

class ConfigurationError(PlumeNavSimError):
    """Raised when configuration parameters are invalid."""
    pass

class PerformanceError(PlumeNavSimError):
    """Raised when performance requirements are not met."""
    pass
```

**Validation Patterns**
```python
class ValidationMixin:
    """Mixin providing common validation patterns for protocol implementations."""
    
    def validate_array_shape(self, array: np.ndarray, expected_shape: Tuple[int, ...],
                            name: str) -> None:
        """Validate NumPy array has expected shape."""
        if array.shape != expected_shape:
            raise ValueError(
                f"{name} must have shape {expected_shape}, got {array.shape}"
            )
    
    def validate_array_range(self, array: np.ndarray, min_val: float, max_val: float,
                           name: str) -> None:
        """Validate NumPy array values are within specified range."""
        if np.any(array < min_val) or np.any(array > max_val):
            raise ValueError(
                f"{name} values must be in range [{min_val}, {max_val}], "
                f"got range [{np.min(array)}, {np.max(array)}]"
            )
    
    def validate_positive(self, value: Union[int, float], name: str) -> None:
        """Validate numeric value is positive."""
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
```

### Testing Strategies

**Protocol Compliance Testing**
```python
import pytest
from typing import get_type_hints

class ProtocolComplianceTestMixin:
    """Mixin for testing protocol compliance."""
    
    def test_protocol_compliance(self, component, protocol_class):
        """Test that component implements all protocol methods."""
        # Check all required methods exist
        for method_name in dir(protocol_class):
            if not method_name.startswith('_'):
                assert hasattr(component, method_name), \
                    f"Component missing required method: {method_name}"
        
        # Check type annotations match
        protocol_hints = get_type_hints(protocol_class)
        component_hints = get_type_hints(type(component))
        
        for method_name, expected_type in protocol_hints.items():
            if hasattr(component, method_name):
                actual_type = component_hints.get(method_name)
                assert actual_type == expected_type, \
                    f"Method {method_name} type mismatch: expected {expected_type}, got {actual_type}"

class TestSourceProtocolCompliance(ProtocolComplianceTestMixin):
    """Test cases for SourceProtocol implementations."""
    
    @pytest.fixture
    def point_source(self):
        return PointSource(position=(50, 50), emission_rate=1000.0)
    
    def test_point_source_compliance(self, point_source):
        self.test_protocol_compliance(point_source, SourceProtocol)
    
    def test_emission_rate_non_negative(self, point_source):
        rate = point_source.get_emission_rate()
        assert rate >= 0.0, "Emission rate must be non-negative"
    
    def test_position_within_bounds(self, point_source):
        x, y = point_source.get_position()
        assert isinstance(x, float) and isinstance(y, float)
        # Additional bounds checking based on specific requirements
    
    def test_performance_requirements(self, point_source):
        import time
        
        # Test emission rate performance
        start_time = time.time()
        for _ in range(1000):
            point_source.get_emission_rate()
        duration = time.time() - start_time
        
        assert duration < 0.1, f"get_emission_rate too slow: {duration:.3f}s for 1000 calls"
```

**Integration Testing**
```python
class TestProtocolIntegration:
    """Test protocol integration with core systems."""
    
    def test_navigator_source_integration(self):
        """Test navigator integrates correctly with source protocol."""
        # Create components
        navigator = NavigatorFactory.single_agent(position=(10, 10))
        source = PointSource(position=(50, 50), emission_rate=1000.0)
        
        # Test integration
        navigator.source = source  # Dependency injection
        
        # Verify source is accessible
        assert navigator.source is not None
        assert isinstance(navigator.source.get_emission_rate(), float)
    
    def test_multi_component_integration(self):
        """Test full component suite integration."""
        # Create all v1.0 components
        components = {
            'source': PointSource(position=(50, 50), emission_rate=1000.0),
            'boundary_policy': TerminatePolicy(domain_bounds=(100, 100)),
            'action_interface': Continuous2DAction(max_linear_velocity=2.0),
            'recorder': ParquetRecorder(output_dir="./test_data"),
            'stats_aggregator': StandardStatsAggregator(),
            'agent_initializer': UniformRandomInitializer(domain_bounds=(100, 100))
        }
        
        # Validate all components
        validation_results = NavigatorFactory.validate_v1_component_suite(**components)
        assert all(validation_results.values()), f"Component validation failed: {validation_results}"
```

## Configuration Patterns

### Hydra Integration

The protocol system integrates seamlessly with Hydra's configuration management, enabling powerful configuration-driven component composition and runtime parameter control.

#### Component Composition

**Modular Configuration Groups**
```yaml
# conf/base/source/point_source.yaml
_target_: plume_nav_sim.core.sources.PointSource
position: [50.0, 50.0]
emission_rate: 1000.0

# conf/base/source/dynamic_source.yaml
_target_: plume_nav_sim.core.sources.DynamicSource
initial_position: [25.0, 75.0]
emission_pattern: "sinusoidal"
period: 60.0
base_rate: 800.0

# conf/base/boundary/terminate.yaml
_target_: plume_nav_sim.core.boundaries.TerminatePolicy
domain_bounds: [100, 100]
status_on_violation: "oob"

# conf/base/boundary/bounce.yaml
_target_: plume_nav_sim.core.boundaries.BouncePolicy
domain_bounds: [100, 100]
energy_loss: 0.1
```

**Configuration Composition**
```yaml
# conf/config.yaml
defaults:
  - base/navigator: single_agent
  - base/source: point_source
  - base/boundary: terminate
  - base/action: continuous_2d
  - base/record: parquet
  - base/stats: standard
  - base/agent_init: uniform_random
  - _self_

# Override parameters
navigator:
  position: [10.0, 20.0]
  max_speed: 2.5

source:
  emission_rate: 1500.0

boundary:
  domain_bounds: [150, 150]
```

#### Dependency Injection

**Automatic Component Instantiation**
```python
@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main experiment function with Hydra configuration."""
    
    # Create components using dependency injection
    navigator = NavigatorFactory.from_config(cfg.navigator)
    source = NavigatorFactory.create_source(cfg.source)
    boundary_policy = NavigatorFactory.create_boundary_policy(cfg.boundary)
    action_interface = NavigatorFactory.create_action_interface(cfg.action)
    recorder = NavigatorFactory.create_recorder(cfg.record)
    stats_aggregator = NavigatorFactory.create_stats_aggregator(cfg.stats)
    agent_initializer = NavigatorFactory.create_agent_initializer(cfg.agent_init)
    
    # Create complete modular environment
    env = NavigatorFactory.create_modular_environment(
        navigator_config=cfg.navigator,
        plume_model_config=cfg.plume_model,
        source_config=cfg.source,
        boundary_policy_config=cfg.boundary,
        action_interface_config=cfg.action,
        recorder_config=cfg.record,
        stats_aggregator_config=cfg.stats,
        agent_initializer_config=cfg.agent_init
    )
    
    # Run experiment
    run_experiment(env, cfg)
```

#### Runtime Switching

**Command-Line Component Selection**
```bash
# Switch to different source implementation
python main.py source=dynamic_source source.period=30.0

# Use bounce boundary instead of terminate
python main.py boundary=bounce boundary.energy_loss=0.2

# Change to discrete action interface
python main.py action=cardinal_discrete action.move_speed=1.5

# Use HDF5 recorder instead of Parquet
python main.py record=hdf5 record.compression=gzip
```

**Multi-Run Parameter Sweeps**
```yaml
# conf/sweep.yaml
defaults:
  - base/config
  
hydra:
  mode: MULTIRUN
  sweep:
    dir: ./outputs/${now:%Y-%m-%d_%H-%M-%S}
    subdir: ${hydra:job.num}

# Parameter sweep configuration
source:
  emission_rate: choice(500, 1000, 1500, 2000)

boundary:
  _target_: choice(
    "plume_nav_sim.core.boundaries.TerminatePolicy",
    "plume_nav_sim.core.boundaries.BouncePolicy"
  )

navigator:
  max_speed: range(1.0, 3.0, 0.5)
```

### Environment Variable Overrides

**Runtime Configuration Control**
```python
# Environment variable override support
import os
from hydra import compose, initialize_config_store
from omegaconf import OmegaConf

def create_environment_with_overrides():
    """Create environment with environment variable overrides."""
    
    # Initialize Hydra
    with initialize_config_store(config_path="../conf"):
        cfg = compose(config_name="config")
        
        # Apply environment variable overrides
        overrides = {}
        
        # Source configuration overrides
        if 'PLUME_SOURCE_EMISSION_RATE' in os.environ:
            overrides['source.emission_rate'] = float(os.environ['PLUME_SOURCE_EMISSION_RATE'])
        
        if 'PLUME_SOURCE_POSITION' in os.environ:
            pos_str = os.environ['PLUME_SOURCE_POSITION']
            overrides['source.position'] = [float(x) for x in pos_str.split(',')]
        
        # Navigator configuration overrides
        if 'PLUME_NAVIGATOR_MAX_SPEED' in os.environ:
            overrides['navigator.max_speed'] = float(os.environ['PLUME_NAVIGATOR_MAX_SPEED'])
        
        # Boundary configuration overrides
        if 'PLUME_BOUNDARY_TYPE' in os.environ:
            boundary_type = os.environ['PLUME_BOUNDARY_TYPE']
            if boundary_type == 'terminate':
                overrides['boundary._target_'] = 'plume_nav_sim.core.boundaries.TerminatePolicy'
            elif boundary_type == 'bounce':
                overrides['boundary._target_'] = 'plume_nav_sim.core.boundaries.BouncePolicy'
        
        # Apply overrides
        with open_dict(cfg):
            for key, value in overrides.items():
                OmegaConf.set(cfg, key, value)
        
        return NavigatorFactory.create_modular_environment(
            navigator_config=cfg.navigator,
            plume_model_config=cfg.plume_model,
            source_config=cfg.source,
            boundary_policy_config=cfg.boundary
        )

# Usage example
env = create_environment_with_overrides()
```

### Validation Schemas

**Pydantic Configuration Validation**
```python
from pydantic import BaseModel, validator, Field
from typing import List, Union, Tuple

class SourceConfig(BaseModel):
    """Source configuration schema with validation."""
    
    _target_: str = Field(..., description="Source implementation class path")
    position: Tuple[float, float] = Field(..., description="Source position (x, y)")
    emission_rate: float = Field(gt=0, description="Emission rate (must be positive)")
    
    @validator('position')
    def validate_position(cls, v):
        if len(v) != 2:
            raise ValueError("Position must be 2D coordinate tuple")
        if any(coord < 0 for coord in v):
            raise ValueError("Position coordinates must be non-negative")
        return v

class BoundaryPolicyConfig(BaseModel):
    """Boundary policy configuration schema with validation."""
    
    _target_: str = Field(..., description="Boundary policy implementation class path")
    domain_bounds: Tuple[float, float] = Field(..., description="Domain size (width, height)")
    
    @validator('domain_bounds')
    def validate_domain_bounds(cls, v):
        if len(v) != 2:
            raise ValueError("Domain bounds must be (width, height) tuple")
        if any(bound <= 0 for bound in v):
            raise ValueError("Domain bounds must be positive")
        return v

class ActionInterfaceConfig(BaseModel):
    """Action interface configuration schema with validation."""
    
    _target_: str = Field(..., description="Action interface implementation class path")
    max_linear_velocity: float = Field(gt=0, le=10, description="Maximum linear velocity")
    max_angular_velocity: float = Field(gt=0, le=180, description="Maximum angular velocity (degrees)")

class RecorderConfig(BaseModel):
    """Recorder configuration schema with validation."""
    
    _target_: str = Field(..., description="Recorder implementation class path")
    output_dir: str = Field(..., description="Output directory path")
    compression: str = Field("snappy", description="Compression method")
    buffer_size: int = Field(1000, gt=0, description="Buffer size for batching")
    
    @validator('compression')
    def validate_compression(cls, v):
        valid_compressions = {'snappy', 'gzip', 'brotli', 'lz4', 'none'}
        if v not in valid_compressions:
            raise ValueError(f"Compression must be one of {valid_compressions}")
        return v

class ExperimentConfig(BaseModel):
    """Complete experiment configuration with nested validation."""
    
    source: SourceConfig
    boundary: BoundaryPolicyConfig
    action: ActionInterfaceConfig
    record: RecorderConfig
    
    # Experiment parameters
    num_episodes: int = Field(100, gt=0, description="Number of episodes to run")
    max_steps_per_episode: int = Field(1000, gt=0, description="Maximum steps per episode")
    random_seed: int = Field(42, description="Random seed for reproducibility")
```

## Migration and Compatibility

### V0.3 to V1.0 Migration

The migration from plume_nav_sim v0.3.0 to v1.0 involves transforming from a project-specific implementation into a general-purpose, extensible simulation toolkit. This section provides comprehensive migration guidance.

#### Backward Compatibility

**Legacy API Support**
The v1.0 architecture maintains backward compatibility through compatibility shims and adapter patterns:

```python
# Legacy v0.3.0 usage pattern
from plume_nav_sim.envs.plume_navigation_env import PlumeNavigationEnv

# Still works in v1.0 with compatibility layer
env = PlumeNavigationEnv(
    position=(10.0, 20.0),
    max_speed=2.0,
    domain_bounds=(100, 100)
)

# Internally, this creates v1.0 components with legacy configuration
```

**Configuration Migration**
```python
def migrate_v03_config_to_v10(legacy_config: dict) -> dict:
    """Convert v0.3.0 configuration to v1.0 modular format."""
    
    v10_config = {
        'navigator': {
            '_target_': 'plume_nav_sim.core.controllers.SingleAgentController',
            'position': legacy_config.get('position', (0.0, 0.0)),
            'max_speed': legacy_config.get('max_speed', 1.0),
            'orientation': legacy_config.get('orientation', 0.0)
        },
        'source': {
            '_target_': 'plume_nav_sim.core.sources.PointSource',
            'position': legacy_config.get('source_position', (50.0, 50.0)),
            'emission_rate': legacy_config.get('source_strength', 1000.0)
        },
        'boundary': {
            '_target_': 'plume_nav_sim.core.boundaries.TerminatePolicy',
            'domain_bounds': legacy_config.get('domain_bounds', (100, 100))
        },
        'action': {
            '_target_': 'plume_nav_sim.core.actions.Continuous2DAction',
            'max_linear_velocity': legacy_config.get('max_speed', 1.0),
            'max_angular_velocity': 45.0
        }
    }
    
    # Add optional components if present
    if 'recording_enabled' in legacy_config and legacy_config['recording_enabled']:
        v10_config['record'] = {
            '_target_': 'plume_nav_sim.recording.backends.parquet_backend.ParquetRecorder',
            'output_dir': legacy_config.get('output_dir', './data'),
            'compression': 'snappy'
        }
    
    return v10_config
```

#### Deprecation Warnings

**Graceful Transition Support**
```python
import warnings
from typing import Optional

class LegacyNavigationEnvironment:
    """Deprecated v0.3.0 environment with migration warnings."""
    
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "LegacyNavigationEnvironment is deprecated and will be removed in v1.1. "
            "Please migrate to the new modular architecture using NavigatorFactory.create_modular_environment(). "
            "See migration guide: docs/migration_guide_v1.md",
            DeprecationWarning,
            stacklevel=2
        )
        
        # Convert legacy parameters to v1.0 configuration
        v10_config = self._convert_legacy_params(*args, **kwargs)
        
        # Create v1.0 environment internally
        self._v10_env = NavigatorFactory.create_modular_environment(**v10_config)
    
    def step(self, action):
        """Legacy step method with API conversion."""
        # Convert legacy action format to v1.0
        v10_action = self._convert_legacy_action(action)
        
        # Use v1.0 environment
        obs, reward, terminated, truncated, info = self._v10_env.step(v10_action)
        
        # Convert back to legacy format if needed
        return self._convert_v10_observation(obs), reward, terminated or truncated, info
    
    def _convert_legacy_params(self, *args, **kwargs) -> dict:
        """Convert legacy constructor parameters to v1.0 configuration."""
        # Implementation details for parameter conversion
        pass
    
    def _convert_legacy_action(self, action):
        """Convert legacy action format to v1.0 action interface."""
        pass
    
    def _convert_v10_observation(self, obs):
        """Convert v1.0 observation to legacy format."""
        pass
```

#### Upgrade Strategies

**Incremental Migration Approach**

1. **Phase 1: Compatibility Layer**
   ```python
   # Immediate compatibility - no code changes required
   from plume_nav_sim.envs.plume_navigation_env import PlumeNavigationEnv
   
   env = PlumeNavigationEnv(config)  # Works with v1.0 via compatibility layer
   ```

2. **Phase 2: Configuration Migration**
   ```python
   # Update configuration to v1.0 format
   from plume_nav_sim.migration import migrate_config_v03_to_v10
   
   legacy_config = load_legacy_config("experiment.yaml")
   v10_config = migrate_config_v03_to_v10(legacy_config)
   
   env = NavigatorFactory.create_modular_environment(**v10_config)
   ```

3. **Phase 3: Component-by-Component Migration**
   ```python
   # Gradually adopt new components
   env = NavigatorFactory.create_modular_environment(
       navigator_config=legacy_navigator_config,  # Keep legacy navigator
       source_config=v10_source_config,          # Use new source system
       boundary_policy_config=v10_boundary_config, # Use new boundary system
       # Add new components incrementally
   )
   ```

4. **Phase 4: Full V1.0 Adoption**
   ```python
   # Complete migration to v1.0 architecture
   @hydra.main(config_path="../conf", config_name="config")
   def main(cfg: DictConfig) -> None:
       env = NavigatorFactory.create_modular_environment(
           navigator_config=cfg.navigator,
           plume_model_config=cfg.plume_model,
           source_config=cfg.source,
           boundary_policy_config=cfg.boundary,
           action_interface_config=cfg.action,
           recorder_config=cfg.record,
           stats_aggregator_config=cfg.stats,
           agent_initializer_config=cfg.agent_init
       )
   ```

#### Breaking Changes

**API Changes Requiring Code Updates**

1. **Import Path Changes**
   ```python
   # v0.3.0
   from plume_nav_sim.controllers import SingleAgentController
   
   # v1.0
   from plume_nav_sim.core.controllers import SingleAgentController
   ```

2. **Configuration Structure Changes**
   ```yaml
   # v0.3.0 configuration
   position: [10.0, 20.0]
   max_speed: 2.0
   source_position: [50.0, 50.0]
   
   # v1.0 configuration
   navigator:
     position: [10.0, 20.0]
     max_speed: 2.0
   source:
     position: [50.0, 50.0]
   ```

3. **Method Signature Changes**
   ```python
   # v0.3.0
   env.reset(position=(10, 20))
   
   # v1.0
   env.reset()  # Position set via configuration or agent_initializer
   ```

#### Transition Guide

**Migration Checklist**

- [ ] **Audit Current Code**: Identify all v0.3.0 dependencies and usage patterns
- [ ] **Install v1.0**: Update to plume_nav_sim v1.0 with compatibility features enabled
- [ ] **Test Compatibility**: Ensure existing code works with compatibility layer
- [ ] **Migrate Configuration**: Convert configuration files to v1.0 modular format
- [ ] **Update Imports**: Update import statements to new module structure
- [ ] **Adopt New Features**: Gradually adopt new v1.0 components and capabilities
- [ ] **Update Tests**: Modify test suites to work with new architecture
- [ ] **Performance Validation**: Verify performance requirements are still met
- [ ] **Documentation Update**: Update internal documentation and examples
- [ ] **Remove Legacy Code**: Clean up legacy compatibility code once migration complete

**Migration Tools**
```python
# Automated migration script
def migrate_project_v03_to_v10(project_path: str, backup: bool = True) -> None:
    """Automated migration tool for v0.3.0 to v1.0."""
    
    if backup:
        create_backup(project_path)
    
    # Update import statements
    update_imports(project_path)
    
    # Convert configuration files
    convert_configs(project_path)
    
    # Update test files
    update_tests(project_path)
    
    # Generate migration report
    generate_migration_report(project_path)
    
    print("Migration completed. Review migration_report.md for details.")

# Usage
migrate_project_v03_to_v10("./my_plume_navigation_project")
```

This comprehensive API reference provides the foundation for understanding and implementing the protocol-based architecture in plume_nav_sim v1.0. The protocol system enables true modularity, extensibility, and maintainability while preserving backward compatibility and providing clear migration paths for existing users.