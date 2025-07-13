# Boundary Policy Framework API Reference

## Overview

The Boundary Policy Framework (Feature F-015) provides pluggable boundary handling strategies for domain edge management in plume navigation simulations. This protocol-based architecture enables runtime selection of boundary behaviors without code changes, supporting diverse experimental setups and research requirements.

### Core Architecture

The framework implements the **BoundaryPolicyProtocol** interface to define standardized boundary handling across four primary policy types:

- **TerminateBoundary**: Episodes end when agents reach domain boundaries (status = "oob")
- **BounceBoundary**: Agents reflect off boundaries with configurable energy conservation  
- **WrapBoundary**: Periodic boundary conditions with seamless position wrapping
- **ClipBoundary**: Hard position constraints preventing boundary crossing

### Key Features

- **Protocol-Based Design**: Pluggable boundary behaviors via `BoundaryPolicyProtocol` interface
- **Vectorized Operations**: Efficient multi-agent support for 100+ agents with <1ms performance
- **Configuration-Driven Selection**: Zero-code boundary policy switching via Hydra configuration
- **Performance Optimization**: Meets ≤33ms step latency requirements with vectorized NumPy operations
- **Multi-Agent Scalability**: Optimized for 100-agent scenarios with vectorized boundary checking

### Integration Points

- **Environment Integration**: `PlumeNavigationEnv` uses boundary policies for episode management
- **Controller Integration**: `SingleAgentController` and `MultiAgentController` delegate boundary handling
- **Configuration System**: Hydra config group `conf/base/boundary/` enables runtime policy selection
- **Statistics Integration**: Boundary violations tracked for research metrics and analysis

---

## BoundaryPolicyProtocol Interface

The `BoundaryPolicyProtocol` defines the standardized interface that all boundary policy implementations must satisfy. This protocol ensures consistent behavior across different boundary handling strategies while enabling performance optimization through vectorized operations.

### Protocol Definition

```python
from typing import Protocol, Union, Optional, Tuple
import numpy as np

@runtime_checkable
class BoundaryPolicyProtocol(Protocol):
    """
    Protocol defining configurable boundary handling strategies for domain edge management.
    
    Performance Requirements:
    - apply_policy(): <1ms for 100 agents with vectorized operations
    - check_violations(): <0.5ms for boundary detection across all agents
    - get_termination_status(): <0.1ms for episode termination decisions
    - Memory efficiency: <1MB for boundary state management
    """
```

### Core Methods

#### apply_policy()

Applies boundary policy to agent positions and optionally velocities with vectorized operations for multi-agent efficiency.

```python
def apply_policy(
    self, 
    positions: np.ndarray, 
    velocities: Optional[np.ndarray] = None
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Apply boundary policy to agent positions and optionally velocities.
    
    Args:
        positions: Agent positions as array with shape (n_agents, 2) for multiple
            agents or (2,) for single agent. Coordinates in environment units.
        velocities: Optional agent velocities with same shape as positions.
            Required for physics-based policies like bounce behavior.
            
    Returns:
        Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]: 
            - If velocities not provided: corrected positions array
            - If velocities provided: tuple of (corrected_positions, corrected_velocities)
            
    Performance:
        Must execute in <1ms for 100 agents with vectorized operations.
    """
```

**Policy-Specific Behavior:**
- **Terminate**: Positions returned unchanged (termination handled separately)
- **Bounce**: Positions and velocities corrected for elastic/inelastic collisions
- **Wrap**: Positions wrapped to opposite boundary with velocity preservation
- **Clip**: Positions constrained to domain bounds with optional velocity damping

#### check_violations()

Detects boundary violations using vectorized operations for sub-millisecond performance with large agent populations.

```python
def check_violations(self, positions: np.ndarray) -> np.ndarray:
    """
    Detect boundary violations for given agent positions.
    
    Args:
        positions: Agent positions as array with shape (n_agents, 2) for multiple
            agents or (2,) for single agent. Coordinates in environment units.
            
    Returns:
        np.ndarray: Boolean array with shape (n_agents,) or scalar bool for single
            agent. True indicates boundary violation requiring policy application.
            
    Performance:
        Must execute in <0.5ms for boundary detection across 100 agents.
    """
```

**Violation Detection Criteria:**
- Position outside domain bounds (all policies)
- Velocity pointing outward at boundary (bounce policy)
- Distance from boundary below threshold (predictive policies)

#### get_termination_status()

Returns episode termination status for boundary policy with semantic information for episode management.

```python
def get_termination_status(self) -> str:
    """
    Get episode termination status for boundary policy.
    
    Returns:
        str: Termination status string indicating boundary policy behavior.
            Common values:
            - "oob": Out of bounds termination (TerminatePolicy)
            - "continue": Episode continues with correction (BouncePolicy, WrapPolicy, ClipPolicy)
            - "boundary_contact": Boundary interaction without termination
            
    Performance:
        Must execute in <0.1ms for immediate episode management decisions.
    """
```

### Method Integration Patterns

```python
# Standard boundary policy application pattern
violations = boundary_policy.check_violations(agent_positions)
if np.any(violations):
    corrected_positions = boundary_policy.apply_policy(agent_positions)
    episode_done = (boundary_policy.get_termination_status() == "oob")

# Physics-based boundary handling with velocities
if boundary_policy.requires_velocities():
    corrected_pos, corrected_vel = boundary_policy.apply_policy(
        agent_positions, agent_velocities
    )
else:
    corrected_pos = boundary_policy.apply_policy(agent_positions)
```

---

## Boundary Policy Implementations

### TerminateBoundary

Episodes terminate when agents violate domain boundaries, implementing traditional "out of bounds" behavior for controlled navigation experiments.

#### Class Definition

```python
class TerminateBoundary:
    """
    Boundary policy that terminates episodes when agents reach domain boundaries.
    
    Key Features:
    - Zero-cost boundary violation handling with no position modification
    - Vectorized violation detection for multi-agent scenarios
    - Configurable domain bounds with optional coordinate restrictions
    - Integration with episode management via termination status reporting
    
    Performance Characteristics:
    - check_violations(): O(n) vectorized operation, <0.1ms for 100 agents
    - apply_policy(): O(1) no-op operation, <0.01ms regardless of agent count
    - Memory usage: <1KB for policy state management
    """
    
    def __init__(
        self,
        domain_bounds: Tuple[float, float],
        allow_negative_coords: bool = False,
        status_on_violation: str = "oob"
    ):
        """
        Initialize termination boundary policy with domain constraints.
        
        Args:
            domain_bounds: Domain size as (width, height) defining valid region
            allow_negative_coords: Whether negative coordinates are allowed
            status_on_violation: Termination status string for episode management
        """
```

#### Usage Examples

```python
# Basic episode termination boundary
policy = TerminateBoundary(domain_bounds=(100, 100))
violations = policy.check_violations(agent_positions)
if violations.any():
    episode_done = (policy.get_termination_status() == "oob")

# Multi-agent boundary violation checking
positions = np.array([[50, 50], [105, 75], [25, 110]])  # One out of bounds
violations = policy.check_violations(positions)
# Returns [False, True, True] for domain bounds (100, 100)

# Custom termination status
policy = TerminateBoundary(
    domain_bounds=(100, 100),
    status_on_violation="boundary_exit"
)
```

### BounceBoundary

Implements elastic collision behavior at domain edges with configurable energy conservation and realistic physics modeling.

#### Class Definition

```python
class BounceBoundary:
    """
    Boundary policy implementing elastic collision behavior at domain edges.
    
    Key Features:
    - Realistic collision physics with energy conservation modeling
    - Configurable elasticity coefficient for material property simulation
    - Vectorized reflection calculations for multi-agent efficiency
    - Corner collision handling with proper momentum conservation
    
    Performance Characteristics:
    - apply_policy(): O(n) vectorized operation, <0.5ms for 100 agents
    - check_violations(): O(n) vectorized operation, <0.1ms for 100 agents
    - Memory usage: <1KB for policy state and collision parameters
    """
    
    def __init__(
        self,
        domain_bounds: Tuple[float, float],
        elasticity: float = 1.0,
        energy_loss: float = 0.0,
        allow_negative_coords: bool = False
    ):
        """
        Initialize bounce boundary policy with collision physics parameters.
        
        Args:
            domain_bounds: Domain size as (width, height) defining valid region
            elasticity: Coefficient of restitution [0, 1] for collision energy conservation
            energy_loss: Additional energy dissipation factor [0, 1] for realistic modeling
            allow_negative_coords: Whether negative coordinates are allowed
        """
```

#### Usage Examples

```python
# Elastic boundary collisions with perfect energy conservation
policy = BounceBoundary(domain_bounds=(100, 100), elasticity=1.0)
corrected_pos, corrected_vel = policy.apply_policy(positions, velocities)

# Inelastic collisions with energy dissipation
policy = BounceBoundary(
    domain_bounds=(100, 100), 
    elasticity=0.8,
    energy_loss=0.1
)
# 80% velocity reflection + 10% additional energy loss

# Dynamic elasticity adjustment during simulation
policy.set_elasticity(0.9)  # Update collision parameters
policy.configure(energy_loss=0.05)  # Reduce energy dissipation
```

#### Physics Model

The bounce policy implements realistic collision physics:

```python
# Velocity reflection calculation
new_velocity = -old_velocity * elasticity * (1 - energy_loss)

# Position correction for boundary penetration
corrected_position = boundary_edge + (boundary_edge - violating_position)

# Independent X/Y collision handling for corner cases
if x_violation:
    positions[idx, 0] = x_boundary - (pos[0] - x_boundary)
    velocities[idx, 0] = -vel[0] * elasticity * (1 - energy_loss)
```

### WrapBoundary

Implements periodic boundary conditions creating toroidal topology where agents exiting one domain edge appear on the opposite side.

#### Class Definition

```python
class WrapBoundary:
    """
    Boundary policy implementing periodic boundary conditions with position wrapping.
    
    Key Features:
    - Seamless position wrapping for toroidal domain topology
    - Velocity preservation during boundary transitions
    - Vectorized wrapping operations for multi-agent efficiency
    - Zero energy loss during wrapping transitions
    
    Performance Characteristics:
    - apply_policy(): O(n) vectorized operation, <0.2ms for 100 agents
    - check_violations(): O(n) vectorized operation, <0.1ms for 100 agents
    - Memory usage: <1KB for policy state and domain parameters
    """
    
    def __init__(
        self,
        domain_bounds: Tuple[float, float],
        allow_negative_coords: bool = False
    ):
        """
        Initialize wrap boundary policy with periodic domain parameters.
        
        Args:
            domain_bounds: Domain size as (width, height) defining wrapping region
            allow_negative_coords: Whether negative coordinates are allowed before wrapping
        """
```

#### Usage Examples

```python
# Periodic boundary conditions for continuous exploration
policy = WrapBoundary(domain_bounds=(100, 100))
wrapped_positions = policy.apply_policy(out_of_bounds_positions)

# Position wrapping with velocity preservation
policy = WrapBoundary(domain_bounds=(200, 150))
wrapped_pos, unchanged_vel = policy.apply_policy(positions, velocities)

# Toroidal navigation examples:
# Agent at (105, 50) wraps to (5, 50) for domain (100, 100)
# Agent at (-10, 75) wraps to (90, 75) for domain (100, 100)
```

#### Wrapping Algorithm

```python
# Efficient wrapping using modular arithmetic
wrapped_positions = positions.copy()

# Wrap x coordinates to [0, x_max)
wrapped_positions[:, 0] = np.mod(wrapped_positions[:, 0], domain_bounds[0])

# Wrap y coordinates to [0, y_max)  
wrapped_positions[:, 1] = np.mod(wrapped_positions[:, 1], domain_bounds[1])
```

### ClipBoundary

Enforces strict spatial constraints by clipping agent positions to remain within domain bounds with optional velocity damping.

#### Class Definition

```python
class ClipBoundary:
    """
    Boundary policy implementing hard position constraints to prevent boundary crossing.
    
    Key Features:
    - Hard position constraints preventing boundary crossing
    - Optional velocity damping at boundaries to reduce pressure effects
    - Vectorized clipping operations for multi-agent efficiency
    - Zero overshoot guarantee for critical spatial constraints
    
    Performance Characteristics:
    - apply_policy(): O(n) vectorized operation, <0.2ms for 100 agents
    - check_violations(): O(n) vectorized operation, <0.1ms for 100 agents
    - Memory usage: <1KB for policy state and clipping parameters
    """
    
    def __init__(
        self,
        domain_bounds: Tuple[float, float],
        velocity_damping: float = 1.0,
        damp_at_boundary: bool = False,
        allow_negative_coords: bool = False
    ):
        """
        Initialize clip boundary policy with constraint parameters.
        
        Args:
            domain_bounds: Domain size as (width, height) defining clipping region
            velocity_damping: Velocity scaling factor [0, 1] when at boundaries
            damp_at_boundary: Whether to apply velocity damping at boundary contact
            allow_negative_coords: Whether negative coordinates are allowed
        """
```

#### Usage Examples

```python
# Hard boundary constraints with position clipping
policy = ClipBoundary(domain_bounds=(100, 100))
clipped_positions = policy.apply_policy(agent_positions)

# Position clipping with velocity damping at boundaries
policy = ClipBoundary(
    domain_bounds=(100, 100),
    velocity_damping=0.8,
    damp_at_boundary=True
)
clipped_pos, damped_vel = policy.apply_policy(positions, velocities)

# Strict spatial confinement guarantees
policy = ClipBoundary(domain_bounds=(50, 50))
assert np.all(clipped_positions <= (50, 50))
assert np.all(clipped_positions >= (0, 0))
```

#### Clipping Algorithm

```python
# Vectorized position clipping to domain bounds
clipped_pos = positions.copy()

# Clip x coordinates to valid range
clipped_pos[:, 0] = np.maximum(clipped_pos[:, 0], x_min)
clipped_pos[:, 0] = np.minimum(clipped_pos[:, 0], x_max)

# Clip y coordinates to valid range
clipped_pos[:, 1] = np.maximum(clipped_pos[:, 1], y_min)
clipped_pos[:, 1] = np.minimum(clipped_pos[:, 1], y_max)
```

### Policy Comparison Table

| Policy | Termination | Position Change | Velocity Change | Use Case |
|--------|-------------|----------------|----------------|----------|
| **Terminate** | Yes ("oob") | None | None | Controlled exploration studies |
| **Bounce** | No ("continue") | Reflection | Reversed + damping | Realistic physics simulation |
| **Wrap** | No ("continue") | Periodic wrapping | Preserved | Infinite exploration space |
| **Clip** | No ("continue") | Hard constraints | Optional damping | Strict confinement requirements |

---

## Configuration Examples

The boundary policy framework integrates with Hydra's configuration system via the `conf/base/boundary/` config group, enabling runtime policy selection without code changes.

### TerminateBoundary Configuration

```yaml
# conf/base/boundary/terminate.yaml
_target_: plume_nav_sim.core.boundaries.TerminateBoundary

# Domain boundary configuration
domain_bounds: 
  - 100.0  # Domain width (x-axis extent)
  - 100.0  # Domain height (y-axis extent)

# Coordinate system configuration
allow_negative_coords: false

# Episode termination status
status_on_violation: "oob"

# Environment variable overrides
boundary_terminate_status: ${oc.env:BOUNDARY_TERMINATE_STATUS,${status_on_violation}}
boundary_domain_bounds: ${oc.env:BOUNDARY_DOMAIN_BOUNDS,${domain_bounds}}
```

#### Usage Examples

```yaml
# Basic termination boundary configuration
boundary:
  _target_: plume_nav_sim.core.boundaries.TerminateBoundary
  domain_bounds: [100, 100]
  status_on_violation: "oob"

# Large domain with custom termination status
boundary:
  _target_: plume_nav_sim.core.boundaries.TerminateBoundary
  domain_bounds: [500, 300]
  status_on_violation: "boundary_exit"
  allow_negative_coords: false
```

### BounceBoundary Configuration

```yaml
# conf/base/boundary/bounce.yaml
_target_: plume_nav_sim.core.boundaries.BounceBoundary

# Domain bounds with environment variable support
domain_bounds: 
  - ${oc.env:BOUNDARY_DOMAIN_WIDTH,100.0}
  - ${oc.env:BOUNDARY_DOMAIN_HEIGHT,100.0}

# Physics parameters
elasticity: ${oc.env:BOUNDARY_BOUNCE_RESTITUTION,0.8}
energy_loss: ${oc.env:BOUNDARY_BOUNCE_ENERGY_LOSS,0.05}

# Advanced collision parameters
collision_config:
  collision_threshold: ${oc.env:BOUNDARY_COLLISION_THRESHOLD,1e-6}
  max_reflection_angle: ${oc.env:BOUNDARY_MAX_REFLECTION_ANGLE,3.14159}
  velocity_damping: ${oc.env:BOUNDARY_VELOCITY_DAMPING,1.0}

# Performance optimization
performance:
  use_vectorized_operations: ${oc.env:BOUNDARY_USE_VECTORIZED,true}
  optimization_level: ${oc.env:BOUNDARY_OPTIMIZATION_LEVEL,2}
```

#### Physics Configuration Examples

```yaml
# Perfect elastic collisions (no energy loss)
boundary:
  _target_: plume_nav_sim.core.boundaries.BounceBoundary
  domain_bounds: [100, 100]
  elasticity: 1.0
  energy_loss: 0.0

# Realistic material behavior (some energy dissipation)
boundary:
  _target_: plume_nav_sim.core.boundaries.BounceBoundary
  domain_bounds: [100, 100]
  elasticity: 0.8
  energy_loss: 0.1

# High damping environment (significant energy loss)
boundary:
  _target_: plume_nav_sim.core.boundaries.BounceBoundary
  domain_bounds: [100, 100]
  elasticity: 0.6
  energy_loss: 0.3
```

### WrapBoundary Configuration

```yaml
# conf/base/boundary/wrap.yaml
_target_: plume_nav_sim.core.boundaries.WrapBoundary

# Core parameters
domain_bounds: [100.0, 100.0]
allow_negative_coords: false

# Runtime configuration overrides
boundary_wrap_enabled: ${env:BOUNDARY_WRAP_ENABLED, true}
domain_width: ${env:BOUNDARY_DOMAIN_WIDTH, ${.domain_bounds[0]}}
domain_height: ${env:BOUNDARY_DOMAIN_HEIGHT, ${.domain_bounds[1]}}

# Axis-specific wrapping controls
wrap_x_axis: ${env:BOUNDARY_WRAP_X, true}
wrap_y_axis: ${env:BOUNDARY_WRAP_Y, true}

# Performance optimization
batch_size: 100
preallocate_arrays: true
enable_early_exit: true
```

#### Advanced Wrapping Examples

```yaml
# Toroidal domain with full wrapping
boundary:
  _target_: plume_nav_sim.core.boundaries.WrapBoundary
  domain_bounds: [200, 150]
  wrap_x_axis: true
  wrap_y_axis: true

# Mixed boundary conditions (wrap X, terminate Y)
boundary:
  _target_: plume_nav_sim.core.boundaries.WrapBoundary
  domain_bounds: [100, 100]
  wrap_x_axis: true
  wrap_y_axis: false
```

### ClipBoundary Configuration

```yaml
# conf/base/boundary/clip.yaml
_target_: plume_nav_sim.core.boundaries.ClipBoundary

# Core clipping behavior
clip_mode: "hard"
preserve_velocity_direction: true

# Boundary parameters
boundary_params:
  x_min: ${oc.env:BOUNDARY_X_MIN, null}
  x_max: ${oc.env:BOUNDARY_X_MAX, null}
  y_min: ${oc.env:BOUNDARY_Y_MIN, null}
  y_max: ${oc.env:BOUNDARY_Y_MAX, null}
  margin: ${oc.env:BOUNDARY_CLIP_MARGIN, 0.0}

# Velocity handling
velocity_handling:
  enabled: ${oc.env:BOUNDARY_CLIP_PRESERVE_VELOCITY, true}
  damping_factor: ${oc.env:BOUNDARY_CLIP_DAMPING, 1.0}
  zero_inward_velocity: true

# Performance settings
performance:
  vectorized: true
  batch_size: 100
  use_numpy_clip: true
```

### Hydra Composition Patterns

#### Runtime Policy Selection

```bash
# Command-line boundary policy override
python train.py boundary=bounce boundary.elasticity=0.9

# Environment variable configuration
export BOUNDARY_BOUNCE_RESTITUTION=0.95
export BOUNDARY_DOMAIN_WIDTH=200
python train.py boundary=bounce

# Configuration composition
python train.py \
  boundary=clip \
  boundary.velocity_handling.damping_factor=0.8 \
  boundary.boundary_params.margin=2.0
```

#### Multi-Environment Deployment

```yaml
# Development environment
defaults:
  - boundary: terminate
  - override boundary: bounce

# Production environment  
defaults:
  - boundary: wrap
  - boundary/performance: optimized

# Testing environment
defaults:
  - boundary: clip
  - boundary/validation: strict
```

### Environment Variable Reference

#### Core Parameters

```bash
# Domain configuration
BOUNDARY_DOMAIN_WIDTH=200          # Domain width override
BOUNDARY_DOMAIN_HEIGHT=150         # Domain height override
BOUNDARY_ALLOW_NEGATIVE_COORDS=true # Coordinate system setting

# BounceBoundary physics
BOUNDARY_BOUNCE_RESTITUTION=0.9    # Elasticity coefficient [0.0-1.0]
BOUNDARY_BOUNCE_ENERGY_LOSS=0.02   # Energy dissipation [0.0-1.0]

# WrapBoundary controls
BOUNDARY_WRAP_ENABLED=true         # Enable/disable wrapping
BOUNDARY_WRAP_X=true               # X-axis wrapping
BOUNDARY_WRAP_Y=false              # Y-axis wrapping only

# ClipBoundary settings
BOUNDARY_CLIP_MARGIN=1.0           # Clipping margin
BOUNDARY_CLIP_DAMPING=0.8          # Velocity damping factor
```

#### Performance Tuning

```bash
# Optimization settings
BOUNDARY_USE_VECTORIZED=true       # Enable vectorized operations
BOUNDARY_OPTIMIZATION_LEVEL=3      # Optimization level [1-3]
BOUNDARY_BATCH_SIZE=100            # Processing batch size

# Debug and monitoring
BOUNDARY_LOG_COLLISIONS=true       # Enable collision logging
BOUNDARY_TRACK_STATS=true          # Track statistics
BOUNDARY_DEBUG=true                # Verbose debug output
```

---

## Performance and Integration

### Vectorized Multi-Agent Operations

The boundary policy framework is optimized for multi-agent scenarios with vectorized NumPy operations that achieve sub-millisecond performance for 100+ agents.

#### Performance Benchmarks

| Operation | Single Agent | 10 Agents | 100 Agents | Performance Target |
|-----------|--------------|-----------|------------|-------------------|
| `check_violations()` | <0.01ms | <0.05ms | <0.1ms | <0.5ms |
| `apply_policy()` | <0.01ms | <0.1ms | <0.5ms | <1.0ms |
| `get_termination_status()` | <0.001ms | <0.001ms | <0.001ms | <0.1ms |

#### Vectorized Implementation Example

```python
def check_violations(self, positions: np.ndarray) -> np.ndarray:
    """Vectorized boundary violation detection for multi-agent scenarios."""
    # Handle single agent case
    single_agent = positions.ndim == 1
    if single_agent:
        positions = positions.reshape(1, -1)
    
    # Vectorized boundary checking for all agents simultaneously
    x_coords = positions[:, 0]
    y_coords = positions[:, 1]
    
    # Efficient logical operations across all agents
    x_violations = np.logical_or(x_coords < self.x_min, x_coords > self.x_max)
    y_violations = np.logical_or(y_coords < self.y_min, y_coords > self.y_max)
    violations = np.logical_or(x_violations, y_violations)
    
    return violations[0] if single_agent else violations
```

### Controller Integration

Boundary policies integrate seamlessly with navigation controllers through the protocol-based dependency injection pattern.

#### SingleAgentController Integration

```python
class SingleAgentController:
    def __init__(self, boundary_policy: BoundaryPolicyProtocol, **kwargs):
        self.boundary_policy = boundary_policy
        
    def step(self, action: Dict[str, Any]) -> None:
        """Navigation step with boundary policy integration."""
        # Update agent position based on action
        new_position = self._compute_new_position(action)
        
        # Apply boundary policy
        violations = self.boundary_policy.check_violations(new_position)
        if violations.any():
            corrected_position = self.boundary_policy.apply_policy(new_position)
            self.position = corrected_position
            
            # Handle episode termination if needed
            status = self.boundary_policy.get_termination_status()
            if status == "oob":
                self.episode_done = True
        else:
            self.position = new_position
```

#### MultiAgentController Integration

```python
class MultiAgentController:
    def __init__(self, boundary_policy: BoundaryPolicyProtocol, **kwargs):
        self.boundary_policy = boundary_policy
        
    def step(self, actions: np.ndarray) -> None:
        """Vectorized multi-agent step with boundary policy."""
        # Update all agent positions simultaneously
        new_positions = self._compute_new_positions(actions)
        
        # Vectorized boundary violation checking
        violations = self.boundary_policy.check_violations(new_positions)
        
        if np.any(violations):
            # Apply policy to all agents (vectorized operation)
            corrected_positions = self.boundary_policy.apply_policy(new_positions)
            self.positions = corrected_positions
            
            # Handle termination for terminate policy
            status = self.boundary_policy.get_termination_status()
            if status == "oob":
                # Mark episodes as done for violating agents
                self.episode_done[violations] = True
        else:
            self.positions = new_positions
```

### Episode Termination Logic

Boundary policies integrate with episode management through standardized termination status reporting.

#### Episode Management Pattern

```python
def _handle_episode_termination(self, boundary_policy: BoundaryPolicyProtocol) -> bool:
    """Standard episode termination handling for boundary policies."""
    status = boundary_policy.get_termination_status()
    
    # Map boundary status to episode termination
    termination_mapping = {
        "oob": True,              # Out of bounds - terminate episode
        "boundary_exit": True,    # Custom termination status
        "continue": False,        # Episode continues with correction
        "boundary_contact": False # Boundary interaction without termination
    }
    
    return termination_mapping.get(status, False)
```

#### Info Dictionary Integration

```python
def _update_info_dict(self, info: Dict[str, Any], boundary_policy: BoundaryPolicyProtocol) -> None:
    """Add boundary policy information to episode info."""
    violations = boundary_policy.check_violations(self.positions)
    
    info.update({
        "boundary_violations": violations.sum() if hasattr(violations, 'sum') else int(violations),
        "boundary_status": boundary_policy.get_termination_status(),
        "boundary_policy_type": boundary_policy.__class__.__name__
    })
    
    # Add policy-specific metrics
    if hasattr(boundary_policy, 'get_collision_count'):
        info["collision_count"] = boundary_policy.get_collision_count()
    if hasattr(boundary_policy, 'get_wrap_count'):
        info["wrap_count"] = boundary_policy.get_wrap_count()
```

### Performance Optimization Guidelines

#### Memory Management

```python
# Pre-allocate arrays for boundary checking to avoid allocation overhead
class OptimizedBoundaryPolicy:
    def __init__(self, max_agents: int = 100):
        # Pre-allocate working arrays
        self._position_buffer = np.zeros((max_agents, 2))
        self._violation_buffer = np.zeros(max_agents, dtype=bool)
        self._correction_buffer = np.zeros((max_agents, 2))
        
    def check_violations(self, positions: np.ndarray) -> np.ndarray:
        """Memory-efficient violation checking with pre-allocated buffers."""
        n_agents = positions.shape[0]
        
        # Use pre-allocated buffer to avoid memory allocation
        working_positions = self._position_buffer[:n_agents]
        working_positions[:] = positions
        
        # Vectorized operations on pre-allocated arrays
        violations = self._violation_buffer[:n_agents]
        # ... boundary checking logic ...
        
        return violations
```

#### Computational Optimization

```python
# Early exit optimization for performance-critical scenarios
def optimized_apply_policy(self, positions: np.ndarray) -> np.ndarray:
    """Apply boundary policy with early exit optimization."""
    # Quick check if any violations exist
    violations = self.check_violations(positions)
    
    # Early exit if no violations (common case)
    if not np.any(violations):
        return positions
    
    # Only process violating agents for efficiency
    violating_indices = np.where(violations)[0]
    violating_positions = positions[violating_indices]
    
    # Apply corrections only to violating agents
    corrected_positions = positions.copy()
    corrected_positions[violating_indices] = self._apply_corrections(violating_positions)
    
    return corrected_positions
```

### Scalability Guidelines

#### 100-Agent Performance Requirements

```python
# Configuration for optimal 100-agent performance
BOUNDARY_CONFIG = {
    "vectorized_operations": True,
    "batch_size": 100,
    "optimization_level": 3,
    "preallocate_arrays": True,
    "enable_early_exit": True,
    "use_numpy_optimized": True
}

# Performance monitoring for 100-agent scenarios
import time

def benchmark_boundary_performance(boundary_policy, n_agents=100, n_iterations=1000):
    """Benchmark boundary policy performance for large agent populations."""
    positions = np.random.rand(n_agents, 2) * 100
    
    # Benchmark violation checking
    start_time = time.time()
    for _ in range(n_iterations):
        violations = boundary_policy.check_violations(positions)
    check_time = (time.time() - start_time) / n_iterations
    
    # Benchmark policy application
    start_time = time.time()
    for _ in range(n_iterations):
        corrected = boundary_policy.apply_policy(positions)
    apply_time = (time.time() - start_time) / n_iterations
    
    print(f"Performance for {n_agents} agents:")
    print(f"  check_violations(): {check_time*1000:.3f}ms (target: <0.5ms)")
    print(f"  apply_policy(): {apply_time*1000:.3f}ms (target: <1.0ms)")
    
    # Verify performance targets
    assert check_time < 0.0005, f"check_violations() too slow: {check_time*1000:.3f}ms"
    assert apply_time < 0.001, f"apply_policy() too slow: {apply_time*1000:.3f}ms"
```

---

## Advanced Usage

### Custom Boundary Policy Implementation

Researchers can implement custom boundary policies by adhering to the `BoundaryPolicyProtocol` interface while leveraging the framework's performance optimizations.

#### Custom Policy Template

```python
from typing import Union, Optional, Tuple
import numpy as np
from plume_nav_sim.core.protocols import BoundaryPolicyProtocol

class CustomBoundaryPolicy:
    """
    Template for implementing custom boundary policies.
    
    This template provides the required interface implementation with
    performance optimization guidelines and integration patterns.
    """
    
    def __init__(
        self,
        domain_bounds: Tuple[float, float],
        custom_param: float = 1.0,
        **kwargs
    ):
        """Initialize custom boundary policy with domain and custom parameters."""
        self.domain_bounds = domain_bounds
        self.custom_param = custom_param
        
        # Cache boundary limits for efficient operations
        self.x_min, self.y_min = 0.0, 0.0
        self.x_max, self.y_max = domain_bounds
        
        # Pre-allocate arrays for performance optimization
        self._max_agents = kwargs.get('max_agents', 100)
        self._position_buffer = np.zeros((self._max_agents, 2))
        self._violation_buffer = np.zeros(self._max_agents, dtype=bool)
    
    def apply_policy(
        self, 
        positions: np.ndarray, 
        velocities: Optional[np.ndarray] = None
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Implement custom boundary behavior.
        
        Performance Requirements:
        - Must execute in <1ms for 100 agents
        - Use vectorized NumPy operations where possible
        - Minimize memory allocations
        """
        # Handle single agent case
        single_agent = positions.ndim == 1
        if single_agent:
            positions = positions.reshape(1, -1)
            if velocities is not None:
                velocities = velocities.reshape(1, -1)
        
        # Implement custom boundary logic here
        corrected_positions = self._apply_custom_boundary_logic(positions)
        corrected_velocities = self._apply_custom_velocity_logic(velocities) if velocities is not None else None
        
        # Return in original format
        if single_agent:
            if corrected_velocities is not None:
                return corrected_positions[0], corrected_velocities[0]
            return corrected_positions[0]
        
        if corrected_velocities is not None:
            return corrected_positions, corrected_velocities
        return corrected_positions
    
    def check_violations(self, positions: np.ndarray) -> np.ndarray:
        """
        Detect custom boundary violations.
        
        Performance Requirements:
        - Must execute in <0.5ms for 100 agents
        - Use vectorized operations for multi-agent scenarios
        """
        # Handle single agent case
        single_agent = positions.ndim == 1
        if single_agent:
            positions = positions.reshape(1, -1)
        
        # Implement custom violation detection logic
        violations = self._detect_custom_violations(positions)
        
        return violations[0] if single_agent else violations
    
    def get_termination_status(self) -> str:
        """Return custom termination status."""
        return "custom_behavior"  # or "continue", "oob", etc.
    
    def _apply_custom_boundary_logic(self, positions: np.ndarray) -> np.ndarray:
        """Implement custom position correction logic."""
        # Example: magnetic boundary that attracts agents back
        corrected = positions.copy()
        
        # Vectorized custom logic implementation
        violations = self._detect_custom_violations(positions)
        if np.any(violations):
            violating_indices = np.where(violations)[0]
            # Apply custom correction to violating agents
            for idx in violating_indices:
                # Custom boundary behavior implementation
                corrected[idx] = self._apply_magnetic_attraction(positions[idx])
        
        return corrected
    
    def _detect_custom_violations(self, positions: np.ndarray) -> np.ndarray:
        """Implement custom violation detection."""
        # Example: violations based on distance from center
        center = np.array([self.x_max/2, self.y_max/2])
        distances = np.linalg.norm(positions - center, axis=1)
        max_radius = min(self.x_max, self.y_max) / 2 * self.custom_param
        
        return distances > max_radius
```

#### Advanced Custom Policy Example

```python
class GradientBoundaryPolicy:
    """
    Advanced boundary policy with gradient-based repulsion.
    
    This policy implements smooth repulsion forces near boundaries
    rather than hard constraints, providing realistic force fields.
    """
    
    def __init__(
        self,
        domain_bounds: Tuple[float, float],
        repulsion_strength: float = 10.0,
        boundary_width: float = 5.0
    ):
        self.domain_bounds = domain_bounds
        self.repulsion_strength = repulsion_strength
        self.boundary_width = boundary_width
        
        # Pre-compute boundary regions for efficiency
        self.x_max, self.y_max = domain_bounds
        self.inner_bounds = (
            self.x_max - boundary_width,
            self.y_max - boundary_width
        )
    
    def apply_policy(self, positions: np.ndarray, velocities: Optional[np.ndarray] = None):
        """Apply gradient-based repulsion forces near boundaries."""
        # Calculate repulsion forces
        forces = self._calculate_repulsion_forces(positions)
        
        # Apply forces as velocity modifications
        if velocities is not None:
            corrected_velocities = velocities + forces * 0.1  # Force scaling
            return positions, corrected_velocities
        else:
            # Apply forces as position corrections
            corrected_positions = positions + forces * 0.01  # Position scaling
            return corrected_positions
    
    def _calculate_repulsion_forces(self, positions: np.ndarray) -> np.ndarray:
        """Calculate vectorized repulsion forces for all agents."""
        forces = np.zeros_like(positions)
        
        # X-axis repulsion
        near_right = positions[:, 0] > self.inner_bounds[0]
        forces[near_right, 0] = -self.repulsion_strength * (
            positions[near_right, 0] - self.inner_bounds[0]
        ) / self.boundary_width
        
        near_left = positions[:, 0] < self.boundary_width
        forces[near_left, 0] = self.repulsion_strength * (
            self.boundary_width - positions[near_left, 0]
        ) / self.boundary_width
        
        # Y-axis repulsion (similar logic)
        near_top = positions[:, 1] > self.inner_bounds[1]
        forces[near_top, 1] = -self.repulsion_strength * (
            positions[near_top, 1] - self.inner_bounds[1]
        ) / self.boundary_width
        
        near_bottom = positions[:, 1] < self.boundary_width
        forces[near_bottom, 1] = self.repulsion_strength * (
            self.boundary_width - positions[near_bottom, 1]
        ) / self.boundary_width
        
        return forces
```

### Factory Function Usage

The `create_boundary_policy()` factory function enables dynamic policy instantiation from configuration without explicit class imports.

#### Factory Function Interface

```python
def create_boundary_policy(
    policy_type: str,
    domain_bounds: Tuple[float, float],
    **kwargs: Any
) -> BoundaryPolicyProtocol:
    """
    Factory function for creating boundary policy instances with runtime selection.
    
    Supported Policy Types:
    - "terminate": TerminateBoundary for episode termination on boundary violation
    - "bounce": BounceBoundary for elastic collision behavior at boundaries
    - "wrap": WrapBoundary for periodic boundary conditions with position wrapping
    - "clip": ClipBoundary for hard position constraints preventing boundary crossing
    
    Args:
        policy_type: Boundary policy type identifier string
        domain_bounds: Domain size as (width, height) defining valid region
        **kwargs: Policy-specific configuration parameters passed to constructor
        
    Returns:
        BoundaryPolicyProtocol: Configured boundary policy instance
    """
```

#### Factory Usage Examples

```python
# Termination boundary with custom status
policy = create_boundary_policy(
    "terminate", 
    domain_bounds=(100, 100),
    status_on_violation="boundary_exit"
)

# Elastic bounce boundary with energy loss
policy = create_boundary_policy(
    "bounce",
    domain_bounds=(200, 150), 
    elasticity=0.8,
    energy_loss=0.1
)

# Periodic wrapping boundary
policy = create_boundary_policy(
    "wrap",
    domain_bounds=(100, 100)
)

# Hard clipping boundary with velocity damping
policy = create_boundary_policy(
    "clip",
    domain_bounds=(50, 50),
    velocity_damping=0.7,
    damp_at_boundary=True
)

# Configuration-driven instantiation
config = {
    'policy_type': 'bounce',
    'domain_bounds': (100, 100),
    'elasticity': 0.9
}
policy = create_boundary_policy(**config)
```

### Runtime Policy Switching

Advanced scenarios may require dynamic boundary policy changes during simulation execution.

#### Dynamic Policy Management

```python
class DynamicBoundaryManager:
    """
    Manages runtime boundary policy switching for adaptive simulations.
    
    Enables research scenarios where boundary behavior changes based on
    environmental conditions or experimental phases.
    """
    
    def __init__(self, initial_policy: BoundaryPolicyProtocol):
        self.current_policy = initial_policy
        self.policy_history = [initial_policy]
        self.switch_count = 0
    
    def switch_policy(
        self, 
        new_policy_config: Dict[str, Any],
        transition_steps: int = 0
    ) -> None:
        """
        Switch to new boundary policy with optional transition period.
        
        Args:
            new_policy_config: Configuration for new boundary policy
            transition_steps: Number of steps to blend between policies
        """
        new_policy = create_boundary_policy(**new_policy_config)
        
        if transition_steps > 0:
            # Implement gradual transition
            self._apply_gradual_transition(new_policy, transition_steps)
        else:
            # Immediate switch
            self.current_policy = new_policy
        
        self.policy_history.append(new_policy)
        self.switch_count += 1
    
    def apply_policy(self, positions: np.ndarray, **kwargs):
        """Delegate to current active policy."""
        return self.current_policy.apply_policy(positions, **kwargs)
    
    def check_violations(self, positions: np.ndarray) -> np.ndarray:
        """Delegate to current active policy."""
        return self.current_policy.check_violations(positions)
    
    def get_termination_status(self) -> str:
        """Delegate to current active policy."""
        return self.current_policy.get_termination_status()
```

#### Adaptive Boundary Scenarios

```python
# Example: Switch from wrap to terminate based on episode progress
class ProgressiveBoundaryManager:
    """Boundary policy that becomes more restrictive over time."""
    
    def __init__(self, domain_bounds: Tuple[float, float]):
        self.domain_bounds = domain_bounds
        self.episode_step = 0
        
        # Start with permissive wrapping policy
        self.early_policy = create_boundary_policy("wrap", domain_bounds)
        
        # Switch to restrictive termination policy later
        self.late_policy = create_boundary_policy("terminate", domain_bounds)
        
        self.switch_threshold = 500  # Switch after 500 steps
    
    def step(self) -> None:
        """Update episode step counter."""
        self.episode_step += 1
    
    def get_active_policy(self) -> BoundaryPolicyProtocol:
        """Return appropriate policy based on episode progress."""
        if self.episode_step < self.switch_threshold:
            return self.early_policy
        else:
            return self.late_policy
    
    def apply_policy(self, positions: np.ndarray, **kwargs):
        """Apply appropriate policy based on episode progress."""
        active_policy = self.get_active_policy()
        return active_policy.apply_policy(positions, **kwargs)

# Usage in environment
boundary_manager = ProgressiveBoundaryManager(domain_bounds=(100, 100))

# In environment step loop
boundary_manager.step()
corrected_positions = boundary_manager.apply_policy(agent_positions)
```

### Research Applications

#### Boundary Policy Comparison Studies

```python
def compare_boundary_policies(
    policy_configs: List[Dict[str, Any]],
    test_scenarios: List[np.ndarray],
    metrics: List[str] = ["termination_rate", "position_deviation", "velocity_change"]
) -> Dict[str, Dict[str, float]]:
    """
    Compare different boundary policies across test scenarios.
    
    Enables systematic evaluation of boundary policy effects on
    navigation performance and agent behavior patterns.
    """
    results = {}
    
    for config in policy_configs:
        policy = create_boundary_policy(**config)
        policy_name = f"{config['policy_type']}_{config.get('elasticity', 'default')}"
        
        policy_results = {}
        
        for scenario_name, positions in test_scenarios:
            # Evaluate policy on test scenario
            violations = policy.check_violations(positions)
            
            if np.any(violations):
                corrected = policy.apply_policy(positions)
                
                # Calculate metrics
                policy_results[f"{scenario_name}_violation_rate"] = violations.mean()
                policy_results[f"{scenario_name}_position_change"] = np.linalg.norm(
                    corrected - positions, axis=1
                ).mean()
                policy_results[f"{scenario_name}_termination"] = (
                    policy.get_termination_status() == "oob"
                )
        
        results[policy_name] = policy_results
    
    return results

# Example usage for research
test_scenarios = [
    ("edge_agents", np.array([[101, 50], [50, 101], [99, 99]])),
    ("corner_violation", np.array([[105, 105], [102, 98]])),
    ("bulk_violation", np.random.rand(100, 2) * 120)  # Some out of bounds
]

policy_configs = [
    {"policy_type": "terminate", "domain_bounds": (100, 100)},
    {"policy_type": "bounce", "domain_bounds": (100, 100), "elasticity": 0.8},
    {"policy_type": "bounce", "domain_bounds": (100, 100), "elasticity": 1.0},
    {"policy_type": "wrap", "domain_bounds": (100, 100)},
    {"policy_type": "clip", "domain_bounds": (100, 100)}
]

comparison_results = compare_boundary_policies(policy_configs, test_scenarios)
```

### Best Practices

#### Performance Optimization

1. **Vectorization**: Always use vectorized NumPy operations for multi-agent scenarios
2. **Memory Management**: Pre-allocate arrays when processing large agent populations
3. **Early Exit**: Implement early exit optimization when no violations are detected
4. **Batch Processing**: Process agents in optimized batch sizes (typically 100)

#### Configuration Management

1. **Environment Variables**: Use environment variables for runtime parameter overrides
2. **Validation**: Implement parameter validation to prevent invalid configurations
3. **Documentation**: Document all configuration parameters with examples and constraints
4. **Migration Support**: Provide legacy compatibility during framework transitions

#### Research Integration

1. **Metrics Collection**: Implement boundary interaction metrics for research analysis
2. **Reproducibility**: Ensure deterministic behavior with proper random seeding
3. **Visualization**: Provide debugging visualizations for boundary policy behavior
4. **Extensibility**: Design custom policies following the established protocol patterns

### Troubleshooting Guide

#### Common Issues and Solutions

**Performance Issues with Large Agent Populations**
- Ensure vectorized operations are enabled in configuration
- Check memory pre-allocation settings for optimal performance
- Verify batch size configuration matches agent population
- Monitor for memory allocation overhead in hot paths

**Unexpected Boundary Behavior**
- Verify domain bounds configuration matches environment setup
- Check coordinate system settings (allow_negative_coords)
- Validate policy-specific parameters (elasticity, energy_loss, etc.)
- Enable debug logging to trace boundary policy decisions

**Configuration Override Problems**
- Verify environment variable naming conventions
- Check Hydra configuration composition order
- Validate parameter type conversions (string to float/bool)
- Review configuration schema validation rules

**Integration with Custom Environments**
- Ensure BoundaryPolicyProtocol compliance for custom policies
- Verify controller integration follows established patterns
- Check episode termination logic for proper status handling
- Validate vectorized operations with custom agent representations

---

## Summary

The Boundary Policy Framework provides a comprehensive, performance-optimized solution for domain edge management in plume navigation simulations. Key benefits include:

- **Flexibility**: Four boundary policy types supporting diverse research scenarios
- **Performance**: Vectorized operations achieving <1ms latency for 100 agents
- **Extensibility**: Protocol-based design enabling custom policy implementations
- **Configuration**: Runtime policy selection via Hydra configuration system
- **Integration**: Seamless controller integration with episode management

The framework supports the plume_nav_sim v1.0 architecture goal of creating a general-purpose, extensible simulation toolkit while maintaining the performance requirements for real-time simulation scenarios.