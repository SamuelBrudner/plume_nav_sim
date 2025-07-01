# Plume Navigation Simulator Migration Guide

**From Monolithic VideoPlume to Modular Architecture & Gymnasium 0.29.x**

This comprehensive guide provides step-by-step instructions for migrating from the legacy monolithic plume navigation simulator to the new modular, extensible architecture while also transitioning from OpenAI Gym 0.26 to Gymnasium 0.29.x API in plume_nav_sim library (v0.4.0).

## Table of Contents

### Core Architecture Migration
1. [Architecture Overview](#architecture-overview)
2. [Modular Components Migration](#modular-components-migration)
3. [Plume Model Migration](#plume-model-migration)
4. [Sensor System Migration](#sensor-system-migration)
5. [Agent Architecture Migration](#agent-architecture-migration)
6. [Configuration System Migration](#configuration-system-migration)

### Gymnasium API Migration
7. [Gymnasium Overview](#gymnasium-overview)
8. [Quick Start Migration](#quick-start-migration)
9. [API Differences](#api-differences)
10. [Step-by-Step Migration](#step-by-step-migration)
11. [Compatibility Shim Usage](#compatibility-shim-usage)
12. [Environment Registration](#environment-registration)
13. [Code Examples](#code-examples)
14. [Performance Considerations](#performance-considerations)

### Troubleshooting & Support
15. [Migration Troubleshooting](#migration-troubleshooting)
16. [Performance Optimization](#performance-optimization)
17. [Protocol Compliance Validation](#protocol-compliance-validation)
18. [Migration Timeline](#migration-timeline)

---

## Architecture Overview

### Major Architectural Transformation

The plume_nav_sim library v0.4.0 introduces a fundamental shift from a monolithic video-based simulator to a highly configurable, extensible modular architecture. This transformation enables researchers to:

- **Switch between plume models** without code changes (Gaussian, Turbulent, Video-based)
- **Use different sensing modalities** through pluggable sensor protocols
- **Support both memory-based and memory-less agents** in the same framework
- **Configure wind dynamics** with realistic environmental physics
- **Extend functionality** through protocol-based interfaces

### What Changed

| Component | Legacy (v0.3.x) | Modular (v0.4.0) |
|-----------|-----------------|-------------------|
| **Plume Generation** | VideoPlume only | PlumeModelProtocol (Gaussian, Turbulent, Video) |
| **Sensing** | Hard-coded odor sampling | SensorProtocol (Binary, Concentration, Gradient) |
| **Wind Effects** | Static or none | WindFieldProtocol (Constant, Turbulent, TimeVarying) |
| **Agent Design** | Fixed navigation logic | NavigatorProtocol with optional memory interfaces |
| **Configuration** | Monolithic YAML | Modular Hydra configs with dependency injection |
| **Extensibility** | Code modifications required | Protocol-based extension points |

### Migration Philosophy

The migration follows a **zero-breaking-changes approach**:
- **Existing code continues to work** with automatic detection and compatibility layers
- **New projects can adopt modular components** incrementally
- **Performance is maintained or improved** with sub-10ms step execution
- **Configuration-driven flexibility** replaces code modifications

### Key Benefits

✅ **Research Flexibility**: Switch between simple and complex plume physics via configuration  
✅ **Agent Agnosticism**: Support reactive and cognitive navigation strategies uniformly  
✅ **Performance Scalability**: Optimized execution for single agents to 100+ swarm scenarios  
✅ **Extension Points**: Add custom sensors, plume models, and behaviors without core changes  
✅ **Reproducible Science**: Deterministic simulation with seed management and config tracking  

---

## Modular Components Migration

### Core Architecture Components

The new modular architecture introduces several protocol-based components that replace monolithic implementations:

#### 1. PlumeModelProtocol
**Purpose**: Pluggable odor plume generation with different physics models

**Old Approach (VideoPlume only)**:
```python
# Fixed to video-based plume data
from plume_nav_sim.components import VideoPlume
plume = VideoPlume(video_path="plume_movie.mp4")
concentration = plume.get_concentration_at(position)
```

**New Approach (Multiple Models)**:
```python
# Configuration-driven model selection
from plume_nav_sim.core.protocols import NavigatorFactory

# Fast analytical model for prototyping
plume_config = {
    '_target_': 'plume_nav_sim.models.plume.GaussianPlumeModel',
    'source_position': (50, 50),
    'source_strength': 1000.0,
    'sigma_x': 5.0,
    'sigma_y': 3.0
}
plume_model = NavigatorFactory.create_plume_model(plume_config)

# Or realistic turbulent model for research
plume_config = {
    '_target_': 'plume_nav_sim.models.plume.TurbulentPlumeModel',
    'filament_count': 500,
    'turbulence_intensity': 0.3
}
plume_model = NavigatorFactory.create_plume_model(plume_config)

# Uniform interface across all models
concentration = plume_model.concentration_at(position)
plume_model.step(dt=1.0)  # Advance temporal dynamics
```

#### 2. SensorProtocol
**Purpose**: Configurable sensing modalities replacing hard-coded odor sampling

**Old Approach (Hard-coded sampling)**:
```python
# Fixed odor sampling in controller
class OldController:
    def sample_odor(self, env_array):
        # Hard-coded bilinear interpolation
        return self._interpolate_concentration(self.position, env_array)
    
    def step(self, env_array, dt):
        odor_level = self.sample_odor(env_array)
        # Navigation logic based on direct sampling
```

**New Approach (Sensor abstraction)**:
```python
# Pluggable sensor configurations
from plume_nav_sim.core.sensors import BinarySensor, ConcentrationSensor, GradientSensor

# Binary detection with configurable threshold
binary_sensor = BinarySensor(threshold=0.1, false_positive_rate=0.02)

# Quantitative measurement with noise modeling
conc_sensor = ConcentrationSensor(dynamic_range=(0, 1), resolution=0.001)

# Spatial gradient for directional navigation
gradient_sensor = GradientSensor(spatial_resolution=(0.5, 0.5))

# Sensor-based controller with protocol compliance
class ModernController:
    def __init__(self, sensors):
        self.sensors = sensors
    
    def step(self, plume_state, dt):
        # Sensor readings through protocol interface
        sensor_data = {}
        for sensor in self.sensors:
            if hasattr(sensor, 'detect'):
                sensor_data['detection'] = sensor.detect(plume_state, self.position)
            if hasattr(sensor, 'measure'):
                sensor_data['concentration'] = sensor.measure(plume_state, self.position)
            if hasattr(sensor, 'compute_gradient'):
                sensor_data['gradient'] = sensor.compute_gradient(plume_state, self.position)
        
        # Navigation logic based on sensor abstractions
        self._navigate_with_sensors(sensor_data)
```

#### 3. WindFieldProtocol
**Purpose**: Environmental dynamics for realistic plume transport

**Old Approach (No wind or basic effects)**:
```python
# Limited or no wind dynamics
plume = VideoPlume(video_path="plume_movie.mp4")
# Wind effects, if any, were baked into video data
```

**New Approach (Dynamic wind modeling)**:
```python
# Configurable wind field implementations
wind_configs = {
    'constant': {
        '_target_': 'plume_nav_sim.models.wind.ConstantWindField',
        'velocity': (2.0, 0.5)  # East-northeast wind
    },
    'turbulent': {
        '_target_': 'plume_nav_sim.models.wind.TurbulentWindField',
        'mean_velocity': (3.0, 1.0),
        'turbulence_intensity': 0.2
    }
}

wind_field = NavigatorFactory.create_wind_field(wind_configs['turbulent'])

# Integration with plume models
plume_model = GaussianPlumeModel(
    source_position=(50, 50),
    wind_field=wind_field,  # Wind affects plume transport
    enable_wind_field=True
)

# Temporal evolution with wind effects
for t in range(100):
    wind_field.step(dt=1.0)     # Update wind dynamics
    plume_model.step(dt=1.0)    # Update plume with wind transport
    agent_velocity = wind_field.velocity_at(agent_position)  # Agent feels wind
```

---

## Plume Model Migration

### Migrating from VideoPlume to Modular Models

#### Legacy VideoPlume Usage
```python
# Old monolithic approach
from plume_nav_sim.components import VideoPlume

env_config = {
    "video_path": "data/plume_movie.mp4",
    "frame_start": 0,
    "frame_end": 1000,
    # Fixed to video data only
}

plume = VideoPlume(**env_config)
env = PlumeNavigationEnv(plume=plume)
```

#### Modern Modular Approach

**Step 1: Choose Plume Model Type**

```python
# Fast analytical model for algorithm development
plume_config_gaussian = {
    '_target_': 'plume_nav_sim.models.plume.GaussianPlumeModel',
    'source_position': (50, 50),
    'source_strength': 1000.0,
    'sigma_x': 5.0,
    'sigma_y': 3.0,
    'wind_speed': 0.0,        # Simple constant wind
    'wind_direction': 0.0
}

# Realistic turbulent model for fidelity research
plume_config_turbulent = {
    '_target_': 'plume_nav_sim.models.plume.TurbulentPlumeModel',
    'source_position': (50, 50),
    'filament_count': 500,
    'turbulence_intensity': 0.3,
    'wind_field': {
        '_target_': 'plume_nav_sim.models.wind.TurbulentWindField',
        'mean_velocity': (2.0, 0.5),
        'turbulence_intensity': 0.2
    }
}

# Backward-compatible video model
plume_config_video = {
    '_target_': 'plume_nav_sim.models.plume.VideoPlumeAdapter',
    'video_path': 'data/plume_movie.mp4',
    'frame_start': 0,
    'frame_end': 1000
}
```

**Step 2: Use Hydra Configuration Files**

Create `conf/base/plume_models/gaussian.yaml`:
```yaml
# @package plume_model
_target_: plume_nav_sim.models.plume.GaussianPlumeModel
source_position: [50.0, 50.0]
source_strength: 1000.0
sigma_x: 5.0
sigma_y: 3.0
wind_speed: 0.0
wind_direction: 0.0
background_concentration: 0.0
max_concentration: 1.0
```

Create `conf/base/plume_models/turbulent.yaml`:
```yaml
# @package plume_model
_target_: plume_nav_sim.models.plume.TurbulentPlumeModel
source_position: [50.0, 50.0]
filament_count: 500
turbulence_intensity: 0.3
dissipation_rate: 0.1
wind_field:
  _target_: plume_nav_sim.models.wind.TurbulentWindField
  mean_velocity: [2.0, 0.5]
  turbulence_intensity: 0.2
```

**Step 3: Update Environment Creation**

```python
# Configuration-driven environment setup
import hydra
from omegaconf import DictConfig
from plume_nav_sim.envs import PlumeNavigationEnv

@hydra.main(config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    # Automatic model instantiation through Hydra
    env = PlumeNavigationEnv.from_config(cfg)
    # Environment automatically uses configured plume model
    
# Command-line model switching without code changes
# python run_experiment.py plume_model=gaussian
# python run_experiment.py plume_model=turbulent  
# python run_experiment.py plume_model=video
```

#### Configuration Migration Mapping

| Legacy VideoPlume Parameter | Modern Equivalent | Notes |
|------------------------------|-------------------|-------|
| `video_path` | `plume_config.video_path` (VideoPlumeAdapter) | Direct mapping for video models |
| `frame_start`, `frame_end` | `plume_config.frame_start`, `frame_end` | Video-specific parameters |
| N/A (fixed video physics) | `plume_config.source_position` | Now configurable source location |
| N/A (baked into video) | `plume_config.source_strength` | Adjustable emission rate |
| N/A (video-dependent) | `plume_config.sigma_x`, `sigma_y` | Configurable dispersion |
| N/A (static) | `wind_field` configuration | Dynamic wind effects |

---

## Sensor System Migration

### Migrating from Hard-Coded Odor Sampling to SensorProtocol

#### Legacy Hard-Coded Approach
```python
# Old controller with fixed sensing logic
class LegacyController:
    def __init__(self, position, max_speed):
        self.position = np.array(position)
        self.max_speed = max_speed
    
    def sample_odor(self, env_array):
        """Hard-coded bilinear interpolation sampling."""
        x, y = self.position
        # Fixed interpolation logic
        concentration = self._bilinear_interpolate(env_array, x, y)
        return concentration
    
    def sample_multiple_sensors(self, env_array, distance=5.0, angle=45.0):
        """Hard-coded multi-sensor sampling."""
        positions = self._compute_sensor_positions(distance, angle)
        readings = []
        for pos in positions:
            reading = self._bilinear_interpolate(env_array, pos[0], pos[1])
            readings.append(reading)
        return np.array(readings)
```

#### Modern Sensor-Based Approach

**Step 1: Configure Sensors**
```python
# Sensor configurations for different research needs
sensor_configs = [
    # Binary detection for simple navigation
    {
        '_target_': 'plume_nav_sim.core.sensors.BinarySensor',
        'threshold': 0.1,
        'false_positive_rate': 0.02,
        'false_negative_rate': 0.01
    },
    
    # Quantitative measurement for gradient following
    {
        '_target_': 'plume_nav_sim.core.sensors.ConcentrationSensor',
        'dynamic_range': (0, 1),
        'resolution': 0.001,
        'noise_std': 0.05
    },
    
    # Spatial gradient for directional cues
    {
        '_target_': 'plume_nav_sim.core.sensors.GradientSensor',
        'spatial_resolution': (0.5, 0.5),
        'method': 'central_difference'
    }
]

# Create sensors through factory
sensors = NavigatorFactory.create_sensors(sensor_configs)
```

**Step 2: Update Controller Implementation**
```python
from plume_nav_sim.core.protocols import NavigatorProtocol, SensorProtocol

class ModernController:
    """Modern controller with sensor-based perception."""
    
    def __init__(self, position, max_speed, sensors):
        self.position = np.array(position)
        self.max_speed = max_speed
        self.sensors = sensors
    
    def sample_environment(self, plume_state):
        """Sensor-based environment sampling."""
        sensor_data = {}
        
        for i, sensor in enumerate(self.sensors):
            # Use sensor protocol methods
            if hasattr(sensor, 'detect'):
                sensor_data[f'detection_{i}'] = sensor.detect(plume_state, self.position)
            
            if hasattr(sensor, 'measure'):
                sensor_data[f'concentration_{i}'] = sensor.measure(plume_state, self.position)
            
            if hasattr(sensor, 'compute_gradient'):
                sensor_data[f'gradient_{i}'] = sensor.compute_gradient(plume_state, self.position)
        
        return sensor_data
    
    def step(self, plume_state, dt=1.0):
        """Navigation step with sensor-based perception."""
        # Get sensor readings through protocol interface
        sensor_readings = self.sample_environment(plume_state)
        
        # Navigation logic based on sensor abstractions
        self._navigate_with_sensor_data(sensor_readings)
        
        # Update position
        self.position += self.velocity * dt
```

**Step 3: Configuration-Driven Sensor Setup**

Create `conf/base/sensors/multi_modal.yaml`:
```yaml
# @package sensors
# Multi-modal sensor configuration
- _target_: plume_nav_sim.core.sensors.BinarySensor
  threshold: 0.1
  false_positive_rate: 0.02
  
- _target_: plume_nav_sim.core.sensors.ConcentrationSensor
  dynamic_range: [0, 1]
  resolution: 0.001
  noise_std: 0.05
  
- _target_: plume_nav_sim.core.sensors.GradientSensor
  spatial_resolution: [0.5, 0.5]
  method: central_difference
```

Create `conf/base/sensors/simple.yaml`:
```yaml
# @package sensors  
# Simple concentration sensing
- _target_: plume_nav_sim.core.sensors.ConcentrationSensor
  dynamic_range: [0, 1]
  resolution: 0.01
```

### Sensor Migration Mapping

| Legacy Method | Modern Sensor | Configuration |
|---------------|---------------|---------------|
| `sample_odor()` | `ConcentrationSensor.measure()` | `dynamic_range`, `resolution`, `noise_std` |
| Binary threshold logic | `BinarySensor.detect()` | `threshold`, `false_positive_rate` |
| Manual gradient computation | `GradientSensor.compute_gradient()` | `spatial_resolution`, `method` |
| `sample_multiple_sensors()` | Multiple sensor instances | List of sensor configurations |

---

## Agent Architecture Migration

### Migrating from Fixed Logic to Agent-Agnostic Design

The new architecture supports both memory-based and memory-less agents through optional interfaces without enforcing specific cognitive architectures.

#### Legacy Approach (Fixed Memory Assumptions)
```python
# Old agent with forced memory management
class LegacyAgent:
    def __init__(self):
        self.position_history = []  # Forced memory
        self.odor_history = []      # All agents had memory
        
    def step(self, env_array, dt):
        # Memory operations forced on all agents
        self.position_history.append(self.position.copy())
        odor = self.sample_odor(env_array)
        self.odor_history.append(odor)
        
        # Navigation logic assumes memory
        if len(self.odor_history) > 10:
            gradient = self._compute_temporal_gradient()
            # ... memory-dependent logic
```

#### Modern Approach (Optional Memory Interface)

**Memory-Less Reactive Agent**:
```python
from plume_nav_sim.core.protocols import NavigatorProtocol

class ReactiveAgent:
    """Simple reactive agent without memory requirements."""
    
    def __init__(self, position, max_speed, sensors):
        self.position = np.array(position)
        self.orientation = 0.0
        self.speed = 0.0
        self.max_speed = max_speed
        self.sensors = sensors
        # No memory attributes - purely reactive
    
    @property
    def positions(self):
        return self.position.reshape(1, -1)
    
    @property 
    def orientations(self):
        return np.array([self.orientation])
    
    def step(self, plume_state, dt=1.0):
        """Reactive navigation without memory."""
        # Get current sensor readings
        sensor_data = self._sample_sensors(plume_state)
        
        # Reactive decision making
        if sensor_data.get('concentration', 0) > 0.1:
            # Simple gradient following
            gradient = sensor_data.get('gradient', np.array([0, 0]))
            target_orientation = np.arctan2(gradient[1], gradient[0]) * 180 / np.pi
            self.orientation = target_orientation
            self.speed = self.max_speed
        else:
            # Random search when no odor detected
            self.orientation += np.random.uniform(-45, 45)
            self.speed = self.max_speed * 0.5
        
        # Update position
        velocity = self.speed * np.array([
            np.cos(np.radians(self.orientation)),
            np.sin(np.radians(self.orientation))
        ])
        self.position += velocity * dt
    
    # Optional memory interface - no-op for reactive agents
    def load_memory(self, memory_data=None):
        pass  # No memory to load
    
    def save_memory(self):
        return None  # No memory to save
```

**Memory-Based Cognitive Agent**:
```python
class CognitiveAgent:
    """Memory-based agent with spatial mapping and planning."""
    
    def __init__(self, position, max_speed, sensors, map_resolution=1.0):
        self.position = np.array(position)
        self.orientation = 0.0
        self.speed = 0.0
        self.max_speed = max_speed
        self.sensors = sensors
        
        # Memory components - optional for this agent type
        self.spatial_map = {}
        self.trajectory_history = []
        self.odor_history = []
        self.belief_state = {}
        self.map_resolution = map_resolution
    
    def step(self, plume_state, dt=1.0):
        """Cognitive navigation with memory and planning."""
        # Get sensor readings
        sensor_data = self._sample_sensors(plume_state)
        
        # Update memory structures
        self._update_spatial_map(sensor_data)
        self._update_trajectory_history()
        self._update_belief_state(sensor_data)
        
        # Plan based on accumulated knowledge
        target_location = self._plan_next_waypoint()
        
        # Execute planned movement
        direction_to_target = target_location - self.position
        if np.linalg.norm(direction_to_target) > 0:
            self.orientation = np.arctan2(direction_to_target[1], direction_to_target[0]) * 180 / np.pi
            self.speed = self.max_speed
        
        # Update position
        velocity = self.speed * np.array([
            np.cos(np.radians(self.orientation)),
            np.sin(np.radians(self.orientation))
        ])
        self.position += velocity * dt
    
    # Memory interface implementation
    def load_memory(self, memory_data=None):
        """Load previous episode memory."""
        if memory_data:
            self.spatial_map = memory_data.get('spatial_map', {})
            self.trajectory_history = memory_data.get('trajectory_history', [])
            self.belief_state = memory_data.get('belief_state', {})
    
    def save_memory(self):
        """Save current memory state."""
        return {
            'spatial_map': self.spatial_map,
            'trajectory_history': self.trajectory_history[-1000:],  # Keep last 1000 steps
            'belief_state': self.belief_state,
            'metadata': {'timestamp': time.time(), 'version': '1.0'}
        }
    
    def _update_spatial_map(self, sensor_data):
        """Update spatial concentration map."""
        grid_pos = tuple(np.round(self.position / self.map_resolution).astype(int))
        concentration = sensor_data.get('concentration', 0)
        self.spatial_map[grid_pos] = concentration
    
    def _plan_next_waypoint(self):
        """Plan next exploration/exploitation target."""
        if not self.spatial_map:
            # No map yet - explore randomly
            return self.position + np.random.uniform(-10, 10, 2)
        
        # Find highest concentration areas
        max_concentration = max(self.spatial_map.values())
        high_conc_positions = [
            pos for pos, conc in self.spatial_map.items() 
            if conc > 0.8 * max_concentration
        ]
        
        if high_conc_positions:
            # Move toward highest concentration area
            target_grid = np.array(high_conc_positions[0])
            return target_grid * self.map_resolution
        else:
            # Explore unvisited areas
            return self._find_unexplored_area()
```

#### Configuration-Based Agent Selection

Create `conf/base/agents/reactive.yaml`:
```yaml
# @package navigator
_target_: plume_nav_sim.examples.agents.ReactiveAgent
position: [0.0, 0.0]
max_speed: 2.0
sensors:
  - _target_: plume_nav_sim.core.sensors.ConcentrationSensor
    dynamic_range: [0, 1]
    resolution: 0.01
```

Create `conf/base/agents/cognitive.yaml`:
```yaml
# @package navigator  
_target_: plume_nav_sim.examples.agents.CognitiveAgent
position: [0.0, 0.0]
max_speed: 2.0
map_resolution: 1.0
enable_memory: true
sensors:
  - _target_: plume_nav_sim.core.sensors.ConcentrationSensor
    dynamic_range: [0, 1]
    resolution: 0.001
  - _target_: plume_nav_sim.core.sensors.GradientSensor
    spatial_resolution: [0.5, 0.5]
```

### NavigatorProtocol Extensions

#### New Optional Interfaces
```python
# Memory management (optional)
def load_memory(self, memory_data: Optional[Dict[str, Any]] = None) -> None:
    """Load agent memory - optional interface."""
    pass  # Default: no-op for memory-less agents

def save_memory(self) -> Optional[Dict[str, Any]]:
    """Save agent memory - optional interface.""" 
    return None  # Default: no memory to save

# Extensibility hooks (optional)
def compute_additional_obs(self, base_obs: dict) -> dict:
    """Add custom observations - extensibility hook."""
    return {}  # Default: no additional observations

def compute_extra_reward(self, base_reward: float, info: dict) -> float:
    """Add reward shaping - extensibility hook."""
    return 0.0  # Default: no additional reward

def on_episode_end(self, final_info: dict) -> None:
    """Handle episode completion - extensibility hook."""
    pass  # Default: no special handling
```

---

## Configuration System Migration

### Migrating from Monolithic to Modular Hydra Configuration

#### Legacy Configuration Structure
```python
# Old monolithic configuration
config = {
    "video_path": "data/plume_movie.mp4",
    "navigator": {
        "position": [0, 0],
        "max_speed": 2.0
    },
    "environment": {
        "max_episode_steps": 1000,
        "reward_type": "concentration"
    }
    # All settings in single configuration
}
```

#### Modern Modular Configuration

**Main Configuration (`conf/config.yaml`)**:
```yaml
# Main configuration with component selection
defaults:
  - base_config
  - plume_model: gaussian           # Pluggable plume model
  - wind_field: constant           # Optional wind dynamics
  - sensors: multi_modal           # Configurable sensing
  - navigator: single_agent        # Agent configuration
  - _self_

# Global simulation settings
simulation:
  max_episode_steps: 1000
  seed: 42
  
# Environment settings
environment:
  reward_type: concentration
  performance_monitoring: true
  
# Component integration settings
integration:
  enable_wind_field: true
  enable_sensors: true  
  enable_memory: false
```

**Component Configurations**:

`conf/base/plume_models/gaussian.yaml`:
```yaml
# @package plume_model
_target_: plume_nav_sim.models.plume.GaussianPlumeModel
source_position: [50.0, 50.0]
source_strength: 1000.0
sigma_x: 5.0
sigma_y: 3.0
wind_speed: 0.0
wind_direction: 0.0
```

`conf/base/wind_fields/constant.yaml`:
```yaml
# @package wind_field
_target_: plume_nav_sim.models.wind.ConstantWindField
velocity: [2.0, 0.5]
```

`conf/base/wind_fields/turbulent.yaml`:
```yaml
# @package wind_field
_target_: plume_nav_sim.models.wind.TurbulentWindField
mean_velocity: [3.0, 1.0]
turbulence_intensity: 0.2
gust_frequency: 0.1
```

#### Command-Line Configuration Switching

```bash
# Switch plume models without code changes
python run_experiment.py plume_model=gaussian
python run_experiment.py plume_model=turbulent
python run_experiment.py plume_model=video

# Combine different components
python run_experiment.py plume_model=gaussian wind_field=turbulent sensors=multi_modal

# Override specific parameters
python run_experiment.py plume_model=gaussian plume_model.source_strength=2000 navigator.max_speed=3.0

# Enable/disable features
python run_experiment.py integration.enable_wind_field=false integration.enable_memory=true
```

#### Programmatic Configuration

```python
import hydra
from omegaconf import DictConfig, OmegaConf
from plume_nav_sim.core.protocols import NavigatorFactory

@hydra.main(config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    """Main experiment function with modular configuration."""
    
    # Create modular environment from configuration
    env = NavigatorFactory.create_modular_environment(
        navigator_config=cfg.navigator,
        plume_model_config=cfg.plume_model,
        wind_field_config=cfg.wind_field if cfg.integration.enable_wind_field else None,
        sensor_configs=cfg.sensors if cfg.integration.enable_sensors else None,
        max_episode_steps=cfg.simulation.max_episode_steps
    )
    
    # Run experiment with configured components
    obs, info = env.reset(seed=cfg.simulation.seed)
    
    for step in range(cfg.simulation.max_episode_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            obs, info = env.reset()
    
    env.close()

if __name__ == "__main__":
    main()
```

### Configuration Migration Mapping

| Legacy Approach | Modern Modular Approach | Benefits |
|-----------------|-------------------------|----------|
| Single monolithic config | Component-specific configs | Reusable, composable configurations |
| Hard-coded model selection | `plume_model=model_name` CLI | Runtime model switching |
| Fixed environment settings | Hydra defaults system | Hierarchical configuration inheritance |
| Manual parameter overrides | CLI parameter override syntax | Easy parameter sweeps |
| Code changes for new models | Protocol-based instantiation | No code changes needed |

---

## Gymnasium Overview

The plume_nav_sim library v0.3.0 introduces full support for Gymnasium 0.29.x while maintaining backward compatibility with legacy OpenAI Gym patterns. This migration enables:

- **Modern API compliance** with 5-tuple step returns and improved reset handling
- **Enhanced type safety** with better type annotations and validation
- **Performance optimizations** with frame caching and sub-10ms step execution
- **Extensibility hooks** for custom observations, rewards, and episode handling
- **Automatic compatibility** through the built-in compatibility shim

### Key Benefits

✅ **Zero Breaking Changes**: Existing code continues to work unchanged  
✅ **Automatic Detection**: Smart API detection based on imports and usage patterns  
✅ **Performance Gains**: Optimized execution with <10ms step times  
✅ **Future-Proof**: Built on Gymnasium's actively maintained codebase  

---

## Quick Start Migration

### For New Projects (Recommended)

```python
# Modern Gymnasium approach
import gymnasium as gym
from plume_nav_sim.environments import GymnasiumEnv

# Create environment with new ID
env = gym.make("PlumeNavSim-v0", video_path="data/plume_movie.mp4")

# Modern reset (returns tuple)
obs, info = env.reset(seed=42)

# Modern step (returns 5-tuple)
obs, reward, terminated, truncated, info = env.step(action)

# Handle episode completion
if terminated or truncated:
    obs, info = env.reset()
```

### For Existing Projects (Compatibility Mode)

```python
# Legacy gym approach - continues to work
from plume_nav_sim.shims import gym_make

# Uses compatibility shim with deprecation warning
env = gym_make("PlumeNavSim-v0", video_path="data/plume_movie.mp4")

# Legacy reset (returns observation only)
obs = env.reset()

# Legacy step (returns 4-tuple)
obs, reward, done, info = env.step(action)

# Handle episode completion
if done:
    obs = env.reset()
```

---

## API Differences

### Key Changes Summary

| Component | Legacy Gym 0.26 | Modern Gymnasium 0.29.x |
|-----------|------------------|--------------------------|
| **Import** | `import gym` | `import gymnasium` |
| **Environment ID** | `OdorPlumeNavigation-v1` | `PlumeNavSim-v0` |
| **Reset Return** | `obs` | `obs, info` |
| **Step Return** | `obs, reward, done, info` | `obs, reward, terminated, truncated, info` |
| **Seed Parameter** | `env.seed(seed)` | `env.reset(seed=seed)` |

### Detailed API Comparison

#### 1. Environment Reset

**Legacy Gym:**
```python
obs = env.reset()
env.seed(42)  # Separate seed call
```

**Modern Gymnasium:**
```python
obs, info = env.reset(seed=42)  # Seed integrated into reset
```

#### 2. Environment Step

**Legacy Gym (4-tuple):**
```python
obs, reward, done, info = env.step(action)
if done:
    # Episode finished (any reason)
    obs = env.reset()
```

**Modern Gymnasium (5-tuple):**
```python
obs, reward, terminated, truncated, info = env.step(action)
if terminated or truncated:
    # terminated: episode ended naturally (success/failure)
    # truncated: episode ended due to time/step limits
    obs, info = env.reset()
```

#### 3. Done Flag Logic

**Legacy (single flag):**
```python
done = env_terminated_naturally or time_limit_reached
```

**Modern (separate flags):**
```python
terminated = env_terminated_naturally  # Success/failure
truncated = time_limit_reached         # Time/step limit
```

---

## Step-by-Step Migration

### Phase 1: Install Dependencies

```bash
# Update to Gymnasium
pip install "gymnasium>=0.29.0"

# Optional: Remove legacy gym to avoid confusion
pip uninstall gym

# Update plume_nav_sim
pip install "plume_nav_sim>=0.3.0"
```

### Phase 2: Update Imports

**Before:**
```python
import gym
from stable_baselines3 import PPO
```

**After:**
```python
import gymnasium as gym  # Note: Import as 'gym' for minimal changes
from stable_baselines3 import PPO
```

### Phase 3: Update Environment Creation

**Before:**
```python
env = gym.make("OdorPlumeNavigation-v1", video_path="data/plume.mp4")
```

**After:**
```python
env = gym.make("PlumeNavSim-v0", video_path="data/plume.mp4")
```

### Phase 4: Update Reset Logic

**Before:**
```python
def reset_environment(env, seed=None):
    if seed is not None:
        env.seed(seed)
    obs = env.reset()
    return obs
```

**After:**
```python
def reset_environment(env, seed=None):
    obs, info = env.reset(seed=seed)
    return obs  # Return only obs for backward compatibility
    # Or return obs, info for full modern API
```

### Phase 5: Update Step Logic

**Before:**
```python
def run_episode(env, policy):
    obs = env.reset()
    total_reward = 0
    
    while True:
        action = policy.predict(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        
        if done:
            break
    
    return total_reward
```

**After:**
```python
def run_episode(env, policy, seed=None):
    obs, info = env.reset(seed=seed)
    total_reward = 0
    
    while True:
        action = policy.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            break
    
    return total_reward
```

### Phase 6: Update Training Loops

**Before:**
```python
for episode in range(num_episodes):
    obs = env.reset()
    episode_reward = 0
    
    while True:
        action = agent.act(obs)
        next_obs, reward, done, info = env.step(action)
        
        agent.learn(obs, action, reward, next_obs, done)
        
        obs = next_obs
        episode_reward += reward
        
        if done:
            break
```

**After:**
```python
for episode in range(num_episodes):
    obs, info = env.reset(seed=episode)  # Optional: seed for reproducibility
    episode_reward = 0
    
    while True:
        action = agent.act(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        # Pass both flags to agent (or combine as needed)
        done = terminated or truncated
        agent.learn(obs, action, reward, next_obs, done)
        
        obs = next_obs
        episode_reward += reward
        
        if terminated or truncated:
            break
```

---

## Compatibility Shim Usage

The plume_nav_sim library provides a compatibility shim for seamless migration without code changes.

### Automatic API Detection

The shim automatically detects your usage pattern:

```python
# Detected as legacy usage (emits deprecation warning)
import gym
env = gym.make("OdorPlumeNavigation-v1")

# Detected as modern usage (no warning)
import gymnasium
env = gymnasium.make("PlumeNavSim-v0")
```

### Manual Compatibility Mode

For explicit control over API mode:

```python
from plume_nav_sim.shims import gym_make

# Force legacy mode (4-tuple returns)
env = gym_make("PlumeNavSim-v0", 
               video_path="data/plume.mp4",
               _force_legacy_api=True)

# Explicit modern mode
from plume_nav_sim.environments import GymnasiumEnv
env = GymnasiumEnv(video_path="data/plume.mp4")
```

### Working with Existing Codebases

For large codebases where gradual migration is preferred:

```python
# Step 1: Use shim to maintain existing behavior
from plume_nav_sim.shims import gym_make

def create_env():
    # This returns legacy 4-tuple format automatically
    return gym_make("PlumeNavSim-v0", video_path="data/plume.mp4")

# Step 2: Gradually migrate individual functions
def modern_create_env():
    import gymnasium as gym
    env = gym.make("PlumeNavSim-v0", video_path="data/plume.mp4")
    return env  # Returns modern 5-tuple format
```

---

## Environment Registration

### New Environment IDs

| Purpose | Legacy ID | Modern ID | Status |
|---------|-----------|-----------|---------|
| Primary | `OdorPlumeNavigation-v1` | `PlumeNavSim-v0` | ✅ Recommended |
| Legacy Support | `OdorPlumeNavigation-v0` | `PlumeNavSim-v0` | ⚠️ Deprecated |

### Registration Details

**Modern Gymnasium Registration:**
```python
import gymnasium as gym
from plume_nav_sim.environments import register_environments

# Register all environments
register_environments()

# Available environments
envs = gym.envs.registry.env_specs
print("PlumeNavSim-v0" in envs)  # True
```

**Custom Environment Configuration:**
```python
import gymnasium as gym

# With custom configuration
env = gym.make(
    "PlumeNavSim-v0",
    video_path="data/custom_plume.mp4",
    max_episode_steps=500,
    initial_position=(320, 240),
    max_speed=2.0,
    include_multi_sensor=True,
    num_sensors=3
)
```

---

## Code Examples

### Example 1: Basic Environment Usage

```python
"""Basic environment usage with modern Gymnasium API."""
import gymnasium as gym
import numpy as np

# Create environment
env = gym.make("PlumeNavSim-v0", 
               video_path="data/plume_movie.mp4",
               max_episode_steps=1000)

# Reset with seed for reproducibility
obs, info = env.reset(seed=42)
print(f"Initial observation keys: {list(obs.keys())}")
print(f"Initial info: {info}")

# Run a few steps
for step in range(5):
    # Sample random action
    action = env.action_space.sample()
    
    # Execute step
    obs, reward, terminated, truncated, info = env.step(action)
    
    print(f"Step {step + 1}:")
    print(f"  Reward: {reward:.3f}")
    print(f"  Terminated: {terminated}")
    print(f"  Truncated: {truncated}")
    print(f"  Agent position: {obs['agent_position']}")
    print(f"  Odor concentration: {obs['odor_concentration']:.3f}")
    
    if terminated or truncated:
        print(f"Episode ended after {step + 1} steps")
        obs, info = env.reset()
        break

env.close()
```

### Example 2: Training with Stable-Baselines3

```python
"""Training example with stable-baselines3 and Gymnasium."""
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

# Create vectorized environments
def make_env():
    return gym.make("PlumeNavSim-v0", 
                   video_path="data/plume_movie.mp4",
                   max_episode_steps=500)

# Create training and evaluation environments
train_env = make_vec_env(make_env, n_envs=4)
eval_env = make_vec_env(make_env, n_envs=1)

# Create evaluation callback
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./logs/best_model",
    log_path="./logs/eval",
    eval_freq=5000,
    deterministic=True,
    render=False
)

# Create and train model
model = PPO(
    "MultiInputPolicy",
    train_env,
    verbose=1,
    tensorboard_log="./logs/tensorboard"
)

# Train the model
model.learn(
    total_timesteps=100000,
    callback=eval_callback
)

# Save final model
model.save("ppo_plume_navigation")

# Test trained model
obs = eval_env.reset()
for _ in range(100):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = eval_env.step(action)
    if done.any():
        obs = eval_env.reset()
```

### Example 3: Custom Environment Extensions

```python
"""Example showing extensibility hooks for custom processing."""
import gymnasium as gym
import numpy as np
from plume_nav_sim.environments import GymnasiumEnv

class CustomPlumeEnv(GymnasiumEnv):
    """Extended environment with custom observations and rewards."""
    
    def compute_additional_obs(self, base_obs):
        """Add custom observations."""
        # Add distance to estimated source
        agent_pos = base_obs["agent_position"]
        source_estimate = getattr(self, "_source_estimate", [320, 240])
        distance = np.linalg.norm(agent_pos - source_estimate)
        
        return {
            "distance_to_source": np.array([distance], dtype=np.float32),
            "exploration_progress": np.array([self._get_exploration_progress()], dtype=np.float32)
        }
    
    def compute_extra_reward(self, base_reward, info):
        """Add reward shaping."""
        # Bonus for reducing distance to source
        if hasattr(self, "_previous_distance"):
            current_distance = info.get("distance_to_source", 0)
            distance_change = self._previous_distance - current_distance
            distance_bonus = distance_change * 0.1
            self._previous_distance = current_distance
            return distance_bonus
        else:
            self._previous_distance = info.get("distance_to_source", 0)
            return 0.0
    
    def on_episode_end(self, final_info):
        """Custom episode-end processing."""
        episode_length = final_info.get("step", 0)
        final_reward = final_info.get("total_reward", 0)
        
        print(f"Episode completed:")
        print(f"  Length: {episode_length} steps")
        print(f"  Total reward: {final_reward:.2f}")
        print(f"  Final odor: {final_info.get('odor_concentration', 0):.3f}")
    
    def _get_exploration_progress(self):
        """Calculate exploration progress."""
        return float(np.sum(self._exploration_grid) / self._exploration_grid.size)

# Use custom environment
env = CustomPlumeEnv(video_path="data/plume_movie.mp4")
obs, info = env.reset(seed=42)

# Custom observations are automatically included
print(f"Available observations: {list(obs.keys())}")
```

### Example 4: Performance Monitoring

```python
"""Example showing performance monitoring and optimization."""
import gymnasium as gym
import time
import numpy as np
from plume_nav_sim.utils.frame_cache import FrameCache

# Create frame cache for performance
cache = FrameCache(mode="lru", max_size_mb=512)

# Create environment with cache
env = gym.make("PlumeNavSim-v0",
               video_path="data/plume_movie.mp4",
               frame_cache=cache,
               performance_monitoring=True)

# Performance benchmark
obs, info = env.reset(seed=42)
step_times = []

print("Running performance benchmark...")
for i in range(1000):
    start_time = time.perf_counter()
    
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    step_time = time.perf_counter() - start_time
    step_times.append(step_time * 1000)  # Convert to milliseconds
    
    if terminated or truncated:
        obs, info = env.reset()
    
    # Print progress every 100 steps
    if (i + 1) % 100 == 0:
        avg_time = np.mean(step_times[-100:])
        cache_stats = env.get_cache_stats()
        print(f"Steps {i+1-99}-{i+1}: {avg_time:.2f}ms avg, "
              f"cache hit rate: {cache_stats['hit_rate']:.1%}")

# Final performance summary
avg_step_time = np.mean(step_times)
p95_step_time = np.percentile(step_times, 95)
cache_stats = env.get_cache_stats()

print(f"\nPerformance Summary:")
print(f"  Average step time: {avg_step_time:.2f}ms")
print(f"  95th percentile: {p95_step_time:.2f}ms")
print(f"  Target compliance: {avg_step_time <= 10}")
print(f"  Cache hit rate: {cache_stats['hit_rate']:.1%}")
print(f"  Memory usage: {cache_stats['memory_usage_mb']:.1f}MB")

env.close()
```

---

## Performance Considerations

### Frame Caching Optimization

The new version includes enhanced frame caching for optimal performance:

```python
from plume_nav_sim.utils.frame_cache import FrameCache

# Configure cache based on available memory
cache_configs = {
    "high_memory": FrameCache(mode="all", max_size_mb=2048),     # Preload all frames
    "balanced": FrameCache(mode="lru", max_size_mb=1024),       # LRU cache
    "low_memory": None                                          # No cache
}

# Choose appropriate config
cache = cache_configs["balanced"]
env = gym.make("PlumeNavSim-v0", 
               video_path="data/plume_movie.mp4",
               frame_cache=cache)
```

### Performance Targets

| Metric | Target | Notes |
|--------|--------|--------|
| Step time | <10ms average | Critical for real-time training |
| Cache hit rate | >90% | With LRU cache enabled |
| Memory usage | <2GB per process | With frame cache limits |
| Reset time | <20ms | Including cache warmup |

### Monitoring Performance

```python
# Check performance metrics
performance_info = info.get("perf_stats", {})
if performance_info:
    print(f"Step time: {performance_info['step_time_ms']:.2f}ms")
    print(f"Cache hit rate: {performance_info['cache_hit_rate']:.1%}")
    
    # Warn if performance targets not met
    if performance_info['step_time_ms'] > 10:
        print("⚠️ Step time exceeds 10ms target")
```

---

## Migration Troubleshooting

### Common Modular Architecture Migration Issues

#### 1. Protocol Compliance Errors

**Problem:**
```python
TypeError: Can't instantiate abstract class CustomPlumeModel with abstract methods concentration_at
```

**Solution:**
```python
# Ensure custom implementations satisfy protocol requirements
from plume_nav_sim.core.protocols import PlumeModelProtocol

class CustomPlumeModel:
    """Custom plume model implementing PlumeModelProtocol."""
    
    def concentration_at(self, positions: np.ndarray) -> np.ndarray:
        """Required method implementation."""
        # Must implement this method
        return np.zeros(positions.shape[0])
    
    def step(self, dt: float = 1.0) -> None:
        """Required method implementation.""" 
        # Must implement this method
        pass
    
    def reset(self, **kwargs: Any) -> None:
        """Required method implementation."""
        # Must implement this method
        pass

# Validate protocol compliance
from plume_nav_sim.core.protocols import NavigatorFactory
is_valid = NavigatorFactory.validate_protocol_compliance(
    CustomPlumeModel(), PlumeModelProtocol
)
assert is_valid, "Must implement PlumeModelProtocol"
```

#### 2. Configuration Loading Errors

**Problem:**
```python
ConfigError: Cannot find primary config module 'plume_nav_sim.models.plume.GaussianPlumeModel'
```

**Solution:**
```python
# Check module paths and imports
# Option 1: Verify Hydra target path
_target_: plume_nav_sim.models.plume.gaussian_plume.GaussianPlumeModel  # Correct full path

# Option 2: Use factory method instead of direct instantiation
from plume_nav_sim.core.protocols import NavigatorFactory
config = {
    'type': 'GaussianPlumeModel',
    'source_position': (50, 50),
    'source_strength': 1000.0
}
plume_model = NavigatorFactory.create_plume_model(config)

# Option 3: Check if modules are available
try:
    from plume_nav_sim.models.plume.gaussian_plume import GaussianPlumeModel
    print("Module available")
except ImportError as e:
    print(f"Module not available: {e}")
```

#### 3. Sensor Configuration Issues

**Problem:**
```python
AttributeError: 'ConcentrationSensor' object has no attribute 'detect'
```

**Solution:**
```python
# Check sensor method availability before calling
sensor_data = {}
for sensor in self.sensors:
    # Check which methods the sensor implements
    if hasattr(sensor, 'detect'):
        sensor_data['detection'] = sensor.detect(plume_state, positions)
    if hasattr(sensor, 'measure'):
        sensor_data['concentration'] = sensor.measure(plume_state, positions)
    if hasattr(sensor, 'compute_gradient'):
        sensor_data['gradient'] = sensor.compute_gradient(plume_state, positions)

# Or use specific sensor types
from plume_nav_sim.core.sensors import BinarySensor, ConcentrationSensor, GradientSensor

# BinarySensor has detect() method
binary_sensor = BinarySensor(threshold=0.1)
detection = binary_sensor.detect(plume_state, positions)

# ConcentrationSensor has measure() method  
conc_sensor = ConcentrationSensor(dynamic_range=(0, 1))
concentration = conc_sensor.measure(plume_state, positions)

# GradientSensor has compute_gradient() method
grad_sensor = GradientSensor(spatial_resolution=(0.5, 0.5))
gradient = grad_sensor.compute_gradient(plume_state, positions)
```

#### 4. Memory Interface Confusion

**Problem:**
```python
# Trying to use memory methods on memory-less agents
agent.save_memory()  # Returns None unexpectedly
```

**Solution:**
```python
# Check if agent supports memory before using
def safely_save_memory(agent):
    """Safely save agent memory if supported."""
    if hasattr(agent, 'save_memory'):
        memory = agent.save_memory()
        if memory is not None:
            return memory
        else:
            print("Agent does not maintain memory")
            return {}
    else:
        print("Agent does not support memory interface")
        return {}

# Or use isinstance check for memory-enabled agents
from plume_nav_sim.examples.agents import CognitiveAgent, ReactiveAgent

if isinstance(agent, CognitiveAgent):
    memory = agent.save_memory()  # Safe - cognitive agents have memory
elif isinstance(agent, ReactiveAgent):
    print("Reactive agent - no memory to save")  # Expected behavior
```

#### 5. Component Integration Issues

**Problem:**
```python
# Components not integrating properly
ValueError: WindField not compatible with PlumeModel
```

**Solution:**
```python
# Ensure components are designed to work together
plume_config = {
    '_target_': 'plume_nav_sim.models.plume.GaussianPlumeModel',
    'source_position': (50, 50),
    'enable_wind_field': True,  # Enable wind integration
    'wind_field': {
        '_target_': 'plume_nav_sim.models.wind.ConstantWindField',
        'velocity': (2.0, 0.5)
    }
}

# Or create components separately and integrate manually
wind_field = NavigatorFactory.create_wind_field({
    'type': 'ConstantWindField', 
    'velocity': (2.0, 0.5)
})

plume_model = NavigatorFactory.create_plume_model({
    'type': 'GaussianPlumeModel',
    'source_position': (50, 50),
    'wind_field': wind_field
})
```

### Performance Migration Issues

#### 1. Slow Step Execution

**Problem:**
```python
# Step times >10ms with new modular architecture
Warning: Step execution exceeded 10ms target
```

**Solutions:**
```python
# Option 1: Optimize sensor configuration
sensor_configs = [
    {
        '_target_': 'plume_nav_sim.core.sensors.ConcentrationSensor',
        'dynamic_range': (0, 1),
        'resolution': 0.01,  # Lower resolution for speed
        'cache_readings': True  # Enable caching
    }
]

# Option 2: Use simpler plume models for development
plume_config = {
    '_target_': 'plume_nav_sim.models.plume.GaussianPlumeModel',  # Faster than turbulent
    'concentration_cutoff': 1e-4,  # Higher cutoff for efficiency
    'enable_wind_field': False  # Disable wind for simpler physics
}

# Option 3: Enable performance monitoring to identify bottlenecks
env = PlumeNavigationEnv.from_config(cfg, performance_monitoring=True)
obs, reward, terminated, truncated, info = env.step(action)
perf_stats = info.get('perf_stats', {})
print(f"Component timings: {perf_stats}")
```

#### 2. Memory Usage Issues

**Problem:**
```python
# High memory usage with multiple sensors and wind fields
MemoryError: Unable to allocate memory for sensor readings
```

**Solutions:**
```python
# Option 1: Limit sensor history
sensor_config = {
    '_target_': 'plume_nav_sim.core.sensors.ConcentrationSensor',
    'memory_limit': 100,  # Limit historical readings
    'enable_history': False  # Disable history if not needed
}

# Option 2: Use memory-efficient wind fields
wind_config = {
    '_target_': 'plume_nav_sim.models.wind.ConstantWindField',  # Minimal memory
    'velocity': (2.0, 0.5)
    # Avoid TurbulentWindField if memory is constrained
}

# Option 3: Clear caches periodically
if hasattr(env, 'clear_component_caches'):
    env.clear_component_caches()
```

### Configuration Migration Issues

#### 1. Hydra Configuration Errors

**Problem:**
```python
ConfigCompositionException: Cannot compose configuration
```

**Solution:**
```python
# Check configuration structure and package directives
# Ensure config files have correct @package directives

# conf/base/plume_models/gaussian.yaml
# @package plume_model  <- Important package directive
_target_: plume_nav_sim.models.plume.GaussianPlumeModel

# conf/base/sensors/simple.yaml  
# @package sensors  <- Important package directive
- _target_: plume_nav_sim.core.sensors.ConcentrationSensor
  dynamic_range: [0, 1]

# Main config.yaml defaults structure
defaults:
  - base_config
  - plume_model: gaussian  # References gaussian.yaml 
  - sensors: simple        # References simple.yaml
  - _self_
```

#### 2. Parameter Override Issues

**Problem:**
```python
# Parameter overrides not working
python run_experiment.py plume_model.source_strength=2000  # Ignored
```

**Solution:**
```python
# Check override syntax and package structure
# Correct syntax:
python run_experiment.py plume_model.source_strength=2000

# Verify parameter path in configuration
# In gaussian.yaml:
source_strength: 1000.0  # This parameter can be overridden

# Check if parameter is under correct package
# If parameter is nested:
python run_experiment.py plume_model.physics.source_strength=2000

# Debug configuration resolution
import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path="../conf", config_name="config")
def debug_config(cfg: DictConfig):
    print("Resolved configuration:")
    print(OmegaConf.to_yaml(cfg))
    print(f"Source strength: {cfg.plume_model.source_strength}")
```

---

## Performance Optimization

### Modular Architecture Performance Tuning

#### 1. Component Selection for Performance

```python
# Performance hierarchy (fastest to slowest)
performance_configs = {
    'fastest': {
        'plume_model': 'gaussian',      # Analytical computation
        'wind_field': 'constant',       # Simple constant wind
        'sensors': 'simple',            # Single concentration sensor
        'enable_wind_field': False      # Disable wind physics
    },
    'balanced': {
        'plume_model': 'gaussian',      # Analytical computation  
        'wind_field': 'constant',       # Simple wind effects
        'sensors': 'multi_modal',       # Multiple sensor types
        'enable_wind_field': True       # Enable wind physics
    },
    'realistic': {
        'plume_model': 'turbulent',     # Complex physics
        'wind_field': 'turbulent',      # Realistic wind dynamics
        'sensors': 'multi_modal',       # Full sensor suite
        'enable_wind_field': True       # Full physics
    }
}

# Select based on requirements
config_type = 'balanced'  # Good compromise
python run_experiment.py --config-name=config_${config_type}
```

#### 2. Sensor Optimization

```python
# High-performance sensor configuration
sensor_configs = [
    {
        '_target_': 'plume_nav_sim.core.sensors.ConcentrationSensor',
        'dynamic_range': (0, 1),
        'resolution': 0.01,             # Lower resolution = faster
        'enable_noise': False,          # Disable noise computation
        'cache_readings': True,         # Enable result caching
        'batch_processing': True        # Vectorized operations
    }
]

# Memory-efficient sensor settings
sensor_configs = [
    {
        '_target_': 'plume_nav_sim.core.sensors.ConcentrationSensor',
        'memory_limit': 10,             # Minimal history
        'enable_history': False,        # No temporal storage
        'lazy_evaluation': True         # Compute on demand
    }
]
```

#### 3. Plume Model Optimization

```python
# Optimized Gaussian plume configuration
gaussian_config = {
    '_target_': 'plume_nav_sim.models.plume.GaussianPlumeModel',
    'source_position': (50, 50),
    'source_strength': 1000.0,
    'concentration_cutoff': 1e-4,      # Higher cutoff = fewer computations
    'enable_vectorized_ops': True,     # Batch processing
    'cache_spatial_grid': True,        # Precompute spatial grid
    'temporal_resolution': 1.0         # Lower resolution = faster
}

# Performance monitoring configuration
gaussian_config['performance_monitoring'] = {
    'enable_profiling': True,
    'log_slow_operations': True,
    'timing_threshold_ms': 1.0
}
```

#### 4. Multi-Agent Performance Scaling

```python
# Vectorized multi-agent configuration
multi_agent_config = {
    'navigator': {
        '_target_': 'plume_nav_sim.core.controllers.MultiAgentController',
        'positions': [[i*5, j*5] for i in range(10) for j in range(10)],  # 100 agents
        'enable_vectorized_ops': True,     # Essential for performance
        'batch_size': 50,                  # Process in batches
        'parallel_processing': True        # Use multiprocessing
    },
    'plume_model': {
        '_target_': 'plume_nav_sim.models.plume.GaussianPlumeModel',
        'enable_batch_queries': True,      # Batch concentration queries
        'vectorized_computation': True     # NumPy vectorization
    }
}
```

---

## Protocol Compliance Validation

### Validating Custom Implementations

#### 1. PlumeModelProtocol Compliance

```python
from plume_nav_sim.core.protocols import PlumeModelProtocol, NavigatorFactory

def validate_plume_model(model_instance):
    """Validate custom plume model implementation."""
    
    # Protocol compliance check
    is_compliant = NavigatorFactory.validate_protocol_compliance(
        model_instance, PlumeModelProtocol
    )
    
    if not is_compliant:
        print("❌ Model does not implement PlumeModelProtocol")
        return False
    
    # Method signature validation
    required_methods = ['concentration_at', 'step', 'reset']
    for method_name in required_methods:
        if not hasattr(model_instance, method_name):
            print(f"❌ Missing required method: {method_name}")
            return False
        
        method = getattr(model_instance, method_name)
        if not callable(method):
            print(f"❌ {method_name} is not callable")
            return False
    
    # Functional validation
    try:
        # Test concentration query
        test_positions = np.array([[10, 20], [30, 40]])
        concentrations = model_instance.concentration_at(test_positions)
        
        if not isinstance(concentrations, np.ndarray):
            print("❌ concentration_at must return numpy array")
            return False
        
        if concentrations.shape[0] != test_positions.shape[0]:
            print("❌ concentration_at output shape mismatch")
            return False
        
        # Test temporal step
        model_instance.step(dt=1.0)
        
        # Test reset
        model_instance.reset()
        
        print("✅ PlumeModelProtocol validation passed")
        return True
        
    except Exception as e:
        print(f"❌ Functional validation failed: {e}")
        return False

# Example usage
custom_model = CustomPlumeModel()
is_valid = validate_plume_model(custom_model)
```

#### 2. SensorProtocol Compliance

```python
from plume_nav_sim.core.protocols import SensorProtocol

def validate_sensor(sensor_instance):
    """Validate custom sensor implementation."""
    
    # Check protocol compliance
    is_compliant = NavigatorFactory.validate_protocol_compliance(
        sensor_instance, SensorProtocol
    )
    
    # Check sensor-specific methods
    sensor_methods = []
    if hasattr(sensor_instance, 'detect'):
        sensor_methods.append('detect')
    if hasattr(sensor_instance, 'measure'):
        sensor_methods.append('measure')
    if hasattr(sensor_instance, 'compute_gradient'):
        sensor_methods.append('compute_gradient')
    
    if not sensor_methods:
        print("❌ Sensor must implement at least one of: detect, measure, compute_gradient")
        return False
    
    # Functional testing
    try:
        mock_plume_state = MockPlumeState()
        test_positions = np.array([[15, 25]])
        
        for method_name in sensor_methods:
            method = getattr(sensor_instance, method_name)
            result = method(mock_plume_state, test_positions)
            
            if not isinstance(result, (np.ndarray, float, bool)):
                print(f"❌ {method_name} must return numpy array, float, or bool")
                return False
        
        print(f"✅ SensorProtocol validation passed for methods: {sensor_methods}")
        return True
        
    except Exception as e:
        print(f"❌ Sensor functional validation failed: {e}")
        return False
```

#### 3. NavigatorProtocol Compliance

```python
from plume_nav_sim.core.protocols import NavigatorProtocol

def validate_navigator(navigator_instance):
    """Validate custom navigator implementation."""
    
    # Protocol compliance
    is_compliant = NavigatorFactory.validate_protocol_compliance(
        navigator_instance, NavigatorProtocol
    )
    
    # Required properties validation
    required_properties = [
        'positions', 'orientations', 'speeds', 'max_speeds', 
        'angular_velocities', 'num_agents'
    ]
    
    for prop_name in required_properties:
        if not hasattr(navigator_instance, prop_name):
            print(f"❌ Missing required property: {prop_name}")
            return False
        
        try:
            value = getattr(navigator_instance, prop_name)
            if prop_name == 'num_agents':
                if not isinstance(value, int) or value < 1:
                    print(f"❌ {prop_name} must be positive integer")
                    return False
            else:
                if not isinstance(value, np.ndarray):
                    print(f"❌ {prop_name} must be numpy array")
                    return False
        except Exception as e:
            print(f"❌ Error accessing {prop_name}: {e}")
            return False
    
    # Required methods validation
    required_methods = ['reset', 'step', 'sample_odor']
    for method_name in required_methods:
        if not hasattr(navigator_instance, method_name):
            print(f"❌ Missing required method: {method_name}")
            return False
    
    # Optional methods check (for memory interface)
    optional_methods = ['load_memory', 'save_memory']
    memory_support = all(hasattr(navigator_instance, method) for method in optional_methods)
    
    if memory_support:
        print("✅ Memory interface supported")
    else:
        print("ℹ️ Memory interface not implemented (optional)")
    
    print("✅ NavigatorProtocol validation passed")
    return True
```

#### 4. Integration Testing

```python
def test_component_integration():
    """Test integration between modular components."""
    
    # Create components
    plume_config = {
        '_target_': 'plume_nav_sim.models.plume.GaussianPlumeModel',
        'source_position': (50, 50),
        'source_strength': 1000.0
    }
    plume_model = NavigatorFactory.create_plume_model(plume_config)
    
    sensor_configs = [{
        '_target_': 'plume_nav_sim.core.sensors.ConcentrationSensor',
        'dynamic_range': (0, 1)
    }]
    sensors = NavigatorFactory.create_sensors(sensor_configs)
    
    navigator_config = {
        'position': (0, 0),
        'max_speed': 2.0,
        'sensors': sensors
    }
    navigator = NavigatorFactory.from_config(navigator_config)
    
    # Test integration
    try:
        # Test plume model step
        plume_model.step(dt=1.0)
        
        # Test concentration query
        concentrations = plume_model.concentration_at(navigator.positions)
        
        # Test sensor readings
        for sensor in sensors:
            if hasattr(sensor, 'measure'):
                reading = sensor.measure(plume_model, navigator.positions)
        
        # Test navigator step
        navigator.step(plume_model, dt=1.0)
        
        print("✅ Component integration test passed")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False
```

---

## Troubleshooting (Gymnasium API)

### Common Migration Issues

#### 1. Import Errors

**Problem:**
```python
ImportError: No module named 'gym'
```

**Solution:**
```python
# Install gymnasium instead of legacy gym
pip install "gymnasium>=0.29.0"

# Update imports
import gymnasium as gym  # Instead of: import gym
```

#### 2. Tuple Length Mismatch

**Problem:**
```python
ValueError: too many values to unpack (expected 4)
obs, reward, done, info = env.step(action)  # Fails with 5-tuple
```

**Solution:**
```python
# Option 1: Update to modern API
obs, reward, terminated, truncated, info = env.step(action)
done = terminated or truncated

# Option 2: Use compatibility shim
from plume_nav_sim.shims import gym_make
env = gym_make("PlumeNavSim-v0")  # Returns 4-tuple automatically
```

#### 3. Environment Not Found

**Problem:**
```python
gym.error.UnregisteredEnv: No registered env with id: PlumeNavSim-v0
```

**Solution:**
```python
# Ensure environment is registered
from plume_nav_sim.environments import register_environments
register_environments()

# Then create environment
env = gym.make("PlumeNavSim-v0")
```

#### 4. Reset Return Format

**Problem:**
```python
TypeError: 'tuple' object has no attribute 'shape'
obs = env.reset()  # Returns (obs, info) in modern API
```

**Solution:**
```python
# Option 1: Update to modern format
obs, info = env.reset(seed=42)

# Option 2: Extract observation only
reset_result = env.reset()
if isinstance(reset_result, tuple):
    obs, info = reset_result
else:
    obs = reset_result
```

#### 5. Seed Handling

**Problem:**
```python
AttributeError: 'GymnasiumEnv' object has no attribute 'seed'
env.seed(42)  # Legacy seed method not available
```

**Solution:**
```python
# Use seed parameter in reset
obs, info = env.reset(seed=42)

# Or use action_space/observation_space seeding
env.action_space.seed(42)
env.observation_space.seed(42)
```

### Performance Issues

#### 1. Slow Step Times

**Symptoms:**
- Step times >10ms consistently
- Training slower than expected

**Solutions:**
```python
# Enable frame caching
from plume_nav_sim.utils.frame_cache import FrameCache
cache = FrameCache(mode="lru", max_size_mb=1024)
env = gym.make("PlumeNavSim-v0", frame_cache=cache)

# Disable performance monitoring in production
env = gym.make("PlumeNavSim-v0", performance_monitoring=False)

# Use vectorized environments
from stable_baselines3.common.env_util import make_vec_env
env = make_vec_env(lambda: gym.make("PlumeNavSim-v0"), n_envs=4)
```

#### 2. Memory Issues

**Symptoms:**
- Out of memory errors
- Gradually increasing memory usage

**Solutions:**
```python
# Use memory-limited cache
cache = FrameCache(mode="lru", max_size_mb=512)  # Smaller cache

# Or disable caching entirely
env = gym.make("PlumeNavSim-v0", frame_cache=None)

# Clear cache periodically
env.clear_cache()
```

### Compatibility Issues

#### 1. Mixed API Usage

**Problem:**
```python
# Code mixes legacy and modern patterns
obs = env.reset()  # Legacy
obs, reward, terminated, truncated, info = env.step(action)  # Modern
```

**Solution:**
```python
# Be consistent with API usage
obs, info = env.reset(seed=42)  # Modern
obs, reward, terminated, truncated, info = env.step(action)  # Modern

# Or use compatibility mode throughout
from plume_nav_sim.shims import gym_make
env = gym_make("PlumeNavSim-v0")
obs = env.reset()  # Legacy
obs, reward, done, info = env.step(action)  # Legacy
```

#### 2. Framework Compatibility

**Problem:**
```python
# Framework expects legacy format
some_rl_library.train(env)  # Expects 4-tuple
```

**Solution:**
```python
# Use compatibility wrapper
from plume_nav_sim.environments.compat import wrap_environment
wrapped_env = wrap_environment(env)
some_rl_library.train(wrapped_env)

# Or check framework documentation for Gymnasium support
```

### Debugging Tools

#### 1. API Compatibility Check

```python
from plume_nav_sim.environments.compat import validate_compatibility

# Check environment compatibility
results = validate_compatibility(env, test_episodes=3)
print(f"Status: {results['overall_status']}")
print(f"Recommendations: {results['recommendations']}")
```

#### 2. Performance Profiling

```python
# Enable detailed logging
import logging
logging.getLogger("plume_nav_sim").setLevel(logging.DEBUG)

# Use performance monitoring
env = gym.make("PlumeNavSim-v0", performance_monitoring=True)
obs, info = env.reset()

# Check step info for timing details
obs, reward, terminated, truncated, info = env.step(action)
perf_stats = info.get("perf_stats", {})
print(f"Step timing: {perf_stats}")
```

#### 3. Environment Diagnostics

```python
from plume_nav_sim.environments import diagnose_environment_setup

# Get system diagnostic information
diagnostics = diagnose_environment_setup()
print(f"System status: {diagnostics}")
```

---

## Migration Timeline

### Modular Architecture Migration Schedule

#### Current Status (v0.4.0)

✅ **Fully Supported:**
- Modular plume model architecture (Gaussian, Turbulent, Video-adapter)
- Sensor-based perception system with configurable modalities
- Agent-agnostic design supporting memory-based and memory-less strategies
- Wind field integration with realistic environmental dynamics
- Protocol-based extensibility hooks for custom implementations
- Hydra configuration system with component dependency injection
- Performance optimization maintaining <10ms step latency

⚠️ **Deprecated but Functional:**
- Direct VideoPlume instantiation (use VideoPlumeAdapter instead)
- Hard-coded odor sampling in controllers (use SensorProtocol)
- Monolithic configuration files (migrate to modular component configs)
- Fixed memory assumptions in agent logic (use optional memory interface)

#### Migration Phases

| Version | Status | Architectural Changes |
|---------|--------|----------------------|
| **v0.4.0** (Current) | ✅ Full modular support | - PlumeModelProtocol introduced<br>- SensorProtocol-based perception<br>- Optional memory interfaces<br>- Wind field integration<br>- Backward compatibility maintained |
| **v0.5.0** (Q2 2024) | ⚠️ Enhanced validation | - Protocol compliance enforcement<br>- Performance monitoring required<br>- Configuration validation tools<br>- Migration assistance utilities |
| **v0.6.0** (Q3 2024) | ⚠️ Legacy support optional | - Monolithic patterns require explicit flags<br>- Enhanced component libraries<br>- Advanced sensor implementations<br>- Memory management optimizations |
| **v1.0.0** (Q4 2024) | ❌ Legacy patterns removed | - Pure modular architecture only<br>- Protocol compliance required<br>- Optimized for modern workflows |

### Gymnasium API Migration Schedule

#### Current Status (v0.4.0)

✅ **Fully Supported:**
- Modern Gymnasium 0.29.x API with 5-tuple step returns
- Enhanced `PlumeNavSim-v0` environment ID
- Extensibility hooks for custom observations and rewards
- Performance monitoring with frame caching optimization
- Dual API compatibility detection and conversion

⚠️ **Deprecated but Functional:**
- Legacy Gym 0.26 API patterns (emits warnings)
- Legacy environment IDs (`OdorPlumeNavigation-v*`)
- 4-tuple step returns through compatibility shim
- Separate seed() method calls

#### API Migration Timeline

| Version | Status | API Changes |
|---------|--------|-------------|
| **v0.4.0** (Current) | ✅ Full compatibility | - Gymnasium 0.29.x support<br>- Automatic API detection<br>- Performance-optimized modern API |
| **v0.5.0** (Q2 2024) | ⚠️ Increased warnings | - More prominent deprecation notices<br>- Enhanced migration tools<br>- Performance penalties for legacy usage |
| **v0.6.0** (Q3 2024) | ⚠️ Legacy requires flags | - Legacy support requires explicit enablement<br>- Default to modern API only<br>- Comprehensive migration documentation |
| **v1.0.0** (Q4 2024) | ❌ Legacy support removed | - Modern Gymnasium API only<br>- Legacy compatibility shim removed<br>- Optimized for modern workflows |

### Migration Priority Matrix

| Component/Usage | Migration Urgency | Action Required |
|-----------------|-------------------|-----------------|
| **New research projects** | **Low** | Use modular architecture from start |
| **Algorithm development** | **Medium** | Migrate to sensor-based perception |
| **Production experiments** | **Medium** | Plan migration by v0.6.0 |
| **Legacy codebases** | **High** | Begin modular migration immediately |
| **VideoPlume workflows** | **Medium** | Switch to VideoPlumeAdapter wrapper |
| **Custom agent implementations** | **Medium** | Adopt NavigatorProtocol interface |
| **Hard-coded sensing logic** | **High** | Migrate to SensorProtocol abstractions |
| **Monolithic configurations** | **Low** | Gradually adopt Hydra modular configs |

### Preparing for v1.0.0

To ensure smooth transition to the fully modular v1.0.0 release:

#### 1. Architecture Updates
- **Adopt modular components:** Use PlumeModelProtocol implementations
- **Implement sensor abstractions:** Replace hard-coded odor sampling  
- **Update agent designs:** Use NavigatorProtocol with optional memory
- **Migrate configurations:** Convert to Hydra modular component configs

#### 2. API Modernization
- **Update dependencies:** Install Gymnasium ≥0.29.0
- **Test compatibility:** Run validation tools on existing code
- **Migrate gradually:** Use compatibility detection during transition
- **Update CI/CD:** Test with both legacy and modern API patterns

#### 3. Performance Validation
- **Benchmark components:** Ensure <10ms step execution with new architecture
- **Optimize configurations:** Select efficient component combinations
- **Monitor performance:** Use built-in profiling and metrics collection
- **Scale testing:** Validate multi-agent scenarios with vectorized operations

#### 4. Protocol Compliance
- **Validate implementations:** Use protocol compliance checking tools
- **Test integrations:** Ensure components work together correctly
- **Document extensions:** Clearly specify custom protocol implementations
- **Prepare fallbacks:** Handle component unavailability gracefully

### Getting Migration Help

#### Documentation Resources
- **Architecture Guide:** [Extending PlumeNavSim](./extending_plume_nav_sim.md)
- **Protocol References:** Complete API documentation for all protocols
- **Configuration Examples:** Sample modular configurations in `conf/` directory
- **Performance Guides:** Optimization recommendations for different scenarios

#### Migration Tools
- **Protocol Validation:** Built-in compliance checking utilities
- **Configuration Converters:** Automated migration from monolithic configs
- **Performance Profilers:** Component-level timing and memory analysis
- **Compatibility Testing:** Automated validation for existing workflows

#### Community Support
- **GitHub Issues:** Report migration problems and get assistance
- **Discussion Forums:** Community-driven migration experiences and solutions
- **Sample Implementations:** Reference implementations for common patterns
- **Migration Workshops:** Scheduled community training sessions

#### Professional Support
- **Migration Consulting:** Expert assistance for complex legacy systems
- **Custom Implementation:** Protocol-compliant component development
- **Performance Optimization:** Specialized tuning for demanding scenarios
- **Training Programs:** Team education on modular architecture patterns

### Migration Success Metrics

Track your migration progress with these key indicators:

#### Technical Metrics
- ✅ **Protocol Compliance:** All custom components implement required protocols
- ✅ **Performance Targets:** Step execution <10ms for realistic scenarios
- ✅ **Configuration Modularity:** Environment setup through component selection
- ✅ **API Modernization:** Gymnasium 0.29.x API usage throughout codebase

#### Workflow Metrics  
- ✅ **Development Velocity:** Faster experiment setup through configuration
- ✅ **Research Flexibility:** Easy switching between plume physics models
- ✅ **Code Reusability:** Shared components across different experiments
- ✅ **Maintenance Efficiency:** Reduced code duplication and cleaner abstractions

#### Research Impact Metrics
- ✅ **Experiment Reproducibility:** Deterministic results with seed management
- ✅ **Algorithm Comparability:** Consistent evaluation across navigation strategies
- ✅ **Publication Quality:** Rigorous experimental methodology support
- ✅ **Collaboration Effectiveness:** Shared component libraries and configurations

---

## Deprecation Timeline (Gymnasium API)

### Current Status (v0.3.0)

✅ **Fully Supported:**
- Modern Gymnasium 0.29.x API
- New `PlumeNavSim-v0` environment ID
- Enhanced performance with frame caching
- Extensibility hooks for custom processing

⚠️ **Deprecated but Functional:**
- Legacy Gym 0.26 API (emits warnings)
- Legacy environment IDs (`OdorPlumeNavigation-v*`)
- 4-tuple step returns (legacy mode)
- Separate seed() method calls

### Timeline

| Version | Status | Changes |
|---------|--------|---------|
| **v0.3.0** (Current) | ✅ Full compatibility | - Gymnasium support added<br>- Compatibility shim introduced<br>- Deprecation warnings for legacy usage |
| **v0.4.0** (Q2 2024) | ⚠️ Legacy warnings increased | - More prominent deprecation warnings<br>- Performance optimizations for modern API<br>- Enhanced migration tools |
| **v0.5.0** (Q3 2024) | ⚠️ Legacy support optional | - Legacy support requires explicit flag<br>- Default to modern API only<br>- Comprehensive migration documentation |
| **v1.0.0** (Q4 2024) | ❌ Legacy support removed | - Modern Gymnasium API only<br>- Legacy compatibility shim removed<br>- Performance optimized for modern usage |

### Migration Urgency

| Usage Pattern | Urgency | Action Required |
|---------------|---------|-----------------|
| New projects | **Low** | Use modern API from start |
| Active development | **Medium** | Migrate during next sprint |
| Production systems | **Medium** | Plan migration by v0.5.0 |
| Legacy codebases | **High** | Begin migration immediately |

### Preparing for v1.0.0

To ensure smooth transition to v1.0.0:

1. **Update dependencies:** Install Gymnasium ≥0.29.0
2. **Test compatibility:** Run validation tools on your codebase
3. **Migrate gradually:** Use compatibility shim during transition
4. **Update CI/CD:** Test with both legacy and modern APIs
5. **Train team:** Ensure developers understand new API patterns

### Getting Help

- **Documentation:** [Gymnasium Documentation](https://gymnasium.farama.org/)
- **Migration Tools:** Built-in compatibility validation and diagnostics
- **Community Support:** GitHub issues and discussions
- **Professional Support:** Available through support channels

---

## Summary

The plume_nav_sim v0.4.0 migration encompasses two major transformations: the architectural evolution from monolithic VideoPlume to a modular, extensible framework, and the API modernization from OpenAI Gym 0.26 to Gymnasium 0.29.x. These changes provide unprecedented research flexibility, performance optimization, and future-proofing for odor navigation research.

### Key Architectural Achievements

**🔧 Modular Component Architecture:**
- **Pluggable plume models:** Switch between Gaussian, Turbulent, and Video-based physics via configuration
- **Configurable sensing:** Replace hard-coded sampling with flexible SensorProtocol implementations  
- **Agent-agnostic design:** Support both reactive and memory-based navigation strategies uniformly
- **Wind field integration:** Realistic environmental dynamics with configurable complexity
- **Protocol-based extensibility:** Add custom components without modifying core simulation logic

**⚡ Performance & Scalability:**
- **Sub-10ms step execution:** Maintained across architectural complexity levels
- **Vectorized operations:** Efficient multi-agent scenarios with 100+ concurrent agents
- **Memory efficiency:** Optimized resource usage with configurable caching strategies
- **Real-time compatibility:** 30+ FPS simulation support for interactive visualization

**🔬 Research Enablement:**
- **Configuration-driven experiments:** Model selection and parameter tuning without code changes
- **Reproducible science:** Deterministic simulation with comprehensive seed management
- **Comparative studies:** Consistent evaluation framework across navigation algorithms
- **Extensible methodology:** Protocol compliance ensures algorithmic interoperability

### Migration Benefits

**✅ Backward Compatibility Preserved:**
- **Existing VideoPlume workflows** continue functioning through VideoPlumeAdapter wrapper
- **Legacy Gym API patterns** supported through automatic compatibility detection  
- **Gradual migration paths** minimize disruption to ongoing research projects
- **Performance maintained or improved** across all compatibility modes

**🚀 Enhanced Research Capabilities:**
- **Rapid prototyping:** Fast analytical models for algorithm development and testing
- **Realistic validation:** Complex turbulent physics for publication-quality research
- **Multi-modal sensing:** Binary detection, concentration measurement, gradient computation
- **Cognitive modeling:** Optional memory interfaces supporting planning and learning agents

**📊 Improved Workflow Efficiency:**
- **Reduced code duplication:** Shared protocol implementations across experiments
- **Faster iteration cycles:** Configuration-based parameter sweeps and model comparisons
- **Enhanced debugging:** Component-level performance monitoring and validation tools
- **Collaborative development:** Standardized interfaces enabling team coordination

### Migration Recommendations

**For New Projects:**
- Start with modular architecture and Gymnasium API from day one
- Use Hydra configuration system for flexible experiment management
- Adopt sensor-based perception for future extensibility
- Leverage protocol-based design for algorithmic modularity

**For Existing Projects:**
- Begin with API modernization (Gymnasium 0.29.x) while maintaining current architecture
- Gradually migrate sensing logic from hard-coded to SensorProtocol-based implementations
- Convert VideoPlume usage to VideoPlumeAdapter for improved modularity
- Plan full architectural migration for next major experimental campaign

**For Production Systems:**
- Implement comprehensive testing with both legacy and modern API patterns
- Validate performance benchmarks across migration phases  
- Plan staged deployment with rollback capabilities
- Complete migration by v0.6.0 to avoid legacy support complications

### Timeline Commitment

The migration timeline provides structured transition support with clear milestones:

- **v0.4.0 (Current):** Full modular support with backward compatibility
- **v0.5.0 (Q2 2024):** Enhanced validation and migration assistance tools
- **v0.6.0 (Q3 2024):** Legacy patterns require explicit enablement
- **v1.0.0 (Q4 2024):** Pure modular architecture with optimized performance

### Getting Started

**Immediate Actions:**
1. **Install Gymnasium:** `pip install "gymnasium>=0.29.0"`
2. **Update imports:** Use `import gymnasium as gym` in new code
3. **Try modular configs:** Experiment with `plume_model=gaussian` parameter switching
4. **Validate existing code:** Run compatibility checking tools on current implementations

**Next Steps:**
1. **Adopt sensor abstractions:** Replace direct odor sampling with SensorProtocol implementations
2. **Implement protocol compliance:** Ensure custom components satisfy interface requirements
3. **Configure component integration:** Set up modular Hydra configurations for experiments
4. **Optimize performance:** Use profiling tools to maintain <10ms step execution targets

**Long-term Planning:**
1. **Design protocol-compliant algorithms:** Build navigation strategies using standard interfaces
2. **Develop component libraries:** Create reusable implementations for common patterns
3. **Establish testing frameworks:** Validate component interactions and performance requirements
4. **Document custom extensions:** Ensure protocol compliance for sharing and collaboration

The comprehensive migration support, extensive documentation, and community resources ensure a smooth transition to the enhanced plume navigation simulation framework. The modular architecture and modern API provide a robust foundation for cutting-edge olfactory navigation research with unprecedented flexibility and performance.

For detailed technical guidance, protocol specifications, and additional migration resources, consult the complete project documentation and community support channels.