# Odor Plume Navigation Library

A reusable Python library for simulating agent navigation through odor plumes with sophisticated Hydra-based configuration management, designed for integration with Kedro pipelines, reinforcement learning frameworks, and machine learning/neural network analyses.

## Overview

The Odor Plume Navigation library provides a comprehensive toolkit for research-grade simulation of how agents navigate through odor plumes. Designed as an importable library, it offers clean APIs, modular architecture, and enterprise-grade configuration management for seamless integration into research workflows.

### Key Features

- **Reusable Library Architecture**: Import and use in any Python project
- **Standardized RL Integration**: Gymnasium API compliance for seamless integration with stable-baselines3 and modern RL frameworks
- **Hydra Configuration Management**: Sophisticated hierarchical configuration with environment variable integration
- **Multi-Framework Integration**: Compatible with Kedro, RL frameworks, and ML/neural network analyses
- **CLI Interface**: Command-line tools for automation, batch processing, and RL training workflows
- **Docker-Ready**: Containerized development and deployment environments
- **Dual Workflow Support**: Poetry and pip installation methods
- **Research-Grade Quality**: Type-safe, well-documented, and thoroughly tested

## Installation

### Prerequisites

- Python 3.9 or higher
- Poetry (recommended) or pip for dependency management
- Docker and docker-compose (optional, for containerized development)

### Installation Methods

#### Poetry Installation (Recommended)

```bash
# Install from PyPI
poetry add odor_plume_nav

# For development with all optional dependencies
poetry add odor_plume_nav --group dev,docs,viz

# For reinforcement learning capabilities
poetry add odor_plume_nav --group rl
```

#### Pip Installation

```bash
# Standard installation
pip install odor_plume_nav

# Development installation with optional dependencies
pip install "odor_plume_nav[dev,docs,viz]"

# Installation with reinforcement learning dependencies
pip install "odor_plume_nav[rl]"

# Full installation with all optional dependencies
pip install "odor_plume_nav[dev,docs,viz,rl]"
```

#### Development Installation

```bash
# Clone repository
git clone https://github.com/organization/odor_plume_nav.git
cd odor_plume_nav

# Poetry development setup (recommended)
poetry install --with dev,docs,viz,rl
poetry shell

# Alternative: pip development setup
pip install -e ".[dev,docs,viz,rl]"
```

#### Docker-Based Development Environment

```bash
# Full development environment with database and pgAdmin
docker-compose up --build

# Library container only
docker build -t odor_plume_nav .
docker run -it odor_plume_nav
```

## Library Usage Patterns

### For Kedro Projects

```python
from odor_plume_nav import Navigator, VideoPlume
from odor_plume_nav.config import NavigatorConfig
from hydra import compose, initialize

# Kedro pipeline integration
def create_navigation_pipeline():
    with initialize(config_path="../conf", version_base=None):
        cfg = compose(config_name="config")
        
        # Create components using Hydra configuration
        navigator = Navigator.from_config(cfg.navigator)
        video_plume = VideoPlume.from_config(cfg.video_plume)
        
        return navigator, video_plume

# Use in Kedro nodes
def navigation_node(navigator: Navigator, video_plume: VideoPlume) -> dict:
    """Kedro node for odor plume navigation simulation."""
    results = navigator.simulate(video_plume, duration=cfg.simulation.max_duration)
    return {"trajectory": results.trajectory, "sensor_data": results.sensor_data}
```

### For Reinforcement Learning Projects

#### Using the Gymnasium Environment

```python
from odor_plume_nav.api.navigation import create_gymnasium_environment
from odor_plume_nav.utils import set_global_seed
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym

# Create Gymnasium-compliant environment
def create_rl_environment(config_path: str = "conf/config.yaml"):
    """Create a Gymnasium environment for RL training."""
    
    # Set deterministic behavior for RL training
    set_global_seed(42)
    
    # Create environment using the factory function
    env = create_gymnasium_environment(config_path)
    
    # Optional: wrap with additional preprocessors
    # env = NormalizeObservation(env)
    # env = ClipAction(env)
    
    return env

# Train with stable-baselines3
def train_rl_agent():
    """Train an RL agent using PPO algorithm."""
    
    # Create environment
    env = create_rl_environment()
    
    # Verify environment compatibility
    from gymnasium.utils.env_checker import check_env
    check_env(env)
    
    # Create vectorized environment for training
    vec_env = DummyVecEnv([lambda: env])
    
    # Initialize PPO agent
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        verbose=1
    )
    
    # Train the model
    model.learn(total_timesteps=100000)
    
    # Save trained model
    model.save("ppo_plume_navigation")
    
    return model

# Evaluate trained agent
def evaluate_agent(model_path: str = "ppo_plume_navigation"):
    """Evaluate a trained RL agent."""
    
    env = create_rl_environment()
    model = PPO.load(model_path)
    
    obs, _ = env.reset()
    total_reward = 0
    
    for _ in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            break
    
    print(f"Total reward: {total_reward}")
    return total_reward
```

#### Advanced RL Integration

```python
from odor_plume_nav.environments.gymnasium_env import OdorPlumeNavigationEnv
from odor_plume_nav.environments.wrappers import NormalizeObservation, RewardShaping
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

# Custom environment configuration
def create_advanced_rl_setup():
    """Create advanced RL training setup with custom wrappers."""
    
    # Create base environment
    base_env = OdorPlumeNavigationEnv(
        video_path="data/complex_plume.mp4",
        max_episode_steps=1000,
        reward_shaping={'odor_weight': 1.0, 'distance_weight': 0.5, 'control_penalty': 0.1}
    )
    
    # Apply preprocessing wrappers
    env = NormalizeObservation(base_env)
    env = RewardShaping(env, dense_rewards=True)
    
    return env

# Training with advanced callbacks
def train_advanced_agent():
    """Train agent with evaluation callbacks and checkpointing."""
    
    # Training and evaluation environments
    train_env = DummyVecEnv([create_advanced_rl_setup for _ in range(4)])
    eval_env = create_advanced_rl_setup()
    
    # Create SAC agent for continuous control
    model = SAC(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        buffer_size=1000000,
        learning_starts=10000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        verbose=1
    )
    
    # Setup callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./logs/",
        log_path="./logs/",
        eval_freq=10000,
        deterministic=True,
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path="./checkpoints/",
        name_prefix="sac_plume_nav"
    )
    
    # Train with callbacks
    model.learn(
        total_timesteps=500000,
        callback=[eval_callback, checkpoint_callback]
    )
    
    return model
```

### Legacy RL Environment (Pre-Gymnasium)

```python
from odor_plume_nav.core import NavigatorProtocol
from odor_plume_nav.api import create_navigator
from odor_plume_nav.utils import set_global_seed

# Legacy RL environment integration (for migration reference)
class OdorPlumeRLEnv(gym.Env):
    def __init__(self, config_path: str = "conf/config.yaml"):
        super().__init__()
        
        # Set deterministic behavior for RL training
        set_global_seed(42)
        
        # Create navigator from configuration
        self.navigator = create_navigator(config_path)
        self.video_plume = VideoPlume.from_config(config_path)
        
        # Define RL action and observation spaces
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(64, 64, 1), dtype=np.uint8
        )
    
    def step(self, action):
        # Execute action and get next state
        self.navigator.update(action)
        observation = self.video_plume.get_sensor_reading(self.navigator.position)
        reward = self._calculate_reward()
        done = self._check_termination()
        return observation, reward, done, {}
```

### For ML/Neural Network Analyses

```python
from odor_plume_nav.utils import set_global_seed
from odor_plume_nav.data import VideoPlume
from odor_plume_nav.api.navigation import run_plume_simulation
import torch
import numpy as np

# Neural network training data generation
def generate_training_data(num_episodes: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    """Generate training data for neural navigation models."""
    
    # Set reproducible seeds for ML workflows
    set_global_seed(42)
    
    # Load configuration with ML-optimized parameters
    with initialize(config_path="../conf"):
        cfg = compose(config_name="config", overrides=[
            "simulation.recording.export_format=numpy",
            "performance.numpy.precision=float32"
        ])
    
    # Generate diverse navigation scenarios
    trajectories = []
    sensor_readings = []
    
    for episode in range(num_episodes):
        # Create randomized navigator for data diversity
        navigator = Navigator.from_config(cfg.navigator)
        navigator.position = np.random.uniform(0, 100, 2)
        
        # Run simulation
        results = run_plume_simulation(navigator, video_plume, cfg)
        
        trajectories.append(results.trajectory)
        sensor_readings.append(results.sensor_data)
    
    return np.array(trajectories), np.array(sensor_readings)

# PyTorch dataset integration
class NavigationDataset(torch.utils.data.Dataset):
    def __init__(self, config_path: str = "conf/config.yaml"):
        self.trajectories, self.sensor_data = generate_training_data()
    
    def __len__(self):
        return len(self.trajectories)
    
    def __getitem__(self, idx):
        return {
            'trajectory': torch.tensor(self.trajectories[idx], dtype=torch.float32),
            'sensor_reading': torch.tensor(self.sensor_data[idx], dtype=torch.float32)
        }
```

## Reinforcement Learning

### Overview

The Odor Plume Navigation library provides comprehensive reinforcement learning integration through the Gymnasium API, enabling seamless compatibility with modern RL frameworks and algorithms. The RL integration transforms the existing simulation framework into a standardized RL environment while preserving all core navigation, sensing, and visualization capabilities.

### Gymnasium Environment

#### Environment Interface

The `OdorPlumeNavigationEnv` class provides a fully Gymnasium-compliant environment implementing the standard `reset()`, `step()`, `render()`, and `close()` methods:

```python
from odor_plume_nav.environments.gymnasium_env import OdorPlumeNavigationEnv
import gymnasium as gym

# Create environment instance
env = OdorPlumeNavigationEnv(
    video_path="data/plume_video.mp4",
    max_episode_steps=1000,
    render_mode="human"  # or "rgb_array" for headless
)

# Standard Gymnasium workflow
observation, info = env.reset(seed=42)
for step in range(1000):
    action = env.action_space.sample()  # Random action
    observation, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        observation, info = env.reset()

env.close()
```

#### Action and Observation Spaces

**Action Space**: `gymnasium.spaces.Box`
- **Shape**: `(2,)` - Continuous control vector
- **Range**: `[-1.0, 1.0]` for both dimensions
- **Components**:
  - `action[0]`: Linear speed control (normalized)
  - `action[1]`: Angular velocity control (normalized)

**Observation Space**: `gymnasium.spaces.Dict`
- **odor_concentration**: `Box(shape=(1,), dtype=float32)` - Current sensor reading
- **agent_position**: `Box(shape=(2,), dtype=float32)` - Agent [x, y] coordinates
- **agent_orientation**: `Box(shape=(1,), dtype=float32)` - Agent heading in radians
- **plume_gradient**: `Box(shape=(2,), dtype=float32, low=-1.0, high=1.0)` - Estimated gradient direction (optional)

```python
# Access observation components
obs, info = env.reset()
odor_value = obs['odor_concentration'][0]
agent_x, agent_y = obs['agent_position']
agent_heading = obs['agent_orientation'][0]
```

#### Environment Factory Function

```python
from odor_plume_nav.api.navigation import create_gymnasium_environment
from hydra import compose, initialize

# Create environment from Hydra configuration
with initialize(config_path="../conf", version_base=None):
    cfg = compose(config_name="config")
    env = create_gymnasium_environment(cfg)

# Create environment with parameter overrides
env = create_gymnasium_environment(
    cfg,
    max_episode_steps=2000,
    reward_shaping={'odor_weight': 2.0, 'efficiency_bonus': 0.5}
)

# Environment validation
from gymnasium.utils.env_checker import check_env
check_env(env)  # Verify Gymnasium API compliance
```

### RL Training Workflows

#### Command-Line Training Interface

The library provides comprehensive CLI commands for RL training workflows:

```bash
# Basic PPO training
plume-nav-sim train --algorithm PPO

# SAC training with custom parameters
plume-nav-sim train --algorithm SAC \
    --total-timesteps 500000 \
    --learning-rate 3e-4 \
    --batch-size 256

# Multi-environment parallel training
plume-nav-sim train --algorithm PPO \
    --n-envs 8 \
    --env-parallel \
    --total-timesteps 1000000

# Training with checkpointing and evaluation
plume-nav-sim train --algorithm SAC \
    --checkpoint-dir ./checkpoints \
    --save-freq 50000 \
    --eval-freq 10000 \
    --eval-episodes 20

# Custom configuration with parameter overrides
plume-nav-sim train --algorithm PPO \
    navigator.max_speed=15.0 \
    simulation.max_episode_steps=2000 \
    rl.learning_rate=1e-4 \
    --config-name rl_config
```

#### Algorithm Support

**Proximal Policy Optimization (PPO)**
- **Use Case**: Robust policy learning with good sample efficiency
- **Best For**: Continuous control tasks with stable learning
- **Default Hyperparameters**: Optimized for odor plume navigation

```bash
# PPO with custom hyperparameters
plume-nav-sim train --algorithm PPO \
    --learning-rate 3e-4 \
    --n-steps 2048 \
    --batch-size 64 \
    --n-epochs 10 \
    --clip-range 0.2
```

**Soft Actor-Critic (SAC)**
- **Use Case**: Sample-efficient learning for continuous control
- **Best For**: Exploration-heavy environments with sparse rewards
- **Features**: Off-policy learning with entropy regularization

```bash
# SAC with custom configuration
plume-nav-sim train --algorithm SAC \
    --learning-rate 3e-4 \
    --buffer-size 1000000 \
    --batch-size 256 \
    --learning-starts 10000
```

**Twin Delayed DDPG (TD3)**
- **Use Case**: Deterministic policy learning with reduced overestimation
- **Best For**: High-dimensional continuous control tasks

```bash
# TD3 training
plume-nav-sim train --algorithm TD3 \
    --learning-rate 1e-3 \
    --batch-size 256 \
    --policy-delay 2
```

#### Training Progress Monitoring

```bash
# Training with comprehensive logging
plume-nav-sim train --algorithm PPO \
    --tensorboard-log ./tensorboard_logs \
    --verbose 1 \
    --progress-bar

# Training with custom evaluation metrics
plume-nav-sim train --algorithm SAC \
    --eval-freq 10000 \
    --eval-episodes 20 \
    --eval-log-path ./evaluation_logs
```

### Environment Wrappers

#### Available Wrappers

```python
from odor_plume_nav.environments.wrappers import (
    NormalizeObservation,
    RewardShaping,
    FrameStack,
    ActionClipping
)

# Apply normalization wrapper
env = NormalizeObservation(base_env, epsilon=1e-8)

# Add reward shaping
env = RewardShaping(
    env,
    odor_weight=1.0,
    distance_weight=0.5,
    control_penalty=0.1,
    efficiency_bonus=0.2
)

# Frame stacking for temporal awareness
env = FrameStack(env, num_stack=4)

# Action space clipping for safety
env = ActionClipping(env, min_action=-0.8, max_action=0.8)
```

#### Custom Wrapper Development

```python
import gymnasium as gym
from gymnasium.wrappers import Wrapper

class CustomPlumeWrapper(Wrapper):
    """Custom wrapper for specialized preprocessing."""
    
    def __init__(self, env, custom_parameter=1.0):
        super().__init__(env)
        self.custom_parameter = custom_parameter
    
    def step(self, action):
        # Custom action preprocessing
        modified_action = self.preprocess_action(action)
        obs, reward, terminated, truncated, info = self.env.step(modified_action)
        
        # Custom observation/reward postprocessing
        modified_obs = self.postprocess_observation(obs)
        modified_reward = self.shape_reward(reward, obs)
        
        return modified_obs, modified_reward, terminated, truncated, info
    
    def preprocess_action(self, action):
        """Apply custom action transformations."""
        return action * self.custom_parameter
    
    def postprocess_observation(self, obs):
        """Apply custom observation transformations."""
        return obs
    
    def shape_reward(self, reward, obs):
        """Apply custom reward shaping."""
        return reward + self.compute_bonus(obs)
```

## Migration Guide: Gym to Gymnasium API

### Overview

The Odor Plume Navigation library has been refactored to support the modern Gymnasium API while maintaining full backward compatibility with existing gym-based code. This migration guide helps you understand the changes and provides clear paths for updating your code.

### API Compatibility Matrix

| API Version | Step Return Signature | Environment ID | Import Statement |
|-------------|----------------------|----------------|-----------------|
| **Legacy Gym** | `(obs, reward, done, info)` | `OdorPlumeNavigation-v1` | `import gym` |
| **New Gymnasium** | `(obs, reward, terminated, truncated, info)` | `PlumeNavSim-v0` | `import gymnasium` |

### Dual API Support

The library automatically detects which API you're using and returns the appropriate tuple format:

#### Legacy Gym API (4-tuple) - Maintained for Backward Compatibility

```python
import gym
from odor_plume_nav.environments import register_environments

# Register environments for gym usage
register_environments()

# Legacy gym usage continues to work unchanged
env = gym.make('OdorPlumeNavigation-v1')
obs = env.reset()

for step in range(1000):
    action = env.action_space.sample()
    # Returns 4-tuple: (obs, reward, done, info)
    obs, reward, done, info = env.step(action)
    
    if done:
        obs = env.reset()
        
env.close()
```

#### New Gymnasium API (5-tuple) - Recommended for New Projects

```python
import gymnasium as gym
from odor_plume_nav.environments import register_environments

# Register environments for gymnasium usage
register_environments()

# New gymnasium environment with enhanced API
env = gym.make('PlumeNavSim-v0')
obs, info = env.reset(seed=42)

for step in range(1000):
    action = env.action_space.sample()
    # Returns 5-tuple: (obs, reward, terminated, truncated, info)
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        obs, info = env.reset()
        
env.close()
```

### Environment Registration Details

#### New PlumeNavSim-v0 Environment

The new `PlumeNavSim-v0` environment provides enhanced features and full Gymnasium compliance:

**Key Features:**
- **Gymnasium 0.29.x API compliance** with 5-tuple step returns
- **Enhanced observation space** with structured dictionary observations
- **Improved reward shaping** with separated termination conditions
- **Seed support** for deterministic episode initialization
- **Performance optimizations** maintaining ≤10ms step() execution time

**Environment Specification:**
```python
from odor_plume_nav.api.navigation import create_gymnasium_environment
import gymnasium as gym

# Create environment with factory function
env = create_gymnasium_environment(config_path="conf/config.yaml")

# Or use gymnasium.make() with registration
env = gym.make('PlumeNavSim-v0', 
               max_episode_steps=1000,
               render_mode="human")

# Environment information
print(f"Action space: {env.action_space}")
print(f"Observation space: {env.observation_space}")
```

**Action Space:**
```python
# Continuous control: Box(2,) with range [-1.0, 1.0]
action_space = gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
# action[0]: Linear velocity control
# action[1]: Angular velocity control
```

**Observation Space:**
```python
# Dictionary observation space with multiple components
observation_space = gymnasium.spaces.Dict({
    'odor_concentration': gymnasium.spaces.Box(
        shape=(1,), low=0.0, high=1.0, dtype=np.float32
    ),
    'agent_position': gymnasium.spaces.Box(
        shape=(2,), low=0.0, high=100.0, dtype=np.float32
    ),
    'agent_orientation': gymnasium.spaces.Box(
        shape=(1,), low=-np.pi, high=np.pi, dtype=np.float32
    ),
    'plume_gradient': gymnasium.spaces.Box(
        shape=(2,), low=-1.0, high=1.0, dtype=np.float32
    )
})
```

### Migration Strategies

#### Strategy 1: Gradual Migration (Recommended)

Maintain both APIs during transition period:

```python
# Phase 1: Test new API alongside existing code
def create_environment(use_gymnasium=False):
    if use_gymnasium:
        import gymnasium as gym
        env = gym.make('PlumeNavSim-v0')
        return env, "gymnasium"
    else:
        import gym
        env = gym.make('OdorPlumeNavigation-v1')
        return env, "gym"

# Phase 2: Validate consistency between APIs
def validate_api_consistency():
    gym_env, _ = create_environment(use_gymnasium=False)
    gymnasium_env, _ = create_environment(use_gymnasium=True)
    
    # Compare observation and action spaces
    assert gym_env.action_space == gymnasium_env.action_space
    # Note: observation spaces may differ due to enhanced features
    
    gym_env.close()
    gymnasium_env.close()

# Phase 3: Switch to Gymnasium for new development
env, api_type = create_environment(use_gymnasium=True)
obs, info = env.reset(seed=42)

for step in range(1000):
    action = env.action_space.sample()
    if api_type == "gymnasium":
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    else:
        obs, reward, done, info = env.step(action)
    
    if done:
        if api_type == "gymnasium":
            obs, info = env.reset()
        else:
            obs = env.reset()
```

#### Strategy 2: Direct Migration

For new projects or major refactoring:

```python
# Before: Legacy gym implementation
"""
import gym
env = gym.make('OdorPlumeNavigation-v1')
obs = env.reset()
obs, reward, done, info = env.step(action)
"""

# After: Modern gymnasium implementation
import gymnasium as gym
env = gym.make('PlumeNavSim-v0')
obs, info = env.reset(seed=42)
obs, reward, terminated, truncated, info = env.step(action)
done = terminated or truncated  # Convert to legacy done flag if needed
```

### Stable-Baselines3 Integration Updates

#### Legacy Integration
```python
from stable_baselines3 import PPO
import gym

env = gym.make('OdorPlumeNavigation-v1')
model = PPO("MlpPolicy", env)
model.learn(total_timesteps=100000)
```

#### Updated Gymnasium Integration
```python
from stable_baselines3 import PPO
import gymnasium as gym

env = gym.make('PlumeNavSim-v0')
model = PPO("MlpPolicy", env)
model.learn(total_timesteps=100000)
```

### Troubleshooting Common Migration Issues

#### Issue 1: Import Errors
```python
# Problem: ModuleNotFoundError for gymnasium
# Solution: Install RL dependencies
pip install "odor_plume_nav[rl]"
# or
poetry add "odor_plume_nav[rl]"
```

#### Issue 2: Step Return Tuple Length
```python
# Problem: Unpacking wrong number of values
# Solution: Use detection wrapper
def step_with_compatibility(env, action):
    result = env.step(action)
    if len(result) == 4:
        # Legacy gym API
        obs, reward, done, info = result
        return obs, reward, done, False, info  # Convert to 5-tuple
    else:
        # Modern gymnasium API
        return result

# Usage
obs, reward, terminated, truncated, info = step_with_compatibility(env, action)
done = terminated or truncated
```

#### Issue 3: Reset Method Signature
```python
# Problem: Different reset signatures
# Solution: Use compatibility wrapper
def reset_with_compatibility(env, seed=None):
    if hasattr(env, 'seed') and seed is not None:
        env.seed(seed)
        return env.reset()
    else:
        # Modern gymnasium reset with seed parameter
        if seed is not None:
            return env.reset(seed=seed)
        else:
            return env.reset()

# Usage works with both APIs
result = reset_with_compatibility(env, seed=42)
if isinstance(result, tuple):
    obs, info = result  # Gymnasium
else:
    obs = result  # Legacy gym
    info = {}
```

### Deprecation Warnings

When using legacy gym imports, you'll see helpful deprecation warnings:

```python
import gym
# UserWarning: You are using the legacy gym API. Consider migrating to gymnasium
# for enhanced features and future compatibility. See migration guide in README.md

env = gym.make('OdorPlumeNavigation-v1')
# UserWarning: Environment 'OdorPlumeNavigation-v1' is deprecated. 
# Use 'PlumeNavSim-v0' with gymnasium for new features and improvements.
```

### Migration from Legacy Simulation API

#### Migration Guide: Simulation to Gymnasium

**Legacy Simulation Approach:**
```python
# Old approach - direct simulation execution
from odor_plume_nav.api.navigation import (
    create_navigator, create_video_plume, run_plume_simulation
)

navigator = create_navigator(cfg.navigator)
video_plume = create_video_plume(cfg.video_plume)
positions, orientations, readings = run_plume_simulation(
    navigator, video_plume, cfg.simulation
)
```

**New Gymnasium Approach:**
```python
# New approach - Gymnasium environment
from odor_plume_nav.api.navigation import create_gymnasium_environment

env = create_gymnasium_environment(cfg)
obs, info = env.reset(seed=42)

positions, orientations, readings = [], [], []
for step in range(1000):
    action = your_policy(obs)  # Your control policy
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Extract data for analysis
    positions.append(obs['agent_position'])
    orientations.append(obs['agent_orientation'])
    readings.append(obs['odor_concentration'])
    
    if terminated or truncated:
        break
```

#### Backward Compatibility

The library maintains full backward compatibility with existing simulation APIs:

```python
# Legacy factory functions still available
from odor_plume_nav.api.navigation import (
    create_navigator_from_config,  # Legacy alias
    create_video_plume_from_config,  # Legacy alias
    run_simulation  # Legacy alias
)

# Original API continues to work unchanged
navigator = create_navigator_from_config(cfg.navigator)
video_plume = create_video_plume_from_config(cfg.video_plume)
results = run_simulation(navigator, video_plume, cfg.simulation)
```

#### Migration Strategies

**Gradual Migration:**
```python
# Phase 1: Use both APIs in parallel
legacy_results = run_plume_simulation(navigator, video_plume, cfg)
gymnasium_env = create_gymnasium_environment(cfg)

# Phase 2: Validate consistency
assert verify_results_consistency(legacy_results, gymnasium_env)

# Phase 3: Switch to Gymnasium for new features
model = PPO("MlpPolicy", gymnasium_env)
model.learn(total_timesteps=100000)
```

**Configuration Migration:**
```yaml
# Legacy configuration (still supported)
navigator:
  position: [50.0, 50.0]
  orientation: 45.0
  max_speed: 10.0

# Enhanced RL configuration
rl:
  environment:
    max_episode_steps: 1000
    reward_shaping:
      odor_weight: 1.0
      distance_weight: 0.5
  training:
    algorithm: "PPO"
    total_timesteps: 500000
    learning_rate: 3e-4
```

### Best Practices and Examples

#### Training Configuration Templates

**PPO Configuration for Quick Prototyping:**
```python
from stable_baselines3 import PPO

def quick_ppo_training():
    env = create_gymnasium_environment()
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        verbose=1
    )
    model.learn(total_timesteps=100000)
    return model
```

**SAC Configuration for Sample Efficiency:**
```python
from stable_baselines3 import SAC

def efficient_sac_training():
    env = create_gymnasium_environment()
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=1000000,
        batch_size=256,
        learning_starts=1000,
        train_freq=1,
        verbose=1
    )
    model.learn(total_timesteps=200000)
    return model
```

#### Hyperparameter Optimization

```python
import optuna
from stable_baselines3 import PPO

def optimize_hyperparameters(trial):
    """Optuna hyperparameter optimization for RL training."""
    
    # Suggest hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    n_steps = trial.suggest_categorical("n_steps", [1024, 2048, 4096])
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    
    # Create environment and model
    env = create_gymnasium_environment()
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        verbose=0
    )
    
    # Train and evaluate
    model.learn(total_timesteps=50000)
    
    # Evaluation metric
    mean_reward = evaluate_policy(model, env, n_eval_episodes=10)[0]
    return mean_reward

# Run optimization
study = optuna.create_study(direction="maximize")
study.optimize(optimize_hyperparameters, n_trials=50)
```

#### Multi-Environment Training

```python
from stable_baselines3.common.vec_env import SubprocVecEnv

def create_vectorized_env(n_envs=4):
    """Create vectorized environment for parallel training."""
    
    def make_env():
        env = create_gymnasium_environment()
        # Add environment-specific wrappers here
        return env
    
    return SubprocVecEnv([make_env for _ in range(n_envs)])

# Training with vectorized environments
def parallel_training():
    vec_env = create_vectorized_env(n_envs=8)
    model = PPO("MlpPolicy", vec_env, verbose=1)
    model.learn(total_timesteps=1000000)
    return model
```

## Structured Logging with Loguru

### Overview

The library features a modern structured logging system built on Loguru, replacing traditional print statements and basic logging with comprehensive JSON-formatted logs, correlation IDs, and performance monitoring integration.

### Basic Logging Configuration

#### Default Setup

The logging system is automatically configured when importing the library:

```python
from odor_plume_nav.utils.logging_setup import setup_logging
from loguru import logger

# Automatic setup with default configuration
setup_logging()

# Use structured logging throughout your code
logger.info("Starting odor plume navigation simulation")
logger.debug("Navigator initialized", position=[10.0, 15.0], orientation=45.0)
```

#### Custom Logging Configuration

```python
from odor_plume_nav.utils.logging_setup import setup_logging
from loguru import logger

# Custom logging configuration
setup_logging(
    level="DEBUG",
    format_type="json",
    file_rotation="50 MB",
    retention="2 weeks",
    compression="gzip",
    correlation_id=True
)

# Structured logging with context
logger.bind(
    component="navigator",
    experiment_id="exp_001",
    agent_count=5
).info("Multi-agent simulation started")
```

### Configuration Examples

#### Development Environment Configuration

```python
# conf/config.yaml - Development logging setup
logging:
  # Core logging configuration
  level: "DEBUG"
  console_enabled: true
  file_enabled: true
  
  # Loguru-specific configuration
  loguru:
    format: "json"  # Options: text, json
    colorize: true  # Colorize console output
    diagnose: true  # Include detailed exception information
    
    # File logging configuration
    file:
      path: "logs/odor_plume_nav_{time:YYYY-MM-DD}.log"
      rotation: "10 MB"
      retention: "1 week"
      compression: "gzip"
      level: "DEBUG"
    
    # Console logging configuration  
    console:
      level: "INFO"
      colorize: true
      format: "text"  # Human-readable for development
    
    # Performance monitoring integration
    performance:
      enabled: true
      slow_threshold: 0.033  # 33ms (30 FPS target)
      memory_tracking: true
      step_timing: true
    
    # Correlation ID tracking
    correlation:
      enabled: true
      auto_generate: true
      propagate_context: true
```

#### Production Environment Configuration

```python
# conf/local/production.yaml - Production logging setup
logging:
  level: "INFO"
  console_enabled: false  # Reduce noise in production
  file_enabled: true
  
  loguru:
    format: "json"  # Structured logs for analysis
    colorize: false
    diagnose: false  # Security: don't expose stack traces
    
    # Centralized logging configuration
    file:
      path: "/var/log/odor_plume_nav/app_{time:YYYY-MM-DD}.log"
      rotation: "100 MB"
      retention: "30 days"
      compression: "gzip"
      level: "INFO"
    
    # Syslog integration for centralized logging
    syslog:
      enabled: true
      address: "localhost"
      port: 514
      facility: "local0"
      format: "json"
    
    # Performance monitoring for production
    performance:
      enabled: true
      slow_threshold: 0.010  # Stricter production threshold
      memory_tracking: false  # Reduce overhead
      alert_on_degradation: true
    
    # Request correlation tracking
    correlation:
      enabled: true
      header_name: "X-Correlation-ID"
      propagate_downstream: true
```

### Usage Patterns

#### Basic Structured Logging

```python
from loguru import logger

# Simple structured logging
logger.info("Simulation started")
logger.debug("Configuration loaded", config_path="conf/config.yaml")
logger.warning("Performance degradation detected", fps=25.5, threshold=30.0)
logger.error("Navigation failed", error="obstacle_collision", position=[15.2, 22.1])

# Exception logging with context
try:
    navigator.update_position(invalid_position)
except ValueError as e:
    logger.exception("Invalid position update", 
                    position=invalid_position, 
                    component="navigator")
```

#### Component-Specific Logging

```python
from loguru import logger

class Navigator:
    def __init__(self, config):
        # Component-specific logger with context binding
        self.logger = logger.bind(component="navigator", id=id(self))
        self.logger.info("Navigator initialized", config=config.dict())
    
    def update_position(self, position):
        self.logger.debug("Position update", 
                         old_position=self.position,
                         new_position=position,
                         timestamp=time.time())
        self.position = position

class VideoPlume:
    def __init__(self, video_path):
        self.logger = logger.bind(component="video_plume", video=video_path)
        self.logger.info("Video plume loaded", 
                        path=video_path,
                        frame_count=self.frame_count,
                        resolution=self.resolution)
    
    def get_frame(self, frame_idx):
        start_time = time.perf_counter()
        frame = self._load_frame(frame_idx)
        duration = time.perf_counter() - start_time
        
        self.logger.debug("Frame retrieved",
                         frame_idx=frame_idx,
                         duration_ms=duration * 1000,
                         frame_shape=frame.shape)
        return frame
```

#### Performance Monitoring Integration

```python
from loguru import logger
from odor_plume_nav.utils.performance import performance_monitor
import time

class SimulationRunner:
    def __init__(self):
        self.logger = logger.bind(component="simulation")
    
    @performance_monitor
    def step(self, action):
        """Simulation step with automatic performance logging."""
        start_time = time.perf_counter()
        
        # Execute simulation step
        obs, reward, terminated, truncated, info = self._internal_step(action)
        
        # Automatic performance logging via decorator
        duration = time.perf_counter() - start_time
        
        # Structured step logging
        self.logger.debug("Simulation step completed",
                         step=self.step_count,
                         duration_ms=duration * 1000,
                         fps=1.0 / duration if duration > 0 else float('inf'),
                         agent_position=obs['agent_position'],
                         reward=reward,
                         terminated=terminated)
        
        return obs, reward, terminated, truncated, info
```

#### Correlation ID Tracking

```python
from loguru import logger
from odor_plume_nav.utils.logging_setup import generate_correlation_id
import contextvars

# Correlation context for request tracking
correlation_id = contextvars.ContextVar('correlation_id')

def run_experiment(experiment_config):
    # Generate unique correlation ID for experiment
    corr_id = generate_correlation_id()
    correlation_id.set(corr_id)
    
    # All logging within this context includes correlation ID
    with logger.contextualize(correlation_id=corr_id, experiment=experiment_config.name):
        logger.info("Experiment started", config=experiment_config.dict())
        
        try:
            # Run simulation components
            navigator = create_navigator(experiment_config.navigator)
            results = run_simulation(navigator, experiment_config.simulation)
            
            logger.info("Experiment completed successfully", 
                       total_steps=results.step_count,
                       duration=results.duration,
                       final_reward=results.total_reward)
            
        except Exception as e:
            logger.exception("Experiment failed", 
                           error=str(e),
                           error_type=type(e).__name__)
            raise
```

### Advanced Logging Features

#### Custom Log Formatters

```python
from loguru import logger
import json

def custom_json_formatter(record):
    """Custom JSON formatter with additional metadata."""
    log_entry = {
        "timestamp": record["time"].isoformat(),
        "level": record["level"].name,
        "message": record["message"],
        "module": record["module"],
        "function": record["function"],
        "line": record["line"],
        # Add custom fields
        "service": "odor_plume_nav",
        "version": "0.2.0",
        "environment": os.getenv("ENVIRONMENT_TYPE", "development")
    }
    
    # Include extra fields from structured logging
    if record["extra"]:
        log_entry["extra"] = record["extra"]
    
    # Include exception information if present
    if record["exception"]:
        log_entry["exception"] = {
            "type": record["exception"].type.__name__,
            "value": str(record["exception"].value),
            "traceback": record["exception"].traceback.format()
        }
    
    return json.dumps(log_entry)

# Configure custom formatter
logger.add("logs/structured.log", 
          format=custom_json_formatter,
          rotation="50 MB",
          retention="2 weeks")
```

#### Log Filtering and Sampling

```python
from loguru import logger

def filter_performance_logs(record):
    """Filter out high-frequency performance logs in production."""
    if record["extra"].get("component") == "performance":
        # Only log every 100th performance record
        return record["extra"].get("step", 0) % 100 == 0
    return True

def filter_debug_in_production(record):
    """Suppress debug logs in production environment."""
    if os.getenv("ENVIRONMENT_TYPE") == "production":
        return record["level"].no >= logger.level("INFO").no
    return True

# Apply filters
logger.add("logs/filtered.log", 
          filter=lambda record: filter_performance_logs(record) and 
                               filter_debug_in_production(record))
```

#### Integration with External Logging Systems

```python
from loguru import logger
import logging

# Integration with standard Python logging
class InterceptHandler(logging.Handler):
    """Intercept standard logging and redirect to Loguru."""
    
    def emit(self, record):
        # Get corresponding Loguru level
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        
        # Find caller from where logging was called
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        
        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )

# Configure integration
logging.basicConfig(handlers=[InterceptHandler()], level=0)

# Configure third-party library logging
for library in ["matplotlib", "stable_baselines3", "gymnasium"]:
    logging.getLogger(library).handlers = [InterceptHandler()]
```

### Environment Variable Configuration

Set up logging through environment variables for deployment flexibility:

```bash
# .env file for development
LOG_LEVEL=DEBUG
LOG_FORMAT=text
LOG_FILE_ENABLED=true
LOG_CONSOLE_COLORIZE=true
LOG_CORRELATION_ENABLED=true
LOG_PERFORMANCE_TRACKING=true

# Production environment variables
export LOG_LEVEL=INFO
export LOG_FORMAT=json
export LOG_FILE_PATH=/var/log/odor_plume_nav/app.log
export LOG_ROTATION=100MB
export LOG_RETENTION=30days
export LOG_COMPRESSION=gzip
export LOG_SYSLOG_ENABLED=true
```

Use in configuration:

```python
# Automatic environment variable integration
logging:
  level: ${oc.env:LOG_LEVEL,"INFO"}
  console_enabled: ${oc.env:LOG_CONSOLE_ENABLED,"true"}
  file_enabled: ${oc.env:LOG_FILE_ENABLED,"true"}
  
  loguru:
    format: ${oc.env:LOG_FORMAT,"json"}
    colorize: ${oc.env:LOG_CONSOLE_COLORIZE,"false"}
    
    file:
      path: ${oc.env:LOG_FILE_PATH,"logs/odor_plume_nav.log"}
      rotation: ${oc.env:LOG_ROTATION,"10 MB"}
      retention: ${oc.env:LOG_RETENTION,"1 week"}
      compression: ${oc.env:LOG_COMPRESSION,"gzip"}
```

## Command-Line Interface

The library provides comprehensive CLI commands for automation and batch processing.

### Available Commands

```bash
# Run a simulation with default configuration
plume-nav-sim run

# Run with parameter overrides
plume-nav-sim run navigator.max_speed=2.0 simulation.fps=60

# Parameter sweep execution
plume-nav-sim run --multirun navigator.max_speed=1.0,2.0,3.0 video_plume.kernel_size=3,5,7

# Reinforcement learning training commands
plume-nav-sim train --algorithm PPO
plume-nav-sim train --algorithm SAC --total-timesteps 500000
plume-nav-sim train --algorithm TD3 --n-envs 4 --env-parallel

# Visualization commands
plume-nav-sim visualize --input-path outputs/experiment_results.npz
plume-nav-sim visualize --animation --save-video output.mp4

# Configuration validation
plume-nav-sim config validate
plume-nav-sim config show

# Environment setup
plume-nav-sim setup --create-dirs --init-config
```

### RL Training Commands

```bash
# Basic algorithm training
plume-nav-sim train --algorithm PPO
plume-nav-sim train --algorithm SAC
plume-nav-sim train --algorithm TD3

# Training with custom parameters
plume-nav-sim train --algorithm PPO \
    --total-timesteps 1000000 \
    --learning-rate 3e-4 \
    --n-steps 2048 \
    --batch-size 64

# Parallel training with multiple environments
plume-nav-sim train --algorithm SAC \
    --n-envs 8 \
    --env-parallel \
    --total-timesteps 2000000

# Training with checkpointing and evaluation
plume-nav-sim train --algorithm PPO \
    --checkpoint-dir ./checkpoints \
    --save-freq 50000 \
    --eval-freq 10000 \
    --eval-episodes 20 \
    --tensorboard-log ./logs

# Training with environment-specific parameters
plume-nav-sim train --algorithm SAC \
    navigator.max_speed=15.0 \
    simulation.max_episode_steps=2000 \
    rl.reward_shaping.odor_weight=2.0 \
    --config-name rl_training_config

# Advanced training options
plume-nav-sim train --algorithm PPO \
    --learning-rate 1e-4 \
    --clip-range 0.1 \
    --entropy-coef 0.01 \
    --vf-coef 0.5 \
    --max-grad-norm 0.5 \
    --gae-lambda 0.95 \
    --verbose 1 \
    --progress-bar
```

### CLI Integration Examples

```bash
# Research workflow automation
#!/bin/bash
# Multi-condition RL experiment execution
for algorithm in PPO SAC TD3; do
    for lr in 1e-4 3e-4 1e-3; do
        plume-nav-sim train \
            --algorithm $algorithm \
            --learning-rate $lr \
            --total-timesteps 500000 \
            --checkpoint-dir "./experiments/${algorithm}_lr_${lr}" \
            hydra.job.name="${algorithm}_lr_${lr}"
    done
done

# Batch evaluation of trained models
plume-nav-sim evaluate \
    --model-dir experiments/ \
    --eval-episodes 100 \
    --output-format csv \
    --metrics reward,success_rate,episode_length

# Automated hyperparameter search
plume-nav-sim train --algorithm PPO \
    --optuna-trials 50 \
    --optuna-study-name ppo_optimization \
    --optuna-db sqlite:///optimization.db
```

## Configuration System

The library uses a sophisticated Hydra-based configuration hierarchy with dataclass-based structured configs, supporting environment variable integration, parameter sweeps, and multi-environment deployment with full type safety and validation.

### Modern Structured Configuration Architecture

The refactored configuration system replaces unstructured YAML with Pydantic-validated dataclasses, providing:

- **Type Safety**: Automatic validation of all configuration parameters
- **IDE Support**: Full autocomplete and type hints in development
- **Runtime Validation**: Configuration errors caught at startup, not runtime
- **Schema Evolution**: Backward-compatible configuration upgrades
- **Documentation**: Self-documenting configuration through type annotations

### Configuration Structure

```
conf/
├── base.yaml          # Foundation defaults with dataclass annotations
├── config.yaml        # User customizations with structured config composition
├── rl/                # RL-specific structured configurations
│   ├── algorithms/    # Algorithm-specific hyperparameters with validation
│   │   ├── ppo.yaml   # PPO algorithm dataclass configuration
│   │   ├── sac.yaml   # SAC algorithm dataclass configuration
│   │   └── td3.yaml   # TD3 algorithm dataclass configuration
│   ├── environments/  # Environment configurations with type enforcement
│   │   ├── basic.yaml      # Basic environment dataclass config
│   │   ├── advanced.yaml   # Advanced environment dataclass config
│   │   └── multi_agent.yaml # Multi-agent dataclass config
│   └── training.yaml  # Training pipeline structured configuration
└── local/             # Local development with secret management
    ├── credentials.yaml.template
    ├── development.yaml
    ├── production.yaml
    └── paths.yaml.template
```

### Dataclass-Based Configuration Examples

#### Structured Configuration Models

The library defines comprehensive dataclass models with automatic validation:

```python
from dataclasses import dataclass, field
from typing import Optional, List, Union
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
import numpy as np

@dataclass
class NavigatorConfig:
    """Structured configuration for navigator parameters with validation."""
    
    # Core navigation parameters with type enforcement
    position: Optional[List[float]] = None
    orientation: float = 0.0
    speed: float = 0.0
    max_speed: float = 1.0
    angular_velocity: float = 0.0
    
    # Multi-agent configuration with structured defaults
    positions: Optional[List[List[float]]] = None
    orientations: Optional[List[float]] = None
    speeds: Optional[List[float]] = None
    max_speeds: Optional[List[float]] = None
    angular_velocities: Optional[List[float]] = None
    num_agents: Optional[int] = None
    
    # Control parameters with validation constraints
    control: ControlConfig = field(default_factory=lambda: ControlConfig())
    formation: FormationConfig = field(default_factory=lambda: FormationConfig())
    
    def __post_init__(self):
        """Post-initialization validation."""
        # Validate speed constraints
        if self.speed > self.max_speed:
            raise ValueError(f"Speed {self.speed} cannot exceed max_speed {self.max_speed}")
        
        # Validate multi-agent consistency
        if self.positions is not None:
            if self.num_agents is None:
                self.num_agents = len(self.positions)
            elif len(self.positions) != self.num_agents:
                raise ValueError("Number of positions must match num_agents")

@dataclass
class ControlConfig:
    """Control system configuration with performance constraints."""
    acceleration: float = 0.1
    turning_rate: float = 30.0
    sensor_range: float = 10.0
    sensor_noise: float = 0.0
    sensor_resolution: float = 1.0
    
    def __post_init__(self):
        """Validate control parameters."""
        if self.acceleration <= 0:
            raise ValueError("Acceleration must be positive")
        if self.sensor_range <= 0:
            raise ValueError("Sensor range must be positive")

@dataclass 
class FormationConfig:
    """Formation control configuration for multi-agent scenarios."""
    type: str = "grid"  # Options: grid, line, circle, custom
    spacing: float = 5.0
    maintain_formation: bool = False
    communication_range: float = 15.0
    
    def __post_init__(self):
        """Validate formation parameters."""
        valid_types = {"grid", "line", "circle", "custom"}
        if self.type not in valid_types:
            raise ValueError(f"Formation type must be one of {valid_types}")

@dataclass
class VideoPlumeConfig:
    """Video plume processing configuration with OpenCV integration."""
    video_path: str = MISSING  # Required field
    flip: bool = False
    grayscale: bool = True
    kernel_size: int = 0
    kernel_sigma: float = 1.0
    threshold: Optional[float] = None
    normalize: bool = True
    
    # Preprocessing configuration with structured validation
    preprocessing: PreprocessingConfig = field(default_factory=lambda: PreprocessingConfig())
    sampling: SamplingConfig = field(default_factory=lambda: SamplingConfig())
    
    def __post_init__(self):
        """Validate video processing parameters."""
        if self.kernel_size < 0:
            raise ValueError("Kernel size must be non-negative")
        if self.kernel_size > 0 and self.kernel_size % 2 == 0:
            raise ValueError("Kernel size must be odd when > 0")
        if self.threshold is not None and not 0.0 <= self.threshold <= 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")

@dataclass
class GymnasiumEnvironmentConfig:
    """Structured configuration for Gymnasium environment with type safety."""
    max_episode_steps: int = 1000
    render_mode: Optional[str] = None
    
    # Action space configuration with validation
    action_space: ActionSpaceConfig = field(default_factory=lambda: ActionSpaceConfig())
    
    # Observation space with structured components
    observation_space: ObservationSpaceConfig = field(default_factory=lambda: ObservationSpaceConfig())
    
    # Reward shaping with performance optimization
    reward_shaping: RewardShapingConfig = field(default_factory=lambda: RewardShapingConfig())
    
    # Termination conditions with clear criteria
    termination: TerminationConfig = field(default_factory=lambda: TerminationConfig())

@dataclass
class RewardShapingConfig:
    """Reward shaping configuration with algorithm optimization."""
    odor_weight: float = 1.0
    distance_weight: float = 0.5
    control_penalty: float = 0.1
    efficiency_bonus: float = 0.2
    success_reward: float = 10.0
    
    def __post_init__(self):
        """Validate reward parameters."""
        if self.odor_weight < 0:
            raise ValueError("Odor weight must be non-negative")

@dataclass
class LoguruLoggingConfig:
    """Structured logging configuration with Loguru integration."""
    level: str = "INFO"
    format: str = "json"  # Options: text, json
    colorize: bool = True
    diagnose: bool = True
    
    # File logging configuration
    file: FileLoggingConfig = field(default_factory=lambda: FileLoggingConfig())
    
    # Console logging configuration  
    console: ConsoleLoggingConfig = field(default_factory=lambda: ConsoleLoggingConfig())
    
    # Performance monitoring integration
    performance: PerformanceLoggingConfig = field(default_factory=lambda: PerformanceLoggingConfig())
    
    # Correlation ID tracking
    correlation: CorrelationConfig = field(default_factory=lambda: CorrelationConfig())

# Register structured configurations with Hydra ConfigStore
cs = ConfigStore.instance()
cs.store(name="navigator_config", node=NavigatorConfig)
cs.store(name="video_plume_config", node=VideoPlumeConfig)
cs.store(name="gymnasium_env_config", node=GymnasiumEnvironmentConfig)
cs.store(name="logging_config", node=LoguruLoggingConfig)
```

#### Structured Configuration Usage

##### Loading and Validation

```python
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from odor_plume_nav.config.models import NavigatorConfig, VideoPlumeConfig

# Initialize Hydra with structured config support
with initialize(config_path="../conf", version_base=None):
    # Compose configuration with automatic validation
    cfg = compose(config_name="config")
    
    # Access structured configurations with full type safety
    navigator_config: NavigatorConfig = cfg.navigator
    video_plume_config: VideoPlumeConfig = cfg.video_plume
    
    # Configuration automatically validated at load time
    print(f"Navigator max speed: {navigator_config.max_speed}")
    print(f"Video path: {video_plume_config.video_path}")
    
    # Type errors caught by IDE and runtime validation
    # navigator_config.max_speed = "invalid"  # TypeError caught immediately
```

##### Configuration Composition and Overrides

```python
# Configuration composition with structured validation
with initialize(config_path="../conf"):
    # Base configuration with overrides
    cfg = compose(
        config_name="config",
        overrides=[
            "navigator.max_speed=2.0",  # Validated against NavigatorConfig
            "video_plume.grayscale=false",  # Type-checked boolean
            "logging.level=DEBUG",  # Validated against allowed levels
            "+gymnasium=advanced_env"  # Add structured environment config
        ]
    )
    
    # All overrides automatically validated against dataclass schemas
    assert isinstance(cfg.navigator.max_speed, float)
    assert isinstance(cfg.video_plume.grayscale, bool)
```

##### Factory Pattern with Structured Configs

```python
from odor_plume_nav.config.models import NavigatorConfig
from odor_plume_nav.core import Navigator

def create_navigator_from_structured_config(config: NavigatorConfig) -> Navigator:
    """Create navigator with full type safety and validation."""
    
    # Configuration already validated by dataclass post_init
    navigator = Navigator(
        position=config.position,
        orientation=config.orientation,
        max_speed=config.max_speed,
        control_config=config.control
    )
    
    # Multi-agent configuration handling
    if config.positions is not None:
        navigator.configure_multi_agent(
            positions=config.positions,
            orientations=config.orientations,
            formation=config.formation
        )
    
    return navigator

# Usage with automatic validation
navigator_config = NavigatorConfig(
    position=[10.0, 15.0],
    max_speed=2.5,
    control=ControlConfig(acceleration=0.2, turning_rate=45.0)
)

navigator = create_navigator_from_structured_config(navigator_config)
```

#### Development vs. Production Configuration

```python
# Development configuration with enhanced debugging
@dataclass
class DevelopmentConfig:
    """Development environment structured configuration."""
    environment: str = "development"
    debug_mode: bool = True
    verbose_output: bool = True
    
    navigator: NavigatorConfig = field(default_factory=lambda: NavigatorConfig(
        max_speed=1.0,  # Conservative speed for debugging
        control=ControlConfig(sensor_noise=0.0)  # No noise for reproducibility
    ))
    
    logging: LoguruLoggingConfig = field(default_factory=lambda: LoguruLoggingConfig(
        level="DEBUG",
        format="text",  # Human-readable for development
        colorize=True,
        performance=PerformanceLoggingConfig(enabled=True, step_timing=True)
    ))

# Production configuration with performance optimization
@dataclass  
class ProductionConfig:
    """Production environment structured configuration."""
    environment: str = "production"
    debug_mode: bool = False
    verbose_output: bool = False
    
    navigator: NavigatorConfig = field(default_factory=lambda: NavigatorConfig(
        max_speed=2.0,  # Higher performance in production
        control=ControlConfig(sensor_noise=0.1)  # Realistic noise
    ))
    
    logging: LoguruLoggingConfig = field(default_factory=lambda: LoguruLoggingConfig(
        level="INFO",
        format="json",  # Structured logs for analysis
        colorize=False,
        performance=PerformanceLoggingConfig(
            enabled=True,
            slow_threshold=0.010,  # Stricter production threshold
            alert_on_degradation=True
        )
    ))

# Register environment-specific configurations
cs.store(name="development_config", node=DevelopmentConfig)
cs.store(name="production_config", node=ProductionConfig)
```

#### Configuration Validation and Error Handling

```python
from pydantic import ValidationError
from hydra.errors import ConfigCompositionException

def load_validated_config(config_name: str = "config"):
    """Load configuration with comprehensive validation and error handling."""
    
    try:
        with initialize(config_path="../conf", version_base=None):
            cfg = compose(config_name=config_name)
            
            # Additional business logic validation
            validate_configuration_constraints(cfg)
            
            return cfg
            
    except ValidationError as e:
        print(f"Configuration validation error: {e}")
        print("Please check your configuration files for type errors.")
        raise
        
    except ConfigCompositionException as e:
        print(f"Configuration composition error: {e}")
        print("Please check your configuration file structure.")
        raise

def validate_configuration_constraints(config):
    """Additional validation beyond dataclass constraints."""
    
    # Cross-component validation
    if config.navigator.max_speed > 5.0 and config.environment.debug_mode:
        logger.warning(
            "High speed in debug mode may affect debugging",
            max_speed=config.navigator.max_speed
        )
    
    # Performance constraint validation
    if config.video_plume.kernel_size > 7:
        estimated_fps = estimate_processing_fps(config.video_plume)
        if estimated_fps < 30:
            raise ValueError(
                f"Configuration may not meet 30 FPS requirement. "
                f"Estimated FPS: {estimated_fps}"
            )
    
    # Resource availability validation
    if config.navigator.num_agents and config.navigator.num_agents > 100:
        available_memory = get_available_memory()
        required_memory = estimate_memory_usage(config.navigator.num_agents)
        if required_memory > available_memory:
            raise ValueError(
                f"Insufficient memory for {config.navigator.num_agents} agents. "
                f"Required: {required_memory}MB, Available: {available_memory}MB"
            )
```

### RL Configuration Examples

#### Structured Algorithm Configuration

Using dataclass-based RL algorithm configuration with automatic validation:

```python
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from hydra.core.config_store import ConfigStore

@dataclass
class PPOAlgorithmConfig:
    """Structured PPO algorithm configuration with hyperparameter validation."""
    algorithm: str = "PPO"
    
    # Core PPO hyperparameters with validation
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    clip_range_vf: Optional[float] = None
    entropy_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    
    # Training configuration with type safety
    training: PPOTrainingConfig = field(default_factory=lambda: PPOTrainingConfig())
    
    def __post_init__(self):
        """Validate PPO hyperparameters."""
        if not 0 < self.learning_rate < 1:
            raise ValueError("Learning rate must be between 0 and 1")
        if self.n_steps <= 0:
            raise ValueError("Number of steps must be positive")
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        if not 0 <= self.gamma <= 1:
            raise ValueError("Gamma must be between 0 and 1")

@dataclass
class PPOTrainingConfig:
    """PPO training pipeline configuration."""
    total_timesteps: int = 1000000
    eval_freq: int = 10000
    eval_episodes: int = 20
    save_freq: int = 50000
    tensorboard_log: str = "./tensorboard_logs"
    verbose: int = 1
    
    def __post_init__(self):
        """Validate training parameters."""
        if self.total_timesteps <= 0:
            raise ValueError("Total timesteps must be positive")
        if self.eval_freq <= 0:
            raise ValueError("Evaluation frequency must be positive")

@dataclass
class SACAlgorithmConfig:
    """Structured SAC algorithm configuration with parameter constraints."""
    algorithm: str = "SAC"
    
    # SAC-specific hyperparameters
    learning_rate: float = 3e-4
    buffer_size: int = 1000000
    learning_starts: int = 10000
    batch_size: int = 256
    tau: float = 0.005
    gamma: float = 0.99
    train_freq: int = 1
    gradient_steps: int = 1
    ent_coef: str = "auto"  # or float value
    target_update_interval: int = 1
    target_entropy: str = "auto"  # or float value
    
    training: SACTrainingConfig = field(default_factory=lambda: SACTrainingConfig())

@dataclass
class SACTrainingConfig:
    """SAC training configuration with off-policy optimization."""
    total_timesteps: int = 500000
    eval_freq: int = 5000
    eval_episodes: int = 10
    save_freq: int = 25000
    verbose: int = 1

# Register algorithm configurations
cs.store(group="rl/algorithms", name="ppo", node=PPOAlgorithmConfig)
cs.store(group="rl/algorithms", name="sac", node=SACAlgorithmConfig)
```

#### YAML Configuration with Structured Config Integration

```yaml
# conf/rl/algorithms/ppo.yaml - now with structured config annotations
# @package rl.algorithm
_target_: odor_plume_nav.config.models.PPOAlgorithmConfig

# All parameters automatically validated against dataclass
algorithm: PPO
learning_rate: 3e-4
n_steps: 2048
batch_size: 64
n_epochs: 10
gamma: 0.99
gae_lambda: 0.95
clip_range: 0.2
clip_range_vf: null
entropy_coef: 0.0
vf_coef: 0.5
max_grad_norm: 0.5

training:
  total_timesteps: 1000000
  eval_freq: 10000
  eval_episodes: 20
  save_freq: 50000
  tensorboard_log: ./tensorboard_logs
  verbose: 1
```

```yaml
# conf/rl/algorithms/sac.yaml
algorithm: SAC
hyperparameters:
  learning_rate: 3e-4
  buffer_size: 1000000
  learning_starts: 10000
  batch_size: 256
  tau: 0.005
  gamma: 0.99
  train_freq: 1
  gradient_steps: 1
  ent_coef: auto
  target_update_interval: 1
  target_entropy: auto
  
training:
  total_timesteps: 500000
  eval_freq: 5000
  eval_episodes: 10
  save_freq: 25000
  verbose: 1
```

#### Structured Environment Configuration

```python
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import numpy as np

@dataclass
class ActionSpaceConfig:
    """Action space configuration with automatic Gymnasium space creation."""
    type: str = "Box"
    low: List[float] = field(default_factory=lambda: [-1.0, -1.0])
    high: List[float] = field(default_factory=lambda: [1.0, 1.0])
    dtype: str = "float32"
    
    def create_space(self):
        """Create Gymnasium action space from configuration."""
        import gymnasium as gym
        return gym.spaces.Box(
            low=np.array(self.low, dtype=self.dtype),
            high=np.array(self.high, dtype=self.dtype)
        )

@dataclass
class ObservationComponentConfig:
    """Individual observation component configuration."""
    type: str = "Box"
    shape: List[int] = field(default_factory=list)
    low: float = 0.0
    high: float = 1.0
    dtype: str = "float32"

@dataclass
class ObservationSpaceConfig:
    """Structured observation space with component validation."""
    odor_concentration: ObservationComponentConfig = field(
        default_factory=lambda: ObservationComponentConfig(
            shape=[1], low=0.0, high=1.0
        )
    )
    agent_position: ObservationComponentConfig = field(
        default_factory=lambda: ObservationComponentConfig(
            shape=[2], low=0.0, high=100.0
        )
    )
    agent_orientation: ObservationComponentConfig = field(
        default_factory=lambda: ObservationComponentConfig(
            shape=[1], low=-np.pi, high=np.pi
        )
    )
    plume_gradient: ObservationComponentConfig = field(
        default_factory=lambda: ObservationComponentConfig(
            shape=[2], low=-1.0, high=1.0
        )
    )
    
    def create_space(self):
        """Create Gymnasium Dict observation space."""
        import gymnasium as gym
        
        components = {}
        for name, component in self.__dict__.items():
            components[name] = gym.spaces.Box(
                low=component.low,
                high=component.high,
                shape=component.shape,
                dtype=getattr(np, component.dtype)
            )
        
        return gym.spaces.Dict(components)

@dataclass
class AdvancedEnvironmentConfig:
    """Advanced environment configuration with full validation."""
    max_episode_steps: int = 2000
    render_mode: Optional[str] = None  # 'human', 'rgb_array', or None
    
    # Structured space configurations
    action_space: ActionSpaceConfig = field(default_factory=ActionSpaceConfig)
    observation_space: ObservationSpaceConfig = field(default_factory=ObservationSpaceConfig)
    
    # Reward and termination with validation
    reward_shaping: RewardShapingConfig = field(default_factory=lambda: RewardShapingConfig(
        odor_weight=1.0,
        distance_weight=0.5,
        control_penalty=0.1,
        efficiency_bonus=0.2,
        success_reward=10.0
    ))
    
    termination: TerminationConfig = field(default_factory=lambda: TerminationConfig(
        max_distance_from_start=150.0,
        min_odor_threshold=0.01,
        success_odor_threshold=0.8
    ))
    
    def __post_init__(self):
        """Validate environment configuration."""
        if self.max_episode_steps <= 0:
            raise ValueError("Max episode steps must be positive")
        
        valid_render_modes = {None, 'human', 'rgb_array'}
        if self.render_mode not in valid_render_modes:
            raise ValueError(f"Render mode must be one of {valid_render_modes}")

@dataclass
class TerminationConfig:
    """Termination condition configuration with clear criteria."""
    max_distance_from_start: float = 150.0
    min_odor_threshold: float = 0.01
    success_odor_threshold: float = 0.8
    max_steps_without_progress: Optional[int] = None
    
    def __post_init__(self):
        """Validate termination parameters."""
        if not 0 <= self.min_odor_threshold <= 1:
            raise ValueError("Min odor threshold must be between 0 and 1")
        if not 0 <= self.success_odor_threshold <= 1:
            raise ValueError("Success odor threshold must be between 0 and 1")
        if self.min_odor_threshold >= self.success_odor_threshold:
            raise ValueError("Min threshold must be less than success threshold")

# Register environment configurations
cs.store(group="rl/environments", name="advanced", node=AdvancedEnvironmentConfig)
```

#### YAML Configuration with Structured Config Integration

```yaml
# conf/rl/environments/advanced.yaml - with dataclass validation
# @package rl.environment
_target_: odor_plume_nav.config.models.AdvancedEnvironmentConfig

environment:
  max_episode_steps: 2000
  render_mode: null  # Automatically validated against allowed values
  
action_space:
  type: Box
  low: [-1.0, -1.0]
  high: [1.0, 1.0]
  dtype: float32

observation_space:
  odor_concentration:
    type: Box
    shape: [1]
    low: 0.0
    high: 1.0
    dtype: float32
  agent_position:
    type: Box
    shape: [2]
    low: 0.0  # Broadcast to [0.0, 0.0] automatically
    high: 100.0  # Broadcast to [100.0, 100.0] automatically
    dtype: float32
  agent_orientation:
    type: Box
    shape: [1]
    low: -3.14159
    high: 3.14159
    dtype: float32

reward_shaping:
  odor_weight: 1.0
  distance_weight: 0.5
  control_penalty: 0.1
  efficiency_bonus: 0.2
  success_reward: 10.0
  
termination:
  max_distance_from_start: 150.0
  min_odor_threshold: 0.01
  success_odor_threshold: 0.8
```

#### Training Pipeline Configuration

```yaml
# conf/rl/training.yaml
defaults:
  - algorithms: ppo
  - environments: advanced
  - _self_

training_pipeline:
  parallel_envs: 4
  evaluation:
    enabled: true
    frequency: 10000
    episodes: 20
    deterministic: true
    
  checkpointing:
    enabled: true
    frequency: 50000
    keep_best: true
    max_checkpoints: 5
    
  logging:
    tensorboard: true
    wandb: false
    csv: true
    console_level: INFO
    
  callbacks:
    early_stopping:
      enabled: false
      patience: 100000
      min_delta: 0.01
    learning_rate_scheduler:
      enabled: false
      schedule: linear
      final_lr: 1e-6
```

### Basic Configuration Usage

```python
from hydra import compose, initialize
from odor_plume_nav.api import create_gymnasium_environment

# Basic RL configuration loading
with initialize(config_path="../conf", version_base=None):
    cfg = compose(config_name="config", config_path="../conf/rl")
    env = create_gymnasium_environment(cfg.environment)

# Dynamic parameter overrides for RL training
with initialize(config_path="../conf"):
    cfg = compose(
        config_name="config",
        overrides=[
            "rl/algorithms=sac",
            "rl/environments=advanced",
            "rl.training.total_timesteps=1000000",
            "rl.environment.reward_shaping.odor_weight=2.0"
        ]
    )
```

### Environment Variable Integration

The configuration system supports secure credential management through environment variables:

```yaml
# conf/config.yaml
database:
  url: ${oc.env:DATABASE_URL,sqlite:///local.db}
  username: ${oc.env:DB_USER,admin}
  password: ${oc.env:DB_PASSWORD}

video_plume:
  video_path: ${oc.env:VIDEO_PATH,data/videos/example_plume.mp4}
  
navigator:
  max_speed: ${oc.env:NAVIGATOR_MAX_SPEED,1.5}

rl:
  tensorboard_log: ${oc.env:TENSORBOARD_LOG_DIR,./tensorboard_logs}
  checkpoint_dir: ${oc.env:CHECKPOINT_DIR,./checkpoints}
  model_save_path: ${oc.env:MODEL_SAVE_PATH,./models}
```

#### Environment Variable Setup

Create a `.env` file in your project root:

```bash
# .env file
ENVIRONMENT_TYPE=development
DATABASE_URL=postgresql://user:password@localhost:5432/plume_nav
VIDEO_PATH=/data/experiments/high_resolution_plume.mp4
NAVIGATOR_MAX_SPEED=2.0
DEBUG=true
LOG_LEVEL=INFO

# RL-specific environment variables
TENSORBOARD_LOG_DIR=/data/tensorboard_logs
CHECKPOINT_DIR=/data/checkpoints
MODEL_SAVE_PATH=/data/trained_models
WANDB_PROJECT=odor_plume_navigation
WANDB_API_KEY=your_wandb_api_key
```

### Migration from Legacy Configuration

#### Migration Guide: Unstructured YAML to Dataclass Configuration

The library has evolved from unstructured YAML configuration to Pydantic-validated dataclass configuration, providing type safety, IDE support, and runtime validation.

#### Legacy vs. Modern Configuration Comparison

**Legacy Unstructured Configuration (v0.1.x):**
```yaml
# configs/default.yaml - Old approach
navigator:
  position: [10.0, 15.0]  # No type validation
  max_speed: "2.0"        # String accepted, potential runtime errors
  invalid_field: "value" # Unknown fields silently ignored

video_plume:
  video_path: null        # Missing required field not caught
  kernel_size: 2          # Even numbers allowed, causes OpenCV errors
  threshold: 1.5          # Invalid range not validated
```

**Modern Structured Configuration (v0.2.x):**
```yaml
# conf/config.yaml - New approach with validation
# @package _global_
_target_: odor_plume_nav.config.models.NavigatorConfig

navigator:
  position: [10.0, 15.0]  # Type-validated as List[float]
  max_speed: 2.0          # Must be float, validated at load time
  # invalid_field: "value" # Rejected with clear error message

video_plume:
  video_path: "data/example.mp4"  # Required field enforced
  kernel_size: 3          # Must be odd, validated in __post_init__
  threshold: 0.8          # Range [0.0, 1.0] enforced automatically
```

#### Step-by-Step Migration Process

##### Step 1: Install Updated Dependencies

```bash
# Ensure you have the latest version with structured config support
pip install "odor_plume_nav[rl]>=0.2.0"
# or
poetry add "odor_plume_nav[rl]>=0.2.0"
```

##### Step 2: Update Configuration Directory Structure

```bash
# Old structure
configs/
├── default.yaml
├── example_user_config.yaml
└── README.md

# New structure
conf/
├── base.yaml          # Replaces default.yaml
├── config.yaml        # Replaces example_user_config.yaml  
├── rl/                # New: RL-specific configurations
│   ├── algorithms/
│   ├── environments/
│   └── training.yaml
└── local/             # New: environment-specific overrides
    ├── development.yaml
    └── production.yaml
```

##### Step 3: Convert Configuration Loading Code

**Legacy Loading Approach:**
```python
# Old approach - error-prone manual loading
from odor_plume_nav.services.config_loader import load_config
import yaml

config = load_config("configs/default.yaml")
# No validation, runtime errors possible
navigator = create_navigator(
    position=config["navigator"]["position"],  # Dictionary access
    max_speed=float(config["navigator"]["max_speed"])  # Manual type conversion
)
```

**Modern Structured Loading:**
```python
# New approach - type-safe with automatic validation
from hydra import compose, initialize
from odor_plume_nav.config.models import NavigatorConfig
from odor_plume_nav.core import Navigator

with initialize(config_path="../conf", version_base=None):
    cfg = compose(config_name="config")
    
    # Fully typed configuration object
    navigator_config: NavigatorConfig = cfg.navigator
    
    # Type safety and IDE autocomplete
    navigator = Navigator.from_config(navigator_config)
    # All validation happened at cfg composition time
```

##### Step 4: Update Parameter Override Patterns

**Legacy Override Pattern:**
```python
# Old approach - manual dictionary manipulation
config = load_config("configs/default.yaml")
config["navigator"]["max_speed"] = 2.5  # No validation
config["video_plume"]["nonexistent"] = "value"  # Silently accepted
```

**Modern Override Pattern:**
```python
# New approach - validated overrides
with initialize(config_path="../conf"):
    cfg = compose(
        config_name="config",
        overrides=[
            "navigator.max_speed=2.5",  # Validated against NavigatorConfig
            # "navigator.nonexistent=value"  # Rejected with clear error
        ]
    )
```

##### Step 5: Update CLI Integration

**Legacy CLI Usage:**
```bash
# Old approach - manual parameter passing
python scripts/run_simulation.py \
    --config configs/my_config.yaml \
    --max-speed 2.0 \
    --video-path data/video.mp4
```

**Modern CLI Usage:**
```bash
# New approach - Hydra integration with validation
plume-nav-sim run \
    navigator.max_speed=2.0 \
    video_plume.video_path=data/video.mp4 \
    --config-name my_config
```

#### Configuration Validation Benefits

##### Immediate Error Detection

```python
# Example validation errors caught at startup:

# Type validation
navigator:
  max_speed: "invalid"  # ValidationError: Input should be a valid number

# Range validation  
video_plume:
  threshold: 1.5        # ValidationError: Threshold must be between 0.0 and 1.0

# Required field validation
video_plume:
  # video_path missing   # ValidationError: Field required

# Custom business logic validation
navigator:
  speed: 3.0
  max_speed: 2.0        # ValidationError: Speed cannot exceed max_speed

# Cross-component validation
gymnasium:
  max_episode_steps: -100  # ValidationError: Max episode steps must be positive
```

##### IDE Integration Benefits

```python
from odor_plume_nav.config.models import NavigatorConfig

# Full IDE autocomplete and type hints
config = NavigatorConfig(
    position=[10.0, 15.0],  # IDE knows this is List[float]
    max_speed=2.0,          # IDE knows this is float
    # IDE will suggest valid field names and types
)

# Type checking with mypy
def create_navigator(config: NavigatorConfig) -> Navigator:
    # mypy verifies all attribute access
    return Navigator(
        position=config.position,      # Validated List[float]
        max_speed=config.max_speed,    # Validated float
    )
```

#### Backward Compatibility Strategy

The library maintains backward compatibility during the transition period:

```python
# Compatibility loading function for legacy configurations
from odor_plume_nav.config.legacy import load_legacy_config
from odor_plume_nav.config.models import migrate_legacy_config

def load_config_with_migration(config_path: str):
    """Load configuration with automatic migration from legacy format."""
    
    if config_path.endswith('configs/'):  # Legacy path pattern
        # Load legacy configuration
        legacy_config = load_legacy_config(config_path)
        
        # Migrate to structured format with validation
        structured_config = migrate_legacy_config(legacy_config)
        
        # Warn user about deprecated usage
        logger.warning(
            "Using legacy configuration format. "
            "Consider migrating to structured configuration in conf/ directory.",
            legacy_path=config_path
        )
        
        return structured_config
    
    else:
        # Load modern structured configuration
        with initialize(config_path=config_path):
            return compose(config_name="config")

# Usage supports both formats during migration
config = load_config_with_migration("configs/")  # Legacy support
# config = load_config_with_migration("conf/")   # Modern approach
```

#### Migration Checklist

- [ ] **Install updated dependencies** with dataclass support
- [ ] **Create new `conf/` directory** structure  
- [ ] **Migrate configuration files** from `configs/` to `conf/`
- [ ] **Add structured config annotations** to YAML files
- [ ] **Update configuration loading code** to use Hydra compose()
- [ ] **Replace manual parameter access** with typed configuration objects
- [ ] **Update CLI usage** to use new Hydra integration
- [ ] **Test configuration validation** with intentional errors
- [ ] **Update documentation** and examples to use structured configs
- [ ] **Remove legacy configuration files** after validation

If migrating from the old `configs/` structure to the new Hydra-based `conf/` system:

#### Legacy Structure (Old)
```
configs/
├── default.yaml
├── example_user_config.yaml
└── README.md
```

#### New Hydra Structure
```
conf/
├── base.yaml          # Replaces default.yaml
├── config.yaml        # Replaces example_user_config.yaml
├── rl/                # New: RL-specific configurations
│   ├── algorithms/
│   ├── environments/
│   └── training.yaml
└── local/             # New: environment-specific overrides
    ├── development.yaml
    └── production.yaml
```

#### Migration Steps

1. **Copy base parameters**: Move `configs/default.yaml` content to `conf/base.yaml`
2. **User customizations**: Move `configs/example_user_config.yaml` to `conf/config.yaml`
3. **RL configuration**: Create new `conf/rl/` directory with algorithm and environment configs
4. **Environment setup**: Create environment-specific files in `conf/local/`
5. **Update imports**: Change from:
   ```python
   # Old approach
   from odor_plume_nav.services.config_loader import load_config
   config = load_config("configs/default.yaml")
   ```
   
   To:
   ```python
   # New Hydra approach
   from hydra import compose, initialize
   with initialize(config_path="../conf"):
       cfg = compose(config_name="config")
   ```

6. **CLI migration**: Replace manual script execution with new CLI commands:
   ```bash
   # Old approach
   python scripts/run_simulation.py --config configs/my_config.yaml
   
   # New approach (simulation)
   plume-nav-sim run --config-name my_config
   
   # New approach (RL training)
   plume-nav-sim train --algorithm PPO --config-name my_rl_config
   ```

## Development Workflow

### Setup Development Environment

```bash
# Clone and setup
git clone https://github.com/organization/odor_plume_nav.git
cd odor_plume_nav

# Poetry setup (recommended)
poetry install --with dev,docs,viz,rl
poetry shell

# Install pre-commit hooks
pre-commit install

# Alternative: Make-based setup
make setup-dev
```

### Makefile Commands

The project includes comprehensive Makefile automation:

```bash
# Development commands
make install-dev       # Poetry install with dev dependencies
make setup-dev         # Complete development environment setup
make install          # Traditional pip install (fallback)

# Code quality
make format           # Run black and isort formatting
make lint            # Run flake8 linting
make type-check      # Run mypy type checking
make test            # Run pytest with coverage
make test-all        # Run all quality checks

# Build and distribution
make build           # Build wheel and sdist
make poetry-build    # Build using Poetry
make clean           # Clean build artifacts

# Documentation
make docs            # Build Sphinx documentation
make docs-serve      # Serve documentation locally

# Docker commands
make docker-build    # Build Docker image
make docker-run      # Run container
make docker-dev      # Development environment with docker-compose
```

### Pre-commit Hooks

The project includes automated code quality checks:

```yaml
# .pre-commit-config.yaml (example hooks)
repos:
  - repo: https://github.com/psf/black
    rev: 23.12.0
    hooks:
      - id: black
  
  - repo: https://github.com/pycqa/isort
    rev: 5.13.0
    hooks:
      - id: isort
  
  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
```

### Testing Strategy

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=odor_plume_nav --cov-report=html

# Run specific test categories
pytest -m "not slow"          # Skip slow tests
pytest -m integration         # Integration tests only
pytest tests/unit/            # Unit tests only
pytest tests/rl/              # RL-specific tests only

# Run with parallel execution
pytest -n auto               # Parallel execution

# Test RL environments specifically
pytest tests/rl/test_gymnasium_env.py -v
pytest tests/rl/test_training_workflows.py -v
```

## Advanced Features

### Multi-Run Experiment Management

```bash
# Systematic parameter exploration
plume-nav-sim run --multirun \
  navigator.max_speed=1.0,1.5,2.0 \
  navigator.angular_velocity=0.1,0.2,0.3 \
  video_plume.gaussian_blur.sigma=1.0,2.0,3.0

# RL hyperparameter sweeps
plume-nav-sim train --multirun \
  --algorithm PPO \
  rl.hyperparameters.learning_rate=1e-4,3e-4,1e-3 \
  rl.hyperparameters.n_steps=1024,2048,4096 \
  rl.training.total_timesteps=500000

# Results organized automatically in:
# outputs/multirun/2024-01-15_10-30-00/
# ├── run_0_navigator.max_speed=1.0,navigator.angular_velocity=0.1,video_plume.gaussian_blur.sigma=1.0/
# ├── run_1_navigator.max_speed=1.0,navigator.angular_velocity=0.1,video_plume.gaussian_blur.sigma=2.0/
# └── ...
```

### Docker-Compose Development Environment

The library includes a complete development infrastructure:

```yaml
# docker-compose.yml
version: '3.8'
services:
  odor_plume_nav:
    build: .
    volumes:
      - ./src:/app/src
      - ./conf:/app/conf
      - ./data:/app/data
      - ./outputs:/app/outputs
    depends_on:
      - postgres
    environment:
      - DATABASE_URL=postgresql://user:password@postgres:5432/odor_nav
      - ENVIRONMENT_TYPE=development

  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: odor_nav
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  pgadmin:
    image: dpage/pgadmin4:latest
    environment:
      PGADMIN_DEFAULT_EMAIL: admin@example.com
      PGADMIN_DEFAULT_PASSWORD: admin
    ports:
      - "8080:80"
    depends_on:
      - postgres

volumes:
  postgres_data:
```

### Performance Optimization

```python
# Configure for high-performance computation
from odor_plume_nav.utils import configure_performance

# Optimize NumPy and OpenCV for multi-core systems
configure_performance(
    numpy_threads=8,
    opencv_threads=6,
    use_gpu=True
)

# Environment variable configuration
export NUMPY_THREADS=8
export OPENCV_OPENCL=true
export MATPLOTLIB_BACKEND=Agg  # Headless mode for batch processing
```

## Project Structure

```
odor_plume_nav/
├── src/odor_plume_nav/           # Main library package
│   ├── __init__.py                   # Public API exports
│   ├── api/                          # Public interfaces
│   │   ├── __init__.py
│   │   └── navigation.py             # Main API functions
│   ├── cli/                          # Command-line interface
│   │   ├── __init__.py
│   │   └── main.py                   # CLI entry point
│   ├── config/                       # Configuration management
│   │   ├── __init__.py
│   │   └── schemas.py                # Pydantic validation schemas
│   ├── core/                         # Core business logic
│   │   ├── __init__.py
│   │   ├── navigator.py              # Navigation protocols
│   │   ├── controllers.py            # Agent controllers
│   │   └── sensors.py                # Sensor strategies
│   ├── data/                         # Data processing
│   │   ├── __init__.py
│   │   └── video_plume.py            # Video plume processing
│   ├── db/                           # Database integration (future)
│   │   └── session.py                # Session management
│   ├── environments/                 # RL environments (new)
│   │   ├── __init__.py
│   │   ├── gymnasium_env.py          # Main Gymnasium environment
│   │   ├── spaces.py                 # Action/observation space definitions
│   │   ├── wrappers.py               # Environment preprocessing wrappers
│   │   └── video_plume.py            # Video plume environment
│   ├── rl/                           # RL utilities (new)
│   │   ├── __init__.py
│   │   ├── training.py               # Training utilities and workflows
│   │   └── policies.py               # Custom policy implementations
│   └── utils/                        # Utilities
│       ├── __init__.py
│       ├── seed_manager.py           # Reproducibility
│       ├── visualization.py          # Plotting and animation
│       └── logging.py                # Logging configuration
├── conf/                             # Hydra configuration
│   ├── base.yaml                     # Foundation defaults
│   ├── config.yaml                   # User customizations
│   ├── rl/                           # RL-specific configurations
│   │   ├── algorithms/               # Algorithm hyperparameters
│   │   ├── environments/             # Environment configurations
│   │   └── training.yaml             # Training pipeline settings
│   └── local/                        # Environment-specific
│       ├── credentials.yaml.template
│       └── paths.yaml.template
├── tests/                            # Test suite
│   ├── unit/                         # Unit tests
│   ├── integration/                  # Integration tests
│   └── rl/                           # RL-specific tests
├── notebooks/                        # Example notebooks
│   ├── demos/                        # Demonstration notebooks
│   ├── rl_examples/                  # RL training examples
│   └── exploratory/                  # Research notebooks
├── workflow/                         # Workflow definitions (future)
│   ├── dvc/                          # DVC pipelines
│   └── snakemake/                    # Snakemake workflows
├── docker-compose.yml               # Development environment
├── Dockerfile                       # Container image
├── Makefile                         # Development automation
├── pyproject.toml                   # Package configuration
└── README.md                        # This file
```

## Integration Examples

### Jupyter Notebook Integration

```python
# notebook_example.ipynb
from hydra import compose, initialize
from odor_plume_nav import Navigator, VideoPlume
from odor_plume_nav.utils import set_global_seed

# Setup reproducible environment
set_global_seed(42)

# Load configuration in notebook
with initialize(config_path="../conf", version_base=None):
    cfg = compose(config_name="config", overrides=[
        "visualization.animation.enabled=true",
        "visualization.plotting.figure_size=[14,10]"
    ])

# Create and run simulation
navigator = Navigator.from_config(cfg.navigator)
video_plume = VideoPlume.from_config(cfg.video_plume)

# Interactive visualization
results = navigator.simulate(video_plume, duration=60)
results.plot_trajectory(interactive=True)
```

### Kedro Pipeline Integration

```python
# kedro_pipeline_example.py
from kedro.pipeline import Pipeline, node
from odor_plume_nav.api import create_navigator, run_plume_simulation

def create_navigation_pipeline(**kwargs) -> Pipeline:
    """Create Kedro pipeline for odor plume navigation."""
    
    return Pipeline([
        node(
            func=create_navigator,
            inputs=["params:navigator_config"],
            outputs="navigator",
            name="create_navigator_node"
        ),
        node(
            func=run_plume_simulation,
            inputs=["navigator", "video_plume", "params:simulation_config"],
            outputs="simulation_results",
            name="run_simulation_node"
        ),
        node(
            func=analyze_trajectory,
            inputs=["simulation_results"],
            outputs="trajectory_analysis",
            name="analyze_trajectory_node"
        )
    ])
```

## Contributing

We welcome contributions! Please see our contributing guidelines:

1. **Fork the repository** and create a feature branch
2. **Setup development environment**: `make setup-dev`
3. **Make changes** with appropriate tests
4. **Run quality checks**: `make test-all`
5. **Submit pull request** with clear description

### Development Standards

- **Code Style**: Black formatting, isort imports, flake8 compliance
- **Type Safety**: MyPy static type checking required
- **Test Coverage**: Minimum 80% coverage for new code
- **Documentation**: Docstrings for all public APIs
- **Commit Messages**: Conventional commits format

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this library in your research, please cite:

```bibtex
@software{odor_plume_navigation_library,
  title={Odor Plume Navigation Library},
  author={Samuel Brudner},
  year={2024},
  url={https://github.com/organization/odor_plume_nav},
  version={0.1.0}
}
```

## Support and Documentation

- **Documentation**: [https://odor-plume-nav.readthedocs.io](https://odor-plume-nav.readthedocs.io)
- **Issues**: [GitHub Issues](https://github.com/organization/odor_plume_nav/issues)
- **Discussions**: [GitHub Discussions](https://github.com/organization/odor_plume_nav/discussions)
- **API Reference**: Generated automatically from docstrings

## Changelog

### Version 0.2.0 (API Consistency and Integration Hardening Release)

#### Core Refactoring and Modernization

- **Gymnasium 0.29.x API Compliance**: Full upgrade from legacy gym to modern Gymnasium with pinned 0.29.x dependency for stability
- **Dual API Support**: Automatic detection and compatibility layer supporting both 4-tuple legacy gym and 5-tuple Gymnasium APIs without breaking changes
- **New Environment ID**: Introduced `PlumeNavSim-v0` environment alongside existing `OdorPlumeNavigation-v1` for enhanced features
- **Backward Compatibility**: Zero breaking changes - all existing gym-based code continues to work unchanged

#### Structured Configuration Revolution

- **Dataclass-Based Configuration**: Complete migration from unstructured YAML to Pydantic-validated dataclass configuration
- **Type Safety**: Full type validation with IDE autocomplete and mypy support throughout configuration system
- **Runtime Validation**: Configuration errors caught at startup rather than runtime, improving reliability
- **Hydra 1.3+ Integration**: Enhanced structured config support with ConfigStore registration and automatic schema validation

#### Centralized Logging Architecture

- **Loguru Integration**: Replaced ad-hoc print statements and basic logging with structured JSON logging system
- **Correlation ID Tracking**: Cross-component request correlation for distributed debugging and analysis
- **Performance Monitoring**: Integrated logging with performance threshold monitoring and automatic alerting
- **Structured Output**: JSON-formatted logs with configurable sinks for development and production environments

#### Enhanced Developer Experience

- **Migration Guides**: Comprehensive documentation for gym→gymnasium and unstructured→structured config transitions
- **Configuration Examples**: Extensive examples of dataclass configuration patterns for all components
- **Logging Configuration**: Detailed examples of Loguru setup for development and production environments
- **API Documentation**: Updated examples showing dual API support and best practices

#### Performance and Quality Improvements

- **Test Coverage Enhancement**: Expanded test coverage targeting ≥70% overall, ≥80% for new code
- **Cross-Repository Integration**: CI verification against `place_mem_rl` main branch and v0.2.0 tag
- **Performance Preservation**: Maintained ≤10ms average step() time on Intel i7-9700K single thread
- **Memory Efficiency**: Optimized memory usage patterns for large-scale multi-agent scenarios

#### Integration and Compatibility

- **stable-baselines3 Compatibility**: Seamless integration with latest stable-baselines3 versions
- **Gymnasium Ecosystem**: Full compatibility with gymnasium wrappers and utilities
- **Legacy Support**: Comprehensive backward compatibility layer with deprecation warnings
- **Environment Registration**: Enhanced registration system supporting both legacy and modern environment IDs

#### Documentation and Examples

- **Migration Documentation**: Step-by-step guides for API and configuration transitions
- **Structured Config Examples**: Comprehensive examples of dataclass-based configuration patterns
- **Logging Examples**: Detailed Loguru configuration and usage patterns
- **Best Practices**: Updated recommendations for modern development workflows

### Version 0.1.0 (Initial Release)

- **Library Architecture**: Transformed from standalone application to importable library
- **Hydra Configuration**: Sophisticated hierarchical configuration management
- **CLI Interface**: Comprehensive command-line tools with Click framework
- **Multi-Framework Support**: Integration patterns for Kedro, RL, and ML workflows
- **Docker Support**: Containerized development and deployment environments
- **Dual Workflows**: Poetry and pip installation support
- **Enhanced Documentation**: Comprehensive usage examples and migration guides

For detailed changes, see [CHANGELOG.md](CHANGELOG.md).