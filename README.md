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
obs, info = env.reset()

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

The library uses a sophisticated Hydra-based configuration hierarchy that supports environment variable integration, parameter sweeps, and multi-environment deployment.

### Configuration Structure

```
conf/
├── base.yaml          # Foundation defaults and core parameters
├── config.yaml        # User customizations and environment-specific overrides
├── rl/                # RL-specific configurations
│   ├── algorithms/    # Algorithm-specific hyperparameters
│   │   ├── ppo.yaml
│   │   ├── sac.yaml
│   │   └── td3.yaml
│   ├── environments/  # Environment configurations
│   │   ├── basic.yaml
│   │   ├── advanced.yaml
│   │   └── multi_agent.yaml
│   └── training.yaml  # Training pipeline configuration
└── local/             # Local development and deployment-specific settings
    ├── credentials.yaml.template
    ├── development.yaml
    ├── production.yaml
    └── paths.yaml.template
```

### RL Configuration Examples

#### Algorithm Configuration

```yaml
# conf/rl/algorithms/ppo.yaml
algorithm: PPO
hyperparameters:
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

#### Environment Configuration

```yaml
# conf/rl/environments/advanced.yaml
environment:
  max_episode_steps: 2000
  render_mode: null  # 'human', 'rgb_array', or null
  
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
    low: [0.0, 0.0]
    high: [100.0, 100.0]
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

### Version 0.2.0 (Gymnasium Integration Release)

- **Gymnasium API Integration**: Full compliance with Gymnasium environment interface
- **RL Training Workflows**: CLI commands for PPO, SAC, and TD3 algorithm training
- **Environment Wrappers**: Preprocessing and reward shaping capabilities
- **Migration Support**: Backward compatibility with existing simulation APIs
- **Enhanced Documentation**: Comprehensive RL integration examples and best practices

### Version 0.1.0 (Initial Release)

- **Library Architecture**: Transformed from standalone application to importable library
- **Hydra Configuration**: Sophisticated hierarchical configuration management
- **CLI Interface**: Comprehensive command-line tools with Click framework
- **Multi-Framework Support**: Integration patterns for Kedro, RL, and ML workflows
- **Docker Support**: Containerized development and deployment environments
- **Dual Workflows**: Poetry and pip installation support
- **Enhanced Documentation**: Comprehensive usage examples and migration guides

For detailed changes, see [CHANGELOG.md](CHANGELOG.md).