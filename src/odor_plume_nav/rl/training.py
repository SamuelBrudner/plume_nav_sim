"""
Comprehensive RL training utilities module for odor plume navigation.

This module provides the core orchestration layer for reinforcement learning workflows,
enabling automated policy optimization with stable-baselines3 algorithms. It serves as
the primary interface for RL training operations, supporting algorithm factory functions,
vectorized environment management, training progress monitoring, model checkpointing,
and comprehensive experiment tracking.

The module integrates seamlessly with the existing Gymnasium environment wrapper and
provides advanced features for production-ready RL training including error recovery,
performance optimization, and extensive monitoring capabilities.

Key Features:
- Algorithm factory functions for PPO, SAC, TD3 with optimized hyperparameters
- Vectorized environment support with both sync and async execution patterns
- Comprehensive training progress monitoring with real-time metrics and ETA estimation
- Automatic model checkpointing with configurable frequency and resumption capabilities
- Integration with TensorBoard and Weights & Biases for experiment tracking
- Advanced error recovery mechanisms for long-running training sessions
- Performance optimization targeting ≥30 FPS simulation with vectorized environments
- Hydra configuration integration for flexible hyperparameter management

Technical Architecture:
- Factory pattern for algorithm instantiation with validated configurations
- Context managers for training session lifecycle management
- Performance monitoring with sub-millisecond precision timing
- Memory-efficient checkpoint serialization and storage
- Thread-safe progress tracking for concurrent training workflows
- Comprehensive error classification and recovery strategies
"""

from __future__ import annotations
import os
import sys
import time
import uuid
import warnings
import traceback
import threading
from typing import (
    Dict, List, Tuple, Optional, Any, Union, Callable, Type, 
    Protocol, runtime_checkable, ContextManager
)
from pathlib import Path
from contextlib import contextmanager
from dataclasses import dataclass, asdict, field
from enum import Enum
import pickle
import json
from abc import ABC, abstractmethod

import numpy as np
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator

# Stable-baselines3 imports with graceful fallbacks
try:
    import stable_baselines3 as sb3
    from stable_baselines3 import PPO, SAC, TD3, A2C, DQN
    from stable_baselines3.common.base_class import BaseAlgorithm
    from stable_baselines3.common.vec_env import (
        VecEnv, DummyVecEnv, SubprocVecEnv, 
        SyncVectorEnv, AsyncVectorEnv
    )
    from stable_baselines3.common.callbacks import (
        BaseCallback, CheckpointCallback, EvalCallback, 
        StopTrainingOnRewardThreshold, CallbackList
    )
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.utils import safe_mean
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    # Mock classes to prevent import errors
    BaseAlgorithm = object
    VecEnv = object
    BaseCallback = object

# Gymnasium vectorization support
try:
    import gymnasium as gym
    from gymnasium.vector import SyncVectorEnv as GymSyncVectorEnv
    from gymnasium.vector import AsyncVectorEnv as GymAsyncVectorEnv
    GYMNASIUM_AVAILABLE = True
except ImportError:
    GYMNASIUM_AVAILABLE = False
    gym = None

# Optional monitoring and visualization imports
try:
    import tensorboard
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

# Core odor plume navigation imports
from odor_plume_nav.environments.gymnasium_env import GymnasiumEnv
from odor_plume_nav.api.navigation import create_gymnasium_environment
from odor_plume_nav.config.models import SimulationConfig
from odor_plume_nav.utils.seed_manager import set_global_seed, get_seed_context, SeedConfig
from odor_plume_nav.utils.logging_setup import get_enhanced_logger, PerformanceMetrics

# Set up module logger with RL training context
logger = get_enhanced_logger(__name__)


# Training session states and configuration
class TrainingStatus(Enum):
    """Training session status enumeration."""
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


class VectorEnvType(Enum):
    """Vectorized environment execution types."""
    SYNC = "sync"
    ASYNC = "async"
    DUMMY = "dummy"  # SB3 DummyVecEnv
    SUBPROC = "subproc"  # SB3 SubprocVecEnv


class AlgorithmType(Enum):
    """Supported RL algorithm types."""
    PPO = "ppo"
    SAC = "sac"
    TD3 = "td3"
    A2C = "a2c"
    DQN = "dqn"


@dataclass
class TrainingMetrics:
    """Comprehensive training metrics container."""
    
    # Basic training progress
    episode: int = 0
    total_timesteps: int = 0
    elapsed_time: float = 0.0
    
    # Performance metrics
    fps: float = 0.0
    steps_per_second: float = 0.0
    episodes_per_minute: float = 0.0
    
    # Learning progress indicators
    mean_reward: float = 0.0
    reward_std: float = 0.0
    episode_length_mean: float = 0.0
    
    # Algorithm-specific metrics
    loss: Optional[float] = None
    policy_loss: Optional[float] = None
    value_loss: Optional[float] = None
    entropy_loss: Optional[float] = None
    
    # Resource utilization
    memory_usage_mb: float = 0.0
    cpu_percent: float = 0.0
    
    # ETA and completion estimates
    eta_seconds: Optional[float] = None
    completion_percentage: float = 0.0
    
    # Error tracking
    error_count: int = 0
    last_error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for logging and monitoring."""
        return asdict(self)
    
    def format_eta(self) -> str:
        """Format ETA in human-readable format."""
        if self.eta_seconds is None or self.eta_seconds <= 0:
            return "Unknown"
        
        hours, remainder = divmod(int(self.eta_seconds), 3600)
        minutes, seconds = divmod(remainder, 60)
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        else:
            return f"{minutes:02d}:{seconds:02d}"


class TrainingConfig(BaseModel):
    """
    Comprehensive training configuration with Hydra integration.
    
    Provides type-safe configuration management for RL training workflows
    with validation, defaults, and environment variable interpolation support.
    """
    
    # Algorithm configuration
    algorithm: AlgorithmType = Field(
        default=AlgorithmType.PPO,
        description="RL algorithm to use for training"
    )
    
    total_timesteps: int = Field(
        default=100000,
        gt=0,
        description="Total number of timesteps for training"
    )
    
    # Environment configuration
    n_envs: int = Field(
        default=4,
        ge=1,
        le=32,
        description="Number of parallel environments"
    )
    
    env_type: VectorEnvType = Field(
        default=VectorEnvType.ASYNC,
        description="Type of vectorized environment execution"
    )
    
    # Training hyperparameters
    learning_rate: float = Field(
        default=3e-4,
        gt=0.0,
        description="Learning rate for optimizer"
    )
    
    batch_size: Optional[int] = Field(
        default=None,
        description="Batch size for training (algorithm-specific default if None)"
    )
    
    n_epochs: Optional[int] = Field(
        default=None,
        description="Number of epochs per update (PPO/A2C specific)"
    )
    
    # Monitoring and checkpointing
    checkpoint_freq: int = Field(
        default=10000,
        gt=0,
        description="Save checkpoint every N timesteps"
    )
    
    eval_freq: int = Field(
        default=5000,
        gt=0,
        description="Evaluate policy every N timesteps"
    )
    
    eval_episodes: int = Field(
        default=10,
        gt=0,
        description="Number of episodes for evaluation"
    )
    
    # Logging and monitoring
    tensorboard_log: Optional[Path] = Field(
        default=None,
        description="TensorBoard log directory"
    )
    
    wandb_project: Optional[str] = Field(
        default=None,
        description="Weights & Biases project name"
    )
    
    wandb_run_name: Optional[str] = Field(
        default=None,
        description="Weights & Biases run name"
    )
    
    # Performance and optimization
    verbose: int = Field(
        default=1,
        ge=0,
        le=2,
        description="Verbosity level (0=silent, 1=info, 2=debug)"
    )
    
    device: str = Field(
        default="auto",
        description="Device for training (auto, cpu, cuda)"
    )
    
    # Error recovery configuration
    max_retries: int = Field(
        default=3,
        ge=0,
        description="Maximum number of training retry attempts"
    )
    
    checkpoint_on_error: bool = Field(
        default=True,
        description="Save checkpoint when training error occurs"
    )
    
    # Seed configuration
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducible training"
    )
    
    # Hydra target for factory instantiation
    target_: str = Field(
        default="odor_plume_nav.rl.training.create_trainer",
        description="Hydra target for automatic instantiation"
    )
    
    @field_validator('n_envs')
    @classmethod
    def validate_n_envs(cls, v):
        """Validate number of environments is reasonable."""
        import multiprocessing
        max_envs = min(32, multiprocessing.cpu_count() * 2)
        if v > max_envs:
            warnings.warn(f"n_envs={v} may be too high for this system (max recommended: {max_envs})")
        return v
    
    @field_validator('device')
    @classmethod
    def validate_device(cls, v):
        """Validate device specification."""
        if v not in ["auto", "cpu", "cuda"] and not v.startswith("cuda:"):
            raise ValueError(f"Invalid device specification: {v}")
        return v
    
    model_config = ConfigDict(
        extra="allow",
        str_strip_whitespace=True,
        validate_assignment=True
    )


@runtime_checkable
class TrainingCallback(Protocol):
    """Protocol for training callback functions."""
    
    def on_training_start(self, trainer: 'RLTrainer') -> None:
        """Called when training starts."""
        ...
    
    def on_step(self, trainer: 'RLTrainer', metrics: TrainingMetrics) -> bool:
        """Called after each training step. Return False to stop training."""
        ...
    
    def on_training_end(self, trainer: 'RLTrainer', success: bool) -> None:
        """Called when training ends (success or failure)."""
        ...


class ProgressMonitor:
    """
    Real-time training progress monitoring with performance tracking.
    
    Provides comprehensive metrics collection, ETA estimation, and
    throughput measurement for RL training workflows.
    """
    
    def __init__(self, total_timesteps: int):
        self.total_timesteps = total_timesteps
        self.start_time = time.time()
        self.last_update_time = self.start_time
        
        # Metrics tracking
        self.metrics = TrainingMetrics()
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.timestep_history: List[Tuple[float, int]] = []  # (time, timesteps)
        
        # Performance monitoring
        self.fps_window = 100  # Window size for FPS calculation
        self.update_interval = 1.0  # Minimum seconds between metric updates
        
        # Thread-safe locks
        self._lock = threading.Lock()
        
        logger.info(f"Progress monitor initialized for {total_timesteps:,} timesteps")
    
    def update(self, timesteps: int, episode_rewards: Optional[List[float]] = None,
               episode_lengths: Optional[List[int]] = None, **kwargs) -> TrainingMetrics:
        """
        Update progress metrics with latest training data.
        
        Args:
            timesteps: Current total timesteps completed
            episode_rewards: List of recent episode rewards
            episode_lengths: List of recent episode lengths
            **kwargs: Additional algorithm-specific metrics
            
        Returns:
            Updated TrainingMetrics object
        """
        with self._lock:
            current_time = time.time()
            
            # Update basic progress
            self.metrics.total_timesteps = timesteps
            self.metrics.elapsed_time = current_time - self.start_time
            self.metrics.completion_percentage = (timesteps / self.total_timesteps) * 100
            
            # Update episode data
            if episode_rewards:
                self.episode_rewards.extend(episode_rewards)
                # Keep only recent episodes for efficiency
                if len(self.episode_rewards) > 1000:
                    self.episode_rewards = self.episode_rewards[-1000:]
                
                self.metrics.mean_reward = float(np.mean(self.episode_rewards[-100:]))
                self.metrics.reward_std = float(np.std(self.episode_rewards[-100:]))
                self.metrics.episode = len(self.episode_rewards)
            
            if episode_lengths:
                self.episode_lengths.extend(episode_lengths)
                if len(self.episode_lengths) > 1000:
                    self.episode_lengths = self.episode_lengths[-1000:]
                
                self.metrics.episode_length_mean = float(np.mean(self.episode_lengths[-100:]))
            
            # Update performance metrics
            if current_time - self.last_update_time >= self.update_interval:
                self._update_performance_metrics(current_time, timesteps)
                self.last_update_time = current_time
            
            # Update algorithm-specific metrics
            for key, value in kwargs.items():
                if hasattr(self.metrics, key):
                    setattr(self.metrics, key, value)
            
            # Calculate ETA
            self._calculate_eta(timesteps)
            
            return self.metrics
    
    def _update_performance_metrics(self, current_time: float, timesteps: int) -> None:
        """Update FPS and throughput metrics."""
        # Add current timestamp data point
        self.timestep_history.append((current_time, timesteps))
        
        # Keep only recent history for FPS calculation
        cutoff_time = current_time - 60.0  # Last 60 seconds
        self.timestep_history = [
            (t, ts) for t, ts in self.timestep_history if t >= cutoff_time
        ]
        
        # Calculate FPS and throughput
        if len(self.timestep_history) >= 2:
            time_diff = self.timestep_history[-1][0] - self.timestep_history[0][0]
            step_diff = self.timestep_history[-1][1] - self.timestep_history[0][1]
            
            if time_diff > 0:
                self.metrics.steps_per_second = step_diff / time_diff
                self.metrics.fps = self.metrics.steps_per_second
        
        # Calculate episodes per minute
        if self.metrics.episode > 0 and self.metrics.elapsed_time > 0:
            self.metrics.episodes_per_minute = (self.metrics.episode / self.metrics.elapsed_time) * 60
        
        # Update resource utilization
        self._update_resource_metrics()
    
    def _update_resource_metrics(self) -> None:
        """Update CPU and memory usage metrics."""
        try:
            import psutil
            process = psutil.Process()
            self.metrics.memory_usage_mb = process.memory_info().rss / 1024 / 1024
            self.metrics.cpu_percent = process.cpu_percent()
        except (ImportError, Exception) as e:
            logger.debug(f"Could not update resource metrics: {e}")
    
    def _calculate_eta(self, timesteps: int) -> None:
        """Calculate estimated time to completion."""
        if timesteps > 0 and self.metrics.steps_per_second > 0:
            remaining_steps = self.total_timesteps - timesteps
            self.metrics.eta_seconds = remaining_steps / self.metrics.steps_per_second
        else:
            self.metrics.eta_seconds = None
    
    def get_summary(self) -> str:
        """Get formatted training progress summary."""
        progress_bar = self._create_progress_bar()
        
        summary_lines = [
            f"Training Progress: {self.metrics.completion_percentage:.1f}%",
            progress_bar,
            f"Steps: {self.metrics.total_timesteps:,}/{self.total_timesteps:,}",
            f"Episodes: {self.metrics.episode:,}",
            f"Elapsed: {self._format_time(self.metrics.elapsed_time)}",
            f"ETA: {self.metrics.format_eta()}",
            f"FPS: {self.metrics.fps:.1f}",
            f"Mean Reward: {self.metrics.mean_reward:.2f} ± {self.metrics.reward_std:.2f}",
            f"Episode Length: {self.metrics.episode_length_mean:.1f}",
            f"Memory: {self.metrics.memory_usage_mb:.1f}MB"
        ]
        
        return "\n".join(summary_lines)
    
    def _create_progress_bar(self, width: int = 50) -> str:
        """Create text-based progress bar."""
        progress = self.metrics.completion_percentage / 100
        filled = int(width * progress)
        bar = "█" * filled + "░" * (width - filled)
        return f"[{bar}] {self.metrics.completion_percentage:.1f}%"
    
    def _format_time(self, seconds: float) -> str:
        """Format time duration in human-readable format."""
        hours, remainder = divmod(int(seconds), 3600)
        minutes, secs = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class AlgorithmFactory:
    """
    Factory class for creating and configuring RL algorithms.
    
    Provides optimized default configurations for different algorithms
    and supports custom hyperparameter overrides through configuration.
    """
    
    # Optimized hyperparameters for odor plume navigation
    DEFAULT_CONFIGS = {
        AlgorithmType.PPO: {
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "policy_kwargs": {
                "net_arch": [64, 64],
                "activation_fn": "tanh"
            }
        },
        
        AlgorithmType.SAC: {
            "learning_rate": 3e-4,
            "buffer_size": 100000,
            "learning_starts": 1000,
            "batch_size": 256,
            "tau": 0.005,
            "gamma": 0.99,
            "train_freq": 1,
            "gradient_steps": 1,
            "ent_coef": "auto",
            "policy_kwargs": {
                "net_arch": [256, 256],
                "activation_fn": "relu"
            }
        },
        
        AlgorithmType.TD3: {
            "learning_rate": 3e-4,
            "buffer_size": 100000,
            "learning_starts": 1000,
            "batch_size": 256,
            "tau": 0.005,
            "gamma": 0.99,
            "train_freq": (1, "step"),
            "gradient_steps": 1,
            "policy_delay": 2,
            "target_policy_noise": 0.2,
            "target_noise_clip": 0.5,
            "policy_kwargs": {
                "net_arch": [256, 256],
                "activation_fn": "relu"
            }
        },
        
        AlgorithmType.A2C: {
            "learning_rate": 7e-4,
            "n_steps": 5,
            "gamma": 0.99,
            "gae_lambda": 1.0,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "policy_kwargs": {
                "net_arch": [64, 64],
                "activation_fn": "tanh"
            }
        },
        
        AlgorithmType.DQN: {
            "learning_rate": 1e-4,
            "buffer_size": 50000,
            "learning_starts": 1000,
            "batch_size": 32,
            "tau": 1.0,
            "gamma": 0.99,
            "train_freq": 4,
            "gradient_steps": 1,
            "target_update_interval": 1000,
            "exploration_fraction": 0.1,
            "exploration_initial_eps": 1.0,
            "exploration_final_eps": 0.05,
            "policy_kwargs": {
                "net_arch": [128, 128],
                "activation_fn": "relu"
            }
        }
    }
    
    @classmethod
    def create_algorithm(
        cls,
        algorithm_type: AlgorithmType,
        env: Union[GymnasiumEnv, VecEnv],
        config: TrainingConfig,
        **kwargs
    ) -> BaseAlgorithm:
        """
        Create and configure RL algorithm instance.
        
        Args:
            algorithm_type: Type of algorithm to create
            env: Training environment (single or vectorized)
            config: Training configuration
            **kwargs: Additional algorithm-specific parameters
            
        Returns:
            Configured algorithm instance
            
        Raises:
            ValueError: If algorithm type is not supported
            ImportError: If stable-baselines3 is not available
        """
        if not SB3_AVAILABLE:
            raise ImportError(
                "stable-baselines3 is required for RL training. "
                "Install with: pip install 'odor_plume_nav[rl]'"
            )
        
        logger.info(f"Creating {algorithm_type.value.upper()} algorithm")
        
        # Get default configuration for algorithm
        default_config = cls.DEFAULT_CONFIGS.get(algorithm_type, {}).copy()
        
        # Apply user overrides
        if config.learning_rate:
            default_config["learning_rate"] = config.learning_rate
        
        if config.batch_size:
            default_config["batch_size"] = config.batch_size
        
        if config.n_epochs and algorithm_type in [AlgorithmType.PPO, AlgorithmType.A2C]:
            default_config["n_epochs"] = config.n_epochs
        
        # Apply additional kwargs
        default_config.update(kwargs)
        
        # Configure tensorboard logging
        tensorboard_log = str(config.tensorboard_log) if config.tensorboard_log else None
        
        # Create algorithm instance
        algorithm_classes = {
            AlgorithmType.PPO: PPO,
            AlgorithmType.SAC: SAC,
            AlgorithmType.TD3: TD3,
            AlgorithmType.A2C: A2C,
            AlgorithmType.DQN: DQN
        }
        
        if algorithm_type not in algorithm_classes:
            raise ValueError(f"Unsupported algorithm type: {algorithm_type}")
        
        algorithm_class = algorithm_classes[algorithm_type]
        
        # Handle policy selection based on environment observation space
        if hasattr(env, 'observation_space'):
            obs_space = env.observation_space
        else:
            # For vectorized environments, get observation space from first env
            obs_space = env.get_attr('observation_space')[0]
        
        if hasattr(obs_space, 'spaces') and isinstance(obs_space.spaces, dict):
            # Dict observation space requires MultiInputPolicy
            policy = "MultiInputPolicy"
        else:
            # Simple observation space uses MlpPolicy
            policy = "MlpPolicy"
        
        logger.info(f"Using policy: {policy}")
        logger.debug(f"Algorithm config: {default_config}")
        
        # Create algorithm with error handling
        try:
            algorithm = algorithm_class(
                policy=policy,
                env=env,
                verbose=config.verbose,
                device=config.device,
                seed=config.seed,
                tensorboard_log=tensorboard_log,
                **default_config
            )
            
            logger.info(
                f"{algorithm_type.value.upper()} algorithm created successfully",
                extra={
                    "policy": policy,
                    "device": algorithm.device,
                    "learning_rate": algorithm.learning_rate,
                    "tensorboard_log": tensorboard_log
                }
            )
            
            return algorithm
            
        except Exception as e:
            logger.error(f"Failed to create {algorithm_type.value.upper()} algorithm: {e}")
            raise


class VectorEnvManager:
    """
    Manager for vectorized environment creation and configuration.
    
    Handles both Gymnasium and stable-baselines3 vectorized environments
    with optimization for different execution patterns and performance requirements.
    """
    
    @classmethod
    def create_vec_env(
        cls,
        env_factory: Callable[[], GymnasiumEnv],
        n_envs: int,
        env_type: VectorEnvType,
        monitor_dir: Optional[Path] = None,
        seed: Optional[int] = None
    ) -> VecEnv:
        """
        Create vectorized environment with specified configuration.
        
        Args:
            env_factory: Function that creates a single environment instance
            n_envs: Number of parallel environments
            env_type: Type of vectorization to use
            monitor_dir: Directory for environment monitoring logs
            seed: Base seed for environment seeding
            
        Returns:
            Configured vectorized environment
            
        Raises:
            ValueError: If environment type is not supported
            ImportError: If required dependencies are not available
        """
        logger.info(f"Creating {env_type.value} vectorized environment with {n_envs} processes")
        
        # Create wrapper function for monitoring if directory provided
        if monitor_dir:
            monitor_dir = Path(monitor_dir)
            monitor_dir.mkdir(parents=True, exist_ok=True)
            
            def monitored_env_factory(rank: int = 0):
                env = env_factory()
                env = Monitor(env, str(monitor_dir / f"env_{rank}"))
                return env
            
            wrapped_factory = monitored_env_factory
        else:
            wrapped_factory = env_factory
        
        # Handle seeding for deterministic training
        if seed is not None:
            def seeded_env_factory(rank: int = 0):
                env = wrapped_factory()
                env.seed(seed + rank)
                return env
            
            final_factory = seeded_env_factory
        else:
            final_factory = wrapped_factory
        
        # Create vectorized environment based on type
        if env_type == VectorEnvType.DUMMY:
            # SB3 DummyVecEnv - sequential execution
            vec_env = DummyVecEnv([lambda: final_factory(i) for i in range(n_envs)])
            
        elif env_type == VectorEnvType.SUBPROC:
            # SB3 SubprocVecEnv - parallel processes
            if not SB3_AVAILABLE:
                raise ImportError("stable-baselines3 required for SubprocVecEnv")
            vec_env = SubprocVecEnv([lambda: final_factory(i) for i in range(n_envs)])
            
        elif env_type == VectorEnvType.SYNC:
            # Gymnasium SyncVectorEnv
            if not GYMNASIUM_AVAILABLE:
                raise ImportError("gymnasium required for SyncVectorEnv")
            
            # Create list of environment factories
            env_fns = [lambda: final_factory(i) for i in range(n_envs)]
            vec_env = GymSyncVectorEnv(env_fns)
            
            # Wrap in SB3 compatibility layer if needed
            from stable_baselines3.common.vec_env import VecEnvWrapper
            vec_env = VecEnvWrapper(vec_env)
            
        elif env_type == VectorEnvType.ASYNC:
            # Gymnasium AsyncVectorEnv
            if not GYMNASIUM_AVAILABLE:
                raise ImportError("gymnasium required for AsyncVectorEnv")
            
            # Create list of environment factories
            env_fns = [lambda: final_factory(i) for i in range(n_envs)]
            vec_env = GymAsyncVectorEnv(env_fns)
            
            # Wrap in SB3 compatibility layer if needed
            from stable_baselines3.common.vec_env import VecEnvWrapper
            vec_env = VecEnvWrapper(vec_env)
            
        else:
            raise ValueError(f"Unsupported vectorized environment type: {env_type}")
        
        logger.info(
            f"Vectorized environment created successfully",
            extra={
                "env_type": env_type.value,
                "n_envs": n_envs,
                "monitor_enabled": monitor_dir is not None,
                "seeded": seed is not None
            }
        )
        
        return vec_env
    
    @classmethod
    def get_optimal_env_count(cls, max_envs: Optional[int] = None) -> int:
        """
        Determine optimal number of environments based on system resources.
        
        Args:
            max_envs: Maximum number of environments to create
            
        Returns:
            Recommended number of environments
        """
        try:
            import multiprocessing
            cpu_count = multiprocessing.cpu_count()
        except Exception:
            cpu_count = 4  # Fallback
        
        # Conservative approach: use CPU count - 1 to leave system resources
        optimal = max(1, cpu_count - 1)
        
        # Apply user-specified maximum
        if max_envs:
            optimal = min(optimal, max_envs)
        
        # Performance testing suggests 8-16 environments work well for most systems
        optimal = min(optimal, 16)
        
        logger.info(f"Optimal environment count: {optimal} (CPU cores: {cpu_count})")
        return optimal


class TrainingCallbacks:
    """
    Collection of training callback implementations for monitoring and control.
    
    Provides callbacks for progress monitoring, checkpointing, evaluation,
    and integration with external monitoring systems.
    """
    
    class ProgressCallback(BaseCallback):
        """Callback for real-time progress monitoring and logging."""
        
        def __init__(self, monitor: ProgressMonitor, log_interval: int = 1000):
            super().__init__()
            self.monitor = monitor
            self.log_interval = log_interval
            self.last_log_step = 0
        
        def _on_step(self) -> bool:
            """Called after each environment step."""
            # Update progress monitor
            episode_rewards = []
            episode_lengths = []
            
            # Extract episode information if available
            if hasattr(self.training_env, 'get_attr'):
                try:
                    # Get episode rewards from monitor wrappers
                    episode_rewards = self.training_env.get_attr('episode_rewards')
                    episode_lengths = self.training_env.get_attr('episode_lengths')
                except Exception:
                    pass
            
            # Update metrics
            metrics = self.monitor.update(
                timesteps=self.num_timesteps,
                episode_rewards=episode_rewards,
                episode_lengths=episode_lengths
            )
            
            # Log progress at specified intervals
            if self.num_timesteps - self.last_log_step >= self.log_interval:
                logger.info(
                    f"Training progress: {metrics.completion_percentage:.1f}% | "
                    f"Steps: {metrics.total_timesteps:,} | "
                    f"FPS: {metrics.fps:.1f} | "
                    f"Mean reward: {metrics.mean_reward:.2f} | "
                    f"ETA: {metrics.format_eta()}"
                )
                self.last_log_step = self.num_timesteps
            
            return True
    
    class TensorBoardCallback(BaseCallback):
        """Callback for TensorBoard logging integration."""
        
        def __init__(self, log_dir: Path, monitor: ProgressMonitor):
            super().__init__()
            self.log_dir = Path(log_dir)
            self.monitor = monitor
            self.writer = None
            
            if TENSORBOARD_AVAILABLE:
                self.log_dir.mkdir(parents=True, exist_ok=True)
                self.writer = SummaryWriter(str(self.log_dir))
                logger.info(f"TensorBoard logging enabled: {self.log_dir}")
        
        def _on_step(self) -> bool:
            """Log metrics to TensorBoard."""
            if self.writer is None:
                return True
            
            metrics = self.monitor.metrics
            
            # Log scalar metrics
            self.writer.add_scalar("training/mean_reward", metrics.mean_reward, self.num_timesteps)
            self.writer.add_scalar("training/reward_std", metrics.reward_std, self.num_timesteps)
            self.writer.add_scalar("training/episode_length", metrics.episode_length_mean, self.num_timesteps)
            self.writer.add_scalar("training/fps", metrics.fps, self.num_timesteps)
            self.writer.add_scalar("training/steps_per_second", metrics.steps_per_second, self.num_timesteps)
            self.writer.add_scalar("resources/memory_mb", metrics.memory_usage_mb, self.num_timesteps)
            self.writer.add_scalar("resources/cpu_percent", metrics.cpu_percent, self.num_timesteps)
            
            # Log algorithm-specific metrics
            if metrics.loss is not None:
                self.writer.add_scalar("training/loss", metrics.loss, self.num_timesteps)
            if metrics.policy_loss is not None:
                self.writer.add_scalar("training/policy_loss", metrics.policy_loss, self.num_timesteps)
            if metrics.value_loss is not None:
                self.writer.add_scalar("training/value_loss", metrics.value_loss, self.num_timesteps)
            
            return True
        
        def _on_training_end(self) -> None:
            """Close TensorBoard writer."""
            if self.writer:
                self.writer.close()
    
    class WandBCallback(BaseCallback):
        """Callback for Weights & Biases integration."""
        
        def __init__(self, project: str, run_name: Optional[str], config: TrainingConfig, monitor: ProgressMonitor):
            super().__init__()
            self.project = project
            self.run_name = run_name
            self.config = config
            self.monitor = monitor
            self.wandb_run = None
            
            if WANDB_AVAILABLE:
                self._init_wandb()
        
        def _init_wandb(self):
            """Initialize Weights & Biases run."""
            try:
                self.wandb_run = wandb.init(
                    project=self.project,
                    name=self.run_name,
                    config=self.config.model_dump(),
                    tags=["rl_training", "odor_plume_navigation"]
                )
                logger.info(f"Weights & Biases tracking enabled: {self.project}")
            except Exception as e:
                logger.error(f"Failed to initialize Weights & Biases: {e}")
                self.wandb_run = None
        
        def _on_step(self) -> bool:
            """Log metrics to Weights & Biases."""
            if self.wandb_run is None:
                return True
            
            metrics = self.monitor.metrics
            
            # Prepare metrics dictionary
            log_dict = {
                "timesteps": self.num_timesteps,
                "mean_reward": metrics.mean_reward,
                "reward_std": metrics.reward_std,
                "episode_length": metrics.episode_length_mean,
                "fps": metrics.fps,
                "steps_per_second": metrics.steps_per_second,
                "memory_mb": metrics.memory_usage_mb,
                "cpu_percent": metrics.cpu_percent,
                "completion_percentage": metrics.completion_percentage
            }
            
            # Add algorithm-specific metrics
            if metrics.loss is not None:
                log_dict["loss"] = metrics.loss
            if metrics.policy_loss is not None:
                log_dict["policy_loss"] = metrics.policy_loss
            if metrics.value_loss is not None:
                log_dict["value_loss"] = metrics.value_loss
            
            # Log to W&B
            try:
                wandb.log(log_dict, step=self.num_timesteps)
            except Exception as e:
                logger.warning(f"Failed to log to Weights & Biases: {e}")
            
            return True
        
        def _on_training_end(self) -> None:
            """Finish Weights & Biases run."""
            if self.wandb_run:
                try:
                    wandb.finish()
                except Exception as e:
                    logger.warning(f"Error finishing W&B run: {e}")


class CheckpointManager:
    """
    Advanced model checkpointing with resumption capabilities.
    
    Manages model saving, loading, and training state restoration
    for long-running training sessions with error recovery.
    """
    
    def __init__(self, checkpoint_dir: Path, save_freq: int = 10000):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.save_freq = save_freq
        
        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # State tracking
        self.last_save_step = 0
        self.checkpoint_files: List[Path] = []
        
        logger.info(f"Checkpoint manager initialized: {self.checkpoint_dir}")
    
    def save_checkpoint(
        self,
        model: BaseAlgorithm,
        timesteps: int,
        metrics: TrainingMetrics,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Save model checkpoint with training state.
        
        Args:
            model: Algorithm instance to save
            timesteps: Current training timesteps
            metrics: Current training metrics
            metadata: Additional metadata to save
            
        Returns:
            Path to saved checkpoint file
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"checkpoint_{timesteps}_{timestamp}"
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_name}.zip"
        
        try:
            # Save model
            model.save(str(checkpoint_path))
            
            # Save additional state information
            state_info = {
                "timesteps": timesteps,
                "metrics": metrics.to_dict(),
                "timestamp": timestamp,
                "model_class": model.__class__.__name__,
                "metadata": metadata or {}
            }
            
            state_path = self.checkpoint_dir / f"{checkpoint_name}_state.json"
            with open(state_path, 'w') as f:
                json.dump(state_info, f, indent=2, default=str)
            
            # Update tracking
            self.checkpoint_files.append(checkpoint_path)
            self.last_save_step = timesteps
            
            # Cleanup old checkpoints (keep last 5)
            self._cleanup_checkpoints()
            
            logger.info(f"Checkpoint saved: {checkpoint_path}")
            return checkpoint_path
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise
    
    def load_checkpoint(self, checkpoint_path: Path) -> Tuple[BaseAlgorithm, Dict[str, Any]]:
        """
        Load model checkpoint and training state.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Tuple of (loaded_model, state_info)
            
        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            ValueError: If checkpoint is corrupted
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        try:
            # Load state information
            state_path = checkpoint_path.with_name(
                checkpoint_path.stem.replace(".zip", "") + "_state.json"
            )
            
            state_info = {}
            if state_path.exists():
                with open(state_path, 'r') as f:
                    state_info = json.load(f)
            
            logger.info(f"Loaded checkpoint: {checkpoint_path}")
            return checkpoint_path, state_info
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise ValueError(f"Corrupted checkpoint: {e}")
    
    def get_latest_checkpoint(self) -> Optional[Path]:
        """Get path to most recent checkpoint."""
        checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_*.zip"))
        
        if not checkpoint_files:
            return None
        
        # Sort by modification time
        checkpoint_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return checkpoint_files[0]
    
    def _cleanup_checkpoints(self, max_checkpoints: int = 5):
        """Remove old checkpoints, keeping the most recent ones."""
        if len(self.checkpoint_files) <= max_checkpoints:
            return
        
        # Sort by creation time
        sorted_checkpoints = sorted(
            self.checkpoint_files,
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        
        # Remove old checkpoints
        for old_checkpoint in sorted_checkpoints[max_checkpoints:]:
            try:
                old_checkpoint.unlink()
                # Also remove corresponding state file
                state_file = old_checkpoint.with_name(
                    old_checkpoint.stem.replace(".zip", "") + "_state.json"
                )
                if state_file.exists():
                    state_file.unlink()
                
                logger.debug(f"Removed old checkpoint: {old_checkpoint}")
            except Exception as e:
                logger.warning(f"Failed to remove old checkpoint {old_checkpoint}: {e}")
        
        # Update tracking list
        self.checkpoint_files = sorted_checkpoints[:max_checkpoints]


class RLTrainer:
    """
    Comprehensive RL trainer with advanced monitoring and error recovery.
    
    Main orchestrator for RL training workflows, providing a unified interface
    for algorithm training with comprehensive monitoring, checkpointing, and
    error recovery capabilities.
    """
    
    def __init__(
        self,
        env_factory: Callable[[], GymnasiumEnv],
        config: TrainingConfig,
        output_dir: Optional[Path] = None
    ):
        """
        Initialize RL trainer with configuration.
        
        Args:
            env_factory: Function that creates environment instances
            config: Training configuration
            output_dir: Directory for outputs (checkpoints, logs, etc.)
        """
        self.env_factory = env_factory
        self.config = config
        self.output_dir = Path(output_dir) if output_dir else Path("./rl_training_output")
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.log_dir = self.output_dir / "logs"
        self.tensorboard_dir = self.output_dir / "tensorboard"
        
        # Initialize components
        self.vec_env: Optional[VecEnv] = None
        self.algorithm: Optional[BaseAlgorithm] = None
        self.progress_monitor: Optional[ProgressMonitor] = None
        self.checkpoint_manager: Optional[CheckpointManager] = None
        
        # Training state
        self.status = TrainingStatus.INITIALIZING
        self.session_id = str(uuid.uuid4())[:8]
        self.start_timestep = 0
        self.callbacks: List[TrainingCallback] = []
        
        # Error tracking
        self.error_count = 0
        self.last_error: Optional[Exception] = None
        
        logger.info(
            f"RL Trainer initialized",
            extra={
                "session_id": self.session_id,
                "algorithm": config.algorithm.value,
                "total_timesteps": config.total_timesteps,
                "n_envs": config.n_envs,
                "output_dir": str(self.output_dir)
            }
        )
    
    def setup_training(self) -> None:
        """
        Initialize training components and environment.
        
        Raises:
            RuntimeError: If setup fails
        """
        try:
            logger.info("Setting up training environment and components")
            
            # Set global seed for reproducibility
            if self.config.seed is not None:
                set_global_seed(self.config.seed)
                logger.info(f"Global seed set to: {self.config.seed}")
            
            # Create vectorized environment
            self._setup_environment()
            
            # Create algorithm
            self._setup_algorithm()
            
            # Initialize monitoring and checkpointing
            self._setup_monitoring()
            
            # Setup callbacks
            self._setup_callbacks()
            
            self.status = TrainingStatus.RUNNING
            logger.info("Training setup completed successfully")
            
        except Exception as e:
            self.status = TrainingStatus.FAILED
            self.last_error = e
            logger.error(f"Training setup failed: {e}")
            raise RuntimeError(f"Failed to setup training: {e}") from e
    
    def _setup_environment(self) -> None:
        """Setup vectorized training environment."""
        logger.info(f"Creating {self.config.env_type.value} vectorized environment")
        
        self.vec_env = VectorEnvManager.create_vec_env(
            env_factory=self.env_factory,
            n_envs=self.config.n_envs,
            env_type=self.config.env_type,
            monitor_dir=self.log_dir / "monitor",
            seed=self.config.seed
        )
        
        logger.info(f"Vectorized environment created with {self.config.n_envs} environments")
    
    def _setup_algorithm(self) -> None:
        """Setup RL algorithm with configuration."""
        logger.info(f"Creating {self.config.algorithm.value.upper()} algorithm")
        
        # Set TensorBoard log directory
        if self.config.tensorboard_log is None:
            self.config.tensorboard_log = self.tensorboard_dir
        
        self.algorithm = AlgorithmFactory.create_algorithm(
            algorithm_type=self.config.algorithm,
            env=self.vec_env,
            config=self.config
        )
        
        logger.info(f"Algorithm created: {self.algorithm.__class__.__name__}")
    
    def _setup_monitoring(self) -> None:
        """Setup progress monitoring and checkpointing."""
        # Progress monitor
        self.progress_monitor = ProgressMonitor(self.config.total_timesteps)
        
        # Checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=self.checkpoint_dir,
            save_freq=self.config.checkpoint_freq
        )
        
        logger.info("Monitoring and checkpointing initialized")
    
    def _setup_callbacks(self) -> None:
        """Setup training callbacks for monitoring and logging."""
        callback_list = []
        
        # Progress monitoring callback
        progress_callback = TrainingCallbacks.ProgressCallback(
            monitor=self.progress_monitor,
            log_interval=1000
        )
        callback_list.append(progress_callback)
        
        # Checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=self.config.checkpoint_freq,
            save_path=str(self.checkpoint_dir),
            name_prefix="checkpoint"
        )
        callback_list.append(checkpoint_callback)
        
        # Evaluation callback
        eval_callback = EvalCallback(
            eval_env=self.env_factory(),
            eval_freq=self.config.eval_freq,
            n_eval_episodes=self.config.eval_episodes,
            best_model_save_path=str(self.output_dir / "best_model"),
            log_path=str(self.log_dir / "eval"),
            deterministic=True
        )
        callback_list.append(eval_callback)
        
        # TensorBoard callback
        if TENSORBOARD_AVAILABLE and self.config.tensorboard_log:
            tb_callback = TrainingCallbacks.TensorBoardCallback(
                log_dir=self.tensorboard_dir,
                monitor=self.progress_monitor
            )
            callback_list.append(tb_callback)
        
        # Weights & Biases callback
        if WANDB_AVAILABLE and self.config.wandb_project:
            wandb_callback = TrainingCallbacks.WandBCallback(
                project=self.config.wandb_project,
                run_name=self.config.wandb_run_name or f"training_{self.session_id}",
                config=self.config,
                monitor=self.progress_monitor
            )
            callback_list.append(wandb_callback)
        
        # Combine all callbacks
        self.callback_list = CallbackList(callback_list)
        
        logger.info(f"Setup {len(callback_list)} training callbacks")
    
    def train(self, resume_from_checkpoint: Optional[Path] = None) -> TrainingMetrics:
        """
        Execute training with comprehensive monitoring and error recovery.
        
        Args:
            resume_from_checkpoint: Path to checkpoint to resume from
            
        Returns:
            Final training metrics
            
        Raises:
            RuntimeError: If training fails after all retry attempts
        """
        if self.status != TrainingStatus.RUNNING:
            self.setup_training()
        
        # Handle checkpoint resumption
        if resume_from_checkpoint:
            self._resume_from_checkpoint(resume_from_checkpoint)
        
        training_successful = False
        attempt = 1
        max_attempts = self.config.max_retries + 1
        
        while attempt <= max_attempts and not training_successful:
            try:
                logger.info(f"Starting training attempt {attempt}/{max_attempts}")
                
                # Execute training
                self._execute_training()
                
                training_successful = True
                self.status = TrainingStatus.COMPLETED
                logger.info("Training completed successfully")
                
            except Exception as e:
                self.error_count += 1
                self.last_error = e
                
                logger.error(f"Training attempt {attempt} failed: {e}")
                
                if attempt < max_attempts:
                    # Save checkpoint on error if configured
                    if self.config.checkpoint_on_error and self.algorithm:
                        try:
                            self._save_error_checkpoint()
                        except Exception as checkpoint_error:
                            logger.error(f"Failed to save error checkpoint: {checkpoint_error}")
                    
                    # Wait before retry
                    retry_delay = min(30 * attempt, 300)  # Exponential backoff, max 5 minutes
                    logger.info(f"Retrying training in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    
                    attempt += 1
                else:
                    self.status = TrainingStatus.FAILED
                    logger.error(f"Training failed after {max_attempts} attempts")
                    raise RuntimeError(f"Training failed: {e}") from e
        
        # Return final metrics
        return self.progress_monitor.metrics if self.progress_monitor else TrainingMetrics()
    
    def _execute_training(self) -> None:
        """Execute the core training loop."""
        logger.info(f"Starting training for {self.config.total_timesteps:,} timesteps")
        
        # Notify callbacks of training start
        for callback in self.callbacks:
            if hasattr(callback, 'on_training_start'):
                callback.on_training_start(self)
        
        try:
            # Execute stable-baselines3 training
            self.algorithm.learn(
                total_timesteps=self.config.total_timesteps - self.start_timestep,
                callback=self.callback_list,
                log_interval=10,
                reset_num_timesteps=False
            )
            
        except Exception as e:
            logger.error(f"Training execution failed: {e}")
            raise
        
        finally:
            # Notify callbacks of training end
            success = self.status == TrainingStatus.COMPLETED
            for callback in self.callbacks:
                if hasattr(callback, 'on_training_end'):
                    callback.on_training_end(self, success)
    
    def _resume_from_checkpoint(self, checkpoint_path: Path) -> None:
        """Resume training from saved checkpoint."""
        logger.info(f"Resuming training from checkpoint: {checkpoint_path}")
        
        try:
            checkpoint_path, state_info = self.checkpoint_manager.load_checkpoint(checkpoint_path)
            
            # Load model
            self.algorithm = self.algorithm.load(str(checkpoint_path), env=self.vec_env)
            
            # Restore training state
            if state_info:
                self.start_timestep = state_info.get("timesteps", 0)
                logger.info(f"Resuming from timestep: {self.start_timestep:,}")
            
        except Exception as e:
            logger.error(f"Failed to resume from checkpoint: {e}")
            raise
    
    def _save_error_checkpoint(self) -> None:
        """Save checkpoint when training error occurs."""
        try:
            current_timesteps = getattr(self.algorithm, 'num_timesteps', 0)
            current_metrics = self.progress_monitor.metrics if self.progress_monitor else TrainingMetrics()
            
            checkpoint_path = self.checkpoint_manager.save_checkpoint(
                model=self.algorithm,
                timesteps=current_timesteps,
                metrics=current_metrics,
                metadata={
                    "error_checkpoint": True,
                    "error_message": str(self.last_error),
                    "error_count": self.error_count
                }
            )
            
            logger.info(f"Error checkpoint saved: {checkpoint_path}")
            
        except Exception as e:
            logger.error(f"Failed to save error checkpoint: {e}")
    
    def stop_training(self) -> None:
        """Stop training gracefully."""
        if self.status == TrainingStatus.RUNNING:
            self.status = TrainingStatus.STOPPED
            logger.info("Training stopped by user request")
    
    def get_progress_summary(self) -> str:
        """Get formatted training progress summary."""
        if self.progress_monitor:
            return self.progress_monitor.get_summary()
        else:
            return "Training not started"
    
    def cleanup(self) -> None:
        """Cleanup training resources."""
        try:
            if self.vec_env:
                self.vec_env.close()
            
            logger.info("Training resources cleaned up")
            
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")


# Public API factory functions

def create_trainer(
    env_config: Union[Dict[str, Any], TrainingConfig],
    training_config: Optional[TrainingConfig] = None,
    output_dir: Optional[Path] = None
) -> RLTrainer:
    """
    Factory function to create configured RL trainer.
    
    Args:
        env_config: Environment configuration or training config with env params
        training_config: Training-specific configuration (if env_config is env params)
        output_dir: Directory for training outputs
        
    Returns:
        Configured RLTrainer instance
        
    Examples:
        >>> # Basic trainer creation
        >>> trainer = create_trainer(
        ...     env_config={"video_path": "data/plume.mp4"},
        ...     training_config=TrainingConfig(algorithm=AlgorithmType.PPO)
        ... )
        >>> 
        >>> # With comprehensive configuration
        >>> env_config = {
        ...     "video_path": "experiments/plume_data.mp4",
        ...     "max_speed": 2.5,
        ...     "include_multi_sensor": True
        ... }
        >>> training_config = TrainingConfig(
        ...     algorithm=AlgorithmType.SAC,
        ...     total_timesteps=500000,
        ...     n_envs=8,
        ...     tensorboard_log=Path("./logs/tensorboard")
        ... )
        >>> trainer = create_trainer(env_config, training_config)
    """
    logger.info("Creating RL trainer with factory function")
    
    # Handle configuration types
    if isinstance(env_config, TrainingConfig) and training_config is None:
        # env_config is actually the full training config
        config = env_config
        env_params = {}
    else:
        # Separate environment and training configs
        config = training_config or TrainingConfig()
        env_params = env_config if isinstance(env_config, dict) else {}
    
    # Create environment factory
    def env_factory() -> GymnasiumEnv:
        return create_gymnasium_environment(**env_params)
    
    # Create trainer
    trainer = RLTrainer(
        env_factory=env_factory,
        config=config,
        output_dir=output_dir
    )
    
    logger.info(f"RL trainer created with {config.algorithm.value.upper()} algorithm")
    return trainer


def train_policy(
    env_config: Dict[str, Any],
    algorithm: AlgorithmType = AlgorithmType.PPO,
    total_timesteps: int = 100000,
    n_envs: int = 4,
    output_dir: Optional[Path] = None,
    **kwargs
) -> Tuple[BaseAlgorithm, TrainingMetrics]:
    """
    Simplified function for quick policy training.
    
    Args:
        env_config: Environment configuration parameters
        algorithm: Algorithm type to use
        total_timesteps: Total training timesteps
        n_envs: Number of parallel environments
        output_dir: Output directory for results
        **kwargs: Additional training configuration parameters
        
    Returns:
        Tuple of (trained_model, final_metrics)
        
    Examples:
        >>> # Quick PPO training
        >>> model, metrics = train_policy(
        ...     env_config={"video_path": "data/plume.mp4"},
        ...     algorithm=AlgorithmType.PPO,
        ...     total_timesteps=50000
        ... )
        >>> 
        >>> # Multi-environment SAC training
        >>> model, metrics = train_policy(
        ...     env_config={"video_path": "data/experiment.mp4", "max_speed": 3.0},
        ...     algorithm=AlgorithmType.SAC,
        ...     total_timesteps=200000,
        ...     n_envs=8,
        ...     learning_rate=1e-3,
        ...     tensorboard_log=Path("./logs")
        ... )
    """
    logger.info(f"Starting quick policy training with {algorithm.value.upper()}")
    
    # Create training configuration
    config = TrainingConfig(
        algorithm=algorithm,
        total_timesteps=total_timesteps,
        n_envs=n_envs,
        **kwargs
    )
    
    # Create and setup trainer
    trainer = create_trainer(env_config, config, output_dir)
    trainer.setup_training()
    
    # Execute training
    metrics = trainer.train()
    
    # Return trained model and metrics
    return trainer.algorithm, metrics


def evaluate_trained_policy(
    model: BaseAlgorithm,
    env_config: Dict[str, Any],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False
) -> Dict[str, float]:
    """
    Evaluate trained policy performance.
    
    Args:
        model: Trained algorithm instance
        env_config: Environment configuration for evaluation
        n_eval_episodes: Number of episodes to evaluate
        deterministic: Use deterministic policy
        render: Enable environment rendering
        
    Returns:
        Dictionary of evaluation metrics
        
    Examples:
        >>> # Evaluate trained model
        >>> eval_metrics = evaluate_trained_policy(
        ...     model=trained_model,
        ...     env_config={"video_path": "data/test_plume.mp4"},
        ...     n_eval_episodes=20
        ... )
        >>> print(f"Mean reward: {eval_metrics['mean_reward']:.2f}")
    """
    logger.info(f"Evaluating policy for {n_eval_episodes} episodes")
    
    # Create evaluation environment
    eval_env = create_gymnasium_environment(**env_config)
    if render:
        eval_env.render_mode = "human"
    
    try:
        # Use stable-baselines3 evaluation
        if SB3_AVAILABLE:
            mean_reward, std_reward = evaluate_policy(
                model=model,
                env=eval_env,
                n_eval_episodes=n_eval_episodes,
                deterministic=deterministic,
                render=render
            )
            
            results = {
                "mean_reward": float(mean_reward),
                "std_reward": float(std_reward),
                "n_episodes": n_eval_episodes
            }
        else:
            # Fallback manual evaluation
            results = _manual_policy_evaluation(
                model, eval_env, n_eval_episodes, deterministic
            )
        
        logger.info(f"Evaluation completed: {results}")
        return results
        
    finally:
        eval_env.close()


def _manual_policy_evaluation(
    model: BaseAlgorithm,
    env: GymnasiumEnv,
    n_episodes: int,
    deterministic: bool
) -> Dict[str, float]:
    """Manual policy evaluation fallback."""
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            episode_length += 1
            done = terminated or truncated
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
    
    return {
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "mean_length": float(np.mean(episode_lengths)),
        "n_episodes": n_episodes
    }


# Export public API
__all__ = [
    # Core classes
    "RLTrainer",
    "TrainingConfig",
    "TrainingMetrics",
    "TrainingStatus",
    "AlgorithmType",
    "VectorEnvType",
    
    # Factory classes
    "AlgorithmFactory",
    "VectorEnvManager",
    
    # Monitoring and callbacks
    "ProgressMonitor",
    "TrainingCallbacks",
    "CheckpointManager",
    
    # Factory functions
    "create_trainer",
    "train_policy",
    "evaluate_trained_policy",
    
    # Protocol
    "TrainingCallback"
]