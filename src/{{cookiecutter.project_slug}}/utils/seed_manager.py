"""
Reproducibility management system for deterministic experiment execution.

This module provides global seed management and experiment-level reproducibility
controls ensuring consistent results across different computing environments and
experiment runs. Integrates with Hydra configuration system and Loguru logging
for comprehensive experiment tracking and reproducibility validation.

Features:
- Global seed management with NumPy and Python random state control
- Hydra configuration integration for seed parameter management
- Cross-platform consistency ensuring deterministic results across environments
- Random state preservation capabilities for experiment checkpointing
- Automatic seed context binding for logging system integration
- Initialization timing and seed value logging for experiment tracking
"""

import os
import sys
import time
import random
import hashlib
import platform
from typing import Optional, Dict, Any, Union, ContextManager, Callable
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from loguru import logger
from hydra.core.config_store import ConfigStore
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from ..config.schemas import BaseModel, Field


class SeedConfig(BaseModel):
    """Configuration schema for seed management."""
    
    seed: Optional[int] = Field(
        default=None,
        description="Global random seed for reproducible experiments. If None, generates from system entropy."
    )
    numpy_seed: Optional[int] = Field(
        default=None,
        description="Specific seed for NumPy random generator. Defaults to global seed if not provided."
    )
    python_seed: Optional[int] = Field(
        default=None,
        description="Specific seed for Python random module. Defaults to global seed if not provided."
    )
    auto_seed: bool = Field(
        default=True,
        description="Automatically generate seed from system entropy if no seed provided."
    )
    hash_environment: bool = Field(
        default=True,
        description="Include environment characteristics in seed generation for better cross-platform consistency."
    )
    validate_initialization: bool = Field(
        default=True,
        description="Perform validation of random state initialization and log initial values."
    )
    preserve_state: bool = Field(
        default=False,
        description="Enable random state preservation for experiment checkpointing."
    )
    log_seed_context: bool = Field(
        default=True,
        description="Automatically inject seed context into all log records."
    )

    class Config:
        extra = "forbid"  # Strict validation for reproducibility


class SeedManager:
    """
    Global seed management system for reproducible experiment execution.
    
    Provides centralized control over random number generation across NumPy,
    Python's random module, and other stochastic components. Integrates with
    Hydra configuration system and Loguru logging for comprehensive experiment
    tracking.
    
    Features:
    - Global seed management with environment-aware generation
    - Cross-platform deterministic execution guarantees
    - Random state preservation and restoration capabilities
    - Automatic logging context injection for experiment tracking
    - Performance-optimized initialization (<100ms requirement)
    """
    
    _instance: Optional['SeedManager'] = None
    _initialized: bool = False
    _current_seed: Optional[int] = None
    _numpy_generator: Optional[np.random.Generator] = None
    _initial_state: Optional[Dict[str, Any]] = None
    _run_id: Optional[str] = None
    _environment_hash: Optional[str] = None
    
    def __new__(cls) -> 'SeedManager':
        """Singleton implementation ensuring single global seed manager."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize seed manager (only once due to singleton pattern)."""
        if not self._initialized:
            self._logger = logger.bind(module=__name__)
            self._config: Optional[SeedConfig] = None
            self._start_time: Optional[float] = None
            SeedManager._initialized = True
    
    @classmethod
    def reset(cls) -> None:
        """Reset singleton instance for testing purposes."""
        cls._instance = None
        cls._initialized = False
        cls._current_seed = None
        cls._numpy_generator = None
        cls._initial_state = None
        cls._run_id = None
        cls._environment_hash = None
    
    def initialize(
        self,
        config: Optional[Union[SeedConfig, DictConfig, Dict[str, Any]]] = None,
        run_id: Optional[str] = None
    ) -> int:
        """
        Initialize global seed management system.
        
        Args:
            config: Seed configuration (SeedConfig, DictConfig, or dict).
                   If None, attempts to load from Hydra global config.
            run_id: Unique run identifier for experiment tracking.
                    If None, generates from seed and timestamp.
        
        Returns:
            The initialized global seed value.
        
        Raises:
            ValueError: If seed initialization fails validation.
            RuntimeError: If initialization exceeds performance requirements.
        """
        start_time = time.perf_counter()
        self._start_time = start_time
        
        try:
            # Load configuration
            self._config = self._load_config(config)
            
            # Generate or validate seed
            seed = self._determine_seed()
            
            # Initialize random number generators
            self._initialize_generators(seed)
            
            # Generate run ID and environment hash
            self._run_id = run_id or self._generate_run_id(seed)
            self._environment_hash = self._generate_environment_hash()
            
            # Store initial state for preservation
            if self._config.preserve_state:
                self._initial_state = self._capture_state()
            
            # Validate initialization if requested
            if self._config.validate_initialization:
                self._validate_initialization()
            
            # Setup logging context binding
            if self._config.log_seed_context:
                self._setup_logging_context()
            
            # Performance validation
            initialization_time = (time.perf_counter() - start_time) * 1000
            if initialization_time > 100:  # 100ms requirement
                self._logger.warning(
                    f"Seed initialization exceeded performance requirement: {initialization_time:.2f}ms > 100ms"
                )
            
            self._logger.info(
                f"Seed manager initialized successfully",
                extra={
                    "seed": seed,
                    "run_id": self._run_id,
                    "initialization_time_ms": f"{initialization_time:.2f}",
                    "environment_hash": self._environment_hash,
                    "numpy_version": np.__version__,
                    "platform": platform.platform(),
                }
            )
            
            return seed
            
        except Exception as e:
            initialization_time = (time.perf_counter() - start_time) * 1000
            self._logger.error(
                f"Seed manager initialization failed after {initialization_time:.2f}ms: {e}",
                extra={"error_type": type(e).__name__}
            )
            raise RuntimeError(f"Seed manager initialization failed: {e}") from e
    
    def _load_config(self, config: Optional[Union[SeedConfig, DictConfig, Dict[str, Any]]]) -> SeedConfig:
        """Load and validate seed configuration."""
        if config is None:
            # Attempt to load from Hydra global config
            config = self._load_from_hydra()
        
        if isinstance(config, SeedConfig):
            return config
        elif isinstance(config, (DictConfig, dict)):
            return SeedConfig(**dict(config))
        else:
            self._logger.warning("No valid seed configuration found, using defaults")
            return SeedConfig()
    
    def _load_from_hydra(self) -> Dict[str, Any]:
        """Load seed configuration from Hydra global config."""
        try:
            if GlobalHydra().is_initialized():
                hydra_cfg = GlobalHydra.instance().cfg
                if hydra_cfg and "seed_manager" in hydra_cfg:
                    return OmegaConf.to_container(hydra_cfg.seed_manager, resolve=True)
                elif hydra_cfg and "seed" in hydra_cfg:
                    # Fallback to direct seed parameter
                    return {"seed": hydra_cfg.seed}
        except Exception as e:
            self._logger.debug(f"Could not load seed config from Hydra: {e}")
        
        return {}
    
    def _determine_seed(self) -> int:
        """Determine the global seed value based on configuration."""
        if self._config.seed is not None:
            seed = self._config.seed
            self._logger.debug(f"Using configured seed: {seed}")
        elif self._config.auto_seed:
            seed = self._generate_entropy_seed()
            self._logger.debug(f"Generated entropy-based seed: {seed}")
        else:
            raise ValueError("No seed provided and auto_seed is disabled")
        
        # Validate seed range
        if not (0 <= seed <= 2**32 - 1):
            raise ValueError(f"Seed must be in range [0, 2^32-1], got: {seed}")
        
        SeedManager._current_seed = seed
        return seed
    
    def _generate_entropy_seed(self) -> int:
        """Generate seed from system entropy with optional environment hashing."""
        # Base entropy from system
        entropy_sources = [
            os.urandom(8),
            str(time.time_ns()).encode(),
            str(os.getpid()).encode(),
        ]
        
        if self._config.hash_environment:
            # Add environment characteristics for cross-platform consistency
            env_data = [
                platform.platform().encode(),
                platform.python_version().encode(),
                np.__version__.encode(),
                str(sys.maxsize).encode(),  # Architecture indicator
            ]
            entropy_sources.extend(env_data)
        
        # Combine entropy sources
        combined_entropy = b''.join(entropy_sources)
        
        # Generate deterministic seed from entropy
        hash_obj = hashlib.sha256(combined_entropy)
        seed = int.from_bytes(hash_obj.digest()[:4], byteorder='big')
        
        return seed
    
    def _initialize_generators(self, seed: int) -> None:
        """Initialize NumPy and Python random generators."""
        # Python random module
        python_seed = self._config.python_seed or seed
        random.seed(python_seed)
        
        # NumPy random generator (modern API)
        numpy_seed = self._config.numpy_seed or seed
        self._numpy_generator = np.random.default_rng(numpy_seed)
        
        # Set global NumPy random state for legacy compatibility
        np.random.seed(numpy_seed)
        
        self._logger.debug(
            f"Initialized random generators",
            extra={
                "python_seed": python_seed,
                "numpy_seed": numpy_seed,
                "generator_type": type(self._numpy_generator).__name__
            }
        )
    
    def _generate_run_id(self, seed: int) -> str:
        """Generate unique run identifier."""
        timestamp = int(time.time() * 1000)  # Millisecond precision
        run_data = f"{seed}_{timestamp}_{os.getpid()}"
        run_hash = hashlib.md5(run_data.encode()).hexdigest()[:8]
        return f"run_{run_hash}"
    
    def _generate_environment_hash(self) -> str:
        """Generate environment characteristics hash for reproducibility tracking."""
        env_data = [
            platform.platform(),
            platform.python_version(),
            np.__version__,
            str(sys.maxsize),
            str(sys.byteorder),
        ]
        
        combined = '_'.join(env_data)
        return hashlib.md5(combined.encode()).hexdigest()[:8]
    
    def _capture_state(self) -> Dict[str, Any]:
        """Capture current random state for preservation."""
        return {
            'python_state': random.getstate(),
            'numpy_legacy_state': np.random.get_state(),
            'numpy_generator_state': self._numpy_generator.bit_generator.state if self._numpy_generator else None,
            'seed': self._current_seed,
            'timestamp': time.time()
        }
    
    def _validate_initialization(self) -> None:
        """Validate random state initialization and log initial values."""
        # Test deterministic behavior
        test_samples = {
            'python_random': random.random(),
            'numpy_legacy': np.random.random(),
            'numpy_generator': self._numpy_generator.random() if self._numpy_generator else None
        }
        
        self._logger.debug(
            "Random state validation samples",
            extra={
                "validation_samples": test_samples,
                "seed": self._current_seed
            }
        )
        
        # Additional validation: ensure generators are properly seeded
        if self._numpy_generator is None:
            raise RuntimeError("NumPy generator not properly initialized")
    
    def _setup_logging_context(self) -> None:
        """Setup automatic seed context injection into log records."""
        def add_seed_context(record):
            """Add seed context to log record."""
            if 'extra' not in record:
                record['extra'] = {}
            
            record['extra'].update({
                'seed': self._current_seed,
                'run_id': self._run_id,
                'environment_hash': self._environment_hash
            })
            
            return record
        
        # Configure Loguru to include seed context
        logger.configure(patcher=add_seed_context)
        
        self._logger.debug("Seed context binding enabled for logging system")
    
    @property
    def current_seed(self) -> Optional[int]:
        """Get current global seed value."""
        return self._current_seed
    
    @property
    def run_id(self) -> Optional[str]:
        """Get current run identifier."""
        return self._run_id
    
    @property
    def environment_hash(self) -> Optional[str]:
        """Get environment characteristics hash."""
        return self._environment_hash
    
    @property
    def numpy_generator(self) -> Optional[np.random.Generator]:
        """Get current NumPy random generator."""
        return self._numpy_generator
    
    def get_state(self) -> Optional[Dict[str, Any]]:
        """
        Get current random state for checkpointing.
        
        Returns:
            Dictionary containing current random states, or None if preserve_state is disabled.
        """
        if not self._config or not self._config.preserve_state:
            return None
        
        return self._capture_state()
    
    def restore_state(self, state: Dict[str, Any]) -> None:
        """
        Restore random state from checkpoint.
        
        Args:
            state: State dictionary from get_state().
        
        Raises:
            ValueError: If state format is invalid.
            RuntimeError: If state restoration fails.
        """
        if not self._config or not self._config.preserve_state:
            raise RuntimeError("State preservation not enabled")
        
        try:
            # Restore Python random state
            if 'python_state' in state:
                random.setstate(state['python_state'])
            
            # Restore NumPy legacy state
            if 'numpy_legacy_state' in state:
                np.random.set_state(state['numpy_legacy_state'])
            
            # Restore NumPy generator state (if available)
            if 'numpy_generator_state' in state and state['numpy_generator_state']:
                if self._numpy_generator:
                    self._numpy_generator.bit_generator.state = state['numpy_generator_state']
            
            # Update current seed tracking
            if 'seed' in state:
                SeedManager._current_seed = state['seed']
            
            self._logger.info(
                "Random state restored from checkpoint",
                extra={
                    "restored_seed": state.get('seed'),
                    "checkpoint_timestamp": state.get('timestamp')
                }
            )
            
        except Exception as e:
            self._logger.error(f"Failed to restore random state: {e}")
            raise RuntimeError(f"State restoration failed: {e}") from e
    
    @contextmanager
    def temporary_seed(self, seed: int) -> ContextManager[int]:
        """
        Context manager for temporary seed override.
        
        Args:
            seed: Temporary seed value.
        
        Yields:
            The temporary seed value.
        
        Example:
            with seed_manager.temporary_seed(42):
                # Operations using seed 42
                result = np.random.random()
        """
        if not self._config or not self._config.preserve_state:
            raise RuntimeError("Temporary seed requires preserve_state=True")
        
        # Save current state
        original_state = self._capture_state()
        original_seed = self._current_seed
        
        try:
            # Set temporary seed
            self._initialize_generators(seed)
            SeedManager._current_seed = seed
            
            self._logger.debug(
                f"Entering temporary seed context",
                extra={"temporary_seed": seed, "original_seed": original_seed}
            )
            
            yield seed
            
        finally:
            # Restore original state
            self.restore_state(original_state)
            
            self._logger.debug(
                f"Exiting temporary seed context",
                extra={"restored_seed": original_seed}
            )
    
    def generate_experiment_seeds(self, count: int, base_seed: Optional[int] = None) -> list[int]:
        """
        Generate deterministic sequence of seeds for experiment runs.
        
        Args:
            count: Number of seeds to generate.
            base_seed: Base seed for generation. Uses current seed if None.
        
        Returns:
            List of deterministic seed values.
        """
        seed = base_seed or self._current_seed
        if seed is None:
            raise RuntimeError("No seed available for experiment seed generation")
        
        # Use separate generator to avoid affecting main random state
        temp_gen = np.random.default_rng(seed)
        
        # Generate sequence of seeds
        seeds = temp_gen.integers(0, 2**32 - 1, size=count, dtype=np.uint32).tolist()
        
        self._logger.debug(
            f"Generated {count} experiment seeds",
            extra={
                "base_seed": seed,
                "generated_count": count,
                "first_seed": seeds[0] if seeds else None
            }
        )
        
        return seeds
    
    def validate_reproducibility(self, reference_values: Dict[str, float], tolerance: float = 1e-10) -> bool:
        """
        Validate reproducibility by comparing with reference values.
        
        Args:
            reference_values: Expected values for validation.
            tolerance: Numerical tolerance for floating-point comparison.
        
        Returns:
            True if validation passes, False otherwise.
        """
        current_state = self._capture_state()
        
        try:
            # Reset to initial state
            if self._initial_state:
                self.restore_state(self._initial_state)
            else:
                # Re-initialize with current seed
                self._initialize_generators(self._current_seed)
            
            # Generate test values
            test_values = {
                'python_random': random.random(),
                'numpy_legacy': np.random.random(),
                'numpy_generator': self._numpy_generator.random() if self._numpy_generator else 0.0
            }
            
            # Compare with reference
            validation_passed = True
            for key, expected in reference_values.items():
                if key in test_values:
                    actual = test_values[key]
                    if abs(actual - expected) > tolerance:
                        validation_passed = False
                        self._logger.error(
                            f"Reproducibility validation failed for {key}",
                            extra={
                                "expected": expected,
                                "actual": actual,
                                "difference": abs(actual - expected),
                                "tolerance": tolerance
                            }
                        )
            
            if validation_passed:
                self._logger.info("Reproducibility validation passed")
            
            return validation_passed
            
        finally:
            # Restore current state
            self.restore_state(current_state)


# Global seed manager instance
_global_seed_manager: Optional[SeedManager] = None


def get_seed_manager() -> SeedManager:
    """Get the global seed manager instance."""
    global _global_seed_manager
    if _global_seed_manager is None:
        _global_seed_manager = SeedManager()
    return _global_seed_manager


def set_global_seed(
    seed: Optional[int] = None,
    config: Optional[Union[SeedConfig, DictConfig, Dict[str, Any]]] = None,
    run_id: Optional[str] = None
) -> int:
    """
    Set global seed for reproducible experiments.
    
    Convenience function for initializing global seed management.
    
    Args:
        seed: Global seed value. If None, uses config or auto-generation.
        config: Seed configuration. If None, uses defaults.
        run_id: Experiment run identifier.
    
    Returns:
        The initialized global seed value.
    
    Example:
        # Basic usage
        seed = set_global_seed(42)
        
        # With configuration
        config = SeedConfig(seed=42, validate_initialization=True)
        seed = set_global_seed(config=config)
        
        # Auto-generation
        seed = set_global_seed()  # Uses entropy-based generation
    """
    seed_manager = get_seed_manager()
    
    # If seed provided directly, create config
    if seed is not None and config is None:
        config = SeedConfig(seed=seed)
    elif seed is not None and config is not None:
        # Override config seed with provided value
        if isinstance(config, dict):
            config = dict(config)
            config['seed'] = seed
        elif isinstance(config, DictConfig):
            config = OmegaConf.to_container(config)
            config['seed'] = seed
        elif isinstance(config, SeedConfig):
            config = config.model_copy(update={'seed': seed})
    
    return seed_manager.initialize(config=config, run_id=run_id)


def get_current_seed() -> Optional[int]:
    """Get current global seed value."""
    seed_manager = get_seed_manager()
    return seed_manager.current_seed


def get_numpy_generator() -> Optional[np.random.Generator]:
    """Get current NumPy random generator."""
    seed_manager = get_seed_manager()
    return seed_manager.numpy_generator


# Register configuration schema with Hydra
cs = ConfigStore.instance()
cs.store(name="seed_manager_config", node=SeedConfig)


__all__ = [
    "SeedConfig",
    "SeedManager", 
    "get_seed_manager",
    "set_global_seed",
    "get_current_seed",
    "get_numpy_generator",
]