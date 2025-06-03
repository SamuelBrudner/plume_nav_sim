"""
Reproducibility management system providing global seed management and deterministic experiment execution.

This module ensures reproducible research outcomes across different computing environments 
and experiment runs by providing centralized random state management. It integrates with 
Hydra configuration system and provides experiment-level seed control with comprehensive 
logging integration for experiment tracking.

The seed manager guarantees:
- Deterministic results across different computing environments
- Cross-platform consistency (Windows, macOS, Linux)
- Integration with Hydra configuration for seed parameter management
- Automatic seed context binding for logging injection
- Fast initialization (<100ms) for real-time experiment execution
- Random state preservation capabilities for experiment checkpointing

Examples:
    Basic seed management:
        >>> from {{cookiecutter.project_slug}}.utils.seed_manager import set_global_seed
        >>> set_global_seed(42)
        >>> # All subsequent random operations will be deterministic
        
    Context-aware seeding with logging:
        >>> from {{cookiecutter.project_slug}}.utils.seed_manager import SeedManager
        >>> with SeedManager(seed=42, experiment_id="exp_001") as sm:
        ...     # Random operations are deterministic and logged
        ...     result = run_simulation()
        
    Hydra configuration integration:
        >>> from {{cookiecutter.project_slug}}.utils.seed_manager import configure_from_hydra
        >>> configure_from_hydra(cfg)  # Automatically sets seed from config
"""

import os
import sys
import time
import random
import hashlib
import threading
from pathlib import Path
from contextlib import contextmanager
from typing import Optional, Dict, Any, Union, Generator, Tuple
from dataclasses import dataclass, field

import numpy as np

# Hydra imports with fallback for environments without Hydra
try:
    from hydra.core.hydra_config import HydraConfig
    from omegaconf import DictConfig, OmegaConf
    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False
    HydraConfig = None
    DictConfig = dict
    OmegaConf = None

# Loguru imports with fallback
try:
    from loguru import logger
    LOGURU_AVAILABLE = True
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    LOGURU_AVAILABLE = False

# Import configuration schemas
try:
    from ..config.schemas import NavigatorConfig, VideoPlumeConfig
    CONFIG_SCHEMAS_AVAILABLE = True
except ImportError:
    # Fallback for environments without config schemas
    CONFIG_SCHEMAS_AVAILABLE = False
    NavigatorConfig = dict
    VideoPlumeConfig = dict


@dataclass
class RandomState:
    """
    Comprehensive random state snapshot for reproducibility and checkpointing.
    
    Captures the complete random state from NumPy, Python random module, and 
    system-specific entropy sources to enable exact experiment reproduction
    and state restoration across different computing environments.
    
    Attributes:
        numpy_state: NumPy random generator state tuple
        python_state: Python random module state tuple
        seed_value: Original seed value used for initialization
        timestamp: State capture timestamp for tracking
        experiment_id: Associated experiment identifier
        platform_info: Platform-specific information for cross-platform validation
        state_checksum: Verification checksum for state integrity
    """
    
    numpy_state: Optional[Tuple] = None
    python_state: Optional[Tuple] = None
    seed_value: Optional[int] = None
    timestamp: float = field(default_factory=time.time)
    experiment_id: Optional[str] = None
    platform_info: Dict[str, str] = field(default_factory=dict)
    state_checksum: Optional[str] = None
    
    def __post_init__(self):
        """Generate platform information and state checksum after initialization."""
        if not self.platform_info:
            self.platform_info = {
                'platform': sys.platform,
                'python_version': sys.version_info[:3],
                'numpy_version': np.__version__,
                'architecture': sys.maxsize > 2**32 and '64bit' or '32bit',
                'byte_order': sys.byteorder
            }
        
        if not self.state_checksum:
            self.state_checksum = self._generate_checksum()
    
    def _generate_checksum(self) -> str:
        """Generate verification checksum for state integrity validation."""
        state_data = f"{self.seed_value}_{self.timestamp}_{self.experiment_id}"
        if self.numpy_state:
            state_data += f"_{hash(str(self.numpy_state))}"
        if self.python_state:
            state_data += f"_{hash(str(self.python_state))}"
        
        return hashlib.md5(state_data.encode()).hexdigest()[:16]
    
    def validate_integrity(self) -> bool:
        """Validate state integrity using checksum verification."""
        return self.state_checksum == self._generate_checksum()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert random state to dictionary for serialization."""
        return {
            'numpy_state': self.numpy_state,
            'python_state': self.python_state,
            'seed_value': self.seed_value,
            'timestamp': self.timestamp,
            'experiment_id': self.experiment_id,
            'platform_info': self.platform_info,
            'state_checksum': self.state_checksum
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RandomState':
        """Create random state from dictionary for deserialization."""
        return cls(**data)


class SeedManager:
    """
    Advanced seed manager providing global reproducibility control and experiment tracking.
    
    This class implements comprehensive random state management with Hydra configuration
    integration, cross-platform consistency, and automatic logging integration. It ensures
    deterministic experiment execution while providing experiment-level context tracking
    and state preservation capabilities.
    
    Features:
    - Global seed management for NumPy and Python random modules
    - Experiment-level context tracking with unique identifiers
    - Automatic Hydra configuration integration
    - Cross-platform consistency validation
    - Random state preservation and restoration for checkpointing
    - Comprehensive logging integration with context binding
    - Performance-optimized initialization (<100ms requirement)
    - Thread-safe operations for concurrent experiment execution
    
    Examples:
        Basic usage:
            >>> manager = SeedManager(seed=42)
            >>> manager.initialize()
            >>> # All random operations are now deterministic
            
        Context manager usage:
            >>> with SeedManager(seed=42, experiment_id="exp_001") as sm:
            ...     # Automatic initialization and cleanup
            ...     result = run_experiment()
            
        State preservation:
            >>> state = manager.capture_state()
            >>> # ... run some random operations ...
            >>> manager.restore_state(state)
            >>> # Random state is restored to previous point
    """
    
    _instances: Dict[str, 'SeedManager'] = {}
    _lock = threading.Lock()
    _global_state: Optional[RandomState] = None
    
    def __init__(
        self,
        seed: Optional[int] = None,
        experiment_id: Optional[str] = None,
        auto_initialize: bool = True,
        strict_validation: bool = True,
        enable_logging: bool = True
    ):
        """
        Initialize seed manager with experiment-specific configuration.
        
        Args:
            seed: Random seed value for reproducibility (None for auto-generation)
            experiment_id: Unique experiment identifier for tracking
            auto_initialize: Whether to automatically initialize random state
            strict_validation: Enable strict cross-platform validation
            enable_logging: Enable comprehensive logging integration
        """
        self.seed = seed or self._generate_seed()
        self.experiment_id = experiment_id or self._generate_experiment_id()
        self.strict_validation = strict_validation
        self.enable_logging = enable_logging
        
        # Performance tracking for <100ms requirement
        self._initialization_time: Optional[float] = None
        self._state_history: List[RandomState] = []
        self._context_stack: List[Dict[str, Any]] = []
        
        # Hydra integration
        self._hydra_config: Optional[DictConfig] = None
        self._config_checksum: Optional[str] = None
        
        # Thread safety
        self._local_lock = threading.Lock()
        
        # Store instance for global access
        with self._lock:
            self._instances[self.experiment_id] = self
        
        if auto_initialize:
            self.initialize()
    
    def __enter__(self) -> 'SeedManager':
        """Context manager entry with automatic state capture."""
        if not self._initialization_time:
            self.initialize()
        
        # Capture initial state for restoration on exit
        initial_state = self.capture_state()
        self._context_stack.append({
            'initial_state': initial_state,
            'experiment_id': self.experiment_id,
            'entry_time': time.time()
        })
        
        if self.enable_logging and LOGURU_AVAILABLE:
            logger.bind(
                seed_value=self.seed,
                experiment_id=self.experiment_id,
                seed_manager_context=True
            ).info(f"Entering seed manager context for experiment {self.experiment_id}")
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with automatic cleanup and state restoration."""
        if not self._context_stack:
            return
        
        context = self._context_stack.pop()
        exit_time = time.time()
        context_duration = exit_time - context['entry_time']
        
        if self.enable_logging and LOGURU_AVAILABLE:
            logger.bind(
                seed_value=self.seed,
                experiment_id=self.experiment_id,
                context_duration_ms=context_duration * 1000,
                seed_manager_context=True
            ).info(f"Exiting seed manager context for experiment {self.experiment_id}")
        
        # Optional: Restore initial state (uncomment if needed)
        # self.restore_state(context['initial_state'])
    
    def initialize(self) -> float:
        """
        Initialize global random state with performance tracking.
        
        Performs comprehensive random state initialization including NumPy random
        generator, Python random module, and platform-specific entropy sources.
        Tracks initialization time to ensure <100ms performance requirement.
        
        Returns:
            float: Initialization time in milliseconds
            
        Raises:
            RuntimeError: If initialization exceeds performance requirements
        """
        start_time = time.perf_counter()
        
        with self._local_lock:
            try:
                # Initialize NumPy random state
                np.random.seed(self.seed)
                
                # Initialize Python random module
                random.seed(self.seed)
                
                # Set environment variable for other libraries
                os.environ['PYTHONHASHSEED'] = str(self.seed)
                
                # Capture initial state after initialization
                initial_state = self.capture_state()
                self._state_history.append(initial_state)
                
                # Store global state
                with self._lock:
                    SeedManager._global_state = initial_state
                
                # Calculate initialization time
                end_time = time.perf_counter()
                self._initialization_time = (end_time - start_time) * 1000  # Convert to milliseconds
                
                # Validate performance requirement (<100ms)
                if self._initialization_time > 100:
                    warning_msg = (
                        f"Seed initialization took {self._initialization_time:.2f}ms, "
                        f"exceeding 100ms performance requirement"
                    )
                    if self.enable_logging and LOGURU_AVAILABLE:
                        logger.warning(warning_msg)
                    elif self.strict_validation:
                        raise RuntimeError(warning_msg)
                
                # Log successful initialization
                if self.enable_logging and LOGURU_AVAILABLE:
                    logger.bind(
                        seed_value=self.seed,
                        experiment_id=self.experiment_id,
                        initialization_time_ms=self._initialization_time,
                        platform=sys.platform,
                        numpy_version=np.__version__,
                        python_version=sys.version_info[:3]
                    ).info(
                        f"Seed manager initialized successfully "
                        f"(seed={self.seed}, time={self._initialization_time:.2f}ms)"
                    )
                
                return self._initialization_time
                
            except Exception as e:
                error_msg = f"Failed to initialize seed manager: {str(e)}"
                if self.enable_logging and LOGURU_AVAILABLE:
                    logger.error(error_msg)
                raise RuntimeError(error_msg) from e
    
    def capture_state(self) -> RandomState:
        """
        Capture current random state for checkpointing and reproduction.
        
        Creates a comprehensive snapshot of all random state components including
        NumPy random generator state, Python random module state, and metadata
        for cross-platform validation and experiment tracking.
        
        Returns:
            RandomState: Complete random state snapshot
        """
        try:
            numpy_state = np.random.get_state()
            python_state = random.getstate()
            
            return RandomState(
                numpy_state=numpy_state,
                python_state=python_state,
                seed_value=self.seed,
                timestamp=time.time(),
                experiment_id=self.experiment_id
            )
            
        except Exception as e:
            error_msg = f"Failed to capture random state: {str(e)}"
            if self.enable_logging and LOGURU_AVAILABLE:
                logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def restore_state(self, state: RandomState) -> bool:
        """
        Restore random state from previous snapshot for exact reproduction.
        
        Restores all random number generators to the exact state captured
        in the provided RandomState snapshot, enabling precise experiment
        reproduction and checkpoint restoration.
        
        Args:
            state: RandomState snapshot to restore
            
        Returns:
            bool: True if restoration was successful
            
        Raises:
            ValueError: If state validation fails
            RuntimeError: If state restoration fails
        """
        if not isinstance(state, RandomState):
            raise ValueError("Invalid state object provided")
        
        if self.strict_validation and not state.validate_integrity():
            raise ValueError("State integrity validation failed")
        
        try:
            with self._local_lock:
                # Restore NumPy random state
                if state.numpy_state:
                    np.random.set_state(state.numpy_state)
                
                # Restore Python random state
                if state.python_state:
                    random.setstate(state.python_state)
                
                # Update tracking information
                self.seed = state.seed_value
                self.experiment_id = state.experiment_id
                
                if self.enable_logging and LOGURU_AVAILABLE:
                    logger.bind(
                        seed_value=self.seed,
                        experiment_id=self.experiment_id,
                        restored_timestamp=state.timestamp,
                        state_checksum=state.state_checksum
                    ).info(f"Random state restored successfully for experiment {self.experiment_id}")
                
                return True
                
        except Exception as e:
            error_msg = f"Failed to restore random state: {str(e)}"
            if self.enable_logging and LOGURU_AVAILABLE:
                logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def configure_from_hydra(self, cfg: Union[DictConfig, Dict[str, Any]]) -> bool:
        """
        Configure seed manager from Hydra configuration with validation.
        
        Extracts seed configuration from Hydra config and applies it to the
        seed manager. Supports nested configuration paths and environment
        variable interpolation through Hydra's resolver system.
        
        Args:
            cfg: Hydra configuration object or dictionary
            
        Returns:
            bool: True if configuration was successful
            
        Raises:
            ValueError: If configuration validation fails
        """
        if not cfg:
            if self.enable_logging and LOGURU_AVAILABLE:
                logger.warning("Empty configuration provided to seed manager")
            return False
        
        try:
            # Store Hydra configuration
            self._hydra_config = cfg
            
            # Extract seed from various possible configuration paths
            seed_value = None
            config_paths = [
                'seed',
                'random_seed', 
                'experiment.seed',
                'simulation.seed',
                'navigator.seed',
                'reproducibility.seed'
            ]
            
            for path in config_paths:
                try:
                    if HYDRA_AVAILABLE and isinstance(cfg, DictConfig):
                        seed_value = OmegaConf.select(cfg, path)
                    else:
                        # Handle regular dictionaries
                        keys = path.split('.')
                        value = cfg
                        for key in keys:
                            if isinstance(value, dict) and key in value:
                                value = value[key]
                            else:
                                value = None
                                break
                        seed_value = value
                    
                    if seed_value is not None:
                        break
                        
                except Exception:
                    continue
            
            # Apply seed if found
            if seed_value is not None:
                self.seed = int(seed_value)
                
                # Re-initialize with new seed
                self.initialize()
                
                if self.enable_logging and LOGURU_AVAILABLE:
                    logger.bind(
                        seed_value=self.seed,
                        experiment_id=self.experiment_id,
                        config_source="hydra"
                    ).info(f"Seed manager configured from Hydra config (seed={self.seed})")
                
                return True
            else:
                if self.enable_logging and LOGURU_AVAILABLE:
                    logger.info("No seed configuration found in Hydra config, using default")
                return False
                
        except Exception as e:
            error_msg = f"Failed to configure from Hydra config: {str(e)}"
            if self.enable_logging and LOGURU_AVAILABLE:
                logger.error(error_msg)
            raise ValueError(error_msg) from e
    
    def bind_to_logger(self) -> Dict[str, Any]:
        """
        Create logger context binding for automatic seed injection.
        
        Generates a dictionary of seed-related context information that can
        be bound to Loguru logger instances for automatic injection into all
        log records within the seed manager's scope.
        
        Returns:
            Dict[str, Any]: Logger context binding dictionary
        """
        context = {
            'seed_value': self.seed,
            'experiment_id': self.experiment_id,
            'seed_manager_active': True,
            'platform': sys.platform,
            'initialization_time_ms': self._initialization_time
        }
        
        # Add Hydra context if available
        if self._hydra_config and HYDRA_AVAILABLE:
            try:
                hydra_context = HydraConfig.get()
                context.update({
                    'hydra_job_name': hydra_context.job.name,
                    'hydra_config_name': hydra_context.job.config_name,
                    'hydra_output_dir': hydra_context.runtime.output_dir
                })
            except Exception:
                # Hydra context not available, skip
                pass
        
        return context
    
    def get_reproducibility_info(self) -> Dict[str, Any]:
        """
        Generate comprehensive reproducibility information for experiment documentation.
        
        Returns:
            Dict[str, Any]: Complete reproducibility metadata
        """
        info = {
            'seed_value': self.seed,
            'experiment_id': self.experiment_id,
            'initialization_time_ms': self._initialization_time,
            'platform_info': {
                'platform': sys.platform,
                'python_version': list(sys.version_info[:3]),
                'numpy_version': np.__version__,
                'architecture': sys.maxsize > 2**32 and '64bit' or '32bit',
                'byte_order': sys.byteorder
            },
            'environment_variables': {
                'PYTHONHASHSEED': os.environ.get('PYTHONHASHSEED'),
                'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES'),
                'OMP_NUM_THREADS': os.environ.get('OMP_NUM_THREADS')
            },
            'timestamp': time.time(),
            'state_history_count': len(self._state_history)
        }
        
        # Add current random state checksums for validation
        current_state = self.capture_state()
        info['current_state_checksum'] = current_state.state_checksum
        
        return info
    
    @staticmethod
    def _generate_seed() -> int:
        """Generate cryptographically secure random seed."""
        return int.from_bytes(os.urandom(4), byteorder='big') % (2**31)
    
    @staticmethod
    def _generate_experiment_id() -> str:
        """Generate unique experiment identifier."""
        timestamp = int(time.time() * 1000000)  # Microsecond precision
        random_component = random.randint(1000, 9999)
        return f"exp_{timestamp}_{random_component}"
    
    @classmethod
    def get_global_state(cls) -> Optional[RandomState]:
        """Get the current global random state."""
        with cls._lock:
            return cls._global_state
    
    @classmethod
    def get_instance(cls, experiment_id: str) -> Optional['SeedManager']:
        """Get seed manager instance by experiment ID."""
        with cls._lock:
            return cls._instances.get(experiment_id)
    
    @classmethod
    def list_active_experiments(cls) -> List[str]:
        """List all active experiment IDs."""
        with cls._lock:
            return list(cls._instances.keys())


# Global convenience functions for backward compatibility and simple usage

_global_manager: Optional[SeedManager] = None
_global_lock = threading.Lock()


def set_global_seed(
    seed: int,
    experiment_id: Optional[str] = None,
    enable_logging: bool = True
) -> SeedManager:
    """
    Set global random seed with comprehensive reproducibility management.
    
    This function provides a simple interface for global seed management while
    maintaining all advanced features of the SeedManager class. It ensures
    deterministic behavior across NumPy, Python random module, and other
    random number generators used throughout the system.
    
    Args:
        seed: Random seed value for reproducibility
        experiment_id: Optional experiment identifier for tracking
        enable_logging: Enable comprehensive logging integration
        
    Returns:
        SeedManager: Global seed manager instance
        
    Examples:
        >>> set_global_seed(42)
        >>> # All subsequent random operations are deterministic
        
        >>> manager = set_global_seed(42, experiment_id="test_run")
        >>> state = manager.capture_state()
        >>> # ... perform random operations ...
        >>> manager.restore_state(state)
    """
    global _global_manager
    
    with _global_lock:
        _global_manager = SeedManager(
            seed=seed,
            experiment_id=experiment_id,
            auto_initialize=True,
            enable_logging=enable_logging
        )
        
        if enable_logging and LOGURU_AVAILABLE:
            # Bind seed context to logger for automatic injection
            logger.configure(extra=_global_manager.bind_to_logger())
    
    return _global_manager


def get_global_seed_manager() -> Optional[SeedManager]:
    """
    Get the current global seed manager instance.
    
    Returns:
        Optional[SeedManager]: Global seed manager or None if not initialized
    """
    global _global_manager
    with _global_lock:
        return _global_manager


def configure_from_hydra(cfg: Union[DictConfig, Dict[str, Any]]) -> bool:
    """
    Configure global seed management from Hydra configuration.
    
    Convenience function for integrating seed management with Hydra-based
    configuration systems. Automatically extracts seed configuration and
    initializes global seed management.
    
    Args:
        cfg: Hydra configuration object or dictionary
        
    Returns:
        bool: True if configuration was successful
        
    Examples:
        >>> from hydra import compose, initialize
        >>> with initialize(config_path="../conf"):
        ...     cfg = compose(config_name="config")
        ...     configure_from_hydra(cfg)
    """
    global _global_manager
    
    # Create global manager if it doesn't exist
    if not _global_manager:
        with _global_lock:
            _global_manager = SeedManager(auto_initialize=False)
    
    return _global_manager.configure_from_hydra(cfg)


@contextmanager
def seed_context(
    seed: int,
    experiment_id: Optional[str] = None
) -> Generator[SeedManager, None, None]:
    """
    Context manager for temporary seed management with automatic restoration.
    
    Provides a context-managed approach to seed management that automatically
    captures the initial random state, applies the specified seed, and restores
    the original state upon context exit.
    
    Args:
        seed: Random seed value for the context
        experiment_id: Optional experiment identifier
        
    Yields:
        SeedManager: Configured seed manager instance
        
    Examples:
        >>> with seed_context(42) as sm:
        ...     # All random operations use seed 42
        ...     result = np.random.random(10)
        >>> # Original random state is restored
    """
    # Capture current state if global manager exists
    original_state = None
    if _global_manager:
        original_state = _global_manager.capture_state()
    
    # Create temporary seed manager
    temp_manager = SeedManager(
        seed=seed,
        experiment_id=experiment_id,
        auto_initialize=True
    )
    
    try:
        yield temp_manager
    finally:
        # Restore original state if it existed
        if original_state and _global_manager:
            _global_manager.restore_state(original_state)


def get_reproducibility_report() -> Dict[str, Any]:
    """
    Generate comprehensive reproducibility report for experiment documentation.
    
    Returns:
        Dict[str, Any]: Complete reproducibility metadata and system information
    """
    report = {
        'timestamp': time.time(),
        'system_info': {
            'platform': sys.platform,
            'python_version': list(sys.version_info[:3]),
            'numpy_version': np.__version__,
            'architecture': sys.maxsize > 2**32 and '64bit' or '32bit',
            'byte_order': sys.byteorder
        },
        'environment_variables': {
            key: os.environ.get(key) for key in [
                'PYTHONHASHSEED', 'CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS',
                'NUMBA_DISABLE_JIT', 'PYTHONDONTWRITEBYTECODE'
            ] if os.environ.get(key)
        },
        'active_experiments': SeedManager.list_active_experiments()
    }
    
    # Add global manager information if available
    if _global_manager:
        report['global_seed_manager'] = _global_manager.get_reproducibility_info()
    
    # Add global state information
    global_state = SeedManager.get_global_state()
    if global_state:
        report['global_random_state'] = {
            'seed_value': global_state.seed_value,
            'checksum': global_state.state_checksum,
            'timestamp': global_state.timestamp
        }
    
    return report


# Export public API
__all__ = [
    'SeedManager',
    'RandomState', 
    'set_global_seed',
    'get_global_seed_manager',
    'configure_from_hydra',
    'seed_context',
    'get_reproducibility_report'
]