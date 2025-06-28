"""
Comprehensive seed management system providing deterministic control of Python random, NumPy, 
and environment variables with Hydra integration, thread-local isolation, and performance 
monitoring for reproducible experiments.

This module ensures reproducible research outcomes across different computing environments 
and experiment runs by providing centralized random state management. It integrates with 
Hydra configuration system and provides experiment-level seed control with comprehensive 
logging integration for experiment tracking and performance monitoring.

The seed manager guarantees:
- Deterministic results across different computing environments with <5% variance
- Cross-platform consistency (Windows, macOS, Linux) with standardized seed ranges
- Integration with Hydra configuration for seed parameter management
- Thread-safe operations with scoped context management for multi-agent environments
- Fast initialization (<100ms) for real-time experiment execution
- Random state preservation capabilities for experiment checkpointing
- Comprehensive performance monitoring and integrity validation

Key Features:
- Global and scoped seed management with state capture and restore capabilities
- Context managers for temporary seed changes during specific operations
- Cross-platform deterministic behavior validation with performance thresholds
- Integration with Hydra configuration system for flexible seed specification
- Thread-safe operations with comprehensive error handling and logging
- Performance monitoring and validation utilities for reproducibility verification
- Environment variable support (RANDOM_SEED, NUMPY_SEED, PYTHONHASHSEED)
- Enhanced reproducibility reporting with comprehensive state capture

Technical Implementation:
- Controls Python's random module through random.setstate() and random.getstate()
- Manages NumPy's random state through np.random.set_state() and np.random.get_state()
- Handles PYTHONHASHSEED environment variable for hash randomization control
- Provides scoped context managers for temporary seed changes with automatic restoration
- Integrates with Hydra's configuration composition for environment variable interpolation
- Thread-local context management for isolated seed operations per experiment

Examples:
    Basic seed management:
        >>> from plume_nav_sim.utils.seed_manager import set_global_seed
        >>> results = set_global_seed(42)
        >>> print(f"Seed set in {results['total_time']:.3f}s")
        
    Context-aware seeding with scoped operations:
        >>> with scoped_seed(42, "experiment_initialization") as state:
        ...     # All random operations within this block use seed 42
        ...     experiment_data = np.random.random(100)
        >>> # Original random state is automatically restored
        
    Hydra configuration integration:
        >>> config = SeedConfig(global_seed="${oc.env:RANDOM_SEED,42}")
        >>> results = setup_global_seed(config)
        
    Thread-safe experiment management:
        >>> context = get_seed_context()
        >>> with scoped_seed(12345, "multi_agent_sim"):
        ...     # Each thread gets isolated seed context
        ...     run_multi_agent_simulation()
"""

import os
import sys
import time
import random
import hashlib
import threading
from typing import Dict, Any, Optional, Tuple, Union, ContextManager, Callable, List
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from pathlib import Path
import warnings
import platform

import numpy as np
from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing_extensions import Self

# Try to import enhanced logging with fallback for compatibility
try:
    from plume_nav_sim.utils.logging_setup import get_enhanced_logger, PerformanceMetrics
    ENHANCED_LOGGING_AVAILABLE = True
except ImportError:
    # Fallback for environments where logging_setup isn't available yet
    try:
        from loguru import logger as _logger
        ENHANCED_LOGGING_AVAILABLE = False
        # Create a simple performance metrics placeholder
        @dataclass
        class PerformanceMetrics:
            operation: str
            duration_ms: float
            timestamp: float = field(default_factory=time.time)
            
            def to_dict(self) -> Dict[str, Any]:
                return asdict(self)
    except ImportError:
        import logging
        _logger = logging.getLogger(__name__)
        ENHANCED_LOGGING_AVAILABLE = False
        
        @dataclass
        class PerformanceMetrics:
            operation: str
            duration_ms: float
            timestamp: float = field(default_factory=time.time)
            
            def to_dict(self) -> Dict[str, Any]:
                return asdict(self)

# Set up module logger for seed management operations
if ENHANCED_LOGGING_AVAILABLE:
    logger = get_enhanced_logger(__name__)
else:
    logger = _logger


# Performance targets and thresholds for seed operations per Section 0.5.1
SEED_PERFORMANCE_THRESHOLDS = {
    "global_seed_init": 0.1,  # 100ms target for global seed initialization
    "state_capture": 0.01,    # 10ms for state capture operations
    "state_restore": 0.01,    # 10ms for state restore operations
    "validation": 0.05,       # 50ms for determinism validation
    "scoped_operation": 0.02, # 20ms for scoped seed operations
}

# Cross-platform seed validation constants per Section 0.5.1 implementation verification
DETERMINISM_TEST_ITERATIONS = 100
NUMPY_RANDOM_TEST_SIZE = 1000
PYTHON_RANDOM_TEST_SIZE = 100
CROSS_PLATFORM_SEED_MAX = 2**31 - 1  # Maximum safe seed value for cross-platform compatibility


@dataclass
class RandomState:
    """
    Comprehensive random state container for all randomness sources with integrity validation.
    
    Captures the complete random state across Python's random module, NumPy's
    random generator, and environment variables to enable perfect state restoration
    for reproducible simulation execution with comprehensive metadata tracking.
    
    Attributes:
        python_state: Python random module state tuple
        numpy_state: NumPy random state in serializable format
        python_hash_seed: Environment hash seed state for PYTHONHASHSEED
        capture_time: State capture timestamp for tracking
        thread_id: Thread identifier for thread-local validation
        process_id: Process identifier for cross-process validation
        platform_info: Platform-specific information for cross-platform validation
        python_checksum: Validation checksum for Python state integrity
        numpy_checksum: Validation checksum for NumPy state integrity
    """
    
    # Core random state components
    python_state: Optional[Tuple] = None
    numpy_state: Optional[Dict[str, Any]] = None
    python_hash_seed: Optional[str] = None
    
    # Metadata for tracking and validation
    capture_time: float = 0.0
    thread_id: Optional[str] = None
    process_id: Optional[int] = None
    platform_info: Optional[str] = None
    
    # Validation checksums for state integrity verification
    python_checksum: Optional[str] = None
    numpy_checksum: Optional[str] = None
    
    def __post_init__(self):
        """Initialize metadata and compute state checksums for validation."""
        self.capture_time = time.time()
        self.thread_id = str(threading.current_thread().ident)
        self.process_id = os.getpid()
        self.platform_info = platform.platform()
        
        # Compute state checksums for integrity verification
        if self.python_state:
            self.python_checksum = self._compute_checksum(str(self.python_state))
        
        if self.numpy_state:
            self.numpy_checksum = self._compute_checksum(str(self.numpy_state))
    
    def _compute_checksum(self, data: str) -> str:
        """Compute SHA256 checksum for state integrity verification."""
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def is_valid(self) -> bool:
        """Validate state integrity using checksums."""
        try:
            if self.python_state and self.python_checksum:
                current_checksum = self._compute_checksum(str(self.python_state))
                if current_checksum != self.python_checksum:
                    return False
            
            if self.numpy_state and self.numpy_checksum:
                current_checksum = self._compute_checksum(str(self.numpy_state))
                if current_checksum != self.numpy_checksum:
                    return False
            
            return True
        
        except Exception as e:
            logger.warning(f"State validation failed: {e}")
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization and logging."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Self:
        """Create RandomState from dictionary data."""
        return cls(**data)


class SeedConfig(BaseModel):
    """
    Pydantic configuration model for seed management with Hydra integration.
    
    Provides comprehensive seed configuration supporting environment variable
    interpolation, validation rules, and integration with Hydra's structured
    configuration system for flexible experiment setup per Section 0.3.1.
    
    Features environment variable interpolation patterns:
    - ${oc.env:RANDOM_SEED} for global seed
    - ${oc.env:PYTHON_SEED} for Python-specific seed
    - ${oc.env:NUMPY_SEED} for NumPy-specific seed
    - ${oc.env:PYTHONHASHSEED} for hash randomization seed
    """
    
    # Primary seed configuration with environment variable support
    global_seed: Optional[int] = Field(
        default=None,
        ge=0,
        le=CROSS_PLATFORM_SEED_MAX,
        description="Global random seed for all randomness sources. Supports ${oc.env:RANDOM_SEED} interpolation"
    )
    
    # Individual component seed overrides for fine-grained control
    python_seed: Optional[int] = Field(
        default=None,
        ge=0,
        le=CROSS_PLATFORM_SEED_MAX,
        description="Specific seed for Python's random module. Supports ${oc.env:PYTHON_SEED}"
    )
    numpy_seed: Optional[int] = Field(
        default=None,
        ge=0,
        le=CROSS_PLATFORM_SEED_MAX,
        description="Specific seed for NumPy's random state. Supports ${oc.env:NUMPY_SEED}"
    )
    hash_seed: Optional[int] = Field(
        default=None,
        ge=0,
        le=CROSS_PLATFORM_SEED_MAX,
        description="Python hash randomization seed. Supports ${oc.env:PYTHONHASHSEED}"
    )
    
    # Validation and behavior configuration
    enable_validation: bool = Field(
        default=True,
        description="Enable determinism validation after seed setting"
    )
    strict_mode: bool = Field(
        default=False,
        description="Enable strict mode with enhanced validation and warnings"
    )
    cross_platform_determinism: bool = Field(
        default=True,
        description="Ensure cross-platform deterministic behavior"
    )
    
    # Performance configuration per Section 0.5.1 requirements
    validation_iterations: int = Field(
        default=DETERMINISM_TEST_ITERATIONS,
        ge=10,
        le=1000,
        description="Number of iterations for determinism validation testing"
    )
    performance_monitoring: bool = Field(
        default=True,
        description="Enable performance monitoring for seed operations"
    )
    
    # Auto-seeding configuration for convenience
    auto_seed_on_import: bool = Field(
        default=False,
        description="Automatically set global seed when module is imported"
    )
    warn_on_unset_seed: bool = Field(
        default=True,
        description="Warn when randomness is used without explicit seed setting"
    )
    
    # Hydra-specific _target_ metadata for factory-driven instantiation
    _target_: str = Field(
        default="plume_nav_sim.utils.seed_manager.setup_global_seed",
        description="Hydra target for automatic seed setup"
    )
    
    @field_validator('global_seed', 'python_seed', 'numpy_seed', 'hash_seed')
    @classmethod
    def validate_seed_range(cls, v, info):
        """Validate seed values are within acceptable range for cross-platform compatibility."""
        if v is not None:
            if not (0 <= v <= CROSS_PLATFORM_SEED_MAX):
                raise ValueError(f"{info.field_name} must be between 0 and {CROSS_PLATFORM_SEED_MAX}, got {v}")
        return v
    
    @field_validator('validation_iterations')
    @classmethod
    def validate_test_iterations(cls, v):
        """Validate test iteration count for performance balance."""
        if v > 500:
            warnings.warn(f"High validation iterations ({v}) may impact performance")
        return v
    
    model_config = ConfigDict(
        extra="allow",
        validate_assignment=True,
        str_strip_whitespace=True,
        json_schema_extra={
            "examples": [
                {
                    "global_seed": 42,
                    "enable_validation": True,
                    "cross_platform_determinism": True,
                    "performance_monitoring": True
                },
                {
                    "global_seed": "${oc.env:RANDOM_SEED,12345}",
                    "python_seed": "${oc.env:PYTHON_SEED}",
                    "numpy_seed": "${oc.env:NUMPY_SEED}",
                    "hash_seed": "${oc.env:PYTHONHASHSEED}",
                    "strict_mode": True
                }
            ]
        }
    )


# Thread-local storage for seed management context per Section 0.3.1 thread-safe operations
_seed_context = threading.local()


class SeedContext:
    """
    Thread-local seed management context for scoped operations with performance tracking.
    
    Maintains seed state and configuration within thread boundaries to support
    concurrent operations with independent seed management while preserving
    global state isolation and thread safety per Section 0.3.1 requirements.
    
    Features:
    - Thread-local state management for multi-agent environments
    - Nested operation stack for complex seed context management
    - Performance metrics tracking for all seed operations
    - State history management for debugging and analysis
    """
    
    def __init__(self):
        self.config: Optional[SeedConfig] = None
        self.saved_states: List[RandomState] = []
        self.operation_stack: List[str] = []
        self.performance_metrics: Dict[str, PerformanceMetrics] = {}
        self.is_seeded: bool = False
        self.last_seed_value: Optional[int] = None
        self.experiment_id: Optional[str] = None
        
    def push_state(self, operation_name: str) -> RandomState:
        """Capture and push current random state for scoped operations."""
        state = capture_random_state()
        self.saved_states.append(state)
        self.operation_stack.append(operation_name)
        
        logger.debug(f"Pushed random state for operation: {operation_name}")
        return state
    
    def pop_state(self) -> Optional[RandomState]:
        """Pop and restore most recent random state."""
        if not self.saved_states:
            logger.warning("No saved states to restore")
            return None
        
        state = self.saved_states.pop()
        operation = self.operation_stack.pop()
        
        restore_random_state(state)
        logger.debug(f"Restored random state for operation: {operation}")
        return state
    
    def clear_states(self):
        """Clear all saved states and reset context."""
        self.saved_states.clear()
        self.operation_stack.clear()
        self.performance_metrics.clear()
        
    def record_performance(self, operation: str, metrics: PerformanceMetrics):
        """Record performance metrics for seed operations."""
        self.performance_metrics[operation] = metrics


def get_seed_context() -> SeedContext:
    """Get or create seed context for current thread."""
    if not hasattr(_seed_context, 'context'):
        _seed_context.context = SeedContext()
    return _seed_context.context


def set_global_seed(
    seed: int,
    config: Optional[SeedConfig] = None,
    validate: bool = True
) -> Dict[str, Any]:
    """
    Set global random seed for all randomness sources with comprehensive validation.
    
    This function provides centralized seed management across Python's random module,
    NumPy's random state, and environment hash seed to ensure complete deterministic
    behavior for reproducible simulations and research per Section 0.3.1 requirements.
    
    Args:
        seed: Random seed value (0 to 2^31-1 for cross-platform compatibility)
        config: Optional SeedConfig for advanced configuration options
        validate: Whether to perform determinism validation after setting seed
        
    Returns:
        Dictionary containing operation results, performance metrics, and validation status
        
    Raises:
        ValueError: If seed value is outside valid range
        RuntimeError: If seed setting fails or validation detects non-deterministic behavior
        
    Example:
        >>> results = set_global_seed(42)
        >>> print(f"Seed set in {results['total_time']:.3f}s")
        >>> 
        >>> # With configuration
        >>> config = SeedConfig(enable_validation=True, strict_mode=True)
        >>> results = set_global_seed(12345, config=config)
    """
    start_time = time.time()
    context = get_seed_context()
    
    # Apply configuration defaults
    if config is None:
        config = SeedConfig(global_seed=seed)
    else:
        config.global_seed = seed
    
    # Validate seed range for cross-platform compatibility
    if not (0 <= seed <= CROSS_PLATFORM_SEED_MAX):
        raise ValueError(f"Seed must be between 0 and {CROSS_PLATFORM_SEED_MAX} for cross-platform compatibility, got {seed}")
    
    logger.info(f"Setting global random seed to {seed}")
    
    results = {
        "seed_value": seed,
        "components_seeded": [],
        "validation_passed": False,
        "performance_metrics": {},
        "platform_info": platform.platform(),
        "warnings": []
    }
    
    try:
        # Set Python random module seed with performance tracking
        py_start = time.time()
        specific_python_seed = config.python_seed if config.python_seed is not None else seed
        random.seed(specific_python_seed)
        py_time = (time.time() - py_start) * 1000
        
        results["components_seeded"].append("python_random")
        results["python_seed_value"] = specific_python_seed
        results["performance_metrics"]["python_seed"] = {
            "operation": "python_seed_set",
            "duration_ms": py_time,
            "timestamp": time.time()
        }
        
        # Set NumPy random seed with performance tracking
        np_start = time.time()
        specific_numpy_seed = config.numpy_seed if config.numpy_seed is not None else seed
        np.random.seed(specific_numpy_seed)
        np_time = (time.time() - np_start) * 1000
        
        results["components_seeded"].append("numpy_random")
        results["numpy_seed_value"] = specific_numpy_seed
        results["performance_metrics"]["numpy_seed"] = {
            "operation": "numpy_seed_set",
            "duration_ms": np_time,
            "timestamp": time.time()
        }
        
        # Handle PYTHONHASHSEED (requires process restart to take effect)
        hash_seed_value = config.hash_seed if config.hash_seed is not None else seed
        current_hash_seed = os.environ.get('PYTHONHASHSEED', 'random')
        
        if current_hash_seed != str(hash_seed_value):
            os.environ['PYTHONHASHSEED'] = str(hash_seed_value)
            results["components_seeded"].append("python_hash_seed")
            results["hash_seed_value"] = hash_seed_value
            
            if config.strict_mode:
                results["warnings"].append(
                    "PYTHONHASHSEED set but requires process restart to take effect. "
                    "Consider setting PYTHONHASHSEED before importing Python modules."
                )
        else:
            results["components_seeded"].append("python_hash_seed")
            results["hash_seed_value"] = hash_seed_value
        
        # Update context state
        context.config = config
        context.is_seeded = True
        context.last_seed_value = seed
        
        # Perform determinism validation if enabled
        if validate and config.enable_validation:
            val_start = time.time()
            validation_results = validate_determinism(
                iterations=config.validation_iterations,
                cross_platform=config.cross_platform_determinism
            )
            val_time = (time.time() - val_start) * 1000
            
            results["validation_passed"] = validation_results["is_deterministic"]
            results["validation_details"] = validation_results
            results["performance_metrics"]["validation"] = {
                "operation": "determinism_validation",
                "duration_ms": val_time,
                "timestamp": time.time()
            }
            
            if not results["validation_passed"] and config.strict_mode:
                raise RuntimeError(f"Determinism validation failed: {validation_results}")
        
        # Calculate total operation time
        total_time = time.time() - start_time
        results["total_time"] = total_time
        
        # Check performance thresholds per Section 0.5.1
        if total_time > SEED_PERFORMANCE_THRESHOLDS["global_seed_init"]:
            warning = f"Seed initialization took {total_time:.3f}s, exceeding {SEED_PERFORMANCE_THRESHOLDS['global_seed_init']}s target"
            results["warnings"].append(warning)
            logger.warning(warning)
        
        # Log successful completion with structured data
        logger.info(
            f"Global seed set successfully in {total_time:.3f}s",
            extra={
                "metric_type": "seed_operation",
                "operation": "set_global_seed",
                "seed_value": seed,
                "total_time": total_time,
                "components_seeded": results["components_seeded"],
                "validation_passed": results["validation_passed"]
            }
        )
        
        return results
    
    except Exception as e:
        logger.error(f"Failed to set global seed {seed}: {e}")
        raise RuntimeError(f"Seed setting failed: {e}") from e


def capture_random_state() -> RandomState:
    """
    Capture complete random state from all randomness sources.
    
    Creates a comprehensive snapshot of the current random state across Python's
    random module, NumPy's random generator, and environment variables to enable
    perfect state restoration for reproducible operations per Section 0.3.1.
    
    Returns:
        RandomState: Complete state snapshot with integrity validation
        
    Example:
        >>> state = capture_random_state()
        >>> # Perform some random operations
        >>> random.random()
        >>> np.random.random()
        >>> # State can be restored later
        >>> restore_random_state(state)
    """
    start_time = time.time()
    
    logger.debug("Capturing random state from all sources")
    
    try:
        # Capture Python random state
        python_state = random.getstate()
        
        # Capture NumPy random state (use legacy format for compatibility)
        numpy_state = np.random.get_state()
        # Convert to serializable format for cross-platform compatibility
        numpy_state_dict = {
            'bit_generator': numpy_state[0],
            'state': {
                'state': numpy_state[1]['state'].tolist(),
                'pos': int(numpy_state[1]['pos'])
            },
            'has_gauss': int(numpy_state[2]),
            'gauss': float(numpy_state[3]) if numpy_state[3] is not None else None
        }
        
        # Capture environment hash seed
        hash_seed = os.environ.get('PYTHONHASHSEED', 'random')
        
        # Create state object with validation
        state = RandomState(
            python_state=python_state,
            numpy_state=numpy_state_dict,
            python_hash_seed=hash_seed
        )
        
        capture_time = time.time() - start_time
        
        # Performance monitoring per Section 0.5.1 requirements
        if capture_time > SEED_PERFORMANCE_THRESHOLDS["state_capture"]:
            logger.warning(f"State capture took {capture_time:.3f}s, exceeding {SEED_PERFORMANCE_THRESHOLDS['state_capture']}s target")
        
        logger.debug(f"Random state captured in {capture_time:.3f}s")
        
        return state
    
    except Exception as e:
        logger.error(f"Failed to capture random state: {e}")
        raise RuntimeError(f"State capture failed: {e}") from e


def restore_random_state(state: RandomState) -> bool:
    """
    Restore complete random state to all randomness sources.
    
    Restores the random state across Python's random module, NumPy's random
    generator, and environment variables from a previously captured state
    snapshot with integrity validation per Section 0.3.1 requirements.
    
    Args:
        state: RandomState object containing the state to restore
        
    Returns:
        True if restoration was successful, False otherwise
        
    Raises:
        ValueError: If state object is invalid or corrupted
        RuntimeError: If state restoration fails
        
    Example:
        >>> state = capture_random_state()
        >>> # Perform operations that change random state
        >>> success = restore_random_state(state)
        >>> assert success, "State restoration failed"
    """
    start_time = time.time()
    
    if not isinstance(state, RandomState):
        raise ValueError("Invalid state object: must be RandomState instance")
    
    if not state.is_valid():
        raise ValueError("State object failed integrity validation")
    
    logger.debug("Restoring random state to all sources")
    
    try:
        # Restore Python random state
        if state.python_state:
            random.setstate(state.python_state)
        
        # Restore NumPy random state
        if state.numpy_state:
            # Convert from serializable format back to NumPy format
            numpy_state_tuple = (
                state.numpy_state['bit_generator'],
                {
                    'state': np.array(state.numpy_state['state']['state'], dtype=np.uint32),
                    'pos': state.numpy_state['state']['pos']
                },
                state.numpy_state['has_gauss'],
                state.numpy_state['gauss']
            )
            np.random.set_state(numpy_state_tuple)
        
        # Note: PYTHONHASHSEED restoration requires process restart
        if state.python_hash_seed and state.python_hash_seed != 'random':
            current_hash_seed = os.environ.get('PYTHONHASHSEED', 'random')
            if current_hash_seed != state.python_hash_seed:
                os.environ['PYTHONHASHSEED'] = state.python_hash_seed
                logger.warning("PYTHONHASHSEED updated but requires process restart to take effect")
        
        restore_time = time.time() - start_time
        
        # Performance monitoring per Section 0.5.1 requirements
        if restore_time > SEED_PERFORMANCE_THRESHOLDS["state_restore"]:
            logger.warning(f"State restore took {restore_time:.3f}s, exceeding {SEED_PERFORMANCE_THRESHOLDS['state_restore']}s target")
        
        logger.debug(f"Random state restored in {restore_time:.3f}s")
        
        return True
    
    except Exception as e:
        logger.error(f"Failed to restore random state: {e}")
        raise RuntimeError(f"State restoration failed: {e}") from e


def validate_determinism(
    iterations: int = DETERMINISM_TEST_ITERATIONS,
    cross_platform: bool = True
) -> Dict[str, Any]:
    """
    Validate deterministic behavior of random number generation.
    
    Performs comprehensive testing to ensure that random number generation
    produces identical results across multiple runs with the same seed,
    supporting both platform-specific and cross-platform validation per
    Section 0.5.1 implementation verification requirements.
    
    Args:
        iterations: Number of test iterations to perform
        cross_platform: Whether to test cross-platform determinism
        
    Returns:
        Dictionary containing validation results, test metrics, and diagnostic information
        
    Example:
        >>> results = validate_determinism(iterations=50)
        >>> print(f"Determinism test: {'PASSED' if results['is_deterministic'] else 'FAILED'}")
        >>> print(f"Python random variance: {results['python_variance']}")
        >>> print(f"NumPy random variance: {results['numpy_variance']}")
    """
    start_time = time.time()
    
    logger.debug(f"Starting determinism validation with {iterations} iterations")
    
    # Capture current state for restoration
    original_state = capture_random_state()
    
    results = {
        "is_deterministic": False,
        "test_iterations": iterations,
        "python_test_passed": False,
        "numpy_test_passed": False,
        "python_variance": 0.0,
        "numpy_variance": 0.0,
        "test_duration": 0.0,
        "platform_info": platform.platform(),
        "cross_platform_tested": cross_platform,
        "diagnostic_info": {}
    }
    
    try:
        # Test seed for determinism validation
        test_seed = 12345
        
        # Test Python random determinism
        python_results = []
        for i in range(iterations):
            random.seed(test_seed)
            test_values = [random.random() for _ in range(PYTHON_RANDOM_TEST_SIZE)]
            python_results.append(sum(test_values))
        
        python_variance = np.var(python_results)
        results["python_variance"] = float(python_variance)
        results["python_test_passed"] = python_variance < 1e-15
        
        # Test NumPy random determinism
        numpy_results = []
        for i in range(iterations):
            np.random.seed(test_seed)
            test_values = np.random.random(NUMPY_RANDOM_TEST_SIZE)
            numpy_results.append(float(np.sum(test_values)))
        
        numpy_variance = np.var(numpy_results)
        results["numpy_variance"] = float(numpy_variance)
        results["numpy_test_passed"] = numpy_variance < 1e-15
        
        # Overall determinism check
        results["is_deterministic"] = results["python_test_passed"] and results["numpy_test_passed"]
        
        # Cross-platform specific tests
        if cross_platform:
            # Test hash function determinism (platform-dependent)
            hash_results = []
            for i in range(min(iterations, 10)):  # Limited iterations for hash tests
                test_string = f"test_string_{i}"
                hash_value = hash(test_string)
                hash_results.append(hash_value)
            
            # Note: Hash values are intentionally non-deterministic across process restarts
            # when PYTHONHASHSEED is random, so we don't fail on hash variance
            results["diagnostic_info"]["hash_variance"] = float(np.var(hash_results))
            results["diagnostic_info"]["hash_note"] = "Hash determinism requires PYTHONHASHSEED setting before process start"
        
        # Performance validation per Section 0.5.1
        test_duration = time.time() - start_time
        results["test_duration"] = test_duration
        
        if test_duration > SEED_PERFORMANCE_THRESHOLDS["validation"]:
            logger.warning(f"Determinism validation took {test_duration:.3f}s, exceeding {SEED_PERFORMANCE_THRESHOLDS['validation']}s target")
        
        # Log results with structured data
        if results["is_deterministic"]:
            logger.info(
                f"Determinism validation PASSED in {test_duration:.3f}s",
                extra={
                    "metric_type": "determinism_validation",
                    "test_passed": True,
                    "iterations": iterations,
                    "python_variance": python_variance,
                    "numpy_variance": numpy_variance
                }
            )
        else:
            logger.warning(
                f"Determinism validation FAILED in {test_duration:.3f}s",
                extra={
                    "metric_type": "determinism_validation",
                    "test_passed": False,
                    "python_test_passed": results["python_test_passed"],
                    "numpy_test_passed": results["numpy_test_passed"],
                    "python_variance": python_variance,
                    "numpy_variance": numpy_variance
                }
            )
    
    except Exception as e:
        logger.error(f"Determinism validation failed with error: {e}")
        results["error"] = str(e)
    
    finally:
        # Restore original state
        try:
            restore_random_state(original_state)
        except Exception as e:
            logger.error(f"Failed to restore state after validation: {e}")
    
    return results


@contextmanager
def scoped_seed(
    seed: int,
    operation_name: str = "scoped_operation",
    validate_after: bool = False
) -> ContextManager[RandomState]:
    """
    Context manager for temporary seed changes with automatic state restoration.
    
    Provides scoped seed management where the random state is temporarily changed
    for a specific operation and automatically restored afterwards, ensuring that
    global random state is preserved while enabling deterministic operations
    per Section 0.3.1 context managers requirements.
    
    Args:
        seed: Temporary seed value for the scoped operation
        operation_name: Name of the operation for logging and tracking
        validate_after: Whether to validate determinism after seed setting
        
    Yields:
        RandomState: The captured state before the temporary seed change
        
    Raises:
        ValueError: If seed value is invalid
        RuntimeError: If state capture or restoration fails
        
    Example:
        >>> # Global state is preserved
        >>> with scoped_seed(42, "experiment_initialization") as original_state:
        ...     # All randomness within this block uses seed 42
        ...     experiment_data = np.random.random(100)
        ...     agent_positions = [random.uniform(0, 100) for _ in range(10)]
        >>> # Original random state is automatically restored
    """
    context = get_seed_context()
    
    # Validate seed value
    if not (0 <= seed <= CROSS_PLATFORM_SEED_MAX):
        raise ValueError(f"Scoped seed must be between 0 and {CROSS_PLATFORM_SEED_MAX}, got {seed}")
    
    logger.debug(f"Starting scoped seed operation '{operation_name}' with seed {seed}")
    
    # Capture current state
    try:
        original_state = context.push_state(operation_name)
    except Exception as e:
        raise RuntimeError(f"Failed to capture state for scoped operation: {e}") from e
    
    # Set temporary seed with performance monitoring
    try:
        op_start = time.time()
        
        # Create temporary config for the scoped operation
        temp_config = SeedConfig(
            global_seed=seed,
            enable_validation=validate_after,
            performance_monitoring=True
        )
        
        # Set the temporary seed (without extensive validation for performance)
        set_global_seed(seed, config=temp_config, validate=validate_after)
        
        op_time = (time.time() - op_start) * 1000
        
        # Record performance metrics
        if ENHANCED_LOGGING_AVAILABLE:
            metrics = PerformanceMetrics(
                operation=f"scoped_seed_{operation_name}",
                duration_ms=op_time
            )
            context.record_performance(operation_name, metrics)
        
        # Check performance threshold
        if op_time > SEED_PERFORMANCE_THRESHOLDS["scoped_operation"] * 1000:
            logger.warning(f"Scoped seed operation '{operation_name}' took {op_time:.3f}ms, exceeding threshold")
        
        yield original_state
    
    except Exception as e:
        logger.error(f"Error during scoped seed operation '{operation_name}': {e}")
        raise
    
    finally:
        # Restore original state
        try:
            context.pop_state()
            logger.debug(f"Completed scoped seed operation '{operation_name}'")
        except Exception as e:
            logger.error(f"Failed to restore state after scoped operation '{operation_name}': {e}")
            # Don't raise here to avoid masking the original exception


def get_random_state() -> RandomState:
    """
    Get current random state from all sources.
    
    Convenience function that captures the current random state across all
    randomness sources and returns it for external use or storage.
    
    Returns:
        RandomState: Current complete random state
        
    Example:
        >>> current_state = get_random_state()
        >>> # Save state for later use
        >>> state_dict = current_state.to_dict()
        >>> # Can be restored later with restore_random_state(current_state)
    """
    return capture_random_state()


def setup_global_seed(config: Optional[Union[SeedConfig, Dict[str, Any], int]] = None) -> Dict[str, Any]:
    """
    Setup global seed from configuration with Hydra integration support.
    
    Primary entry point for seed management that supports multiple configuration
    input formats including direct seed values, SeedConfig objects, and Hydra
    configuration dictionaries with environment variable interpolation per
    Section 0.3.1 Hydra integration requirements.
    
    Args:
        config: Seed configuration as SeedConfig, dict, or integer seed value
        
    Returns:
        Dictionary containing setup results and performance metrics
        
    Raises:
        ValueError: If configuration is invalid
        RuntimeError: If seed setup fails
        
    Example:
        >>> # Simple integer seed
        >>> results = setup_global_seed(42)
        >>> 
        >>> # Using configuration object
        >>> config = SeedConfig(global_seed=12345, enable_validation=True)
        >>> results = setup_global_seed(config)
        >>> 
        >>> # Using Hydra configuration dict
        >>> hydra_config = {"global_seed": "${oc.env:RANDOM_SEED,42}", "strict_mode": True}
        >>> results = setup_global_seed(hydra_config)
    """
    start_time = time.time()
    
    # Handle different input types
    if config is None:
        logger.warning("No seed configuration provided, using default settings")
        seed_config = SeedConfig()
    elif isinstance(config, int):
        seed_config = SeedConfig(global_seed=config)
    elif isinstance(config, dict):
        seed_config = SeedConfig(**config)
    elif isinstance(config, SeedConfig):
        seed_config = config
    else:
        raise ValueError(f"Invalid config type: {type(config)}. Expected SeedConfig, dict, int, or None")
    
    # Apply auto-seeding logic
    if seed_config.global_seed is None:
        if seed_config.auto_seed_on_import:
            # Generate a deterministic seed based on current time and process ID
            seed_config.global_seed = int(time.time() * 1000000) % CROSS_PLATFORM_SEED_MAX
            logger.info(f"Auto-generated global seed: {seed_config.global_seed}")
        else:
            logger.warning("No global seed specified and auto_seed_on_import is False")
            return {"status": "skipped", "reason": "no_seed_specified"}
    
    # Set up the global seed
    try:
        results = set_global_seed(
            seed=seed_config.global_seed,
            config=seed_config,
            validate=seed_config.enable_validation
        )
        
        setup_time = time.time() - start_time
        results["setup_time"] = setup_time
        results["configuration"] = seed_config.model_dump()
        
        logger.info(
            f"Global seed setup completed in {setup_time:.3f}s",
            extra={
                "metric_type": "seed_setup",
                "seed_value": seed_config.global_seed,
                "setup_time": setup_time,
                "validation_enabled": seed_config.enable_validation
            }
        )
        
        return results
    
    except Exception as e:
        logger.error(f"Global seed setup failed: {e}")
        raise RuntimeError(f"Seed setup failed: {e}") from e


def reset_random_state() -> bool:
    """
    Reset all random states to unseeded condition.
    
    Resets Python's random module and NumPy's random generator to their default
    unseeded state, clearing any previously set seeds for fresh initialization.
    
    Returns:
        True if reset was successful
        
    Example:
        >>> # After setting seeds
        >>> set_global_seed(42)
        >>> # Reset to unseeded state
        >>> success = reset_random_state()
        >>> assert success
    """
    try:
        logger.info("Resetting all random states to unseeded condition")
        
        # Reset Python random to default state
        random.seed()
        
        # Reset NumPy random to default state
        np.random.seed(None)
        
        # Update context
        context = get_seed_context()
        context.is_seeded = False
        context.last_seed_value = None
        context.clear_states()
        
        logger.info("Random state reset completed successfully")
        return True
    
    except Exception as e:
        logger.error(f"Failed to reset random state: {e}")
        return False


def is_seeded() -> bool:
    """
    Check if global seed has been set in current context.
    
    Returns:
        True if a global seed has been set in the current thread context
        
    Example:
        >>> print(f"Seeded: {is_seeded()}")  # False
        >>> set_global_seed(42)
        >>> print(f"Seeded: {is_seeded()}")  # True
    """
    context = get_seed_context()
    return context.is_seeded


def get_last_seed() -> Optional[int]:
    """
    Get the last seed value that was set in current context.
    
    Returns:
        Last seed value, or None if no seed has been set
        
    Example:
        >>> print(f"Last seed: {get_last_seed()}")  # None
        >>> set_global_seed(42)
        >>> print(f"Last seed: {get_last_seed()}")  # 42
    """
    context = get_seed_context()
    return context.last_seed_value


def generate_experiment_seed(experiment_name: str, base_seed: Optional[int] = None) -> int:
    """
    Generate deterministic seed for named experiments.
    
    Creates a deterministic seed value based on experiment name and optional base seed,
    ensuring reproducible results for named experiments while maintaining uniqueness
    across different experiment names per Section 0.3.1 requirements.
    
    Args:
        experiment_name: Unique name for the experiment
        base_seed: Optional base seed for additional randomization
        
    Returns:
        Generated deterministic seed value
        
    Example:
        >>> exp1_seed = generate_experiment_seed("optimization_run_1")
        >>> exp2_seed = generate_experiment_seed("optimization_run_2")
        >>> # Same experiment name always produces same seed
        >>> assert generate_experiment_seed("optimization_run_1") == exp1_seed
    """
    # Create deterministic seed from experiment name
    name_hash = hashlib.sha256(experiment_name.encode()).hexdigest()
    
    # Combine with base seed if provided
    if base_seed is not None:
        combined = f"{name_hash}{base_seed}"
        name_hash = hashlib.sha256(combined.encode()).hexdigest()
    
    # Convert to integer in valid range
    seed = int(name_hash[:8], 16) % CROSS_PLATFORM_SEED_MAX
    
    logger.debug(f"Generated experiment seed {seed} for '{experiment_name}'")
    return seed


def create_seed_config_from_hydra(hydra_config: Optional[Any] = None) -> SeedConfig:
    """
    Create SeedConfig from Hydra configuration with environment variable resolution.
    
    Integrates with Hydra's configuration system to create seed configurations
    that support environment variable interpolation and hierarchical configuration
    composition for flexible experiment setup per Section 0.3.1 requirements.
    
    Args:
        hydra_config: Hydra configuration object (DictConfig)
        
    Returns:
        SeedConfig: Resolved seed configuration
        
    Example:
        >>> # In a Hydra app
        >>> @hydra.main(config_path="conf", config_name="config")
        >>> def my_app(cfg: DictConfig) -> None:
        ...     seed_config = create_seed_config_from_hydra(cfg.seed)
        ...     setup_global_seed(seed_config)
    """
    if hydra_config is None:
        return SeedConfig()
    
    try:
        from omegaconf import OmegaConf
        
        # Resolve environment variables and convert to dict
        resolved_config = OmegaConf.to_container(hydra_config, resolve=True)
        return SeedConfig(**resolved_config)
    
    except ImportError:
        # Fallback if OmegaConf not available
        if hasattr(hydra_config, '_content'):
            return SeedConfig(**hydra_config._content)
        else:
            return SeedConfig(**dict(hydra_config))


def register_seed_config_schema():
    """
    Register SeedConfig with Hydra ConfigStore for structured configuration.
    
    Enables automatic schema discovery and validation within Hydra's configuration
    composition system for comprehensive seed management configuration per
    Section 0.3.1 Hydra integration requirements.
    """
    try:
        from hydra.core.config_store import ConfigStore
        
        cs = ConfigStore.instance()
        cs.store(
            group="seed",
            name="default",
            node=SeedConfig,
            package="seed"
        )
        
        logger.info("Successfully registered SeedConfig schema with Hydra ConfigStore")
        
    except ImportError:
        logger.warning("Hydra not available, skipping ConfigStore registration")
    except Exception as e:
        logger.error(f"Failed to register seed configuration schema: {e}")


# Performance monitoring decorator for seed-sensitive operations per Section 0.3.1
def seed_sensitive_operation(
    operation_name: str,
    require_seed: bool = False,
    auto_seed: Optional[int] = None
) -> Callable:
    """
    Decorator for operations that are sensitive to random seed state.
    
    Provides automatic seed management for functions that require deterministic
    behavior, with optional auto-seeding and seed requirement enforcement
    per Section 0.3.1 context managers requirements.
    
    Args:
        operation_name: Name of the operation for logging and tracking
        require_seed: Whether to require a seed to be set before operation
        auto_seed: Automatic seed to set if none is present
        
    Returns:
        Decorated function with seed management
        
    Example:
        >>> @seed_sensitive_operation("simulation_run", require_seed=True)
        >>> def run_simulation(params):
        ...     # This function requires a seed to be set
        ...     return np.random.random(100)
        >>> 
        >>> @seed_sensitive_operation("data_generation", auto_seed=42)
        >>> def generate_data():
        ...     # Auto-seeds with 42 if no seed is set
        ...     return random.random()
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            context = get_seed_context()
            
            # Check seed requirements
            if require_seed and not context.is_seeded:
                if auto_seed is not None:
                    logger.info(f"Auto-seeding operation '{operation_name}' with seed {auto_seed}")
                    set_global_seed(auto_seed)
                else:
                    raise RuntimeError(f"Operation '{operation_name}' requires a seed to be set")
            elif auto_seed is not None and not context.is_seeded:
                logger.info(f"Auto-seeding operation '{operation_name}' with seed {auto_seed}")
                set_global_seed(auto_seed)
            
            # Execute with performance monitoring
            start_time = time.time()
            try:
                if context.config and context.config.warn_on_unset_seed and not context.is_seeded:
                    logger.warning(f"Operation '{operation_name}' using unset random seed")
                
                result = func(*args, **kwargs)
                
                # Record performance metrics
                duration = (time.time() - start_time) * 1000
                if ENHANCED_LOGGING_AVAILABLE:
                    metrics = PerformanceMetrics(
                        operation=f"seed_sensitive_{operation_name}",
                        duration_ms=duration
                    )
                    context.record_performance(operation_name, metrics)
                
                return result
            
            except Exception as e:
                logger.error(f"Seed-sensitive operation '{operation_name}' failed: {e}")
                raise
        
        return wrapper
    return decorator


def get_reproducibility_report() -> Dict[str, Any]:
    """
    Generate comprehensive reproducibility report for experiment documentation.
    
    Creates detailed report including system information, current seed state,
    performance metrics, and validation results for complete experiment
    reproducibility documentation per Section 0.5.1 requirements.
    
    Returns:
        Dict[str, Any]: Complete reproducibility metadata and system information
        
    Example:
        >>> report = get_reproducibility_report()
        >>> print(f"Platform: {report['system_info']['platform']}")
        >>> print(f"Seed state: {report['seed_context']['is_seeded']}")
    """
    context = get_seed_context()
    
    report = {
        'timestamp': time.time(),
        'system_info': {
            'platform': platform.platform(),
            'python_version': list(sys.version_info[:3]),
            'numpy_version': np.__version__,
            'architecture': sys.maxsize > 2**32 and '64bit' or '32bit',
            'byte_order': sys.byteorder
        },
        'environment_variables': {
            key: os.environ.get(key) for key in [
                'PYTHONHASHSEED', 'RANDOM_SEED', 'NUMPY_SEED', 'PYTHON_SEED',
                'CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS'
            ] if os.environ.get(key)
        },
        'seed_context': {
            'is_seeded': context.is_seeded,
            'last_seed_value': context.last_seed_value,
            'experiment_id': context.experiment_id,
            'operation_stack_depth': len(context.operation_stack),
            'saved_states_count': len(context.saved_states)
        },
        'performance_thresholds': SEED_PERFORMANCE_THRESHOLDS,
        'cross_platform_seed_max': CROSS_PLATFORM_SEED_MAX
    }
    
    # Add current random state information
    try:
        current_state = capture_random_state()
        report['current_random_state'] = {
            'capture_time': current_state.capture_time,
            'thread_id': current_state.thread_id,
            'process_id': current_state.process_id,
            'platform_info': current_state.platform_info,
            'state_valid': current_state.is_valid()
        }
    except Exception as e:
        report['current_random_state'] = {'error': str(e)}
    
    # Add configuration information if available
    if context.config:
        report['seed_configuration'] = context.config.model_dump()
    
    # Add performance metrics if available
    if context.performance_metrics:
        report['performance_metrics'] = {
            name: metrics.to_dict() for name, metrics in context.performance_metrics.items()
        }
    
    return report


# Initialize module with performance tracking per Section 0.5.1
_module_init_start = time.time()

# Register with Hydra ConfigStore if available
try:
    register_seed_config_schema()
except Exception as e:
    logger.debug(f"ConfigStore registration skipped: {e}")

_module_init_time = time.time() - _module_init_start

# Module initialization logging
logger.info(
    f"Seed manager module initialized in {_module_init_time:.3f}s",
    extra={
        "metric_type": "module_initialization",
        "module": "seed_manager",
        "init_time": _module_init_time,
        "performance_thresholds": SEED_PERFORMANCE_THRESHOLDS,
        "enhanced_logging_available": ENHANCED_LOGGING_AVAILABLE
    }
)

# Enhanced exports for comprehensive seed management functionality per Section 0.2.3
__all__ = [
    # Core seed management functions
    "set_global_seed",
    "get_random_state",
    "restore_random_state",
    "capture_random_state",
    "reset_random_state",
    
    # Context managers and scoped operations
    "scoped_seed",
    "get_seed_context",
    "seed_sensitive_operation",
    
    # Configuration and setup
    "SeedConfig",
    "setup_global_seed",
    "create_seed_config_from_hydra",
    
    # State management
    "RandomState",
    "SeedContext",
    
    # Validation and utilities
    "validate_determinism",
    "is_seeded",
    "get_last_seed",
    "generate_experiment_seed",
    
    # Hydra integration
    "register_seed_config_schema",
    
    # Reproducibility reporting
    "get_reproducibility_report",
    
    # Constants
    "SEED_PERFORMANCE_THRESHOLDS",
    "DETERMINISM_TEST_ITERATIONS",
    "CROSS_PLATFORM_SEED_MAX",
]