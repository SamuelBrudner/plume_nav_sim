"""
Global Seed Management Utilities for Reproducible Experiments.

This module provides comprehensive seed management functionality for ensuring reproducible
experiments across the odor plume navigation system. It implements thread-safe operations,
Gymnasium 0.29.x API compliance, and centralized seeding across Python's random, NumPy,
and environment variables.

Key Features:
- Global seeding utility supporting Python random, NumPy, and PYTHONHASHSEED
- Gymnasium 0.29.x compliant seed parameter support for reset() methods
- Thread-safe seed management for concurrent environment execution
- Seed validation and determinism checking for research reproducibility
- Integration with centralized Loguru logging for comprehensive observability
- Support for correlation context and experiment tracking
- Performance monitoring with threshold-based warnings

Integration Points:
- Centralized Loguru logging from logging_setup module
- Configuration integration via SimulationConfig models
- Thread-safe operations with automatic cleanup
- Environment variable management for cross-process reproducibility
"""

import os
import sys
import time
import random
import threading
import hashlib
import warnings
from typing import Optional, Union, Dict, Any, List, Tuple, Callable, ContextManager
from contextlib import contextmanager
from functools import wraps
from dataclasses import dataclass, field
from pathlib import Path

# Core scientific computing libraries
import numpy as np

# Import configuration and logging dependencies
from odor_plume_nav.utils.logging_setup import (
    get_logger, 
    correlation_context, 
    get_correlation_context,
    PerformanceMetrics
)
from odor_plume_nav.config.models import SimulationConfig

# Module logger with enhanced context
logger = get_logger(__name__)

# Thread-local storage for seed context management
_seed_context = threading.local()

# Global seed registry for tracking and validation
_global_seed_registry: Dict[str, Any] = {}
_registry_lock = threading.RLock()

# Performance tracking for seed operations
SEED_PERFORMANCE_THRESHOLDS = {
    "global_seed_setup": 0.1,  # 100ms threshold for global seed operations
    "numpy_seed": 0.05,        # 50ms threshold for NumPy seeding
    "thread_seed_setup": 0.02, # 20ms threshold for thread-local seeding
    "validation_check": 0.01,  # 10ms threshold for validation operations
}

# Supported random number generators and their seeding functions
RNG_COMPONENTS = {
    "python_random": random.seed,
    "numpy_random": np.random.seed,
    "pythonhashseed": lambda seed: os.environ.update({"PYTHONHASHSEED": str(seed)}),
}


@dataclass
class SeedContext:
    """
    Thread-local seed context for managing reproducibility state.
    
    Maintains comprehensive seed state including global seed values, per-component
    seeds, validation state, and performance metrics for debugging and analysis.
    """
    
    # Core seed state
    global_seed: Optional[int] = None
    component_seeds: Dict[str, int] = field(default_factory=dict)
    thread_id: str = field(default_factory=lambda: str(threading.current_thread().ident))
    
    # Validation and tracking
    is_seeded: bool = False
    seed_timestamp: Optional[float] = None
    validation_hash: Optional[str] = None
    
    # Performance metrics
    setup_duration: Optional[float] = None
    component_timings: Dict[str, float] = field(default_factory=dict)
    
    # Experiment correlation
    correlation_id: Optional[str] = None
    experiment_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize correlation context integration."""
        try:
            context = get_correlation_context()
            self.correlation_id = context.correlation_id
            self.experiment_metadata = dict(context.experiment_metadata)
        except Exception:
            # Graceful fallback if correlation context unavailable
            self.correlation_id = f"seed_context_{id(self)}"
    
    def get_state_hash(self) -> str:
        """Generate deterministic hash of current seed state for validation."""
        state_data = {
            "global_seed": self.global_seed,
            "component_seeds": sorted(self.component_seeds.items()),
            "thread_id": self.thread_id,
        }
        
        state_str = str(state_data).encode("utf-8")
        return hashlib.sha256(state_str).hexdigest()[:16]
    
    def bind_logger_context(self) -> Dict[str, Any]:
        """Get context dictionary for structured logging."""
        return {
            "seed_context": {
                "global_seed": self.global_seed,
                "is_seeded": self.is_seeded,
                "thread_id": self.thread_id,
                "validation_hash": self.validation_hash,
                "setup_duration": self.setup_duration,
            },
            "correlation_id": self.correlation_id,
            **self.experiment_metadata
        }


def get_seed_context() -> SeedContext:
    """Get or create seed context for current thread."""
    if not hasattr(_seed_context, 'context'):
        _seed_context.context = SeedContext()
    return _seed_context.context


def set_seed_context(context: SeedContext):
    """Set seed context for current thread."""
    _seed_context.context = context


@contextmanager
def seed_context_manager(
    seed: Optional[int] = None,
    experiment_name: Optional[str] = None,
    **metadata
) -> ContextManager[SeedContext]:
    """
    Context manager for scoped seed management with automatic cleanup.
    
    Provides isolated seed context for experiments or test cases with
    automatic restoration of previous state and comprehensive logging.
    
    Args:
        seed: Optional seed value for the context
        experiment_name: Optional experiment name for tracking
        **metadata: Additional metadata for correlation context
        
    Yields:
        SeedContext: Isolated seed context for the scope
        
    Example:
        >>> with seed_context_manager(42, experiment_name="test_run") as ctx:
        ...     # All seeding operations use isolated context
        ...     set_global_seed(42)
        ...     # Context automatically restored on exit
    """
    # Create new isolated context
    new_context = SeedContext()
    if experiment_name:
        new_context.experiment_metadata["experiment_name"] = experiment_name
    new_context.experiment_metadata.update(metadata)
    
    # Store previous context
    old_context = getattr(_seed_context, 'context', None)
    
    with correlation_context(
        operation_name="seed_context_scope",
        experiment_name=experiment_name,
        **metadata
    ):
        try:
            # Set new context
            set_seed_context(new_context)
            
            # Apply seed if provided
            if seed is not None:
                set_global_seed(seed)
            
            logger.debug(
                f"Seed context established",
                extra=new_context.bind_logger_context()
            )
            
            yield new_context
            
        finally:
            # Restore previous context
            if old_context:
                set_seed_context(old_context)
            elif hasattr(_seed_context, 'context'):
                delattr(_seed_context, 'context')
            
            logger.debug(
                f"Seed context restored",
                extra=new_context.bind_logger_context()
            )


def set_global_seed(
    seed: int,
    components: Optional[List[str]] = None,
    validate: bool = True,
    performance_tracking: bool = True
) -> Dict[str, Any]:
    """
    Set global seed across all random number generators with comprehensive validation.
    
    This function provides centralized seeding for Python's random, NumPy, and
    PYTHONHASHSEED environment variable, ensuring reproducibility across all
    stochastic components of the system.
    
    Args:
        seed: Integer seed value (0 <= seed <= 2^32-1)
        components: Optional list of specific components to seed
                   (default: all supported components)
        validate: Whether to perform post-seeding validation
        performance_tracking: Whether to track timing metrics
        
    Returns:
        Dict containing seeding results and performance metrics
        
    Raises:
        ValueError: If seed is invalid or validation fails
        RuntimeError: If seeding operation fails
        
    Example:
        >>> result = set_global_seed(42)
        >>> print(f"Seeded {len(result['components'])} components")
        >>> 
        >>> # Seed specific components only
        >>> set_global_seed(123, components=["python_random", "numpy_random"])
    """
    start_time = time.perf_counter()
    context = get_seed_context()
    
    # Validate seed value
    if not isinstance(seed, int) or not (0 <= seed <= 2**32 - 1):
        raise ValueError(f"Seed must be an integer between 0 and 2^32-1, got {seed}")
    
    # Default to all components if none specified
    if components is None:
        components = list(RNG_COMPONENTS.keys())
    
    # Validate requested components
    invalid_components = set(components) - set(RNG_COMPONENTS.keys())
    if invalid_components:
        raise ValueError(f"Invalid components: {invalid_components}. "
                        f"Supported: {list(RNG_COMPONENTS.keys())}")
    
    results = {
        "seed": seed,
        "components": {},
        "performance": {},
        "validation": {},
        "thread_id": context.thread_id,
        "timestamp": time.time(),
    }
    
    with correlation_context("global_seed_setup", seed=seed, components=components):
        logger.info(
            f"Setting global seed to {seed}",
            extra={"seed": seed, "components": components}
        )
        
        # Seed each component with timing
        for component in components:
            component_start = time.perf_counter()
            
            try:
                seeding_func = RNG_COMPONENTS[component]
                
                # Handle special case for PYTHONHASHSEED
                if component == "pythonhashseed":
                    # Store original value for potential restoration
                    original_value = os.environ.get("PYTHONHASHSEED")
                    seeding_func(seed)
                    results["components"][component] = {
                        "status": "success",
                        "original_value": original_value,
                        "new_value": str(seed)
                    }
                else:
                    seeding_func(seed)
                    results["components"][component] = {"status": "success"}
                
                # Record component-specific timing
                component_duration = time.perf_counter() - component_start
                results["performance"][component] = component_duration
                context.component_timings[component] = component_duration
                
                # Check performance threshold
                threshold = SEED_PERFORMANCE_THRESHOLDS.get(f"{component}_seed", 0.1)
                if performance_tracking and component_duration > threshold:
                    logger.warning(
                        f"Slow seeding detected for {component}",
                        extra={
                            "component": component,
                            "duration": component_duration,
                            "threshold": threshold,
                            "metric_type": "slow_seed_operation"
                        }
                    )
                
                logger.debug(f"Successfully seeded {component}")
                
            except Exception as e:
                error_msg = f"Failed to seed {component}: {e}"
                logger.error(error_msg, extra={"component": component, "error": str(e)})
                results["components"][component] = {
                    "status": "error",
                    "error": str(e)
                }
                
                # Continue with other components but track failure
                if component in ["python_random", "numpy_random"]:
                    # These are critical components
                    raise RuntimeError(f"Critical seeding failure: {error_msg}")
        
        # Update context state
        context.global_seed = seed
        context.component_seeds.update({comp: seed for comp in components})
        context.is_seeded = True
        context.seed_timestamp = time.time()
        
        # Perform validation if requested
        if validate:
            validation_start = time.perf_counter()
            validation_results = validate_seed_determinism(seed, components)
            validation_duration = time.perf_counter() - validation_start
            
            results["validation"] = validation_results
            results["performance"]["validation"] = validation_duration
            context.validation_hash = validation_results.get("state_hash")
        
        # Record total timing
        total_duration = time.perf_counter() - start_time
        results["performance"]["total"] = total_duration
        context.setup_duration = total_duration
        
        # Update global registry
        with _registry_lock:
            _global_seed_registry[context.thread_id] = {
                "seed": seed,
                "components": components,
                "timestamp": time.time(),
                "validation_hash": context.validation_hash,
            }
        
        # Check overall performance threshold
        threshold = SEED_PERFORMANCE_THRESHOLDS["global_seed_setup"]
        if performance_tracking and total_duration > threshold:
            logger.warning(
                f"Slow global seed setup",
                extra={
                    "duration": total_duration,
                    "threshold": threshold,
                    "seed": seed,
                    "metric_type": "slow_global_seed"
                }
            )
        
        logger.success(
            f"Global seed {seed} applied successfully",
            extra={
                "seed": seed,
                "components_count": len([c for c in results["components"].values() 
                                      if c["status"] == "success"]),
                "total_duration": total_duration,
                **context.bind_logger_context()
            }
        )
    
    return results


def validate_seed_determinism(
    expected_seed: int,
    components: Optional[List[str]] = None,
    sample_operations: int = 10
) -> Dict[str, Any]:
    """
    Validate that seeding produced deterministic behavior across components.
    
    Performs sample random operations and validates consistent behavior to
    ensure proper seeding took effect.
    
    Args:
        expected_seed: The seed value that should be active
        components: Components to validate (default: all seeded components)
        sample_operations: Number of sample operations for validation
        
    Returns:
        Dict containing validation results and diagnostic information
        
    Example:
        >>> set_global_seed(42)
        >>> results = validate_seed_determinism(42)
        >>> assert results["is_deterministic"]
    """
    context = get_seed_context()
    
    if components is None:
        components = list(context.component_seeds.keys())
        if not components:
            components = ["python_random", "numpy_random"]
    
    validation_results = {
        "expected_seed": expected_seed,
        "is_deterministic": True,
        "component_results": {},
        "sample_data": {},
        "state_hash": None,
        "validation_timestamp": time.time(),
    }
    
    logger.debug(
        f"Validating seed determinism for seed {expected_seed}",
        extra={"seed": expected_seed, "components": components}
    )
    
    # Validate each component
    for component in components:
        component_result = {"status": "unknown", "samples": [], "is_consistent": True}
        
        try:
            if component == "python_random":
                # Test Python random module
                test_samples = [random.random() for _ in range(sample_operations)]
                component_result["samples"] = test_samples
                component_result["status"] = "validated"
                
                # Check for obvious non-randomness (all same values)
                if len(set(test_samples)) <= 1:
                    component_result["is_consistent"] = False
                    validation_results["is_deterministic"] = False
                    logger.warning(f"Python random appears non-functional")
                    
            elif component == "numpy_random":
                # Test NumPy random
                test_samples = np.random.random(sample_operations).tolist()
                component_result["samples"] = test_samples
                component_result["status"] = "validated"
                
                # Check for obvious non-randomness
                if len(set(test_samples)) <= 1:
                    component_result["is_consistent"] = False
                    validation_results["is_deterministic"] = False
                    logger.warning(f"NumPy random appears non-functional")
                    
            elif component == "pythonhashseed":
                # Validate environment variable setting
                current_value = os.environ.get("PYTHONHASHSEED")
                expected_value = str(expected_seed)
                
                component_result["current_value"] = current_value
                component_result["expected_value"] = expected_value
                component_result["is_consistent"] = (current_value == expected_value)
                component_result["status"] = "validated"
                
                if not component_result["is_consistent"]:
                    validation_results["is_deterministic"] = False
                    logger.warning(
                        f"PYTHONHASHSEED mismatch: expected {expected_value}, got {current_value}"
                    )
            
        except Exception as e:
            component_result["status"] = "error"
            component_result["error"] = str(e)
            component_result["is_consistent"] = False
            validation_results["is_deterministic"] = False
            logger.error(f"Validation failed for {component}: {e}")
        
        validation_results["component_results"][component] = component_result
    
    # Generate state hash for tracking
    validation_results["state_hash"] = context.get_state_hash()
    
    # Store sample data for reproducibility checking
    validation_results["sample_data"] = {
        comp: result.get("samples", [])
        for comp, result in validation_results["component_results"].items()
        if "samples" in result
    }
    
    if validation_results["is_deterministic"]:
        logger.debug(
            f"Seed determinism validation passed",
            extra={"seed": expected_seed, "state_hash": validation_results["state_hash"]}
        )
    else:
        logger.error(
            f"Seed determinism validation failed",
            extra={"seed": expected_seed, "failed_components": [
                comp for comp, result in validation_results["component_results"].items()
                if not result["is_consistent"]
            ]}
        )
    
    return validation_results


def get_gymnasium_seed_parameter(seed: Optional[int] = None) -> Dict[str, Any]:
    """
    Generate Gymnasium 0.29.x compliant seed parameter for reset() methods.
    
    Provides properly formatted seed parameter dictionary compatible with
    Gymnasium's expected seed format, supporting both explicit seeds and
    automatic seed generation.
    
    Args:
        seed: Optional explicit seed value
              If None, uses current global seed or generates new one
        
    Returns:
        Dict containing seed parameter for Gymnasium reset() method
        
    Example:
        >>> # For environment reset with specific seed
        >>> seed_params = get_gymnasium_seed_parameter(42)
        >>> obs, info = env.reset(**seed_params)
        >>> 
        >>> # For environment reset with automatic seed
        >>> seed_params = get_gymnasium_seed_parameter()
        >>> obs, info = env.reset(**seed_params)
    """
    context = get_seed_context()
    
    if seed is None:
        # Use current global seed if available
        if context.global_seed is not None:
            seed = context.global_seed
        else:
            # Generate new seed if none available
            seed = np.random.randint(0, 2**31 - 1)
            logger.info(f"Generated automatic seed: {seed}")
    
    # Validate seed range for Gymnasium compatibility
    if not isinstance(seed, int) or not (0 <= seed <= 2**31 - 1):
        raise ValueError(f"Gymnasium seed must be integer in range [0, 2^31-1], got {seed}")
    
    seed_params = {"seed": seed}
    
    logger.debug(
        f"Generated Gymnasium seed parameter",
        extra={"seed": seed, "context_seed": context.global_seed}
    )
    
    return seed_params


def setup_reproducible_environment(
    config: Union[SimulationConfig, Dict[str, Any], None] = None,
    force_reseed: bool = False
) -> Dict[str, Any]:
    """
    Setup comprehensive reproducible environment from configuration.
    
    Configures all random number generators and environment variables for
    reproducible experiments based on SimulationConfig or explicit parameters.
    
    Args:
        config: SimulationConfig object, dict, or None for defaults
        force_reseed: Whether to force re-seeding even if already seeded
        
    Returns:
        Dict containing setup results and applied configuration
        
    Example:
        >>> from odor_plume_nav.config.models import SimulationConfig
        >>> config = SimulationConfig(random_seed=42)
        >>> result = setup_reproducible_environment(config)
        >>> print(f"Environment seeded with {result['applied_seed']}")
    """
    context = get_seed_context()
    
    # Skip if already seeded and not forcing
    if context.is_seeded and not force_reseed:
        logger.debug(
            f"Environment already seeded with {context.global_seed}",
            extra=context.bind_logger_context()
        )
        return {
            "status": "already_seeded",
            "applied_seed": context.global_seed,
            "skipped": True,
            **context.bind_logger_context()
        }
    
    # Extract seed from configuration
    seed = None
    experiment_name = None
    
    if config is not None:
        if isinstance(config, SimulationConfig):
            seed = config.random_seed
            experiment_name = config.experiment_name
        elif isinstance(config, dict):
            seed = config.get("random_seed")
            experiment_name = config.get("experiment_name")
    
    # Generate seed if not provided
    if seed is None:
        seed = np.random.randint(0, 2**31 - 1)
        logger.info(f"Generated random seed: {seed}")
    
    with correlation_context(
        "reproducible_environment_setup",
        seed=seed,
        experiment_name=experiment_name,
        force_reseed=force_reseed
    ):
        logger.info(
            f"Setting up reproducible environment",
            extra={
                "seed": seed,
                "experiment_name": experiment_name,
                "force_reseed": force_reseed
            }
        )
        
        # Apply global seeding
        seed_results = set_global_seed(seed, validate=True)
        
        # Additional environment setup
        setup_results = {
            "status": "success",
            "applied_seed": seed,
            "experiment_name": experiment_name,
            "force_reseed": force_reseed,
            "seed_results": seed_results,
            "environment_setup": {},
        }
        
        # Set additional reproducibility environment variables
        env_vars = {
            "PYTHONHASHSEED": str(seed),
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",  # For PyTorch determinism
            "PYTHONDONTWRITEBYTECODE": "1",        # Avoid .pyc variations
        }
        
        for var, value in env_vars.items():
            original_value = os.environ.get(var)
            os.environ[var] = value
            setup_results["environment_setup"][var] = {
                "original": original_value,
                "new": value
            }
            logger.debug(f"Set {var}={value}")
        
        # Update context with experiment metadata
        if experiment_name:
            context.experiment_metadata["experiment_name"] = experiment_name
        
        logger.success(
            f"Reproducible environment setup complete",
            extra={
                "seed": seed,
                "components_seeded": len(seed_results["components"]),
                "env_vars_set": len(env_vars),
                **context.bind_logger_context()
            }
        )
    
    return setup_results


def create_thread_safe_seeder() -> Callable[[int], None]:
    """
    Create thread-safe seeding function for concurrent environment execution.
    
    Returns a seeding function that can be safely called from multiple threads
    without interference, maintaining proper isolation between concurrent
    environment instances.
    
    Returns:
        Callable seeding function that accepts seed integer
        
    Example:
        >>> seeder = create_thread_safe_seeder()
        >>> 
        >>> def worker_thread(thread_id):
        ...     seeder(thread_id + 1000)  # Thread-specific seed
        ...     # Thread-isolated random state
        >>> 
        >>> threads = [threading.Thread(target=worker_thread, args=(i,)) 
        ...           for i in range(5)]
        >>> for t in threads: t.start()
    """
    thread_lock = threading.RLock()
    
    def thread_safe_seed(seed: int) -> None:
        """Thread-safe seeding implementation."""
        with thread_lock:
            thread_id = threading.current_thread().ident
            
            logger.debug(
                f"Thread-safe seeding initiated",
                extra={"seed": seed, "thread_id": thread_id}
            )
            
            # Create thread-specific context if needed
            if not hasattr(_seed_context, 'context'):
                _seed_context.context = SeedContext()
            
            # Apply seeding to thread-local components
            set_global_seed(seed, components=["python_random", "numpy_random"])
            
            logger.debug(
                f"Thread-safe seeding complete",
                extra={"seed": seed, "thread_id": thread_id}
            )
    
    return thread_safe_seed


def generate_experiment_seed(
    experiment_name: str,
    base_seed: Optional[int] = None,
    hash_length: int = 8
) -> int:
    """
    Generate deterministic seed for named experiments.
    
    Creates reproducible seeds based on experiment names, enabling
    consistent seeding across experiment runs while maintaining
    different seeds for different experiments.
    
    Args:
        experiment_name: Name of the experiment
        base_seed: Optional base seed for additional entropy
        hash_length: Length of hash to use for seed generation
        
    Returns:
        Deterministic integer seed for the experiment
        
    Example:
        >>> seed1 = generate_experiment_seed("baseline_ppo")
        >>> seed2 = generate_experiment_seed("baseline_ppo")
        >>> assert seed1 == seed2  # Deterministic
        >>> 
        >>> seed3 = generate_experiment_seed("modified_ppo")
        >>> assert seed1 != seed3  # Different experiments get different seeds
    """
    # Create deterministic hash from experiment name and optional base seed
    hash_input = experiment_name
    if base_seed is not None:
        hash_input += f"_{base_seed}"
    
    # Generate hash and convert to seed
    hash_obj = hashlib.sha256(hash_input.encode("utf-8"))
    hash_hex = hash_obj.hexdigest()[:hash_length]
    seed = int(hash_hex, 16) % (2**31 - 1)  # Ensure valid seed range
    
    logger.debug(
        f"Generated experiment seed",
        extra={
            "experiment_name": experiment_name,
            "base_seed": base_seed,
            "generated_seed": seed,
            "hash_input": hash_input
        }
    )
    
    return seed


def performance_aware_seeding_wrapper(func: Callable) -> Callable:
    """
    Decorator for performance-aware seeding operations.
    
    Wraps seeding functions with performance monitoring and automatic
    threshold warnings for operations that exceed expected timing.
    
    Args:
        func: Function to wrap with performance monitoring
        
    Returns:
        Wrapped function with performance tracking
        
    Example:
        >>> @performance_aware_seeding_wrapper
        ... def custom_seeding_operation(seed):
        ...     set_global_seed(seed)
        ...     # Additional custom seeding logic
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        operation_name = f"{func.__name__}_seeding"
        
        with correlation_context(operation_name):
            try:
                result = func(*args, **kwargs)
                duration = time.perf_counter() - start_time
                
                # Check performance threshold
                threshold = SEED_PERFORMANCE_THRESHOLDS.get(operation_name, 0.1)
                if duration > threshold:
                    logger.warning(
                        f"Slow seeding operation: {operation_name}",
                        extra={
                            "operation": operation_name,
                            "duration": duration,
                            "threshold": threshold,
                            "metric_type": "slow_seed_wrapper"
                        }
                    )
                else:
                    logger.debug(
                        f"Seeding operation completed",
                        extra={
                            "operation": operation_name,
                            "duration": duration
                        }
                    )
                
                return result
                
            except Exception as e:
                duration = time.perf_counter() - start_time
                logger.error(
                    f"Seeding operation failed: {operation_name}",
                    extra={
                        "operation": operation_name,
                        "duration": duration,
                        "error": str(e)
                    }
                )
                raise
    
    return wrapper


def get_global_seed_registry() -> Dict[str, Any]:
    """
    Get copy of global seed registry for debugging and analysis.
    
    Returns:
        Dict containing current global seed registry state
        
    Example:
        >>> set_global_seed(42)
        >>> registry = get_global_seed_registry()
        >>> print(f"Current seeds: {registry}")
    """
    with _registry_lock:
        return dict(_global_seed_registry)


def clear_seed_registry(confirm: bool = False) -> None:
    """
    Clear global seed registry (use with caution).
    
    Args:
        confirm: Must be True to actually clear registry
        
    Example:
        >>> clear_seed_registry(confirm=True)  # Only if you're sure
    """
    if not confirm:
        raise ValueError("Must set confirm=True to clear seed registry")
    
    with _registry_lock:
        _global_seed_registry.clear()
        logger.warning("Global seed registry cleared")


# Enhanced exports for comprehensive seed management functionality
__all__ = [
    # Core seeding functions
    "set_global_seed",
    "validate_seed_determinism",
    "setup_reproducible_environment",
    
    # Gymnasium integration
    "get_gymnasium_seed_parameter",
    
    # Thread safety
    "create_thread_safe_seeder",
    
    # Context management
    "SeedContext",
    "get_seed_context",
    "set_seed_context",
    "seed_context_manager",
    
    # Experiment utilities
    "generate_experiment_seed",
    
    # Performance monitoring
    "performance_aware_seeding_wrapper",
    
    # Registry management
    "get_global_seed_registry",
    "clear_seed_registry",
    
    # Constants
    "RNG_COMPONENTS",
    "SEED_PERFORMANCE_THRESHOLDS",
]