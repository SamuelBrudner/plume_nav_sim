
"""
Enhanced Testing Utilities for Odor Plume Navigation Core Modules.

This module provides centralized testing infrastructure including:
- Gymnasium API compliance validation utilities
- Hypothesis property-based testing framework integration
- Centralized Loguru logging configuration for test environments
- Performance testing utilities with step() latency validation (≤10ms requirement)
- Shared fixtures and utilities for coordinate frame consistency testing

The module integrates with the enhanced logging and configuration systems to ensure
comprehensive test coverage while maintaining performance requirements.
"""

import sys
import time
import threading
from contextlib import contextmanager
from typing import Any, Dict, Optional, Union, Callable, List
from pathlib import Path

# Core testing framework imports
import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch

# Gymnasium API compliance validation
try:
    import gymnasium
    from gymnasium.utils.env_checker import check_env
    GYMNASIUM_AVAILABLE = True
except ImportError:
    gymnasium = None
    check_env = None
    GYMNASIUM_AVAILABLE = False

# Hypothesis property-based testing framework
try:
    import hypothesis
    from hypothesis import given, assume, strategies as st
    from hypothesis.extra.numpy import arrays, array_shapes
    HYPOTHESIS_AVAILABLE = True
except ImportError:
    hypothesis = None
    given = None
    assume = None
    st = None
    arrays = None
    array_shapes = None
    HYPOTHESIS_AVAILABLE = False

# Import centralized logging configuration
try:
    from odor_plume_nav.utils.logging_setup import (
        LoggingConfig, 
        setup_logger, 
        get_logger,
        correlation_context,
        PerformanceMetrics
    )
    LOGGING_AVAILABLE = True
except ImportError:
    LoggingConfig = None
    setup_logger = None
    get_logger = None
    correlation_context = None
    PerformanceMetrics = None
    LOGGING_AVAILABLE = False

# Import configuration models for test setup
try:
    from odor_plume_nav.config.models import (
        NavigatorConfig,
        SingleAgentConfig,
        MultiAgentConfig,
        VideoPlumeConfig,
        SimulationConfig
    )
    CONFIG_MODELS_AVAILABLE = True
except ImportError:
    NavigatorConfig = None
    SingleAgentConfig = None
    MultiAgentConfig = None
    VideoPlumeConfig = None
    SimulationConfig = None
    CONFIG_MODELS_AVAILABLE = False


# Performance monitoring constants
STEP_LATENCY_THRESHOLD_MS = 10.0  # ≤10ms requirement for step() operations
FRAME_TIME_THRESHOLD_MS = 33.3    # ≥30 FPS requirement
MAX_MEMORY_PER_AGENT_MB = 0.1     # <10MB per 100 agents = 0.1MB per agent

# Test environment configuration
TEST_LOGGING_CONFIG = {
    "environment": "testing",
    "level": "INFO",
    "format": "minimal",
    "console_enabled": True,
    "file_enabled": False,
    "correlation_enabled": False,
    "memory_tracking": False,
    "enable_performance": True,
}


class TestPerformanceMonitor:
    """
    Performance monitoring utility for test validation with ≤10ms step() requirement.
    
    Provides high-resolution timing and memory tracking to ensure test performance
    meets the specified latency requirements for real-time simulation.
    """
    
    def __init__(self, operation_name: str = "test_operation"):
        self.operation_name = operation_name
        self.start_time = None
        self.end_time = None
        self.duration_ms = None
        self.memory_before = None
        self.memory_after = None
        
    def start_monitoring(self):
        """Begin performance monitoring with high-resolution timing."""
        self.start_time = time.perf_counter()
        try:
            import psutil
            process = psutil.Process()
            self.memory_before = process.memory_info().rss / 1024 / 1024  # MB
        except ImportError:
            self.memory_before = None
    
    def stop_monitoring(self):
        """Complete performance monitoring and calculate metrics."""
        self.end_time = time.perf_counter()
        self.duration_ms = (self.end_time - self.start_time) * 1000
        
        try:
            import psutil
            process = psutil.Process()
            self.memory_after = process.memory_info().rss / 1024 / 1024  # MB
        except ImportError:
            self.memory_after = None
    
    @property
    def memory_delta_mb(self) -> Optional[float]:
        """Calculate memory usage delta in MB."""
        if self.memory_before is not None and self.memory_after is not None:
            return self.memory_after - self.memory_before
        return None
    
    def assert_step_performance(self, threshold_ms: float = STEP_LATENCY_THRESHOLD_MS):
        """Assert that operation meets step() latency requirements."""
        if self.duration_ms is None:
            raise ValueError("Performance monitoring not completed")
        
        assert self.duration_ms <= threshold_ms, (
            f"{self.operation_name} took {self.duration_ms:.2f}ms, "
            f"exceeds threshold of {threshold_ms:.2f}ms"
        )
    
    def assert_frame_performance(self, threshold_ms: float = FRAME_TIME_THRESHOLD_MS):
        """Assert that operation meets frame time requirements (≥30 FPS)."""
        if self.duration_ms is None:
            raise ValueError("Performance monitoring not completed")
        
        assert self.duration_ms <= threshold_ms, (
            f"{self.operation_name} took {self.duration_ms:.2f}ms, "
            f"exceeds frame time threshold of {threshold_ms:.2f}ms (required for ≥30 FPS)"
        )
    
    def assert_memory_efficiency(self, num_agents: int = 1):
        """Assert memory usage meets efficiency requirements."""
        if self.memory_delta_mb is None:
            return  # Skip if memory tracking unavailable
        
        max_memory_mb = num_agents * MAX_MEMORY_PER_AGENT_MB
        assert self.memory_delta_mb <= max_memory_mb, (
            f"{self.operation_name} used {self.memory_delta_mb:.3f}MB for {num_agents} agent(s), "
            f"exceeds limit of {max_memory_mb:.3f}MB"
        )


@contextmanager
def performance_timer(operation_name: str = "test_operation"):
    """
    Context manager for performance timing with automatic assertion.
    
    Args:
        operation_name: Name of the operation being timed
        
    Yields:
        TestPerformanceMonitor: Performance monitor for the operation
        
    Example:
        >>> with performance_timer("env_step") as perf:
        ...     obs, reward, done, info = env.step(action)
        >>> perf.assert_step_performance()
    """
    monitor = TestPerformanceMonitor(operation_name)
    monitor.start_monitoring()
    try:
        yield monitor
    finally:
        monitor.stop_monitoring()


def validate_gymnasium_api_compliance(env, fast_check: bool = False):
    """
    Validate Gymnasium API compliance using gymnasium.utils.env_checker.
    
    Args:
        env: Environment instance to validate
        fast_check: Whether to perform fast validation (skips some checks)
        
    Raises:
        AssertionError: If environment fails API compliance checks
        ImportError: If gymnasium is not available
    """
    if not GYMNASIUM_AVAILABLE:
        pytest.skip("Gymnasium not available for API compliance testing")
    
    if check_env is None:
        pytest.skip("gymnasium.utils.env_checker not available")
    
    try:
        # Use fast_check to avoid time-consuming validation in CI
        check_env(env, warn=True, skip_render_check=fast_check)
    except Exception as e:
        pytest.fail(f"Gymnasium API compliance check failed: {e}")


def create_test_config(config_type: str = "navigator", **overrides) -> Dict[str, Any]:
    """
    Create test configuration dictionaries with enhanced validation.
    
    Args:
        config_type: Type of configuration ("navigator", "video_plume", "simulation")
        **overrides: Configuration parameter overrides
        
    Returns:
        Dictionary configuration for testing
    """
    base_configs = {
        "navigator": {
            "mode": "single",
            "position": [50.0, 50.0],
            "orientation": 0.0,
            "speed": 1.0,
            "max_speed": 2.0,
            "angular_velocity": 0.1,
        },
        "multi_navigator": {
            "mode": "multi",
            "num_agents": 3,
            "positions": [[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]],
            "orientations": [0.0, 90.0, 180.0],
            "speeds": [1.0, 1.5, 0.8],
            "max_speeds": [2.0, 2.0, 2.0],
            "angular_velocities": [0.1, 0.1, 0.1],
        },
        "video_plume": {
            "video_path": "/tmp/test_plume.mp4",
            "flip": False,
            "grayscale": True,
            "kernel_size": 5,
            "kernel_sigma": 1.0,
            "normalize": True,
        },
        "simulation": {
            "max_steps": 100,
            "step_size": 1.0,
            "record_trajectory": True,
            "output_format": "numpy",
            "enable_visualization": False,
            "random_seed": 42,
        },
    }
    
    config = base_configs.get(config_type, {}).copy()
    config.update(overrides)
    return config


# Hypothesis strategies for property-based testing
if HYPOTHESIS_AVAILABLE:
    # Coordinate strategies for position testing
    coordinate_strategy = st.floats(
        min_value=-1000.0, 
        max_value=1000.0, 
        allow_nan=False, 
        allow_infinity=False
    )
    
    position_strategy = st.tuples(coordinate_strategy, coordinate_strategy)
    
    # Angle strategies for orientation testing
    angle_strategy = st.floats(
        min_value=0.0, 
        max_value=360.0, 
        allow_nan=False, 
        allow_infinity=False
    )
    
    # Speed strategies for navigation testing
    speed_strategy = st.floats(
        min_value=0.0, 
        max_value=10.0, 
        allow_nan=False, 
        allow_infinity=False
    )
    
    # Multi-agent list strategies
    multi_agent_positions_strategy = st.lists(
        position_strategy,
        min_size=1,
        max_size=10
    )
    
    multi_agent_angles_strategy = lambda size: st.lists(
        angle_strategy,
        min_size=size,
        max_size=size
    )
    
    multi_agent_speeds_strategy = lambda size: st.lists(
        speed_strategy,
        min_size=size,
        max_size=size
    )


def setup_test_logging():
    """
    Configure centralized Loguru logging for test environment.
    
    Sets up logging with minimal output and performance monitoring
    suitable for test execution environments.
    """
    if not LOGGING_AVAILABLE:
        # Fallback to basic logging if Loguru not available
        import logging
        logging.basicConfig(level=logging.WARNING)
        return
    
    # Configure minimal logging for test environment
    config = LoggingConfig(**TEST_LOGGING_CONFIG)
    setup_logger(config)


def coordinate_frame_consistency_test(transform_func: Callable, 
                                    test_positions: List[tuple],
                                    tolerance: float = 1e-10) -> bool:
    """
    Test coordinate frame transformations for consistency and precision.
    
    Args:
        transform_func: Function that transforms coordinates
        test_positions: List of (x, y) coordinate tuples to test
        tolerance: Numerical tolerance for floating-point comparisons
        
    Returns:
        True if all transformations maintain consistency
        
    Raises:
        AssertionError: If coordinate frame consistency is violated
    """
    for pos in test_positions:
        x, y = pos
        
        # Test identity transformation (transform then inverse)
        transformed = transform_func(x, y)
        if hasattr(transform_func, 'inverse'):
            restored = transform_func.inverse(*transformed)
            
            # Check coordinate preservation within tolerance
            assert abs(restored[0] - x) < tolerance, (
                f"X coordinate not preserved: {x} -> {transformed[0]} -> {restored[0]}"
            )
            assert abs(restored[1] - y) < tolerance, (
                f"Y coordinate not preserved: {y} -> {transformed[1]} -> {restored[1]}"
            )
        
        # Test numerical stability
        assert not np.isnan(transformed[0]), f"Transform produced NaN for x={x}, y={y}"
        assert not np.isnan(transformed[1]), f"Transform produced NaN for x={x}, y={y}"
        assert not np.isinf(transformed[0]), f"Transform produced Inf for x={x}, y={y}"
        assert not np.isinf(transformed[1]), f"Transform produced Inf for x={x}, y={y}"
    
    return True


# Test fixtures for common testing scenarios
@pytest.fixture
def test_logger():
    """Provide configured test logger."""
    if LOGGING_AVAILABLE:
        return get_logger("test")
    else:
        return Mock()


@pytest.fixture
def performance_monitor():
    """Provide performance monitor for test timing."""
    return TestPerformanceMonitor


@pytest.fixture
def navigator_config():
    """Provide standard navigator configuration for testing."""
    return create_test_config("navigator")


@pytest.fixture
def multi_agent_config():
    """Provide multi-agent configuration for testing."""
    return create_test_config("multi_navigator")


@pytest.fixture
def video_plume_config():
    """Provide video plume configuration for testing."""
    return create_test_config("video_plume")


@pytest.fixture
def simulation_config():
    """Provide simulation configuration for testing."""
    return create_test_config("simulation")


# Performance testing decorators
def requires_performance_validation(threshold_ms: float = STEP_LATENCY_THRESHOLD_MS):
    """
    Decorator to ensure test operations meet performance requirements.
    
    Args:
        threshold_ms: Maximum allowed duration in milliseconds
        
    Example:
        @requires_performance_validation(10.0)
        def test_step_performance(env):
            env.step(action)
    """
    def decorator(test_func):
        def wrapper(*args, **kwargs):
            with performance_timer(test_func.__name__) as perf:
                result = test_func(*args, **kwargs)
            perf.assert_step_performance(threshold_ms)
            return result
        return wrapper
    return decorator


def requires_gymnasium_compliance(fast_check: bool = True):
    """
    Decorator to ensure environments pass Gymnasium API compliance.
    
    Args:
        fast_check: Whether to perform fast validation
        
    Example:
        @requires_gymnasium_compliance()
        def test_environment_api(env):
            # Test will automatically validate API compliance
            pass
    """
    def decorator(test_func):
        def wrapper(*args, **kwargs):
            # Extract environment from args/kwargs
            env = None
            if args and hasattr(args[0], 'step'):
                env = args[0]
            elif 'env' in kwargs:
                env = kwargs['env']
            
            if env is not None:
                validate_gymnasium_api_compliance(env, fast_check)
            
            return test_func(*args, **kwargs)
        return wrapper
    return decorator


# Initialize test logging on module import
setup_test_logging()


# Export testing utilities
__all__ = [
    # Performance monitoring
    "TestPerformanceMonitor",
    "performance_timer",
    "requires_performance_validation",
    
    # API compliance validation
    "validate_gymnasium_api_compliance", 
    "requires_gymnasium_compliance",
    
    # Configuration utilities
    "create_test_config",
    "setup_test_logging",
    
    # Coordinate frame testing
    "coordinate_frame_consistency_test",
    
    # Availability flags
    "GYMNASIUM_AVAILABLE",
    "HYPOTHESIS_AVAILABLE", 
    "LOGGING_AVAILABLE",
    "CONFIG_MODELS_AVAILABLE",
    
    # Constants
    "STEP_LATENCY_THRESHOLD_MS",
    "FRAME_TIME_THRESHOLD_MS",
    "MAX_MEMORY_PER_AGENT_MB",
]

# Conditional exports based on availability
if HYPOTHESIS_AVAILABLE:
    __all__.extend([
        "coordinate_strategy",
        "position_strategy", 
        "angle_strategy",
        "speed_strategy",
        "multi_agent_positions_strategy",
        "multi_agent_angles_strategy",
        "multi_agent_speeds_strategy",
    ])