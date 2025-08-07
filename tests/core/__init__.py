"""
Core Testing Infrastructure for Plume Navigation Simulation.

This module provides centralized testing utilities, performance monitoring,
and common test infrastructure used across the entire test suite.

Key Components:
- TestPerformanceMonitor: Comprehensive performance monitoring for test validation
- performance_timer: Context manager for timing test operations
- Test configuration utilities and factories
- Hypothesis property-based testing strategies
- Common test constants and thresholds
- Logging setup for test environments
- Gymnasium API compliance validation utilities

Performance Requirements:
- Step execution: <10ms per simulation step
- Frame processing: <33.3ms per frame for real-time requirements
- Memory usage: <0.1MB per agent for efficient multi-agent scenarios
- Test execution: Fail tests that exceed performance thresholds

Author: Blitzy Agent
Version: 1.0.0
"""

import sys
import time
import threading
import warnings
from contextlib import contextmanager
from typing import Any, Dict, Optional, Union, Callable, List, Tuple
from pathlib import Path
import psutil

# Core testing framework imports
import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch

# Optional dependency handling
try:
    import gymnasium
    from gymnasium.utils.env_checker import check_env
    GYMNASIUM_AVAILABLE = True
except ImportError:
    gymnasium = None
    check_env = None
    GYMNASIUM_AVAILABLE = False

try:
    import hypothesis
    from hypothesis import given, strategies as st
    from hypothesis.extra.numpy import arrays, array_shapes
    HYPOTHESIS_AVAILABLE = True
except ImportError:
    hypothesis = None
    given = None
    st = None
    arrays = None
    array_shapes = None
    HYPOTHESIS_AVAILABLE = False

# Performance monitoring constants
STEP_LATENCY_THRESHOLD_MS = 10.0      # Maximum allowed step execution time
FRAME_TIME_THRESHOLD_MS = 33.3        # Real-time frame processing threshold
MAX_MEMORY_PER_AGENT_MB = 0.1         # Memory limit per agent
SENSOR_SAMPLING_THRESHOLD_MS = 1.0    # Sensor sampling latency threshold
NAVIGATION_UPDATE_THRESHOLD_MS = 5.0  # Navigation update threshold

# Test environment configuration
TEST_LOGGING_CONFIG = {
    "environment": "testing",
    "level": "WARNING",
    "format": "minimal", 
    "console_enabled": True,
    "file_enabled": False,
    "correlation_enabled": False,
    "memory_tracking": True,
    "enable_performance": True,
}


class TestPerformanceMonitor:
    """
    Comprehensive performance monitoring for test validation.
    
    Provides timing, memory tracking, and performance validation utilities
    for ensuring test operations meet performance requirements.
    """
    
    def __init__(self, test_name: str = "unknown_test"):
        self.test_name = test_name
        self.start_time = None
        self.end_time = None
        self.duration_ms = None
        self.memory_before = None
        self.memory_after = None
        self.process = psutil.Process()
        
    def start_monitoring(self):
        """Begin performance monitoring."""
        self.start_time = time.perf_counter()
        try:
            self.memory_before = self.process.memory_info().rss / 1024 / 1024  # MB
        except:
            self.memory_before = None
    
    def stop_monitoring(self):
        """Complete performance monitoring and calculate metrics."""
        self.end_time = time.perf_counter()
        self.duration_ms = (self.end_time - self.start_time) * 1000
        
        try:
            self.memory_after = self.process.memory_info().rss / 1024 / 1024  # MB
        except:
            self.memory_after = None
    
    @property
    def memory_delta_mb(self) -> Optional[float]:
        """Calculate memory usage delta in MB."""
        if self.memory_before is not None and self.memory_after is not None:
            return self.memory_after - self.memory_before
        return None
    
    def assert_step_performance(self, threshold_ms: float = STEP_LATENCY_THRESHOLD_MS):
        """Assert test operation meets step performance requirements."""
        if self.duration_ms is None:
            raise ValueError("Performance monitoring not completed")
        
        assert self.duration_ms <= threshold_ms, (
            f"Test '{self.test_name}' took {self.duration_ms:.2f}ms, "
            f"exceeds step threshold of {threshold_ms:.2f}ms"
        )
    
    def assert_frame_performance(self, threshold_ms: float = FRAME_TIME_THRESHOLD_MS):
        """Assert test operation meets frame processing requirements."""
        if self.duration_ms is None:
            raise ValueError("Performance monitoring not completed")
        
        assert self.duration_ms <= threshold_ms, (
            f"Test '{self.test_name}' took {self.duration_ms:.2f}ms, "
            f"exceeds frame threshold of {threshold_ms:.2f}ms"
        )
    
    def assert_memory_efficiency(self, max_memory_mb: float = MAX_MEMORY_PER_AGENT_MB):
        """Assert memory usage meets efficiency requirements."""
        if self.memory_delta_mb is None:
            return  # Skip if memory tracking unavailable
        
        assert self.memory_delta_mb <= max_memory_mb, (
            f"Test '{self.test_name}' used {self.memory_delta_mb:.3f}MB, "
            f"exceeds limit of {max_memory_mb:.3f}MB"
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all performance metrics as dictionary."""
        return {
            "test_name": self.test_name,
            "duration_ms": self.duration_ms,
            "memory_delta_mb": self.memory_delta_mb,
            "start_time": self.start_time,
            "end_time": self.end_time
        }


@contextmanager
def performance_timer(test_name: str = "test_operation", 
                     auto_assert: bool = True,
                     threshold_ms: float = STEP_LATENCY_THRESHOLD_MS):
    """
    Context manager for performance timing with automatic validation.
    
    Args:
        test_name: Name of the test operation being timed
        auto_assert: Whether to automatically assert performance requirements
        threshold_ms: Performance threshold in milliseconds
        
    Yields:
        TestPerformanceMonitor: Performance monitor for the operation
        
    Example:
        >>> with performance_timer("navigation_step", threshold_ms=5.0) as perf:
        ...     navigator.step()
        >>> # Performance automatically validated on exit
    """
    monitor = TestPerformanceMonitor(test_name)
    monitor.start_monitoring()
    try:
        yield monitor
    finally:
        monitor.stop_monitoring()
        if auto_assert:
            monitor.assert_step_performance(threshold_ms)
            monitor.assert_memory_efficiency()


def requires_performance_validation(threshold_ms: float = STEP_LATENCY_THRESHOLD_MS,
                                  memory_mb: float = MAX_MEMORY_PER_AGENT_MB):
    """
    Decorator to ensure test operations meet performance requirements.
    
    Args:
        threshold_ms: Maximum allowed duration in milliseconds
        memory_mb: Maximum allowed memory usage in MB
        
    Example:
        @requires_performance_validation(5.0)
        def test_fast_operation():
            # Test will automatically validate performance
            pass
    """
    def decorator(test_func):
        def wrapper(*args, **kwargs):
            with performance_timer(test_func.__name__, threshold_ms=threshold_ms) as perf:
                result = test_func(*args, **kwargs)
            perf.assert_memory_efficiency(memory_mb)
            return result
        return wrapper
    return decorator


def validate_gymnasium_api_compliance(env: Any) -> bool:
    """
    Validate that an environment complies with Gymnasium API requirements.
    
    Args:
        env: Environment to validate
        
    Returns:
        True if environment is compliant
        
    Raises:
        AssertionError: If environment fails compliance checks
    """
    if not GYMNASIUM_AVAILABLE:
        pytest.skip("Gymnasium not available for API compliance testing")
    
    # Use gymnasium's built-in checker if available
    if check_env is not None:
        try:
            check_env(env, warn=True)
            return True
        except Exception as e:
            pytest.fail(f"Gymnasium API compliance failed: {e}")
    
    # Fallback manual checks
    required_methods = ['reset', 'step', 'render', 'close']
    for method_name in required_methods:
        assert hasattr(env, method_name), f"Missing required method: {method_name}"
        assert callable(getattr(env, method_name)), f"Method not callable: {method_name}"
    
    # Check that reset returns proper format
    obs, info = env.reset()
    assert obs is not None, "reset() must return observation"
    assert isinstance(info, dict), "reset() must return info dict"
    
    # Check that step returns proper format
    action = env.action_space.sample() if hasattr(env, 'action_space') else 0
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs is not None, "step() must return observation"
    assert isinstance(reward, (int, float, np.number)), "step() must return numeric reward"
    assert isinstance(terminated, bool), "step() must return boolean terminated"
    assert isinstance(truncated, bool), "step() must return boolean truncated"
    assert isinstance(info, dict), "step() must return info dict"
    
    return True


def create_test_config(config_type: str = "default", **overrides) -> Dict[str, Any]:
    """
    Create test configuration dictionaries for various test scenarios.
    
    Args:
        config_type: Type of configuration to create
        **overrides: Configuration parameter overrides
        
    Returns:
        Dictionary configuration for testing
    """
    base_configs = {
        "default": {
            "simulation": {
                "max_steps": 100,
                "step_size": 0.1,
                "random_seed": 42
            },
            "navigator": {
                "position": [0.0, 0.0],
                "orientation": 0.0,
                "speed": 1.0,
                "max_speed": 2.0
            },
            "environment": {
                "domain_bounds": [[0, 50], [0, 50]]
            }
        },
        "performance": {
            "simulation": {
                "max_steps": 1000,
                "step_size": 0.01,  # Smaller step for stress testing
                "random_seed": 42
            },
            "navigator": {
                "position": [0.0, 0.0],
                "orientation": 0.0,
                "speed": 2.0,
                "max_speed": 5.0
            },
            "environment": {
                "domain_bounds": [[0, 100], [0, 100]]
            }
        },
        "minimal": {
            "simulation": {
                "max_steps": 10,
                "step_size": 1.0,
                "random_seed": 42
            },
            "navigator": {
                "position": [0.0, 0.0],
                "orientation": 0.0,
                "speed": 1.0,
                "max_speed": 1.0
            }
        }
    }
    
    config = base_configs.get(config_type, base_configs["default"]).copy()
    
    # Apply overrides recursively
    def apply_overrides(base_dict, override_dict):
        for key, value in override_dict.items():
            if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
                apply_overrides(base_dict[key], value)
            else:
                base_dict[key] = value
    
    apply_overrides(config, overrides)
    return config


def setup_test_logging():
    """
    Configure logging for test environment.
    
    Sets up minimal logging configuration suitable for test execution
    with performance monitoring enabled.
    """
    try:
        # Try to use the project's logging setup if available
        from plume_nav_sim.utils.logging_setup import setup_logger, LoggingConfig
        config = LoggingConfig(**TEST_LOGGING_CONFIG)
        setup_logger(config)
    except ImportError:
        # Fallback to basic logging configuration
        import logging
        logging.basicConfig(
            level=logging.WARNING,
            format='%(levelname)s: %(message)s'
        )


# Hypothesis strategies for property-based testing
if HYPOTHESIS_AVAILABLE:
    # Coordinate strategies
    coordinate_strategy = st.floats(
        min_value=-100.0,
        max_value=100.0,
        allow_nan=False,
        allow_infinity=False
    )
    
    # Position strategies
    position_strategy = st.tuples(coordinate_strategy, coordinate_strategy)
    
    # Angle strategies
    angle_strategy = st.floats(
        min_value=0.0,
        max_value=2*np.pi,
        allow_nan=False,
        allow_infinity=False
    )
    
    # Speed strategies
    speed_strategy = st.floats(
        min_value=0.0,
        max_value=10.0,
        allow_nan=False,
        allow_infinity=False
    )
    
    # Time step strategies
    time_step_strategy = st.floats(
        min_value=0.01,
        max_value=1.0,
        allow_nan=False,
        allow_infinity=False
    )
    
    # Multi-position strategies
    position_list_strategy = st.lists(
        position_strategy,
        min_size=1,
        max_size=10
    )


# Test fixtures for common test scenarios
@pytest.fixture
def performance_monitor():
    """Provide performance monitor for test validation."""
    return TestPerformanceMonitor


@pytest.fixture
def test_config():
    """Provide default test configuration."""
    return create_test_config("default")


@pytest.fixture
def minimal_config():
    """Provide minimal test configuration for quick tests."""
    return create_test_config("minimal")


@pytest.fixture
def performance_config():
    """Provide performance test configuration."""
    return create_test_config("performance")


# Initialize test logging on module import
setup_test_logging()


# Export all testing utilities
__all__ = [
    # Performance monitoring
    "TestPerformanceMonitor",
    "performance_timer", 
    "requires_performance_validation",
    
    # API compliance validation
    "validate_gymnasium_api_compliance",
    
    # Configuration utilities
    "create_test_config",
    
    # Logging setup
    "setup_test_logging",
    
    # Constants
    "STEP_LATENCY_THRESHOLD_MS",
    "FRAME_TIME_THRESHOLD_MS", 
    "MAX_MEMORY_PER_AGENT_MB",
    "SENSOR_SAMPLING_THRESHOLD_MS",
    "NAVIGATION_UPDATE_THRESHOLD_MS",
    
    # Availability flags
    "GYMNASIUM_AVAILABLE",
    "HYPOTHESIS_AVAILABLE",
]

# Conditional exports based on availability
if HYPOTHESIS_AVAILABLE:
    __all__.extend([
        "coordinate_strategy",
        "position_strategy",
        "angle_strategy", 
        "speed_strategy",
        "time_step_strategy",
        "position_list_strategy",
    ])