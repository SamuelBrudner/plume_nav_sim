"""Test utilities for performance monitoring and validation."""

import time
import functools
from contextlib import contextmanager
from typing import Any, Dict, Optional, Union
from types import SimpleNamespace
import warnings

# Performance threshold constants
STEP_LATENCY_THRESHOLD_MS = 10.0  # 10ms threshold for step operations

class PerformanceMonitor:
    """Monitor performance metrics during tests."""
    
    def __init__(self):
        self.measurements = {}
        self.thresholds = {}
    
    def add_measurement(self, name: str, duration_ms: float):
        """Add a performance measurement."""
        if name not in self.measurements:
            self.measurements[name] = []
        self.measurements[name].append(duration_ms)
    
    def set_threshold(self, name: str, threshold_ms: float):
        """Set performance threshold for a measurement."""
        self.thresholds[name] = threshold_ms
    
    def check_thresholds(self):
        """Check if all measurements meet their thresholds."""
        violations = []
        for name, measurements in self.measurements.items():
            if name in self.thresholds:
                avg_time = sum(measurements) / len(measurements)
                if avg_time > self.thresholds[name]:
                    violations.append(f"{name}: {avg_time:.2f}ms > {self.thresholds[name]}ms")
        return violations

    def record_step(self, duration_ms: float, label: str = "step"):
        """Record timing information for a simulation step."""
        if duration_ms < 0:
            raise ValueError("duration_ms must be non-negative")
        self.add_measurement(label, duration_ms)


@contextmanager
def performance_timer(name: str = "operation"):
    """Context manager to time operations."""
    start_time = time.perf_counter()
    result = SimpleNamespace(duration_ms=0.0)
    try:
        yield result
    finally:
        end_time = time.perf_counter()
        result.duration_ms = (end_time - start_time) * 1000
        print(f"{name} took {result.duration_ms:.2f}ms")


def requires_performance_validation(threshold_ms: float):
    """Decorator to validate performance of test functions.
    
    Args:
        threshold_ms: Maximum allowed execution time in milliseconds
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
            finally:
                end_time = time.perf_counter()
                duration_ms = (end_time - start_time) * 1000
                
                if duration_ms > threshold_ms:
                    warnings.warn(
                        f"Performance test {func.__name__} exceeded threshold: "
                        f"{duration_ms:.2f}ms > {threshold_ms}ms",
                        UserWarning
                    )
            return result
        return wrapper
    return decorator


def create_test_config(
    sensor_type: str = "binary",
    threshold: float = 0.1,
    **kwargs
) -> Dict[str, Any]:
    """Create test configuration for sensors.
    
    Args:
        sensor_type: Type of sensor ('binary', 'concentration', 'gradient')
        threshold: Detection threshold for binary sensor
        **kwargs: Additional configuration parameters
    
    Returns:
        Dictionary with test configuration
    """
    base_config = {
        "sensor_type": sensor_type,
        "enable_logging": False,
        "enable_performance_monitoring": True,
    }
    
    if sensor_type == "binary":
        base_config.update({
            "threshold": threshold,
            "detection_radius": kwargs.get("detection_radius", 10.0),
        })
    elif sensor_type == "concentration":
        base_config.update({
            "dynamic_range": kwargs.get("dynamic_range", (0.0, 1.0)),
            "noise_level": kwargs.get("noise_level", 0.0),
        })
    elif sensor_type == "gradient":
        base_config.update({
            "method": kwargs.get("method", "central"),
            "epsilon": kwargs.get("epsilon", 1e-6),
        })
    
    # Add any additional parameters
    base_config.update(kwargs)
    
    return base_config