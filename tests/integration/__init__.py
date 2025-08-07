"""Integration test utilities."""

import time
import warnings
from typing import Any, Dict, Optional
from contextlib import contextmanager

# Migration testing constants
MIGRATION_SLA_THRESHOLD_MS = 50.0  # 50ms threshold for migration performance
MIGRATION_TOLERANCE = 1e-6  # Tolerance for behavioral parity
MIGRATION_DETERMINISTIC_EXECUTIONS = 5  # Number of executions for deterministic validation
MIGRATION_VALIDATION_STEPS = 100  # Number of steps for migration validation

class IntegrationTestPerformanceMonitor:
    """Monitor performance metrics during integration tests."""
    
    def __init__(self):
        self.measurements = {}
        self.migration_metrics = {}
    
    def add_migration_measurement(self, name: str, duration_ms: float):
        """Add a migration performance measurement."""
        if name not in self.migration_metrics:
            self.migration_metrics[name] = []
        self.migration_metrics[name].append(duration_ms)
    
    def check_migration_sla(self):
        """Check if migration measurements meet SLA."""
        violations = []
        for name, measurements in self.migration_metrics.items():
            avg_time = sum(measurements) / len(measurements)
            if avg_time > MIGRATION_SLA_THRESHOLD_MS:
                violations.append(f"Migration {name}: {avg_time:.2f}ms > {MIGRATION_SLA_THRESHOLD_MS}ms")
        return violations


def create_migration_test_config(**kwargs) -> Dict[str, Any]:
    """Create test configuration for migration tests."""
    config = {
        "enable_migration_mode": True,
        "enable_deprecation_warnings": True,
        "enable_behavioral_validation": True,
        "migration_tolerance": MIGRATION_TOLERANCE,
        "performance_monitoring": True,
    }
    config.update(kwargs)
    return config


def validate_behavioral_parity(old_result: Any, new_result: Any, tolerance: float = MIGRATION_TOLERANCE) -> bool:
    """Validate that old and new implementations produce equivalent results."""
    try:
        import numpy as np
        if hasattr(old_result, '__array__') and hasattr(new_result, '__array__'):
            return np.allclose(old_result, new_result, atol=tolerance)
        elif isinstance(old_result, (int, float)) and isinstance(new_result, (int, float)):
            return abs(old_result - new_result) <= tolerance
        else:
            return old_result == new_result
    except ImportError:
        # Fallback without numpy
        if isinstance(old_result, (int, float)) and isinstance(new_result, (int, float)):
            return abs(old_result - new_result) <= tolerance
        else:
            return old_result == new_result


def validate_deterministic_seeding(func, seed: int = 42, num_runs: int = 3) -> bool:
    """Validate that a function produces deterministic results with seeding."""
    results = []
    for _ in range(num_runs):
        try:
            import numpy as np
            np.random.seed(seed)
        except ImportError:
            import random
            random.seed(seed)
        
        result = func()
        results.append(result)
    
    # Check if all results are identical
    first_result = results[0]
    return all(validate_behavioral_parity(first_result, result) for result in results[1:])


def validate_deprecation_warnings(func, expected_warning_count: int = 1) -> bool:
    """Validate that deprecated functionality generates appropriate warnings."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        func()
        
        deprecation_warnings = [warning for warning in w if issubclass(warning.category, DeprecationWarning)]
        return len(deprecation_warnings) >= expected_warning_count


@contextmanager
def setup_migration_test_environment():
    """Setup environment for migration testing."""
    old_warnings_setting = warnings.filters.copy()
    try:
        warnings.simplefilter("always", DeprecationWarning)
        yield
    finally:
        warnings.filters[:] = old_warnings_setting


__all__ = [
    "IntegrationTestPerformanceMonitor",
    "create_migration_test_config", 
    "validate_behavioral_parity",
    "validate_deterministic_seeding",
    "validate_deprecation_warnings",
    "setup_migration_test_environment",
    "MIGRATION_SLA_THRESHOLD_MS",
    "MIGRATION_TOLERANCE",
    "MIGRATION_DETERMINISTIC_EXECUTIONS",
    "MIGRATION_VALIDATION_STEPS"
]