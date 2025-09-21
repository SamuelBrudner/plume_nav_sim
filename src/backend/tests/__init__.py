"""
Lightweight test package initialization for plume_nav_sim test suite.

This module provides minimal test package setup without heavy import-time orchestration
to avoid test collection issues. Test discovery is handled by pytest's natural discovery.
"""

# Minimal imports to avoid collection issues
import os
import sys

# Add src to path if needed for absolute imports
_src_path = os.path.join(os.path.dirname(__file__), "..")
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

# Internal imports - Test infrastructure and fixtures
# Import fixtures lazily so the trimmed educational build can run a subset of tests
# without pulling in optional dependencies.  When the import fails we expose lightweight
# stubs that either no-op or trigger skips when exercised.
try:  # pragma: no cover - import guard for optional fixtures
    from .conftest import (  # Pytest fixtures for comprehensive test environment management; Context manager classes for resource management and performance tracking
        PerformanceTracker,
        TestEnvironmentManager,
        integration_test_env,
        performance_test_env,
        reproducibility_test_env,
        unit_test_env,
    )
except (
    ModuleNotFoundError
) as exc:  # pragma: no cover - exercised only in minimal kata builds
    raise ImportError(
        "Required test fixtures are missing; ensure src/backend/tests/conftest.py is available."
    ) from exc

# Internal imports - Main test classes and functions from test modules
try:  # pragma: no cover - import guard for optional API compliance tests
    from .test_environment_api import (  # Main environment API testing class for Gymnasium compliance validation; Core API compliance test functions
        TestEnvironmentAPI,
        test_environment_inheritance,
        test_gymnasium_api_compliance,
        test_seeding_and_reproducibility,
    )
except (
    ModuleNotFoundError
) as exc:  # pragma: no cover - exercised only in minimal kata builds
    raise ImportError(
        "Environment API tests are unavailable; ensure test_environment_api.py is present."
    ) from exc

# Performance modules are optional in the pared-down kata repository.  Guard the imports
# so we can still run lightweight contract tests without the heavy Gymnasium dependency
# tree.
try:  # pragma: no cover - import guard for optional performance tests
    from .test_performance import (  # Performance validation test class for benchmarking and performance target verification; Performance test functions for latency and resource validation
        TestPerformanceValidation,
        test_comprehensive_performance_suite,
        test_environment_step_latency_performance,
        test_memory_usage_constraints,
    )
except (
    ModuleNotFoundError,
    ImportError,
) as exc:  # pragma: no cover - exercised only in minimal kata builds
    raise ImportError(
        "Performance test suite could not be imported; ensure test_performance.py and its dependencies are available."
    ) from exc

try:  # pragma: no cover - import guard for optional integration tests
    from .test_integration import (  # Main integration test class for cross-component coordination and system-level testing; Integration test functions for end-to-end workflow validation
        TestEnvironmentIntegration,
        test_complete_episode_workflow,
        test_cross_component_seeding,
        test_system_level_performance,
    )
except (
    ModuleNotFoundError
) as exc:  # pragma: no cover - exercised only in minimal kata builds
    raise ImportError(
        "The integration test module is missing; add tests/test_integration.py to restore coverage."
    ) from exc

# Package metadata and version information
__version__ = "0.0.1"  # Test package version for compatibility and version tracking

# Test package identifier for test discovery and organization
TEST_PACKAGE_NAME = "plume_nav_sim_tests"

# Test category classification for organized execution
TEST_CATEGORIES = ["unit", "integration", "performance", "reproducibility", "edge_case"]

# Registry of pytest markers for test categorization and selective execution
PYTEST_MARKERS = {
    "unit": "Unit tests for individual component functionality",
    "integration": "Integration tests for cross-component interactions",
    "performance": "Performance tests for latency and resource validation",
    "reproducibility": "Reproducibility tests for deterministic behavior",
    "edge_case": "Edge case tests for boundary conditions and error handling",
}

# Shared test configuration dictionary for cross-module test settings
SHARED_TEST_CONFIG = {
    "default_timeout": 30.0,
    "memory_threshold_mb": 50.0,
    "performance_multiplier": 1.0,
    "cleanup_validation_enabled": True,
    "detailed_reporting_enabled": True,
}

TEST_DISCOVERY_ENABLED = True
