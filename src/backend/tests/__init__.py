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

# Internal imports - ensure the shared conftest module is available and fail fast if missing.
try:
    from . import conftest as _shared_conftest  # type: ignore
except Exception as import_error:  # pragma: no cover - exercised via guardrail tests
    raise ImportError(
        "The plume_nav_sim backend test suite requires tests/conftest.py. "
        "Ensure the shared test fixtures are available before running tests."
    ) from import_error
else:
    if not hasattr(_shared_conftest, "_require_psutil"):
        raise ImportError(
            "tests/conftest.py is missing required guard `_require_psutil`. "
            "Verify the correct shared conftest module is available."
        )

_PYTEST_AUTOCONFIG = True  # Marker for test tooling; no functional effect

"""
Avoid importing individual test modules here. Pytest will discover tests on its
own, and importing them at package import time can cause import-time side
effects and optional-dependency failures (e.g., psutil in performance tests).
"""

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
