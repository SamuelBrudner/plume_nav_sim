"""
Minimal data package initializer.

This module intentionally avoids importing heavy submodules at import time to
keep test collection reliable. Utilities can be imported directly from their
respective modules, e.g.:
  - data.benchmark_data
  - data.test_scenarios
  - data.example_configs
"""

__all__: list[str] = []
