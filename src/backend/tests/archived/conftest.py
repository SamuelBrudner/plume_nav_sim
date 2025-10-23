"""Configuration for archived tests.

These tests exercise implementation details that may no longer match current APIs.
They are skipped by default to avoid cluttering the test suite during active development.
"""

import pytest

# Skip all tests in this directory by default
pytestmark = pytest.mark.skip(reason="Archived tests - implementation details may be outdated")
