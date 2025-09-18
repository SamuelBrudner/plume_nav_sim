"""Lightweight namespace package for the trimmed plume_nav_sim test suite."""

from __future__ import annotations

import pytest

__all__ = ["require_full_test_suite"]


def require_full_test_suite() -> None:
    """Signal that the expansive upstream test helpers are not bundled."""

    pytest.skip(
        "The full plume_nav_sim test harness is not included in this kata-oriented repository."
    )
