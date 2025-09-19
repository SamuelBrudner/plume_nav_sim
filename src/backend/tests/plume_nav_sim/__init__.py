"""Lightweight namespace package for the trimmed plume_nav_sim test suite."""

from __future__ import annotations

import logging

__all__ = ["require_full_test_suite"]

_LOGGER = logging.getLogger(__name__)


def require_full_test_suite() -> None:
    """Raise an explicit error indicating the trimmed test harness."""

    message = (
        "The full plume_nav_sim test suite is not included in this kata repository; "
        "install the upstream test harness to exercise these helpers."
    )
    _LOGGER.error(message)
    raise RuntimeError(message)
