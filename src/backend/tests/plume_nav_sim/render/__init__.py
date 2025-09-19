"""Render-specific tests namespace stub for the trimmed test environment."""

from __future__ import annotations

import logging

__all__: list[str] = []

_LOGGER = logging.getLogger(__name__)


def __getattr__(name: str):
    message = (
        f"Render test helper '{name}' is not included in the trimmed plume_nav_sim test suite. "
        "Install the full test harness to access render-specific fixtures."
    )
    _LOGGER.error(message)
    raise AttributeError(message)
