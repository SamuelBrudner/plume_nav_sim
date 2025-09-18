"""Regression tests for the :mod:`plume_nav_sim.core.action_processor` module.

These tests focus on the observable behaviour of :class:`ActionProcessor` when it
receives a valid action. They intentionally avoid asserting against the internal
implementation so the suite remains resilient to refactors while still verifying
that valid inputs produce the expected movement and metadata.
"""

from __future__ import annotations

import importlib
from typing import Any

import pytest

from plume_nav_sim.core.enums import Action
from plume_nav_sim.core.geometry import Coordinates, GridSize


MODULE_NAME = "plume_nav_sim.core.action_processor"


def _load_action_processor_module() -> Any:
    """Import the action processor module and fail the test if it cannot be loaded."""

    try:
        return importlib.import_module(MODULE_NAME)
    except Exception as exc:  # pragma: no cover - exercised by failing import
        pytest.fail(
            f"Importing {MODULE_NAME} should succeed for the regression tests, "
            f"but raised {exc.__class__.__name__}: {exc}"
        )


def _create_processor(module: Any) -> Any:
    """Create an ``ActionProcessor`` for a simple grid and fail loudly on error."""

    grid = GridSize(5, 5)
    try:
        return module.ActionProcessor(grid)
    except Exception as exc:  # pragma: no cover - exercised when initialisation breaks
        pytest.fail(
            "ActionProcessor should initialise for a valid grid size, but raised "
            f"{exc.__class__.__name__}: {exc}"
        )


def _process(module: Any, processor: Any, action: Action, position: Coordinates) -> Any:
    """Run ``process_action`` while converting unexpected exceptions to test failures."""

    try:
        return processor.process_action(action, position)
    except Exception as exc:  # pragma: no cover - exercised when processing fails
        pytest.fail(
            "process_action should handle valid inputs without raising, but raised "
            f"{exc.__class__.__name__}: {exc}"
        )


def test_process_action_moves_right_without_boundary_hit() -> None:
    """A valid RIGHT action should move the agent one cell to the east."""

    module = _load_action_processor_module()
    processor = _create_processor(module)

    start = Coordinates(1, 1)
    result = _process(module, processor, Action.RIGHT, start)

    assert result.action_valid is True
    assert result.final_position == Coordinates(2, 1)
    assert result.movement_delta == (1, 0)
    assert result.boundary_hit is False
