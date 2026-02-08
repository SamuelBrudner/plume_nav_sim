"""Legacy ActionProcessor shim.

This module is kept for backwards compatibility with early versions of the
project (and regression tests) that referenced :mod:`plume_nav_sim.core.action_processor`.

The modern codebase models actions via action-processor components under
``plume_nav_sim.actions``. This shim provides a minimal, self-contained grid
action processor with a stable return type for simple movement semantics.
"""

from __future__ import annotations

from dataclasses import dataclass

from .enums import Action
from .geometry import Coordinates, GridSize


@dataclass(frozen=True)
class ProcessResult:
    action_valid: bool
    final_position: Coordinates
    movement_delta: tuple[int, int]
    boundary_hit: bool


class ActionProcessor:
    """Process simple grid actions with boundary clamping."""

    def __init__(self, grid: GridSize) -> None:
        self._grid = grid

    def process_action(self, action: Action, position: Coordinates) -> ProcessResult:
        try:
            act = Action(int(action))
        except Exception:
            return ProcessResult(
                action_valid=False,
                final_position=position,
                movement_delta=(0, 0),
                boundary_hit=False,
            )

        dx, dy = act.to_vector()
        new_x = int(position.x) + int(dx)
        new_y = int(position.y) + int(dy)

        clamped_x = max(0, min(new_x, int(self._grid.width) - 1))
        clamped_y = max(0, min(new_y, int(self._grid.height) - 1))
        boundary_hit = (clamped_x != new_x) or (clamped_y != new_y)

        final = Coordinates(clamped_x, clamped_y)
        movement = (final.x - int(position.x), final.y - int(position.y))
        return ProcessResult(
            action_valid=True,
            final_position=final,
            movement_delta=movement,
            boundary_hit=boundary_hit,
        )


__all__ = ["ActionProcessor", "ProcessResult"]

