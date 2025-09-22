"""
Core enumerations for the plume navigation simulation.
"""

from enum import Enum, IntEnum
from typing import Tuple

from .constants import (
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_UP,
    MOVEMENT_VECTORS,
)


class Action(IntEnum):
    """Enumeration for discrete agent actions."""

    UP = ACTION_UP
    RIGHT = ACTION_RIGHT
    DOWN = ACTION_DOWN
    LEFT = ACTION_LEFT

    def to_vector(self) -> Tuple[int, int]:
        """Convert action to a movement vector."""
        try:
            return MOVEMENT_VECTORS[self.value]
        except KeyError:
            from ..utils.exceptions import ValidationError

            raise ValidationError(f"Invalid action value {self.value}")

    def opposite(self) -> "Action":
        """Get the opposite action."""
        opposite_map = {
            Action.UP: Action.DOWN,
            Action.DOWN: Action.UP,
            Action.RIGHT: Action.LEFT,
            Action.LEFT: Action.RIGHT,
        }
        return opposite_map[self]

    def is_horizontal(self) -> bool:
        """Check if the action is horizontal."""
        return self in (Action.LEFT, Action.RIGHT)

    def is_vertical(self) -> bool:
        """Check if the action is vertical."""
        return self in (Action.UP, Action.DOWN)


class RenderMode(Enum):
    """Enumeration for visualization modes."""

    RGB_ARRAY = "rgb_array"
    HUMAN = "human"

    def is_programmatic(self) -> bool:
        """Check if the render mode is programmatic."""
        return self == RenderMode.RGB_ARRAY

    def requires_display(self) -> bool:
        """Check if the render mode requires a display."""
        return self == RenderMode.HUMAN

    def get_output_format(self) -> str:
        """Get the expected output format."""
        if self == RenderMode.RGB_ARRAY:
            return "np.ndarray[H,W,3] uint8"
        return "Interactive matplotlib window (returns None)"
