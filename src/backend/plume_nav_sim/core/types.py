"""Core types for plume_nav_sim."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Any, Sequence

import numpy as np

try:  # pragma: no cover - numpy<1.20 compatibility
    from numpy.typing import NDArray
except ImportError:  # pragma: no cover
    NDArray = np.ndarray  # type: ignore[assignment]

DEFAULT_GRID_SIZE = (128, 128)
DEFAULT_SOURCE_LOCATION = (64, 64)
DEFAULT_MAX_STEPS = 1000
DEFAULT_GOAL_RADIUS = float(np.finfo(np.float32).eps)
DEFAULT_PLUME_SIGMA = 12.0

ACTION_UP, ACTION_RIGHT, ACTION_DOWN, ACTION_LEFT = 0, 1, 2, 3
ACTION_SPACE_SIZE = 4
MOVEMENT_VECTORS = {
    ACTION_UP: (0, 1),
    ACTION_RIGHT: (1, 0),
    ACTION_DOWN: (0, -1),
    ACTION_LEFT: (-1, 0),
}


class Action(IntEnum):
    """Discrete movement actions for grid navigation."""
    UP = ACTION_UP
    RIGHT = ACTION_RIGHT
    DOWN = ACTION_DOWN
    LEFT = ACTION_LEFT

    def to_vector(self) -> tuple[int, int]:
        return MOVEMENT_VECTORS[int(self)]


class RenderMode(Enum):
    """Rendering output modes."""
    RGB_ARRAY = "rgb_array"
    HUMAN = "human"

    def is_programmatic(self) -> bool:
        return self == RenderMode.RGB_ARRAY

    def requires_display(self) -> bool:
        return self == RenderMode.HUMAN

    def get_output_format(self) -> str:
        if self == RenderMode.RGB_ARRAY:
            return "np.ndarray[H,W,3] uint8"
        return "Interactive matplotlib window (returns None)"


@dataclass(frozen=True)
class Coordinates:
    """2D integer coordinates on the grid."""
    x: int
    y: int

    def to_tuple(self) -> tuple[int, int]:
        return (int(self.x), int(self.y))

    def is_within_bounds(self, grid_bounds: "GridSize") -> bool:
        return 0 <= self.x < grid_bounds.width and 0 <= self.y < grid_bounds.height

    def distance_to(self, other: "Coordinates") -> float:
        return calculate_euclidean_distance(self, other)

    def move(
        self,
        movement: Action | tuple[int, int],
        bounds: "GridSize | None" = None,
    ) -> "Coordinates":
        if isinstance(movement, Action):
            dx, dy = movement.to_vector()
        else:
            dx, dy = movement
        new_x = self.x + int(dx)
        new_y = self.y + int(dy)
        if bounds is not None:
            new_x = max(0, min(new_x, bounds.width - 1))
            new_y = max(0, min(new_y, bounds.height - 1))
        return Coordinates(new_x, new_y)

    def to_array_index(self, grid_bounds: "GridSize") -> tuple[int, int]:
        return (self.y, self.x)


@dataclass(frozen=True)
class GridSize:
    """Grid dimensions for the environment."""

    width: int
    height: int

    def to_tuple(self) -> tuple[int, int]:
        return (int(self.width), int(self.height))

    def total_cells(self) -> int:
        return int(self.width) * int(self.height)

    def center(self) -> Coordinates:
        return Coordinates(self.width // 2, self.height // 2)

    def contains(self, coord: Coordinates) -> bool:
        return 0 <= coord.x < self.width and 0 <= coord.y < self.height

    def estimate_memory_mb(self, field_dtype: np.dtype | None = None) -> float:
        dtype = field_dtype if field_dtype is not None else np.float32
        bytes_per_cell = np.dtype(dtype).itemsize
        return (self.total_cells() * bytes_per_cell) / (1024 * 1024)


@dataclass
class AgentState:
    """Mutable agent state during episode."""

    position: Coordinates
    orientation: float = 0.0
    step_count: int = 0
    total_reward: float = 0.0
    goal_reached: bool = False

    def update_position(self, new_position: Coordinates) -> None:
        self.position = new_position

    def add_reward(self, reward: float) -> None:
        self.total_reward += float(reward)
        if reward > 0:
            self.goal_reached = True

    def increment_step(self) -> None:
        self.step_count += 1

    def mark_goal_reached(self) -> None:
        self.goal_reached = True


CoordinateType = Coordinates | tuple[int, int] | Sequence[int]
GridDimensions = GridSize | tuple[int, int] | Sequence[int]
MovementVector = tuple[int, int]
ActionType = Action | int | np.ndarray | NDArray[Any]
Observation = NDArray[np.floating] | dict[str, Any] | tuple[Any, ...]
Info = dict[str, Any]
ObservationType = Observation
InfoType = Info

if NDArray is not None:
    RGBArray = NDArray[np.uint8]
else:  # pragma: no cover - numpy<1.20
    RGBArray = np.ndarray  # type: ignore[assignment]


def calculate_euclidean_distance(coord1: Coordinates, coord2: Coordinates) -> float:
    """Calculate Euclidean distance between two coordinates."""
    return math.hypot(coord1.x - coord2.x, coord1.y - coord2.y)


def create_coordinates(value: CoordinateType, y: int | None = None) -> Coordinates:
    """Create Coordinates from various input types."""
    if y is not None:
        return Coordinates(int(value), int(y))  # type: ignore[arg-type]
    if isinstance(value, Coordinates):
        return value
    if isinstance(value, Sequence) and len(value) == 2:
        return Coordinates(int(value[0]), int(value[1]))
    raise TypeError("Coordinates must be Coordinates or length-2 sequence")


def create_grid_size(value: GridDimensions, height: int | None = None) -> GridSize:
    """Create GridSize from various input types."""
    if height is not None:
        return GridSize(int(value), int(height))  # type: ignore[arg-type]
    if isinstance(value, GridSize):
        return value
    if isinstance(value, Sequence) and len(value) == 2:
        return GridSize(int(value[0]), int(value[1]))
    raise TypeError("GridSize must be GridSize or length-2 sequence")


def validate_action(action: ActionType) -> Action:
    """Validate and convert action input to Action enum."""
    if isinstance(action, Action):
        return action
    if isinstance(action, (np.integer, int)) and int(action) in MOVEMENT_VECTORS:
        return Action(int(action))
    raise TypeError("Action must be Action enum or integer in action space")


def get_movement_vector(action: ActionType) -> MovementVector:
    """Get movement vector for an action."""
    return validate_action(action).to_vector()


__all__ = [
    "Action",
    "ActionType",
    "AgentState",
    "CoordinateType",
    "Coordinates",
    "GridDimensions",
    "GridSize",
    "Info",
    "InfoType",
    "MovementVector",
    "Observation",
    "ObservationType",
    "RGBArray",
    "RenderMode",
    "calculate_euclidean_distance",
    "create_coordinates",
    "create_grid_size",
    "get_movement_vector",
    "validate_action",
]
