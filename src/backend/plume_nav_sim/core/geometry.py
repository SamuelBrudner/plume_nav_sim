"""
Core geometry types for the plume navigation simulation.
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from .constants import FIELD_DTYPE, MEMORY_LIMIT_PLUME_FIELD_MB
from .enums import Action


@dataclass(frozen=True)
class Coordinates:
    """Immutable data class for 2D grid coordinates."""

    x: int
    y: int

    def __post_init__(self):
        from ..utils.exceptions import ValidationError

        if not isinstance(self.x, int) or not isinstance(self.y, int):
            raise ValidationError(
                f"Coordinates must be integers, got x={type(self.x).__name__}, y={type(self.y).__name__}"
            )
        # Note: Negative coordinates are allowed per contract (core_types.md)
        # They represent off-grid positions, useful for boundary logic

    def distance_to(self, other: "Coordinates", high_precision: bool = False) -> float:
        """Calculate Euclidean distance to another coordinate."""
        from ..utils.exceptions import ValidationError

        if not isinstance(other, Coordinates):
            raise ValidationError(
                f"Distance calculation requires Coordinates instance, got {type(other).__name__}"
            )
        return calculate_euclidean_distance(self, other, high_precision=high_precision)

    def move(
        self,
        movement: Union[Action, Tuple[int, int]],
        bounds: Optional["GridSize"] = None,
    ) -> "Coordinates":
        """Create new Coordinates by applying a movement."""
        from ..utils.exceptions import ValidationError

        if isinstance(movement, Action):
            dx, dy = movement.to_vector()
        elif isinstance(movement, tuple) and len(movement) == 2:
            dx, dy = movement
        else:
            raise ValidationError(
                f"Movement must be Action or tuple[int, int], got {type(movement).__name__}"
            )

        new_x = self.x + dx
        new_y = self.y + dy

        # Clamp only when explicit bounds are provided. Without bounds, negative
        # coordinates are allowed (per contract) to represent off-grid positions.
        if bounds is not None:
            new_x = max(0, min(new_x, bounds.width - 1))
            new_y = max(0, min(new_y, bounds.height - 1))

        return Coordinates(new_x, new_y)

    def is_within_bounds(self, grid_bounds: "GridSize") -> bool:
        """Check if coordinates are within grid boundaries."""
        return 0 <= self.x < grid_bounds.width and 0 <= self.y < grid_bounds.height

    def manhattan_distance_to(self, other: "Coordinates") -> int:
        """Calculate Manhattan distance to another coordinate."""
        return abs(self.x - other.x) + abs(self.y - other.y)

    def to_tuple(self) -> Tuple[int, int]:
        """Convert coordinates to a tuple."""
        return (self.x, self.y)

    def clone(self) -> "Coordinates":
        """Return self, as Coordinates are immutable."""
        return self

    def to_array_index(self, grid_bounds: "GridSize") -> Tuple[int, int]:
        """Return (row, col) indices for NumPy arrays.

        NumPy arrays are indexed as [row, col] which corresponds to (y, x).
        This helper converts Coordinates (x, y) to (y, x) and validates bounds.
        """
        if not self.is_within_bounds(grid_bounds):
            from ..utils.exceptions import ValidationError

            raise ValidationError(
                f"Coordinate ({self.x}, {self.y}) outside grid bounds {grid_bounds.width}x{grid_bounds.height}"
            )
        return (self.y, self.x)


@dataclass(frozen=True)
class GridSize:
    """Immutable data class for 2D grid dimensions."""

    width: int
    height: int

    def __post_init__(self):
        from ..utils.exceptions import ValidationError

        if not isinstance(self.width, int) or not isinstance(self.height, int):
            raise ValidationError(
                f"Invalid grid_size: dimensions must be integers, got width={type(self.width).__name__}, height={type(self.height).__name__}"
            )
        if self.width <= 0 or self.height <= 0:
            # Include keywords expected by tests: 'negative' or 'invalid'
            raise ValidationError(
                f"Invalid grid_size: dimensions must be positive; negative or zero values are invalid: got ({self.width}, {self.height})"
            )
        max_dimension = 1024
        if self.width > max_dimension or self.height > max_dimension:
            raise ValidationError(
                f"Grid dimensions exceed maximum size {max_dimension}"
            )

    def total_cells(self) -> int:
        """Calculate the total number of cells in the grid."""
        return self.width * self.height

    def center(self) -> Coordinates:
        """Calculate the center coordinates of the grid.

        Contract: core_types.md - GridSize.center()
        Returns: Coordinates at (width//2, height//2)
        """
        return Coordinates(self.width // 2, self.height // 2)

    def contains(self, coord: Coordinates) -> bool:
        """Check if coordinate is within grid bounds.

        Contract: core_types.md - GridSize.contains()
        Returns: True if 0 <= coord.x < width AND 0 <= coord.y < height
        """
        return (0 <= coord.x < self.width) and (0 <= coord.y < self.height)

    def estimate_memory_mb(self, field_dtype: Optional[np.dtype] = None) -> float:
        """Estimate memory usage for the plume field."""
        dtype = field_dtype if field_dtype is not None else FIELD_DTYPE
        bytes_per_cell = np.dtype(dtype).itemsize
        total_bytes = self.total_cells() * bytes_per_cell
        return total_bytes / (1024 * 1024)

    def contains_coordinates(self, coordinates: Coordinates) -> bool:
        """Check if coordinates are within the grid."""
        return coordinates.is_within_bounds(self)

    def is_performance_feasible(
        self, performance_targets: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Check if the grid size is within performance targets."""
        memory_estimate = self.estimate_memory_mb()
        if memory_estimate > MEMORY_LIMIT_PLUME_FIELD_MB:
            return False
        if performance_targets:
            if (
                "max_total_cells" in performance_targets
                and self.total_cells() > performance_targets["max_total_cells"]
            ):
                return False
            if (
                "max_memory_mb" in performance_targets
                and memory_estimate > performance_targets["max_memory_mb"]
            ):
                return False
        return True

    def to_tuple(self) -> Tuple[int, int]:
        """Convert grid size to a tuple."""
        return (self.width, self.height)


def calculate_euclidean_distance(
    coord1: Coordinates, coord2: Coordinates, high_precision: bool = False
) -> float:
    """Utility function to calculate Euclidean distance."""
    dx = coord1.x - coord2.x
    dy = coord1.y - coord2.y
    if high_precision:
        return math.sqrt(float(dx**2) + float(dy**2))
    return math.sqrt(dx**2 + dy**2)
