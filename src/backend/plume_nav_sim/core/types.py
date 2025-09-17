
"""
Lightweight core types for tests and minimal environment wiring.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict, Any
import numpy as np

from .enums import Action, RenderMode
from .geometry import Coordinates, GridSize
from .typing import RGBArray

# Factories expected by tests

def create_coordinates(x: int, y: int) -> Coordinates:
    return Coordinates(x, y)

def create_grid_size(width: int, height: int) -> GridSize:
    return GridSize(width, height)

# Minimal EnvironmentConfig with validation used by BaseEnvironment
@dataclass
class EnvironmentConfig:
    grid_size: GridSize
    source_location: Coordinates
    max_steps: int
    goal_radius: float

    def validate(self) -> bool:
        if not isinstance(self.grid_size, GridSize):
            raise ValueError('grid_size must be GridSize')
        if not isinstance(self.source_location, Coordinates):
            raise ValueError('source_location must be Coordinates')
        if not isinstance(self.max_steps, int) or self.max_steps <= 0:
            raise ValueError('max_steps must be positive int')
        if not isinstance(self.goal_radius, (int, float)) or self.goal_radius < 0:
            raise ValueError('goal_radius must be non-negative number')
        return True

    def estimate_resources(self) -> Dict[str, Any]:
        # Rough memory estimate for a single float32 field of size grid
        total_cells = self.grid_size.width * self.grid_size.height
        memory_mb = (total_cells * np.dtype(np.float32).itemsize) / (1024*1024)
        return {'memory_mb': memory_mb}
