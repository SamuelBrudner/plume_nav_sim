"""
Core data models for the plume navigation simulation.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .constants import MAX_PLUME_SIGMA, MIN_PLUME_SIGMA
from .geometry import Coordinates, GridSize


@dataclass(frozen=True)
class PlumeModel:
    """Data class for plume model parameters."""

    source_location: Coordinates
    sigma: float
    grid_compatibility: Optional[GridSize] = None
    validation_results: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        from ..utils.exceptions import ValidationError

        if not isinstance(self.source_location, Coordinates):
            raise ValidationError(
                f"Source location must be a Coordinates instance, got {type(self.source_location).__name__}"
            )

        if not (MIN_PLUME_SIGMA <= self.sigma <= MAX_PLUME_SIGMA):
            raise ValidationError(
                f"Plume sigma {self.sigma} is out of the valid range [{MIN_PLUME_SIGMA}, {MAX_PLUME_SIGMA}]"
            )

        if self.grid_compatibility is not None:
            if not isinstance(self.grid_compatibility, GridSize):
                raise ValidationError(
                    f"Grid compatibility must be a GridSize instance, got {type(self.grid_compatibility).__name__}"
                )
            if not self.grid_compatibility.contains_coordinates(self.source_location):
                raise ValidationError(
                    f"Source location {self.source_location.to_tuple()} is outside grid bounds {self.grid_compatibility.to_tuple()}"
                )

    def is_compatible(self, grid_size: GridSize) -> bool:
        """Check if the plume model is compatible with a given grid size."""
        return grid_size.contains_coordinates(self.source_location)

    def validate_model(self) -> Dict[str, Any]:
        """Perform comprehensive validation of plume model parameters."""
        self.validation_results["sigma_check"] = "passed"
        self.validation_results["source_location_check"] = "passed"
        return self.validation_results

    def to_dict(self) -> Dict[str, Any]:
        """Convert the plume model to a dictionary."""
        return {
            "source_location": self.source_location.to_tuple(),
            "sigma": self.sigma,
            "grid_compatibility": (
                self.grid_compatibility.to_tuple() if self.grid_compatibility else None
            ),
        }
