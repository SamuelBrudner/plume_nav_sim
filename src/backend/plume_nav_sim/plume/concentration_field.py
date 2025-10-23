"""
Core concentration field data structure for plume_nav_sim package providing efficient 2D concentration
field management, mathematical operations, position-based sampling, and integration with plume models.
Serves as the foundation for all plume concentration calculations with optimized NumPy operations,
caching support, and comprehensive validation for environment observation generation and rendering
pipeline integration.

This module implements:
- ConcentrationField class for 2D concentration field management with efficient sampling and caching
- Factory functions for field creation with parameter validation and memory estimation
- Mathematical operations for Gaussian concentration formula calculations and field normalization
- Performance-optimized field generation with <10ms generation time for 128×128 grids
- Component interface integration for plume model and rendering pipeline data access
- Specialized exception classes for field generation and sampling error handling
"""

import copy  # >=3.10 - Deep copying of concentration field arrays for safe field manipulation and caching
import functools  # >=3.10 - Caching decorators for performance optimization of field generation and validation operations
import warnings  # >=3.10 - Warning management for performance issues, memory concerns, and deprecated field operations
from typing import (  # >=3.10 - Type hints for method parameters, return types, and optional parameter specifications
    Any,
    Dict,
    List,
    Optional,
)

# External imports with version comments
import numpy as np  # >=2.1.0 - Core mathematical operations, array management, meshgrid generation, and efficient vectorized calculations

from ..core.constants import (
    CONCENTRATION_RANGE,
    DEFAULT_GRID_SIZE,
    DEFAULT_PLUME_SIGMA,
    FIELD_DTYPE,
    GAUSSIAN_PRECISION,
    MEMORY_LIMIT_PLUME_FIELD_MB,
    PERFORMANCE_TARGET_PLUME_GENERATION_MS,
)

# Internal imports from core types and constants modules
from ..core.types import Coordinates, CoordinateType, GridDimensions, GridSize
from ..utils.exceptions import ComponentError, ResourceError

# Internal imports from logging utilities for component-specific monitoring
from ..utils.logging import ComponentLogger, PerformanceTimer, get_component_logger

# Internal imports from validation and exception handling modules
from ..utils.validation import ValidationError, validate_coordinates, validate_grid_size

# Global constants for field caching and interpolation configuration
_FIELD_CACHE_SIZE = 100
_INTERPOLATION_METHODS = ["nearest", "linear", "cubic"]
_DEFAULT_INTERPOLATION = "nearest"
_FIELD_MEMORY_WARNING_THRESHOLD = 64
_GENERATION_PERFORMANCE_CACHE = {}

# Module exports for concentration field interface
__all__ = [
    "ConcentrationField",
    "create_concentration_field",
    "validate_field_parameters",
    "estimate_field_memory",
    "clear_field_cache",
    "FieldGenerationError",
    "FieldSamplingError",
]


class FieldGenerationError(ComponentError):
    """
    Specialized exception class for concentration field generation failures including mathematical
    errors, memory constraints, and parameter validation issues with detailed error context and
    recovery suggestions for development debugging and automated error handling.

    This exception provides detailed context for field generation failures including grid parameters,
    sigma values, memory constraints, and specific recovery actions for resolving generation issues.
    """

    def __init__(
        self,
        message: str,
        grid_size: Optional[GridSize] = None,
        generation_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize field generation error with context information and parameter details for
        detailed error reporting and debugging support with component-specific recovery guidance.

        Args:
            message: Primary error description for field generation failure
            grid_size: Grid dimensions that caused or were involved in the generation failure
            generation_params: Parameters used during failed field generation for debugging context
        """
        # Call parent ComponentError constructor with HIGH severity for generation failures
        super().__init__(
            message=message,
            component_name="concentration_field",
            operation_name="generate_field",
            underlying_error=None,
        )

        # Store grid_size for memory and dimension analysis in recovery suggestions
        self.grid_size = grid_size
        # Store generation_params for parameter debugging and analysis
        self.generation_params = generation_params or {}
        # Initialize error_context with field generation specific information
        self.error_context = {
            "operation_type": "field_generation",
            # Represent grid size as a simple tuple to avoid relying on helper methods
            # not guaranteed on the GridSize dataclass in this trimmed build.
            "grid_size": (grid_size.width, grid_size.height) if grid_size else None,
            "generation_params": self.generation_params,
        }

        # Set component-specific recovery suggestions based on error context
        recovery_suggestion = self._generate_recovery_suggestion()
        self.set_recovery_suggestion(recovery_suggestion)

    def _generate_recovery_suggestion(self) -> str:
        """Generate specific recovery suggestion based on grid size and generation parameters."""
        if self.grid_size:
            # Check for memory-related issues based on grid size
            estimated_memory = self.grid_size.estimate_memory_mb()
            if estimated_memory > MEMORY_LIMIT_PLUME_FIELD_MB:
                return f"Reduce grid size from {self.grid_size.width}×{self.grid_size.height} to stay within {MEMORY_LIMIT_PLUME_FIELD_MB}MB memory limit"

        # Check generation parameters for common issues
        if "sigma" in self.generation_params:
            sigma = self.generation_params["sigma"]
            if sigma <= 0:
                return (
                    "Ensure sigma parameter is positive for valid Gaussian calculations"
                )
            elif sigma < GAUSSIAN_PRECISION:
                return (
                    f"Increase sigma above {GAUSSIAN_PRECISION} for numerical stability"
                )

        # Default recovery suggestion for field generation failures
        return "Check grid size and sigma parameters, reduce memory usage, or clear field cache"

    def get_recovery_suggestions(self) -> List[str]:
        """
        Generate specific recovery suggestions based on field generation error context and
        parameters for automated error handling and user guidance.

        Returns:
            List of prioritized recovery actions for resolving field generation issues
        """
        suggestions = []

        # Analyze grid_size for memory-related recovery suggestions
        if self.grid_size:
            memory_mb = self.grid_size.estimate_memory_mb()
            if memory_mb > MEMORY_LIMIT_PLUME_FIELD_MB:
                suggestions.append(
                    f"Reduce grid size to stay within {MEMORY_LIMIT_PLUME_FIELD_MB}MB limit"
                )
                # Suggest specific reduced dimensions
                scale_factor = 0.8
                new_width = int(self.grid_size.width * scale_factor)
                new_height = int(self.grid_size.height * scale_factor)
                suggestions.append(f"Try reduced dimensions: {new_width}×{new_height}")

        # Check generation_params for parameter adjustment recommendations
        if self.generation_params and "sigma" in self.generation_params:
            sigma = self.generation_params["sigma"]
            if sigma <= 0:
                suggestions.append(
                    "Set sigma to positive value (recommended: 8.0-16.0)"
                )
            elif sigma > 50:
                suggestions.append("Reduce sigma for better gradient definition")

        # General recovery actions
        suggestions.extend(
            [
                "Clear field cache to free memory",
                "Use smaller grid size for initial testing",
                "Check available system memory",
            ]
        )

        return suggestions[:5]  # Return top 5 suggestions


class FieldSamplingError(ComponentError):
    """
    Specialized exception class for concentration field sampling failures including bounds violations,
    interpolation errors, and cache issues with position context and sampling parameter analysis for
    debugging position-based sampling operations.

    This exception provides detailed context for sampling failures including position coordinates,
    sampling methods, bounds checking results, and specific recovery actions for position errors.
    """

    def __init__(
        self,
        message: str,
        position: Optional[CoordinateType] = None,
        sampling_method: Optional[str] = None,
    ):
        """
        Initialize field sampling error with position and method information for detailed
        sampling error analysis and position-specific debugging support.

        Args:
            message: Primary error description for field sampling failure
            position: Position coordinates that caused sampling failure for bounds analysis
            sampling_method: Sampling method that was used during failed operation
        """
        # Call parent ComponentError constructor with MEDIUM severity for sampling failures
        super().__init__(
            message=message,
            component_name="concentration_field",
            operation_name="sample_at",
            underlying_error=None,
        )

        # Store position for bounds checking and coordinate analysis
        self.position = position
        # Store sampling_method for method-specific error handling
        self.sampling_method = sampling_method or "nearest"
        # Initialize sampling_context with position and method details
        self.sampling_context = {
            "operation_type": "field_sampling",
            "position": self._serialize_position(position),
            "sampling_method": self.sampling_method,
            "interpolation_available": sampling_method in _INTERPOLATION_METHODS,
        }

        # Set recovery suggestions for sampling parameter corrections
        recovery_suggestion = self._generate_sampling_recovery_suggestion()
        self.set_recovery_suggestion(recovery_suggestion)

    def _serialize_position(
        self, position: Optional[CoordinateType]
    ) -> Optional[Dict[str, Any]]:
        """Serialize position for safe storage in error context."""
        if position is None:
            return None

        if isinstance(position, Coordinates):
            return {"x": position.x, "y": position.y, "type": "Coordinates"}
        elif isinstance(position, (tuple, list)) and len(position) == 2:
            return {"x": position[0], "y": position[1], "type": "tuple"}
        else:
            return {"position": str(position), "type": type(position).__name__}

    def _generate_sampling_recovery_suggestion(self) -> str:
        """Generate recovery suggestion based on sampling error context."""
        if self.position and (
            isinstance(self.position, (tuple, list)) and len(self.position) == 2
        ):
            x, y = self.position
            # Note: Negative coordinates allowed per contract, but may indicate issues
            if x > 1000 or y > 1000:  # Reasonable upper bound check
                return "Check position coordinates are within reasonable grid bounds"
            elif x < -100 or y < -100:  # Very negative might indicate error
                return "Position appears far outside typical grid bounds"

        # Check sampling method issues
        if self.sampling_method not in _INTERPOLATION_METHODS:
            return f"Use valid sampling method: {', '.join(_INTERPOLATION_METHODS)}"

        return "Validate position coordinates are within grid bounds before sampling"

    def get_position_analysis(self) -> Dict[str, Any]:
        """
        Analyze position parameter for bounds violations and coordinate validity for detailed
        position error debugging and coordinate system validation.

        Returns:
            Dictionary containing position analysis with bounds checking and coordinate validation
        """
        analysis = {
            "position_provided": self.position is not None,
            "position_type": type(self.position).__name__ if self.position else None,
            "sampling_method": self.sampling_method,
            "analysis_timestamp": np.float64(
                functools.lru_cache.cache_info
            ),  # Mock timestamp
        }

        # Analyze position coordinates for bounds violations and validity
        if self.position:
            if isinstance(self.position, Coordinates):
                analysis.update(
                    {
                        "coordinates": {"x": self.position.x, "y": self.position.y},
                        "coordinate_format": "Coordinates_object",
                        "bounds_checkable": True,
                    }
                )
            elif isinstance(self.position, (tuple, list)) and len(self.position) == 2:
                x, y = self.position
                analysis.update(
                    {
                        "coordinates": {"x": x, "y": y},
                        "coordinate_format": "tuple_list",
                        "bounds_checkable": True,
                        "negative_coordinates": x < 0 or y < 0,
                        "large_coordinates": x > 1000 or y > 1000,
                    }
                )
            else:
                analysis.update(
                    {
                        "coordinate_format": "invalid",
                        "bounds_checkable": False,
                        "format_error": "Position must be Coordinates object or (x, y) tuple",
                    }
                )

        # Check coordinate format and type consistency
        analysis["sampling_method_valid"] = (
            self.sampling_method in _INTERPOLATION_METHODS
        )

        # Generate position correction recommendations
        recommendations = []
        if not analysis.get("bounds_checkable", False):
            recommendations.append(
                "Convert position to Coordinates object or (x, y) tuple"
            )
        if analysis.get("negative_coordinates", False):
            recommendations.append("Ensure coordinates are non-negative")
        if not analysis.get("sampling_method_valid", False):
            recommendations.append(
                f"Use valid sampling method: {_INTERPOLATION_METHODS[0]}"
            )

        analysis["recommendations"] = recommendations

        return analysis


class ConcentrationField:
    """
    Core concentration field data structure providing efficient 2D concentration field management,
    mathematical operations, position-based sampling, and caching support for plume navigation
    simulations with optimized NumPy operations and comprehensive validation.

    This class implements efficient concentration field operations including:
    - Lazy field generation with Gaussian mathematical formula calculations
    - O(1) position-based sampling with interpolation support and bounds checking
    - Memory-efficient caching with configurable cache size and performance monitoring
    - Field validation with mathematical property checking and integrity verification
    - Performance optimization with vectorized NumPy operations and timing analysis
    """

    def __init__(self, grid_size: GridSize, enable_caching: bool = True):
        """
        Initialize concentration field with grid dimensions, caching configuration, and performance
        monitoring setup for efficient field operations and memory management with lazy generation.

        Args:
            grid_size: Grid dimensions using GridSize dataclass for field boundaries
            enable_caching: Enable performance optimization through result caching
        """
        # Store grid_size with validation using validate_grid_size function
        try:
            validate_grid_size(
                grid_size, check_memory_limits=True, validate_performance=True
            )
            self.grid_size = grid_size
        except ValidationError as e:
            raise FieldGenerationError(
                f"Invalid grid size for concentration field: {e}",
                grid_size=grid_size,
            ) from e

        # Initialize enable_caching flag for performance optimization control
        self.enable_caching = enable_caching

        # Set field_array to None for lazy field generation
        self.field_array: Optional[np.ndarray] = None

        # Initialize is_generated flag to False indicating field needs generation
        self.is_generated = False

        # Create component logger using get_component_logger for concentration field operations
        from ..utils.logging import ComponentType

        self.logger: ComponentLogger = get_component_logger(
            component_name="concentration_field",
            component_type=ComponentType.PLUME_MODEL,
            enable_performance_tracking=True,
        )

        # Initialize empty generation_params dictionary for parameter tracking
        self.generation_params: Dict[str, Any] = {}

        # Set last_generation_time_ms to None for performance monitoring
        self.last_generation_time_ms: Optional[float] = None

        # Initialize empty sampling_cache dictionary if caching enabled
        self.sampling_cache: Dict[str, float] = {} if enable_caching else {}

        # Initialize cache statistics counters (cache_hits, cache_misses) to zero
        self.cache_hits = 0
        self.cache_misses = 0

        # Set interpolation_method to default nearest neighbor for performance
        self.interpolation_method = _DEFAULT_INTERPOLATION

    def generate_field(  # noqa: C901
        self,
        source_location: Coordinates,
        sigma: float = DEFAULT_PLUME_SIGMA,
        force_regeneration: bool = False,
        normalize_field: bool = True,
    ) -> np.ndarray:
        """
        Generate 2D concentration field using Gaussian formula with vectorized NumPy operations,
        performance monitoring, and validation for efficient mathematical plume calculations with
        caching support and memory optimization.

        Args:
            source_location: Source position using Coordinates for plume center location
            sigma: Gaussian dispersion parameter for plume spread control
            force_regeneration: Force field regeneration even if already generated
            normalize_field: Normalize field values to [0,1] range for consistency

        Returns:
            Generated 2D concentration field array with proper normalization and validation

        Raises:
            FieldGenerationError: If field generation fails due to parameters or memory constraints
        """
        # Check if field already generated and force_regeneration is False for performance optimization
        if (
            self.is_generated
            and not force_regeneration
            and self.field_array is not None
        ):
            self.logger.debug("Using existing generated field")
            return self.field_array

        try:
            # Validate source_location is within grid bounds using validate_coordinates
            source_location = validate_coordinates(
                coordinates=source_location,
                grid_bounds=self.grid_size,
            )

            # Validate sigma parameter range for mathematical stability and numerical precision
            if sigma <= 0:
                raise ValidationError(f"Sigma must be positive, got {sigma}")
            if sigma < GAUSSIAN_PRECISION:
                warnings.warn(
                    f"Small sigma ({sigma}) may cause numerical precision issues",
                    RuntimeWarning,
                    stacklevel=2,
                )

        except ValidationError as e:
            raise FieldGenerationError(
                f"Field generation validation failed: {e}",
                grid_size=self.grid_size,
                generation_params={
                    "source_location": source_location,
                    "sigma": sigma,
                },
            ) from e

        # Start performance timing using PerformanceTimer for generation monitoring
        with PerformanceTimer(
            "field_generation", logger=self.logger, auto_log=True
        ) as timer:
            try:
                # Create coordinate meshgrids using numpy.meshgrid for vectorized calculations
                x_coords = np.arange(self.grid_size.width, dtype=FIELD_DTYPE)
                y_coords = np.arange(self.grid_size.height, dtype=FIELD_DTYPE)
                x_mesh, y_mesh = np.meshgrid(x_coords, y_coords, indexing="xy")

                # Calculate distance arrays from source location using vectorized operations
                dx = x_mesh - source_location.x
                dy = y_mesh - source_location.y
                distance_squared = dx * dx + dy * dy

                # Add memory usage metric for performance tracking
                array_memory_mb = (
                    distance_squared.nbytes + x_mesh.nbytes + y_mesh.nbytes
                ) / (1024 * 1024)
                timer.add_metric("memory_mb", array_memory_mb)

                # Apply Gaussian formula: exp(-distance²/(2*σ²)) using NumPy vectorized operations
                sigma_squared_2 = 2.0 * float(sigma) * float(sigma)
                # Compute in default numpy precision then cast to FIELD_DTYPE
                field = np.exp(-distance_squared / sigma_squared_2).astype(FIELD_DTYPE)

                # Normalize field to [0,1] range if normalize_field enabled with peak at source
                if normalize_field:
                    field_max = np.max(field)
                    if field_max > GAUSSIAN_PRECISION:
                        field = field / field_max
                        # Ensure source location has concentration of 1.0
                        source_idx_y, source_idx_x = source_location.to_array_index(
                            self.grid_size
                        )
                        field[source_idx_y, source_idx_x] = 1.0

                # Ensure array dtype is FIELD_DTYPE for consistent memory usage and precision
                self.field_array = (
                    field if field.dtype == FIELD_DTYPE else field.astype(FIELD_DTYPE)
                )

                # Store field_array and set is_generated to True with generation parameter tracking
                self.is_generated = True
                self.generation_params = {
                    "source_location": source_location,
                    "sigma": sigma,
                    "normalize_field": normalize_field,
                    "grid_size": self.grid_size,
                    "generation_timestamp": timer.start_time,
                }

                # Log generation timing and validate against PERFORMANCE_TARGET_PLUME_GENERATION_MS
                if timer.duration_ms is not None:
                    self.last_generation_time_ms = timer.duration_ms
                    if (
                        self.last_generation_time_ms
                        > PERFORMANCE_TARGET_PLUME_GENERATION_MS
                    ):
                        self.logger.warning(
                            f"Field generation took {self.last_generation_time_ms:.2f}ms, "
                            f"exceeds target {PERFORMANCE_TARGET_PLUME_GENERATION_MS}ms"
                        )

                # Clear sampling cache since field has changed
                if self.enable_caching:
                    self.sampling_cache.clear()

                # Return generated field array ready for sampling and rendering operations
                return self.field_array

            except Exception as e:
                raise FieldGenerationError(
                    f"Field generation computation failed: {e}",
                    grid_size=self.grid_size,
                    generation_params={
                        "source_location": source_location,
                        "sigma": sigma,
                    },
                ) from e

    def sample_at(  # noqa: C901
        self,
        position: CoordinateType,
        interpolate: bool = False,
        validate_bounds: bool = True,
        use_cache: bool = True,
    ) -> float:
        """
        Sample concentration value at specified position with bounds checking, interpolation support,
        and caching optimization for efficient agent observation generation and position queries.

        Args:
            position: Position coordinates for sampling using CoordinateType union
            interpolate: Enable interpolation for smooth sampling between grid points
            validate_bounds: Enable bounds checking for position validation
            use_cache: Enable result caching for performance optimization

        Returns:
            Concentration value at specified position with proper precision and validation

        Raises:
            FieldSamplingError: If sampling fails due to bounds violation or interpolation error
        """
        try:
            # Ensure field is generated by calling generate_field if field_array is None
            if not self.is_generated or self.field_array is None:
                raise FieldSamplingError(
                    "Cannot sample from ungenerated field. Call generate_field first.",
                    position=position,
                    sampling_method=self.interpolation_method,
                )

            # Convert position to Coordinates object using validate_coordinates if needed
            if isinstance(position, Coordinates):
                coords = position
            elif isinstance(position, (tuple, list)) and len(position) == 2:
                coords = Coordinates(x=int(position[0]), y=int(position[1]))
            else:
                raise FieldSamplingError(
                    f"Invalid position format: {type(position)}. Expected Coordinates or (x, y) tuple.",
                    position=position,
                    sampling_method=self.interpolation_method,
                )

            # Check position bounds if validate_bounds enabled using is_within_bounds method
            if validate_bounds and not coords.is_within_bounds(self.grid_size):
                raise FieldSamplingError(
                    f"Position {coords.x}, {coords.y} is outside grid bounds "
                    f"{self.grid_size.width}×{self.grid_size.height}",
                    position=position,
                    sampling_method=self.interpolation_method,
                )

            # Check sampling cache if use_cache enabled and caching is enabled
            cache_key = f"{coords.x}_{coords.y}_{interpolate}"
            if use_cache and self.enable_caching and cache_key in self.sampling_cache:
                self.cache_hits += 1
                return self.sampling_cache[cache_key]

            # Convert coordinates to array indices using to_array_index method
            try:
                array_y, array_x = coords.to_array_index(self.grid_size)
            except Exception as e:
                raise FieldSamplingError(
                    f"Failed to convert coordinates to array indices: {e}",
                    position=position,
                    sampling_method=self.interpolation_method,
                ) from e

            # Sample concentration value from field_array using array indexing
            if interpolate and self.interpolation_method == "linear":
                # Apply linear interpolation for smoother sampling
                concentration = self._interpolate_bilinear(coords.x, coords.y)
            else:
                # Use nearest neighbor sampling for performance
                concentration = float(self.field_array[array_y, array_x])

            # Validate sampled value is within CONCENTRATION_RANGE for data integrity
            if not (CONCENTRATION_RANGE[0] <= concentration <= CONCENTRATION_RANGE[1]):
                self.logger.warning(
                    f"Sampled concentration {concentration} outside valid range {CONCENTRATION_RANGE}"
                )
                concentration = np.clip(
                    concentration, CONCENTRATION_RANGE[0], CONCENTRATION_RANGE[1]
                )

            # Cache result if use_cache enabled and caching is active
            if use_cache and self.enable_caching:
                # Limit cache size to prevent memory bloat
                if len(self.sampling_cache) >= _FIELD_CACHE_SIZE:
                    # Remove oldest entries (simple FIFO)
                    cache_keys = list(self.sampling_cache.keys())
                    for old_key in cache_keys[: len(cache_keys) // 2]:
                        del self.sampling_cache[old_key]

                self.sampling_cache[cache_key] = concentration
                self.cache_misses += 1
            else:
                self.cache_misses += 1

            # Return concentration value with appropriate precision and validation
            return float(concentration)

        except FieldSamplingError:
            # Re-raise FieldSamplingError without modification
            raise
        except Exception as e:
            raise FieldSamplingError(
                f"Sampling operation failed: {e}",
                position=position,
                sampling_method=self.interpolation_method,
            ) from e

    def _interpolate_bilinear(self, x: float, y: float) -> float:
        """
        Perform bilinear interpolation for smooth concentration sampling between grid points.

        Args:
            x: X coordinate (float) for interpolation
            y: Y coordinate (float) for interpolation

        Returns:
            Interpolated concentration value
        """
        # Get integer coordinates and fractional parts
        x0, x1 = int(np.floor(x)), int(np.ceil(x))
        y0, y1 = int(np.floor(y)), int(np.ceil(y))

        # Clamp to grid boundaries
        x0 = np.clip(x0, 0, self.grid_size.width - 1)
        x1 = np.clip(x1, 0, self.grid_size.width - 1)
        y0 = np.clip(y0, 0, self.grid_size.height - 1)
        y1 = np.clip(y1, 0, self.grid_size.height - 1)

        # Get fractional parts for interpolation weights
        fx = x - x0
        fy = y - y0

        # Sample corner values
        c00 = self.field_array[y0, x0]
        c10 = self.field_array[y0, x1]
        c01 = self.field_array[y1, x0]
        c11 = self.field_array[y1, x1]

        # Perform bilinear interpolation
        c0 = c00 * (1 - fx) + c10 * fx
        c1 = c01 * (1 - fx) + c11 * fx
        concentration = c0 * (1 - fy) + c1 * fy

        return float(concentration)

    # Compatibility layer expected by tests: expose `.field` ndarray and `.sample()` API
    @property
    def field(self) -> np.ndarray:
        """Return the underlying field array (reference) for analysis/tests."""
        return self.get_field_array(copy_array=False, ensure_generated=True)

    def sample(self, position: CoordinateType) -> float:
        """Compatibility wrapper for sample_at(position)."""
        return self.sample_at(
            position, interpolate=False, validate_bounds=True, use_cache=True
        )

    def get_field_array(
        self,
        copy_array: bool = True,
        ensure_generated: bool = True,
        validate_field: bool = False,
    ) -> np.ndarray:
        """
        Return complete concentration field array with optional copying and validation for
        rendering pipeline integration and external processing with memory safety controls.

        Args:
            copy_array: Create deep copy of array for safe external access
            ensure_generated: Generate field if not already generated
            validate_field: Validate field integrity before returning

        Returns:
            Complete concentration field array with proper format and validation

        Raises:
            FieldGenerationError: If field generation is required but fails
        """
        # Generate field if not generated and ensure_generated is True
        if ensure_generated and (not self.is_generated or self.field_array is None):
            raise FieldGenerationError(
                "Field array requested but field not generated. Call generate_field first.",
                grid_size=self.grid_size,
            )

        if self.field_array is None:
            raise FieldGenerationError(
                "Field array is None. Cannot return field data.",
                grid_size=self.grid_size,
            )

        # Validate field integrity if validate_field enabled with range and shape checking
        if validate_field:
            validation_result = self.validate_field(
                check_mathematical_properties=True,
                check_memory_usage=False,
                check_performance=False,
            )
            if not validation_result["is_valid"]:
                self.logger.warning(
                    f"Field validation found issues: {validation_result['issues']}"
                )

        # Create deep copy of field_array if copy_array enabled for safe external access
        if copy_array:
            array_copy = copy.deepcopy(self.field_array)
            self.logger.debug("Returned deep copy of concentration field array")
            return array_copy
        else:
            # Log field access for performance monitoring and usage tracking
            self.logger.debug("Returned reference to concentration field array")
            return self.field_array

    def update_field(
        self,
        new_source_location: Optional[Coordinates] = None,
        new_sigma: Optional[float] = None,
        auto_regenerate: bool = True,
        clear_cache: bool = True,
    ) -> bool:
        """
        Update concentration field with new parameters including source location and sigma with
        validation, regeneration, and performance tracking for dynamic field modifications.

        Args:
            new_source_location: New source position for field center update
            new_sigma: New sigma parameter for plume dispersion modification
            auto_regenerate: Automatically regenerate field with new parameters
            clear_cache: Clear sampling cache due to parameter changes

        Returns:
            True if field updated successfully, False if validation failed

        Raises:
            FieldGenerationError: If parameter validation or regeneration fails
        """
        try:
            return self._extracted_from_update_field_26(
                new_source_location, new_sigma, clear_cache, auto_regenerate
            )
        except (ValidationError, FieldGenerationError) as e:
            self.logger.error(f"Field update failed: {e}", exception=e)
            return False
        except Exception as e:
            raise FieldGenerationError(
                f"Unexpected error during field update: {e}",
                grid_size=self.grid_size,
                generation_params=self.generation_params,
            ) from e

    # TODO Rename this here and in `update_field`
    def _extracted_from_update_field_26(  # noqa: C901
        self, new_source_location, new_sigma, clear_cache, auto_regenerate
    ):
        # Check if parameters actually changed to avoid unnecessary regeneration
        current_params = self.generation_params
        params_changed = False

        if new_source_location is not None:
            old_location = current_params.get("source_location")
            if old_location is None or (
                old_location.x != new_source_location.x
                or old_location.y != new_source_location.y
            ):
                params_changed = True

        if new_sigma is not None:
            old_sigma = current_params.get("sigma", DEFAULT_PLUME_SIGMA)
            if abs(old_sigma - new_sigma) > GAUSSIAN_PRECISION:
                params_changed = True

        if not params_changed:
            self.logger.debug("No field parameter changes detected, skipping update")
            return True

        # Validate new parameters using existing validation functions
        if new_source_location is not None:
            new_source_location = validate_coordinates(
                coordinates=new_source_location,
                grid_bounds=self.grid_size,
            )

        if new_sigma is not None and new_sigma <= 0:
            raise ValidationError(f"New sigma must be positive, got {new_sigma}")

        # Update generation_params with new values and modification timestamp
        if new_source_location is not None:
            current_params["source_location"] = new_source_location
        if new_sigma is not None:
            current_params["sigma"] = new_sigma
        current_params["last_update"] = np.float64(0)  # Placeholder timestamp

        # Clear sampling cache if clear_cache enabled or parameters changed significantly
        if clear_cache and self.enable_caching:
            cache_cleared = len(self.sampling_cache)
            self.sampling_cache.clear()
            self.logger.debug(f"Cleared {cache_cleared} cached sampling results")

        # Regenerate field if auto_regenerate enabled using generate_field method
        if auto_regenerate:
            source_loc = new_source_location or current_params.get("source_location")
            sigma = new_sigma or current_params.get("sigma", DEFAULT_PLUME_SIGMA)

            if source_loc is None:
                raise FieldGenerationError(
                    "Cannot regenerate field: no source location available",
                    grid_size=self.grid_size,
                    generation_params=current_params,
                )

            self.generate_field(
                source_location=source_loc,
                sigma=sigma,
                force_regeneration=True,
                normalize_field=True,
            )

            # Log parameter update with old and new values for monitoring
            self.logger.info(
                f"Field updated - Source: {source_loc}, Sigma: {sigma}, "
                f"Generation time: {self.last_generation_time_ms:.2f}ms"
            )

        return True

    def validate_field(  # noqa: C901
        self,
        check_mathematical_properties: bool = True,
        check_memory_usage: bool = False,
        check_performance: bool = False,
    ) -> Dict[str, Any]:
        """
        Validate current field state including array properties, value ranges, mathematical
        properties, and memory usage with comprehensive error reporting for field integrity.

        Args:
            check_mathematical_properties: Validate mathematical consistency and properties
            check_memory_usage: Check memory usage against system limits
            check_performance: Validate performance metrics against targets

        Returns:
            Field validation result with status, issues, and optimization recommendations
        """
        validation_result = {
            "is_valid": True,
            "issues": [],
            "warnings": [],
            "recommendations": [],
            "validation_timestamp": np.float64(0),  # Placeholder
            "checks_performed": {
                "basic_properties": True,
                "mathematical_properties": check_mathematical_properties,
                "memory_usage": check_memory_usage,
                "performance": check_performance,
            },
        }

        try:
            # Check field_array exists and has proper shape matching grid_size
            if self.field_array is None:
                validation_result["is_valid"] = False
                validation_result["issues"].append(
                    "Field array is None - field not generated"
                )
                return validation_result

            expected_shape = (self.grid_size.height, self.grid_size.width)
            if self.field_array.shape != expected_shape:
                validation_result["is_valid"] = False
                validation_result["issues"].append(
                    f"Field array shape {self.field_array.shape} does not match "
                    f"grid size {expected_shape}"
                )

            # Validate array dtype matches FIELD_DTYPE for consistency
            if self.field_array.dtype != FIELD_DTYPE:
                validation_result["warnings"].append(
                    f"Field array dtype {self.field_array.dtype} differs from "
                    f"expected {FIELD_DTYPE}"
                )

            # Check concentration values are within CONCENTRATION_RANGE bounds
            field_min = np.min(self.field_array)
            field_max = np.max(self.field_array)

            if field_min < CONCENTRATION_RANGE[0] or field_max > CONCENTRATION_RANGE[1]:
                validation_result["warnings"].append(
                    f"Field values [{field_min:.6f}, {field_max:.6f}] outside "
                    f"expected range {CONCENTRATION_RANGE}"
                )

            # Check for NaN or infinite values that could cause calculation errors
            nan_count = np.sum(np.isnan(self.field_array))
            inf_count = np.sum(np.isinf(self.field_array))

            if nan_count > 0:
                validation_result["is_valid"] = False
                validation_result["issues"].append(
                    f"Field contains {nan_count} NaN values"
                )

            if inf_count > 0:
                validation_result["is_valid"] = False
                validation_result["issues"].append(
                    f"Field contains {inf_count} infinite values"
                )

            # Validate mathematical properties if check_mathematical_properties enabled
            if check_mathematical_properties:
                # Check field normalization and peak location accuracy
                if "source_location" in self.generation_params:
                    source_loc = self.generation_params["source_location"]
                    try:
                        source_y, source_x = source_loc.to_array_index(self.grid_size)
                        source_value = self.field_array[source_y, source_x]

                        if abs(source_value - 1.0) > GAUSSIAN_PRECISION:
                            validation_result["warnings"].append(
                                f"Source location value {source_value:.6f} not normalized to 1.0"
                            )
                    except Exception as e:
                        validation_result["warnings"].append(
                            f"Could not validate source location normalization: {e}"
                        )

                # Check for reasonable gradient properties
                gradient_x, gradient_y = np.gradient(self.field_array)
                max_gradient = np.max(np.sqrt(gradient_x**2 + gradient_y**2))
                if max_gradient > 1.0:
                    validation_result["warnings"].append(
                        f"High gradient detected: {max_gradient:.3f} - may indicate numerical issues"
                    )

            # Check memory usage against limits if check_memory_usage enabled
            if check_memory_usage:
                field_memory_mb = self.field_array.nbytes / (1024 * 1024)
                if field_memory_mb > MEMORY_LIMIT_PLUME_FIELD_MB:
                    validation_result["warnings"].append(
                        f"Field memory usage {field_memory_mb:.1f}MB exceeds limit "
                        f"{MEMORY_LIMIT_PLUME_FIELD_MB}MB"
                    )

                # Check cache memory usage
                if self.enable_caching:
                    cache_memory_estimate = (
                        len(self.sampling_cache) * 8
                    )  # Rough estimate
                    validation_result["cache_memory_bytes"] = cache_memory_estimate

            # Validate performance metrics if check_performance enabled
            if check_performance and self.last_generation_time_ms is not None:
                if (
                    self.last_generation_time_ms
                    > PERFORMANCE_TARGET_PLUME_GENERATION_MS
                ):
                    validation_result["warnings"].append(
                        f"Generation time {self.last_generation_time_ms:.2f}ms exceeds "
                        f"target {PERFORMANCE_TARGET_PLUME_GENERATION_MS}ms"
                    )

                # Check cache efficiency
                total_samples = self.cache_hits + self.cache_misses
                if total_samples > 0:
                    hit_rate = self.cache_hits / total_samples
                    if hit_rate < 0.5:
                        validation_result["recommendations"].append(
                            f"Low cache hit rate ({hit_rate:.1%}) - consider increasing cache size"
                        )

            # Generate optimization recommendations for memory and performance
            if (
                len(validation_result["warnings"]) == 0
                and len(validation_result["issues"]) == 0
            ):
                validation_result["recommendations"].append(
                    "Field validation passed all checks"
                )
            else:
                if field_memory_mb > _FIELD_MEMORY_WARNING_THRESHOLD:
                    validation_result["recommendations"].append(
                        "Consider reducing grid size to lower memory usage"
                    )
                if (
                    self.last_generation_time_ms
                    and self.last_generation_time_ms
                    > PERFORMANCE_TARGET_PLUME_GENERATION_MS
                ):
                    validation_result["recommendations"].append(
                        "Consider optimizing field generation or reducing grid complexity"
                    )

        except Exception as e:
            validation_result["is_valid"] = False
            validation_result["issues"].append(
                f"Field validation failed with error: {e}"
            )

        return validation_result

    def get_field_statistics(
        self,
        include_distribution_analysis: bool = False,
        include_performance_data: bool = True,
        include_memory_analysis: bool = False,
    ) -> Dict[str, Any]:
        """
        Calculate and return comprehensive field statistics including concentration distribution,
        peak analysis, memory usage, and performance metrics for debugging and analysis.

        Args:
            include_distribution_analysis: Include detailed concentration distribution statistics
            include_performance_data: Include performance timing and cache efficiency metrics
            include_memory_analysis: Include memory usage breakdown and optimization analysis

        Returns:
            Comprehensive field statistics dictionary with distribution, performance, and memory information
        """
        if self.field_array is None:
            return {
                "field_generated": False,
                "error": "Field not generated - call generate_field first",
            }

        stats = {
            "field_generated": self.is_generated,
            "grid_size": {
                "width": self.grid_size.width,
                "height": self.grid_size.height,
                "total_cells": self.grid_size.total_cells(),
            },
            "generation_timestamp": self.generation_params.get("last_update", 0),
        }

        # Calculate basic field statistics (min, max, mean, std) using NumPy functions
        stats["basic_statistics"] = {
            "min_value": float(np.min(self.field_array)),
            "max_value": float(np.max(self.field_array)),
            "mean_value": float(np.mean(self.field_array)),
            "std_value": float(np.std(self.field_array)),
            "median_value": float(np.median(self.field_array)),
        }

        # Include concentration distribution analysis if include_distribution_analysis enabled
        if include_distribution_analysis:
            # Calculate percentiles and distribution shape
            percentiles = [10, 25, 50, 75, 90, 95, 99]
            percentile_values = np.percentile(self.field_array, percentiles)

            stats["distribution_analysis"] = {
                "percentiles": dict(zip(percentiles, percentile_values)),
                "concentration_above_50%": float(
                    np.sum(self.field_array > 0.5) / self.field_array.size
                ),
                "concentration_above_10%": float(
                    np.sum(self.field_array > 0.1) / self.field_array.size
                ),
                "zero_concentration_cells": int(
                    np.sum(self.field_array < GAUSSIAN_PRECISION)
                ),
            }

            # Peak analysis with location
            peak_location = np.unravel_index(
                np.argmax(self.field_array), self.field_array.shape
            )
            stats["distribution_analysis"]["peak_location"] = {
                "array_indices": peak_location,
                "coordinates": (
                    peak_location[1],
                    peak_location[0],
                ),  # Convert to (x, y)
                "peak_value": float(self.field_array[peak_location]),
            }

        # Add performance metrics if include_performance_data enabled
        if include_performance_data:
            total_samples = self.cache_hits + self.cache_misses
            hit_rate = (self.cache_hits / total_samples) if total_samples > 0 else 0.0

            stats["performance_data"] = {
                "last_generation_time_ms": self.last_generation_time_ms,
                "cache_statistics": {
                    "cache_enabled": self.enable_caching,
                    "cache_hits": self.cache_hits,
                    "cache_misses": self.cache_misses,
                    "hit_rate": hit_rate,
                    "cached_entries": len(self.sampling_cache),
                },
                "generation_parameters": self.generation_params.copy(),
            }

            # Performance comparison with targets
            if self.last_generation_time_ms is not None:
                stats["performance_data"]["performance_comparison"] = {
                    "target_ms": PERFORMANCE_TARGET_PLUME_GENERATION_MS,
                    "actual_ms": self.last_generation_time_ms,
                    "meets_target": self.last_generation_time_ms
                    <= PERFORMANCE_TARGET_PLUME_GENERATION_MS,
                    "performance_ratio": self.last_generation_time_ms
                    / PERFORMANCE_TARGET_PLUME_GENERATION_MS,
                }

        # Include memory usage analysis if include_memory_analysis enabled
        if include_memory_analysis:
            field_memory_bytes = self.field_array.nbytes
            field_memory_mb = field_memory_bytes / (1024 * 1024)

            cache_memory_estimate = (
                len(self.sampling_cache) * 32
            )  # Rough estimate per entry
            total_memory_mb = field_memory_mb + (cache_memory_estimate / (1024 * 1024))

            stats["memory_analysis"] = {
                "field_array_bytes": field_memory_bytes,
                "field_array_mb": field_memory_mb,
                "cache_memory_estimate_bytes": cache_memory_estimate,
                "total_memory_mb": total_memory_mb,
                "memory_efficiency": {
                    "bytes_per_cell": field_memory_bytes / self.grid_size.total_cells(),
                    "within_limit": field_memory_mb <= MEMORY_LIMIT_PLUME_FIELD_MB,
                    "memory_limit_mb": MEMORY_LIMIT_PLUME_FIELD_MB,
                    "usage_percentage": (field_memory_mb / MEMORY_LIMIT_PLUME_FIELD_MB)
                    * 100,
                },
            }

        # Include field quality metrics and mathematical property analysis
        stats["field_quality"] = {
            "value_range_valid": (
                CONCENTRATION_RANGE[0]
                <= stats["basic_statistics"]["min_value"]
                <= stats["basic_statistics"]["max_value"]
                <= CONCENTRATION_RANGE[1]
            ),
            "no_invalid_values": {
                "nan_count": int(np.sum(np.isnan(self.field_array))),
                "inf_count": int(np.sum(np.isinf(self.field_array))),
                "negative_count": int(np.sum(self.field_array < 0)),
            },
            "normalization_check": abs(stats["basic_statistics"]["max_value"] - 1.0)
            < GAUSSIAN_PRECISION,
        }

        return stats

    def clear_cache(self) -> Dict[str, Any]:
        """
        Clear sampling cache and reset cache statistics with memory cleanup for performance
        optimization and testing scenarios with detailed cleanup reporting.

        Returns:
            Cache clearing report with memory freed and performance impact analysis
        """
        if not self.enable_caching:
            return {
                "cache_enabled": False,
                "action_taken": "none",
                "message": "Caching is disabled",
            }

        # Calculate memory freed from cache clearing operations
        entries_cleared = len(self.sampling_cache)
        estimated_memory_freed = entries_cleared * 32  # Rough estimate per cache entry

        # Clear sampling_cache dictionary and reset cache statistics
        self.sampling_cache.clear()
        old_hits = self.cache_hits
        old_misses = self.cache_misses

        # Reset cache performance counters (hits, misses) to zero
        self.cache_hits = 0
        self.cache_misses = 0

        # Log cache clearing operation with memory and performance impact
        self.logger.debug(
            f"Cache cleared: {entries_cleared} entries, "
            f"~{estimated_memory_freed} bytes freed"
        )

        # Return cache clearing report with statistics and memory analysis
        return {
            "cache_enabled": True,
            "action_taken": "cache_cleared",
            "entries_cleared": entries_cleared,
            "estimated_memory_freed_bytes": estimated_memory_freed,
            "previous_statistics": {
                "cache_hits": old_hits,
                "cache_misses": old_misses,
                "hit_rate": (
                    (old_hits / (old_hits + old_misses))
                    if (old_hits + old_misses) > 0
                    else 0.0
                ),
            },
            "cleanup_timestamp": np.float64(0),  # Placeholder timestamp
        }

    def clone(
        self,
        new_grid_size: Optional[GridSize] = None,
        copy_field_data: bool = True,
        preserve_cache: bool = False,
        copy_performance_stats: bool = False,
    ) -> "ConcentrationField":
        """
        Create deep copy of concentration field with optional parameter modifications while
        preserving field data and configuration for testing and analysis scenarios.

        Args:
            new_grid_size: Optional new grid size for cloned field
            copy_field_data: Copy generated field array data to clone
            preserve_cache: Preserve sampling cache in cloned field
            copy_performance_stats: Copy performance statistics to clone

        Returns:
            Cloned concentration field with optional modifications and preserved data

        Raises:
            FieldGenerationError: If cloning fails due to parameter or memory issues
        """
        try:
            # Create new ConcentrationField instance with same or modified grid_size
            clone_grid_size = (
                new_grid_size if new_grid_size is not None else self.grid_size
            )
            cloned_field = ConcentrationField(
                grid_size=clone_grid_size, enable_caching=self.enable_caching
            )

            # Copy field_array data if copy_field_data enabled and field is generated
            if copy_field_data and self.is_generated and self.field_array is not None:
                if new_grid_size is not None and new_grid_size != self.grid_size:
                    # Need to regenerate field for different grid size
                    if "source_location" in self.generation_params:
                        source_loc = self.generation_params["source_location"]
                        sigma = self.generation_params.get("sigma", DEFAULT_PLUME_SIGMA)

                        # Adjust source location if needed for new grid size
                        if not source_loc.is_within_bounds(clone_grid_size):
                            # Center the source in the new grid
                            source_loc = Coordinates(
                                x=clone_grid_size.width // 2,
                                y=clone_grid_size.height // 2,
                            )

                        cloned_field.generate_field(
                            source_location=source_loc,
                            sigma=sigma,
                            normalize_field=True,
                        )
                else:
                    # Direct copy for same grid size
                    cloned_field.field_array = copy.deepcopy(self.field_array)
                    cloned_field.is_generated = True

            # Copy sampling cache if preserve_cache enabled for performance continuity
            if preserve_cache and self.enable_caching:
                cloned_field.sampling_cache = copy.deepcopy(self.sampling_cache)

            # Copy performance statistics if copy_performance_stats enabled
            if copy_performance_stats:
                cloned_field.cache_hits = self.cache_hits
                cloned_field.cache_misses = self.cache_misses
                cloned_field.last_generation_time_ms = self.last_generation_time_ms

            # Preserve generation parameters and configuration settings
            cloned_field.generation_params = copy.deepcopy(self.generation_params)
            cloned_field.interpolation_method = self.interpolation_method

            # Log cloning operation with parameter differences and memory usage
            memory_estimate = clone_grid_size.estimate_memory_mb()
            self.logger.info(
                f"Field cloned - Grid: {clone_grid_size.width}×{clone_grid_size.height}, "
                f"Data copied: {copy_field_data}, Cache preserved: {preserve_cache}, "
                f"Memory: ~{memory_estimate:.1f}MB"
            )

            # Return cloned field ready for independent operations and modifications
            return cloned_field

        except Exception as e:
            raise FieldGenerationError(
                f"Field cloning failed: {e}",
                grid_size=self.grid_size,
                generation_params={
                    "clone_target_grid": new_grid_size,
                    "copy_field_data": copy_field_data,
                    "preserve_cache": preserve_cache,
                },
            ) from e


def create_concentration_field(  # noqa: C901
    grid_size: GridDimensions = DEFAULT_GRID_SIZE,
    source_location: Optional[Coordinates] = None,
    sigma: Optional[float] = None,
    enable_caching: bool = True,
    validate_parameters: bool = True,
) -> ConcentrationField:
    """
    Factory function to create ConcentrationField instances with parameter validation, memory
    estimation, and optimized configuration for different use cases including plume model
    integration and rendering pipeline requirements.

    Args:
        grid_size: Grid dimensions using GridDimensions type alias for flexible initialization
        source_location: Optional source position, defaults to grid center if not provided
        sigma: Optional dispersion parameter, defaults to DEFAULT_PLUME_SIGMA if not provided
        enable_caching: Enable result caching for performance optimization
        validate_parameters: Enable comprehensive parameter validation before creation

    Returns:
        Configured ConcentrationField instance ready for plume calculations and sampling operations

    Raises:
        FieldGenerationError: If field creation fails due to invalid parameters or resource constraints
    """
    try:
        # Validate grid_size parameter using validate_grid_size function with memory limit checking
        if isinstance(grid_size, (tuple, list)):
            if len(grid_size) != 2:
                raise ValidationError(
                    f"Grid size tuple must have 2 elements, got {len(grid_size)}"
                )
            grid_obj = GridSize(width=int(grid_size[0]), height=int(grid_size[1]))
        elif isinstance(grid_size, GridSize):
            grid_obj = grid_size
        else:
            raise ValidationError(f"Invalid grid_size type: {type(grid_size)}")

        if validate_parameters:
            validate_grid_size(
                grid_size=grid_obj,
                check_memory_limits=True,
                validate_performance=True,
            )

        # Set source_location to grid center if not provided using grid_size.center() method
        if source_location is None:
            center_x = grid_obj.width // 2
            center_y = grid_obj.height // 2
            source_location = Coordinates(x=center_x, y=center_y)
        elif validate_parameters:
            # Validate source_location coordinates within grid bounds using validate_coordinates
            source_location = validate_coordinates(
                coordinates=source_location,
                grid_bounds=grid_obj,
            )

        # Set sigma to DEFAULT_PLUME_SIGMA if not provided with range validation
        if sigma is None:
            sigma = DEFAULT_PLUME_SIGMA
        elif validate_parameters and sigma <= 0:
            raise ValidationError(f"Sigma must be positive, got {sigma}")

        # Estimate memory requirements using estimate_field_memory function
        memory_estimate = estimate_field_memory(
            grid_size=grid_obj,
            include_cache_overhead=enable_caching,
            include_intermediate_arrays=True,
        )

        # Check memory estimate against MEMORY_LIMIT_PLUME_FIELD_MB constraint
        total_memory = memory_estimate["total_memory_mb"]
        if total_memory > MEMORY_LIMIT_PLUME_FIELD_MB:
            raise ResourceError(
                f"Estimated memory usage {total_memory:.1f}MB exceeds limit {MEMORY_LIMIT_PLUME_FIELD_MB}MB",
                resource_type="memory",
                current_usage=total_memory,
                limit_exceeded=MEMORY_LIMIT_PLUME_FIELD_MB,
            )

        # Create ConcentrationField instance with validated parameters and caching configuration
        field = ConcentrationField(grid_size=grid_obj, enable_caching=enable_caching)

        # Generate field with provided parameters if source_location and sigma are specified
        field.generate_field(
            source_location=source_location,
            sigma=sigma,
            force_regeneration=False,
            normalize_field=True,
        )

        # Log field creation with parameters and memory estimate for monitoring
        field.logger.info(
            f"Concentration field created - Grid: {grid_obj.width}×{grid_obj.height}, "
            f"Source: ({source_location.x}, {source_location.y}), Sigma: {sigma}, "
            f"Memory: ~{total_memory:.1f}MB"
        )

        # Return configured ConcentrationField ready for mathematical operations
        return field

    except (ValidationError, ResourceError):
        # Re-raise validation and resource errors without modification
        raise
    except Exception as e:
        raise FieldGenerationError(
            f"Concentration field creation failed: {e}",
            grid_size=grid_obj if "grid_obj" in locals() else None,
            generation_params={
                "grid_size": grid_size,
                "source_location": source_location,
                "sigma": sigma,
                "enable_caching": enable_caching,
            },
        ) from e


def validate_field_parameters(  # noqa: C901
    grid_size: GridDimensions,
    source_location: CoordinateType,
    sigma: float,
    check_memory_limits: bool = True,
    validate_performance: bool = True,
) -> Dict[str, Any]:
    """
    Comprehensive validation function for concentration field parameters including grid dimensions,
    source location bounds, sigma range, mathematical consistency, and resource constraints with
    detailed error reporting and optimization recommendations.

    Args:
        grid_size: Grid dimensions to validate using GridDimensions type alias
        source_location: Source coordinates to validate using CoordinateType union
        sigma: Gaussian dispersion parameter to validate for mathematical stability
        check_memory_limits: Enable memory usage validation against system constraints
        validate_performance: Enable performance feasibility analysis with timing estimates

    Returns:
        Validation result dictionary with status, warnings, memory estimates, and parameter recommendations

    Raises:
        ValidationError: If critical validation failures are detected that prevent field creation
    """
    validation_result = {
        "is_valid": True,
        "critical_errors": [],
        "warnings": [],
        "recommendations": [],
        "memory_analysis": {},
        "performance_analysis": {},
        "parameter_analysis": {
            "grid_size": "pending",
            "source_location": "pending",
            "sigma": "pending",
        },
    }

    try:
        # Validate grid_size using validate_grid_size with dimension and memory checking
        try:
            if isinstance(grid_size, (tuple, list)):
                grid_obj = GridSize(width=int(grid_size[0]), height=int(grid_size[1]))
            else:
                grid_obj = grid_size

            validate_grid_size(
                grid_size=grid_obj,
                check_memory_limits=check_memory_limits,
                validate_performance=validate_performance,
            )
            validation_result["parameter_analysis"]["grid_size"] = "valid"

        except ValidationError as e:
            validation_result["is_valid"] = False
            validation_result["critical_errors"].append(
                f"Grid size validation failed: {e}"
            )
            validation_result["parameter_analysis"]["grid_size"] = "invalid"
            grid_obj = None

        # Validate source_location using validate_coordinates with grid bounds verification
        try:
            if isinstance(source_location, Coordinates):
                source_coords = source_location
            elif (
                isinstance(source_location, (tuple, list)) and len(source_location) == 2
            ):
                source_coords = Coordinates(
                    x=int(source_location[0]), y=int(source_location[1])
                )
            else:
                raise ValidationError(
                    "Source location must be Coordinates or (x, y) tuple"
                )

            if grid_obj:
                source_coords = validate_coordinates(
                    coordinates=source_coords,
                    grid_bounds=grid_obj,
                )

            validation_result["parameter_analysis"]["source_location"] = "valid"

        except ValidationError as e:
            validation_result["is_valid"] = False
            validation_result["critical_errors"].append(
                f"Source location validation failed: {e}"
            )
            validation_result["parameter_analysis"]["source_location"] = "invalid"
            source_coords = None

        # Check sigma is within valid range for Gaussian calculations with numerical stability
        try:
            if not isinstance(sigma, (int, float)):
                raise ValidationError(f"Sigma must be numeric, got {type(sigma)}")

            if sigma <= 0:
                raise ValidationError(f"Sigma must be positive, got {sigma}")

            if sigma < GAUSSIAN_PRECISION:
                validation_result["warnings"].append(
                    f"Small sigma ({sigma}) may cause numerical precision issues"
                )
                validation_result["recommendations"].append(
                    f"Consider sigma >= {GAUSSIAN_PRECISION} for numerical stability"
                )

            validation_result["parameter_analysis"]["sigma"] = "valid"

        except ValidationError as e:
            validation_result["is_valid"] = False
            validation_result["critical_errors"].append(f"Sigma validation failed: {e}")
            validation_result["parameter_analysis"]["sigma"] = "invalid"

        # Estimate memory usage if check_memory_limits enabled and validate against system constraints
        if check_memory_limits and grid_obj:
            memory_analysis = estimate_field_memory(
                grid_size=grid_obj,
                include_cache_overhead=True,
                include_intermediate_arrays=True,
            )
            validation_result["memory_analysis"] = memory_analysis

            if memory_analysis["total_memory_mb"] > MEMORY_LIMIT_PLUME_FIELD_MB:
                validation_result["warnings"].append(
                    f"Memory usage {memory_analysis['total_memory_mb']:.1f}MB "
                    f"exceeds limit {MEMORY_LIMIT_PLUME_FIELD_MB}MB"
                )
                validation_result["recommendations"].append(
                    "Reduce grid size or disable caching to lower memory usage"
                )

        # Check performance feasibility if validate_performance enabled with timing estimates
        if validate_performance and grid_obj:
            estimated_cells = grid_obj.total_cells()
            # Rough performance estimate based on grid size
            estimated_time_ms = estimated_cells / 100000  # Simplified model

            validation_result["performance_analysis"] = {
                "estimated_generation_time_ms": estimated_time_ms,
                "meets_performance_target": estimated_time_ms
                <= PERFORMANCE_TARGET_PLUME_GENERATION_MS,
                "performance_target_ms": PERFORMANCE_TARGET_PLUME_GENERATION_MS,
            }

            if estimated_time_ms > PERFORMANCE_TARGET_PLUME_GENERATION_MS:
                validation_result["warnings"].append(
                    f"Estimated generation time {estimated_time_ms:.1f}ms "
                    f"exceeds target {PERFORMANCE_TARGET_PLUME_GENERATION_MS}ms"
                )
                validation_result["recommendations"].append(
                    "Consider reducing grid size for better performance"
                )

        # Validate mathematical consistency between sigma and grid dimensions for proper plume shape
        if grid_obj and sigma and source_coords:
            grid_diagonal = np.sqrt(grid_obj.width**2 + grid_obj.height**2)

            if sigma > grid_diagonal / 2:
                validation_result["warnings"].append(
                    f"Large sigma ({sigma}) may create uniform field across grid"
                )
                validation_result["recommendations"].append(
                    f"Consider sigma < {grid_diagonal/2:.1f} for better gradient definition"
                )
            elif sigma < grid_diagonal / 50:
                validation_result["warnings"].append(
                    f"Small sigma ({sigma}) may create very sharp gradients"
                )
                validation_result["recommendations"].append(
                    f"Consider sigma > {grid_diagonal/50:.1f} for smoother fields"
                )

        # Generate optimization recommendations for performance and memory efficiency
        if len(validation_result["critical_errors"]) == 0:
            if not validation_result["warnings"]:
                validation_result["recommendations"].append(
                    "All parameters are within optimal ranges"
                )
            else:
                validation_result["recommendations"].append(
                    "Review warnings for potential optimizations"
                )

        return validation_result

    except Exception as e:
        validation_result["is_valid"] = False
        validation_result["critical_errors"].append(f"Parameter validation failed: {e}")
        return validation_result


def estimate_field_memory(
    grid_size: GridDimensions,
    include_cache_overhead: bool = True,
    include_intermediate_arrays: bool = True,
    cache_size: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Memory estimation function calculating required memory for concentration field storage including
    array overhead, caching requirements, and system resource analysis for resource constraint validation.

    Args:
        grid_size: Grid dimensions for memory calculation using GridDimensions type alias
        include_cache_overhead: Include sampling cache memory in estimation
        include_intermediate_arrays: Include temporary arrays used during field generation
        cache_size: Optional cache size override, defaults to _FIELD_CACHE_SIZE

    Returns:
        Memory estimation dictionary with total usage, component breakdown, and optimization suggestions
    """
    # Convert grid_size to GridSize object for consistent memory calculation
    if isinstance(grid_size, (tuple, list)):
        grid_obj = GridSize(width=int(grid_size[0]), height=int(grid_size[1]))
    else:
        grid_obj = grid_size

    memory_breakdown = {}

    # Calculate base field memory using GridSize.estimate_memory_mb with FIELD_DTYPE
    base_memory_mb = grid_obj.estimate_memory_mb()
    memory_breakdown["field_array_mb"] = base_memory_mb

    # Add cache overhead if include_cache_overhead enabled based on cache_size parameter
    if include_cache_overhead:
        cache_entries = cache_size if cache_size is not None else _FIELD_CACHE_SIZE
        # Estimate: each cache entry is ~32 bytes (key + value + overhead)
        cache_memory_mb = (cache_entries * 32) / (1024 * 1024)
        memory_breakdown["cache_overhead_mb"] = cache_memory_mb
    else:
        memory_breakdown["cache_overhead_mb"] = 0.0

    # Include intermediate array memory if include_intermediate_arrays for generation calculations
    if include_intermediate_arrays:
        # Estimate memory for meshgrid arrays (x_mesh, y_mesh) and distance calculations
        total_cells = grid_obj.total_cells()
        bytes_per_cell = np.dtype(FIELD_DTYPE).itemsize

        # Meshgrid arrays: 2 arrays of same size as field
        meshgrid_memory_mb = (2 * total_cells * bytes_per_cell) / (1024 * 1024)
        # Distance calculation arrays: 2 difference arrays + 1 distance_squared array
        distance_memory_mb = (3 * total_cells * bytes_per_cell) / (1024 * 1024)

        intermediate_memory_mb = meshgrid_memory_mb + distance_memory_mb
        memory_breakdown["intermediate_arrays_mb"] = intermediate_memory_mb
    else:
        memory_breakdown["intermediate_arrays_mb"] = 0.0

    # Add safety margin for NumPy operation overhead and memory fragmentation
    safety_margin_mb = sum(memory_breakdown.values()) * 0.1  # 10% safety margin
    memory_breakdown["safety_margin_mb"] = safety_margin_mb

    # Calculate total memory estimate
    total_memory_mb = sum(memory_breakdown.values())

    # Compare total estimate against MEMORY_LIMIT_PLUME_FIELD_MB constraint
    within_limit = total_memory_mb <= MEMORY_LIMIT_PLUME_FIELD_MB
    usage_percentage = (total_memory_mb / MEMORY_LIMIT_PLUME_FIELD_MB) * 100

    # Generate memory optimization recommendations for large field configurations
    recommendations = []
    if total_memory_mb > MEMORY_LIMIT_PLUME_FIELD_MB:
        recommendations.append(
            f"Reduce grid size to stay within {MEMORY_LIMIT_PLUME_FIELD_MB}MB limit"
        )

        # Calculate suggested grid reduction
        scale_factor = np.sqrt(MEMORY_LIMIT_PLUME_FIELD_MB / total_memory_mb)
        suggested_width = int(grid_obj.width * scale_factor)
        suggested_height = int(grid_obj.height * scale_factor)
        recommendations.append(
            f"Suggested reduced size: {suggested_width}×{suggested_height}"
        )

    if include_cache_overhead and memory_breakdown["cache_overhead_mb"] > 5.0:
        recommendations.append("Consider reducing cache size or disabling caching")

    if not recommendations:
        recommendations.append("Memory usage is within acceptable limits")

    # Return comprehensive memory analysis with usage estimates and recommendations
    return {
        "total_memory_mb": total_memory_mb,
        "memory_breakdown": memory_breakdown,
        "within_memory_limit": within_limit,
        "memory_limit_mb": MEMORY_LIMIT_PLUME_FIELD_MB,
        "usage_percentage": usage_percentage,
        "optimization_recommendations": recommendations,
        "grid_size": {
            "width": grid_obj.width,
            "height": grid_obj.height,
            "total_cells": grid_obj.total_cells(),
        },
    }


def clear_field_cache(
    force_cleanup: bool = False, cache_filter: Optional[str] = None
) -> Dict[str, Any]:
    """
    Cache management function to clear cached concentration fields and reset performance tracking
    with memory cleanup and garbage collection for resource management and testing scenarios.

    Args:
        force_cleanup: Enable aggressive memory cleanup including garbage collection
        cache_filter: Optional filter to selectively clear specific cached fields

    Returns:
        Cache clearing report dictionary with memory freed, performance impact, and cleanup statistics
    """
    cleanup_report = {
        "cache_cleared": False,
        "entries_cleared": 0,
        "memory_freed_estimate_mb": 0.0,
        "performance_baselines_reset": 0,
        "cleanup_actions": [],
    }

    try:
        # Clear cached field arrays from _GENERATION_PERFORMANCE_CACHE
        initial_cache_size = len(_GENERATION_PERFORMANCE_CACHE)
        cleared_entries = 0

        if cache_filter:
            # Apply cache_filter to selectively clear specific cached fields if specified
            keys_to_remove = [
                key
                for key in _GENERATION_PERFORMANCE_CACHE.keys()
                if cache_filter in key
            ]
            for key in keys_to_remove:
                del _GENERATION_PERFORMANCE_CACHE[key]
                cleared_entries += 1
            cleanup_report["cleanup_actions"].append(
                f"Selective cache clear with filter: {cache_filter}"
            )
        else:
            # Clear all cached entries
            cleared_entries = len(_GENERATION_PERFORMANCE_CACHE)
            _GENERATION_PERFORMANCE_CACHE.clear()
            cleanup_report["cleanup_actions"].append("Full cache clear")

        # Calculate memory freed from cache clearing operations
        # Rough estimate: each cache entry ~1MB (varies by field size)
        estimated_memory_freed_mb = cleared_entries * 1.0

        # Reset performance baselines and timing statistics for cleared fields
        # Note: This is a simplified implementation, full version would track specific baselines
        baselines_reset = cleared_entries

        # Force garbage collection if force_cleanup enabled for complete memory cleanup
        if force_cleanup:
            import gc

            collected = gc.collect()
            cleanup_report["cleanup_actions"].append(
                f"Garbage collection: {collected} objects collected"
            )

        # Update cleanup report with results
        cleanup_report.update(
            {
                "cache_cleared": True,
                "entries_cleared": cleared_entries,
                "memory_freed_estimate_mb": estimated_memory_freed_mb,
                "performance_baselines_reset": baselines_reset,
                "initial_cache_size": initial_cache_size,
                "final_cache_size": len(_GENERATION_PERFORMANCE_CACHE),
            }
        )

        # Log cache clearing operation with memory freed and performance impact
        if cleared_entries > 0:
            cleanup_report["cleanup_actions"].append(
                f"Cleared {cleared_entries} cache entries, freed ~{estimated_memory_freed_mb:.1f}MB"
            )

        # Update cache statistics and memory usage tracking
        cleanup_report["cache_statistics"] = {
            "total_cleared": cleared_entries,
            "memory_impact": estimated_memory_freed_mb,
            "cleanup_timestamp": np.float64(0),  # Placeholder timestamp
        }

    except Exception as e:
        cleanup_report.update(
            {
                "cache_cleared": False,
                "error": f"Cache clearing failed: {e}",
                "cleanup_actions": ["Cache clearing error occurred"],
            }
        )

    # Return cleanup report with detailed statistics and memory analysis
    return cleanup_report
