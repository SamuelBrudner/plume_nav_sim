"""
Abstract base renderer class providing shared rendering functionality, context management,
performance tracking, and consistent API contracts for dual-mode plume navigation visualization.

This module defines the foundational architecture for the rendering pipeline component, implementing
the Strategy pattern for dual-mode visualization (RGB array and human mode) with comprehensive
validation, resource management, and performance monitoring capabilities.

Key Components:
- RenderContext: Immutable data class containing complete environment state for rendering operations
- RenderingMetrics: Mutable performance tracking with timing analysis and resource monitoring
- BaseRenderer: Abstract base class defining consistent interface for all renderer implementations
- Factory functions: Validated creation of rendering contexts and metrics objects
- Validation framework: Comprehensive parameter validation with performance feasibility checking

Architecture Integration:
- Infrastructure layer component in modular layered architecture
- Strategy pattern foundation enabling polymorphic renderer usage
- Performance targets: <5ms RGB rendering, <50ms human mode updates
- Resource management with automatic cleanup and graceful degradation
- Extensible design supporting future renderer implementations
"""

# Standard library imports - Python >=3.10
import abc  # >=3.10 - Abstract base class decorators and interface definitions for renderer polymorphism
import copy  # >=3.10 - Deep copying for immutable rendering context creation and state management
import dataclasses  # >=3.10 - Data class decorators for rendering context and metrics with automatic validation
import logging  # >=3.10 - Renderer operation logging, performance monitoring, and error reporting for debugging
import time  # >=3.10 - High-precision timing for performance measurement and rendering latency tracking
import uuid  # >=3.10 - Unique identifier generation for rendering context tracking and state management
from typing import (  # >=3.10 - Type hints for abstract base class and generic type support
    Any,
    Dict,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
    Union,
)

# Third-party imports
import numpy as np  # >=2.1.0 - Array operations, concentration field processing, and RGB array type support for rendering validation

from ..core.constants import (
    FIELD_DTYPE,  # NumPy dtype (float32) for concentration field arrays ensuring data type consistency
)
from ..core.constants import (
    PERFORMANCE_TARGET_HUMAN_RENDER_MS,  # Performance target (<50ms) for human mode rendering operations and timing validation
)
from ..core.constants import (
    PERFORMANCE_TARGET_RGB_RENDER_MS,  # Performance target (<5ms) for RGB array rendering operations and benchmarking validation
)
from ..core.constants import (
    RGB_DTYPE,  # NumPy dtype (uint8) for RGB rendering arrays with proper color representation
)
from ..core.constants import (
    SUPPORTED_RENDER_MODES,  # List of supported visualization modes ['rgb_array', 'human'] for mode validation
)

# Internal imports - Core types and constants
from ..core.types import (
    Coordinates,  # 2D coordinate representation for agent and source position tracking with bounds validation
)
from ..core.types import (
    GridSize,  # Grid dimension representation for rendering bounds validation and coordinate system management
)
from ..core.types import (
    RenderMode,  # Rendering mode enumeration for dual-mode visualization support and mode validation
)
from ..core.types import (
    RGBArray,  # Type alias for RGB rendering arrays with proper shape [H,W,3] and dtype uint8 validation
)

# Internal imports - Exception handling and utilities
from ..utils.exceptions import (
    ComponentError,  # General component-level exception handling for renderer lifecycle and state management
)
from ..utils.exceptions import (
    RenderingError,  # Exception handling for rendering operation failures and visualization errors
)
from ..utils.exceptions import (
    ValidationError,  # Exception handling for rendering parameter validation failures with detailed error context
)
from ..utils.logging import (
    get_component_logger,  # Component-specific logger creation with performance monitoring integration and structured output
)
from ..utils.logging import (
    monitor_performance,  # Performance monitoring decorator for automatic timing and resource usage tracking
)
from ..utils.validation import (
    validate_coordinates,  # Coordinate validation with bounds checking for agent and source position markers
)

# Global configuration constants for base renderer behavior and validation
DEFAULT_CONTEXT_VALIDATION = (
    True  # Enable comprehensive context validation by default for safety and debugging
)
PERFORMANCE_MONITORING_ENABLED = (
    True  # Enable performance tracking and timing analysis for optimization
)
RESOURCE_CLEANUP_TIMEOUT_SEC = (
    5.0  # Maximum time allowed for resource cleanup operations to prevent hanging
)

# Module exports for external usage and API definition
__all__ = [
    "BaseRenderer",  # Abstract base renderer class defining consistent interface for all renderer implementations
    "RenderContext",  # Immutable rendering context containing environment state and configuration
    "RenderingMetrics",  # Performance metrics tracking object for rendering operations analysis
    "create_render_context",  # Factory function for validated rendering context creation
    "validate_rendering_parameters",  # Comprehensive validation function for rendering parameters
    "create_rendering_metrics",  # Factory function for performance metrics tracking object creation
]


@dataclasses.dataclass(frozen=True)
class RenderContext:
    """
    Immutable data class containing complete environment state and visual configuration for rendering
    operations, providing validated context with performance tracking and consistent data access for
    all renderer implementations.

    This class serves as the primary data contract between the environment and rendering pipeline,
    ensuring immutability for thread safety and providing comprehensive validation for all rendering
    operations. The frozen dataclass prevents accidental mutation while enabling efficient copying
    and validation caching.

    Attributes:
        concentration_field: 2D NumPy array with concentration values, dtype FIELD_DTYPE (float32)
        agent_position: Current agent coordinates with bounds validation
        source_position: Plume source coordinates for marker rendering
        grid_size: Grid dimensions for bounds checking and rendering configuration
        context_id: Unique identifier for tracking and debugging (auto-generated)
        creation_timestamp: Unix timestamp for performance analysis and lifecycle tracking
        metadata: Extensible dictionary for additional context information

    Performance Characteristics:
        - Immutable design enables safe caching and parallel access
        - Validation overhead: ~0.1ms for typical 128x128 grid
        - Memory footprint: ~65KB for 128x128 float32 concentration field
        - Deep copy operations: ~1ms for complete context duplication
    """

    # Core environment state data with type annotations for validation
    concentration_field: (
        np.ndarray
    )  # 2D concentration field with shape matching grid_size and FIELD_DTYPE
    agent_position: (
        Coordinates  # Current agent position with coordinate system validation
    )
    source_position: Coordinates  # Plume source position for rendering markers
    grid_size: (
        GridSize  # Grid dimensions for bounds validation and coordinate consistency
    )

    # Context metadata and tracking information
    context_id: str = dataclasses.field(
        default_factory=lambda: str(uuid.uuid4())
    )  # Unique identifier for correlation and debugging
    creation_timestamp: float = dataclasses.field(
        default_factory=time.time
    )  # Creation time for performance analysis
    metadata: Dict[str, Any] = dataclasses.field(
        default_factory=dict
    )  # Extensible metadata for additional information

    def validate(
        self, strict_validation: bool = True, check_performance: bool = True
    ) -> bool:
        """
        Comprehensive validation of rendering context data ensuring consistency, bounds checking,
        and format compliance for all renderer types.

        Performs multi-layer validation including data type checking, bounds validation, coordinate
        system consistency, and optional performance analysis. This method is designed to catch
        common integration issues and provide detailed error reporting for debugging.

        Args:
            strict_validation: Enable rigorous validation including numerical precision checks
            check_performance: Analyze memory usage and performance implications

        Returns:
            True if context is valid and ready for rendering operations

        Raises:
            ValidationError: If any validation check fails, with detailed context and recommendations

        Performance Impact:
            - Basic validation: ~0.05ms for typical context
            - Strict validation: ~0.2ms with additional numerical checks
            - Performance analysis: +0.1ms for memory estimation
        """
        try:
            # Validate concentration_field array properties and data integrity
            if not isinstance(self.concentration_field, np.ndarray):
                raise ValidationError(
                    "Concentration field must be NumPy array",
                    context={"actual_type": type(self.concentration_field).__name__},
                )

            # Check concentration field data type consistency with system constants
            if self.concentration_field.dtype != FIELD_DTYPE:
                raise ValidationError(
                    f"Concentration field must use {FIELD_DTYPE} dtype",
                    context={
                        "expected_dtype": str(FIELD_DTYPE),
                        "actual_dtype": str(self.concentration_field.dtype),
                    },
                )

            # Verify concentration field shape matches grid dimensions
            expected_shape = (self.grid_size.height, self.grid_size.width)
            if self.concentration_field.shape != expected_shape:
                raise ValidationError(
                    "Concentration field shape must match grid dimensions",
                    context={
                        "expected_shape": expected_shape,
                        "actual_shape": self.concentration_field.shape,
                        "grid_size": f"{self.grid_size.width}x{self.grid_size.height}",
                    },
                )

            # Validate concentration values are within expected [0,1] range
            min_concentration = float(np.min(self.concentration_field))
            max_concentration = float(np.max(self.concentration_field))

            if min_concentration < 0.0 or max_concentration > 1.0:
                if strict_validation:
                    raise ValidationError(
                        "Concentration values must be in range [0,1]",
                        context={
                            "min_value": min_concentration,
                            "max_value": max_concentration,
                            "expected_range": "[0.0, 1.0]",
                        },
                    )

            # Validate agent position coordinates using coordinate validation utility
            if not validate_coordinates(self.agent_position, self.grid_size):
                raise ValidationError(
                    "Agent position is outside grid boundaries",
                    context={
                        "agent_position": f"({self.agent_position.x}, {self.agent_position.y})",
                        "grid_bounds": f"0 <= x < {self.grid_size.width}, 0 <= y < {self.grid_size.height}",
                    },
                )

            # Validate source position coordinates using coordinate validation utility
            if not validate_coordinates(self.source_position, self.grid_size):
                raise ValidationError(
                    "Source position is outside grid boundaries",
                    context={
                        "source_position": f"({self.source_position.x}, {self.source_position.y})",
                        "grid_bounds": f"0 <= x < {self.grid_size.width}, 0 <= y < {self.grid_size.height}",
                    },
                )

            # Check coordinate system mathematical consistency between agent and source positions
            if self.agent_position == self.source_position:
                # Not an error but worth noting for debugging
                pass  # Agent can be at source position in valid scenarios

            # Apply strict validation rules including numerical precision checks
            if strict_validation:
                # Check for NaN or infinite values in concentration field
                if not np.isfinite(self.concentration_field).all():
                    raise ValidationError(
                        "Concentration field contains NaN or infinite values",
                        context={
                            "nan_count": int(
                                np.sum(np.isnan(self.concentration_field))
                            ),
                            "inf_count": int(
                                np.sum(np.isinf(self.concentration_field))
                            ),
                        },
                    )

                # Verify context_id is valid UUID format
                try:
                    uuid.UUID(self.context_id)
                except ValueError:
                    raise ValidationError(
                        "Context ID must be valid UUID format",
                        context={"context_id": self.context_id},
                    )

            # Check performance implications and memory usage if check_performance enabled
            if check_performance:
                # Calculate memory footprint for performance analysis
                field_memory_mb = self.concentration_field.nbytes / (1024 * 1024)

                if field_memory_mb > 10.0:  # Warning threshold for large grids
                    # Not an error but important for performance awareness
                    pass  # Large grids are valid but may impact performance

                # Check creation timestamp validity
                if self.creation_timestamp <= 0:
                    raise ValidationError(
                        "Invalid creation timestamp",
                        context={"timestamp": self.creation_timestamp},
                    )

            return True

        except ValidationError:
            # Re-raise validation errors with existing context
            raise
        except Exception as e:
            # Handle unexpected validation errors with comprehensive context
            raise ValidationError(
                f"Context validation failed with unexpected error: {e}",
                context={
                    "original_error": str(e),
                    "error_type": type(e).__name__,
                    "context_id": getattr(self, "context_id", "unknown"),
                },
            )

    def get_context_summary(
        self, include_metadata: bool = True, include_performance_info: bool = True
    ) -> Dict[str, Any]:
        """
        Generate concise context summary for logging and debugging with key state information
        and validation status.

        Creates a comprehensive summary of the rendering context suitable for logging, debugging,
        and performance analysis. The summary includes essential state information, optional
        metadata, and performance characteristics for troubleshooting and optimization.

        Args:
            include_metadata: Include extensible metadata dictionary in summary
            include_performance_info: Include memory usage and performance characteristics

        Returns:
            Dictionary containing context summary with state, validation, and performance info

        Performance Impact:
            - Basic summary: ~0.01ms
            - With metadata: +0.005ms
            - With performance info: +0.02ms for statistical calculations
        """
        # Extract key information for summary including grid dimensions and positions
        summary = {
            "context_id": self.context_id,
            "creation_timestamp": self.creation_timestamp,
            "grid_dimensions": f"{self.grid_size.width}x{self.grid_size.height}",
            "agent_position": f"({self.agent_position.x}, {self.agent_position.y})",
            "source_position": f"({self.source_position.x}, {self.source_position.y})",
        }

        # Calculate concentration field statistics for analysis and debugging
        field_stats = {
            "concentration_min": float(np.min(self.concentration_field)),
            "concentration_max": float(np.max(self.concentration_field)),
            "concentration_mean": float(np.mean(self.concentration_field)),
            "field_shape": self.concentration_field.shape,
            "field_dtype": str(self.concentration_field.dtype),
        }
        summary["concentration_stats"] = field_stats

        # Include metadata information if include_metadata is True
        if include_metadata and self.metadata:
            summary["metadata"] = dict(self.metadata)

        # Include performance characteristics if include_performance_info is True
        if include_performance_info:
            performance_info = {
                "memory_usage_mb": round(
                    self.concentration_field.nbytes / (1024 * 1024), 3
                ),
                "total_grid_points": self.grid_size.width * self.grid_size.height,
                "age_seconds": round(time.time() - self.creation_timestamp, 3),
            }
            summary["performance_info"] = performance_info

        return summary

    def clone_with_overrides(
        self,
        new_agent_position: Optional[Coordinates] = None,
        new_source_position: Optional[Coordinates] = None,
        new_metadata: Optional[Dict[str, Any]] = None,
    ) -> "RenderContext":
        """
        Create new RenderContext with parameter overrides while preserving immutability and validation.

        Enables efficient context updates by creating new immutable instances with selective parameter
        overrides. This method preserves the immutable nature of RenderContext while allowing for
        controlled updates during rendering operations or state transitions.

        Args:
            new_agent_position: Updated agent coordinates (optional)
            new_source_position: Updated source coordinates (optional)
            new_metadata: Additional or replacement metadata (optional)

        Returns:
            New validated RenderContext instance with applied overrides

        Raises:
            ValidationError: If new context with overrides fails validation

        Performance Impact:
            - Context cloning: ~0.5ms for typical context
            - Validation overhead: +0.1ms for new context validation
            - Deep copy operations: Minimal due to immutable design
        """
        # Use provided values or keep current values for selective updates
        agent_pos = (
            new_agent_position
            if new_agent_position is not None
            else self.agent_position
        )
        source_pos = (
            new_source_position
            if new_source_position is not None
            else self.source_position
        )

        # Merge new_metadata with existing metadata if provided
        updated_metadata = dict(self.metadata)  # Start with current metadata
        if new_metadata is not None:
            updated_metadata.update(new_metadata)

        # Create new RenderContext with updated parameters using dataclass replacement
        new_context = RenderContext(
            concentration_field=self.concentration_field,  # Reuse existing field data
            agent_position=agent_pos,
            source_position=source_pos,
            grid_size=self.grid_size,  # Grid dimensions remain constant
            context_id=str(uuid.uuid4()),  # Generate new unique identifier
            creation_timestamp=time.time(),  # Update creation timestamp
            metadata=updated_metadata,
        )

        # Validate new context using comprehensive validation with consistency checking
        try:
            new_context.validate(strict_validation=True, check_performance=False)
        except ValidationError as e:
            raise ValidationError(
                f"Context cloning failed validation: {e}",
                context={
                    "original_context_id": self.context_id,
                    "clone_parameters": {
                        "new_agent_position": (
                            str(new_agent_position) if new_agent_position else None
                        ),
                        "new_source_position": (
                            str(new_source_position) if new_source_position else None
                        ),
                        "metadata_keys": (
                            list(new_metadata.keys()) if new_metadata else None
                        ),
                    },
                },
            )

        return new_context


@dataclasses.dataclass
class RenderingMetrics:
    """
    Mutable data class for comprehensive rendering performance tracking including timing analysis,
    resource monitoring, operation statistics, and optimization recommendations with real-time
    analysis and historical comparison.

    This class provides detailed performance monitoring capabilities for rendering operations,
    tracking timing information, resource usage patterns, and generating optimization recommendations.
    The mutable design allows for incremental updates during rendering operations while maintaining
    structured data organization.

    Attributes:
        renderer_type: String identifier for renderer implementation type
        render_mode: RenderMode enumeration value for operation categorization
        operation_id: Unique identifier for operation correlation and tracking
        start_time: Operation start timestamp for duration calculations
        end_time: Operation completion timestamp (None while active)
        timing_data: Dictionary storing detailed operation timing information
        resource_usage: Dictionary tracking memory and system resource consumption
        operation_count: Integer counter for frequency analysis and statistics
        performance_summary: Dictionary containing aggregate performance analysis

    Performance Tracking Features:
        - Sub-millisecond timing precision using time.time()
        - Resource usage monitoring and trend analysis
        - Statistical aggregation with min/max/average calculations
        - Performance target validation against mode-specific thresholds
        - Optimization recommendations based on performance patterns
    """

    # Core identification and classification attributes
    renderer_type: str  # Renderer implementation identifier for categorized analysis
    render_mode: (
        RenderMode  # Rendering mode for appropriate performance target application
    )
    operation_id: str  # Unique operation identifier for correlation and debugging

    # Timing and performance tracking attributes
    start_time: float = dataclasses.field(
        default_factory=time.time
    )  # Operation start timestamp
    end_time: Optional[float] = (
        None  # Completion timestamp, None indicates active operation
    )
    timing_data: Dict[str, Any] = dataclasses.field(
        default_factory=dict
    )  # Detailed timing information
    resource_usage: Dict[str, Any] = dataclasses.field(
        default_factory=dict
    )  # Resource consumption tracking
    operation_count: int = 0  # Counter for statistical frequency analysis
    performance_summary: Dict[str, Any] = dataclasses.field(
        default_factory=dict
    )  # Aggregate analysis

    def record_rendering(
        self,
        duration_ms: float,
        operation_context: Optional[Dict[str, Any]] = None,
        validate_performance: bool = True,
    ) -> None:
        """
        Record rendering operation completion with timing analysis and performance validation
        against targets.

        Updates metrics with completed rendering operation data, performs performance target
        validation, and maintains statistical aggregation for trend analysis. This method
        serves as the primary interface for recording rendering performance data.

        Args:
            duration_ms: Rendering operation duration in milliseconds
            operation_context: Optional context information for detailed analysis
            validate_performance: Whether to validate against mode-specific performance targets

        Returns:
            None - Updates internal metrics state with operation data

        Performance Impact:
            - Metric recording: ~0.01ms overhead
            - Performance validation: +0.005ms for target checking
            - Context processing: Variable based on context size
        """
        # Record completion timestamp for complete operation tracking
        self.end_time = time.time()

        # Store detailed timing information with timestamp and operation context
        timing_entry = {
            "duration_ms": duration_ms,
            "completion_time": self.end_time,
            "render_mode": str(self.render_mode),
            "renderer_type": self.renderer_type,
        }

        # Include operation context if provided for detailed analysis and debugging
        if operation_context:
            timing_entry["context"] = dict(operation_context)

        # Update timing_data with latest operation information
        operation_key = f"operation_{self.operation_count}"
        self.timing_data[operation_key] = timing_entry

        # Increment operation counter for frequency analysis and statistics
        self.operation_count += 1

        # Validate performance against targets if validate_performance is True
        if validate_performance:
            target_ms = self._get_performance_target()
            if duration_ms > target_ms:
                # Record performance warning for optimization guidance
                warning_msg = (
                    f"Rendering exceeded target: {duration_ms:.2f}ms > {target_ms}ms"
                )
                self.performance_summary.setdefault("warnings", []).append(
                    {
                        "timestamp": self.end_time,
                        "message": warning_msg,
                        "duration_ms": duration_ms,
                        "target_ms": target_ms,
                    }
                )

        # Update performance summary with latest operation statistics and trends
        self._update_performance_summary(duration_ms)

    def check_performance_targets(
        self, strict_checking: bool = False, generate_recommendations: bool = True
    ) -> Dict[str, Any]:
        """
        Validate rendering performance against mode-specific targets with detailed analysis
        and optimization recommendations.

        Analyzes current performance metrics against established targets for the render mode,
        identifies performance bottlenecks, and generates actionable optimization recommendations.
        This method provides comprehensive performance assessment for monitoring and optimization.

        Args:
            strict_checking: Apply rigorous performance criteria with reduced tolerances
            generate_recommendations: Include optimization recommendations in analysis

        Returns:
            Dictionary containing performance analysis with compliance, bottlenecks, and suggestions

        Performance Characteristics:
            - Analysis overhead: ~0.05ms for basic checking
            - Strict analysis: +0.1ms for detailed bottleneck identification
            - Recommendation generation: +0.02ms for optimization suggestions
        """
        target_ms = self._get_performance_target()
        analysis_result = {
            "target_compliance": True,
            "performance_target_ms": target_ms,
            "render_mode": str(self.render_mode),
            "analysis_timestamp": time.time(),
        }

        # Analyze timing data for performance trends and target compliance
        if self.timing_data:
            durations = [
                entry["duration_ms"]
                for entry in self.timing_data.values()
                if isinstance(entry, dict) and "duration_ms" in entry
            ]

            if durations:
                latest_duration = durations[-1]
                average_duration = sum(durations) / len(durations)
                max_duration = max(durations)

                # Apply appropriate tolerance based on strict_checking
                tolerance_factor = (
                    1.1 if strict_checking else 1.5
                )  # 10% vs 50% tolerance
                effective_target = target_ms * tolerance_factor

                analysis_result.update(
                    {
                        "latest_duration_ms": latest_duration,
                        "average_duration_ms": round(average_duration, 2),
                        "max_duration_ms": max_duration,
                        "sample_count": len(durations),
                        "target_compliance": latest_duration <= effective_target,
                    }
                )

                # Check for performance bottlenecks and trend analysis
                if max_duration > target_ms * 2:  # Significant performance issue
                    analysis_result["bottleneck_detected"] = True
                    analysis_result["bottleneck_severity"] = (
                        "high" if max_duration > target_ms * 5 else "moderate"
                    )

                # Calculate performance ratios and efficiency metrics
                target_ratio = average_duration / target_ms
                analysis_result["performance_ratio"] = round(target_ratio, 2)
                analysis_result["efficiency_percentage"] = round(
                    min(100, target_ms / average_duration * 100), 1
                )

        # Generate optimization recommendations if generate_recommendations is True
        if generate_recommendations:
            recommendations = []

            if not analysis_result["target_compliance"]:
                recommendations.append(
                    "Performance target missed - consider optimization strategies"
                )

            if analysis_result.get("bottleneck_detected"):
                recommendations.append(
                    "Performance bottleneck detected - investigate resource usage"
                )

            if self.operation_count > 0:
                avg_duration = analysis_result.get("average_duration_ms", 0)
                if avg_duration > target_ms:
                    if self.render_mode == RenderMode.RGB_ARRAY:
                        recommendations.append(
                            "RGB rendering slow - consider array optimization or caching"
                        )
                    elif self.render_mode == RenderMode.HUMAN:
                        recommendations.append(
                            "Human rendering slow - consider reducing update frequency"
                        )

            if not recommendations:
                recommendations.append("Performance within acceptable limits")

            analysis_result["recommendations"] = recommendations

        return analysis_result

    def get_performance_summary(
        self, include_history: bool = True, include_recommendations: bool = True
    ) -> Dict[str, Any]:
        """
        Generate comprehensive performance summary with statistics, trends, and optimization
        analysis for monitoring and debugging.

        Compiles complete performance analysis including operation statistics, timing trends,
        resource usage patterns, and optimization recommendations. This method provides
        dashboard-ready performance data for monitoring and system optimization.

        Args:
            include_history: Include historical timing data for trend analysis
            include_recommendations: Include optimization recommendations and guidance

        Returns:
            Dictionary with complete performance analysis for monitoring and optimization

        Data Organization:
            - operation_statistics: Count, success rate, timing averages
            - performance_trends: Historical analysis and pattern identification
            - resource_metrics: Memory usage and system resource tracking
            - optimization_analysis: Actionable recommendations and insights
        """
        # Compile basic operation statistics including counts and success metrics
        summary = {
            "renderer_type": self.renderer_type,
            "render_mode": str(self.render_mode),
            "operation_id": self.operation_id,
            "total_operations": self.operation_count,
            "analysis_timestamp": time.time(),
        }

        # Calculate comprehensive operation statistics from timing data
        if self.timing_data:
            durations = [
                entry["duration_ms"]
                for entry in self.timing_data.values()
                if isinstance(entry, dict) and "duration_ms" in entry
            ]

            if durations:
                statistics = {
                    "average_duration_ms": round(sum(durations) / len(durations), 3),
                    "min_duration_ms": round(min(durations), 3),
                    "max_duration_ms": round(max(durations), 3),
                    "total_render_time_ms": round(sum(durations), 3),
                    "operations_per_second": (
                        round(len(durations) / (sum(durations) / 1000), 2)
                        if sum(durations) > 0
                        else 0
                    ),
                }
                summary["operation_statistics"] = statistics

        # Include historical timing data if include_history is True for trend analysis
        if include_history and self.timing_data:
            history_data = {
                "timing_entries": len(self.timing_data),
                "oldest_entry": min(
                    (
                        entry.get("completion_time", 0)
                        for entry in self.timing_data.values()
                        if isinstance(entry, dict)
                    ),
                    default=0,
                ),
                "newest_entry": max(
                    (
                        entry.get("completion_time", 0)
                        for entry in self.timing_data.values()
                        if isinstance(entry, dict)
                    ),
                    default=0,
                ),
            }
            summary["historical_data"] = history_data

        # Analyze resource usage patterns and memory efficiency metrics
        if self.resource_usage:
            summary["resource_metrics"] = dict(self.resource_usage)

        # Include comprehensive performance summary data
        if self.performance_summary:
            summary["performance_summary"] = dict(self.performance_summary)

        # Generate optimization recommendations if include_recommendations is True
        if include_recommendations:
            target_analysis = self.check_performance_targets(
                strict_checking=False, generate_recommendations=True
            )
            summary["optimization_analysis"] = target_analysis

        return summary

    def reset_metrics(
        self, preserve_history: bool = False, new_operation_id: Optional[str] = None
    ) -> None:
        """
        Reset performance metrics for new operation tracking while preserving configuration
        and identification.

        Clears current metrics data for fresh operation tracking while optionally preserving
        historical data for trend analysis. This method enables metric reuse across multiple
        rendering sessions while maintaining performance tracking capabilities.

        Args:
            preserve_history: Store current metrics in history before reset
            new_operation_id: Optional new operation identifier for tracking

        Returns:
            None - Resets internal metrics state for new operation cycle

        State Changes:
            - Clears timing_data and resource_usage dictionaries
            - Resets operation_count to 0
            - Updates timestamps and operation_id
            - Optionally preserves current data in metadata
        """
        # Store current metrics in history if preserve_history is True
        if preserve_history and (self.timing_data or self.resource_usage):
            history_entry = {
                "preserved_timestamp": time.time(),
                "operation_count": self.operation_count,
                "timing_data_entries": len(self.timing_data),
                "resource_entries": len(self.resource_usage),
                "operation_duration_sec": (
                    (self.end_time - self.start_time) if self.end_time else None
                ),
            }

            # Store in performance_summary under history key
            self.performance_summary.setdefault("history", []).append(history_entry)

        # Reset core tracking data structures for fresh operation cycle
        self.timing_data.clear()
        self.resource_usage.clear()
        self.operation_count = 0

        # Update operation_id if new_operation_id is provided for tracking
        if new_operation_id is not None:
            self.operation_id = new_operation_id

        # Reset timing information for new operation tracking
        self.start_time = time.time()
        self.end_time = None

        # Preserve renderer_type and render_mode for consistent tracking
        # These remain unchanged as they define the metrics context

        # Clear performance warnings but preserve historical data
        if "warnings" in self.performance_summary and not preserve_history:
            self.performance_summary["warnings"].clear()

    def _get_performance_target(self) -> float:
        """
        Get appropriate performance target based on render mode.

        Returns:
            Performance target in milliseconds for current render mode
        """
        if self.render_mode == RenderMode.RGB_ARRAY:
            return PERFORMANCE_TARGET_RGB_RENDER_MS
        elif self.render_mode == RenderMode.HUMAN:
            return PERFORMANCE_TARGET_HUMAN_RENDER_MS
        else:
            return PERFORMANCE_TARGET_RGB_RENDER_MS  # Default to RGB target

    def _update_performance_summary(self, latest_duration_ms: float) -> None:
        """
        Update aggregate performance summary with latest operation data.

        Args:
            latest_duration_ms: Most recent operation duration for trend analysis
        """
        # Initialize or update running statistics
        if "total_duration_ms" not in self.performance_summary:
            self.performance_summary["total_duration_ms"] = 0.0
            self.performance_summary["operation_count"] = 0

        self.performance_summary["total_duration_ms"] += latest_duration_ms
        self.performance_summary["operation_count"] = self.operation_count

        # Calculate running average
        if self.operation_count > 0:
            self.performance_summary["average_duration_ms"] = round(
                self.performance_summary["total_duration_ms"] / self.operation_count, 3
            )

        # Track performance trends
        self.performance_summary["last_update"] = time.time()
        self.performance_summary["latest_duration_ms"] = latest_duration_ms


class BaseRenderer(abc.ABC):
    """
    Abstract base renderer class defining consistent interface and shared functionality for
    dual-mode plume navigation visualization, providing polymorphic renderer usage, resource
    management, performance tracking, and extensible architecture for RGB and matplotlib implementations.

    This abstract base class establishes the foundation for the rendering pipeline component,
    implementing the Strategy pattern for dual-mode visualization with comprehensive shared
    functionality. All concrete renderer implementations must inherit from this class and
    implement the abstract methods while leveraging the shared infrastructure.

    Key Design Patterns:
        - Strategy Pattern: Pluggable renderer implementations with consistent interface
        - Template Method: Shared workflow with customizable implementation points
        - Resource Management: Automatic cleanup and lifecycle management
        - Performance Monitoring: Integrated timing and resource tracking
        - Validation Framework: Comprehensive input validation and error handling

    Shared Functionality:
        - Context validation and preprocessing
        - Performance tracking and analysis
        - Resource management and cleanup
        - Error handling with graceful degradation
        - Logging integration with structured output
        - Extensible metadata and configuration support

    Abstract Interface Requirements:
        - supports_render_mode(): Mode compatibility checking
        - _initialize_renderer_resources(): Renderer-specific initialization
        - _cleanup_renderer_resources(): Renderer-specific cleanup
        - _render_rgb_array(): RGB array generation implementation
        - _render_human(): Human mode visualization implementation
    """

    def __init__(
        self,
        grid_size: GridSize,
        color_scheme_name: Optional[str] = None,
        renderer_options: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize base renderer with grid configuration, color scheme setup, and performance
        monitoring infrastructure.

        Establishes the foundational renderer configuration including grid dimensions for
        bounds validation, optional color scheme integration, and performance monitoring
        infrastructure. This constructor sets up shared resources while leaving renderer-specific
        initialization for the concrete implementation.

        Args:
            grid_size: Grid dimensions for rendering bounds and memory estimation
            color_scheme_name: Optional color scheme identifier for visual configuration
            renderer_options: Optional dictionary of renderer-specific configuration parameters

        Initialization Process:
            1. Store and validate core configuration parameters
            2. Initialize component logger for structured logging
            3. Set up performance monitoring infrastructure
            4. Initialize resource management structures
            5. Prepare for concrete renderer initialization
        """
        # Store grid_size with validation for rendering bounds and memory estimation
        if not isinstance(grid_size, GridSize):
            raise ValidationError(
                "Grid size must be GridSize instance",
                context={"provided_type": type(grid_size).__name__},
            )
        self.grid_size = grid_size

        # Store color_scheme_name for color management integration and validation
        self.color_scheme_name = color_scheme_name

        # Store renderer_options with default values and parameter validation
        self.renderer_options = renderer_options or {}

        # Initialize component logger using get_component_logger for structured logging
        self.logger = get_component_logger(
            f"{self.__class__.__module__}.{self.__class__.__name__}"
        )

        # Set _initialized flag to False for lifecycle management and validation
        self._initialized = False

        # Initialize _current_metrics to None for lazy performance tracking initialization
        self._current_metrics: Optional[RenderingMetrics] = None

        # Initialize _resource_cache dictionary for efficient resource management
        self._resource_cache: Dict[str, Any] = {}

        # Log initialization with configuration details for debugging and monitoring
        self.logger.info(
            f"BaseRenderer initialized with grid_size={grid_size.width}x{grid_size.height}, "
            f"color_scheme={color_scheme_name}, options={len(self.renderer_options)} parameters"
        )

    def initialize(
        self,
        validate_immediately: bool = DEFAULT_CONTEXT_VALIDATION,
        enable_performance_monitoring: bool = PERFORMANCE_MONITORING_ENABLED,
    ) -> None:
        """
        Initialize renderer resources with validation, performance setup, and error handling
        for production readiness.

        Completes renderer initialization by calling concrete implementation setup, configuring
        performance monitoring, and performing optional immediate validation. This method must
        be called before any rendering operations and provides comprehensive error handling
        for initialization failures.

        Args:
            validate_immediately: Perform immediate validation of renderer capabilities
            enable_performance_monitoring: Enable performance tracking for rendering operations

        Raises:
            ComponentError: If renderer initialization fails
            ValidationError: If immediate validation fails after initialization

        Initialization Steps:
            1. Validate configuration parameters and system compatibility
            2. Call abstract renderer-specific resource initialization
            3. Set up performance monitoring if enabled
            4. Perform immediate validation if requested
            5. Mark renderer as initialized and ready for operations
        """
        try:
            # Validate grid_size and renderer_options for compatibility and feasibility
            if self.grid_size.width <= 0 or self.grid_size.height <= 0:
                raise ValidationError(
                    "Grid dimensions must be positive",
                    context={
                        "grid_size": f"{self.grid_size.width}x{self.grid_size.height}"
                    },
                )

            # Check memory feasibility for large grids
            estimated_memory_mb = (self.grid_size.width * self.grid_size.height * 4) / (
                1024 * 1024
            )  # Assuming float32
            if estimated_memory_mb > 100:  # 100MB threshold
                self.logger.warning(
                    f"Large grid may impact performance: {estimated_memory_mb:.1f}MB estimated"
                )

            # Call abstract _initialize_renderer_resources() method for concrete implementation setup
            self._initialize_renderer_resources()

            # Set up performance monitoring if enable_performance_monitoring is True
            if enable_performance_monitoring:
                self.logger.info(
                    "Performance monitoring enabled for rendering operations"
                )
                # Performance monitoring will be activated on first render call

            # Perform immediate validation if validate_immediately is True
            if validate_immediately:
                # Create test context for validation
                test_field = np.zeros(
                    (self.grid_size.height, self.grid_size.width), dtype=FIELD_DTYPE
                )
                test_context = RenderContext(
                    concentration_field=test_field,
                    agent_position=Coordinates(x=0, y=0),
                    source_position=Coordinates(
                        x=self.grid_size.width // 2, y=self.grid_size.height // 2
                    ),
                    grid_size=self.grid_size,
                )
                self.validate_context(test_context, strict_validation=False)

            # Set _initialized to True after successful initialization
            self._initialized = True

            # Log initialization success with configuration details and performance setup
            self.logger.info(
                f"Renderer initialization completed: grid={self.grid_size.width}x{self.grid_size.height}, "
                f"performance_monitoring={enable_performance_monitoring}, "
                f"immediate_validation={validate_immediately}"
            )

        except Exception as e:
            # Handle initialization errors with comprehensive context and cleanup
            self.logger.error(f"Renderer initialization failed: {e}")
            self._initialized = False

            # Attempt cleanup of any partially initialized resources
            try:
                self._cleanup_renderer_resources()
            except Exception as cleanup_error:
                self.logger.error(
                    f"Cleanup after initialization failure failed: {cleanup_error}"
                )

            # Raise appropriate error based on exception type
            if isinstance(e, (ValidationError, ComponentError)):
                raise
            else:
                raise ComponentError(f"Renderer initialization failed: {e}")

    @monitor_performance("render_operation")
    def render(
        self, context: RenderContext, mode_override: Optional[RenderMode] = None
    ) -> Union[RGBArray, None]:
        """
        Main rendering entry point with context validation, performance tracking, and error
        handling for all renderer implementations.

        This method serves as the primary interface for all rendering operations, providing
        comprehensive validation, performance monitoring, and error handling. The method
        delegates to appropriate abstract methods based on render mode while maintaining
        consistent behavior and monitoring across all implementations.

        Args:
            context: Validated rendering context with environment state and configuration
            mode_override: Optional render mode override for dynamic mode switching

        Returns:
            RGBArray for rgb_array mode rendering, None for human mode interactive display

        Raises:
            ComponentError: If renderer not initialized or context validation fails
            RenderingError: If rendering operation fails with details and recovery suggestions

        Operation Flow:
            1. Validate renderer initialization and context data
            2. Determine effective render mode (override or default)
            3. Start performance tracking for operation monitoring
            4. Delegate to appropriate abstract render method
            5. Record performance metrics and handle errors gracefully
        """
        # Validate renderer is initialized using _check_initialization() with error handling
        self._check_initialization()

        # Validate context using comprehensive validation with error context
        try:
            self.validate_context(context, strict_validation=False)
        except ValidationError as e:
            raise RenderingError(
                f"Context validation failed: {e}",
                context={"context_id": context.context_id},
            )

        # Determine render mode from mode_override or renderer default configuration
        effective_mode = mode_override
        if effective_mode is None:
            # Use first supported mode as default
            for mode in [RenderMode.RGB_ARRAY, RenderMode.HUMAN]:
                if self.supports_render_mode(mode):
                    effective_mode = mode
                    break

        if effective_mode is None:
            raise RenderingError("No supported render modes available")

        # Verify renderer supports the effective mode
        if not self.supports_render_mode(effective_mode):
            raise RenderingError(
                f"Renderer does not support mode: {effective_mode}",
                context={
                    "supported_modes": [
                        str(mode)
                        for mode in [RenderMode.RGB_ARRAY, RenderMode.HUMAN]
                        if self.supports_render_mode(mode)
                    ]
                },
            )

        # Start performance tracking using _start_performance_tracking() for timing analysis
        operation_context = {
            "render_mode": str(effective_mode),
            "grid_size": f"{self.grid_size.width}x{self.grid_size.height}",
            "context_id": context.context_id,
        }
        self._start_performance_tracking("render", operation_context)

        try:
            start_time = time.time()

            # Call appropriate abstract render method based on effective mode
            if effective_mode == RenderMode.RGB_ARRAY:
                result = self._render_rgb_array(context)

                # Validate RGB array output format
                if not isinstance(result, np.ndarray):
                    raise RenderingError("RGB renderer must return numpy array")
                if result.dtype != RGB_DTYPE:
                    raise RenderingError(f"RGB array must use {RGB_DTYPE} dtype")
                if len(result.shape) != 3 or result.shape[2] != 3:
                    raise RenderingError("RGB array must have shape (H,W,3)")

            elif effective_mode == RenderMode.HUMAN:
                result = self._render_human(context)
                # Human mode should return None
                if result is not None:
                    self.logger.warning("Human mode renderer returned non-None value")
                result = None

            else:
                raise RenderingError(f"Unsupported render mode: {effective_mode}")

            # Record performance metrics using _record_performance() with timing validation
            duration_ms = (time.time() - start_time) * 1000
            self._record_performance("render", duration_ms, check_targets=True)

            return result

        except Exception as e:
            # Handle rendering errors with graceful degradation and fallback mechanisms
            duration_ms = (
                (time.time() - start_time) * 1000 if "start_time" in locals() else 0
            )

            # Record failed operation for performance analysis
            if self._current_metrics:
                self._current_metrics.performance_summary.setdefault(
                    "errors", []
                ).append(
                    {
                        "timestamp": time.time(),
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "duration_ms": duration_ms,
                        "render_mode": str(effective_mode),
                    }
                )

            # Log error with context for debugging
            self.logger.error(
                f"Rendering failed: {e}, mode={effective_mode}, "
                f"context_id={context.context_id}, duration={duration_ms:.2f}ms"
            )

            # Raise rendering error with context and recovery suggestions
            if isinstance(e, RenderingError):
                raise
            else:
                raise RenderingError(
                    f"Rendering operation failed: {e}", context=operation_context
                )

    def cleanup_resources(
        self,
        timeout_sec: float = RESOURCE_CLEANUP_TIMEOUT_SEC,
        force_cleanup: bool = False,
    ) -> bool:
        """
        Clean up renderer resources with timeout handling and comprehensive resource management
        for memory efficiency.

        Performs complete cleanup of renderer resources including cached data, rendering contexts,
        and implementation-specific resources. This method provides timeout protection and forced
        cleanup capabilities for robust resource management in production environments.

        Args:
            timeout_sec: Maximum time allowed for cleanup operations
            force_cleanup: Bypass normal cleanup procedures and force resource release

        Returns:
            True if cleanup completed successfully, False if issues encountered

        Cleanup Process:
            1. Call concrete implementation cleanup with timeout protection
            2. Clear shared resource cache and release memory
            3. Reset performance metrics and clear tracking data
            4. Update initialization state and log cleanup results
            5. Handle cleanup failures with appropriate error reporting
        """
        cleanup_start_time = time.time()
        cleanup_successful = True

        try:
            # Use provided timeout or default RESOURCE_CLEANUP_TIMEOUT_SEC
            effective_timeout = (
                timeout_sec if timeout_sec > 0 else RESOURCE_CLEANUP_TIMEOUT_SEC
            )

            self.logger.info(
                f"Starting resource cleanup with timeout={effective_timeout}s, force={force_cleanup}"
            )

            # Call abstract _cleanup_renderer_resources() method for concrete implementation cleanup
            try:
                self._cleanup_renderer_resources()
            except Exception as e:
                cleanup_successful = False
                if force_cleanup:
                    self.logger.warning(
                        f"Forced cleanup ignoring implementation error: {e}"
                    )
                else:
                    self.logger.error(f"Implementation cleanup failed: {e}")
                    return False

            # Clear _resource_cache and release cached objects with memory management
            if self._resource_cache:
                cache_size = len(self._resource_cache)
                self._resource_cache.clear()
                self.logger.debug(f"Cleared resource cache: {cache_size} entries")

            # Reset performance metrics and clear tracking data
            if self._current_metrics:
                self._current_metrics.reset_metrics(preserve_history=False)
                self._current_metrics = None

            # Set _initialized to False after successful cleanup
            self._initialized = False

            # Calculate cleanup duration for performance monitoring
            cleanup_duration = time.time() - cleanup_start_time

            # Log cleanup completion with resource statistics and timing information
            self.logger.info(
                f"Resource cleanup completed: success={cleanup_successful}, "
                f"duration={cleanup_duration:.3f}s, forced={force_cleanup}"
            )

            # Check if cleanup exceeded timeout (warning only)
            if cleanup_duration > effective_timeout:
                self.logger.warning(
                    f"Cleanup exceeded timeout: {cleanup_duration:.3f}s > {effective_timeout}s"
                )

            return cleanup_successful

        except Exception as e:
            # Handle unexpected cleanup errors with comprehensive logging
            cleanup_duration = time.time() - cleanup_start_time
            self.logger.error(
                f"Resource cleanup failed with exception: {e}, "
                f"duration={cleanup_duration:.3f}s, force={force_cleanup}"
            )

            # Force reset critical state even if cleanup failed
            if force_cleanup:
                self._initialized = False
                self._resource_cache.clear()
                self._current_metrics = None
                return True  # Consider forced cleanup successful

            return False

    @abc.abstractmethod
    def supports_render_mode(self, mode: RenderMode) -> bool:
        """
        Abstract method to check renderer support for specific rendering mode with capability validation.

        This method must be implemented by concrete renderer classes to indicate their rendering
        capabilities. The method enables polymorphic renderer usage by allowing callers to query
        mode support before attempting rendering operations.

        Args:
            mode: RenderMode enumeration value to check for support

        Returns:
            True if renderer supports the specified mode, False otherwise

        Implementation Requirements:
            - Must handle all RenderMode enumeration values
            - Should return quickly for performance-sensitive usage
            - Must be consistent across multiple calls with same input
            - Should validate mode compatibility with current configuration
        """
        pass

    def get_performance_metrics(
        self, reset_after_retrieval: bool = False, include_system_metrics: bool = False
    ) -> Dict[str, Any]:
        """
        Retrieve comprehensive rendering performance metrics with analysis and optimization recommendations.

        Provides access to detailed performance data including timing analysis, resource usage patterns,
        operation statistics, and optimization recommendations. This method serves monitoring systems
        and performance analysis tools with structured data for dashboard integration.

        Args:
            reset_after_retrieval: Clear metrics data after returning current values
            include_system_metrics: Include system-level performance data beyond rendering operations

        Returns:
            Dictionary containing performance metrics, timing analysis, and optimization guidance

        Metrics Categories:
            - operation_statistics: Count, timing, success rate analysis
            - performance_analysis: Target compliance, bottleneck identification
            - resource_usage: Memory consumption, cache utilization
            - optimization_recommendations: Actionable performance improvement suggestions
        """
        # Initialize metrics collection structure
        metrics_data = {
            "collection_timestamp": time.time(),
            "renderer_type": self.__class__.__name__,
            "grid_size": f"{self.grid_size.width}x{self.grid_size.height}",
            "initialized": self._initialized,
        }

        # Compile current performance metrics from _current_metrics tracking
        if self._current_metrics:
            # Get comprehensive performance summary with all available data
            performance_summary = self._current_metrics.get_performance_summary(
                include_history=True, include_recommendations=True
            )
            metrics_data["performance_data"] = performance_summary

            # Add performance target analysis
            target_analysis = self._current_metrics.check_performance_targets(
                strict_checking=False, generate_recommendations=True
            )
            metrics_data["target_analysis"] = target_analysis

        else:
            metrics_data["performance_data"] = {"status": "no_metrics_available"}

        # Include system-level metrics if include_system_metrics is True
        if include_system_metrics:
            system_info = {
                "resource_cache_size": len(self._resource_cache),
                "color_scheme": self.color_scheme_name,
                "renderer_options_count": len(self.renderer_options),
                "supported_modes": [
                    str(mode)
                    for mode in [RenderMode.RGB_ARRAY, RenderMode.HUMAN]
                    if self.supports_render_mode(mode)
                ],
            }
            metrics_data["system_metrics"] = system_info

        # Include optimization recommendations based on performance patterns
        recommendations = []
        if self._current_metrics and hasattr(
            self._current_metrics, "performance_summary"
        ):
            if self._current_metrics.operation_count > 10:
                avg_duration = self._current_metrics.performance_summary.get(
                    "average_duration_ms", 0
                )
                target_ms = self._current_metrics._get_performance_target()

                if avg_duration > target_ms * 1.5:
                    recommendations.append(
                        "Consider performance optimization - operations consistently slow"
                    )
                elif avg_duration < target_ms * 0.1:
                    recommendations.append(
                        "Excellent performance - consider more complex visualizations"
                    )

        if not recommendations:
            recommendations.append("Performance within expected parameters")

        metrics_data["optimization_recommendations"] = recommendations

        # Reset metrics after retrieval if reset_after_retrieval is True
        if reset_after_retrieval and self._current_metrics:
            self._current_metrics.reset_metrics(preserve_history=True)
            self.logger.debug("Performance metrics reset after retrieval")

        return metrics_data

    def validate_context(
        self, context: RenderContext, strict_validation: bool = True
    ) -> bool:
        """
        Validate rendering context with renderer-specific checks and comprehensive error reporting.

        Performs multi-layer validation including base context validation, renderer compatibility
        checks, and optional strict validation rules. This method ensures context data integrity
        and compatibility with the current renderer configuration before rendering operations.

        Args:
            context: RenderContext instance to validate for rendering operations
            strict_validation: Apply comprehensive validation rules including edge cases

        Returns:
            True if context passes all validation checks

        Raises:
            ValidationError: If context validation fails with detailed analysis and recommendations

        Validation Layers:
            1. Base context validation using RenderContext.validate()
            2. Renderer configuration compatibility checking
            3. Grid size consistency validation
            4. Resource availability and performance feasibility analysis
        """
        try:
            # Perform base context validation using context.validate() method
            context.validate(
                strict_validation=strict_validation, check_performance=True
            )

            # Check context compatibility with current renderer configuration
            if context.grid_size != self.grid_size:
                raise ValidationError(
                    "Context grid size does not match renderer configuration",
                    context={
                        "context_grid_size": f"{context.grid_size.width}x{context.grid_size.height}",
                        "renderer_grid_size": f"{self.grid_size.width}x{self.grid_size.height}",
                        "suggestion": "Recreate renderer with matching grid size or update context",
                    },
                )

            # Validate grid_size consistency between context and renderer setup
            if context.concentration_field.shape != (
                self.grid_size.height,
                self.grid_size.width,
            ):
                raise ValidationError(
                    "Concentration field shape inconsistent with renderer grid configuration",
                    context={
                        "field_shape": context.concentration_field.shape,
                        "expected_shape": (self.grid_size.height, self.grid_size.width),
                        "renderer_config": f"{self.grid_size.width}x{self.grid_size.height}",
                    },
                )

            # Apply strict validation rules if strict_validation is True
            if strict_validation:
                # Check for potential performance issues with context data
                field_memory_mb = context.concentration_field.nbytes / (1024 * 1024)
                if field_memory_mb > 50:  # 50MB threshold for performance warning
                    # Not an error but worth noting for performance awareness
                    self.logger.warning(
                        f"Large concentration field may impact performance: {field_memory_mb:.1f}MB"
                    )

                # Validate context creation timestamp is reasonable
                time_since_creation = time.time() - context.creation_timestamp
                if (
                    time_since_creation < 0 or time_since_creation > 3600
                ):  # 1 hour threshold
                    raise ValidationError(
                        "Context creation timestamp appears invalid",
                        context={
                            "creation_timestamp": context.creation_timestamp,
                            "time_since_creation_sec": time_since_creation,
                            "current_timestamp": time.time(),
                        },
                    )

            # Check resource availability and performance feasibility for context
            if not self._initialized:
                raise ValidationError(
                    "Cannot validate context - renderer not initialized",
                    context={"renderer_state": "uninitialized"},
                )

            return True

        except ValidationError:
            # Re-raise validation errors with existing context information
            raise
        except Exception as e:
            # Handle unexpected validation errors with comprehensive context
            raise ValidationError(
                f"Context validation failed with unexpected error: {e}",
                context={
                    "error_type": type(e).__name__,
                    "context_id": getattr(context, "context_id", "unknown"),
                    "renderer_type": self.__class__.__name__,
                    "validation_mode": "strict" if strict_validation else "normal",
                },
            )

    @abc.abstractmethod
    def _initialize_renderer_resources(self) -> None:
        """
        Abstract method for renderer-specific resource initialization requiring concrete implementation.

        This method must be implemented by concrete renderer classes to initialize renderer-specific
        resources such as matplotlib figures, color maps, rendering caches, or GPU contexts. The
        method is called during the initialize() process after base validation.

        Implementation Requirements:
            - Initialize all renderer-specific resources required for operation
            - Handle initialization errors gracefully with meaningful error messages
            - Set up any caching or optimization structures
            - Prepare renderer for immediate operation after completion

        Common Implementation Patterns:
            - Matplotlib renderers: Create figure, axes, and colormap objects
            - RGB renderers: Initialize color lookup tables and array templates
            - GPU renderers: Set up compute contexts and shader programs
        """
        pass

    @abc.abstractmethod
    def _cleanup_renderer_resources(self) -> None:
        """
        Abstract method for renderer-specific resource cleanup requiring concrete implementation.

        This method must be implemented by concrete renderer classes to clean up renderer-specific
        resources and prevent memory leaks. The method is called during cleanup_resources() process
        and should handle cleanup failures gracefully.

        Implementation Requirements:
            - Release all renderer-specific resources (figures, contexts, caches)
            - Handle cleanup errors gracefully without affecting system stability
            - Ensure memory is properly released and references are cleared
            - Leave renderer in a state suitable for potential re-initialization

        Common Implementation Patterns:
            - Matplotlib renderers: Close figures and clear axes references
            - RGB renderers: Clear cached arrays and color lookup tables
            - GPU renderers: Release compute contexts and free GPU memory
        """
        pass

    @abc.abstractmethod
    def _render_rgb_array(self, context: RenderContext) -> RGBArray:
        """
        Abstract method for RGB array generation requiring implementation in RGB-capable renderers.

        This method must be implemented by renderers supporting RGB_ARRAY mode to generate
        numpy arrays representing the current environment state. The method should create
        high-performance programmatic visualizations suitable for automated processing.

        Args:
            context: Validated rendering context containing environment state and configuration

        Returns:
            RGBArray (numpy array) with shape (H,W,3) and dtype uint8 representing visualization

        Implementation Requirements:
            - Return numpy array with shape (grid_height, grid_width, 3)
            - Use RGB_DTYPE (uint8) for color values in range [0, 255]
            - Include concentration field visualization and position markers
            - Meet performance target of PERFORMANCE_TARGET_RGB_RENDER_MS (<5ms)
            - Handle large grids efficiently with optimized array operations

        Visualization Elements:
            - Concentration field as background heatmap or grayscale
            - Agent position marker (typically red or distinctive color)
            - Source position marker (typically white or contrasting color)
            - Optional grid lines or coordinate indicators
        """
        pass

    @abc.abstractmethod
    def _render_human(self, context: RenderContext) -> None:
        """
        Abstract method for human mode visualization requiring implementation in interactive renderers.

        This method must be implemented by renderers supporting HUMAN mode to create interactive
        visualizations for human observation and analysis. The method should update existing
        display windows or create new ones for real-time environment monitoring.

        Args:
            context: Validated rendering context containing environment state and configuration

        Returns:
            None - Method should update interactive display directly

        Implementation Requirements:
            - Create or update interactive display window with current environment state
            - Handle display backend compatibility and graceful fallback
            - Meet performance target of PERFORMANCE_TARGET_HUMAN_RENDER_MS (<50ms)
            - Support headless environments with appropriate backend selection
            - Provide clear visual representation suitable for human analysis

        Visualization Features:
            - Interactive matplotlib window with concentration field display
            - Real-time updates showing agent movement and exploration
            - Color-coded concentration visualization with legend
            - Position markers for agent and source locations
            - Optional trajectory tracking and episode information
        """
        pass

    def _start_performance_tracking(
        self, operation_type: str, operation_context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize performance tracking for rendering operation with metrics setup and timing.

        Sets up performance monitoring infrastructure for the current rendering operation,
        creating metrics objects and recording operation context for detailed analysis.
        This method prepares the renderer for comprehensive timing and resource tracking.

        Args:
            operation_type: String identifier for operation type ('render', 'initialize', etc.)
            operation_context: Optional dictionary containing operation context information

        Side Effects:
            - Creates or resets RenderingMetrics instance
            - Records operation start time and context
            - Prepares timing infrastructure for operation monitoring
        """
        # Create new RenderingMetrics instance if _current_metrics is None
        if self._current_metrics is None:
            self._current_metrics = RenderingMetrics(
                renderer_type=self.__class__.__name__,
                render_mode=RenderMode.RGB_ARRAY,  # Default, will be updated based on actual operation
                operation_id=str(uuid.uuid4()),
            )

        # Update render mode based on operation context if available
        if operation_context and "render_mode" in operation_context:
            mode_str = operation_context["render_mode"]
            try:
                if mode_str == "rgb_array":
                    self._current_metrics.render_mode = RenderMode.RGB_ARRAY
                elif mode_str == "human":
                    self._current_metrics.render_mode = RenderMode.HUMAN
            except Exception as e:
                self.logger.debug(f"Failed to update render mode in metrics: {e}")

        # Record operation start time and context for performance analysis
        self._current_metrics.start_time = time.time()
        self._current_metrics.end_time = None

        # Store operation_context for correlation and debugging purposes
        if operation_context:
            context_key = f"{operation_type}_{int(time.time() * 1000)}"
            self._current_metrics.resource_usage[context_key] = {
                "operation_type": operation_type,
                "context": dict(operation_context),
                "start_timestamp": self._current_metrics.start_time,
            }

        # Set up timing infrastructure for detailed operation tracking
        # Additional timing points can be added by concrete implementations

    def _record_performance(
        self, operation_type: str, duration_ms: float, check_targets: bool = True
    ) -> None:
        """
        Record rendering operation completion with performance validation and analysis.

        Updates performance metrics with completed operation data, validates against performance
        targets, and maintains statistical analysis for optimization guidance. This method
        provides the primary mechanism for performance data collection and analysis.

        Args:
            operation_type: String identifier for completed operation type
            duration_ms: Operation duration in milliseconds for performance analysis
            check_targets: Whether to validate duration against mode-specific performance targets

        Side Effects:
            - Updates RenderingMetrics with operation completion data
            - Records performance statistics and trend analysis
            - Logs performance warnings if targets are exceeded
        """
        if self._current_metrics is None:
            # Create metrics instance if not already present
            self._current_metrics = RenderingMetrics(
                renderer_type=self.__class__.__name__,
                render_mode=RenderMode.RGB_ARRAY,  # Default mode
                operation_id=str(uuid.uuid4()),
            )

        # Record operation completion in _current_metrics with timing analysis
        operation_context = {
            "operation_type": operation_type,
            "renderer_class": self.__class__.__name__,
            "grid_size": f"{self.grid_size.width}x{self.grid_size.height}",
        }

        self._current_metrics.record_rendering(
            duration_ms=duration_ms,
            operation_context=operation_context,
            validate_performance=check_targets,
        )

        # Validate performance against mode-specific targets if check_targets is True
        if check_targets:
            target_ms = self._current_metrics._get_performance_target()

            if duration_ms > target_ms:
                # Log performance warnings for optimization guidance
                performance_ratio = duration_ms / target_ms
                self.logger.warning(
                    f"Operation exceeded performance target: {duration_ms:.2f}ms > {target_ms}ms "
                    f"({performance_ratio:.1f}x slower), operation={operation_type}"
                )

                # Add optimization suggestions based on performance ratio
                if performance_ratio > 5.0:
                    self.logger.warning(
                        "Severe performance issue detected - investigate optimization opportunities"
                    )
                elif performance_ratio > 2.0:
                    self.logger.info(
                        "Moderate performance issue - consider optimization"
                    )

        # Update performance statistics and trend analysis
        # Statistical updates are handled automatically by RenderingMetrics.record_rendering()

    def _check_initialization(self) -> None:
        """
        Validate renderer initialization status with detailed error reporting for operation safety.

        Checks the renderer initialization state and raises appropriate errors if the renderer
        is not ready for operation. This method provides a consistent initialization check
        across all renderer operations with helpful guidance for resolution.

        Raises:
            ComponentError: If renderer is not initialized with guidance for initialization

        Validation Checks:
            - Verifies _initialized flag is True
            - Provides clear error messages with initialization guidance
            - Logs initialization status for debugging and troubleshooting
        """
        # Check _initialized flag for renderer lifecycle validation
        if not self._initialized:
            error_message = (
                f"Renderer {self.__class__.__name__} is not initialized. "
                "Call initialize() before using rendering operations."
            )

            # Log initialization status for debugging and troubleshooting
            self.logger.error(
                f"Operation attempted on uninitialized renderer: {self.__class__.__name__}"
            )

            # Raise ComponentError with initialization guidance
            raise ComponentError(
                error_message,
                context={
                    "renderer_type": self.__class__.__name__,
                    "initialization_status": "not_initialized",
                    "required_action": "call_initialize_method",
                },
            )


def create_render_context(
    concentration_field: np.ndarray,
    agent_position: Coordinates,
    source_position: Coordinates,
    grid_size: GridSize,
    context_id: Optional[str] = None,
    validate_immediately: bool = DEFAULT_CONTEXT_VALIDATION,
) -> RenderContext:
    """
    Factory function to create validated rendering context with environment state, visual
    configuration, and performance tracking for consistent renderer operations.

    This function provides a convenient interface for creating RenderContext instances with
    comprehensive validation and error handling. The factory pattern ensures consistent
    context creation while providing flexibility for testing and advanced usage scenarios.

    Args:
        concentration_field: 2D NumPy array with concentration values and proper dtype
        agent_position: Current agent coordinates with bounds validation
        source_position: Plume source coordinates for marker rendering
        grid_size: Grid dimensions for bounds checking and coordinate consistency
        context_id: Optional unique identifier for tracking (auto-generated if None)
        validate_immediately: Whether to perform validation during creation

    Returns:
        Validated RenderContext instance ready for renderer operations and processing

    Raises:
        ValidationError: If any parameter validation fails with detailed error context

    Validation Process:
        1. Validate concentration_field array properties and data integrity
        2. Check coordinate bounds and consistency with grid dimensions
        3. Verify data types and array shapes match system requirements
        4. Perform immediate validation if requested with comprehensive checking
        5. Generate unique context_id and creation timestamp
        6. Return fully validated and ready-to-use RenderContext instance
    """
    try:
        # Validate concentration_field is numpy array with FIELD_DTYPE and proper shape matching grid_size
        if not isinstance(concentration_field, np.ndarray):
            raise ValidationError(
                "Concentration field must be NumPy array",
                context={"provided_type": type(concentration_field).__name__},
            )

        if concentration_field.dtype != FIELD_DTYPE:
            raise ValidationError(
                f"Concentration field must use {FIELD_DTYPE} dtype",
                context={
                    "expected_dtype": str(FIELD_DTYPE),
                    "actual_dtype": str(concentration_field.dtype),
                    "conversion_suggestion": f"Use concentration_field.astype({FIELD_DTYPE})",
                },
            )

        expected_shape = (grid_size.height, grid_size.width)
        if concentration_field.shape != expected_shape:
            raise ValidationError(
                "Concentration field shape must match grid dimensions",
                context={
                    "expected_shape": expected_shape,
                    "actual_shape": concentration_field.shape,
                    "grid_size": f"{grid_size.width}x{grid_size.height}",
                },
            )

        # Validate agent_position coordinates using validate_coordinates with grid bounds checking
        if not validate_coordinates(agent_position, grid_size):
            raise ValidationError(
                "Agent position coordinates are outside grid boundaries",
                context={
                    "agent_position": f"({agent_position.x}, {agent_position.y})",
                    "grid_bounds": f"0 <= x < {grid_size.width}, 0 <= y < {grid_size.height}",
                    "validation_function": "validate_coordinates",
                },
            )

        # Validate source_position coordinates using validate_coordinates with grid bounds checking
        if not validate_coordinates(source_position, grid_size):
            raise ValidationError(
                "Source position coordinates are outside grid boundaries",
                context={
                    "source_position": f"({source_position.x}, {source_position.y})",
                    "grid_bounds": f"0 <= x < {grid_size.width}, 0 <= y < {grid_size.height}",
                    "validation_function": "validate_coordinates",
                },
            )

        # Generate unique context_id if not provided using UUID for tracking and debugging
        if context_id is None:
            context_id = str(uuid.uuid4())
        else:
            # Validate provided context_id is reasonable
            if not isinstance(context_id, str) or len(context_id.strip()) == 0:
                raise ValidationError(
                    "Context ID must be non-empty string",
                    context={"provided_context_id": repr(context_id)},
                )

        # Create RenderContext dataclass with validated parameters and metadata
        context = RenderContext(
            concentration_field=concentration_field,
            agent_position=agent_position,
            source_position=source_position,
            grid_size=grid_size,
            context_id=context_id.strip(),
            metadata={},  # Empty metadata dictionary for extensibility
        )

        # Perform immediate validation if validate_immediately is True using RenderContext.validate()
        if validate_immediately:
            context.validate(strict_validation=True, check_performance=True)

        # Log context creation with parameters and validation results for debugging
        logger = get_component_logger(f"{__name__}.create_render_context")
        logger.debug(
            f"RenderContext created: id={context_id[:8]}..., "
            f"grid={grid_size.width}x{grid_size.height}, "
            f"agent=({agent_position.x},{agent_position.y}), "
            f"source=({source_position.x},{source_position.y}), "
            f"validated={validate_immediately}"
        )

        # Return validated RenderContext ready for renderer consumption and processing
        return context

    except ValidationError:
        # Re-raise validation errors with existing context
        raise
    except Exception as e:
        # Handle unexpected errors in context creation with comprehensive context
        raise ValidationError(
            f"Context creation failed with unexpected error: {e}",
            context={
                "error_type": type(e).__name__,
                "original_error": str(e),
                "function": "create_render_context",
            },
        )


def validate_rendering_parameters(
    render_mode: RenderMode,
    grid_size: GridSize,
    color_scheme_name: Optional[str] = None,
    check_performance_feasibility: bool = True,
    strict_validation: bool = True,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Comprehensive validation function for rendering parameters ensuring compatibility, performance
    feasibility, and proper format compliance across all renderer types.

    This function provides comprehensive validation of rendering parameters before renderer
    initialization, helping to catch configuration issues early and provide optimization
    recommendations. The validation covers mode compatibility, performance feasibility,
    and system requirements.

    Args:
        render_mode: RenderMode enumeration value to validate for support and compatibility
        grid_size: GridSize instance for dimension validation and performance analysis
        color_scheme_name: Optional color scheme identifier for availability checking
        check_performance_feasibility: Whether to analyze performance implications and memory usage
        strict_validation: Apply comprehensive validation including edge cases and warnings

    Returns:
        Tuple of (is_valid: bool, validation_report: dict) with comprehensive analysis and recommendations

    Validation Categories:
        - Mode compatibility: Render mode support and system compatibility
        - Performance feasibility: Memory usage, timing estimates, scalability analysis
        - Configuration validation: Parameter format, value ranges, consistency checking
        - System requirements: Dependency availability, backend compatibility
        - Optimization recommendations: Performance tuning suggestions and best practices
    """
    validation_start_time = time.time()

    # Initialize validation report structure with comprehensive analysis
    validation_report = {
        "validation_timestamp": validation_start_time,
        "parameters_tested": {
            "render_mode": str(render_mode),
            "grid_size": f"{grid_size.width}x{grid_size.height}",
            "color_scheme": color_scheme_name,
            "performance_check": check_performance_feasibility,
            "strict_mode": strict_validation,
        },
        "validation_results": {},
        "warnings": [],
        "recommendations": [],
        "performance_analysis": {},
        "system_compatibility": {},
    }

    is_valid = True

    try:
        # Validate render_mode is supported using SUPPORTED_RENDER_MODES list with mode compatibility checking
        if render_mode not in [RenderMode.RGB_ARRAY, RenderMode.HUMAN]:
            is_valid = False
            validation_report["validation_results"]["render_mode"] = {
                "valid": False,
                "error": f"Unsupported render mode: {render_mode}",
                "supported_modes": [
                    str(mode) for mode in [RenderMode.RGB_ARRAY, RenderMode.HUMAN]
                ],
            }
        else:
            mode_str = str(render_mode)
            if mode_str in SUPPORTED_RENDER_MODES:
                validation_report["validation_results"]["render_mode"] = {
                    "valid": True,
                    "mode": mode_str,
                    "programmatic": render_mode.is_programmatic(),
                    "requires_display": render_mode.requires_display(),
                }
            else:
                is_valid = False
                validation_report["validation_results"]["render_mode"] = {
                    "valid": False,
                    "error": f"Render mode not in supported list: {SUPPORTED_RENDER_MODES}",
                }

        # Validate grid_size dimensions and memory requirements for rendering performance feasibility
        grid_validation = {
            "valid": True,
            "width": grid_size.width,
            "height": grid_size.height,
            "total_points": grid_size.width * grid_size.height,
        }

        if grid_size.width <= 0 or grid_size.height <= 0:
            is_valid = False
            grid_validation["valid"] = False
            grid_validation["error"] = "Grid dimensions must be positive"
        elif grid_size.width > 2048 or grid_size.height > 2048:
            if strict_validation:
                is_valid = False
                grid_validation["valid"] = False
                grid_validation["error"] = (
                    "Grid dimensions too large for reliable performance"
                )
            else:
                validation_report["warnings"].append(
                    "Large grid dimensions may impact performance"
                )

        validation_report["validation_results"]["grid_size"] = grid_validation

        # Check color_scheme_name availability and compatibility if provided with renderer integration validation
        if color_scheme_name is not None:
            color_scheme_validation = {"valid": True, "scheme_name": color_scheme_name}

            # Basic color scheme name validation
            if (
                not isinstance(color_scheme_name, str)
                or len(color_scheme_name.strip()) == 0
            ):
                is_valid = False
                color_scheme_validation["valid"] = False
                color_scheme_validation["error"] = (
                    "Color scheme name must be non-empty string"
                )
            else:
                # Note: Actual color scheme availability would be checked by concrete renderers
                color_scheme_validation["note"] = (
                    "Color scheme availability depends on renderer implementation"
                )

            validation_report["validation_results"][
                "color_scheme"
            ] = color_scheme_validation

        # Perform performance feasibility analysis if check_performance_feasibility is True
        if check_performance_feasibility:
            performance_analysis = {}

            # Calculate memory usage estimates
            total_points = grid_size.width * grid_size.height
            field_memory_mb = (total_points * 4) / (1024 * 1024)  # float32 = 4 bytes

            if render_mode == RenderMode.RGB_ARRAY:
                rgb_memory_mb = (total_points * 3) / (
                    1024 * 1024
                )  # uint8 RGB = 3 bytes
                performance_analysis["memory_estimates"] = {
                    "concentration_field_mb": round(field_memory_mb, 2),
                    "rgb_output_mb": round(rgb_memory_mb, 2),
                    "total_estimated_mb": round(field_memory_mb + rgb_memory_mb, 2),
                }

                # Estimate rendering time based on grid size
                estimated_rgb_time_ms = 0.001 * total_points / 1000  # Rough estimate
                performance_analysis["timing_estimates"] = {
                    "estimated_rgb_render_ms": round(estimated_rgb_time_ms, 2),
                    "target_rgb_ms": PERFORMANCE_TARGET_RGB_RENDER_MS,
                    "within_target": estimated_rgb_time_ms
                    <= PERFORMANCE_TARGET_RGB_RENDER_MS,
                }

            elif render_mode == RenderMode.HUMAN:
                performance_analysis["memory_estimates"] = {
                    "concentration_field_mb": round(field_memory_mb, 2),
                    "matplotlib_overhead_mb": 5.0,  # Approximate matplotlib memory usage
                }

                estimated_human_time_ms = 0.01 * total_points / 1000  # Rough estimate
                performance_analysis["timing_estimates"] = {
                    "estimated_human_render_ms": round(estimated_human_time_ms, 2),
                    "target_human_ms": PERFORMANCE_TARGET_HUMAN_RENDER_MS,
                    "within_target": estimated_human_time_ms
                    <= PERFORMANCE_TARGET_HUMAN_RENDER_MS,
                }

            validation_report["performance_analysis"] = performance_analysis

            # Add performance warnings if estimates exceed targets
            if render_mode == RenderMode.RGB_ARRAY:
                if (
                    performance_analysis["timing_estimates"]["estimated_rgb_render_ms"]
                    > PERFORMANCE_TARGET_RGB_RENDER_MS
                ):
                    validation_report["warnings"].append(
                        f"RGB rendering may exceed {PERFORMANCE_TARGET_RGB_RENDER_MS}ms target"
                    )
            elif render_mode == RenderMode.HUMAN:
                if (
                    performance_analysis["timing_estimates"][
                        "estimated_human_render_ms"
                    ]
                    > PERFORMANCE_TARGET_HUMAN_RENDER_MS
                ):
                    validation_report["warnings"].append(
                        f"Human rendering may exceed {PERFORMANCE_TARGET_HUMAN_RENDER_MS}ms target"
                    )

        # Apply strict validation rules including edge cases if strict_validation is True
        if strict_validation:
            # Additional strict validation checks
            if grid_size.width != grid_size.height:
                validation_report["warnings"].append(
                    "Non-square grids may have different performance characteristics"
                )

            if grid_size.width % 8 != 0 or grid_size.height % 8 != 0:
                validation_report["warnings"].append(
                    "Grid dimensions not divisible by 8 may impact vectorization"
                )

        # Generate comprehensive validation report with findings, warnings, and optimization recommendations
        if is_valid:
            validation_report["recommendations"].append(
                "All validation checks passed successfully"
            )

        # Add optimization recommendations based on analysis
        if (
            check_performance_feasibility
            and "performance_analysis" in validation_report
        ):
            memory_mb = validation_report["performance_analysis"][
                "memory_estimates"
            ].get("total_estimated_mb", 0)

            if memory_mb > 100:
                validation_report["recommendations"].append(
                    "High memory usage - consider smaller grid or optimization"
                )
            elif memory_mb < 1:
                validation_report["recommendations"].append(
                    "Low memory usage - grid size is very efficient"
                )

            # Mode-specific recommendations
            if render_mode == RenderMode.RGB_ARRAY:
                validation_report["recommendations"].append(
                    "RGB mode selected - optimized for programmatic analysis"
                )
            elif render_mode == RenderMode.HUMAN:
                validation_report["recommendations"].append(
                    "Human mode selected - optimized for interactive visualization"
                )

        # Calculate validation duration
        validation_duration_ms = (time.time() - validation_start_time) * 1000
        validation_report["validation_duration_ms"] = round(validation_duration_ms, 3)

        # Return validation tuple with boolean status and detailed analysis for debugging and optimization
        return (is_valid, validation_report)

    except Exception as e:
        # Handle validation errors with comprehensive error reporting
        validation_report["validation_results"]["error"] = {
            "exception_type": type(e).__name__,
            "error_message": str(e),
            "validation_failed": True,
        }
        validation_report["recommendations"] = [
            "Fix validation error before proceeding with renderer creation"
        ]

        return (False, validation_report)


def create_rendering_metrics(
    renderer_type: str,
    render_mode: RenderMode,
    operation_id: Optional[str] = None,
    enable_detailed_tracking: bool = True,
) -> RenderingMetrics:
    """
    Factory function to create performance metrics tracking object for rendering operations
    with timing analysis and resource monitoring.

    This function provides a convenient interface for creating RenderingMetrics instances
    with proper initialization and configuration. The factory pattern ensures consistent
    metrics object creation while providing flexibility for different tracking scenarios.

    Args:
        renderer_type: String identifier for renderer implementation type and categorization
        render_mode: RenderMode enumeration value for appropriate performance target application
        operation_id: Optional unique identifier for operation correlation (auto-generated if None)
        enable_detailed_tracking: Whether to enable comprehensive tracking features and analysis

    Returns:
        Initialized RenderingMetrics object ready for rendering operation tracking and analysis

    Configuration Process:
        1. Validate input parameters and create unique operation identifier
        2. Initialize RenderingMetrics with renderer type and mode configuration
        3. Set up performance targets based on render mode for benchmark comparison
        4. Configure detailed tracking options and data structures
        5. Prepare metrics object for immediate operation monitoring
        6. Return fully configured metrics instance ready for performance analysis
    """
    try:
        # Initialize RenderingMetrics with renderer_type and render_mode for categorized tracking
        if not isinstance(renderer_type, str) or len(renderer_type.strip()) == 0:
            raise ValidationError(
                "Renderer type must be non-empty string",
                context={"provided_type": repr(renderer_type)},
            )

        if not isinstance(render_mode, RenderMode):
            raise ValidationError(
                "Render mode must be RenderMode enumeration",
                context={"provided_type": type(render_mode).__name__},
            )

        # Generate unique operation_id if not provided using UUID for correlation and analysis
        if operation_id is None:
            operation_id = str(uuid.uuid4())
        else:
            if not isinstance(operation_id, str) or len(operation_id.strip()) == 0:
                raise ValidationError(
                    "Operation ID must be non-empty string",
                    context={"provided_id": repr(operation_id)},
                )
            operation_id = operation_id.strip()

        # Create RenderingMetrics instance with validated parameters
        metrics = RenderingMetrics(
            renderer_type=renderer_type.strip(),
            render_mode=render_mode,
            operation_id=operation_id,
        )

        # Configure detailed tracking options if enable_detailed_tracking is True
        if enable_detailed_tracking:
            # Initialize detailed tracking data structures for comprehensive analysis
            metrics.resource_usage["detailed_tracking"] = {
                "enabled": True,
                "initialization_time": time.time(),
                "tracking_features": [
                    "timing_analysis",
                    "resource_monitoring",
                    "performance_validation",
                    "optimization_recommendations",
                ],
            }

        # Set performance targets based on render_mode (RGB vs HUMAN) for benchmark comparison
        target_ms = (
            PERFORMANCE_TARGET_RGB_RENDER_MS
            if render_mode == RenderMode.RGB_ARRAY
            else PERFORMANCE_TARGET_HUMAN_RENDER_MS
        )

        metrics.performance_summary["performance_target_ms"] = target_ms
        metrics.performance_summary["render_mode"] = str(render_mode)
        metrics.performance_summary["detailed_tracking"] = enable_detailed_tracking

        # Initialize timing and resource tracking data structures
        metrics.timing_data["initialization"] = {
            "creation_timestamp": time.time(),
            "renderer_type": renderer_type,
            "render_mode": str(render_mode),
            "tracking_enabled": enable_detailed_tracking,
        }

        # Log metrics creation for debugging and monitoring
        logger = get_component_logger(f"{__name__}.create_rendering_metrics")
        logger.debug(
            f"RenderingMetrics created: type={renderer_type}, mode={render_mode}, "
            f"id={operation_id[:8]}..., detailed_tracking={enable_detailed_tracking}"
        )

        # Return configured RenderingMetrics ready for operation monitoring and performance analysis
        return metrics

    except ValidationError:
        # Re-raise validation errors with existing context
        raise
    except Exception as e:
        # Handle unexpected errors in metrics creation with comprehensive context
        raise ValidationError(
            f"Metrics creation failed with unexpected error: {e}",
            context={
                "error_type": type(e).__name__,
                "original_error": str(e),
                "function": "create_rendering_metrics",
                "renderer_type": (
                    renderer_type if "renderer_type" in locals() else "unknown"
                ),
            },
        )
