# rendering template system providing reusable, configurable templates for both RGB array generation and matplotlib human mode visualization, with performance optimization, backend compatibility management, and standardized visualization patterns for plume navigation environment rendering.

import copy  # >=3.10 - Deep copying of template configurations for modification and customization without affecting base templates
import functools  # >=3.10 - LRU cache decorator for template caching, performance optimization, and resource management in repeated operations
import time  # >=3.10 - High-precision timing for template performance measurement, benchmarking, and optimization validation
import warnings  # >=3.10 - Template deprecation warnings, backend fallback notifications, and configuration compatibility alerts
from abc import (  # >=3.10 - Abstract base classes for template interface definitions and ensuring implementation compliance
    ABC,
    abstractmethod,
)
from dataclasses import (  # >=3.10 - Template configuration data structures with automatic validation, immutability, and serialization support
    dataclass,
    field,
)
from enum import (  # >=3.10 - Template type enumeration and rendering quality level definitions with type safety
    Enum,
)
from typing import (  # >=3.10 - Type hints for template methods, configuration parameters, and factory functions with enhanced code clarity and IDE support
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

import matplotlib.pyplot as plt  # >=3.9.0 - Interactive plotting interface for matplotlib template creation, figure management, and real-time visualization updates
import numpy as np  # >=2.1.0 - Array operations, RGB array generation, mathematical transformations, and performance-optimized template operations for concentration field rendering

from plume_nav_sim.core.enums import RenderMode

# Core types for template sizing and positioning
from plume_nav_sim.core.geometry import Coordinates, GridSize
from plume_nav_sim.core.types import RGBArray

# Performance constants and configuration
from ..plume_nav_sim.core.constants import (
    BACKEND_PRIORITY_LIST,
    MATPLOTLIB_DEFAULT_FIGSIZE,
    PERFORMANCE_TARGET_HUMAN_RENDER_MS,
    PERFORMANCE_TARGET_RGB_RENDER_MS,
    RGB_DTYPE,
)

# Internal imports for color scheme integration and dual-mode optimization
from .default_colormap import ColorScheme, create_default_scheme

# Global configuration dictionaries for template defaults
DEFAULT_RGB_TEMPLATE_CONFIG = {
    "colormap": "gray",
    "agent_marker_size": (3, 3),
    "source_marker_size": (5, 5),
    "performance_optimized": True,
    "caching_enabled": True,
}

DEFAULT_MATPLOTLIB_TEMPLATE_CONFIG = {
    "figsize": (8, 8),
    "colormap": "gray",
    "backend_fallback": True,
    "interactive_updates": True,
    "animation_interval": 0.001,
}

TEMPLATE_CACHE_SIZE = 20
PERFORMANCE_MONITORING_ENABLED = True
TEMPLATE_REGISTRY = {}

QUALITY_PRESETS = {
    "ultra_fast": "Minimal features for maximum speed",
    "fast": "Balanced features with speed priority",
    "standard": "Full features with balanced performance",
    "quality": "Enhanced quality with relaxed timing",
    "premium": "Maximum quality without time constraints",
}

BACKEND_COMPATIBILITY_MATRIX = {
    "TkAgg": "Linux/macOS GUI",
    "Qt5Agg": "Cross-platform GUI",
    "Agg": "Headless compatible",
    "macosx": "macOS native",
    "webagg": "Browser-based",
}


class TemplateQuality(Enum):
    """Quality levels with predefined performance/feature profiles."""

    ULTRA_FAST = "ultra_fast"
    FAST = "fast"
    STANDARD = "standard"
    QUALITY = "quality"
    PREMIUM = "premium"

    @property
    def performance_target_ms(self) -> float:
        mapping = {
            "ultra_fast": 1.0,
            "fast": 3.0,
            "standard": 5.0,
            "quality": 10.0,
            "premium": float("inf"),
        }
        return mapping[self.value]

    @property
    def feature_set(self) -> Dict[str, bool]:
        mapping = {
            "ultra_fast": {
                "anti_aliasing": False,
                "smooth_markers": False,
                "color_interpolation": False,
                "advanced_caching": True,
            },
            "fast": {
                "anti_aliasing": False,
                "smooth_markers": True,
                "color_interpolation": False,
                "advanced_caching": True,
            },
            "standard": {
                "anti_aliasing": True,
                "smooth_markers": True,
                "color_interpolation": True,
                "advanced_caching": True,
            },
            "quality": {
                "anti_aliasing": True,
                "smooth_markers": True,
                "color_interpolation": True,
                "advanced_caching": True,
                "high_dpi": True,
            },
            "premium": {
                "anti_aliasing": True,
                "smooth_markers": True,
                "color_interpolation": True,
                "advanced_caching": True,
                "high_dpi": True,
                "maximum_precision": True,
            },
        }
        return mapping[self.value]

    @property
    def resource_priority(self) -> int:
        priorities = {
            "ultra_fast": 5,
            "fast": 4,
            "standard": 3,
            "quality": 2,
            "premium": 1,
        }
        return priorities[self.value]

    def get_performance_config(self) -> Dict[str, Any]:
        """
        Returns performance configuration dictionary for template quality level
        including timing targets and resource settings.
        """
        return {
            "performance_target_ms": self.performance_target_ms,
            "resource_priority": self.resource_priority,
            "feature_set": self.feature_set,
            "memory_limit_mb": self.performance_target_ms * 10,  # Rough heuristic
            "optimization_flags": {
                "vectorized_operations": True,
                "memory_pooling": self.feature_set.get("advanced_caching", False),
                "lazy_loading": self.value in ["ultra_fast", "fast"],
            },
        }

    def supports_feature(self, feature_name: str) -> bool:
        """Checks if template quality level supports specific rendering feature."""
        return self.feature_set.get(feature_name, False)


@dataclass
class TemplateConfig:
    """
    Comprehensive configuration data class for rendering templates containing visual settings,
    performance parameters, and customization options with validation and serialization support.
    """

    grid_size: GridSize
    color_scheme: Optional[ColorScheme] = None
    quality_level: TemplateQuality = TemplateQuality.STANDARD
    marker_config: Dict[str, Any] = field(
        default_factory=lambda: {
            "agent_size": (3, 3),
            "agent_color": [255, 0, 0],
            "source_size": (5, 5),
            "source_color": [255, 255, 255],
        }
    )
    performance_options: Dict[str, Any] = field(
        default_factory=lambda: {
            "enable_caching": True,
            "vectorized_operations": True,
            "memory_optimization": True,
        }
    )
    backend_preferences: Dict[str, Any] = field(
        default_factory=lambda: {
            "preferred_backend": "TkAgg",
            "fallback_chain": ["TkAgg", "Qt5Agg", "Agg"],
            "headless_compatible": True,
        }
    )
    caching_enabled: bool = True
    custom_parameters: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize template configuration with defaults and validation."""
        # Initialize color_scheme with default if None provided
        if self.color_scheme is None:
            self.color_scheme = create_default_scheme()

        # Validate grid_size dimensions are positive
        if self.grid_size.width <= 0 or self.grid_size.height <= 0:
            raise ValueError("Grid size dimensions must be positive")

    def validate(self, strict_mode: bool = False) -> Tuple[bool, List[str]]:
        """
        Validates template configuration completeness, consistency, and performance
        feasibility with detailed error reporting.
        """
        validation_errors = []

        # Validate grid_size dimensions and reasonable limits
        if not (8 <= self.grid_size.width <= 2048):
            validation_errors.append(
                f"Grid width {self.grid_size.width} outside reasonable range [8, 2048]"
            )

        if not (8 <= self.grid_size.height <= 2048):
            validation_errors.append(
                f"Grid height {self.grid_size.height} outside reasonable range [8, 2048]"
            )

        # Validate color_scheme if provided
        if self.color_scheme is not None:
            try:
                # Test colormap availability
                self.color_scheme.get_concentration_colormap()
            except Exception as e:
                validation_errors.append(f"Color scheme validation failed: {e}")

        # Check marker_config parameters
        agent_size = self.marker_config.get("agent_size", (3, 3))
        if not isinstance(agent_size, (tuple, list)) or len(agent_size) != 2:
            validation_errors.append("Agent marker size must be (width, height) tuple")

        # Validate performance_options consistency
        if (
            self.performance_options.get("enable_caching", True)
            and not self.caching_enabled
        ):
            validation_errors.append(
                "Performance caching enabled but template caching disabled"
            )

        # Apply strict validation if requested
        if strict_mode:
            # Check memory requirements
            memory_estimate = (self.grid_size.width * self.grid_size.height * 4) / (
                1024 * 1024
            )  # MB
            if memory_estimate > 100:
                validation_errors.append(
                    f"Estimated memory usage {memory_estimate:.1f}MB exceeds strict limit"
                )

            # Validate backend availability
            preferred_backend = self.backend_preferences.get("preferred_backend")
            try:
                import matplotlib

                available_backends = (
                    matplotlib.backend_bases.Backend._backend_map.keys()
                )
                if preferred_backend not in available_backends:
                    validation_errors.append(
                        f"Preferred backend '{preferred_backend}' not available"
                    )
            except Exception:
                pass  # Skip backend validation if matplotlib not available

        is_valid = len(validation_errors) == 0
        return is_valid, validation_errors

    def optimize_for_system(self, system_info: Dict[str, Any]) -> "TemplateConfig":
        """
        Optimizes template configuration based on current system capabilities
        and performance characteristics.
        """
        optimized = copy.deepcopy(self)

        # Analyze system performance capabilities
        cpu_cores = system_info.get("cpu_cores", 1)
        memory_gb = system_info.get("memory_gb", 4)
        display_available = system_info.get("display_available", True)

        # Adjust quality level based on system performance
        if memory_gb < 2:
            optimized.quality_level = TemplateQuality.ULTRA_FAST
        elif memory_gb < 4:
            optimized.quality_level = TemplateQuality.FAST

        # Optimize backend preferences for system
        if not display_available:
            optimized.backend_preferences["preferred_backend"] = "Agg"
            optimized.backend_preferences["fallback_chain"] = ["Agg"]

        # Configure performance options for system capabilities
        if cpu_cores > 2:
            optimized.performance_options["vectorized_operations"] = True
        if memory_gb > 8:
            optimized.performance_options["memory_optimization"] = (
                False  # Allow higher memory usage
            )

        return optimized

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts template configuration to dictionary representation
        for serialization and storage.
        """
        return {
            "grid_size": {
                "width": self.grid_size.width,
                "height": self.grid_size.height,
            },
            "color_scheme": {
                "type": type(self.color_scheme).__name__,
                "config": self.color_scheme.to_dict() if self.color_scheme else None,
            },
            "quality_level": self.quality_level.value,
            "marker_config": self.marker_config,
            "performance_options": self.performance_options,
            "backend_preferences": self.backend_preferences,
            "caching_enabled": self.caching_enabled,
            "custom_parameters": self.custom_parameters,
        }

    def clone(self, overrides: Dict[str, Any] = None) -> "TemplateConfig":
        """
        Creates deep copy of template configuration with optional parameter overrides.
        """
        cloned = copy.deepcopy(self)

        if overrides:
            for key, value in overrides.items():
                if hasattr(cloned, key):
                    setattr(cloned, key, value)
                else:
                    cloned.custom_parameters[key] = value

        return cloned


class BaseRenderTemplate(ABC):
    """
    Abstract base class defining the core interface and shared functionality for all
    rendering templates with consistent API contracts, performance monitoring, and
    resource management.
    """

    def __init__(self, config: TemplateConfig):
        """
        Initialize base template with configuration, color scheme setup, and
        performance monitoring initialization.
        """
        # Validate and store template configuration
        is_valid, errors = config.validate()
        if not is_valid:
            raise ValueError(f"Invalid template configuration: {errors}")

        self.config = config
        self.color_scheme = config.color_scheme or create_default_scheme()

        # Initialize performance metrics tracking
        self.performance_metrics = {
            "total_renders": 0,
            "total_render_time": 0.0,
            "average_render_time": 0.0,
            "last_render_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

        # Initialize template cache and state
        self._initialized = False
        self._cache = {}
        self._last_render_time = 0.0

        # Configure caching based on template settings
        if config.caching_enabled and TEMPLATE_CACHE_SIZE > 0:
            self._setup_caching()

    def _setup_caching(self):
        """Initialize template caching system for performance optimization."""
        cache_size = min(TEMPLATE_CACHE_SIZE, 50)  # Reasonable upper bound

        # Create LRU cache for frequently accessed data
        @functools.lru_cache(maxsize=cache_size)
        def cached_concentration_conversion(concentration_hash: int) -> np.ndarray:
            """Cache concentration field to RGB conversions."""
            pass  # Implementation provided by subclasses

        self._cached_conversion = cached_concentration_conversion

    def initialize(self) -> bool:
        """
        Initializes template-specific resources and validates configuration
        for operation readiness.
        """
        try:
            # Validate template configuration completeness
            is_valid, errors = self.config.validate(strict_mode=True)
            if not is_valid:
                raise RuntimeError(
                    f"Template configuration validation failed: {errors}"
                )

            # Call subclass-specific resource initialization
            self._initialize_template_resources()

            # Test template functionality with minimal rendering operation
            test_concentration = np.zeros((8, 8), dtype=np.float32)
            test_agent_pos = Coordinates(4, 4)
            test_source_pos = Coordinates(2, 2)

            try:
                test_result = self._execute_render(
                    test_concentration, test_agent_pos, test_source_pos
                )
                if (
                    test_result is None
                    and self.config.quality_level != TemplateQuality.PREMIUM
                ):
                    warnings.warn(
                        "Template functionality test returned None - may indicate rendering issues"
                    )
            except Exception as e:
                raise RuntimeError(f"Template functionality test failed: {e}")

            # Set initialization status
            self._initialized = True

            return True

        except Exception as e:
            warnings.warn(f"Template initialization failed: {e}")
            return False

    @abstractmethod
    def _initialize_template_resources(self) -> None:
        """
        Abstract method for template-specific resource initialization
        implemented by concrete subclasses.
        """
        pass

    def render(
        self,
        concentration_field: np.ndarray,
        agent_position: Coordinates,
        source_position: Coordinates,
    ) -> Optional[RGBArray]:
        """
        Main template rendering interface with performance monitoring and error handling.
        """
        if not self._initialized:
            raise RuntimeError("Template not initialized - call initialize() first")

        # Validate input parameters
        if concentration_field.ndim != 2:
            raise ValueError("Concentration field must be 2D array")

        if not (0 <= agent_position.x < concentration_field.shape[1]):
            raise ValueError(f"Agent x position {agent_position.x} outside grid bounds")

        if not (0 <= agent_position.y < concentration_field.shape[0]):
            raise ValueError(f"Agent y position {agent_position.y} outside grid bounds")

        # Start performance timing
        start_time = time.perf_counter()

        try:
            # Execute template-specific rendering logic
            result = self._execute_render(
                concentration_field, agent_position, source_position
            )

            # Update performance metrics
            render_time = time.perf_counter() - start_time
            self._update_performance_metrics(render_time, success=True)

            # Check performance targets
            target_ms = self.config.quality_level.performance_target_ms / 1000.0
            if render_time > target_ms and PERFORMANCE_MONITORING_ENABLED:
                warnings.warn(
                    f"Render time {render_time*1000:.2f}ms exceeded target {target_ms*1000:.2f}ms"
                )

            return result

        except Exception as e:
            # Handle rendering errors with recovery strategies
            render_time = time.perf_counter() - start_time
            self._update_performance_metrics(render_time, success=False)

            warnings.warn(f"Rendering failed after {render_time*1000:.2f}ms: {e}")

            # Attempt graceful degradation
            if hasattr(self, "_fallback_render"):
                try:
                    return self._fallback_render(
                        concentration_field, agent_position, source_position
                    )
                except Exception:
                    pass

            raise

    @abstractmethod
    def _execute_render(
        self,
        concentration_field: np.ndarray,
        agent_position: Coordinates,
        source_position: Coordinates,
    ) -> Optional[RGBArray]:
        """
        Abstract method for template-specific rendering implementation by concrete subclasses.
        """
        pass

    def _update_performance_metrics(self, render_time: float, success: bool):
        """Update performance tracking with latest render timing."""
        self.performance_metrics["total_renders"] += 1
        self._last_render_time = render_time

        if success:
            self.performance_metrics["total_render_time"] += render_time
            self.performance_metrics["average_render_time"] = (
                self.performance_metrics["total_render_time"]
                / self.performance_metrics["total_renders"]
            )

        self.performance_metrics["last_render_time"] = render_time

    def optimize(self, optimization_targets: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimizes template configuration and resources based on usage patterns
        and performance targets.
        """
        optimization_report = {
            "original_config": self.config.to_dict(),
            "optimizations_applied": [],
            "performance_improvement": {},
            "recommendations": [],
        }

        # Analyze current performance
        current_avg_time = self.performance_metrics.get("average_render_time", 0.0)
        target_time = optimization_targets.get("target_render_time_ms", 5.0) / 1000.0

        if current_avg_time > target_time:
            # Apply performance optimizations
            if self.config.quality_level != TemplateQuality.ULTRA_FAST:
                original_quality = self.config.quality_level
                self.config.quality_level = TemplateQuality.FAST
                optimization_report["optimizations_applied"].append(
                    f"Reduced quality from {original_quality.value} to {self.config.quality_level.value}"
                )

            # Enable aggressive caching
            if not self.config.caching_enabled:
                self.config.caching_enabled = True
                self._setup_caching()
                optimization_report["optimizations_applied"].append(
                    "Enabled template caching"
                )

            # Optimize performance options
            self.config.performance_options["memory_optimization"] = True
            self.config.performance_options["vectorized_operations"] = True
            optimization_report["optimizations_applied"].append(
                "Enabled performance optimizations"
            )

        # Memory optimization targets
        memory_target_mb = optimization_targets.get("memory_limit_mb", 50)
        estimated_memory = (
            self.config.grid_size.width * self.config.grid_size.height * 4
        ) / (1024 * 1024)

        if estimated_memory > memory_target_mb:
            optimization_report["recommendations"].append(
                f"Consider reducing grid size - estimated memory usage {estimated_memory:.1f}MB exceeds target {memory_target_mb}MB"
            )

        return optimization_report

    def cleanup(self) -> None:
        """Cleans up template resources and performs memory management."""
        try:
            # Call subclass-specific cleanup
            self._cleanup_template_resources()

            # Clear template cache
            if hasattr(self, "_cache"):
                self._cache.clear()

            # Reset performance metrics
            self.performance_metrics = {
                "total_renders": 0,
                "total_render_time": 0.0,
                "average_render_time": 0.0,
                "last_render_time": 0.0,
                "cache_hits": 0,
                "cache_misses": 0,
            }

            # Reset initialization status
            self._initialized = False

        except Exception as e:
            warnings.warn(f"Template cleanup encountered error: {e}")

    @abstractmethod
    def _cleanup_template_resources(self) -> None:
        """
        Abstract method for template-specific resource cleanup implemented by concrete subclasses.
        """
        pass

    def get_performance_metrics(
        self, include_cache_stats: bool = True
    ) -> Dict[str, Any]:
        """
        Returns comprehensive performance metrics including timing, resource usage,
        and optimization statistics.
        """
        metrics = self.performance_metrics.copy()

        if include_cache_stats and hasattr(self, "_cached_conversion"):
            cache_info = self._cached_conversion.cache_info()
            metrics["cache_stats"] = {
                "hits": cache_info.hits,
                "misses": cache_info.misses,
                "maxsize": cache_info.maxsize,
                "currsize": cache_info.currsize,
                "hit_rate": cache_info.hits
                / max(cache_info.hits + cache_info.misses, 1),
            }

        # Add configuration summary
        metrics["configuration"] = {
            "quality_level": self.config.quality_level.value,
            "grid_size": f"{self.config.grid_size.width}x{self.config.grid_size.height}",
            "caching_enabled": self.config.caching_enabled,
            "performance_target_ms": self.config.quality_level.performance_target_ms,
        }

        return metrics

    def validate_performance(
        self, test_scenarios: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Validates template performance against targets with detailed analysis and recommendations.
        """
        performance_analysis = {
            "meets_targets": True,
            "target_violations": [],
            "performance_summary": {},
            "recommendations": [],
        }

        # Execute performance tests
        test_results = []
        target_time = self.config.quality_level.performance_target_ms / 1000.0

        for scenario_name, scenario_config in test_scenarios.items():
            # Create test data
            grid_size = scenario_config.get("grid_size", (64, 64))
            test_field = np.random.rand(*grid_size).astype(np.float32)
            test_agent = Coordinates(grid_size[1] // 2, grid_size[0] // 2)
            test_source = Coordinates(grid_size[1] // 4, grid_size[0] // 4)

            # Measure performance
            start_time = time.perf_counter()
            try:
                self.render(test_field, test_agent, test_source)
                render_time = time.perf_counter() - start_time

                test_results.append(
                    {
                        "scenario": scenario_name,
                        "render_time": render_time,
                        "success": True,
                        "meets_target": render_time <= target_time,
                    }
                )

                if render_time > target_time:
                    performance_analysis["meets_targets"] = False
                    performance_analysis["target_violations"].append(
                        {
                            "scenario": scenario_name,
                            "actual_time_ms": render_time * 1000,
                            "target_time_ms": target_time * 1000,
                            "excess_time_ms": (render_time - target_time) * 1000,
                        }
                    )

            except Exception as e:
                test_results.append(
                    {
                        "scenario": scenario_name,
                        "render_time": -1,
                        "success": False,
                        "error": str(e),
                    }
                )
                performance_analysis["meets_targets"] = False

        # Generate performance summary
        successful_tests = [r for r in test_results if r["success"]]
        if successful_tests:
            render_times = [r["render_time"] for r in successful_tests]
            performance_analysis["performance_summary"] = {
                "average_render_time_ms": np.mean(render_times) * 1000,
                "max_render_time_ms": np.max(render_times) * 1000,
                "min_render_time_ms": np.min(render_times) * 1000,
                "success_rate": len(successful_tests) / len(test_results),
            }

        # Generate recommendations
        if not performance_analysis["meets_targets"]:
            performance_analysis["recommendations"].extend(
                [
                    "Consider reducing template quality level for better performance",
                    "Enable template caching if not already enabled",
                    "Reduce grid size for memory and performance optimization",
                    "Use vectorized operations in performance options",
                ]
            )

        return performance_analysis["meets_targets"], performance_analysis


class RGBTemplate(BaseRenderTemplate):
    """
    Specialized rendering template for RGB array generation providing high-performance
    programmatic visualization with optimized color application, marker rendering,
    and caching for <5ms generation targets.
    """

    def __init__(self, config: TemplateConfig):
        """
        Initialize RGB template with configuration and pre-compute optimized rendering resources.
        """
        super().__init__(config)

        # Initialize RGB-specific properties
        self._color_lookup_table = None
        self._agent_marker_pattern = None
        self._source_marker_pattern = None
        self._optimization_enabled = config.performance_options.get(
            "vectorized_operations", True
        )
        self._render_cache = {}

    def _initialize_template_resources(self) -> None:
        """
        Initialize RGB template resources including color tables, marker patterns,
        and performance optimization.
        """
        # Create color lookup table for concentration [0,1] to RGB [0,255] mapping
        self._color_lookup_table = np.linspace(0, 255, 256, dtype=RGB_DTYPE)

        # Pre-compute agent marker pattern (3×3 red square)
        agent_size = self.config.marker_config.get("agent_size", (3, 3))
        agent_color = self.config.marker_config.get("agent_color", [255, 0, 0])
        self._agent_marker_pattern = np.full(
            (*agent_size, 3), agent_color, dtype=RGB_DTYPE
        )

        # Pre-compute source marker pattern (5×5 white cross)
        source_size = self.config.marker_config.get("source_size", (5, 5))
        source_color = self.config.marker_config.get("source_color", [255, 255, 255])

        # Create cross pattern
        source_pattern = np.zeros((*source_size, 3), dtype=RGB_DTYPE)
        center_x, center_y = source_size[1] // 2, source_size[0] // 2

        # Horizontal line of cross
        source_pattern[center_y, :] = source_color
        # Vertical line of cross
        source_pattern[:, center_x] = source_color

        self._source_marker_pattern = source_pattern

        # Initialize render cache with size limit
        cache_size = TEMPLATE_CACHE_SIZE if self.config.caching_enabled else 0
        self._render_cache = {} if cache_size > 0 else None

    def _execute_render(
        self,
        concentration_field: np.ndarray,
        agent_position: Coordinates,
        source_position: Coordinates,
    ) -> RGBArray:
        """
        Execute RGB array rendering with optimized concentration mapping and marker application.
        """
        # Create cache key if caching enabled
        cache_key = None
        if self._render_cache is not None:
            cache_key = (
                hash(concentration_field.tobytes()),
                agent_position.x,
                agent_position.y,
                source_position.x,
                source_position.y,
            )

            if cache_key in self._render_cache:
                self.performance_metrics["cache_hits"] += 1
                return self._render_cache[cache_key].copy()

        # Convert concentration field to RGB using optimized color mapping
        height, width = concentration_field.shape
        rgb_array = np.zeros((height, width, 3), dtype=RGB_DTYPE)

        # Apply concentration background using vectorized operations
        self._apply_concentration_background(rgb_array, concentration_field)

        # Apply source marker with boundary checking
        self._apply_source_marker(rgb_array, source_position)

        # Apply agent marker with boundary checking (done last for visibility)
        self._apply_agent_marker(rgb_array, agent_position)

        # Cache result if caching enabled
        if self._render_cache is not None and cache_key is not None:
            if len(self._render_cache) >= TEMPLATE_CACHE_SIZE:
                # Remove oldest entry (simple FIFO replacement)
                oldest_key = next(iter(self._render_cache))
                del self._render_cache[oldest_key]

            self._render_cache[cache_key] = rgb_array.copy()
            self.performance_metrics["cache_misses"] += 1

        return rgb_array

    def _apply_concentration_background(
        self, rgb_array: np.ndarray, concentration_field: np.ndarray
    ) -> None:
        """
        Applies concentration field as grayscale background to RGB array using
        optimized color mapping.
        """
        # Normalize concentration values to [0, 1] range with clipping
        normalized_concentration = np.clip(concentration_field, 0.0, 1.0)

        # Convert to grayscale values [0, 255]
        grayscale_values = (normalized_concentration * 255).astype(RGB_DTYPE)

        # Apply to all RGB channels for grayscale visualization
        if self._optimization_enabled:
            # Vectorized assignment for all channels
            rgb_array[:, :, 0] = grayscale_values
            rgb_array[:, :, 1] = grayscale_values
            rgb_array[:, :, 2] = grayscale_values
        else:
            # Channel-by-channel assignment
            for channel in range(3):
                rgb_array[:, :, channel] = grayscale_values

    def _apply_agent_marker(
        self, rgb_array: np.ndarray, agent_position: Coordinates
    ) -> None:
        """
        Applies pre-computed agent marker pattern to RGB array at specified position
        with boundary handling.
        """
        # Convert agent position to array indices
        agent_y, agent_x = agent_position.to_array_index()

        # Calculate marker boundaries
        marker_h, marker_w = self._agent_marker_pattern.shape[:2]
        half_h, half_w = marker_h // 2, marker_w // 2

        # Calculate slice boundaries with grid boundary checking
        y_start = max(0, agent_y - half_h)
        y_end = min(rgb_array.shape[0], agent_y + half_h + 1)
        x_start = max(0, agent_x - half_w)
        x_end = min(rgb_array.shape[1], agent_x + half_w + 1)

        # Calculate corresponding marker pattern slice
        marker_y_start = max(0, half_h - agent_y)
        marker_y_end = marker_y_start + (y_end - y_start)
        marker_x_start = max(0, half_w - agent_x)
        marker_x_end = marker_x_start + (x_end - x_start)

        # Apply marker pattern with boundary clipping
        try:
            rgb_array[y_start:y_end, x_start:x_end] = self._agent_marker_pattern[
                marker_y_start:marker_y_end, marker_x_start:marker_x_end
            ]
        except (IndexError, ValueError):
            # Fallback for edge cases
            if 0 <= agent_y < rgb_array.shape[0] and 0 <= agent_x < rgb_array.shape[1]:
                rgb_array[agent_y, agent_x] = self.config.marker_config.get(
                    "agent_color", [255, 0, 0]
                )

    def _apply_source_marker(
        self, rgb_array: np.ndarray, source_position: Coordinates
    ) -> None:
        """
        Applies pre-computed source marker pattern to RGB array at specified position
        with cross shape.
        """
        # Convert source position to array indices
        source_y, source_x = source_position.to_array_index()

        # Calculate marker boundaries
        marker_h, marker_w = self._source_marker_pattern.shape[:2]
        half_h, half_w = marker_h // 2, marker_w // 2

        # Calculate slice boundaries with grid boundary checking
        y_start = max(0, source_y - half_h)
        y_end = min(rgb_array.shape[0], source_y + half_h + 1)
        x_start = max(0, source_x - half_w)
        x_end = min(rgb_array.shape[1], source_x + half_w + 1)

        # Calculate corresponding marker pattern slice
        marker_y_start = max(0, half_h - source_y)
        marker_y_end = marker_y_start + (y_end - y_start)
        marker_x_start = max(0, half_w - source_x)
        marker_x_end = marker_x_start + (x_end - x_start)

        # Apply cross pattern with boundary clipping
        try:
            # Only apply non-zero pixels from the cross pattern
            pattern_slice = self._source_marker_pattern[
                marker_y_start:marker_y_end, marker_x_start:marker_x_end
            ]

            # Create mask for cross pixels (non-zero pixels)
            cross_mask = np.any(pattern_slice > 0, axis=2)

            # Apply cross pattern only where mask is True
            rgb_slice = rgb_array[y_start:y_end, x_start:x_end]
            rgb_slice[cross_mask] = pattern_slice[cross_mask]

        except (IndexError, ValueError):
            # Fallback for edge cases - simple cross
            source_color = self.config.marker_config.get(
                "source_color", [255, 255, 255]
            )
            if (
                0 <= source_y < rgb_array.shape[0]
                and 0 <= source_x < rgb_array.shape[1]
            ):
                rgb_array[source_y, source_x] = source_color

    def generate_frame(
        self,
        concentration_field: np.ndarray,
        agent_position: Coordinates,
        source_position: Coordinates,
        use_cache: bool = True,
    ) -> RGBArray:
        """
        Public interface for RGB frame generation with caching and performance monitoring.
        """
        if use_cache and not self.config.caching_enabled:
            use_cache = False

        if not use_cache and self._render_cache is not None:
            # Temporarily disable cache for this render
            original_cache = self._render_cache
            self._render_cache = None
            try:
                result = self.render(
                    concentration_field, agent_position, source_position
                )
            finally:
                self._render_cache = original_cache
            return result
        else:
            return self.render(concentration_field, agent_position, source_position)

    def _cleanup_template_resources(self) -> None:
        """
        Clean up RGB template resources including lookup tables and cached patterns.
        """
        # Release pre-computed patterns and lookup tables
        self._color_lookup_table = None
        self._agent_marker_pattern = None
        self._source_marker_pattern = None

        # Clear render cache
        if self._render_cache is not None:
            self._render_cache.clear()

        # Reset optimization flags
        self._optimization_enabled = False

    def validate_performance(
        self, test_scenarios: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Validates RGB template performance against <5ms generation target with
        comprehensive analysis.
        """
        # Override target time for RGB template (5ms target)
        rgb_target_ms = PERFORMANCE_TARGET_RGB_RENDER_MS
        performance_analysis = {
            "meets_target": True,
            "target_violations": [],
            "performance_data": {},
            "optimization_recommendations": [],
        }

        test_results = []

        for scenario_name, scenario_config in test_scenarios.items():
            # Create test data with various grid sizes
            grid_sizes = scenario_config.get("grid_sizes", [(64, 64), (128, 128)])

            for grid_size in grid_sizes:
                test_field = np.random.rand(*grid_size).astype(np.float32)
                test_agent = Coordinates(grid_size[1] // 2, grid_size[0] // 2)
                test_source = Coordinates(grid_size[1] // 4, grid_size[0] // 4)

                # Multiple timing measurements for accuracy
                render_times = []
                for _ in range(10):
                    start_time = time.perf_counter()
                    try:
                        result = self.render(test_field, test_agent, test_source)
                        render_time = time.perf_counter() - start_time
                        render_times.append(render_time)

                        # Validate result format
                        if not isinstance(result, np.ndarray):
                            raise ValueError("RGB template must return numpy array")
                        if result.shape != (*grid_size, 3):
                            raise ValueError(f"Invalid RGB array shape: {result.shape}")
                        if result.dtype != RGB_DTYPE:
                            raise ValueError(f"Invalid RGB array dtype: {result.dtype}")

                    except Exception as e:
                        test_results.append(
                            {
                                "scenario": f"{scenario_name}_{grid_size[0]}x{grid_size[1]}",
                                "success": False,
                                "error": str(e),
                            }
                        )
                        break

                if render_times:
                    avg_time = np.mean(render_times)
                    max_time = np.max(render_times)

                    meets_target = avg_time <= (rgb_target_ms / 1000.0)

                    test_results.append(
                        {
                            "scenario": f"{scenario_name}_{grid_size[0]}x{grid_size[1]}",
                            "success": True,
                            "avg_render_time_ms": avg_time * 1000,
                            "max_render_time_ms": max_time * 1000,
                            "meets_target": meets_target,
                        }
                    )

                    if not meets_target:
                        performance_analysis["meets_target"] = False
                        performance_analysis["target_violations"].append(
                            {
                                "scenario": f"{scenario_name}_{grid_size[0]}x{grid_size[1]}",
                                "actual_time_ms": avg_time * 1000,
                                "target_time_ms": rgb_target_ms,
                                "excess_time_ms": (avg_time * 1000) - rgb_target_ms,
                            }
                        )

        # Analyze results and generate recommendations
        successful_tests = [r for r in test_results if r["success"]]
        if successful_tests:
            all_times = [r["avg_render_time_ms"] for r in successful_tests]
            performance_analysis["performance_data"] = {
                "overall_average_ms": np.mean(all_times),
                "overall_max_ms": np.max(all_times),
                "success_rate": len(successful_tests) / len(test_results),
                "target_compliance_rate": sum(
                    bool(r["meets_target"]) for r in successful_tests
                )
                / len(successful_tests),
            }

            # Generate optimization recommendations
            if not performance_analysis["meets_target"]:
                recommendations = [
                    "Enable template caching to reduce repeated computation",
                    "Use ULTRA_FAST or FAST quality levels for maximum performance",
                    "Enable vectorized operations in performance options",
                    "Consider reducing marker sizes for faster rendering",
                ]

                # Add memory-specific recommendations for large grids
                large_grid_tests = [
                    r
                    for r in successful_tests
                    if "128x128" in r["scenario"] or "256x256" in r["scenario"]
                ]
                if (
                    large_grid_tests
                    and np.mean([r["avg_render_time_ms"] for r in large_grid_tests])
                    > rgb_target_ms
                ):
                    recommendations.append(
                        "Consider grid size optimization for large environments"
                    )

                performance_analysis["optimization_recommendations"] = recommendations

        return performance_analysis["meets_target"], performance_analysis


class MatplotlibTemplate(BaseRenderTemplate):
    """
    Specialized rendering template for matplotlib human mode visualization providing
    interactive real-time display with backend management, figure optimization,
    and performance tuning for <50ms update targets.
    """

    def __init__(self, config: TemplateConfig):
        """
        Initialize matplotlib template with backend detection, figure setup,
        and interactive configuration.
        """
        super().__init__(config)

        # Initialize matplotlib-specific properties
        self.figure = None
        self.axes = None
        self.concentration_image = None
        self.agent_marker = None
        self.source_marker = None

        # Backend configuration
        self.active_backend = None
        self.backend_interactive = False
        self._backend_fallback_chain = config.backend_preferences.get(
            "fallback_chain", BACKEND_PRIORITY_LIST
        )

        # Animation and update configuration
        self._animation_interval = config.backend_preferences.get(
            "animation_interval", 0.001
        )

    def _initialize_template_resources(self) -> None:
        """
        Initialize matplotlib template resources including backend selection,
        figure creation, and axes configuration.
        """
        # Configure matplotlib backend with fallback support
        backend_success = self.configure_backend(
            self.config.backend_preferences.get("preferred_backend")
        )

        if not backend_success:
            raise RuntimeError("Failed to configure any matplotlib backend")

        # Create matplotlib figure with configured settings
        figsize = self.config.backend_preferences.get(
            "figsize", MATPLOTLIB_DEFAULT_FIGSIZE
        )
        self.figure, self.axes = plt.subplots(figsize=figsize)

        # Configure axes with mathematical coordinate system
        self.axes.set_title("Plume Navigation Environment")
        self.axes.set_xlabel("X Position")
        self.axes.set_ylabel("Y Position")

        # Initialize marker collections for efficient updates
        self.agent_marker = self.axes.scatter(
            [], [], c="red", s=100, marker="s", zorder=3
        )
        self.source_marker = self.axes.scatter(
            [],
            [],
            c="white",
            s=150,
            marker="+",
            linewidths=3,
            edgecolors="black",
            zorder=2,
        )

        # Test matplotlib functionality
        try:
            test_data = np.zeros((8, 8))
            test_image = self.axes.imshow(
                test_data,
                cmap="gray",
                origin="lower",
                extent=[0, 8, 0, 8],
                alpha=0.8,
                zorder=1,
            )
            self.concentration_image = test_image

            # Test figure drawing
            self.figure.canvas.draw()

        except Exception as e:
            warnings.warn(f"Matplotlib functionality test failed: {e}")
            raise RuntimeError("Matplotlib backend test failed")

    def _execute_render(
        self,
        concentration_field: np.ndarray,
        agent_position: Coordinates,
        source_position: Coordinates,
    ) -> None:
        """
        Execute matplotlib visualization update with concentration display and marker positioning.
        """
        try:
            # Update concentration image data if image exists
            if self.concentration_image is not None:
                self.concentration_image.set_data(concentration_field)

                # Configure normalization for [0,1] concentration range
                self.concentration_image.set_clim(vmin=0.0, vmax=1.0)

                # Update extent to match grid dimensions
                height, width = concentration_field.shape
                self.concentration_image.set_extent([0, width, 0, height])
            else:
                # Create concentration plot if not exists
                self.create_concentration_plot(concentration_field)

            # Update agent marker position
            self._update_agent_marker(agent_position)

            # Update source marker position
            self._update_source_marker(source_position)

            # Apply color scheme settings
            if self.color_scheme:
                try:
                    self.color_scheme.configure_matplotlib_axes(self.axes)
                except Exception as e:
                    warnings.warn(f"Color scheme application failed: {e}")

            # Refresh matplotlib display
            self.refresh_display()

        except Exception as e:
            # Handle backend errors with fallback mechanisms
            if "backend" in str(e).lower() or "display" in str(e).lower():
                fallback_success = self._handle_backend_error(e, self.active_backend)
                if not fallback_success:
                    raise RuntimeError(
                        f"Matplotlib rendering failed with all backends: {e}"
                    )
                else:
                    # Retry render with new backend
                    self._execute_render(
                        concentration_field, agent_position, source_position
                    )
            else:
                raise

    def create_concentration_plot(self, concentration_field: np.ndarray) -> None:
        """
        Creates initial matplotlib concentration field plot with proper scaling
        and coordinate system.
        """
        height, width = concentration_field.shape

        # Create concentration heatmap with grayscale colormap
        self.concentration_image = self.axes.imshow(
            concentration_field,
            cmap="gray",
            origin="lower",  # Mathematical coordinates
            extent=[0, width, 0, height],
            alpha=0.8,
            zorder=1,
        )

        # Configure axes limits and aspect ratio
        self.axes.set_xlim(0, width)
        self.axes.set_ylim(0, height)
        self.axes.set_aspect("equal")

        # Add colorbar for concentration scale reference
        try:
            if not hasattr(self, "_colorbar"):
                self._colorbar = self.figure.colorbar(
                    self.concentration_image, ax=self.axes, label="Concentration"
                )
        except Exception:
            pass  # Colorbar is optional

    def refresh_display(self, force_redraw: bool = False) -> None:
        """
        Refreshes matplotlib display with optimized update strategy for
        interactive performance.
        """
        try:
            if force_redraw or self.active_backend == "Agg":
                # Force complete redraw for non-interactive backends
                self.figure.canvas.draw()
            else:
                # Use efficient blitting for interactive backends if supported
                try:
                    # Attempt blitting for performance
                    if hasattr(self.figure.canvas, "copy_from_bbox"):
                        # Advanced blitting implementation would go here
                        # For simplicity, falling back to regular draw
                        self.figure.canvas.draw()
                    else:
                        self.figure.canvas.draw()
                except AttributeError:
                    self.figure.canvas.draw()

            # Allow GUI update with configured interval
            if self.backend_interactive:
                plt.pause(self._animation_interval)

        except Exception as e:
            warnings.warn(f"Display refresh failed: {e}")
            # Attempt fallback refresh method
            try:
                self.figure.canvas.flush_events()
            except Exception:
                pass  # Silent failure for refresh issues

    def configure_backend(self, preferred_backend: Optional[str] = None) -> bool:
        """
        Configures matplotlib backend with fallback chain and compatibility testing.
        """
        backend_candidates = []

        # Add preferred backend if specified
        if preferred_backend:
            backend_candidates.append(preferred_backend)

        # Add fallback chain
        backend_candidates.extend(self._backend_fallback_chain)

        # Remove duplicates while preserving order
        seen = set()
        unique_backends = []
        for backend in backend_candidates:
            if backend not in seen:
                unique_backends.append(backend)
                seen.add(backend)

        # Test each backend in priority order
        for backend in unique_backends:
            try:
                # Test backend availability
                plt.switch_backend(backend)
                current_backend = plt.get_backend()

                if current_backend.lower() == backend.lower():
                    # Backend successfully activated
                    self.active_backend = backend
                    self.backend_interactive = backend not in ["Agg", "svg", "pdf"]

                    # Test basic functionality
                    test_fig, test_ax = plt.subplots(figsize=(1, 1))
                    test_ax.plot([0, 1], [0, 1])
                    plt.close(test_fig)

                    return True

            except (ImportError, RuntimeError, Exception) as e:
                warnings.warn(f"Backend {backend} failed: {e}")
                continue

        # All backends failed
        warnings.warn("All matplotlib backends failed, falling back to Agg")
        try:
            plt.switch_backend("Agg")
            self.active_backend = "Agg"
            self.backend_interactive = False
            return True
        except Exception:
            return False

    def _update_agent_marker(self, agent_position: Coordinates) -> None:
        """
        Updates agent position marker with efficient matplotlib collection update.
        """
        try:
            if self.agent_marker is not None:
                # Convert agent coordinates to matplotlib plot coordinate system
                agent_x, agent_y = agent_position.x, agent_position.y

                # Update marker position using set_offsets
                self.agent_marker.set_offsets([[agent_x, agent_y]])

                # Ensure agent marker is visible with proper color and size
                agent_color = self.config.marker_config.get("agent_color", [255, 0, 0])
                # Convert to normalized color if needed
                if isinstance(agent_color, list) and max(agent_color) > 1:
                    agent_color = [c / 255.0 for c in agent_color]

                self.agent_marker.set_color(agent_color)

        except Exception as e:
            warnings.warn(f"Agent marker update failed: {e}")

    def _update_source_marker(self, source_position: Coordinates) -> None:
        """
        Updates source position marker with white cross symbol and high visibility.
        """
        try:
            if self.source_marker is not None:
                # Convert source coordinates to matplotlib plot coordinate system
                source_x, source_y = source_position.x, source_position.y

                # Update marker position
                self.source_marker.set_offsets([[source_x, source_y]])

                # Ensure high contrast visibility
                source_color = self.config.marker_config.get(
                    "source_color", [255, 255, 255]
                )
                if isinstance(source_color, list) and max(source_color) > 1:
                    source_color = [c / 255.0 for c in source_color]

                self.source_marker.set_color(source_color)

        except Exception as e:
            warnings.warn(f"Source marker update failed: {e}")

    def _handle_backend_error(self, error: Exception, failed_backend: str) -> bool:
        """
        Handles matplotlib backend errors with automatic fallback and recovery.
        """
        warnings.warn(f"Backend {failed_backend} error: {error}")

        # Get remaining backends from fallback chain
        remaining_backends = [
            b
            for b in self._backend_fallback_chain
            if b != failed_backend and b != self.active_backend
        ]

        # Try next available backend
        for backend in remaining_backends:
            try:
                plt.switch_backend(backend)
                self.active_backend = backend
                self.backend_interactive = backend not in ["Agg", "svg", "pdf"]

                # Reinitialize figure with new backend
                figsize = self.config.backend_preferences.get(
                    "figsize", MATPLOTLIB_DEFAULT_FIGSIZE
                )
                plt.close(self.figure)  # Close old figure
                self.figure, self.axes = plt.subplots(figsize=figsize)

                # Reconfigure axes
                self.axes.set_title("Plume Navigation Environment")
                self.axes.set_xlabel("X Position")
                self.axes.set_ylabel("Y Position")

                # Reinitialize markers
                self.agent_marker = self.axes.scatter(
                    [], [], c="red", s=100, marker="s", zorder=3
                )
                self.source_marker = self.axes.scatter(
                    [],
                    [],
                    c="white",
                    s=150,
                    marker="+",
                    linewidths=3,
                    edgecolors="black",
                    zorder=2,
                )
                self.concentration_image = None  # Will be recreated

                return True

            except Exception as e:
                warnings.warn(f"Fallback to backend {backend} failed: {e}")
                continue

        return False

    def _cleanup_template_resources(self) -> None:
        """
        Clean up matplotlib template resources including figure closure and
        backend disconnection.
        """
        try:
            # Close matplotlib figure and clear references
            if self.figure is not None:
                plt.close(self.figure)
                self.figure = None

            # Clear axes and marker objects
            self.axes = None
            self.concentration_image = None
            self.agent_marker = None
            self.source_marker = None

            # Reset backend state
            self.active_backend = None
            self.backend_interactive = False

            # Clear any matplotlib internal caches
            plt.clf()  # Clear current figure
            plt.cla()  # Clear current axes

        except Exception as e:
            warnings.warn(f"Matplotlib cleanup error: {e}")

    def validate_performance(
        self, test_scenarios: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Validates matplotlib template performance against <50ms update target with
        backend compatibility analysis.
        """
        # Target time for matplotlib human mode rendering (50ms)
        matplotlib_target_ms = PERFORMANCE_TARGET_HUMAN_RENDER_MS
        performance_analysis = {
            "meets_target": True,
            "target_violations": [],
            "performance_data": {},
            "backend_analysis": {},
            "optimization_recommendations": [],
        }

        test_results = []

        # Test current backend performance
        performance_analysis["backend_analysis"]["active_backend"] = self.active_backend
        performance_analysis["backend_analysis"][
            "interactive"
        ] = self.backend_interactive

        for scenario_name, scenario_config in test_scenarios.items():
            grid_sizes = scenario_config.get("grid_sizes", [(64, 64), (128, 128)])

            for grid_size in grid_sizes:
                test_field = np.random.rand(*grid_size).astype(np.float32)
                test_agent = Coordinates(grid_size[1] // 2, grid_size[0] // 2)
                test_source = Coordinates(grid_size[1] // 4, grid_size[0] // 4)

                # Multiple measurements for accuracy
                render_times = []
                for _ in range(5):  # Fewer iterations than RGB due to slower matplotlib
                    start_time = time.perf_counter()
                    try:
                        self.render(test_field, test_agent, test_source)
                        render_time = time.perf_counter() - start_time
                        render_times.append(render_time)

                    except Exception as e:
                        test_results.append(
                            {
                                "scenario": f"{scenario_name}_{grid_size[0]}x{grid_size[1]}",
                                "success": False,
                                "error": str(e),
                                "backend": self.active_backend,
                            }
                        )
                        break

                if render_times:
                    avg_time = np.mean(render_times)
                    max_time = np.max(render_times)

                    meets_target = avg_time <= (matplotlib_target_ms / 1000.0)

                    test_results.append(
                        {
                            "scenario": f"{scenario_name}_{grid_size[0]}x{grid_size[1]}",
                            "success": True,
                            "avg_render_time_ms": avg_time * 1000,
                            "max_render_time_ms": max_time * 1000,
                            "meets_target": meets_target,
                            "backend": self.active_backend,
                        }
                    )

                    if not meets_target:
                        performance_analysis["meets_target"] = False
                        performance_analysis["target_violations"].append(
                            {
                                "scenario": f"{scenario_name}_{grid_size[0]}x{grid_size[1]}",
                                "actual_time_ms": avg_time * 1000,
                                "target_time_ms": matplotlib_target_ms,
                                "excess_time_ms": (avg_time * 1000)
                                - matplotlib_target_ms,
                                "backend": self.active_backend,
                            }
                        )

        # Analyze results
        successful_tests = [r for r in test_results if r["success"]]
        if successful_tests:
            all_times = [r["avg_render_time_ms"] for r in successful_tests]
            performance_analysis["performance_data"] = {
                "overall_average_ms": np.mean(all_times),
                "overall_max_ms": np.max(all_times),
                "success_rate": len(successful_tests) / len(test_results),
                "target_compliance_rate": sum(
                    1 for r in successful_tests if r["meets_target"]
                )
                / len(successful_tests),
            }

            # Backend-specific analysis
            backend_times = {}
            for result in successful_tests:
                backend = result["backend"]
                if backend not in backend_times:
                    backend_times[backend] = []
                backend_times[backend].append(result["avg_render_time_ms"])

            performance_analysis["backend_analysis"]["performance_by_backend"] = {
                backend: {
                    "average_ms": np.mean(times),
                    "max_ms": np.max(times),
                    "sample_count": len(times),
                }
                for backend, times in backend_times.items()
            }

            # Generate optimization recommendations
            if not performance_analysis["meets_target"]:
                recommendations = [
                    "Consider using Agg backend for better performance in headless environments",
                    "Reduce animation interval for less frequent updates",
                    "Disable interactive features if not needed",
                    "Use smaller figure sizes to improve rendering speed",
                ]

                # Backend-specific recommendations
                if (
                    self.active_backend in ["TkAgg", "Qt5Agg"]
                    and not self.backend_interactive
                ):
                    recommendations.append(
                        "Switch to Agg backend for non-interactive use"
                    )

                if np.mean(all_times) > matplotlib_target_ms * 2:
                    recommendations.append(
                        "Consider fallback to RGB array mode for critical performance"
                    )

                performance_analysis["optimization_recommendations"] = recommendations

        return performance_analysis["meets_target"], performance_analysis


# Factory Functions for Template Creation


def create_rgb_template(
    grid_size: GridSize,
    color_scheme: Optional[ColorScheme] = None,
    quality_level: Optional[TemplateQuality] = None,
    template_options: Dict[str, Any] = None,
) -> RGBTemplate:
    """
    Factory function to create optimized RGB array rendering template with performance
    tuning, color scheme integration, and customizable marker configurations for
    programmatic visualization.
    """
    # Validate grid_size dimensions and calculate memory requirements
    if grid_size.width <= 0 or grid_size.height <= 0:
        raise ValueError("Grid size dimensions must be positive")

    memory_estimate_mb = (grid_size.width * grid_size.height * 3) / (1024 * 1024)
    if memory_estimate_mb > 500:  # Reasonable upper limit
        warnings.warn(
            f"Large grid size may require {memory_estimate_mb:.1f}MB of memory"
        )

    # Initialize color scheme using default if none provided, optimize for RGB mode
    if color_scheme is None:
        color_scheme = create_default_scheme()

    # Optimize color scheme for RGB rendering mode
    try:
        color_scheme.optimize_for_render_mode(RenderMode.RGB_ARRAY)
    except Exception as e:
        warnings.warn(f"Color scheme optimization failed: {e}")

    # Configure quality level with performance targets
    if quality_level is None:
        quality_level = TemplateQuality.STANDARD

    # Apply template options with defaults
    if template_options is None:
        template_options = {}

    # Merge with RGB template defaults
    merged_options = DEFAULT_RGB_TEMPLATE_CONFIG.copy()
    merged_options.update(template_options)

    # Pre-compute template resources and optimization settings
    marker_config = {
        "agent_size": merged_options.get("agent_marker_size", (3, 3)),
        "agent_color": [255, 0, 0],  # Red for high visibility
        "source_size": merged_options.get("source_marker_size", (5, 5)),
        "source_color": [255, 255, 255],  # White for goal identification
    }

    performance_options = {
        "enable_caching": merged_options.get("caching_enabled", True),
        "vectorized_operations": merged_options.get("performance_optimized", True),
        "memory_optimization": True,
    }

    # Configure template performance monitoring and validation
    backend_preferences = {
        "preferred_backend": "Agg",  # RGB templates don't need interactive backends
        "fallback_chain": ["Agg"],
        "headless_compatible": True,
    }

    # Create template configuration
    config = TemplateConfig(
        grid_size=grid_size,
        color_scheme=color_scheme,
        quality_level=quality_level,
        marker_config=marker_config,
        performance_options=performance_options,
        backend_preferences=backend_preferences,
        caching_enabled=merged_options.get("caching_enabled", True),
        custom_parameters=merged_options,
    )

    # Initialize and validate RGB template
    template = RGBTemplate(config)

    # Configure performance monitoring based on quality level
    if not template.initialize():
        raise RuntimeError("RGB template initialization failed")

    # Validate against RGB rendering performance targets (<5ms)
    target_time = PERFORMANCE_TARGET_RGB_RENDER_MS / 1000.0  # Convert to seconds
    if quality_level.performance_target_ms / 1000.0 > target_time:
        warnings.warn(
            f"Quality level target {quality_level.performance_target_ms}ms exceeds "
            f"RGB rendering target {PERFORMANCE_TARGET_RGB_RENDER_MS}ms"
        )

    return template


def create_matplotlib_template(
    grid_size: GridSize,
    color_scheme: Optional[ColorScheme] = None,
    preferred_backend: Optional[str] = None,
    template_options: Dict[str, Any] = None,
) -> MatplotlibTemplate:
    """
    Factory function to create matplotlib-based human mode visualization template with
    backend management, interactive optimization, and cross-platform compatibility for
    research workflows.
    """
    # Validate grid_size and calculate optimal figure sizing
    if grid_size.width <= 0 or grid_size.height <= 0:
        raise ValueError("Grid size dimensions must be positive")

    # Calculate optimal figure size based on grid dimensions
    aspect_ratio = grid_size.width / grid_size.height
    base_size = 8  # Base figure size

    if aspect_ratio > 1.5:
        figsize = (base_size * 1.2, base_size / 1.2)
    elif aspect_ratio < 0.67:
        figsize = (base_size / 1.2, base_size * 1.2)
    else:
        figsize = (base_size, base_size)

    # Initialize color scheme optimized for matplotlib visualization
    if color_scheme is None:
        color_scheme = create_default_scheme()

    # Optimize for matplotlib rendering with colormap integration
    try:
        color_scheme.optimize_for_render_mode(RenderMode.HUMAN)
    except Exception as e:
        warnings.warn(f"Color scheme optimization for matplotlib failed: {e}")

    # Configure matplotlib backend selection with fallback chain
    backend_chain = BACKEND_PRIORITY_LIST.copy()
    if preferred_backend and preferred_backend not in backend_chain:
        backend_chain.insert(0, preferred_backend)
    elif preferred_backend:
        # Move preferred backend to front
        backend_chain.remove(preferred_backend)
        backend_chain.insert(0, preferred_backend)

    # Apply template options with matplotlib defaults
    if template_options is None:
        template_options = {}

    merged_options = DEFAULT_MATPLOTLIB_TEMPLATE_CONFIG.copy()
    merged_options.update(template_options)
    merged_options["figsize"] = figsize  # Override with calculated size

    # Configure backend preferences with compatibility
    backend_preferences = {
        "preferred_backend": preferred_backend or backend_chain[0],
        "fallback_chain": backend_chain,
        "headless_compatible": merged_options.get("backend_fallback", True),
        "figsize": figsize,
        "animation_interval": merged_options.get("animation_interval", 0.001),
    }

    # Initialize matplotlib figure and axes templates
    marker_config = {
        "agent_size": (100,),  # Matplotlib scatter size
        "agent_color": [1.0, 0.0, 0.0],  # Normalized RGB for matplotlib
        "source_size": (150,),
        "source_color": [1.0, 1.0, 1.0],
    }

    performance_options = {
        "enable_caching": merged_options.get(
            "caching_enabled", False
        ),  # Less useful for matplotlib
        "vectorized_operations": True,
        "memory_optimization": False,  # Matplotlib handles its own optimization
        "interactive_updates": merged_options.get("interactive_updates", True),
    }

    # Configure interactive update mechanisms targeting <50ms performance
    quality_level = TemplateQuality.STANDARD  # Balanced for interactive use
    target_time = PERFORMANCE_TARGET_HUMAN_RENDER_MS

    if target_time < quality_level.performance_target_ms:
        quality_level = TemplateQuality.FAST

    # Create matplotlib template configuration
    config = TemplateConfig(
        grid_size=grid_size,
        color_scheme=color_scheme,
        quality_level=quality_level,
        marker_config=marker_config,
        performance_options=performance_options,
        backend_preferences=backend_preferences,
        caching_enabled=merged_options.get("caching_enabled", False),
        custom_parameters=merged_options,
    )

    # Test backend availability and configure fallback mechanisms
    template = MatplotlibTemplate(config)

    # Initialize with backend compatibility and optimization settings
    if not template.initialize():
        warnings.warn(
            "Matplotlib template initialization failed - attempting fallback configuration"
        )

        # Try with Agg backend as ultimate fallback
        config.backend_preferences["preferred_backend"] = "Agg"
        config.backend_preferences["fallback_chain"] = ["Agg"]

        template = MatplotlibTemplate(config)
        if not template.initialize():
            raise RuntimeError(
                "Matplotlib template initialization failed with all backends"
            )

    return template


def create_custom_template(
    config: TemplateConfig,
    template_type: str,
    custom_parameters: Dict[str, Any] = None,
    validate_performance: bool = True,
) -> Union[RGBTemplate, MatplotlibTemplate]:
    """
    Advanced factory function for creating customized rendering templates with
    user-defined configurations, validation, and performance optimization for
    specialized visualization requirements.
    """
    # Validate template_type against supported types
    supported_types = ["rgb", "matplotlib", "hybrid"]
    if template_type not in supported_types:
        raise ValueError(
            f"Template type '{template_type}' not supported. Use one of: {supported_types}"
        )

    # Parse and validate TemplateConfig for consistency
    is_valid, validation_errors = config.validate(strict_mode=True)
    if not is_valid:
        raise ValueError(f"Invalid template configuration: {validation_errors}")

    # Apply custom_parameters with validation against template type requirements
    if custom_parameters:
        # Validate custom parameters based on template type
        if template_type == "rgb":
            rgb_params = [
                "agent_marker_size",
                "source_marker_size",
                "performance_optimized",
            ]
            invalid_params = [
                p
                for p in custom_parameters.keys()
                if p not in rgb_params and not p.startswith("custom_")
            ]
            if invalid_params:
                warnings.warn(
                    f"Custom parameters {invalid_params} may not apply to RGB templates"
                )

        elif template_type == "matplotlib":
            mpl_params = [
                "figsize",
                "backend_fallback",
                "interactive_updates",
                "animation_interval",
            ]
            invalid_params = [
                p
                for p in custom_parameters.keys()
                if p not in mpl_params and not p.startswith("custom_")
            ]
            if invalid_params:
                warnings.warn(
                    f"Custom parameters {invalid_params} may not apply to matplotlib templates"
                )

        # Merge custom parameters into config
        config.custom_parameters.update(custom_parameters)

    # Create template instance using appropriate factory based on template_type
    if template_type == "rgb":
        template = RGBTemplate(config)
    elif template_type == "matplotlib":
        template = MatplotlibTemplate(config)
    elif template_type == "hybrid":
        # Hybrid template implementation - for future extension
        raise NotImplementedError(
            "Hybrid templates not implemented in proof-of-life version"
        )
    else:
        raise ValueError(f"Unknown template type: {template_type}")

    # Apply custom configuration and specialized settings
    try:
        if not template.initialize():
            raise RuntimeError(f"Custom {template_type} template initialization failed")
    except Exception as e:
        raise RuntimeError(f"Custom template creation failed: {e}")

    # Perform performance validation if requested
    if validate_performance:
        test_scenarios = {
            "basic_test": {"grid_sizes": [(32, 32), (64, 64)]},
            "custom_test": {
                "grid_sizes": [(config.grid_size.width, config.grid_size.height)]
            },
        }

        meets_targets, performance_data = template.validate_performance(test_scenarios)

        if not meets_targets:
            warnings.warn(
                f"Custom template does not meet performance targets: {performance_data.get('target_violations', [])}"
            )

            # Provide optimization recommendations
            recommendations = performance_data.get("optimization_recommendations", [])
            if recommendations:
                warnings.warn(
                    f"Performance optimization recommendations: {recommendations}"
                )

    return template


def register_template(
    template_name: str,
    template: Union[RGBTemplate, MatplotlibTemplate],
    metadata: Dict[str, Any] = None,
    allow_override: bool = False,
) -> bool:
    """
    Registers custom rendering template in global registry with validation,
    conflict checking, and metadata management for template discovery and reuse.
    """
    # Validate template_name uniqueness and naming convention
    if not isinstance(template_name, str) or len(template_name) < 3:
        raise ValueError("Template name must be a string with at least 3 characters")

    if not template_name.replace("_", "").replace("-", "").isalnum():
        raise ValueError(
            "Template name must contain only alphanumeric characters, hyphens, and underscores"
        )

    # Check for existing template
    if template_name in TEMPLATE_REGISTRY and not allow_override:
        warnings.warn(
            f"Template '{template_name}' already registered. Use allow_override=True to replace."
        )
        return False

    # Validate template instance completeness and performance
    if not isinstance(template, (RGBTemplate, MatplotlibTemplate)):
        raise TypeError(
            "Template must be an instance of RGBTemplate or MatplotlibTemplate"
        )

    # Ensure template is properly initialized
    if not template._initialized:
        try:
            if not template.initialize():
                raise RuntimeError("Template initialization failed during registration")
        except Exception as e:
            warnings.warn(f"Template validation failed: {e}")
            return False

    # Check metadata completeness
    if metadata is None:
        metadata = {}

    required_metadata = ["description", "performance_targets", "compatibility"]
    missing_metadata = [field for field in required_metadata if field not in metadata]
    if missing_metadata:
        warnings.warn(f"Missing recommended metadata fields: {missing_metadata}")

    # Test template functionality with sample rendering
    try:
        test_field = np.random.rand(32, 32).astype(np.float32)
        test_agent = Coordinates(16, 16)
        test_source = Coordinates(8, 8)

        result = template.render(test_field, test_agent, test_source)

        # Validate result based on template type
        if isinstance(template, RGBTemplate):
            if not isinstance(result, np.ndarray) or result.shape != (32, 32, 3):
                raise ValueError("RGB template test failed - invalid result format")
        elif isinstance(template, MatplotlibTemplate):
            if result is not None:
                warnings.warn("Matplotlib template returned non-None result")

    except Exception as e:
        warnings.warn(f"Template functionality test failed: {e}")
        return False

    # Handle naming conflicts with override checking
    if template_name in TEMPLATE_REGISTRY:
        if allow_override:
            # Clean up old template
            old_template = TEMPLATE_REGISTRY[template_name]["template"]
            try:
                old_template.cleanup()
            except Exception:
                pass
        else:
            return False

    # Add template to registry with metadata
    registry_entry = {
        "template": template,
        "metadata": {
            "description": metadata.get(
                "description", f"Custom {type(template).__name__}"
            ),
            "template_type": type(template).__name__,
            "grid_size": f"{template.config.grid_size.width}x{template.config.grid_size.height}",
            "quality_level": template.config.quality_level.value,
            "performance_targets": metadata.get("performance_targets", {}),
            "compatibility": metadata.get("compatibility", "Unknown"),
            "registration_time": time.time(),
            "usage_count": 0,
        },
        "performance_history": [],
        "usage_statistics": {
            "total_renders": 0,
            "total_render_time": 0.0,
            "average_render_time": 0.0,
        },
    }

    # Update registration
    TEMPLATE_REGISTRY[template_name] = registry_entry

    return True


def get_template_registry(
    template_type_filter: Optional[str] = None,
    quality_filter: Optional[str] = None,
    include_performance_data: bool = False,
) -> Dict[str, Any]:
    """
    Returns comprehensive template registry with filtering capabilities, performance
    characteristics, and usage recommendations for template discovery and selection.
    """
    # Retrieve all registered templates
    if not TEMPLATE_REGISTRY:
        return {
            "templates": {},
            "summary": {
                "total_templates": 0,
                "available_types": [],
                "quality_levels": [],
            },
        }

    filtered_templates = {}

    # Apply filters
    for template_name, entry in TEMPLATE_REGISTRY.items():
        template = entry["template"]
        metadata = entry["metadata"]

        # Apply template_type_filter
        if template_type_filter:
            valid_types = ["rgb", "matplotlib", "custom"]
            if template_type_filter not in valid_types:
                warnings.warn(
                    f"Invalid template type filter '{template_type_filter}'. Valid options: {valid_types}"
                )
                continue

            if template_type_filter == "rgb" and not isinstance(template, RGBTemplate):
                continue
            elif template_type_filter == "matplotlib" and not isinstance(
                template, MatplotlibTemplate
            ):
                continue
            elif template_type_filter == "custom" and template_name not in [
                "rgb_default",
                "matplotlib_default",
            ]:
                # Custom filter logic could be expanded
                pass

        # Apply quality_filter
        if quality_filter:
            valid_qualities = [q.value for q in TemplateQuality]
            if quality_filter not in valid_qualities:
                warnings.warn(
                    f"Invalid quality filter '{quality_filter}'. Valid options: {valid_qualities}"
                )
                continue

            if metadata.get("quality_level") != quality_filter:
                continue

        # Include template in filtered results
        template_info = {
            "template_type": metadata["template_type"],
            "description": metadata["description"],
            "grid_size": metadata["grid_size"],
            "quality_level": metadata["quality_level"],
            "compatibility": metadata["compatibility"],
            "usage_count": metadata["usage_count"],
            "registration_time": metadata["registration_time"],
        }

        # Include performance data if requested
        if include_performance_data:
            performance_metrics = template.get_performance_metrics()
            template_info["performance_data"] = {
                "average_render_time_ms": performance_metrics.get(
                    "average_render_time", 0.0
                )
                * 1000,
                "total_renders": performance_metrics.get("total_renders", 0),
                "cache_hit_rate": 0.0,
            }

            # Add cache statistics if available
            cache_stats = performance_metrics.get("cache_stats")
            if cache_stats:
                template_info["performance_data"]["cache_hit_rate"] = cache_stats.get(
                    "hit_rate", 0.0
                )

        filtered_templates[template_name] = template_info

    # Generate summary statistics
    all_types = [entry["template_type"] for entry in filtered_templates.values()]
    all_qualities = [entry["quality_level"] for entry in filtered_templates.values()]

    summary = {
        "total_templates": len(filtered_templates),
        "available_types": list(set(all_types)),
        "quality_levels": list(set(all_qualities)),
        "filters_applied": {
            "template_type": template_type_filter,
            "quality_filter": quality_filter,
            "performance_data_included": include_performance_data,
        },
    }

    # Sort templates by performance and popularity
    sorted_templates = dict(
        sorted(
            filtered_templates.items(),
            key=lambda item: (item[1]["usage_count"], -item[1]["registration_time"]),
            reverse=True,
        )
    )

    return {
        "templates": sorted_templates,
        "summary": summary,
        "recommendations": _generate_template_recommendations(sorted_templates),
    }


def _generate_template_recommendations(templates: Dict[str, Any]) -> List[str]:
    """Generate usage recommendations based on available templates."""
    recommendations = []

    if not templates:
        recommendations.append(
            "No templates available - consider creating templates with factory functions"
        )
        return recommendations

    # Recommend based on template types available
    rgb_templates = [
        name
        for name, info in templates.items()
        if info["template_type"] == "RGBTemplate"
    ]
    matplotlib_templates = [
        name
        for name, info in templates.items()
        if info["template_type"] == "MatplotlibTemplate"
    ]

    if rgb_templates:
        most_used_rgb = max(rgb_templates, key=lambda x: templates[x]["usage_count"])
        recommendations.append(
            f"For programmatic rendering, consider using '{most_used_rgb}' (most popular RGB template)"
        )

    if matplotlib_templates:
        most_used_mpl = max(
            matplotlib_templates, key=lambda x: templates[x]["usage_count"]
        )
        recommendations.append(
            f"For interactive visualization, consider using '{most_used_mpl}' (most popular matplotlib template)"
        )

    # Quality level recommendations
    quality_levels = [info["quality_level"] for info in templates.values()]
    if "ultra_fast" in quality_levels:
        recommendations.append(
            "ULTRA_FAST quality templates available for maximum performance requirements"
        )

    if "premium" in quality_levels:
        recommendations.append(
            "PREMIUM quality templates available for highest visual quality"
        )

    return recommendations


def optimize_template(
    template: Union[RGBTemplate, MatplotlibTemplate],
    optimization_targets: Dict[str, Any],
    usage_statistics: Dict[str, Any] = None,
) -> Tuple[Union[RGBTemplate, MatplotlibTemplate], Dict[str, Any]]:
    """
    Optimizes existing rendering template based on usage patterns, system performance
    characteristics, and target requirements for maximum efficiency.
    """
    # Analyze current template performance characteristics
    current_metrics = template.get_performance_metrics()
    optimization_report = {
        "original_performance": current_metrics.copy(),
        "optimizations_applied": [],
        "performance_improvements": {},
        "recommendations": [],
        "configuration_changes": {},
    }

    # Create optimized configuration copy
    optimized_config = template.config.clone()

    # Identify optimization opportunities from targets
    target_render_time_ms = optimization_targets.get("target_render_time_ms", 5.0)
    target_memory_mb = optimization_targets.get("target_memory_mb", 50.0)
    optimization_targets.get("target_cache_hit_rate", 0.8)

    current_render_time = (
        current_metrics.get("average_render_time", 0.0) * 1000
    )  # Convert to ms

    # Apply optimization targets and strategies
    optimizations_needed = current_render_time > target_render_time_ms

    if optimizations_needed:
        # Quality level optimization
        current_quality = optimized_config.quality_level
        if current_quality != TemplateQuality.ULTRA_FAST:
            if target_render_time_ms <= 1.0:
                optimized_config.quality_level = TemplateQuality.ULTRA_FAST
                optimization_report["optimizations_applied"].append(
                    f"Reduced quality from {current_quality.value} to ultra_fast for <1ms target"
                )
            elif target_render_time_ms <= 3.0:
                optimized_config.quality_level = TemplateQuality.FAST
                optimization_report["optimizations_applied"].append(
                    f"Reduced quality from {current_quality.value} to fast for <3ms target"
                )

        # Caching optimization based on usage patterns
        if usage_statistics:
            render_frequency = usage_statistics.get("renders_per_second", 1.0)
            repeat_renders = usage_statistics.get("repeated_renders", 0.1)

            if repeat_renders > 0.3 and not optimized_config.caching_enabled:
                optimized_config.caching_enabled = True
                optimization_report["optimizations_applied"].append(
                    "Enabled caching due to high repeat render rate"
                )

        # Performance options optimization
        if not optimized_config.performance_options.get("vectorized_operations", True):
            optimized_config.performance_options["vectorized_operations"] = True
            optimization_report["optimizations_applied"].append(
                "Enabled vectorized operations"
            )

        if not optimized_config.performance_options.get("memory_optimization", False):
            optimized_config.performance_options["memory_optimization"] = True
            optimization_report["optimizations_applied"].append(
                "Enabled memory optimization"
            )

    # Memory optimization
    estimated_memory = (
        optimized_config.grid_size.width * optimized_config.grid_size.height * 3
    ) / (1024 * 1024)
    if estimated_memory > target_memory_mb:
        # Suggest grid size reduction
        reduction_factor = target_memory_mb / estimated_memory
        suggested_width = int(
            optimized_config.grid_size.width * np.sqrt(reduction_factor)
        )
        suggested_height = int(
            optimized_config.grid_size.height * np.sqrt(reduction_factor)
        )

        optimization_report["recommendations"].append(
            f"Consider reducing grid size to {suggested_width}x{suggested_height} to meet {target_memory_mb}MB target"
        )

    # Configure caching strategies based on usage patterns
    if usage_statistics and optimized_config.caching_enabled:
        access_pattern = usage_statistics.get("access_pattern", "random")
        if access_pattern == "sequential":
            # Sequential access benefits less from caching
            cache_size = min(TEMPLATE_CACHE_SIZE, 5)
        elif access_pattern == "repetitive":
            # Repetitive access benefits greatly from caching
            cache_size = TEMPLATE_CACHE_SIZE
        else:
            cache_size = TEMPLATE_CACHE_SIZE // 2

        # Note: Cache size adjustment would need to be implemented in template initialization
        optimization_report["configuration_changes"]["cache_size"] = cache_size

    # Create optimized template instance
    try:
        if isinstance(template, RGBTemplate):
            optimized_template = RGBTemplate(optimized_config)
        else:
            optimized_template = MatplotlibTemplate(optimized_config)

        # Initialize optimized template
        if not optimized_template.initialize():
            raise RuntimeError("Optimized template initialization failed")

        # Test optimized template performance
        test_field = np.random.rand(64, 64).astype(np.float32)
        test_agent = Coordinates(32, 32)
        test_source = Coordinates(16, 16)

        # Measure optimization improvement
        start_time = time.perf_counter()
        optimized_template.render(test_field, test_agent, test_source)
        optimized_render_time = time.perf_counter() - start_time

        # Calculate improvement
        if current_render_time > 0:
            improvement_percent = (
                (current_render_time / 1000 - optimized_render_time)
                / (current_render_time / 1000)
            ) * 100
            optimization_report["performance_improvements"][
                "render_time_improvement_percent"
            ] = improvement_percent
            optimization_report["performance_improvements"]["new_render_time_ms"] = (
                optimized_render_time * 1000
            )

    except Exception as e:
        warnings.warn(f"Template optimization failed: {e}")
        # Return original template with report
        return template, optimization_report

    # Generate comprehensive optimization report
    if not optimization_report["optimizations_applied"]:
        optimization_report["recommendations"].append(
            "Template already well-optimized for current targets"
        )

    # Add general recommendations
    optimization_report["recommendations"].extend(
        [
            "Monitor performance metrics regularly to identify optimization opportunities",
            "Adjust quality levels based on use case requirements",
            "Enable caching for repetitive rendering scenarios",
            "Consider template-specific optimizations based on usage patterns",
        ]
    )

    return optimized_template, optimization_report


def validate_template_performance(
    template: Union[RGBTemplate, MatplotlibTemplate],
    test_scenarios: Dict[str, Any],
    strict_validation: bool = False,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Comprehensive performance validation function for rendering templates including
    timing benchmarks, memory usage analysis, and compatibility testing.
    """
    # Initialize validation report
    performance_report = {
        "validation_passed": True,
        "template_info": {
            "type": type(template).__name__,
            "quality_level": template.config.quality_level.value,
            "grid_size": f"{template.config.grid_size.width}x{template.config.grid_size.height}",
            "caching_enabled": template.config.caching_enabled,
        },
        "test_results": {},
        "performance_analysis": {},
        "compatibility_results": {},
        "recommendations": [],
    }

    # Determine performance targets based on template type
    if isinstance(template, RGBTemplate):
        target_ms = PERFORMANCE_TARGET_RGB_RENDER_MS
        template_type = "RGB"
    else:
        target_ms = PERFORMANCE_TARGET_HUMAN_RENDER_MS
        template_type = "Matplotlib"

    performance_report["performance_analysis"]["target_time_ms"] = target_ms

    # Execute template rendering benchmarks across test scenarios
    all_test_results = []

    for scenario_name, scenario_config in test_scenarios.items():
        scenario_results = {"scenario_name": scenario_name, "tests": [], "summary": {}}

        # Get grid sizes for testing
        grid_sizes = scenario_config.get("grid_sizes", [(64, 64), (128, 128)])
        iterations = scenario_config.get(
            "iterations", 10 if isinstance(template, RGBTemplate) else 5
        )

        for grid_size in grid_sizes:
            # Create test data
            test_field = np.random.rand(*grid_size).astype(np.float32)
            test_agent = Coordinates(grid_size[1] // 2, grid_size[0] // 2)
            test_source = Coordinates(grid_size[1] // 4, grid_size[0] // 4)

            # Execute multiple iterations for statistical accuracy
            render_times = []
            memory_usage = []
            success_count = 0

            for iteration in range(iterations):
                start_time = time.perf_counter()
                try:
                    # Measure memory usage if possible
                    import psutil

                    process = psutil.Process()
                    memory_before = process.memory_info().rss / 1024 / 1024  # MB

                    # Execute render
                    result = template.render(test_field, test_agent, test_source)

                    # Measure timing and memory
                    render_time = time.perf_counter() - start_time
                    memory_after = process.memory_info().rss / 1024 / 1024  # MB

                    render_times.append(render_time)
                    memory_usage.append(memory_after - memory_before)
                    success_count += 1

                    # Validate result format and quality
                    if isinstance(template, RGBTemplate):
                        if not isinstance(result, np.ndarray):
                            raise ValueError("RGB template must return numpy array")
                        if result.shape != (*grid_size, 3):
                            raise ValueError(f"Invalid RGB shape: {result.shape}")
                        if result.dtype != RGB_DTYPE:
                            raise ValueError(f"Invalid RGB dtype: {result.dtype}")

                except ImportError:
                    # psutil not available, skip memory measurement
                    render_time = time.perf_counter() - start_time
                    render_times.append(render_time)
                    success_count += 1

                except Exception as e:
                    # Test failed
                    performance_report["validation_passed"] = False
                    render_times.append(-1)  # Mark as failed

                    if strict_validation:
                        performance_report["test_results"][
                            f"{scenario_name}_{grid_size[0]}x{grid_size[1]}"
                        ] = {"success": False, "error": str(e), "iteration": iteration}
                        break

            # Analyze results for this grid size
            successful_times = [t for t in render_times if t > 0]

            if successful_times:
                test_result = {
                    "grid_size": f"{grid_size[0]}x{grid_size[1]}",
                    "success_rate": success_count / iterations,
                    "average_time_ms": np.mean(successful_times) * 1000,
                    "max_time_ms": np.max(successful_times) * 1000,
                    "min_time_ms": np.min(successful_times) * 1000,
                    "std_time_ms": np.std(successful_times) * 1000,
                    "meets_target": np.mean(successful_times) * 1000 <= target_ms,
                    "iterations": iterations,
                }

                # Add memory analysis if available
                if memory_usage:
                    test_result["memory_analysis"] = {
                        "average_memory_mb": np.mean(memory_usage),
                        "max_memory_mb": np.max(memory_usage),
                        "memory_efficient": np.mean(memory_usage)
                        < 10,  # Reasonable threshold
                    }

                scenario_results["tests"].append(test_result)
                all_test_results.append(test_result)

                # Check performance against targets
                if not test_result["meets_target"]:
                    performance_report["validation_passed"] = False
            else:
                # All tests failed for this grid size
                performance_report["validation_passed"] = False
                scenario_results["tests"].append(
                    {
                        "grid_size": f"{grid_size[0]}x{grid_size[1]}",
                        "success_rate": 0.0,
                        "error": "All iterations failed",
                    }
                )

        # Summarize scenario results
        successful_tests = [
            t for t in scenario_results["tests"] if t.get("success_rate", 0) > 0.5
        ]
        if successful_tests:
            scenario_results["summary"] = {
                "overall_success_rate": np.mean(
                    [t["success_rate"] for t in successful_tests]
                ),
                "average_time_ms": np.mean(
                    [t["average_time_ms"] for t in successful_tests]
                ),
                "target_compliance_rate": sum(
                    1 for t in successful_tests if t["meets_target"]
                )
                / len(successful_tests),
            }

        performance_report["test_results"][scenario_name] = scenario_results

    # Generate comprehensive performance analysis
    if all_test_results:
        successful_results = [
            r for r in all_test_results if r.get("success_rate", 0) > 0.8
        ]

        if successful_results:
            performance_report["performance_analysis"].update(
                {
                    "overall_average_time_ms": np.mean(
                        [r["average_time_ms"] for r in successful_results]
                    ),
                    "overall_max_time_ms": np.max(
                        [r["max_time_ms"] for r in successful_results]
                    ),
                    "overall_success_rate": np.mean(
                        [r["success_rate"] for r in successful_results]
                    ),
                    "target_compliance_rate": sum(
                        1 for r in successful_results if r["meets_target"]
                    )
                    / len(successful_results),
                    "performance_consistency": {
                        "std_deviation_ms": np.std(
                            [r["average_time_ms"] for r in successful_results]
                        ),
                        "coefficient_of_variation": np.std(
                            [r["average_time_ms"] for r in successful_results]
                        )
                        / np.mean([r["average_time_ms"] for r in successful_results]),
                    },
                }
            )

    # Test backend compatibility for matplotlib templates
    if isinstance(template, MatplotlibTemplate):
        compatibility_results = {}

        original_backend = template.active_backend
        test_backends = ["Agg", "TkAgg", "Qt5Agg"]

        for backend in test_backends:
            try:
                backend_compatible = template.configure_backend(backend)
                if backend_compatible:
                    # Quick functionality test
                    test_field = np.random.rand(32, 32).astype(np.float32)
                    test_agent = Coordinates(16, 16)
                    test_source = Coordinates(8, 8)

                    start_time = time.perf_counter()
                    template.render(test_field, test_agent, test_source)
                    backend_time = time.perf_counter() - start_time

                    compatibility_results[backend] = {
                        "compatible": True,
                        "render_time_ms": backend_time * 1000,
                        "interactive": backend not in ["Agg", "svg", "pdf"],
                    }
                else:
                    compatibility_results[backend] = {
                        "compatible": False,
                        "error": "Backend configuration failed",
                    }

            except Exception as e:
                compatibility_results[backend] = {"compatible": False, "error": str(e)}

        # Restore original backend
        template.configure_backend(original_backend)
        performance_report["compatibility_results"] = compatibility_results

    # Apply strict validation rules if enabled
    if strict_validation:
        strict_failures = []

        # Check for any test failures
        if (
            performance_report["performance_analysis"].get("overall_success_rate", 1.0)
            < 1.0
        ):
            strict_failures.append("Not all test iterations succeeded")

        # Check performance consistency
        consistency = performance_report["performance_analysis"].get(
            "performance_consistency", {}
        )
        cv = consistency.get("coefficient_of_variation", 0)
        if cv > 0.5:  # More than 50% variation
            strict_failures.append(f"High performance variation (CV={cv:.2f})")

        # Check memory efficiency for large grids
        large_grid_tests = [
            r
            for r in all_test_results
            if "128x128" in r["grid_size"] and "memory_analysis" in r
        ]
        if large_grid_tests:
            avg_memory = np.mean(
                [r["memory_analysis"]["average_memory_mb"] for r in large_grid_tests]
            )
            if avg_memory > 25:  # Strict memory limit
                strict_failures.append(
                    f"High memory usage for large grids: {avg_memory:.1f}MB"
                )

        if strict_failures:
            performance_report["validation_passed"] = False
            performance_report["strict_validation_failures"] = strict_failures

    # Generate optimization recommendations based on results
    recommendations = []

    if not performance_report["validation_passed"]:
        recommendations.extend(
            [
                f"Template does not meet {template_type} performance targets (<{target_ms}ms)",
                "Consider reducing quality level for better performance",
                "Enable caching if not already enabled for repetitive operations",
                "Optimize grid sizes based on use case requirements",
            ]
        )

    # Performance-specific recommendations
    avg_time = performance_report["performance_analysis"].get(
        "overall_average_time_ms", 0
    )
    if avg_time > target_ms * 1.5:
        recommendations.append(
            "Significant performance optimization needed - consider ULTRA_FAST quality level"
        )
    elif avg_time > target_ms * 1.2:
        recommendations.append("Minor performance optimization recommended")

    # Memory recommendations
    if isinstance(template, RGBTemplate):
        grid_area = template.config.grid_size.width * template.config.grid_size.height
        if grid_area > 128 * 128:
            recommendations.append(
                "Large grid sizes may impact performance - consider optimization"
            )

    performance_report["recommendations"] = recommendations

    return performance_report["validation_passed"], performance_report


# Export all public interfaces
__all__ = [
    # Core template classes
    "BaseRenderTemplate",
    "RGBTemplate",
    "MatplotlibTemplate",
    # Configuration and quality enums
    "TemplateConfig",
    "TemplateQuality",
    # Factory functions
    "create_rgb_template",
    "create_matplotlib_template",
    "create_custom_template",
    # Registry and management functions
    "register_template",
    "get_template_registry",
    "optimize_template",
    "validate_template_performance",
]
