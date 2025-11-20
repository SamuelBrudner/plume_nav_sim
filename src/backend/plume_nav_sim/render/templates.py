# rendering template system providing reusable, configurable templates for both RGB array generation and matplotlib human mode visualization, with performance optimization, backend compatibility management, and standardized visualization patterns for plume navigation environment rendering.

from __future__ import annotations

import contextlib
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

# Performance constants and configuration
from plume_nav_sim.core.constants import (
    BACKEND_PRIORITY_LIST,
    MATPLOTLIB_DEFAULT_FIGSIZE,
    PERFORMANCE_TARGET_HUMAN_RENDER_MS,
    PERFORMANCE_TARGET_RGB_RENDER_MS,
    RGB_DTYPE,
)
from plume_nav_sim.core.enums import RenderMode

# Core types for template sizing and positioning
from plume_nav_sim.core.geometry import Coordinates, GridSize
from plume_nav_sim.core.types import RGBArray

# Internal imports for color scheme integration and dual-mode optimization
from .colormaps import ColorScheme, create_default_scheme

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


# --- Internal helpers to keep factory functions simple (lower cyclomatic complexity) ---
def _validate_positive_grid_size(grid_size: GridSize) -> None:
    if grid_size.width <= 0 or grid_size.height <= 0:
        raise ValueError("Grid size dimensions must be positive")


def _optimize_scheme_for_mode(
    scheme: Optional[ColorScheme], mode: RenderMode
) -> ColorScheme:
    scheme = scheme or create_default_scheme()
    try:
        scheme.optimize_for_render_mode(mode)
    except Exception as e:
        warnings.warn(f"Color scheme optimization failed: {e}")
    return scheme


def _merge_options(
    defaults: Dict[str, Any], overrides: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    merged = defaults.copy()
    if overrides:
        merged.update(overrides)
    return merged


def _calculate_figsize_for_grid(grid_size: GridSize) -> Tuple[float, float]:
    base = 8.0
    aspect = grid_size.width / grid_size.height
    if aspect > 1.5:
        return (base * 1.2, base / 1.2)
    if aspect < 0.67:
        return (base / 1.2, base * 1.2)
    return (base, base)


def _backend_chain_with_preference(preferred: Optional[str]) -> list[str]:
    chain = BACKEND_PRIORITY_LIST.copy()
    if preferred:
        if preferred in chain:
            chain.remove(preferred)
        chain.insert(0, preferred)
    return chain


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
        errors: List[str] = []

        self._validate_grid_ranges(errors)
        self._validate_color_scheme(errors)
        self._validate_marker_config(errors)
        self._validate_performance_options(errors)

        if strict_mode:
            self._strict_validation(errors)

        return (len(errors) == 0), errors

    # --- TemplateConfig validation helpers (reduce complexity of validate) ---
    def _validate_grid_ranges(self, errors: List[str]) -> None:
        if not (8 <= self.grid_size.width <= 2048):
            errors.append(
                f"Grid width {self.grid_size.width} outside reasonable range [8, 2048]"
            )
        if not (8 <= self.grid_size.height <= 2048):
            errors.append(
                f"Grid height {self.grid_size.height} outside reasonable range [8, 2048]"
            )

    def _validate_color_scheme(self, errors: List[str]) -> None:
        if self.color_scheme is None:
            return
        try:
            self.color_scheme.get_concentration_colormap()
        except Exception as e:
            errors.append(f"Color scheme validation failed: {e}")

    def _validate_marker_config(self, errors: List[str]) -> None:
        agent_size = self.marker_config.get("agent_size", (3, 3))
        if not isinstance(agent_size, (tuple, list)) or len(agent_size) != 2:
            errors.append("Agent marker size must be (width, height) tuple")

    def _validate_performance_options(self, errors: List[str]) -> None:
        if (
            self.performance_options.get("enable_caching", True)
            and not self.caching_enabled
        ):
            errors.append("Performance caching enabled but template caching disabled")

    def _strict_validation(self, errors: List[str]) -> None:
        memory_estimate = (self.grid_size.width * self.grid_size.height * 4) / (
            1024 * 1024
        )  # MB
        if memory_estimate > 100:
            errors.append(
                f"Estimated memory usage {memory_estimate:.1f}MB exceeds strict limit"
            )

        preferred_backend = self.backend_preferences.get("preferred_backend")
        with contextlib.suppress(Exception):
            import matplotlib

            available_backends = matplotlib.backend_bases.Backend._backend_map.keys()
            if preferred_backend not in available_backends:
                errors.append(f"Preferred backend '{preferred_backend}' not available")

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
                raise RuntimeError(f"Template functionality test failed: {e}") from e

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
                with contextlib.suppress(Exception):
                    return self._fallback_render(
                        concentration_field, agent_position, source_position
                    )
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

        if use_cache or self._render_cache is None:
            return self.render(concentration_field, agent_position, source_position)
        # Temporarily disable cache for this render
        original_cache = self._render_cache
        self._render_cache = None
        try:
            result = self.render(concentration_field, agent_position, source_position)
        finally:
            self._render_cache = original_cache
        return result

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
        rgb_target_ms = PERFORMANCE_TARGET_RGB_RENDER_MS

        analysis = {
            "meets_target": True,
            "target_violations": [],
            "performance_data": {},
            "optimization_recommendations": [],
        }

        results = self._rgb_collect_test_results(
            test_scenarios, rgb_target_ms, analysis
        )
        successful = [r for r in results if r.get("success")] if results else []

        if successful:
            analysis["performance_data"] = self._rgb_performance_summary(
                successful, len(results)
            )

            if not analysis["meets_target"]:
                analysis["optimization_recommendations"] = self._rgb_recommendations(
                    successful, rgb_target_ms
                )

        return analysis["meets_target"], analysis

    # --- RGBTemplate performance helpers ---
    def _rgb_collect_test_results(
        self,
        test_scenarios: Dict[str, Any],
        rgb_target_ms: float,
        analysis: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for scenario_name, scenario_config in test_scenarios.items():
            grid_sizes = scenario_config.get("grid_sizes", [(64, 64), (128, 128)])
            for grid in grid_sizes:
                result = self._run_single_rgb_scenario(
                    scenario_name, grid, rgb_target_ms
                )
                results.append(result)

                if result["success"] and not result["meets_target"]:
                    analysis["meets_target"] = False
                    analysis["target_violations"].append(
                        {
                            "scenario": result["scenario"],
                            "actual_time_ms": result["avg_render_time_ms"],
                            "target_time_ms": rgb_target_ms,
                            "excess_time_ms": result["avg_render_time_ms"]
                            - rgb_target_ms,
                        }
                    )
        return results

    def _run_single_rgb_scenario(
        self, scenario_name: str, grid: Tuple[int, int], rgb_target_ms: float
    ) -> Dict[str, Any]:
        test_field = np.random.rand(*grid).astype(np.float32)
        test_agent = Coordinates(grid[1] // 2, grid[0] // 2)
        test_source = Coordinates(grid[1] // 4, grid[0] // 4)

        timings: List[float] = []
        for _ in range(10):
            start = time.perf_counter()
            try:
                result = self.render(test_field, test_agent, test_source)
                elapsed = time.perf_counter() - start
                timings.append(elapsed)

                if not isinstance(result, np.ndarray):
                    raise ValueError("RGB template must return numpy array")
                if result.shape != (*grid, 3):
                    raise ValueError(f"Invalid RGB array shape: {result.shape}")
                if result.dtype != RGB_DTYPE:
                    raise ValueError(f"Invalid RGB array dtype: {result.dtype}")
            except Exception as e:
                return {
                    "scenario": f"{scenario_name}_{grid[0]}x{grid[1]}",
                    "success": False,
                    "error": str(e),
                }

        if timings:
            avg = np.mean(timings)
            mx = np.max(timings)
            meets = avg <= (rgb_target_ms / 1000.0)
            return {
                "scenario": f"{scenario_name}_{grid[0]}x{grid[1]}",
                "success": True,
                "avg_render_time_ms": avg * 1000,
                "max_render_time_ms": mx * 1000,
                "meets_target": meets,
            }
        return {
            "scenario": f"{scenario_name}_{grid[0]}x{grid[1]}",
            "success": False,
            "error": "No timings collected",
        }

    def _rgb_performance_summary(
        self, successful: List[Dict[str, Any]], total_count: int
    ) -> Dict[str, Any]:
        times = [r["avg_render_time_ms"] for r in successful]
        return {
            "overall_average_ms": float(np.mean(times)),
            "overall_max_ms": float(np.max(times)),
            "success_rate": len(successful) / total_count if total_count else 0.0,
            "target_compliance_rate": sum(bool(r["meets_target"]) for r in successful)
            / len(successful),
        }

    def _rgb_recommendations(
        self, successful: List[Dict[str, Any]], rgb_target_ms: float
    ) -> List[str]:
        recs = [
            "Enable template caching to reduce repeated computation",
            "Use ULTRA_FAST or FAST quality levels for maximum performance",
            "Enable vectorized operations in performance options",
            "Consider reducing marker sizes for faster rendering",
        ]
        large = [
            r
            for r in successful
            if "128x128" in r["scenario"] or "256x256" in r["scenario"]
        ]
        if large and np.mean([r["avg_render_time_ms"] for r in large]) > rgb_target_ms:
            recs.append("Consider grid size optimization for large environments")
        return recs


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
            raise RuntimeError("Matplotlib backend test failed") from e

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
            if "backend" not in str(e).lower() and "display" not in str(e).lower():
                raise
            if self._handle_backend_error(e, self.active_backend):
                # Retry render with new backend
                self._execute_render(
                    concentration_field, agent_position, source_position
                )
            else:
                raise RuntimeError(
                    f"Matplotlib rendering failed with all backends: {e}"
                ) from e

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
        with contextlib.suppress(Exception):
            if not hasattr(self, "_colorbar"):
                self._colorbar = self.figure.colorbar(
                    self.concentration_image, ax=self.axes, label="Concentration"
                )

    def refresh_display(self, force_redraw: bool = False) -> None:
        """
        Refreshes matplotlib display with optimized update strategy for
        interactive performance.
        """
        try:
            # Force complete redraw for non-interactive backends
            self.figure.canvas.draw()
            # Allow GUI update with configured interval
            if self.backend_interactive:
                plt.pause(self._animation_interval)

        except Exception as e:
            warnings.warn(f"Display refresh failed: {e}")
            # Attempt fallback refresh method
            with contextlib.suppress(Exception):
                self.figure.canvas.flush_events()

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
                    return self._extracted_from_configure_backend_31(backend)
            except (ImportError, RuntimeError) as e:
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

    # TODO Rename this here and in `configure_backend`
    def _extracted_from_configure_backend_31(self, backend):
        # Backend successfully activated
        self.active_backend = backend
        self.backend_interactive = backend not in ["Agg", "svg", "pdf"]

        # Test basic functionality
        test_fig, test_ax = plt.subplots(figsize=(1, 1))
        test_ax.plot([0, 1], [0, 1])
        plt.close(test_fig)

        return True

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
            backend
            for backend in self._backend_fallback_chain
            if backend not in {failed_backend, self.active_backend}
        ]

        # Try next available backend
        for backend in remaining_backends:
            try:
                return self._extracted_from__handle_backend_error_17(backend)
            except Exception as e:
                warnings.warn(f"Fallback to backend {backend} failed: {e}")
                continue

        return False

    # TODO Rename this here and in `_handle_backend_error`
    def _extracted_from__handle_backend_error_17(self, backend):
        plt.switch_backend(backend)
        self.active_backend = backend
        self.backend_interactive = backend not in {"Agg", "svg", "pdf"}

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

        successful_tests = [r for r in test_results if r["success"]]
        if successful_tests:
            self._extracted_from_validate_performance_87(
                successful_tests,
                test_results,
                performance_analysis,
                matplotlib_target_ms,
            )
        return performance_analysis["meets_target"], performance_analysis

    # TODO Rename this here and in `validate_performance`
    def _extracted_from_validate_performance_87(
        self, successful_tests, test_results, performance_analysis, matplotlib_target_ms
    ):
        all_times = [r["avg_render_time_ms"] for r in successful_tests]
        performance_analysis["performance_data"] = {
            "overall_average_ms": np.mean(all_times),
            "overall_max_ms": np.max(all_times),
            "success_rate": float(np.mean([r["success"] for r in test_results])),
            "target_compliance_rate": float(
                np.mean([r["meets_target"] for r in successful_tests])
            ),
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
                recommendations.append("Switch to Agg backend for non-interactive use")

            if np.mean(all_times) > matplotlib_target_ms * 2:
                recommendations.append(
                    "Consider fallback to RGB array mode for critical performance"
                )

            performance_analysis["optimization_recommendations"] = recommendations


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
    _validate_positive_grid_size(grid_size)

    memory_estimate_mb = (grid_size.width * grid_size.height * 3) / (1024 * 1024)
    if memory_estimate_mb > 500:  # Reasonable upper limit
        warnings.warn(
            f"Large grid size may require {memory_estimate_mb:.1f}MB of memory"
        )

    # Initialize and optimize color scheme for RGB rendering mode
    color_scheme = _optimize_scheme_for_mode(color_scheme, RenderMode.RGB_ARRAY)

    # Configure quality level with performance targets
    if quality_level is None:
        quality_level = TemplateQuality.STANDARD

    # Merge options with defaults
    merged_options = _merge_options(DEFAULT_RGB_TEMPLATE_CONFIG, template_options)

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
    # Validate and size figure
    _validate_positive_grid_size(grid_size)
    figsize = _calculate_figsize_for_grid(grid_size)

    # Initialize color scheme optimized for matplotlib visualization
    color_scheme = _optimize_scheme_for_mode(color_scheme, RenderMode.HUMAN)

    # Configure matplotlib backend selection with fallback chain
    backend_chain = _backend_chain_with_preference(preferred_backend)
    # Apply template options with matplotlib defaults
    merged_options = _merge_options(
        DEFAULT_MATPLOTLIB_TEMPLATE_CONFIG, template_options
    )
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
    # Validate inputs and apply custom parameters
    _validate_template_type(template_type)
    _validate_template_config(config)
    if custom_parameters:
        _validate_custom_parameters(template_type, custom_parameters)
        config.custom_parameters.update(custom_parameters)

    # Instantiate and initialize
    template = _instantiate_template_from_type(config, template_type)
    _initialize_or_raise(template, template_type)

    # Optional performance validation
    if validate_performance:
        test_scenarios = {
            "basic_test": {"grid_sizes": [(32, 32), (64, 64)]},
            "custom_test": {
                "grid_sizes": [(config.grid_size.width, config.grid_size.height)]
            },
        }
        meets, pdata = template.validate_performance(test_scenarios)
        if not meets:
            warnings.warn(
                f"Custom template does not meet performance targets: {pdata.get('target_violations', [])}"
            )
            recommendations = pdata.get("optimization_recommendations", [])
            if recommendations:
                warnings.warn(
                    f"Performance optimization recommendations: {recommendations}"
                )

    return template


def _validate_template_type(template_type: str) -> None:
    supported_types = ["rgb", "matplotlib", "hybrid"]
    if template_type not in supported_types:
        raise ValueError(
            f"Template type '{template_type}' not supported. Use one of: {supported_types}"
        )


def _validate_template_config(config: TemplateConfig) -> None:
    ok, errs = config.validate(strict_mode=True)
    if not ok:
        raise ValueError(f"Invalid template configuration: {errs}")


def _instantiate_template_from_type(
    config: TemplateConfig, template_type: str
) -> Union[RGBTemplate, MatplotlibTemplate]:
    if template_type == "hybrid":
        raise NotImplementedError(
            "Hybrid templates not implemented in proof-of-life version"
        )
    if template_type == "matplotlib":
        return MatplotlibTemplate(config)
    if template_type == "rgb":
        return RGBTemplate(config)
    raise ValueError(f"Unknown template type: {template_type}")


def _initialize_or_raise(
    template: Union[RGBTemplate, MatplotlibTemplate], template_type: str
) -> None:
    try:
        if not template.initialize():
            raise RuntimeError(f"Custom {template_type} template initialization failed")
    except Exception as e:
        raise RuntimeError(f"Custom template creation failed: {e}") from e


def _validate_custom_parameters(template_type: str, params: Dict[str, Any]) -> None:
    if template_type == "matplotlib":
        allowed = {
            "figsize",
            "backend_fallback",
            "interactive_updates",
            "animation_interval",
        }
        invalid = [
            p for p in params if p not in allowed and not p.startswith("custom_")
        ]
        if invalid:
            warnings.warn(
                f"Custom parameters {invalid} may not apply to matplotlib templates"
            )
        return
    if template_type == "rgb":
        allowed = {"agent_marker_size", "source_marker_size", "performance_optimized"}
        invalid = [
            p for p in params if p not in allowed and not p.startswith("custom_")
        ]
        if invalid:
            warnings.warn(f"Custom parameters {invalid} may not apply to RGB templates")


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
    _assert_valid_template_name(template_name)

    # Check for existing template when override not allowed
    if template_name in TEMPLATE_REGISTRY and not allow_override:
        warnings.warn(
            f"Template '{template_name}' already registered. Use allow_override=True to replace."
        )
        return False

    if not isinstance(template, (RGBTemplate, MatplotlibTemplate)):
        raise TypeError(
            "Template must be an instance of RGBTemplate or MatplotlibTemplate"
        )

    if not _ensure_initialized_for_registration(template):
        return False

    metadata = metadata or {}
    _warn_on_missing_metadata(metadata)

    if not _smoke_test_template_result(template):
        return False

    # Handle naming conflicts with override checking
    if template_name in TEMPLATE_REGISTRY and allow_override:
        old_template = TEMPLATE_REGISTRY[template_name]["template"]
        with contextlib.suppress(Exception):
            old_template.cleanup()

    # Add/replace template entry
    TEMPLATE_REGISTRY[template_name] = _build_registry_entry(template, metadata)
    return True


def _assert_valid_template_name(name: str) -> None:
    if not isinstance(name, str) or len(name) < 3:
        raise ValueError("Template name must be a string with at least 3 characters")
    if not name.replace("_", "").replace("-", "").isalnum():
        raise ValueError(
            "Template name must contain only alphanumeric characters, hyphens, and underscores"
        )


def _ensure_initialized_for_registration(
    template: Union[RGBTemplate, MatplotlibTemplate]
) -> bool:
    if template._initialized:
        return True
    try:
        return bool(template.initialize())
    except Exception as e:
        warnings.warn(f"Template validation failed: {e}")
        return False


def _warn_on_missing_metadata(metadata: Dict[str, Any]) -> None:
    required = ["description", "performance_targets", "compatibility"]
    missing = [field for field in required if field not in metadata]
    if missing:
        warnings.warn(f"Missing recommended metadata fields: {missing}")


def _smoke_test_template_result(
    template: Union[RGBTemplate, MatplotlibTemplate]
) -> bool:
    try:
        test_field = np.random.rand(32, 32).astype(np.float32)
        test_agent = Coordinates(16, 16)
        test_source = Coordinates(8, 8)
        result = template.render(test_field, test_agent, test_source)

        if isinstance(template, RGBTemplate):
            if not isinstance(result, np.ndarray) or result.shape != (32, 32, 3):
                raise ValueError("RGB template test failed - invalid result format")
        elif isinstance(template, MatplotlibTemplate):
            if result is not None:
                warnings.warn("Matplotlib template returned non-None result")
        return True
    except Exception as e:
        warnings.warn(f"Template functionality test failed: {e}")
        return False


def _build_registry_entry(
    template: Union[RGBTemplate, MatplotlibTemplate], metadata: Dict[str, Any]
) -> Dict[str, Any]:
    return {
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


def get_template_registry(
    template_type_filter: Optional[str] = None,
    quality_filter: Optional[str] = None,
    include_performance_data: bool = False,
) -> Dict[str, Any]:
    """
    Returns comprehensive template registry with filtering capabilities, performance
    characteristics, and usage recommendations for template discovery and selection.
    """
    # Retrieve and filter
    if not TEMPLATE_REGISTRY:
        return {
            "templates": {},
            "summary": {
                "total_templates": 0,
                "available_types": [],
                "quality_levels": [],
            },
        }

    filtered = _filter_registry_items(
        TEMPLATE_REGISTRY,
        template_type_filter=template_type_filter,
        quality_filter=quality_filter,
        include_performance_data=include_performance_data,
    )

    # Summary and ordering
    all_types = [entry["template_type"] for entry in filtered.values()]
    all_qualities = [entry["quality_level"] for entry in filtered.values()]
    summary = {
        "total_templates": len(filtered),
        "available_types": list(set(all_types)),
        "quality_levels": list(set(all_qualities)),
        "filters_applied": {
            "template_type": template_type_filter,
            "quality_filter": quality_filter,
            "performance_data_included": include_performance_data,
        },
    }

    sorted_templates = dict(
        sorted(
            filtered.items(),
            key=lambda item: (item[1]["usage_count"], -item[1]["registration_time"]),
            reverse=True,
        )
    )

    return {
        "templates": sorted_templates,
        "summary": summary,
        "recommendations": _generate_template_recommendations(sorted_templates),
    }


def _filter_registry_items(
    registry: Dict[str, Any],
    *,
    template_type_filter: Optional[str],
    quality_filter: Optional[str],
    include_performance_data: bool,
) -> Dict[str, Any]:
    filtered: Dict[str, Any] = {}
    for name, entry in registry.items():
        template = entry["template"]
        metadata = entry["metadata"]

        if template_type_filter:
            valid_types = ["rgb", "matplotlib", "custom"]
            if template_type_filter not in valid_types:
                warnings.warn(
                    f"Invalid template type filter '{template_type_filter}'. Valid options: {valid_types}"
                )
                continue
            if template_type_filter == "rgb" and not isinstance(template, RGBTemplate):
                continue
            if template_type_filter == "matplotlib" and not isinstance(
                template, MatplotlibTemplate
            ):
                continue

        if quality_filter:
            valid_qualities = [q.value for q in TemplateQuality]
            if quality_filter not in valid_qualities:
                warnings.warn(
                    f"Invalid quality filter '{quality_filter}'. Valid options: {valid_qualities}"
                )
                continue
            if metadata.get("quality_level") != quality_filter:
                continue

        info = _build_template_info(template, metadata, include_performance_data)
        filtered[name] = info

    return filtered


def _build_template_info(
    template: Union[RGBTemplate, MatplotlibTemplate],
    metadata: Dict[str, Any],
    include_performance_data: bool,
) -> Dict[str, Any]:
    info = {
        "template_type": metadata["template_type"],
        "description": metadata["description"],
        "grid_size": metadata["grid_size"],
        "quality_level": metadata["quality_level"],
        "compatibility": metadata["compatibility"],
        "usage_count": metadata["usage_count"],
        "registration_time": metadata["registration_time"],
    }

    if include_performance_data:
        metrics = template.get_performance_metrics()
        info["performance_data"] = {
            "average_render_time_ms": metrics.get("average_render_time", 0.0) * 1000,
            "total_renders": metrics.get("total_renders", 0),
            "cache_hit_rate": 0.0,
        }
        cache_stats = metrics.get("cache_stats")
        if cache_stats:
            info["performance_data"]["cache_hit_rate"] = cache_stats.get(
                "hit_rate", 0.0
            )

    return info


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
    current_metrics = template.get_performance_metrics()
    report = {
        "original_performance": current_metrics.copy(),
        "optimizations_applied": [],
        "performance_improvements": {},
        "recommendations": [],
        "configuration_changes": {},
    }

    optimized_config = template.config.clone()
    target_render_time_ms = optimization_targets.get("target_render_time_ms", 5.0)
    target_memory_mb = optimization_targets.get("target_memory_mb", 50.0)
    current_rt_ms = current_metrics.get("average_render_time", 0.0) * 1000

    # Apply quality/caching/perf options
    _apply_quality_and_caching_optimizations(
        optimized_config,
        report,
        target_render_time_ms,
        usage_statistics or {},
        current_rt_ms,
    )
    _memory_recommendations(optimized_config, report, target_memory_mb)

    # Cache tuning suggestions
    if usage_statistics and optimized_config.caching_enabled:
        access_pattern = usage_statistics.get("access_pattern", "random")
        cache_size = (
            min(TEMPLATE_CACHE_SIZE, 5)
            if access_pattern == "sequential"
            else (
                TEMPLATE_CACHE_SIZE
                if access_pattern == "repetitive"
                else TEMPLATE_CACHE_SIZE // 2
            )
        )
        report["configuration_changes"]["cache_size"] = cache_size

    # Build optimized instance and measure
    try:
        optimized_template = _instantiate_and_measure_optimized(
            template, optimized_config, current_rt_ms, report
        )
    except Exception as e:
        warnings.warn(f"Template optimization failed: {e}")
        return template, report

    if not report["optimizations_applied"]:
        report["recommendations"].append(
            "Template already well-optimized for current targets"
        )

    report["recommendations"].extend(
        [
            "Monitor performance metrics regularly to identify optimization opportunities",
            "Adjust quality levels based on use case requirements",
            "Enable caching for repetitive rendering scenarios",
            "Consider template-specific optimizations based on usage patterns",
        ]
    )
    return optimized_template, report


def _apply_quality_and_caching_optimizations(
    config: TemplateConfig,
    report: Dict[str, Any],
    target_render_time_ms: float,
    usage_stats: Dict[str, Any],
    current_rt_ms: float,
) -> None:
    if current_rt_ms <= target_render_time_ms:
        return

    current_quality = config.quality_level
    if current_quality != TemplateQuality.ULTRA_FAST:
        if target_render_time_ms <= 1.0:
            config.quality_level = TemplateQuality.ULTRA_FAST
            report["optimizations_applied"].append(
                f"Reduced quality from {current_quality.value} to ultra_fast for <1ms target"
            )
        elif target_render_time_ms <= 3.0:
            config.quality_level = TemplateQuality.FAST
            report["optimizations_applied"].append(
                f"Reduced quality from {current_quality.value} to fast for <3ms target"
            )

    if usage_stats:
        repeat = usage_stats.get("repeated_renders", 0.1)
        if repeat > 0.3 and not config.caching_enabled:
            config.caching_enabled = True
            report["optimizations_applied"].append(
                "Enabled caching due to high repeat render rate"
            )

    if not config.performance_options.get("vectorized_operations", True):
        config.performance_options["vectorized_operations"] = True
        report["optimizations_applied"].append("Enabled vectorized operations")

    if not config.performance_options.get("memory_optimization", False):
        config.performance_options["memory_optimization"] = True
        report["optimizations_applied"].append("Enabled memory optimization")


def _memory_recommendations(
    config: TemplateConfig, report: Dict[str, Any], target_memory_mb: float
) -> None:
    est = (config.grid_size.width * config.grid_size.height * 3) / (1024 * 1024)
    if est <= target_memory_mb:
        return
    factor = target_memory_mb / est
    suggested_width = int(config.grid_size.width * np.sqrt(factor))
    suggested_height = int(config.grid_size.height * np.sqrt(factor))
    report["recommendations"].append(
        f"Consider reducing grid size to {suggested_width}x{suggested_height} to meet {target_memory_mb}MB target"
    )


def _instantiate_and_measure_optimized(
    template: Union[RGBTemplate, MatplotlibTemplate],
    config: TemplateConfig,
    current_rt_ms: float,
    report: Dict[str, Any],
) -> Union[RGBTemplate, MatplotlibTemplate]:
    optimized = (
        RGBTemplate(config)
        if isinstance(template, RGBTemplate)
        else MatplotlibTemplate(config)
    )
    if not optimized.initialize():
        raise RuntimeError("Optimized template initialization failed")

    test_field = np.random.rand(64, 64).astype(np.float32)
    test_agent = Coordinates(32, 32)
    test_source = Coordinates(16, 16)
    start = time.perf_counter()
    optimized.render(test_field, test_agent, test_source)
    opt_time = time.perf_counter() - start

    if current_rt_ms > 0:
        improvement_percent = (
            (current_rt_ms / 1000 - opt_time) / (current_rt_ms / 1000)
        ) * 100
        report["performance_improvements"][
            "render_time_improvement_percent"
        ] = improvement_percent
        report["performance_improvements"]["new_render_time_ms"] = opt_time * 1000

    return optimized


def validate_template_performance(
    template: Union[RGBTemplate, MatplotlibTemplate],
    test_scenarios: Dict[str, Any],
    strict_validation: bool = False,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Comprehensive performance validation function for rendering templates including
    timing benchmarks, memory usage analysis, and compatibility testing.
    """
    # Initialize report and targets
    report = {
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

    target_ms, template_kind = _determine_targets_for_template(template)
    report["performance_analysis"]["target_time_ms"] = target_ms

    # Benchmarks
    all_results = _run_benchmarks_for_template(
        template, test_scenarios, target_ms, strict_validation, report
    )

    # Aggregate
    if all_results:
        successful = [r for r in all_results if r.get("success_rate", 0) > 0.8]
        if successful:
            report["performance_analysis"].update(
                {
                    "overall_average_time_ms": np.mean(
                        [r["average_time_ms"] for r in successful]
                    ),
                    "overall_max_time_ms": np.max(
                        [r["max_time_ms"] for r in successful]
                    ),
                    "overall_success_rate": np.mean(
                        [r["success_rate"] for r in successful]
                    ),
                    "target_compliance_rate": np.mean(
                        [r["meets_target"] for r in successful]
                    ),
                    "performance_consistency": {
                        "std_deviation_ms": np.std(
                            [r["average_time_ms"] for r in successful]
                        ),
                        "coefficient_of_variation": np.std(
                            [r["average_time_ms"] for r in successful]
                        )
                        / np.mean([r["average_time_ms"] for r in successful]),
                    },
                }
            )

    # Compatibility and strict checks
    if isinstance(template, MatplotlibTemplate):
        _extracted_from_validate_template_performance_199(template, report)
    if strict_validation:
        _extracted_from_validate_template_performance_(report, all_results)

    # Recommendations
    recs: List[str] = []
    if not report["validation_passed"]:
        recs.extend(
            [
                f"Template does not meet {template_kind} performance targets (<{target_ms}ms)",
                "Consider reducing quality level for better performance",
                "Enable caching if not already enabled for repetitive operations",
                "Optimize grid sizes based on use case requirements",
            ]
        )

    avg_time = report["performance_analysis"].get("overall_average_time_ms", 0)
    if avg_time > target_ms * 1.5:
        recs.append(
            "Significant performance optimization needed - consider ULTRA_FAST quality level"
        )
    elif avg_time > target_ms * 1.2:
        recs.append("Minor performance optimization recommended")

    if isinstance(template, RGBTemplate):
        grid_area = template.config.grid_size.width * template.config.grid_size.height
        if grid_area > 128 * 128:
            recs.append(
                "Large grid sizes may impact performance - consider optimization"
            )

    report["recommendations"] = recs
    return report["validation_passed"], report


def _determine_targets_for_template(
    template: Union[RGBTemplate, MatplotlibTemplate]
) -> Tuple[float, str]:
    if isinstance(template, RGBTemplate):
        return PERFORMANCE_TARGET_RGB_RENDER_MS, "RGB"
    return PERFORMANCE_TARGET_HUMAN_RENDER_MS, "Matplotlib"


def _run_benchmarks_for_template(
    template: Union[RGBTemplate, MatplotlibTemplate],
    test_scenarios: Dict[str, Any],
    target_ms: float,
    strict_validation: bool,
    report: Dict[str, Any],
) -> List[Dict[str, Any]]:
    all_results: List[Dict[str, Any]] = []
    for scenario_name, scenario_config in test_scenarios.items():
        scenario_results = {"scenario_name": scenario_name, "tests": [], "summary": {}}
        grid_sizes = scenario_config.get("grid_sizes", [(64, 64), (128, 128)])
        iterations = scenario_config.get(
            "iterations", 10 if isinstance(template, RGBTemplate) else 5
        )

        for grid in grid_sizes:
            tr = _run_single_benchmark_scenario(
                template,
                scenario_name,
                grid,
                iterations,
                target_ms,
                strict_validation,
                report,
            )
            if tr:
                scenario_results["tests"].append(tr)
                all_results.append(tr)
            else:
                # Failure case handled inside helper (updates report)
                scenario_results["tests"].append(
                    {
                        "grid_size": f"{grid[0]}x{grid[1]}",
                        "success_rate": 0.0,
                        "error": "All iterations failed",
                    }
                )

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
                "target_compliance_rate": np.mean(
                    [t["meets_target"] for t in successful_tests]
                ),
            }

        report["test_results"][scenario_name] = scenario_results

    return all_results


def _run_single_benchmark_scenario(
    template: Union[RGBTemplate, MatplotlibTemplate],
    scenario_name: str,
    grid: Tuple[int, int],
    iterations: int,
    target_ms: float,
    strict_validation: bool,
    report: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    test_field = np.random.rand(*grid).astype(np.float32)
    test_agent = Coordinates(grid[1] // 2, grid[0] // 2)
    test_source = Coordinates(grid[1] // 4, grid[0] // 4)

    times: List[float] = []
    mem_use: List[float] = []
    success = 0

    for iteration in range(iterations):
        try:
            elapsed, mem_diff = _execute_benchmark_iteration(
                template, test_field, test_agent, test_source, grid
            )
            times.append(elapsed)
            if mem_diff is not None:
                mem_use.append(mem_diff)
            success += 1

        except Exception as e:
            report["validation_passed"] = False
            times.append(-1)
            if strict_validation:
                report["test_results"][f"{scenario_name}_{grid[0]}x{grid[1]}"] = {
                    "success": False,
                    "error": str(e),
                    "iteration": iteration,
                }
                break

    successful_times = [t for t in times if t > 0]
    if not successful_times:
        report["validation_passed"] = False
        return None

    tr = {
        "grid_size": f"{grid[0]}x{grid[1]}",
        "success_rate": success / iterations,
        "average_time_ms": np.mean(successful_times) * 1000,
        "max_time_ms": np.max(successful_times) * 1000,
        "min_time_ms": np.min(successful_times) * 1000,
        "std_time_ms": np.std(successful_times) * 1000,
        "meets_target": np.mean(successful_times) * 1000 <= target_ms,
        "iterations": iterations,
    }
    if mem_use:
        tr["memory_analysis"] = {
            "average_memory_mb": np.mean(mem_use),
            "max_memory_mb": np.max(mem_use),
            "memory_efficient": np.mean(mem_use) < 10,
        }
    if not tr["meets_target"]:
        report["validation_passed"] = False
    return tr


def _execute_benchmark_iteration(
    template: Union[RGBTemplate, MatplotlibTemplate],
    test_field: np.ndarray,
    test_agent: Coordinates,
    test_source: Coordinates,
    grid: Tuple[int, int],
) -> Tuple[float, Optional[float]]:
    try:
        import psutil  # type: ignore

        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024
    except ImportError:
        mem_before = None

    start = time.perf_counter()
    result = template.render(test_field, test_agent, test_source)
    elapsed = time.perf_counter() - start

    if isinstance(template, RGBTemplate):
        if not isinstance(result, np.ndarray):
            raise ValueError("RGB template must return numpy array")
        if result.shape != (*grid, 3):
            raise ValueError(f"Invalid RGB shape: {result.shape}")
        if result.dtype != RGB_DTYPE:
            raise ValueError(f"Invalid RGB dtype: {result.dtype}")

    mem_diff = None
    if mem_before is not None:
        mem_after = process.memory_info().rss / 1024 / 1024
        mem_diff = mem_after - mem_before

    return elapsed, mem_diff


# TODO Rename this here and in `validate_template_performance`
def _extracted_from_validate_template_performance_(
    performance_report, all_test_results
):
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


# TODO Rename this here and in `validate_template_performance`
def _extracted_from_validate_template_performance_199(template, performance_report):
    compatibility_results = {}

    original_backend = template.active_backend
    test_backends = ["Agg", "TkAgg", "Qt5Agg"]

    for backend in test_backends:
        try:
            if template.configure_backend(backend):
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
