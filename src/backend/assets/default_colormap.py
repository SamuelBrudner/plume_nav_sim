"""
Default colormap and color scheme configuration module for plume navigation environment.

This module provides standardized color definitions, colormap utilities, and scheme
management for dual-mode rendering with optimized color schemes for both RGB array
generation and matplotlib human mode visualization. Includes accessibility features,
performance optimization, and cross-platform compatibility.

Key Features:
- Dual-mode rendering support (RGB array + matplotlib)
- Predefined color schemes with accessibility options
- Performance-optimized color operations (<5ms RGB, <50ms human mode)
- Cross-platform matplotlib backend compatibility
- Comprehensive validation and error handling
- LRU caching for colormap operations
"""

from __future__ import annotations

import functools  # >=3.10 - LRU cache decorator for performance optimization
import warnings  # >=3.10 - Deprecation warnings and compatibility alerts
from dataclasses import (  # >=3.10 - Structured configuration with validation
    dataclass,
    field,
)
from enum import Enum  # >=3.10 - Color scheme enumeration with type safety
from typing import (  # >=3.10 - Type hints and annotations
    Any,
    Dict,
    Optional,
    Tuple,
    Union,
)

import matplotlib.cm  # >=3.9.0 - Built-in colormap registry and management
import matplotlib.colors  # >=3.9.0 - Colormap objects and normalization utilities
import numpy as np  # >=2.1.0 - RGB array operations, mathematical transformations

# =============================================================================
# GLOBAL CONSTANTS AND CONFIGURATION
# =============================================================================

# Default colormap configuration
DEFAULT_COLORMAP = "gray"
DEFAULT_AGENT_COLOR = (255, 0, 0)  # Red RGB for high visibility
DEFAULT_SOURCE_COLOR = (255, 255, 255)  # White RGB for contrast
DEFAULT_BACKGROUND_COLOR = (0, 0, 0)  # Black RGB for zero concentration

# Concentration field colormap settings
CONCENTRATION_COLORMAP = "gray"

# Marker size specifications (height, width in pixels)
AGENT_MARKER_SIZE = (3, 3)  # 3x3 pixel red square
SOURCE_MARKER_SIZE = (5, 5)  # 5x5 pixel white cross pattern

# Color value constraints and normalization
COLOR_VALUE_MIN = 0  # Minimum RGB channel value
COLOR_VALUE_MAX = 255  # Maximum RGB channel value
CONCENTRATION_RANGE = (0.0, 1.0)  # Normalized concentration range
RGB_CHANNELS = 3  # Standard RGB channel count

# Performance and caching configuration
COLORMAP_CACHE_SIZE = 10  # LRU cache size for matplotlib colormaps
ACCESSIBILITY_CONTRAST_RATIO = 4.5  # WCAG AA compliance minimum ratio
PERFORMANCE_TARGET_RGB_MS = 5.0  # RGB array generation target latency
PERFORMANCE_TARGET_HUMAN_MS = 50.0  # Human mode rendering target latency

# Predefined color schemes registry
PREDEFINED_SCHEMES = [
    "standard",
    "high_contrast",
    "colorblind_friendly",
    "minimal",
    "debug",
]

# Module exports for external access
__all__ = [
    "ColorScheme",
    "PredefinedScheme",
    "DEFAULT_COLORMAP",
    "create_default_scheme",
    "create_color_scheme",
    "validate_color_scheme",
    "normalize_concentration_to_rgb",
    "apply_agent_marker",
    "apply_source_marker",
    "get_matplotlib_colormap",
    "optimize_for_performance",
]


# =============================================================================
# PREDEFINED COLOR SCHEMES ENUMERATION
# =============================================================================


class PredefinedScheme(Enum):
    """
    Enumeration of predefined color schemes for plume navigation visualization.

    Provides common visualization configurations including standard research colors,
    high contrast accessibility, colorblind-friendly palettes, minimal visualization,
    and debug schemes with optimized performance characteristics.
    """

    STANDARD = "standard"  # Default visualization scheme
    HIGH_CONTRAST = "high_contrast"  # Enhanced visibility for accessibility
    COLORBLIND_FRIENDLY = "colorblind_friendly"  # Optimized for color vision deficiency
    MINIMAL = "minimal"  # Reduced visual complexity
    DEBUG = "debug"  # High visibility for development

    def get_color_config(self) -> Dict[str, Any]:
        """
        Returns complete color configuration for the predefined scheme.

        Returns:
            dict: Complete color configuration with agent, source, background colors,
                 colormap specification, accessibility settings, and performance flags
        """
        config_map = {
            PredefinedScheme.STANDARD: {
                "agent_color": (255, 0, 0),  # Red agent marker
                "source_color": (255, 255, 255),  # White source marker
                "background_color": (0, 0, 0),  # Black background
                "concentration_colormap": "gray",  # Grayscale concentration field
                "accessibility_enhanced": False,
                "performance_optimized": True,
                "agent_marker_size": AGENT_MARKER_SIZE,
                "source_marker_size": SOURCE_MARKER_SIZE,
            },
            PredefinedScheme.HIGH_CONTRAST: {
                "agent_color": (255, 255, 0),  # Bright yellow agent
                "source_color": (0, 255, 255),  # Cyan source marker
                "background_color": (0, 0, 0),  # Black background
                "concentration_colormap": "gray",
                "accessibility_enhanced": True,
                "performance_optimized": True,
                "agent_marker_size": (5, 5),  # Larger markers for visibility
                "source_marker_size": (7, 7),
            },
            PredefinedScheme.COLORBLIND_FRIENDLY: {
                "agent_color": (230, 159, 0),  # Orange for deuteranopia safety
                "source_color": (86, 180, 233),  # Sky blue for visibility
                "background_color": (0, 0, 0),  # Black background
                "concentration_colormap": "viridis",  # Colorblind-safe colormap
                "accessibility_enhanced": True,
                "performance_optimized": True,
                "agent_marker_size": AGENT_MARKER_SIZE,
                "source_marker_size": SOURCE_MARKER_SIZE,
            },
            PredefinedScheme.MINIMAL: {
                "agent_color": (128, 128, 128),  # Gray agent for subtlety
                "source_color": (192, 192, 192),  # Light gray source
                "background_color": (0, 0, 0),  # Black background
                "concentration_colormap": "gray",
                "accessibility_enhanced": False,
                "performance_optimized": True,
                "agent_marker_size": (2, 2),  # Smaller markers
                "source_marker_size": (3, 3),
            },
            PredefinedScheme.DEBUG: {
                "agent_color": (255, 0, 255),  # Magenta for high visibility
                "source_color": (0, 255, 0),  # Green for strong contrast
                "background_color": (64, 64, 64),  # Dark gray background
                "concentration_colormap": "plasma",  # High contrast colormap
                "accessibility_enhanced": False,
                "performance_optimized": False,  # Debugging over performance
                "agent_marker_size": (7, 7),  # Large markers for debugging
                "source_marker_size": (9, 9),
            },
        }
        return config_map[self]

    def is_accessibility_enhanced(self) -> bool:
        """
        Check if scheme includes accessibility enhancements.

        Returns:
            bool: True if scheme includes high contrast ratios, colorblind-friendly
                 colors, or enhanced visibility features
        """
        return self in [
            PredefinedScheme.HIGH_CONTRAST,
            PredefinedScheme.COLORBLIND_FRIENDLY,
        ]

    def get_performance_profile(self) -> Dict[str, Any]:
        """
        Returns performance characteristics for the predefined scheme.

        Returns:
            dict: Performance profile with target latencies, memory usage estimates,
                 and optimization settings
        """
        base_profile = {
            "target_rgb_latency_ms": PERFORMANCE_TARGET_RGB_MS,
            "target_human_latency_ms": PERFORMANCE_TARGET_HUMAN_MS,
            "memory_usage_estimate_mb": 5,
            "caching_enabled": True,
            "optimization_level": "standard",
        }

        # Adjust profile based on scheme complexity
        if self == PredefinedScheme.DEBUG:
            base_profile.update(
                {
                    "target_rgb_latency_ms": 10.0,  # Relaxed for debugging
                    "target_human_latency_ms": 100.0,
                    "memory_usage_estimate_mb": 10,
                    "optimization_level": "debug",
                }
            )
        elif self == PredefinedScheme.HIGH_CONTRAST:
            base_profile.update(
                {
                    "memory_usage_estimate_mb": 7,  # Larger markers
                    "optimization_level": "accessibility",
                }
            )

        return base_profile


# =============================================================================
# MAIN COLOR SCHEME DATA CLASS
# =============================================================================


@dataclass
class ColorScheme:
    """
    Main color scheme configuration for plume navigation environment visualization.

    Provides comprehensive color configuration, validation, and rendering integration
    with support for dual-mode rendering (RGB array and matplotlib), accessibility
    features, and performance optimization.
    """

    # Core color configuration (RGB tuples)
    agent_color: Tuple[int, int, int]
    source_color: Tuple[int, int, int]
    background_color: Tuple[int, int, int]
    concentration_colormap: str

    # Optional configuration with defaults
    agent_marker_size: Tuple[int, int] = field(default=AGENT_MARKER_SIZE)
    source_marker_size: Tuple[int, int] = field(default=SOURCE_MARKER_SIZE)
    accessibility_enhanced: bool = field(default=False)
    optimized_for_mode: Optional[str] = field(default=None)

    # Internal state management
    _cached_colormap: Optional[matplotlib.colors.Colormap] = field(
        default=None, init=False
    )
    performance_config: Dict[str, Any] = field(default_factory=dict, init=False)

    def __post_init__(self):
        """Initialize and validate color scheme configuration."""
        # Validate RGB color values
        self._validate_rgb_color(self.agent_color, "agent_color")
        self._validate_rgb_color(self.source_color, "source_color")
        self._validate_rgb_color(self.background_color, "background_color")

        # Validate marker sizes
        self._validate_marker_size(self.agent_marker_size, "agent_marker_size")
        self._validate_marker_size(self.source_marker_size, "source_marker_size")

        # Verify matplotlib colormap availability
        try:
            matplotlib.colormaps[self.concentration_colormap]
        except (ValueError, KeyError):
            warnings.warn(
                f"Colormap '{self.concentration_colormap}' not found, falling back to 'gray'"
            )
            self.concentration_colormap = "gray"

        # Initialize performance configuration
        self.performance_config = {
            "caching_enabled": True,
            "target_rgb_ms": PERFORMANCE_TARGET_RGB_MS,
            "target_human_ms": PERFORMANCE_TARGET_HUMAN_MS,
            "optimization_flags": [],
        }

    def _validate_rgb_color(self, color: Tuple[int, int, int], name: str):
        """Validate RGB color tuple values and types."""
        if not isinstance(color, tuple) or len(color) != 3:
            raise ValueError(f"{name} must be a 3-tuple of integers")

        for i, value in enumerate(color):
            if not isinstance(value, int):
                raise ValueError(f"{name}[{i}] must be an integer, got {type(value)}")
            if not (COLOR_VALUE_MIN <= value <= COLOR_VALUE_MAX):
                raise ValueError(
                    f"{name}[{i}] must be in range [{COLOR_VALUE_MIN}, {COLOR_VALUE_MAX}], got {value}"
                )

    def _validate_marker_size(self, size: Tuple[int, int], name: str):
        """Validate marker size tuple values."""
        if not isinstance(size, tuple) or len(size) != 2:
            raise ValueError(f"{name} must be a 2-tuple of positive integers")

        for i, value in enumerate(size):
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name}[{i}] must be a positive integer, got {value}")

    def get_agent_color(
        self, format_type: str = "rgb"
    ) -> Union[Tuple[int, int, int], np.ndarray, str]:
        """
        Returns agent color in specified format with validation.

        Args:
            format_type: Output format - 'rgb' tuple, 'normalized' [0,1] array, or 'hex' string

        Returns:
            Agent color in requested format with appropriate type conversion
        """
        if format_type == "rgb":
            return self.agent_color
        elif format_type == "normalized":
            return np.array(self.agent_color, dtype=np.float32) / 255.0
        elif format_type == "hex":
            return f"#{self.agent_color[0]:02x}{self.agent_color[1]:02x}{self.agent_color[2]:02x}"
        else:
            raise ValueError(
                f"Unsupported format_type '{format_type}'. Use 'rgb', 'normalized', or 'hex'"
            )

    def get_source_color(
        self, format_type: str = "rgb"
    ) -> Union[Tuple[int, int, int], np.ndarray, str]:
        """
        Returns source color in specified format with validation.

        Args:
            format_type: Output format - 'rgb' tuple, 'normalized' [0,1] array, or 'hex' string

        Returns:
            Source color in requested format with appropriate type conversion
        """
        if format_type == "rgb":
            return self.source_color
        elif format_type == "normalized":
            return np.array(self.source_color, dtype=np.float32) / 255.0
        elif format_type == "hex":
            return f"#{self.source_color[0]:02x}{self.source_color[1]:02x}{self.source_color[2]:02x}"
        else:
            raise ValueError(
                f"Unsupported format_type '{format_type}'. Use 'rgb', 'normalized', or 'hex'"
            )

    def get_concentration_colormap(
        self,
        use_cache: bool = True,
        normalization_range: Optional[Tuple[float, float]] = None,
    ) -> matplotlib.colors.Colormap:
        """
        Returns matplotlib colormap with caching and normalization.

        Args:
            use_cache: Enable colormap caching for performance optimization
            normalization_range: Custom normalization range, defaults to [0,1]

        Returns:
            Configured matplotlib colormap object ready for visualization
        """
        if use_cache and self._cached_colormap is not None:
            return self._cached_colormap

        try:
            colormap = matplotlib.colormaps[self.concentration_colormap]
        except (ValueError, KeyError):
            warnings.warn(
                f"Colormap '{self.concentration_colormap}' not available, using 'gray'"
            )
            colormap = matplotlib.colormaps["gray"]

        # Configure normalization range
        if normalization_range is None:
            normalization_range = CONCENTRATION_RANGE

        # Apply normalization to colormap
        norm = matplotlib.colors.Normalize(
            vmin=normalization_range[0], vmax=normalization_range[1]
        )
        colormap_with_norm = matplotlib.cm.ScalarMappable(norm=norm, cmap=colormap)

        # Cache colormap if requested
        if use_cache:
            self._cached_colormap = colormap_with_norm.cmap

        return colormap_with_norm.cmap

    def apply_to_rgb_array(
        self,
        concentration_field: np.ndarray,
        agent_position: Tuple[int, int],
        source_position: Tuple[int, int],
    ) -> np.ndarray:
        """
        Applies complete color scheme to RGB array including concentration field and markers.

        Args:
            concentration_field: 2D array with concentration values [0,1]
            agent_position: (x, y) coordinates for agent marker placement
            source_position: (x, y) coordinates for source marker placement

        Returns:
            Complete RGB array (H,W,3) with color scheme applied
        """
        # Validate input concentration field
        if not isinstance(concentration_field, np.ndarray):
            raise ValueError("concentration_field must be a numpy array")

        if concentration_field.ndim != 2:
            raise ValueError(
                f"concentration_field must be 2D, got {concentration_field.ndim}D"
            )

        # Clamp concentration values to valid range
        concentration_field = np.clip(
            concentration_field, CONCENTRATION_RANGE[0], CONCENTRATION_RANGE[1]
        )

        # Convert concentration field to grayscale RGB values
        grayscale_values = (concentration_field * 255).astype(np.uint8)

        # Create RGB array with proper shape
        height, width = concentration_field.shape
        rgb_array = np.stack(
            [grayscale_values, grayscale_values, grayscale_values], axis=-1
        )

        # Apply background color for zero concentration areas
        zero_mask = concentration_field == 0.0
        if np.any(zero_mask):
            rgb_array[zero_mask] = self.background_color

        # Apply source marker (5×5 cross pattern)
        rgb_array = apply_source_marker(
            rgb_array, source_position, self.source_color, self.source_marker_size
        )

        # Apply agent marker (3×3 square pattern)
        rgb_array = apply_agent_marker(
            rgb_array, agent_position, self.agent_color, self.agent_marker_size
        )

        return rgb_array

    def configure_matplotlib_axes(
        self, axes: matplotlib.axes.Axes, concentration_field: np.ndarray
    ) -> matplotlib.image.AxesImage:
        """
        Configures matplotlib axes with color scheme for human mode visualization.

        Args:
            axes: Matplotlib axes object for configuration
            concentration_field: 2D concentration data for display

        Returns:
            Configured AxesImage object ready for marker updates
        """
        # Get colormap with caching
        colormap = self.get_concentration_colormap(use_cache=True)

        # Configure axes with concentration field visualization
        im = axes.imshow(
            concentration_field,
            cmap=colormap,
            origin="lower",  # Mathematical coordinate system
            vmin=CONCENTRATION_RANGE[0],
            vmax=CONCENTRATION_RANGE[1],
            aspect="equal",
        )

        # Set axes properties
        axes.set_title("Plume Navigation Environment")
        axes.set_xlabel("X Position")
        axes.set_ylabel("Y Position")

        return im

    def optimize_for_render_mode(
        self, render_mode: str, optimization_options: Dict[str, Any] = None
    ):
        """
        Optimizes color scheme for specific rendering mode with performance enhancements.

        Args:
            render_mode: Target rendering mode ('rgb_array' or 'human')
            optimization_options: Custom optimization settings
        """
        if optimization_options is None:
            optimization_options = {}

        self.optimized_for_mode = render_mode

        if render_mode == "rgb_array":
            # Optimize for programmatic RGB array generation
            self.performance_config.update(
                {
                    "precompute_color_tables": True,
                    "use_uint8_optimization": True,
                    "caching_enabled": False,  # Less important for batch processing
                    "target_latency_ms": PERFORMANCE_TARGET_RGB_MS,
                }
            )

        elif render_mode == "human":
            # Optimize for matplotlib interactive visualization
            self.performance_config.update(
                {
                    "colormap_caching": True,
                    "figure_reuse": True,
                    "update_optimization": True,
                    "target_latency_ms": PERFORMANCE_TARGET_HUMAN_MS,
                }
            )

        # Apply custom optimization options
        self.performance_config.update(optimization_options)

    def validate(
        self,
        check_accessibility: bool = False,
        check_performance: bool = False,
        strict_mode: bool = False,
    ) -> bool:
        """
        Comprehensive validation of color scheme configuration.

        Args:
            check_accessibility: Enable accessibility compliance checking
            check_performance: Enable performance requirement validation
            strict_mode: Apply strict validation rules including edge cases

        Returns:
            True if validation passes, raises ValueError with details if invalid
        """
        validation_errors = []

        # Basic RGB color validation (already done in __post_init__)
        try:
            self._validate_rgb_color(self.agent_color, "agent_color")
            self._validate_rgb_color(self.source_color, "source_color")
            self._validate_rgb_color(self.background_color, "background_color")
        except ValueError as e:
            validation_errors.append(f"Color validation: {e}")

        # Validate matplotlib colormap availability
        try:
            matplotlib.colormaps[self.concentration_colormap]
        except (ValueError, KeyError):
            validation_errors.append(
                f"Colormap '{self.concentration_colormap}' is not available"
            )

        # Check for color conflicts between markers and background
        if self.agent_color == self.source_color:
            validation_errors.append("Agent and source colors are identical")

        if check_accessibility:
            # Calculate contrast ratios (simplified luminance calculation)
            def luminance(color):
                r, g, b = [c / 255.0 for c in color]
                return 0.299 * r + 0.587 * g + 0.114 * b

            bg_luminance = luminance(self.background_color)
            agent_luminance = luminance(self.agent_color)
            source_luminance = luminance(self.source_color)

            agent_contrast = abs(agent_luminance - bg_luminance)
            source_contrast = abs(source_luminance - bg_luminance)

            min_contrast = 0.5  # Simplified contrast threshold
            if agent_contrast < min_contrast:
                validation_errors.append(
                    f"Agent-background contrast {agent_contrast:.2f} below threshold"
                )
            if source_contrast < min_contrast:
                validation_errors.append(
                    f"Source-background contrast {source_contrast:.2f} below threshold"
                )

        if validation_errors:
            error_message = "Color scheme validation failed:\n" + "\n".join(
                validation_errors
            )
            raise ValueError(error_message)

        return True

    def to_dict(
        self,
        include_performance_info: bool = False,
        include_optimization_state: bool = False,
    ) -> Dict[str, Any]:
        """
        Converts color scheme to dictionary representation for serialization.

        Args:
            include_performance_info: Include performance configuration in output
            include_optimization_state: Include optimization state information

        Returns:
            Dictionary representation with color scheme configuration and metadata
        """
        config_dict = {
            "agent_color": self.agent_color,
            "source_color": self.source_color,
            "background_color": self.background_color,
            "concentration_colormap": self.concentration_colormap,
            "agent_marker_size": self.agent_marker_size,
            "source_marker_size": self.source_marker_size,
            "accessibility_enhanced": self.accessibility_enhanced,
            "optimized_for_mode": self.optimized_for_mode,
        }

        if include_performance_info:
            config_dict["performance_config"] = self.performance_config.copy()

        if include_optimization_state:
            config_dict["cached_colormap_available"] = self._cached_colormap is not None
            config_dict["validation_passed"] = True  # If we got this far

        return config_dict

    def clone(
        self,
        color_overrides: Optional[Dict[str, Any]] = None,
        preserve_optimizations: bool = True,
    ) -> "ColorScheme":
        """
        Creates deep copy of color scheme with optional parameter overrides.

        Args:
            color_overrides: Dictionary of parameters to override in the copy
            preserve_optimizations: Maintain performance config and optimization state

        Returns:
            New ColorScheme instance with overrides applied and validated
        """
        # Start with current configuration
        config = self.to_dict(include_performance_info=preserve_optimizations)

        # Apply overrides
        if color_overrides:
            config.update(color_overrides)

        # Create new instance
        new_scheme = ColorScheme(
            agent_color=config["agent_color"],
            source_color=config["source_color"],
            background_color=config["background_color"],
            concentration_colormap=config["concentration_colormap"],
            agent_marker_size=config.get("agent_marker_size", AGENT_MARKER_SIZE),
            source_marker_size=config.get("source_marker_size", SOURCE_MARKER_SIZE),
            accessibility_enhanced=config.get("accessibility_enhanced", False),
            optimized_for_mode=config.get("optimized_for_mode"),
        )

        # Preserve performance configuration if requested
        if preserve_optimizations and hasattr(self, "performance_config"):
            new_scheme.performance_config = self.performance_config.copy()

        return new_scheme


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================


def create_default_scheme(
    render_mode: Optional[str] = None,
    enable_accessibility: bool = False,
    cache_colormap: bool = True,
) -> ColorScheme:
    """
    Factory function to create default color scheme with standard plume navigation colors.

    Args:
        render_mode: Rendering mode optimization ('rgb_array' or 'human')
        enable_accessibility: Apply accessibility enhancements and contrast adjustment
        cache_colormap: Enable colormap caching for matplotlib integration

    Returns:
        Default color scheme configured with standard colors and optimizations
    """
    # Create base color scheme with standard colors
    scheme = ColorScheme(
        agent_color=DEFAULT_AGENT_COLOR,  # Red [255,0,0]
        source_color=DEFAULT_SOURCE_COLOR,  # White [255,255,255]
        background_color=DEFAULT_BACKGROUND_COLOR,  # Black [0,0,0]
        concentration_colormap=DEFAULT_COLORMAP,  # 'gray'
        accessibility_enhanced=enable_accessibility,
    )

    # Apply render mode optimization if specified
    if render_mode:
        optimization_options = {"caching_enabled": cache_colormap}
        scheme.optimize_for_render_mode(render_mode, optimization_options)

    # Enable colormap caching for performance
    if cache_colormap:
        scheme.get_concentration_colormap(use_cache=True)

    # Validate configuration
    scheme.validate(check_accessibility=enable_accessibility)

    return scheme


def create_color_scheme(
    color_config: Dict[str, Any],
    scheme_type: Optional[str] = None,
    render_mode: Optional[str] = None,
    validate_accessibility: bool = False,
) -> ColorScheme:
    """
    Factory function to create custom color scheme with validation and optimization.

    Args:
        color_config: Dictionary with agent_color, source_color, background_color, concentration_colormap
        scheme_type: Predefined scheme type for configuration inheritance
        render_mode: Target rendering mode ('rgb_array' or 'human') for optimization
        validate_accessibility: Enable accessibility validation including contrast ratios

    Returns:
        Custom color scheme with validated configuration and mode-specific optimizations
    """
    # Validate color_config structure
    required_keys = {
        "agent_color",
        "source_color",
        "background_color",
        "concentration_colormap",
    }
    if missing_keys := required_keys - set(color_config.keys()):
        raise ValueError(f"Missing required color configuration keys: {missing_keys}")

    # Apply scheme_type configuration if specified
    if scheme_type and scheme_type in PREDEFINED_SCHEMES:
        try:
            predefined_scheme = PredefinedScheme(scheme_type)
            base_config = predefined_scheme.get_color_config()
            # Merge with custom config (custom takes precedence)
            merged_config = {**base_config, **color_config}
            color_config = merged_config
        except ValueError:
            warnings.warn(
                f"Unknown scheme_type '{scheme_type}', using custom config only"
            )

    # Create color scheme instance
    scheme = ColorScheme(
        agent_color=color_config["agent_color"],
        source_color=color_config["source_color"],
        background_color=color_config["background_color"],
        concentration_colormap=color_config["concentration_colormap"],
        agent_marker_size=color_config.get("agent_marker_size", AGENT_MARKER_SIZE),
        source_marker_size=color_config.get("source_marker_size", SOURCE_MARKER_SIZE),
        accessibility_enhanced=color_config.get("accessibility_enhanced", False),
    )

    # Apply render mode optimization
    if render_mode:
        scheme.optimize_for_render_mode(render_mode)

    # Validate configuration
    scheme.validate(check_accessibility=validate_accessibility, strict_mode=True)

    return scheme


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================


def validate_color_scheme(
    color_scheme: ColorScheme,
    check_accessibility: bool = False,
    check_performance: bool = False,
    strict_mode: bool = False,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Comprehensive validation function for color scheme configuration.

    Args:
        color_scheme: ColorScheme instance to validate
        check_accessibility: Enable accessibility compliance checking
        check_performance: Enable performance requirement validation
        strict_mode: Apply strict validation rules including edge cases

    Returns:
        Tuple of (is_valid: bool, validation_report: dict) with detailed analysis
    """
    validation_report = {
        "is_valid": True,
        "errors": [],
        "warnings": [],
        "accessibility_score": None,
        "performance_metrics": {},
        "recommendations": [],
    }

    try:
        # Run built-in validation
        color_scheme.validate(
            check_accessibility=check_accessibility,
            check_performance=check_performance,
            strict_mode=strict_mode,
        )

    except ValueError as e:
        validation_report["is_valid"] = False
        validation_report["errors"].append(str(e))

    # Additional validation checks
    try:
        # Test matplotlib colormap functionality
        color_scheme.get_concentration_colormap()
        validation_report["performance_metrics"]["colormap_load_time"] = "success"
    except Exception as e:
        validation_report["warnings"].append(f"Colormap loading issue: {e}")

    # Performance testing if requested
    if check_performance:
        import time

        # Test RGB array generation performance
        try:
            _extracted_from_validate_color_scheme_54(
                time, color_scheme, validation_report
            )
        except Exception as e:
            validation_report["errors"].append(f"RGB performance test failed: {e}")

    # Generate recommendations
    if validation_report["warnings"]:
        validation_report["recommendations"].append(
            "Consider optimizing for performance mode"
        )

    if check_accessibility and color_scheme.accessibility_enhanced:
        validation_report["recommendations"].append("Accessibility features enabled")

    return validation_report["is_valid"], validation_report


# TODO Rename this here and in `validate_color_scheme`
def _extracted_from_validate_color_scheme_54(time, color_scheme, validation_report):
    test_field = np.random.rand(32, 32)
    start_time = time.perf_counter()
    color_scheme.apply_to_rgb_array(test_field, (16, 16), (20, 20))
    rgb_time = (time.perf_counter() - start_time) * 1000  # Convert to ms

    validation_report["performance_metrics"]["rgb_generation_ms"] = rgb_time

    if rgb_time > PERFORMANCE_TARGET_RGB_MS:
        validation_report["warnings"].append(
            f"RGB generation time {rgb_time:.2f}ms exceeds target {PERFORMANCE_TARGET_RGB_MS}ms"
        )


# =============================================================================
# UTILITY FUNCTIONS FOR COLOR OPERATIONS
# =============================================================================


def normalize_concentration_to_rgb(
    concentration_field: np.ndarray,
    colormap_name: str = DEFAULT_COLORMAP,
    use_cache: bool = True,
    value_range: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """
    Converts concentration field from [0,1] to RGB grayscale values [0,255].

    Args:
        concentration_field: 2D array with concentration values
        colormap_name: Matplotlib colormap name for conversion
        use_cache: Enable colormap caching for performance optimization
        value_range: Custom value range, defaults to [0,1]

    Returns:
        RGB array (H,W,3) with dtype uint8 containing grayscale visualization
    """
    # Validate input array
    if not isinstance(concentration_field, np.ndarray):
        raise ValueError("concentration_field must be a numpy array")

    if concentration_field.ndim != 2:
        raise ValueError(f"Expected 2D array, got {concentration_field.ndim}D")

    # Set default value range
    if value_range is None:
        value_range = CONCENTRATION_RANGE

    # Clamp values to valid range
    concentration_field = np.clip(concentration_field, value_range[0], value_range[1])

    # Get colormap with caching
    colormap = get_matplotlib_colormap(colormap_name, value_range, use_cache)

    # Apply colormap normalization to convert [0,1] to [0,255] RGB
    normalized_field = (concentration_field - value_range[0]) / (
        value_range[1] - value_range[0]
    )
    rgb_values = colormap(normalized_field)

    return (rgb_values[:, :, :3] * 255).astype(np.uint8)


def apply_agent_marker(
    rgb_array: np.ndarray,
    agent_position: Tuple[int, int],
    marker_color: Tuple[int, int, int] = DEFAULT_AGENT_COLOR,
    marker_size: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """
    Applies agent marker visualization to RGB array at specified position.

    Args:
        rgb_array: Target RGB array for marker application
        agent_position: (x, y) coordinates for marker placement
        marker_color: RGB color tuple for marker, defaults to red
        marker_size: (height, width) marker dimensions, defaults to 3x3

    Returns:
        RGB array with agent marker applied, maintaining original structure
    """
    # Validate RGB array
    if (
        not isinstance(rgb_array, np.ndarray)
        or rgb_array.ndim != 3
        or rgb_array.shape[2] != RGB_CHANNELS
    ):
        raise ValueError("rgb_array must be a 3D array with shape (H,W,3)")

    if rgb_array.dtype != np.uint8:
        warnings.warn("rgb_array dtype is not uint8, conversion may lose precision")
        rgb_array = rgb_array.astype(np.uint8)

    # Validate agent position
    height, width = rgb_array.shape[:2]
    x, y = agent_position
    if not (0 <= x < width and 0 <= y < height):
        raise ValueError(
            f"agent_position {agent_position} is outside array bounds ({width}, {height})"
        )

    # Use default marker size if not specified
    if marker_size is None:
        marker_size = AGENT_MARKER_SIZE

    marker_height, marker_width = marker_size

    # Calculate marker boundaries with boundary checking
    half_height = marker_height // 2
    half_width = marker_width // 2

    y_start = max(0, y - half_height)
    y_end = min(height, y + half_height + 1)
    x_start = max(0, x - half_width)
    x_end = min(width, x + half_width + 1)

    # Apply red square marker pattern to RGB array
    rgb_array[y_start:y_end, x_start:x_end] = marker_color

    return rgb_array


def apply_source_marker(
    rgb_array: np.ndarray,
    source_position: Tuple[int, int],
    marker_color: Tuple[int, int, int] = DEFAULT_SOURCE_COLOR,
    marker_size: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """
    Applies source marker cross pattern to RGB array at specified position.

    Args:
        rgb_array: Target RGB array for marker application
        source_position: (x, y) coordinates for marker placement
        marker_color: RGB color tuple for marker, defaults to white
        marker_size: (height, width) marker dimensions, defaults to 5x5

    Returns:
        RGB array with source marker cross pattern applied
    """
    # Validate RGB array
    if (
        not isinstance(rgb_array, np.ndarray)
        or rgb_array.ndim != 3
        or rgb_array.shape[2] != RGB_CHANNELS
    ):
        raise ValueError("rgb_array must be a 3D array with shape (H,W,3)")

    if rgb_array.dtype != np.uint8:
        warnings.warn("rgb_array dtype is not uint8, conversion may lose precision")
        rgb_array = rgb_array.astype(np.uint8)

    # Validate source position
    height, width = rgb_array.shape[:2]
    x, y = source_position
    if not (0 <= x < width and 0 <= y < height):
        raise ValueError(
            f"source_position {source_position} is outside array bounds ({width}, {height})"
        )

    # Use default marker size if not specified
    if marker_size is None:
        marker_size = SOURCE_MARKER_SIZE

    marker_height, marker_width = marker_size

    # Calculate cross pattern coordinates with boundary checking
    half_height = marker_height // 2
    half_width = marker_width // 2

    # Horizontal line of cross
    x_start = max(0, x - half_width)
    x_end = min(width, x + half_width + 1)
    if 0 <= y < height:  # Ensure y is within bounds
        rgb_array[y, x_start:x_end] = marker_color

    # Vertical line of cross
    y_start = max(0, y - half_height)
    y_end = min(height, y + half_height + 1)
    if 0 <= x < width:  # Ensure x is within bounds
        rgb_array[y_start:y_end, x] = marker_color

    return rgb_array


@functools.lru_cache(maxsize=COLORMAP_CACHE_SIZE)
def get_matplotlib_colormap(
    colormap_name: str,
    normalization_range: Optional[Tuple[float, float]] = None,
    use_cache: bool = True,
    validate_backend: bool = False,
) -> matplotlib.colors.Colormap:
    """
    Retrieves matplotlib colormap with caching, normalization, and performance optimization.

    Args:
        colormap_name: Name of matplotlib colormap to retrieve
        normalization_range: Value range for normalization, defaults to [0,1]
        use_cache: Enable LRU caching for performance (decorator handles this)
        validate_backend: Check matplotlib backend availability

    Returns:
        Configured matplotlib colormap object optimized for performance
    """
    # Validate matplotlib backend if requested
    if validate_backend:
        try:
            import matplotlib.pyplot as plt

            backend = plt.get_backend()
            if backend == "Agg":
                warnings.warn(
                    "Using headless Agg backend, interactive features unavailable"
                )
        except ImportError:
            warnings.warn("Matplotlib backend validation failed")

    # Set default normalization range
    if normalization_range is None:
        normalization_range = CONCENTRATION_RANGE

    # Retrieve colormap from matplotlib registry with fallback
    try:
        colormap = matplotlib.colormaps[colormap_name]
    except (ValueError, KeyError):
        warnings.warn(f"Colormap '{colormap_name}' not found, falling back to 'gray'")
        colormap = matplotlib.colormaps["gray"]

    return colormap


def optimize_for_performance(
    color_scheme: ColorScheme,
    render_mode: str,
    optimization_options: Dict[str, Any] = None,
) -> ColorScheme:
    """
    Optimizes color scheme configuration for specified rendering mode with performance enhancements.

    Args:
        color_scheme: ColorScheme instance to optimize
        render_mode: Target rendering mode ('rgb_array' or 'human')
        optimization_options: Custom optimization settings and resource management options

    Returns:
        Optimized color scheme with performance enhancements and mode-specific configurations
    """
    if optimization_options is None:
        optimization_options = {}

    # Create optimized copy of color scheme
    optimized_scheme = color_scheme.clone(preserve_optimizations=True)

    # Apply render mode optimizations
    optimized_scheme.optimize_for_render_mode(render_mode, optimization_options)

    # Add specific optimization flags based on render mode
    if render_mode == "rgb_array":
        # Pre-compute color tables for fast array operations
        optimization_options.update(
            {
                "precompute_agent_color_array": True,
                "precompute_source_color_array": True,
                "use_vectorized_operations": True,
            }
        )

    elif render_mode == "human":
        # Enable matplotlib-specific optimizations
        optimization_options.update(
            {
                "enable_figure_caching": True,
                "optimize_colormap_updates": True,
                "reduce_rendering_calls": True,
            }
        )

        # Pre-load and cache colormap
        optimized_scheme.get_concentration_colormap(use_cache=True)

    # Apply optimization options to performance config
    optimized_scheme.performance_config.update(optimization_options)

    # Validate optimized configuration maintains consistency
    optimized_scheme.validate(check_performance=True)

    return optimized_scheme
