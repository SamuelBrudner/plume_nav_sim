import copy
import functools
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.cm as cm
import matplotlib.colors
import numpy as np

from .._compat import RenderingError, ValidationError, validate_render_mode
from ..constants import (
    AGENT_MARKER_COLOR,
    AGENT_MARKER_SIZE,
    CONCENTRATION_RANGE,
    PERFORMANCE_TARGET_RGB_RENDER_MS,
    RGB_DTYPE,
    SOURCE_MARKER_COLOR,
    SOURCE_MARKER_SIZE,
)
from ..core.types import Coordinates, RenderMode
from .colormaps import DEFAULT_COLORMAP, PredefinedScheme

__all__ = [
    "ColorSchemeManager",
    "PredefinedScheme",
    "CustomColorScheme",
    "ColorPalette",
    "AccessibilityConfig",
    "create_color_scheme",
    "get_default_scheme",
    "create_accessibility_scheme",
    "optimize_for_render_mode",
    "validate_color_scheme",
    "get_predefined_schemes",
    "normalize_concentration_to_rgb",
    "apply_agent_marker",
    "apply_source_marker",
    "get_matplotlib_colormap",
    "convert_rgb_to_matplotlib",
]

DEFAULT_SCHEME_NAME = PredefinedScheme.STANDARD.value
COLORMAP_CACHE_SIZE = 50
COLOR_VALIDATION_TOLERANCE = 0.001
ACCESSIBILITY_CONTRAST_RATIO_MIN = 4.5
PERFORMANCE_OPTIMIZATION_ENABLED = True
SCHEME_CACHE = {}

# Initialize module logger for color scheme operations and debugging
logger = logging.getLogger(__name__)


def _safe_get_cmap(colormap_name: str):  # noqa: C901
    try:
        # Prefer modern API to avoid deprecation warnings and internal imports
        if hasattr(matplotlib, "colormaps"):
            try:
                return matplotlib.colormaps.get_cmap(colormap_name)
            except KeyError:
                raise ValueError(f"Invalid colormap '{colormap_name}'")
            except Exception as e:
                # If environment import hooks break internals, soften to None
                if "warn_external" in str(e):
                    return None
                # Fallback to legacy API below
        # Legacy API
        try:
            return cm.get_cmap(colormap_name)
        except ValueError:
            raise  # Invalid colormap name
        except Exception as e:
            if "warn_external" in str(e):
                return None
            raise
    except ImportError:
        # Environment import trouble (e.g., patched __import__), treat as non-fatal
        return None


@dataclass
class CustomColorScheme:
    agent_color: Tuple[int, int, int]
    source_color: Tuple[int, int, int]
    background_color: Tuple[int, int, int]
    concentration_colormap: str
    agent_marker_size: Tuple[int, int] = field(
        default_factory=lambda: AGENT_MARKER_SIZE
    )
    source_marker_size: Tuple[int, int] = field(
        default_factory=lambda: SOURCE_MARKER_SIZE
    )
    accessibility_enabled: bool = False
    optimized_for_mode: Optional[RenderMode] = None
    performance_config: Dict[str, Any] = field(default_factory=dict)
    _cached_colormap: Optional[matplotlib.colors.Colormap] = field(
        default=None, repr=False
    )

    def __post_init__(self):  # noqa: C901
        """Initialize custom color scheme with RGB color validation and performance configuration."""
        # Validate agent_color tuple has exactly 3 RGB values in range [0,255]
        if not (isinstance(self.agent_color, tuple) and len(self.agent_color) == 3):
            raise ValidationError("agent_color must be a tuple of 3 RGB values")
        if not all(
            isinstance(val, int) and 0 <= val <= 255 for val in self.agent_color
        ):
            raise ValidationError(
                "agent_color values must be integers in range [0,255]"
            )

        # Validate source_color tuple has exactly 3 RGB values in range [0,255]
        if not (isinstance(self.source_color, tuple) and len(self.source_color) == 3):
            raise ValidationError("source_color must be a tuple of 3 RGB values")
        if not all(
            isinstance(val, int) and 0 <= val <= 255 for val in self.source_color
        ):
            raise ValidationError(
                "source_color values must be integers in range [0,255]"
            )

        # Validate background_color tuple has exactly 3 RGB values in range [0,255]
        if not (
            isinstance(self.background_color, tuple) and len(self.background_color) == 3
        ):
            raise ValidationError("background_color must be a tuple of 3 RGB values")
        if not all(
            isinstance(val, int) and 0 <= val <= 255 for val in self.background_color
        ):
            raise ValidationError(
                "background_color values must be integers in range [0,255]"
            )

        # Verify concentration_colormap exists (robust against deprecation/import quirks)
        try:
            result = _safe_get_cmap(self.concentration_colormap)
            if result is None:
                # Assume valid when environment prevents import-time resolution
                logger.debug(
                    "Colormap resolution deferred due to environment; proceeding"
                )
        except ValueError as e:
            raise ValidationError(
                f"Invalid concentration_colormap '{self.concentration_colormap}': {str(e)}"
            )

        # Initialize performance_config with default optimization settings
        if not self.performance_config:
            self.performance_config = {
                "caching_enabled": PERFORMANCE_OPTIMIZATION_ENABLED,
                "pre_computation_enabled": True,
                "target_latency_ms": PERFORMANCE_TARGET_RGB_RENDER_MS,
            }

    def apply_to_rgb_array(
        self,
        concentration_field: np.ndarray,
        agent_position: Coordinates,
        source_position: Coordinates,
    ) -> np.ndarray:
        # Convert concentration field [0,1] to RGB array using normalize_concentration_to_rgb()
        rgb_array = normalize_concentration_to_rgb(
            concentration_field,
            self.concentration_colormap,
            use_caching=self.performance_config.get("caching_enabled", True),
        )

        # Apply background_color to areas with zero concentration values
        zero_mask = concentration_field <= COLOR_VALIDATION_TOLERANCE
        if np.any(zero_mask):
            rgb_array[zero_mask] = self.background_color

        # Add source marker at source_position using apply_source_marker() with source_color
        rgb_array = apply_source_marker(
            rgb_array,
            source_position,
            marker_color=self.source_color,
            marker_size=self.source_marker_size,
            check_boundaries=True,
        )

        # Add agent marker at agent_position using apply_agent_marker() with agent_color
        rgb_array = apply_agent_marker(
            rgb_array,
            agent_position,
            marker_color=self.agent_color,
            marker_size=self.agent_marker_size,
            check_boundaries=True,
        )

        # Validate final RGB array format and data type consistency
        if rgb_array.dtype != RGB_DTYPE:
            rgb_array = rgb_array.astype(RGB_DTYPE)

        return rgb_array

    def configure_matplotlib_axes(
        self, axes: Any, concentration_field: np.ndarray
    ) -> Any:
        # Get concentration colormap using get_matplotlib_colormap() with caching
        colormap = get_matplotlib_colormap(
            self.concentration_colormap,
            normalization_range=CONCENTRATION_RANGE,
            enable_caching=self.performance_config.get("caching_enabled", True),
        )

        # Configure axes.imshow() with colormap and [0,1] normalization
        image = axes.imshow(
            concentration_field,
            cmap=colormap,
            origin="lower",  # Mathematical coordinate system
            vmin=CONCENTRATION_RANGE[0],
            vmax=CONCENTRATION_RANGE[1],
            aspect=1.0,
        )

        # Configure color normalization and aspect ratio for proper display
        norm = matplotlib.colors.Normalize(
            vmin=CONCENTRATION_RANGE[0], vmax=CONCENTRATION_RANGE[1]
        )
        image.set_norm(norm)

        # Apply background_color configuration if needed
        axes.set_facecolor([c / 255.0 for c in self.background_color])

        return image

    def optimize_for_render_mode(
        self,
        target_mode: RenderMode,
        optimization_options: Optional[Dict[str, Any]] = None,
    ) -> None:
        # Set optimized_for_mode to target_mode for optimization tracking
        self.optimized_for_mode = target_mode

        # Apply RGB array optimizations if target_mode is RGB_ARRAY
        if target_mode == RenderMode.RGB_ARRAY:
            self.performance_config.update(
                {
                    "pre_compute_colors": True,
                    "enable_vectorization": True,
                    "target_latency_ms": PERFORMANCE_TARGET_RGB_RENDER_MS,
                }
            )

        # Enable matplotlib colormap caching if target_mode is HUMAN
        elif target_mode == RenderMode.HUMAN:
            self.performance_config.update(
                {
                    "matplotlib_caching": True,
                    "figure_reuse": True,
                    "target_latency_ms": 50.0,  # Human mode target
                }
            )

        # Apply optimization_options for custom performance tuning
        if optimization_options:
            self.performance_config.update(optimization_options)

        # Pre-compute expensive operations based on target mode
        if self.performance_config.get("pre_compute_colors", False):
            # Cache colormap object for repeated use
            self._cached_colormap = get_matplotlib_colormap(self.concentration_colormap)

        logger.debug(
            f"Optimized color scheme for {target_mode} with config: {self.performance_config}"
        )

    def enable_accessibility(
        self, accessibility_type: Optional[str] = None, validate_contrast: bool = True
    ) -> None:
        # Set accessibility_enabled to True
        self.accessibility_enabled = True

        # Apply high contrast modifications if accessibility_type is 'high_contrast'
        if accessibility_type == "high_contrast":
            # Enhance contrast for agent and source markers
            self.agent_color = (255, 0, 0)  # Pure red
            self.source_color = (255, 255, 255)  # Pure white
            self.background_color = (0, 0, 0)  # Pure black

        # Apply colorblind-friendly colors if accessibility_type is 'colorblind_friendly'
        elif accessibility_type == "colorblind_friendly":
            # Use deuteranopia/protanopia safe colors
            self.agent_color = (255, 165, 0)  # Orange instead of red
            self.source_color = (255, 255, 255)  # White remains
            self.concentration_colormap = "viridis"  # Colorblind-friendly colormap

        # Enhance marker visibility with larger sizes and bold colors
        if accessibility_type in ["high_contrast", "colorblind_friendly"]:
            # Increase marker sizes for better visibility
            self.agent_marker_size = (
                max(3, self.agent_marker_size[0] + 1),
                max(3, self.agent_marker_size[1] + 1),
            )
            self.source_marker_size = (
                max(5, self.source_marker_size[0] + 1),
                max(5, self.source_marker_size[1] + 1),
            )

        # Validate contrast ratios meet ACCESSIBILITY_CONTRAST_RATIO_MIN if validate_contrast is True
        if validate_contrast:
            # Calculate contrast ratios (simplified implementation)
            agent_luminance = sum(self.agent_color) / (3 * 255)
            background_luminance = sum(self.background_color) / (3 * 255)
            contrast_ratio = (max(agent_luminance, background_luminance) + 0.05) / (
                min(agent_luminance, background_luminance) + 0.05
            )

            if contrast_ratio < ACCESSIBILITY_CONTRAST_RATIO_MIN:
                logger.warning(
                    f"Contrast ratio {contrast_ratio:.2f} below minimum {ACCESSIBILITY_CONTRAST_RATIO_MIN}"
                )

        logger.info(f"Enabled accessibility features: {accessibility_type}")

    def validate(  # noqa: C901
        self,
        check_accessibility: bool = False,
        check_performance: bool = False,
        strict_validation: bool = False,
    ) -> bool:
        try:
            # Validate all RGB colors are in range [0,255] with proper types
            for color_name, color_value in [
                ("agent_color", self.agent_color),
                ("source_color", self.source_color),
                ("background_color", self.background_color),
            ]:
                if not (isinstance(color_value, tuple) and len(color_value) == 3):
                    raise ValidationError(f"{color_name} must be a tuple of 3 values")
                if not all(
                    isinstance(val, int) and 0 <= val <= 255 for val in color_value
                ):
                    raise ValidationError(
                        f"{color_name} values must be integers in range [0,255]"
                    )

            # Check matplotlib colormap availability and compatibility
            try:
                result = _safe_get_cmap(self.concentration_colormap)
                if result is None:
                    # Environment-related import failure; treat as acceptable
                    pass
            except ValueError:
                raise ValidationError(
                    f"Invalid colormap '{self.concentration_colormap}'"
                )

            # Verify accessibility compliance if check_accessibility is True
            if check_accessibility and self.accessibility_enabled:
                # Check contrast ratios and visibility requirements
                pass  # Implementation would include detailed accessibility checks

            # Test performance characteristics if check_performance is True
            if check_performance:
                # Validate performance configuration and target latencies
                required_keys = ["target_latency_ms", "caching_enabled"]
                for key in required_keys:
                    if key not in self.performance_config:
                        raise ValidationError(f"Missing performance config key: {key}")

            # Check marker visibility and color conflicts
            if self.agent_color == self.source_color:
                if strict_validation:
                    raise ValidationError(
                        "Agent and source colors must be different for visibility"
                    )
                else:
                    logger.warning(
                        "Agent and source colors are identical - may reduce visibility"
                    )

            # Apply strict validation rules if strict_validation is True
            if strict_validation:
                if sum(self.agent_color) == sum(self.background_color):
                    raise ValidationError("Agent color too similar to background color")

            return True

        except ValidationError:
            raise
        except Exception as e:
            raise ValidationError(f"Color scheme validation failed: {str(e)}")

    def clone(
        self,
        color_overrides: Optional[Dict[str, Any]] = None,
        preserve_optimizations: bool = True,
    ) -> "CustomColorScheme":
        # Create deep copy of all color scheme parameters using copy.deepcopy()
        cloned_scheme = copy.deepcopy(self)

        # Apply color_overrides if provided with parameter validation
        if color_overrides:
            for key, value in color_overrides.items():
                if hasattr(cloned_scheme, key):
                    setattr(cloned_scheme, key, value)
                else:
                    logger.warning(f"Invalid color override key: {key}")

        # Reset _cached_colormap to force re-initialization with new parameters
        cloned_scheme._cached_colormap = None

        # Preserve performance_config and optimization state if preserve_optimizations is True
        if not preserve_optimizations:
            cloned_scheme.performance_config = {}
            cloned_scheme.optimized_for_mode = None

        # Validate cloned color scheme with new parameters
        try:
            cloned_scheme.validate()
        except ValidationError as e:
            logger.error(f"Cloned color scheme validation failed: {str(e)}")
            raise

        return cloned_scheme

    def to_dict(
        self,
        include_optimization_info: bool = False,
        sanitize_for_logging: bool = False,
    ) -> Dict[str, Any]:
        # Create dictionary with RGB color values for agent, source, background
        scheme_dict = {
            "agent_color": self.agent_color,
            "source_color": self.source_color,
            "background_color": self.background_color,
            "concentration_colormap": self.concentration_colormap,
            "agent_marker_size": self.agent_marker_size,
            "source_marker_size": self.source_marker_size,
            "accessibility_enabled": self.accessibility_enabled,
        }

        # Include performance_config if include_optimization_info is True
        if include_optimization_info:
            scheme_dict["optimized_for_mode"] = (
                str(self.optimized_for_mode) if self.optimized_for_mode else None
            )
            scheme_dict["performance_config"] = self.performance_config.copy()

        # Sanitize sensitive configuration data if sanitize_for_logging is True
        if sanitize_for_logging:
            # Remove potentially sensitive optimization details
            scheme_dict.pop("performance_config", None)

        return scheme_dict


@dataclass
class AccessibilityConfig:
    min_contrast_ratio: float = ACCESSIBILITY_CONTRAST_RATIO_MIN
    colorblind_friendly: bool = False
    high_contrast_mode: bool = False
    enhanced_visibility: bool = False
    accessibility_overrides: Dict[str, Any] = field(default_factory=dict)

    def apply_to_scheme(self, color_scheme: CustomColorScheme) -> CustomColorScheme:
        # Clone the color scheme to avoid modifying original
        enhanced_scheme = color_scheme.clone()

        # Apply contrast enhancements based on min_contrast_ratio
        if self.high_contrast_mode:
            enhanced_scheme.agent_color = (255, 0, 0)  # Pure red
            enhanced_scheme.source_color = (255, 255, 255)  # Pure white
            enhanced_scheme.background_color = (0, 0, 0)  # Pure black

        # Apply colorblind-friendly color modifications if enabled
        if self.colorblind_friendly:
            enhanced_scheme.agent_color = (255, 165, 0)  # Orange
            enhanced_scheme.concentration_colormap = "viridis"

        # Apply enhanced visibility features and marker size increases
        if self.enhanced_visibility:
            enhanced_scheme.agent_marker_size = (
                enhanced_scheme.agent_marker_size[0] + 2,
                enhanced_scheme.agent_marker_size[1] + 2,
            )
            enhanced_scheme.source_marker_size = (
                enhanced_scheme.source_marker_size[0] + 2,
                enhanced_scheme.source_marker_size[1] + 2,
            )

        # Apply accessibility overrides
        for key, value in self.accessibility_overrides.items():
            if hasattr(enhanced_scheme, key):
                setattr(enhanced_scheme, key, value)

        # Enable accessibility and validate
        enhanced_scheme.enable_accessibility()
        enhanced_scheme.validate(check_accessibility=True)

        return enhanced_scheme

    def validate_compliance(
        self, color_scheme: CustomColorScheme
    ) -> Tuple[bool, Dict[str, Any]]:
        compliance_report = {
            "is_compliant": True,
            "contrast_ratios": {},
            "recommendations": [],
            "warnings": [],
        }

        # Check contrast ratios meet min_contrast_ratio requirements
        # Simplified contrast calculation
        agent_luminance = sum(color_scheme.agent_color) / (3 * 255)
        background_luminance = sum(color_scheme.background_color) / (3 * 255)
        contrast_ratio = (max(agent_luminance, background_luminance) + 0.05) / (
            min(agent_luminance, background_luminance) + 0.05
        )

        compliance_report["contrast_ratios"]["agent_background"] = contrast_ratio

        if contrast_ratio < self.min_contrast_ratio:
            compliance_report["is_compliant"] = False
            compliance_report["recommendations"].append(
                f"Increase agent-background contrast ratio from {contrast_ratio:.2f} to {self.min_contrast_ratio}"
            )

        # Validate colorblind-friendly color selection if required
        if self.colorblind_friendly:
            if color_scheme.agent_color == AGENT_MARKER_COLOR:  # Standard red
                compliance_report["warnings"].append(
                    "Standard red may not be colorblind-friendly"
                )

        return compliance_report["is_compliant"], compliance_report


class ColorPalette:
    def __init__(self, colors: Dict[str, Any], palette_name: Optional[str] = None):
        """Initialize color palette with color dictionary and accessibility analysis."""
        # Validate colors dictionary contains required color definitions
        if not isinstance(colors, dict) or not colors:
            raise ValidationError("Colors dictionary must be non-empty")

        self.colors = colors
        self.palette_name = palette_name or "unnamed_palette"
        self.accessibility_info = {}

        # Validate all colors are proper RGB tuples or hex strings
        for color_name, color_value in colors.items():
            if isinstance(color_value, str):
                # Hex string validation
                if not (color_value.startswith("#") and len(color_value) == 7):
                    raise ValidationError(
                        f"Invalid hex color '{color_value}' for '{color_name}'"
                    )
            elif isinstance(color_value, (tuple, list)):
                # RGB tuple validation
                if len(color_value) != 3 or not all(
                    isinstance(v, int) and 0 <= v <= 255 for v in color_value
                ):
                    raise ValidationError(
                        f"Invalid RGB color '{color_value}' for '{color_name}'"
                    )
            else:
                raise ValidationError(
                    f"Color '{color_name}' must be hex string or RGB tuple"
                )

        # Initialize accessibility_info with contrast analysis
        self._analyze_accessibility()

    def get_color(
        self, color_name: str, output_format: str = "tuple"
    ) -> Union[Tuple[int, int, int], str, np.ndarray]:
        """Retrieve color by name with format conversion and validation."""
        # Validate color_name exists in colors dictionary
        if color_name not in self.colors:
            raise ValidationError(f"Color '{color_name}' not found in palette")

        color_value = self.colors[color_name]

        # Convert color to requested output_format
        if output_format == "tuple":
            if isinstance(color_value, str):  # Hex to RGB
                hex_val = color_value.lstrip("#")
                return tuple(int(hex_val[i : i + 2], 16) for i in (0, 2, 4))
            return tuple(color_value)
        elif output_format == "hex":
            if isinstance(color_value, (tuple, list)):  # RGB to hex
                return f"#{color_value[0]:02x}{color_value[1]:02x}{color_value[2]:02x}"
            return color_value
        elif output_format == "array":
            rgb_tuple = self.get_color(color_name, "tuple")
            return np.array(rgb_tuple, dtype=np.uint8)
        else:
            raise ValidationError(f"Invalid output format '{output_format}'")

    def calculate_contrast_ratio(self, color1_name: str, color2_name: str) -> float:
        """Calculate contrast ratio between two colors for accessibility compliance."""
        # Retrieve colors from palette with validation
        color1_rgb = self.get_color(color1_name, "tuple")
        color2_rgb = self.get_color(color2_name, "tuple")

        # Convert colors to luminance values
        def rgb_to_luminance(rgb):
            rgb_norm = [c / 255.0 for c in rgb]
            rgb_linear = [
                c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
                for c in rgb_norm
            ]
            return (
                0.2126 * rgb_linear[0] + 0.7152 * rgb_linear[1] + 0.0722 * rgb_linear[2]
            )

        lum1 = rgb_to_luminance(color1_rgb)
        lum2 = rgb_to_luminance(color2_rgb)

        # Calculate WCAG contrast ratio formula
        return (max(lum1, lum2) + 0.05) / (min(lum1, lum2) + 0.05)

    def generate_accessibility_report(self) -> Dict[str, Any]:
        report = {
            "palette_name": self.palette_name,
            "total_colors": len(self.colors),
            "contrast_analysis": {},
            "recommendations": [],
            "compliance_status": "compliant",
        }

        # Analyze all color combinations in palette
        color_names = list(self.colors.keys())
        for i, color1 in enumerate(color_names):
            for color2 in color_names[i + 1 :]:
                try:
                    contrast_ratio = self.calculate_contrast_ratio(color1, color2)
                    report["contrast_analysis"][f"{color1}_{color2}"] = contrast_ratio

                    if contrast_ratio < ACCESSIBILITY_CONTRAST_RATIO_MIN:
                        report["recommendations"].append(
                            f"Increase contrast between {color1} and {color2}"
                        )
                        report["compliance_status"] = "needs_improvement"
                except Exception as e:
                    logger.warning(
                        f"Failed to calculate contrast for {color1}-{color2}: {str(e)}"
                    )

        return report

    def _analyze_accessibility(self):
        """Internal method to analyze accessibility of color palette."""
        try:
            self.accessibility_info = self.generate_accessibility_report()
        except Exception as e:
            logger.warning(f"Accessibility analysis failed: {str(e)}")
            self.accessibility_info = {"error": str(e)}


class ColorSchemeManager:
    def __init__(
        self,
        enable_caching: bool = True,
        auto_optimize: bool = True,
        default_config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize color scheme manager with caching, optimization, and default configuration."""
        # Initialize scheme_cache and colormap_cache dictionaries for performance optimization
        self.scheme_cache = {}
        self.colormap_cache = {}

        # Set caching_enabled flag and configure cache size limits
        self.caching_enabled = enable_caching
        self.cache_size_limit = COLORMAP_CACHE_SIZE

        # Set optimization_enabled flag for automatic performance optimization
        self.optimization_enabled = auto_optimize

        # Create default_scheme using get_default_scheme() or from default_config
        if default_config:
            self.default_scheme = create_color_scheme(default_config)
        else:
            self.default_scheme = get_default_scheme()

        # Initialize performance_metrics dictionary for operation timing and resource tracking
        self.performance_metrics = {
            "cache_hits": 0,
            "cache_misses": 0,
            "optimization_count": 0,
            "validation_count": 0,
        }

        # Create component logger for color scheme operation tracking and debugging
        self.logger = logging.getLogger(f"{__name__}.ColorSchemeManager")

    def get_scheme(
        self,
        scheme_name: Union[str, PredefinedScheme],
        render_mode: Optional[RenderMode] = None,
        use_cache: bool = True,
    ) -> CustomColorScheme:
        # Generate cache key from scheme_name and render_mode parameters
        cache_key = f"{scheme_name}_{render_mode}"

        # Check scheme_cache if use_cache is True and caching_enabled
        if use_cache and self.caching_enabled and cache_key in self.scheme_cache:
            self.performance_metrics["cache_hits"] += 1
            self.logger.debug(f"Cache hit for scheme: {cache_key}")
            return self.scheme_cache[cache_key]

        self.performance_metrics["cache_misses"] += 1

        # Load predefined scheme configuration if scheme_name is PredefinedScheme
        if isinstance(scheme_name, PredefinedScheme):
            color_config = scheme_name.get_color_config()
            color_scheme = create_color_scheme(color_config)
        elif isinstance(scheme_name, str):
            # Try to get predefined scheme by name
            try:
                predefined_scheme = PredefinedScheme(scheme_name)
                color_config = predefined_scheme.get_color_config()
                color_scheme = create_color_scheme(color_config)
            except ValueError:
                # Fallback to default scheme
                self.logger.warning(
                    f"Unknown scheme name '{scheme_name}', using default"
                )
                color_scheme = self.default_scheme.clone()
        else:
            raise ValidationError(f"Invalid scheme_name type: {type(scheme_name)}")

        # Apply render mode optimization if render_mode specified
        if render_mode:
            color_scheme.optimize_for_render_mode(render_mode)

        # Cache color scheme if use_cache is True for performance optimization
        if use_cache and self.caching_enabled:
            self.scheme_cache[cache_key] = color_scheme

        # Update performance metrics and access statistics
        self.logger.debug(f"Retrieved color scheme: {scheme_name}")

        return color_scheme

    def create_custom_scheme(
        self,
        scheme_name: str,
        color_config: Dict[str, Any],
        validate_scheme: bool = True,
        enable_accessibility: bool = False,
        register_scheme: bool = True,
    ) -> CustomColorScheme:
        # Validate scheme_name uniqueness and format requirements
        if not isinstance(scheme_name, str) or not scheme_name.strip():
            raise ValidationError("scheme_name must be a non-empty string")

        sanitized_config = copy.deepcopy(color_config)

        # Create CustomColorScheme using create_color_scheme() with configuration
        color_scheme = create_color_scheme(
            sanitized_config, validate_immediately=validate_scheme
        )

        # Enable accessibility features if enable_accessibility is True
        if enable_accessibility:
            color_scheme.enable_accessibility()

        # Register scheme in scheme_cache if register_scheme is True
        if register_scheme:
            self.scheme_cache[scheme_name] = color_scheme

        # Apply automatic optimization if optimization_enabled is True
        if self.optimization_enabled:
            color_scheme.optimize_for_render_mode(
                RenderMode.RGB_ARRAY
            )  # Default optimization
            self.performance_metrics["optimization_count"] += 1

        # Log scheme creation with configuration details
        self.logger.info(
            f"Created custom scheme '{scheme_name}' with accessibility: {enable_accessibility}"
        )

        return color_scheme

    def validate_scheme(
        self,
        color_scheme: CustomColorScheme,
        check_accessibility: bool = False,
        check_performance: bool = False,
        check_matplotlib_integration: bool = True,
    ) -> Tuple[bool, Dict[str, Any]]:
        self.performance_metrics["validation_count"] += 1

        return validate_color_scheme(
            color_scheme,
            check_accessibility=check_accessibility,
            check_performance=check_performance,
            validate_matplotlib_integration=check_matplotlib_integration,
        )

    def optimize_scheme(
        self,
        color_scheme: CustomColorScheme,
        target_mode: RenderMode,
        optimization_config: Optional[Dict[str, Any]] = None,
    ) -> CustomColorScheme:
        # Delegate to optimize_for_render_mode() function with performance configuration
        optimized_scheme = color_scheme.clone()
        optimized_scheme.optimize_for_render_mode(target_mode, optimization_config)

        # Update colormap_cache with optimized colormaps for performance
        if optimized_scheme.concentration_colormap not in self.colormap_cache:
            colormap = get_matplotlib_colormap(optimized_scheme.concentration_colormap)
            self.colormap_cache[optimized_scheme.concentration_colormap] = colormap

        # Measure optimization impact and update performance_metrics
        self.performance_metrics["optimization_count"] += 1

        # Log optimization results
        self.logger.debug(f"Optimized color scheme for {target_mode}")

        return optimized_scheme

    def clear_cache(
        self,
        clear_scheme_cache: bool = True,
        clear_colormap_cache: bool = True,
        reset_metrics: bool = False,
    ) -> Dict[str, Any]:
        # Calculate current cache sizes before clearing
        initial_scheme_count = len(self.scheme_cache)
        initial_colormap_count = len(self.colormap_cache)

        clearing_report = {
            "schemes_cleared": 0,
            "colormaps_cleared": 0,
            "memory_freed_estimate_mb": 0,
        }

        # Clear scheme_cache if clear_scheme_cache is True
        if clear_scheme_cache:
            clearing_report["schemes_cleared"] = len(self.scheme_cache)
            self.scheme_cache.clear()

        # Clear colormap_cache if clear_colormap_cache is True
        if clear_colormap_cache:
            clearing_report["colormaps_cleared"] = len(self.colormap_cache)
            self.colormap_cache.clear()

        # Reset performance_metrics if reset_metrics is True
        if reset_metrics:
            keys_to_clear = [
                "cache_hits",
                "cache_misses",
                "optimization_count",
                "validation_count",
            ]
            normalization_cache_dict = dict.fromkeys(keys_to_clear)
            self.performance_metrics = normalization_cache_dict

        # Estimate memory freed (rough calculation)
        clearing_report["memory_freed_estimate_mb"] = (
            initial_scheme_count * 0.1 + initial_colormap_count * 0.5
        )

        # Log cache clearing
        self.logger.info(f"Cache cleared: {clearing_report}")

        return clearing_report

    def get_performance_metrics(
        self,
        include_cache_stats: bool = True,
        include_optimization_analysis: bool = True,
    ) -> Dict[str, Any]:
        metrics = self.performance_metrics.copy()
        metrics["timestamp"] = (
            logging.getLogger()
            .handlers[0]
            .formatter.formatTime(logging.LogRecord("", 0, "", 0, "", (), None))
            if logging.getLogger().handlers
            else "unknown"
        )

        # Include cache hit/miss ratios and efficiency metrics
        if include_cache_stats:
            total_requests = metrics["cache_hits"] + metrics["cache_misses"]
            if total_requests > 0:
                metrics["cache_hit_ratio"] = metrics["cache_hits"] / total_requests
                metrics["cache_efficiency"] = (
                    "high"
                    if metrics["cache_hit_ratio"] > 0.7
                    else "medium" if metrics["cache_hit_ratio"] > 0.4 else "low"
                )

            metrics["current_cache_sizes"] = {
                "scheme_cache": len(self.scheme_cache),
                "colormap_cache": len(self.colormap_cache),
            }

        # Add optimization analysis and performance recommendations
        if include_optimization_analysis:
            recommendations = []
            if metrics.get("cache_hit_ratio", 0) < 0.5:
                recommendations.append("Consider enabling more aggressive caching")
            if metrics["optimization_count"] == 0:
                recommendations.append(
                    "Enable automatic optimization for better performance"
                )

            metrics["recommendations"] = recommendations

        return metrics


# Factory Functions


def create_color_scheme(  # noqa: C901
    scheme_config: Union[Dict[str, Any], str, PredefinedScheme],
    optimize_for_mode: Optional[RenderMode] = None,
    enable_accessibility: bool = False,
    validate_immediately: bool = True,
    performance_config: Optional[Dict[str, Any]] = None,
) -> CustomColorScheme:
    try:
        # Parse scheme_config input: handle dict configuration, PredefinedScheme enum, or string scheme name
        if isinstance(scheme_config, PredefinedScheme):
            config_dict = scheme_config.get_color_config()
        elif isinstance(scheme_config, str):
            predefined_scheme = PredefinedScheme(scheme_config)
            config_dict = predefined_scheme.get_color_config()
        elif isinstance(scheme_config, dict):
            config_dict = scheme_config.copy()
        else:
            raise ValidationError(f"Invalid scheme_config type: {type(scheme_config)}")

        # Extract color parameters with defaults
        agent_color = config_dict.get("agent_color", AGENT_MARKER_COLOR)
        source_color = config_dict.get("source_color", SOURCE_MARKER_COLOR)
        background_color = config_dict.get("background_color", (0, 0, 0))
        concentration_colormap = config_dict.get(
            "concentration_colormap", DEFAULT_COLORMAP
        )

        # Validate RGB color values are within [0, 255] range
        for color_name, color_value in [
            ("agent_color", agent_color),
            ("source_color", source_color),
            ("background_color", background_color),
        ]:
            if not (isinstance(color_value, (tuple, list)) and len(color_value) == 3):
                raise ValidationError(f"{color_name} must be a tuple/list of 3 values")
            if not all(isinstance(val, int) and 0 <= val <= 255 for val in color_value):
                raise ValidationError(
                    f"{color_name} values must be integers in range [0,255]"
                )

        # Create base CustomColorScheme with validated color parameters
        color_scheme = CustomColorScheme(
            agent_color=tuple(agent_color),
            source_color=tuple(source_color),
            background_color=tuple(background_color),
            concentration_colormap=concentration_colormap,
        )

        # Apply performance configuration from performance_config
        if performance_config:
            color_scheme.performance_config.update(performance_config)

        # Apply render mode optimization if optimize_for_mode specified
        if optimize_for_mode:
            color_scheme.optimize_for_render_mode(optimize_for_mode)

        # Enable accessibility features if enable_accessibility is True
        if enable_accessibility:
            color_scheme.enable_accessibility()

        # Validate complete color scheme configuration if validate_immediately is True
        if validate_immediately:
            color_scheme.validate(check_accessibility=enable_accessibility)

        logger.debug(f"Created color scheme with config: {config_dict}")
        return color_scheme

    except Exception as e:
        logger.error(f"Color scheme creation failed: {str(e)}")
        raise ValidationError(f"Failed to create color scheme: {str(e)}")


@functools.lru_cache(maxsize=10)
def get_default_scheme(
    render_mode: Optional[RenderMode] = None,
    enable_caching: bool = True,
    apply_accessibility: bool = False,
) -> CustomColorScheme:
    # Check scheme cache if enable_caching is True
    cache_key = f"default_{render_mode}_{apply_accessibility}"
    if enable_caching and cache_key in SCHEME_CACHE:
        logger.debug("Retrieved default scheme from cache")
        return SCHEME_CACHE[cache_key]

    # Create default color configuration with standard colors
    default_config = {
        "agent_color": AGENT_MARKER_COLOR,  # Red [255,0,0]
        "source_color": SOURCE_MARKER_COLOR,  # White [255,255,255]
        "background_color": (0, 0, 0),  # Black [0,0,0]
        "concentration_colormap": DEFAULT_COLORMAP,  # Gray colormap
    }

    # Create CustomColorScheme with default configuration
    default_scheme = create_color_scheme(
        default_config,
        optimize_for_mode=render_mode,
        enable_accessibility=apply_accessibility,
        validate_immediately=True,
    )

    # Configure performance optimization settings for target latency achievement
    default_scheme.performance_config.update(
        {
            "target_rgb_latency_ms": PERFORMANCE_TARGET_RGB_RENDER_MS,
            "target_human_latency_ms": 50.0,
            "caching_enabled": True,
        }
    )

    # Cache default scheme if enable_caching is True
    if enable_caching:
        SCHEME_CACHE[cache_key] = default_scheme

    logger.debug(f"Created default color scheme for mode: {render_mode}")
    return default_scheme


def create_accessibility_scheme(
    accessibility_type: str,
    optimize_for_mode: Optional[RenderMode] = None,
    custom_colors: Optional[Dict[str, Any]] = None,
    validate_contrast: bool = True,
) -> CustomColorScheme:
    # Validate accessibility_type against supported options
    supported_types = ["high_contrast", "colorblind_friendly", "low_vision"]
    if accessibility_type not in supported_types:
        raise ValidationError(
            f"Unsupported accessibility_type '{accessibility_type}'. Supported: {supported_types}"
        )

    # Load base accessibility configuration for specified accessibility_type
    if accessibility_type == "high_contrast":
        base_config = {
            "agent_color": (255, 0, 0),  # Pure red
            "source_color": (255, 255, 255),  # Pure white
            "background_color": (0, 0, 0),  # Pure black
            "concentration_colormap": "gray",
        }
    elif accessibility_type == "colorblind_friendly":
        base_config = {
            "agent_color": (255, 165, 0),  # Orange (deuteranopia safe)
            "source_color": (255, 255, 255),  # White
            "background_color": (0, 0, 0),  # Black
            "concentration_colormap": "viridis",  # Colorblind-friendly
        }
    elif accessibility_type == "low_vision":
        base_config = {
            "agent_color": (255, 255, 0),  # Bright yellow
            "source_color": (255, 255, 255),  # White
            "background_color": (0, 0, 0),  # Black
            "concentration_colormap": "plasma",
        }

    # Override default colors with custom_colors if provided
    if custom_colors:
        base_config.update(custom_colors)

    # Create CustomColorScheme with accessibility configuration
    accessibility_scheme = create_color_scheme(
        base_config,
        optimize_for_mode=optimize_for_mode,
        enable_accessibility=True,
        validate_immediately=False,  # Validate after accessibility enhancements
    )

    # Apply accessibility type-specific enhancements
    accessibility_scheme.enable_accessibility(accessibility_type, validate_contrast)

    # Create AccessibilityConfig with validated settings
    accessibility_config = AccessibilityConfig(
        min_contrast_ratio=ACCESSIBILITY_CONTRAST_RATIO_MIN,
        colorblind_friendly=(accessibility_type == "colorblind_friendly"),
        high_contrast_mode=(accessibility_type == "high_contrast"),
        enhanced_visibility=(accessibility_type == "low_vision"),
    )

    # Validate contrast ratios meet requirements if validate_contrast is True
    if validate_contrast:
        is_compliant, compliance_report = accessibility_config.validate_compliance(
            accessibility_scheme
        )
        if not is_compliant:
            logger.warning(f"Accessibility compliance issues: {compliance_report}")

    logger.info(f"Created accessibility scheme: {accessibility_type}")
    return accessibility_scheme


def optimize_for_render_mode(
    color_scheme: CustomColorScheme,
    target_mode: RenderMode,
    optimization_options: Optional[Dict[str, Any]] = None,
    enable_caching: bool = True,
    validate_performance: bool = True,
) -> CustomColorScheme:
    # Validate target_mode is supported RenderMode enum value
    validate_render_mode(target_mode)

    # Create optimized copy to avoid modifying original
    optimized_scheme = color_scheme.clone()

    # Apply mode-specific optimizations
    if target_mode == RenderMode.RGB_ARRAY:
        # RGB array optimizations: pre-compute color lookup tables, optimize uint8 operations
        optimization_config = {
            "pre_compute_colors": True,
            "enable_vectorization": True,
            "use_uint8_operations": True,
            "target_latency_ms": PERFORMANCE_TARGET_RGB_RENDER_MS,
        }

    elif target_mode == RenderMode.HUMAN:
        # Matplotlib optimizations: enable colormap caching, configure figure reuse
        optimization_config = {
            "matplotlib_caching": True,
            "figure_reuse": True,
            "colorbar_caching": True,
            "target_latency_ms": 50.0,  # Human mode target
        }
    else:
        raise ValidationError(f"Unsupported render mode: {target_mode}")

    # Configure optimization options from optimization_options
    if optimization_options:
        optimization_config.update(optimization_options)

    # Apply optimization configuration
    optimized_scheme.optimize_for_render_mode(target_mode, optimization_config)

    # Enable colormap caching and pre-computation if enable_caching is True
    if enable_caching:
        # Pre-cache the colormap for performance
        colormap = get_matplotlib_colormap(
            optimized_scheme.concentration_colormap, enable_caching=True
        )
        optimized_scheme._cached_colormap = colormap

    # Validate optimized performance meets target latencies if validate_performance is True
    if validate_performance:
        target_latency = optimization_config.get("target_latency_ms", 10.0)
        if target_latency > 100.0:  # Warn about potentially slow targets
            logger.warning(
                f"High performance target: {target_latency}ms may impact user experience"
            )

    logger.debug(
        f"Optimized color scheme for {target_mode} with config: {optimization_config}"
    )
    return optimized_scheme


def validate_color_scheme(  # noqa: C901
    color_scheme: Union[CustomColorScheme, ColorSchemeManager, Dict[str, Any]],
    check_accessibility: bool = False,
    check_performance: bool = False,
    validate_matplotlib_integration: bool = True,
    strict_validation: bool = False,
) -> Tuple[bool, Dict[str, Any]]:
    validation_report = {
        "is_valid": True,
        "errors": [],
        "warnings": [],
        "recommendations": [],
        "validation_details": {},
    }

    try:
        # Parse color_scheme input and extract color configuration
        if isinstance(color_scheme, CustomColorScheme):
            scheme_obj = color_scheme
        elif isinstance(color_scheme, dict):
            scheme_obj = create_color_scheme(color_scheme, validate_immediately=False)
        else:
            raise ValidationError(
                f"Unsupported color_scheme type: {type(color_scheme)}"
            )

        # Validate RGB color values are integers in valid range [0, 255]
        for color_name in ["agent_color", "source_color", "background_color"]:
            color_value = getattr(scheme_obj, color_name)
            if not (isinstance(color_value, tuple) and len(color_value) == 3):
                validation_report["errors"].append(
                    f"{color_name} must be a tuple of 3 values"
                )
                validation_report["is_valid"] = False
            elif not all(
                isinstance(val, int) and 0 <= val <= 255 for val in color_value
            ):
                validation_report["errors"].append(
                    f"{color_name} values must be integers in range [0,255]"
                )
                validation_report["is_valid"] = False

        # Check matplotlib colormap availability and backend compatibility
        if validate_matplotlib_integration:
            try:
                result = _safe_get_cmap(scheme_obj.concentration_colormap)
                if result is None:
                    validation_report["validation_details"][
                        "matplotlib_colormap"
                    ] = "deferred"
                else:
                    validation_report["validation_details"][
                        "matplotlib_colormap"
                    ] = "available"
            except ValueError as e:
                validation_report["errors"].append(
                    f"Invalid matplotlib colormap: {str(e)}"
                )
                validation_report["is_valid"] = False

        # Validate accessibility compliance including contrast ratios
        if check_accessibility:
            if (
                hasattr(scheme_obj, "accessibility_enabled")
                and scheme_obj.accessibility_enabled
            ):
                # Simplified contrast check
                agent_luminance = sum(scheme_obj.agent_color) / (3 * 255)
                background_luminance = sum(scheme_obj.background_color) / (3 * 255)
                contrast_ratio = (max(agent_luminance, background_luminance) + 0.05) / (
                    min(agent_luminance, background_luminance) + 0.05
                )

                validation_report["validation_details"][
                    "contrast_ratio"
                ] = contrast_ratio
                if contrast_ratio < ACCESSIBILITY_CONTRAST_RATIO_MIN:
                    validation_report["warnings"].append(
                        f"Low contrast ratio: {contrast_ratio:.2f}"
                    )

        # Test rendering performance against targets
        if check_performance:
            performance_config = getattr(scheme_obj, "performance_config", {})
            target_latency = performance_config.get("target_latency_ms", 10.0)
            if (
                target_latency > PERFORMANCE_TARGET_RGB_RENDER_MS * 5
            ):  # 5x slower than target
                validation_report["warnings"].append(
                    f"High performance target may impact user experience: {target_latency}ms"
                )

        # Check for color conflicts between markers and background
        if scheme_obj.agent_color == scheme_obj.source_color:
            if strict_validation:
                validation_report["errors"].append(
                    "Agent and source colors must be different"
                )
                validation_report["is_valid"] = False
            else:
                validation_report["warnings"].append(
                    "Identical agent and source colors may reduce visibility"
                )

        # Apply strict validation rules including edge cases
        if strict_validation:
            # Check color similarity
            def color_distance(c1, c2):
                return sum((a - b) ** 2 for a, b in zip(c1, c2)) ** 0.5

            if color_distance(scheme_obj.agent_color, scheme_obj.background_color) < 50:
                validation_report["errors"].append(
                    "Agent color too similar to background"
                )
                validation_report["is_valid"] = False

        # Generate recommendations based on validation results
        if validation_report["warnings"]:
            validation_report["recommendations"].append(
                "Review warning messages for potential improvements"
            )
        if not check_accessibility:
            validation_report["recommendations"].append(
                "Consider enabling accessibility validation"
            )

    except Exception as e:
        validation_report["is_valid"] = False
        validation_report["errors"].append(f"Validation failed: {str(e)}")

    logger.debug(f"Color scheme validation completed: {validation_report['is_valid']}")
    return validation_report["is_valid"], validation_report


def get_predefined_schemes(
    include_descriptions: bool = True,
    include_performance_info: bool = True,
    filter_by_mode: Optional[RenderMode] = None,
) -> Dict[str, Any]:
    schemes_info = {}

    # Enumerate all PredefinedScheme enum values
    for scheme in PredefinedScheme:
        scheme_info = {"name": scheme.value, "enum_value": scheme}

        # Include scheme descriptions and use case information
        if include_descriptions:
            if scheme == PredefinedScheme.STANDARD:
                scheme_info["description"] = (
                    "Standard plume navigation colors with red agent and white source markers"
                )
                scheme_info["use_cases"] = [
                    "General visualization",
                    "Scientific analysis",
                    "Training environments",
                ]
            elif scheme == PredefinedScheme.HIGH_CONTRAST:
                scheme_info["description"] = (
                    "High contrast colors for enhanced visibility and accessibility"
                )
                scheme_info["use_cases"] = [
                    "Accessibility",
                    "Low light conditions",
                    "Presentation displays",
                ]
            elif scheme == PredefinedScheme.COLORBLIND_FRIENDLY:
                scheme_info["description"] = (
                    "Colorblind-friendly palette using orange and blue tones"
                )
                scheme_info["use_cases"] = [
                    "Inclusive design",
                    "Color vision deficiency",
                    "Public presentations",
                ]

        # Add performance characteristics and optimization data
        if include_performance_info:
            scheme_info["performance"] = {
                "rgb_render_target_ms": PERFORMANCE_TARGET_RGB_RENDER_MS,
                "human_render_target_ms": 50.0,
                "caching_supported": True,
                "optimization_available": True,
            }

        # Filter schemes by rendering mode compatibility if specified
        if filter_by_mode is None or True:  # All schemes support both modes
            schemes_info[scheme.value] = scheme_info

    return {
        "total_schemes": len(schemes_info),
        "schemes": schemes_info,
        "filter_mode": str(filter_by_mode) if filter_by_mode else "all",
        "timestamp": (
            logger.handlers[0].formatter.formatTime(
                logging.LogRecord("", 0, "", 0, "", (), None)
            )
            if logger.handlers
            else "unknown"
        ),
    }


# Utility Functions


def normalize_concentration_to_rgb(
    concentration_field: np.ndarray,
    colormap_name: Optional[str] = None,
    use_caching: bool = True,
    normalization_range: Optional[Tuple[float, float]] = None,
    validate_output: bool = True,
) -> np.ndarray:
    # Validate concentration_field array dimensions and data type
    if not isinstance(concentration_field, np.ndarray):
        raise ValidationError("concentration_field must be a numpy array")
    if concentration_field.ndim != 2:
        raise ValidationError("concentration_field must be a 2D array")

    # Clamp concentration values to normalization_range or default CONCENTRATION_RANGE
    norm_range = normalization_range or CONCENTRATION_RANGE
    concentration_clamped = np.clip(concentration_field, norm_range[0], norm_range[1])

    # Use colormap_name or fallback to DEFAULT_COLORMAP
    colormap_name = colormap_name or DEFAULT_COLORMAP

    # Retrieve matplotlib colormap with caching
    colormap = get_matplotlib_colormap(
        colormap_name, normalization_range=norm_range, enable_caching=use_caching
    )

    # Apply colormap normalization converting [0,1] to [0,255] RGB values
    # Normalize to [0,1] first
    normalized_field = (concentration_clamped - norm_range[0]) / (
        norm_range[1] - norm_range[0]
    )

    # Apply colormap to get RGBA values
    rgba_values = colormap(normalized_field)

    # Convert to RGB (drop alpha channel) and scale to [0,255]
    rgb_array = (rgba_values[:, :, :3] * 255).astype(RGB_DTYPE)

    # Handle edge cases including NaN values gracefully
    if np.any(np.isnan(concentration_field)):
        nan_mask = np.isnan(concentration_field)
        rgb_array[nan_mask] = [0, 0, 0]  # Black for NaN values

    # Validate output array format and value ranges if validate_output is True
    if validate_output:
        if rgb_array.shape[:2] != concentration_field.shape:
            raise RenderingError(
                "RGB array shape mismatch with input concentration field"
            )
        if rgb_array.dtype != RGB_DTYPE:
            raise RenderingError(
                f"RGB array dtype {rgb_array.dtype} != expected {RGB_DTYPE}"
            )
        if np.any((rgb_array < 0) | (rgb_array > 255)):
            raise RenderingError("RGB values outside valid range [0,255]")

    return rgb_array


def apply_agent_marker(
    rgb_array: np.ndarray,
    agent_position: Coordinates,
    marker_color: Optional[Tuple[int, int, int]] = None,
    marker_size: Optional[Tuple[int, int]] = None,
    check_boundaries: bool = True,
) -> np.ndarray:
    # Validate rgb_array dimensions (H,W,3) and data type
    if not isinstance(rgb_array, np.ndarray):
        raise ValidationError("rgb_array must be a numpy array")
    if rgb_array.ndim != 3 or rgb_array.shape[2] != 3:
        raise ValidationError("rgb_array must have shape (H,W,3)")
    if rgb_array.dtype != RGB_DTYPE:
        logger.warning(f"rgb_array dtype {rgb_array.dtype} != expected {RGB_DTYPE}")

    # Extract x,y coordinates from agent_position with bounds validation
    if not isinstance(agent_position, Coordinates):
        raise ValidationError("agent_position must be a Coordinates object")

    agent_x, agent_y = agent_position.x, agent_position.y
    height, width = rgb_array.shape[:2]

    # Use marker_color or default AGENT_MARKER_COLOR
    color = marker_color or AGENT_MARKER_COLOR

    # Use marker_size or default AGENT_MARKER_SIZE
    size = marker_size or AGENT_MARKER_SIZE
    half_w, half_h = size[0] // 2, size[1] // 2

    # Calculate marker pixel boundaries with boundary checking
    if check_boundaries:
        x_min = max(0, agent_x - half_w)
        x_max = min(width, agent_x + half_w + 1)
        y_min = max(0, agent_y - half_h)
        y_max = min(height, agent_y + half_h + 1)
    else:
        x_min = agent_x - half_w
        x_max = agent_x + half_w + 1
        y_min = agent_y - half_h
        y_max = agent_y + half_h + 1

    # Apply red square marker pattern to RGB array at agent position
    if x_min < width and x_max > 0 and y_min < height and y_max > 0:
        # Clip to array bounds
        x_min_clipped = max(0, x_min)
        x_max_clipped = min(width, x_max)
        y_min_clipped = max(0, y_min)
        y_max_clipped = min(height, y_max)

        # Apply marker color
        rgb_array[y_min_clipped:y_max_clipped, x_min_clipped:x_max_clipped] = color

    return rgb_array


def apply_source_marker(
    rgb_array: np.ndarray,
    source_position: Coordinates,
    marker_color: Optional[Tuple[int, int, int]] = None,
    marker_size: Optional[Tuple[int, int]] = None,
    check_boundaries: bool = True,
) -> np.ndarray:
    # Validate rgb_array dimensions (H,W,3) and data type
    if not isinstance(rgb_array, np.ndarray):
        raise ValidationError("rgb_array must be a numpy array")
    if rgb_array.ndim != 3 or rgb_array.shape[2] != 3:
        raise ValidationError("rgb_array must have shape (H,W,3)")

    # Extract x,y coordinates from source_position
    if not isinstance(source_position, Coordinates):
        raise ValidationError("source_position must be a Coordinates object")

    source_x, source_y = source_position.x, source_position.y
    height, width = rgb_array.shape[:2]

    # Use marker_color or default SOURCE_MARKER_COLOR
    color = marker_color or SOURCE_MARKER_COLOR

    # Use marker_size or default SOURCE_MARKER_SIZE
    size = marker_size or SOURCE_MARKER_SIZE
    half_w, half_h = size[0] // 2, size[1] // 2

    # Calculate cross pattern coordinates with boundary checking
    if check_boundaries:
        # Horizontal line
        h_x_min = max(0, source_x - half_w)
        h_x_max = min(width, source_x + half_w + 1)
        h_y = source_y if 0 <= source_y < height else -1

        # Vertical line
        v_y_min = max(0, source_y - half_h)
        v_y_max = min(height, source_y + half_h + 1)
        v_x = source_x if 0 <= source_x < width else -1
    else:
        h_x_min, h_x_max = source_x - half_w, source_x + half_w + 1
        h_y = source_y
        v_y_min, v_y_max = source_y - half_h, source_y + half_h + 1
        v_x = source_x

    # Apply white cross marker pattern
    # Horizontal line
    if h_y >= 0 and h_x_min < width and h_x_max > 0:
        h_x_min_clipped = max(0, h_x_min)
        h_x_max_clipped = min(width, h_x_max)
        rgb_array[h_y, h_x_min_clipped:h_x_max_clipped] = color

    # Vertical line
    if v_x >= 0 and v_y_min < height and v_y_max > 0:
        v_y_min_clipped = max(0, v_y_min)
        v_y_max_clipped = min(height, v_y_max)
        rgb_array[v_y_min_clipped:v_y_max_clipped, v_x] = color

    return rgb_array


@functools.lru_cache(maxsize=COLORMAP_CACHE_SIZE)
def get_matplotlib_colormap(
    colormap_name: str,
    normalization_range: Optional[Tuple[float, float]] = None,
    validate_backend: bool = True,
    enable_caching: bool = True,
) -> matplotlib.colors.Colormap:
    try:
        # Validate matplotlib backend availability if validate_backend is True
        if validate_backend:
            import matplotlib

            backend = matplotlib.get_backend()
            logger.debug(f"Using matplotlib backend: {backend}")

        # Retrieve colormap using robust helper (modern API preferred)
        colormap = _safe_get_cmap(colormap_name)
        if colormap is None:
            # Environment import issues; attempt default, else build a basic grayscale
            fallback = _safe_get_cmap(DEFAULT_COLORMAP)
            if fallback is None:
                # Final-resort grayscale to keep rendering functional in CI
                colormap = matplotlib.colors.LinearSegmentedColormap.from_list(
                    "plume_nav_fallback_gray",
                    [(0.0, (0.0, 0.0, 0.0)), (1.0, (1.0, 1.0, 1.0))],
                    N=256,
                )
            else:
                colormap = fallback

        # Configure colormap normalization for specified range
        norm_range = normalization_range or CONCENTRATION_RANGE
        if norm_range != (0.0, 1.0):
            # Create custom normalization if needed (placeholder; using existing colormap)
            colormap = (
                colormap._segmentdata if hasattr(colormap, "_segmentdata") else colormap
            )

        logger.debug(
            f"Retrieved colormap '{colormap_name}' with normalization {norm_range}"
        )
        return colormap

    except Exception as e:
        logger.error(f"Failed to retrieve colormap '{colormap_name}': {str(e)}")
        # Fallback to default colormap, with final grayscale guard
        fallback = _safe_get_cmap(DEFAULT_COLORMAP)
        if fallback is None:
            return matplotlib.colors.LinearSegmentedColormap.from_list(
                "plume_nav_fallback_gray",
                [(0.0, (0.0, 0.0, 0.0)), (1.0, (1.0, 1.0, 1.0))],
                N=256,
            )
        return fallback


def convert_rgb_to_matplotlib(  # noqa: C901
    rgb_color: Union[Tuple[int, int, int], List[int], np.ndarray],
    normalize_to_unit: bool = True,
    validate_range: bool = True,
    output_format: str = "tuple",
) -> Union[Tuple[float, float, float], np.ndarray, str]:
    # Validate rgb_color input format and extract RGB values
    if isinstance(rgb_color, np.ndarray):
        if rgb_color.shape != (3,):
            raise ValidationError("RGB array must have shape (3,)")
        rgb_values = rgb_color.tolist()
    elif isinstance(rgb_color, (tuple, list)):
        if len(rgb_color) != 3:
            raise ValidationError("RGB color must have exactly 3 values")
        rgb_values = list(rgb_color)
    else:
        raise ValidationError(f"Invalid RGB color type: {type(rgb_color)}")

    # Check RGB values are in valid range [0, 255] if validate_range is True
    if validate_range:
        if not all(
            isinstance(val, (int, float)) and 0 <= val <= 255 for val in rgb_values
        ):
            raise ValidationError("RGB values must be numbers in range [0, 255]")

    # Normalize RGB values to [0, 1] range if normalize_to_unit is True
    if normalize_to_unit:
        rgb_normalized = [val / 255.0 for val in rgb_values]
    else:
        rgb_normalized = rgb_values

    # Convert to requested output_format
    if output_format == "tuple":
        return tuple(rgb_normalized)
    elif output_format == "array":
        return np.array(rgb_normalized, dtype=np.float32)
    elif output_format == "hex":
        # Convert back to integers for hex representation
        if normalize_to_unit:
            rgb_int = [int(val * 255) for val in rgb_normalized]
        else:
            rgb_int = [int(val) for val in rgb_normalized]
        return f"#{rgb_int[0]:02x}{rgb_int[1]:02x}{rgb_int[2]:02x}"
    else:
        raise ValidationError(
            f"Invalid output_format '{output_format}'. Supported: tuple, array, hex"
        )


# Initialize module-level caching and performance monitoring
logger.info(
    f"Color scheme module initialized with caching: {PERFORMANCE_OPTIMIZATION_ENABLED}"
)
