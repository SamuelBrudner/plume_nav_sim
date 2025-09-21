"""
Assets package initialization module for plume_nav_sim providing centralized access to
color schemes, rendering templates, and visualization utilities.

This module exposes public APIs for dual-mode rendering system supporting both RGB array
generation and matplotlib human mode visualization with performance optimization,
accessibility features, and cross-platform compatibility.

The assets package provides:
- Standardized color schemes with accessibility support
- Template-based rendering system for consistent visualization
- Performance-optimized asset utilities
- Cross-platform compatibility and fallback mechanisms
- Clean package interface with minimal coupling
"""

import logging  # >=3.10 - Logger for asset compatibility warnings and debugging
import time  # >=3.10 - Performance timing for asset optimization validation

# External imports with version comments
from typing import (  # >=3.10 - Type hints for factory functions and compatibility validation
    Any,
    Dict,
    Optional,
    Tuple,
    Union,
)

# Internal imports from color management module
from .default_colormap import (
    DEFAULT_COLORMAP,  # Default colormap constant for consistency
)
from .default_colormap import (
    ColorScheme,  # Primary color scheme configuration class for dual-mode rendering
)
from .default_colormap import (
    PredefinedScheme,  # Predefined color scheme enumeration for common visualization scenarios
)
from .default_colormap import (
    create_color_scheme,  # Factory function for custom color scheme creation
)
from .default_colormap import (
    create_default_scheme,  # Factory function for default color scheme creation
)
from .default_colormap import validate_color_scheme  # Color scheme validation utility

# Internal imports from rendering template system (optional in minimal environments)
try:
    from .render_templates import (
        BaseRenderTemplate,  # Abstract base template interface for consistent API
    )
    from .render_templates import (
        MatplotlibTemplate,  # Matplotlib visualization template for human mode rendering
    )
    from .render_templates import (
        RGBTemplate,  # RGB array rendering template for programmatic visualization
    )
    from .render_templates import (
        TemplateConfig,  # Template configuration data structure
    )
    from .render_templates import TemplateQuality  # Template quality level enumeration
    from .render_templates import (
        create_matplotlib_template,  # Factory function for matplotlib template creation
    )
    from .render_templates import (
        create_rgb_template,  # Factory function for RGB template creation
    )
    from .render_templates import (
        get_template_registry,  # Template registry access utility
    )
except (
    ImportError
):  # pragma: no cover - template utilities may be unavailable during tests
    RGBTemplate = MatplotlibTemplate = BaseRenderTemplate = None  # type: ignore[assignment]
    TemplateConfig = TemplateQuality = None  # type: ignore[assignment]

    def create_rgb_template(*args, **kwargs):  # type: ignore[override]
        raise RuntimeError("Rendering templates are unavailable in this environment.")

    def create_matplotlib_template(*args, **kwargs):  # type: ignore[override]
        raise RuntimeError("Rendering templates are unavailable in this environment.")

    def get_template_registry() -> dict:  # type: ignore[override]
        return {}


# Package version and metadata constants
__version__ = "0.0.1"
ASSETS_VERSION = __version__

# Configure logger for asset package operations
_logger = logging.getLogger(__name__)


def get_default_assets(
    render_mode: str, enable_optimization: bool = True
) -> Dict[str, Any]:
    """Convenience function to retrieve default asset configuration including standard color scheme
    and rendering templates for quick setup of plume navigation visualization.

    This function creates a comprehensive default asset configuration optimized for the specified
    render mode, including color schemes, rendering templates, and performance settings.

    Args:
        render_mode (str): Target render mode ('rgb_array' or 'human') for optimization
        enable_optimization (bool): Whether to apply performance optimization settings

    Returns:
        Dict[str, Any]: Dictionary containing default ColorScheme, RGBTemplate, MatplotlibTemplate,
                       and configuration settings optimized for specified render mode

    Raises:
        ValueError: If render_mode is not 'rgb_array' or 'human'
        RuntimeError: If asset creation fails during configuration
    """
    # Validate render_mode parameter against supported modes
    supported_modes = ["rgb_array", "human"]
    if render_mode not in supported_modes:
        raise ValueError(
            f"Render mode '{render_mode}' not supported. Must be one of: {supported_modes}"
        )

    try:
        # Create default color scheme using create_default_scheme with render mode optimization
        _logger.debug(f"Creating default color scheme for render mode: {render_mode}")
        color_scheme = create_default_scheme(
            scheme_type=PredefinedScheme.STANDARD, optimize_for_mode=render_mode
        )

        # Create RGB template using create_rgb_template with default configuration and performance tuning
        rgb_template_config = TemplateConfig(
            quality_level=(
                TemplateQuality.STANDARD
                if not enable_optimization
                else TemplateQuality.FAST
            ),
            enable_caching=enable_optimization,
            target_fps=None,  # No FPS limit for RGB mode
            memory_limit_mb=40,  # Conservative memory limit for plume field
            enable_validation=True,
        )
        rgb_template = create_rgb_template(config=rgb_template_config)

        # Create matplotlib template using create_matplotlib_template with backend compatibility
        matplotlib_template_config = TemplateConfig(
            quality_level=TemplateQuality.STANDARD,
            enable_caching=enable_optimization,
            target_fps=30 if render_mode == "human" else None,
            memory_limit_mb=10,  # Matplotlib figure memory limit
            enable_validation=True,
        )
        matplotlib_template = create_matplotlib_template(
            config=matplotlib_template_config
        )

        # Apply performance optimization if enable_optimization is True
        optimization_settings = {}
        if enable_optimization:
            # Configure performance optimization based on render mode
            if render_mode == "rgb_array":
                optimization_settings = {
                    "prioritize_speed": True,
                    "cache_plume_field": True,
                    "use_fast_rendering": True,
                    "target_latency_ms": 5.0,
                }
            else:  # human mode
                optimization_settings = {
                    "prioritize_quality": True,
                    "enable_interactive_features": True,
                    "use_figure_caching": True,
                    "target_latency_ms": 50.0,
                }

        # Package all assets into comprehensive configuration dictionary
        default_assets = {
            # Core asset components
            "color_scheme": color_scheme,
            "rgb_template": rgb_template,
            "matplotlib_template": matplotlib_template,
            # Template configurations for reference
            "rgb_config": rgb_template_config,
            "matplotlib_config": matplotlib_template_config,
            # Optimization and performance settings
            "optimization_settings": optimization_settings,
            "enable_optimization": enable_optimization,
            "target_render_mode": render_mode,
            # Default constants and compatibility information
            "default_colormap": DEFAULT_COLORMAP,
            "supported_modes": supported_modes,
            # Performance monitoring and validation
            "performance_targets": {
                "rgb_render_ms": 5.0,
                "human_render_ms": 50.0,
                "memory_limit_mb": 50.0,
            },
        }

        # Include version information and compatibility metadata
        default_assets.update(
            {
                "assets_version": ASSETS_VERSION,
                "package_version": __version__,
                "creation_timestamp": time.time(),
                "compatibility_info": {
                    "dual_mode_rendering": True,
                    "accessibility_features": True,
                    "cross_platform": True,
                    "performance_optimized": enable_optimization,
                },
            }
        )

        # Return complete default asset configuration ready for environment setup
        _logger.info(
            f"Successfully created default assets for {render_mode} mode with optimization: {enable_optimization}"
        )
        return default_assets

    except Exception as e:
        _logger.error(f"Failed to create default assets for {render_mode} mode: {e}")
        raise RuntimeError(f"Asset creation failed: {e}") from e


def validate_asset_compatibility(
    color_scheme: ColorScheme,
    render_template: Union[RGBTemplate, MatplotlibTemplate],
    strict_validation: bool = False,
) -> Tuple[bool, Dict[str, Any]]:
    """Validates compatibility between color schemes and rendering templates, ensuring consistent
    visual representation and performance characteristics across dual-mode rendering system.

    This function performs comprehensive compatibility analysis including color scheme validation,
    template configuration checking, performance target alignment, and visual consistency verification.

    Args:
        color_scheme (ColorScheme): Color scheme instance to validate
        render_template (Union[RGBTemplate, MatplotlibTemplate]): Rendering template to check compatibility with
        strict_validation (bool): Whether to apply strict validation rules including edge cases

    Returns:
        Tuple[bool, Dict[str, Any]]: Tuple of (is_compatible: bool, compatibility_report: dict)
                                    containing validation results and detailed compatibility analysis

    Raises:
        TypeError: If color_scheme or render_template are not of expected types
        ValueError: If validation parameters are invalid
    """
    # Input type validation
    if not isinstance(color_scheme, ColorScheme):
        raise TypeError(
            f"Expected ColorScheme instance, got {type(color_scheme).__name__}"
        )

    if not isinstance(render_template, (RGBTemplate, MatplotlibTemplate)):
        raise TypeError(
            f"Expected RGBTemplate or MatplotlibTemplate instance, got {type(render_template).__name__}"
        )

    # Initialize compatibility report structure
    compatibility_report = {
        "validation_timestamp": time.time(),
        "color_scheme_valid": False,
        "template_valid": False,
        "performance_compatible": False,
        "visual_consistency": False,
        "overall_compatible": False,
        "warnings": [],
        "errors": [],
        "recommendations": [],
        "compatibility_score": 0.0,
        "validation_details": {},
    }

    is_compatible = True
    compatibility_score = 0.0

    try:
        # Validate color scheme and template configuration compatibility
        _logger.debug("Validating color scheme configuration")

        # Check color scheme validity using its internal validation
        try:
            color_scheme_validation = color_scheme.validate()
            compatibility_report["color_scheme_valid"] = color_scheme_validation
            if color_scheme_validation:
                compatibility_score += 25.0
            else:
                compatibility_report["errors"].append(
                    "Color scheme failed internal validation"
                )
                is_compatible = False
        except Exception as e:
            compatibility_report["errors"].append(f"Color scheme validation error: {e}")
            is_compatible = False

        # Validate template configuration
        _logger.debug("Validating template configuration")
        try:
            if hasattr(render_template, "validate_performance"):
                template_validation = render_template.validate_performance()
                compatibility_report["template_valid"] = template_validation
                if template_validation:
                    compatibility_score += 25.0
                else:
                    compatibility_report["warnings"].append(
                        "Template performance validation failed"
                    )
            else:
                compatibility_report["template_valid"] = True
                compatibility_score += 25.0
        except Exception as e:
            compatibility_report["errors"].append(f"Template validation error: {e}")
            is_compatible = False

        # Check render mode optimization consistency between assets
        _logger.debug("Checking render mode optimization consistency")
        template_type = type(render_template).__name__

        if template_type == "RGBTemplate":
            # RGB templates work well with all color schemes
            compatibility_report["validation_details"][
                "render_mode_match"
            ] = "rgb_optimized"
            compatibility_score += 15.0
        elif template_type == "MatplotlibTemplate":
            # Matplotlib templates may have specific color scheme requirements
            compatibility_report["validation_details"][
                "render_mode_match"
            ] = "human_optimized"

            # Check if color scheme supports matplotlib rendering
            if (
                hasattr(color_scheme, "supports_matplotlib")
                and not color_scheme.supports_matplotlib()
            ):
                compatibility_report["warnings"].append(
                    "Color scheme may not be optimized for matplotlib rendering"
                )
                compatibility_score += 10.0
            else:
                compatibility_score += 15.0

        # Verify performance target alignment and resource requirements
        _logger.debug("Verifying performance target alignment")

        # Check memory requirements compatibility
        color_scheme_memory = getattr(color_scheme, "estimated_memory_mb", 5.0)
        template_memory = getattr(render_template, "memory_limit_mb", 10.0)
        total_memory = color_scheme_memory + template_memory

        if total_memory > 50.0:  # Total system memory limit
            compatibility_report["warnings"].append(
                f"Combined memory usage ({total_memory:.1f}MB) may exceed system limits"
            )
            compatibility_score += 5.0
        else:
            compatibility_report["performance_compatible"] = True
            compatibility_score += 15.0

        # Validate visual consistency across RGB array and matplotlib rendering
        _logger.debug("Validating visual consistency")

        # Check color space compatibility
        color_space = getattr(color_scheme, "color_space", "RGB")
        if color_space == "RGB":
            compatibility_report["visual_consistency"] = True
            compatibility_score += 10.0
        else:
            compatibility_report["warnings"].append(
                f"Non-RGB color space ({color_space}) may cause rendering inconsistencies"
            )
            compatibility_score += 5.0

        # Apply strict validation rules including edge cases if strict_validation enabled
        if strict_validation:
            _logger.debug("Applying strict validation rules")

            # Check for edge case compatibility
            if hasattr(color_scheme, "handles_edge_cases"):
                if not color_scheme.handles_edge_cases():
                    compatibility_report["warnings"].append(
                        "Color scheme may not handle edge cases properly in strict mode"
                    )
                else:
                    compatibility_score += 5.0

            # Verify numerical precision compatibility
            color_precision = getattr(color_scheme, "numerical_precision", 1e-6)
            template_precision = getattr(render_template, "numerical_precision", 1e-6)

            if abs(color_precision - template_precision) > 1e-8:
                compatibility_report["warnings"].append(
                    "Different numerical precisions may cause rendering artifacts"
                )
            else:
                compatibility_score += 5.0

        # Test actual rendering compatibility with sample operations
        _logger.debug("Testing actual rendering compatibility")
        try:
            # This would ideally test a small sample rendering operation
            # For now, we validate that required methods are available
            required_color_methods = [
                "get_concentration_colormap",
                "apply_to_rgb_array",
            ]
            for method in required_color_methods:
                if not hasattr(color_scheme, method):
                    compatibility_report["errors"].append(
                        f"Color scheme missing required method: {method}"
                    )
                    is_compatible = False
                    break
            else:
                compatibility_score += 5.0

            # Check template rendering methods
            if template_type == "RGBTemplate":
                required_methods = ["generate_frame"]
            else:
                required_methods = ["create_concentration_plot"]

            for method in required_methods:
                if not hasattr(render_template, method):
                    compatibility_report["errors"].append(
                        f"Template missing required method: {method}"
                    )
                    is_compatible = False
                    break
            else:
                compatibility_score += 5.0

        except Exception as e:
            compatibility_report["warnings"].append(
                f"Sample rendering test failed: {e}"
            )

        # Generate comprehensive compatibility report with recommendations
        compatibility_report["compatibility_score"] = min(compatibility_score, 100.0)

        # Determine overall compatibility based on errors and score
        if not compatibility_report["errors"] and compatibility_score >= 70.0:
            compatibility_report["overall_compatible"] = True
        else:
            is_compatible = False

        # Generate recommendations based on validation results
        if compatibility_report["warnings"]:
            compatibility_report["recommendations"].extend(
                [
                    "Review warning messages for potential compatibility improvements",
                    "Consider adjusting color scheme or template configuration",
                ]
            )

        if compatibility_score < 70.0:
            compatibility_report["recommendations"].append(
                "Compatibility score below recommended threshold - consider alternative assets"
            )

        if strict_validation and compatibility_score < 90.0:
            compatibility_report["recommendations"].append(
                "Strict validation active - ensure all edge cases are properly handled"
            )

        # Return validation status and detailed compatibility analysis
        _logger.info(
            f"Asset compatibility validation completed: {is_compatible} "
            f"(score: {compatibility_score:.1f}%)"
        )

        return is_compatible, compatibility_report

    except Exception as e:
        _logger.error(f"Asset compatibility validation failed: {e}")
        compatibility_report["errors"].append(f"Validation process error: {e}")
        compatibility_report["overall_compatible"] = False
        return False, compatibility_report


# Export comprehensive public interface following the specification
__all__ = [
    # Color scheme classes and enums
    "ColorScheme",
    "PredefinedScheme",
    # Rendering template classes
    "RGBTemplate",
    "MatplotlibTemplate",
    "BaseRenderTemplate",
    "TemplateConfig",
    "TemplateQuality",
    # Factory functions for asset creation
    "create_default_scheme",
    "create_color_scheme",
    "validate_color_scheme",
    "create_rgb_template",
    "create_matplotlib_template",
    "get_template_registry",
    # Package constants
    "DEFAULT_COLORMAP",
    "ASSETS_VERSION",
    # Utility functions
    "get_default_assets",
    "validate_asset_compatibility",
]
