"""
Rendering Module Initialization for Plume Navigation Simulation

This module provides unified access to dual-mode plume navigation visualization components
including BaseRenderer abstract interface, NumpyRGBRenderer for programmatic RGB arrays,
MatplotlibRenderer for interactive human mode, ColorSchemeManager for consistent color
management, and factory functions for streamlined renderer creation with performance
optimization and cross-platform compatibility support.

The rendering pipeline supports two primary modes:
- RGB_ARRAY: High-performance NumPy array generation for machine learning pipelines
- HUMAN: Interactive matplotlib visualization for research and debugging

Key Components:
- BaseRenderer: Abstract interface defining rendering contracts
- NumpyRGBRenderer: Optimized RGB array renderer targeting <5ms generation
- MatplotlibRenderer: Interactive visualization with backend management
- ColorSchemeManager: Comprehensive color scheme management with accessibility support
- Factory Functions: Streamlined renderer creation with performance optimization

Architecture Features:
- Cross-platform compatibility with graceful fallback mechanisms
- Performance optimization targeting RGB <5ms, human mode <50ms updates
- Comprehensive error handling and resource management
- Extensive logging and debugging support
- Enterprise-ready configuration management

External Dependencies:
- typing>=3.10: Type hints for factory functions and public API typing
- logging>=3.10: Module-level logging for operations and error reporting
- warnings>=3.10: Compatibility warnings for backend availability and limitations
"""

import atexit
import contextlib
import gc
import importlib
import signal
import sys
from pathlib import Path

# Standard library imports for configuration and logging
from typing import Any, Dict, List, Optional, Tuple


def _import_stdlib_module(module_name: str):
    """Import a standard library module even when repo-local packages shadow it."""

    existing = sys.modules.get(module_name)
    repo_root = Path(__file__).resolve().parents[4]
    repo_root_str = str(repo_root)

    if existing is not None:
        module_file = getattr(existing, "__file__", None)
        module_origin = getattr(getattr(existing, "__spec__", None), "origin", None)
        if (
            module_file
            and not str(Path(module_file).resolve()).startswith(repo_root_str)
        ) or module_origin in {"built-in", "frozen"}:
            return existing

        sys.modules.pop(module_name, None)

    original_path = list(sys.path)
    try:
        sanitized_path: List[str] = []
        for entry in original_path:
            try:
                resolved = str(Path(entry).resolve())
            except (OSError, RuntimeError, ValueError):
                sanitized_path.append(entry)
                continue

            if resolved.startswith(repo_root_str):
                continue

            sanitized_path.append(entry)

        sys.path = sanitized_path
        module = importlib.import_module(module_name)
    finally:
        sys.path = original_path

    return module


logging = _import_stdlib_module("logging")
warnings = _import_stdlib_module("warnings")

# Internal imports from core types module
from ..core.types import GridSize  # noqa: E402
from ..core.types import RenderMode  # noqa: E402
from ..utils.exceptions import ValidationError  # noqa: E402

# Internal imports from base renderer module
from .base_renderer import BaseRenderer  # noqa: E402
from .base_renderer import RenderContext  # noqa: E402
from .base_renderer import RenderingMetrics  # noqa: E402
from .base_renderer import create_render_context  # noqa: E402
from .base_renderer import validate_rendering_parameters  # noqa: E402

# Internal imports from color scheme management module
from .color_schemes import ColorSchemeManager  # noqa: E402
from .color_schemes import CustomColorScheme  # noqa: E402
from .color_schemes import PredefinedScheme  # noqa: E402
from .color_schemes import create_color_scheme  # noqa: E402
from .color_schemes import get_default_scheme  # noqa: E402

# Internal imports from matplotlib visualization module
from .matplotlib_viz import MatplotlibBackendManager  # noqa: E402
from .matplotlib_viz import MatplotlibRenderer  # noqa: E402
from .matplotlib_viz import create_matplotlib_renderer  # noqa: E402
from .matplotlib_viz import detect_matplotlib_capabilities  # noqa: E402

# Internal imports from NumPy RGB renderer module
from .numpy_rgb import NumpyRGBRenderer  # noqa: E402
from .numpy_rgb import create_rgb_renderer  # noqa: E402
from .numpy_rgb import generate_rgb_array_fast  # noqa: E402
from .numpy_rgb import validate_rgb_array_output  # noqa: E402

# Module-level logger configuration
_logger = logging.getLogger("plume_nav_sim.render")
_logger.setLevel(logging.INFO)

# Default configuration dictionaries for renderer optimization
_DEFAULT_RGB_CONFIG = {
    "enable_caching": True,
    "optimize_for_performance": True,
    "validate_output": True,
    "use_float32_precision": True,
    "enable_vectorization": True,
    "memory_pool_size": 64,  # MB
    "cache_size_limit": 100,  # number of cached renders
}

_DEFAULT_MATPLOTLIB_CONFIG = {
    "backend_preferences": ["TkAgg", "Qt5Agg", "Agg"],
    "enable_interactive": True,
    "figure_size": (8, 8),
    "dpi": 100,
    "enable_tight_layout": True,
    "animation_interval": 50,  # ms
    "resource_cleanup_interval": 10,  # renders between cleanup
}

# Global renderer registry for resource tracking and management
_RENDERER_REGISTRY = {}

# Performance and resource management constants
_MAX_CONCURRENT_RENDERERS = 10
_MEMORY_WARNING_THRESHOLD_MB = 500
_PERFORMANCE_WARNING_THRESHOLD_MS = 100


def _get_color_scheme(color_scheme_name: Optional[str]):
    """Create a color scheme or fall back to default with logging.

    Returns a valid color scheme object.
    """
    try:
        if color_scheme_name:
            scheme = create_color_scheme(color_scheme_name, validate_accessibility=True)
            _logger.debug(f"Created custom color scheme: {color_scheme_name}")
            return scheme
        scheme = get_default_scheme()
        _logger.debug("Using default color scheme")
        return scheme
    except Exception as e:  # pragma: no cover - defensive logging
        _logger.warning(f"Color scheme creation failed: {e}, using default")
        return get_default_scheme()


def _merge_renderer_options(renderer_options: Optional[dict]):
    """Merge user renderer_options with module defaults and return two dicts."""
    rgb_config = _DEFAULT_RGB_CONFIG.copy()
    matplotlib_config = _DEFAULT_MATPLOTLIB_CONFIG.copy()

    if renderer_options:
        if "rgb_options" in renderer_options:
            rgb_config.update(renderer_options["rgb_options"])
        if "matplotlib_options" in renderer_options:
            matplotlib_config.update(renderer_options["matplotlib_options"])

    return rgb_config, matplotlib_config


def _create_rgb_renderer_safe(
    grid_size: GridSize, color_scheme, rgb_config: Dict[str, Any], allow_fallback: bool
):
    """Attempt to create the RGB renderer; return (renderer_or_none, error_or_none)."""
    try:
        renderer = create_rgb_renderer(
            grid_size=grid_size,
            color_scheme=color_scheme,
            enable_caching=rgb_config["enable_caching"],
            optimize_for_performance=rgb_config["optimize_for_performance"],
            validate_output=rgb_config["validate_output"],
        )
        _logger.debug("RGB renderer created successfully")
        return renderer, None
    except Exception as e:  # pragma: no cover - defensive logging
        _logger.error(f"RGB renderer creation failed: {e}")
        if not allow_fallback:
            raise
        return None, str(e)


def _create_matplotlib_renderer_safe(
    grid_size: GridSize, color_scheme, mpl_config: Dict[str, Any], allow_fallback: bool
):
    """Attempt to create the Matplotlib renderer; return (renderer_or_none, error_or_none)."""
    try:
        renderer = create_matplotlib_renderer(
            grid_size=grid_size,
            color_scheme=color_scheme,
            backend_preferences=mpl_config["backend_preferences"],
            figure_size=mpl_config["figure_size"],
            enable_interactive=mpl_config["enable_interactive"],
        )
        _logger.debug("Matplotlib renderer created successfully")
        return renderer, None
    except Exception as e:  # pragma: no cover - defensive logging
        _logger.error(f"Matplotlib renderer creation failed: {e}")
        if not allow_fallback:
            raise
        return None, str(e)


def _auto_select_primary_mode(primary_mode: Optional[RenderMode], rgb_renderer):
    """Choose a primary mode if none provided and log the decision."""
    if primary_mode is None:
        selected = RenderMode.RGB_ARRAY if rgb_renderer else RenderMode.HUMAN
        _logger.debug(f"Auto-selected primary mode: {selected}")
        return selected
    return primary_mode


def _build_fallback_config(
    enable_rgb_fallback: bool,
    enable_matplotlib_fallback: bool,
    rgb_renderer,
    matplotlib_renderer,
    primary_mode: RenderMode,
) -> Dict[str, Any]:
    """Create the fallback configuration dict in one place."""
    return {
        "enable_rgb_fallback": enable_rgb_fallback and rgb_renderer is not None,
        "enable_matplotlib_fallback": enable_matplotlib_fallback
        and matplotlib_renderer is not None,
        "primary_mode": primary_mode,
    }


def _test_renderer_rgb(grid_size: GridSize, rgb_renderer):
    """Run a quick render test for the RGB renderer."""
    if not rgb_renderer:
        return None
    try:
        ctx = create_render_context(grid_size)
        t0 = logging.time.time()
        result = rgb_renderer.render(ctx, mode=RenderMode.RGB_ARRAY)
        dt = logging.time.time() - t0
        return {
            "success": True,
            "duration_ms": dt * 1000,
            "array_shape": (result.shape if hasattr(result, "shape") else None),
        }
    except Exception as e:  # pragma: no cover - defensive logging
        return {"success": False, "error": str(e)}


def _test_renderer_matplotlib(grid_size: GridSize, matplotlib_renderer):
    """Run a quick render test for the Matplotlib renderer."""
    if not matplotlib_renderer:
        return None
    try:
        ctx = create_render_context(grid_size)
        t0 = logging.time.time()
        matplotlib_renderer.render(ctx, mode=RenderMode.HUMAN)
        dt = logging.time.time() - t0
        return {"success": True, "duration_ms": dt * 1000}
    except Exception as e:  # pragma: no cover - defensive logging
        return {"success": False, "error": str(e)}


def create_dual_mode_renderer(
    grid_size: GridSize,
    color_scheme_name: Optional[str] = None,
    primary_mode: Optional[RenderMode] = None,
    enable_rgb_fallback: bool = True,
    enable_matplotlib_fallback: bool = True,
    renderer_options: Optional[dict] = None,
) -> Dict[str, Any]:
    """
    Factory function to create dual-mode renderer supporting both RGB array and human
    visualization modes with intelligent backend selection, performance optimization,
    and unified color scheme management for comprehensive plume navigation visualization.

    This function creates a unified rendering system that can seamlessly switch between
    RGB array mode (for programmatic processing) and human mode (for interactive
    visualization), providing fallback mechanisms and performance optimization.

    Args:
        grid_size: Grid dimensions for renderer configuration and memory allocation
        color_scheme_name: Optional color scheme name, defaults to standard scheme
        primary_mode: Preferred rendering mode, defaults to RGB_ARRAY for performance
        enable_rgb_fallback: Whether to enable RGB fallback for matplotlib failures
        enable_matplotlib_fallback: Whether to enable matplotlib fallback for headless environments
        renderer_options: Optional configuration overrides for fine-tuning performance

    Returns:
        dict: Dictionary containing RGB and matplotlib renderers with unified interface
              and fallback configuration, including performance metrics and capability info

    Raises:
        ValidationError: If grid_size validation fails or renderer creation encounters errors
        ConfigurationError: If renderer_options contain invalid configuration parameters

    Example:
        >>> from plume_nav_sim.core.types import GridSize, RenderMode
        >>> grid = GridSize(128, 128)
        >>> dual_renderer = create_dual_mode_renderer(
        ...     grid_size=grid,
        ...     primary_mode=RenderMode.HUMAN,
        ...     color_scheme_name="high_contrast"
        ... )
        >>> rgb_renderer = dual_renderer['rgb_renderer']
        >>> matplotlib_renderer = dual_renderer['matplotlib_renderer']
    """
    _logger.info(f"Creating dual-mode renderer for grid size {grid_size.to_tuple()}")

    try:
        # Validate grid_size dimensions and memory requirements for dual-mode rendering feasibility
        if not isinstance(grid_size, GridSize):
            raise ValidationError(
                "Invalid grid_size type",
                parameter_name="grid_size",
                invalid_value=type(grid_size).__name__,
                expected_format="GridSize instance",
            )

        # Check memory requirements
        estimated_memory = grid_size.estimate_memory_mb()
        if estimated_memory > _MEMORY_WARNING_THRESHOLD_MB:
            _logger.warning(
                f"Grid size {grid_size.to_tuple()} requires {estimated_memory:.1f}MB memory, "
                f"exceeding warning threshold {_MEMORY_WARNING_THRESHOLD_MB}MB"
            )

        # Create unified color scheme
        color_scheme = _get_color_scheme(color_scheme_name)

        # Merge renderer options with defaults
        rgb_config, matplotlib_config = _merge_renderer_options(renderer_options)

        # Configure RGB renderer safe
        rgb_renderer, rgb_error = _create_rgb_renderer_safe(
            grid_size,
            color_scheme,
            rgb_config,
            allow_fallback=enable_matplotlib_fallback,
        )

        # Configure matplotlib renderer safe
        matplotlib_renderer, matplotlib_error = _create_matplotlib_renderer_safe(
            grid_size,
            color_scheme,
            matplotlib_config,
            allow_fallback=enable_rgb_fallback,
        )

        # Set primary_mode for preferred rendering method with automatic fallback
        primary_mode = _auto_select_primary_mode(primary_mode, rgb_renderer)

        # Configure fallback mechanisms
        fallback_config = _build_fallback_config(
            enable_rgb_fallback,
            enable_matplotlib_fallback,
            rgb_renderer,
            matplotlib_renderer,
            primary_mode,
        )

        # Test both renderers functionality with sample operations
        test_results = {}
        rgb_test = _test_renderer_rgb(grid_size, rgb_renderer)
        if rgb_test is not None:
            test_results["rgb_test"] = rgb_test
        mpl_test = _test_renderer_matplotlib(grid_size, matplotlib_renderer)
        if mpl_test is not None:
            test_results["matplotlib_test"] = mpl_test

        class UnifiedRenderer:
            """Unified interface wrapper providing transparent mode switching and fallback handling."""

            def __init__(self, rgb_renderer, matplotlib_renderer, config):
                self.rgb_renderer = rgb_renderer
                self.matplotlib_renderer = matplotlib_renderer
                self.config = config
                self.metrics = RenderingMetrics()

            def render(self, context: RenderContext, mode: RenderMode = None):
                """Render with automatic fallback and performance tracking."""
                if mode is None:
                    mode = self.config["primary_mode"]

                start_time = logging.time.time()

                try:
                    if mode == RenderMode.RGB_ARRAY:
                        if self.rgb_renderer:
                            result = self.rgb_renderer.render(context, mode)
                        elif (
                            self.config["enable_matplotlib_fallback"]
                            and self.matplotlib_renderer
                        ):
                            _logger.warning(
                                "RGB renderer unavailable, falling back to matplotlib"
                            )
                            result = self.matplotlib_renderer.render(
                                context, RenderMode.RGB_ARRAY
                            )
                        else:
                            raise RuntimeError("No RGB rendering capability available")
                    elif self.matplotlib_renderer:
                        result = self.matplotlib_renderer.render(context, mode)
                    elif self.config["enable_rgb_fallback"] and self.rgb_renderer:
                        _logger.warning(
                            "Matplotlib renderer unavailable, falling back to RGB"
                        )
                        result = self.rgb_renderer.render(context, RenderMode.RGB_ARRAY)
                    else:
                        raise RuntimeError("No human rendering capability available")

                    # Record performance metrics
                    duration = logging.time.time() - start_time
                    self.metrics.record_rendering(mode, duration * 1000)

                    return result

                except Exception as e:
                    _logger.error(f"Rendering failed for mode {mode}: {e}")
                    raise

            def get_capabilities(self):
                """Return capability information for the unified renderer."""
                return {
                    "rgb_available": self.rgb_renderer is not None,
                    "matplotlib_available": self.matplotlib_renderer is not None,
                    "fallback_enabled": self.config["enable_rgb_fallback"]
                    or self.config["enable_matplotlib_fallback"],
                    "primary_mode": self.config["primary_mode"],
                }

            # Register dual-mode renderer in _RENDERER_REGISTRY for resource tracking

        # Register dual-mode renderer in _RENDERER_REGISTRY for resource tracking
        renderer_id = f"dual_mode_{len(_RENDERER_REGISTRY)}"
        unified_renderer = UnifiedRenderer(
            rgb_renderer, matplotlib_renderer, fallback_config
        )

        _RENDERER_REGISTRY[renderer_id] = {
            "unified_renderer": unified_renderer,
            "rgb_renderer": rgb_renderer,
            "matplotlib_renderer": matplotlib_renderer,
            "creation_time": logging.time.time(),
            "grid_size": grid_size,
            "color_scheme": color_scheme,
        }

        # Log dual-mode renderer creation with configuration details
        _logger.info(
            f"Dual-mode renderer created successfully: ID={renderer_id}, "
            f"RGB={'available' if rgb_renderer else 'failed'}, "
            f"Matplotlib={'available' if matplotlib_renderer else 'failed'}, "
            f"Primary={primary_mode}"
        )

        # Return comprehensive dual-mode renderer dictionary
        return {
            "unified_renderer": unified_renderer,
            "rgb_renderer": rgb_renderer,
            "matplotlib_renderer": matplotlib_renderer,
            "color_scheme": color_scheme,
            "configuration": {
                "grid_size": grid_size,
                "primary_mode": primary_mode,
                "fallback_config": fallback_config,
                "rgb_config": rgb_config,
                "matplotlib_config": matplotlib_config,
            },
            "capabilities": unified_renderer.get_capabilities(),
            "test_results": test_results,
            "errors": {"rgb_error": rgb_error, "matplotlib_error": matplotlib_error},
            "renderer_id": renderer_id,
        }

    except Exception as e:
        _logger.error(f"Dual-mode renderer creation failed: {e}")
        raise


def validate_renderer_config(
    renderer_config: Dict[str, Any],
    check_system_capabilities: bool = True,
    check_performance_targets: bool = True,
    check_accessibility: bool = False,
    strict_validation: bool = False,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Comprehensive validation function for renderer configuration ensuring compatibility,
    performance feasibility, accessibility compliance, and system capability assessment
    for robust plume navigation visualization setup.

    This function performs thorough validation of renderer configuration parameters,
    system capabilities, performance requirements, and accessibility standards to
    ensure optimal rendering setup and identify potential issues before runtime.

    Args:
        renderer_config: Dictionary containing renderer configuration parameters
        check_system_capabilities: Whether to verify system rendering capabilities
        check_performance_targets: Whether to validate against performance targets
        check_accessibility: Whether to check accessibility compliance for color schemes
        strict_validation: Whether to apply strict validation rules with comprehensive testing

    Returns:
        tuple: (is_valid: bool, validation_report: dict) with comprehensive analysis
               and optimization recommendations for configuration improvement

    Raises:
        ValidationError: If critical validation failures occur in strict mode

    Example:
        >>> config = {
        ...     'grid_size': (128, 128),
        ...     'color_scheme': 'high_contrast',
        ...     'render_modes': ['rgb_array', 'human'],
        ...     'performance_targets': {'rgb_ms': 5, 'human_ms': 50}
        ... }
        >>> is_valid, report = validate_renderer_config(config, check_accessibility=True)
        >>> if not is_valid:
        ...     print("Configuration issues found:", report['warnings'])
    """
    _logger.debug("Starting comprehensive renderer configuration validation")

    validation_report = {
        "is_valid": True,
        "errors": [],
        "warnings": [],
        "recommendations": [],
        "system_capabilities": {},
        "performance_analysis": {},
        "accessibility_analysis": {},
        "configuration_summary": {},
    }

    def _coerce_grid_size(grid_size_val) -> Optional[GridSize]:
        try:
            if isinstance(grid_size_val, (tuple, list)):
                grid_obj_local = GridSize(grid_size_val[0], grid_size_val[1])
            elif isinstance(grid_size_val, GridSize):
                grid_obj_local = grid_size_val
            else:
                validation_report["errors"].append(
                    f"Invalid grid_size type: {type(grid_size_val).__name__}, expected GridSize, tuple, or list"
                )
                validation_report["is_valid"] = False
                return None

            memory_estimate = grid_obj_local.estimate_memory_mb()
            validation_report["configuration_summary"][
                "memory_estimate_mb"
            ] = memory_estimate
            if memory_estimate > _MEMORY_WARNING_THRESHOLD_MB:
                validation_report["warnings"].append(
                    f"Memory estimate {memory_estimate:.1f}MB exceeds warning threshold {_MEMORY_WARNING_THRESHOLD_MB}MB"
                )
                validation_report["recommendations"].append(
                    "Consider reducing grid size to improve memory usage"
                )
            return grid_obj_local
        except Exception as e:
            validation_report["errors"].append(f"Grid size validation failed: {e}")
            validation_report["is_valid"] = False
            return None

    def _check_system_capabilities_section():
        try:
            capabilities = detect_rendering_capabilities(
                test_matplotlib_backends=True,
                test_performance_characteristics=True,
                test_color_scheme_support=True,
                generate_recommendations=True,
            )
            validation_report["system_capabilities"] = capabilities

            if not capabilities.get("numpy_available", False):
                validation_report["errors"].append(
                    "NumPy not available - required for RGB rendering"
                )
                validation_report["is_valid"] = False

            if not capabilities.get("matplotlib_backends", []):
                validation_report["warnings"].append(
                    "No matplotlib backends available - human mode disabled"
                )
                validation_report["recommendations"].append(
                    "Install matplotlib with GUI backend support"
                )
        except Exception as e:
            validation_report["warnings"].append(f"System capability check failed: {e}")

    def _validate_color_scheme_section(color_scheme_val):
        if not color_scheme_val:
            return
        try:
            if isinstance(color_scheme_val, str):
                if hasattr(PredefinedScheme, color_scheme_val.upper()):
                    scheme_obj = getattr(PredefinedScheme, color_scheme_val.upper())
                    validation_report["configuration_summary"][
                        "color_scheme"
                    ] = scheme_obj.value
                else:
                    validation_report["warnings"].append(
                        f"Unknown color scheme: {color_scheme_val}"
                    )
                    validation_report["recommendations"].append(
                        "Use predefined scheme or create custom scheme"
                    )

            if check_accessibility:
                try:
                    scheme_manager = ColorSchemeManager()
                    scheme_result = scheme_manager.validate_scheme(
                        color_scheme_val,
                        check_accessibility=True,
                        check_contrast_ratios=True,
                    )
                    validation_report["accessibility_analysis"] = scheme_result

                    if not scheme_result.get("accessibility_compliant", True):
                        validation_report["warnings"].append(
                            "Color scheme may not meet accessibility standards"
                        )
                        validation_report["recommendations"].append(
                            "Consider using high_contrast or colorblind_friendly schemes"
                        )
                except Exception as e:
                    validation_report["warnings"].append(
                        f"Accessibility validation failed: {e}"
                    )
        except Exception as e:
            validation_report["warnings"].append(f"Color scheme validation failed: {e}")

    def _analyze_performance_section(grid_obj_local: Optional[GridSize]):
        if not (check_performance_targets and grid_obj_local):
            return
        performance_targets = renderer_config.get("performance_targets", {})
        rgb_target = performance_targets.get("rgb_ms", 5)
        human_target = performance_targets.get("human_ms", 50)

        try:
            estimated_rgb_time = grid_obj_local.total_cells() / 1000000 * 2
            estimated_human_time = estimated_rgb_time * 10

            validation_report["performance_analysis"] = {
                "estimated_rgb_ms": estimated_rgb_time,
                "estimated_human_ms": estimated_human_time,
                "rgb_target_ms": rgb_target,
                "human_target_ms": human_target,
                "rgb_feasible": estimated_rgb_time <= rgb_target,
                "human_feasible": estimated_human_time <= human_target,
            }

            if estimated_rgb_time > rgb_target:
                validation_report["warnings"].append(
                    f"RGB rendering may exceed target {rgb_target}ms (estimated {estimated_rgb_time:.1f}ms)"
                )
                validation_report["recommendations"].append(
                    "Enable performance optimization or reduce grid size"
                )

            if estimated_human_time > human_target:
                validation_report["warnings"].append(
                    f"Human rendering may exceed target {human_target}ms (estimated {estimated_human_time:.1f}ms)"
                )
                validation_report["recommendations"].append(
                    "Consider reducing update frequency for human mode"
                )
        except Exception as e:
            validation_report["warnings"].append(f"Performance analysis failed: {e}")

    def _strict_validation_section(grid_obj_local: Optional[GridSize]):
        if not strict_validation:
            return
        if len(validation_report["errors"]) > 0:
            error_msg = (
                f"Strict validation failed: {'; '.join(validation_report['errors'])}"
            )
            _logger.error(error_msg)
            raise ValidationError(
                error_msg,
                parameter_name="renderer_config",
                invalid_value=renderer_config,
            )
        if grid_obj_local:
            try:
                create_render_context(grid_obj_local)
                validation_report["configuration_summary"][
                    "test_context_created"
                ] = True
            except Exception as e:
                validation_report["errors"].append(f"Test context creation failed: {e}")
                validation_report["is_valid"] = False

    def _add_optimization_recommendations(render_modes_val, optimization_settings_val):
        if optimization_settings_val.get("enable_caching", True):
            validation_report["recommendations"].append(
                "Caching enabled - ensure sufficient memory for cache"
            )
        if len(render_modes_val) > 1:
            validation_report["recommendations"].append(
                "Dual-mode rendering - consider using create_dual_mode_renderer()"
            )

    try:
        grid_size_val = renderer_config.get("grid_size")
        color_scheme_val = renderer_config.get("color_scheme", "default")
        render_modes_val = renderer_config.get("render_modes", ["rgb_array"])
        optimization_settings_val = renderer_config.get("optimization_settings", {})

        grid_obj = _coerce_grid_size(grid_size_val)

        if check_system_capabilities:
            _check_system_capabilities_section()

        _validate_color_scheme_section(color_scheme_val)
        _analyze_performance_section(grid_obj)
        _strict_validation_section(grid_obj)
        _add_optimization_recommendations(render_modes_val, optimization_settings_val)

        if len(validation_report["errors"]) > 0:
            validation_report["is_valid"] = False

        _logger.debug(
            f"Configuration validation completed: valid={validation_report['is_valid']}"
        )
        return validation_report["is_valid"], validation_report

    except Exception as e:
        _logger.error(f"Renderer configuration validation failed: {e}")
        validation_report["is_valid"] = False
        validation_report["errors"].append(f"Validation exception: {e}")
        if strict_validation:
            raise
        return False, validation_report


def detect_rendering_capabilities(
    test_matplotlib_backends: bool = True,
    test_performance_characteristics: bool = False,
    test_color_scheme_support: bool = False,
    generate_recommendations: bool = True,
) -> Dict[str, Any]:
    """
    System capability detection function for comprehensive rendering support assessment
    including matplotlib backend availability, NumPy performance characteristics,
    color scheme compatibility, and cross-platform rendering support analysis.

    This function systematically evaluates the current system's rendering capabilities,
    identifying available backends, performance characteristics, and potential limitations
    to provide comprehensive capability reporting and optimization recommendations.

    Args:
        test_matplotlib_backends: Whether to test matplotlib backend availability and compatibility
        test_performance_characteristics: Whether to benchmark NumPy and rendering performance
        test_color_scheme_support: Whether to validate color scheme and colormap support
        generate_recommendations: Whether to generate optimization and configuration recommendations

    Returns:
        dict: Comprehensive capabilities report with backend availability, performance metrics,
              compatibility analysis, and optimization recommendations for informed renderer selection

    Example:
        >>> capabilities = detect_rendering_capabilities(test_performance_characteristics=True)
        >>> if capabilities['matplotlib_backends']:
        ...     print("Available backends:", capabilities['matplotlib_backends'])
        >>> if not capabilities['display_available']:
        ...     print("Running in headless mode")
    """
    _logger.debug("Starting comprehensive rendering capability detection")

    capabilities = {
        "detection_timestamp": logging.time.time(),
        "platform_info": {
            "system": sys.platform,
            "python_version": sys.version,
        },
        "numpy_available": False,
        "numpy_version": None,
        "matplotlib_available": False,
        "matplotlib_version": None,
        "matplotlib_backends": [],
        "display_available": False,
        "performance_characteristics": {},
        "color_scheme_support": {},
        "recommendations": [],
        "warnings": [],
    }

    try:
        # Detect NumPy availability and version compatibility
        try:
            import numpy as np

            capabilities["numpy_available"] = True
            capabilities["numpy_version"] = np.__version__
            _logger.debug(f"NumPy {np.__version__} detected")

            # Basic NumPy performance test
            if test_performance_characteristics:
                try:
                    _extracted_from_detect_rendering_capabilities_65(np, capabilities)
                except Exception as e:
                    capabilities["warnings"].append(
                        f"NumPy performance test failed: {e}"
                    )

        except ImportError:
            capabilities["warnings"].append(
                "NumPy not available - RGB rendering disabled"
            )

        # Test matplotlib backend availability if requested
        if test_matplotlib_backends:
            try:
                _extracted_from_detect_rendering_capabilities_99(
                    test_performance_characteristics, capabilities
                )
            except Exception as e:
                capabilities["warnings"].append(
                    f"Matplotlib capability detection failed: {e}"
                )

        # Assess display environment availability
        display_env = (
            sys.platform != "win32" and "DISPLAY" in sys.os.environ
            if hasattr(sys, "os")
            else True
        )
        if not display_env and sys.platform != "win32":
            capabilities["display_available"] = False
            capabilities["warnings"].append(
                "No DISPLAY environment variable - running in headless mode"
            )

        # Test color scheme support if requested
        if test_color_scheme_support:
            try:
                # Test colormap availability
                if capabilities["matplotlib_available"]:
                    import matplotlib.pyplot as plt

                    colormaps = ["viridis", "plasma", "inferno", "magma", "gray"]
                    available_colormaps = []

                    for cmap in colormaps:
                        with contextlib.suppress(Exception):
                            plt.get_cmap(cmap)
                            available_colormaps.append(cmap)
                    capabilities["color_scheme_support"] = {
                        "available_colormaps": available_colormaps,
                        "total_colormaps": len(available_colormaps),
                        "colormap_support_rating": (
                            "excellent"
                            if len(available_colormaps) >= 4
                            else "good" if len(available_colormaps) >= 2 else "limited"
                        ),
                    }

            except Exception as e:
                capabilities["warnings"].append(
                    f"Color scheme support test failed: {e}"
                )

        # Evaluate platform-specific rendering capabilities
        platform_capabilities = {
            "linux": {"full_support": True, "gui_toolkits": ["tkinter", "qt", "gtk"]},
            "darwin": {
                "full_support": True,
                "gui_toolkits": ["tkinter", "qt", "cocoa"],
            },
            "win32": {
                "full_support": False,
                "community_support": True,
                "gui_toolkits": ["tkinter", "qt"],
            },
        }

        platform = sys.platform
        if platform in platform_capabilities:
            capabilities["platform_support"] = platform_capabilities[platform]
            if not platform_capabilities[platform].get("full_support", False):
                capabilities["warnings"].append(
                    f"Platform {platform} has limited official support - community PRs accepted"
                )

        # Test headless operation compatibility
        if (
            capabilities["matplotlib_available"]
            and not capabilities["display_available"]
        ):
            try:
                import matplotlib

                matplotlib.use("Agg")  # Set headless backend
                capabilities["headless_compatible"] = True
                _logger.debug("Headless operation configured successfully")
            except Exception as e:
                capabilities["headless_compatible"] = False
                capabilities["warnings"].append(f"Headless operation setup failed: {e}")

        # Generate optimization recommendations if requested
        if generate_recommendations:
            _extracted_from_detect_rendering_capabilities_211(
                capabilities, test_performance_characteristics
            )
        # Assess memory availability and performance characteristics
        try:
            import psutil

            memory_info = psutil.virtual_memory()
            capabilities["system_resources"] = {
                "total_memory_gb": memory_info.total / (1024**3),
                "available_memory_gb": memory_info.available / (1024**3),
                "memory_usage_percent": memory_info.percent,
            }

            if memory_info.available < 1024**3:  # Less than 1GB available
                capabilities["warnings"].append(
                    "Low available memory - consider reducing grid size"
                )

        except ImportError:
            capabilities["warnings"].append(
                "psutil not available - cannot assess system resources"
            )

        _logger.info(
            f"Capability detection completed: NumPy={'✓' if capabilities['numpy_available'] else '✗'}, "
            f"Matplotlib={'✓' if capabilities['matplotlib_available'] else '✗'}, "
            f"Display={'✓' if capabilities['display_available'] else '✗'}"
        )

        return capabilities

    except Exception as e:
        _logger.error(f"Rendering capability detection failed: {e}")
        capabilities["warnings"].append(f"Detection error: {e}")
        return capabilities


# TODO Rename this here and in `detect_rendering_capabilities`
def _extracted_from_detect_rendering_capabilities_211(
    capabilities, test_performance_characteristics
):
    recommendations = []

    if not capabilities["numpy_available"]:
        recommendations.append("Install NumPy >=2.1.0 for RGB array rendering support")

    if not capabilities["matplotlib_available"]:
        recommendations.append(
            "Install matplotlib >=3.9.0 for human mode visualization"
        )
    elif not capabilities["matplotlib_backends"]:
        recommendations.append(
            "Install GUI toolkit (tkinter/qt) for interactive visualization"
        )

    if not capabilities["display_available"]:
        recommendations.append(
            "Configure headless operation with Agg backend for server deployment"
        )

    if test_performance_characteristics:
        numpy_perf = capabilities["performance_characteristics"].get(
            "numpy_performance_rating"
        )
        if numpy_perf in ["acceptable", "slow"]:
            recommendations.append(
                "Consider NumPy optimization or reduced grid size for better performance"
            )

    capabilities["recommendations"] = recommendations


# TODO Rename this here and in `detect_rendering_capabilities`
def _extracted_from_detect_rendering_capabilities_99(
    test_performance_characteristics, capabilities
):
    capabilities_result = detect_matplotlib_capabilities(
        test_all_backends=True,
        test_display_capability=True,
        test_performance=test_performance_characteristics,
    )

    capabilities["matplotlib_available"] = capabilities_result.get(
        "matplotlib_available", False
    )
    capabilities["matplotlib_version"] = capabilities_result.get("matplotlib_version")
    capabilities["matplotlib_backends"] = capabilities_result.get(
        "available_backends", []
    )
    capabilities["display_available"] = capabilities_result.get(
        "display_available", False
    )

    if test_performance_characteristics:
        capabilities["performance_characteristics"].update(
            capabilities_result.get("performance_metrics", {})
        )


# TODO Rename this here and in `detect_rendering_capabilities`
def _extracted_from_detect_rendering_capabilities_65(np, capabilities):
    import time

    test_array = np.random.random((128, 128, 3))
    start_time = time.time()
    _ = np.uint8(test_array * 255)
    numpy_duration = time.time() - start_time

    capabilities["performance_characteristics"]["numpy_conversion_ms"] = (
        numpy_duration * 1000
    )
    capabilities["performance_characteristics"]["numpy_performance_rating"] = (
        "excellent"
        if numpy_duration < 0.001
        else (
            "good"
            if numpy_duration < 0.005
            else "acceptable" if numpy_duration < 0.01 else "slow"
        )
    )


def _register_cleanup_handlers(
    renderer_registry: Dict[str, Any], enable_automatic_cleanup: bool = True
) -> None:
    """
    Internal function to register cleanup handlers for renderer resource management
    including matplotlib figure disposal, cache clearing, and memory cleanup for
    proper system resource management and prevention of memory leaks.

    Args:
        renderer_registry: Dictionary containing active renderer instances
        enable_automatic_cleanup: Whether to enable periodic automatic cleanup

    Returns:
        None: Registers cleanup handlers with atexit and signal handlers
    """

    def cleanup_all_renderers():
        """Clean up all registered renderers and their resources."""
        _logger.debug(f"Cleaning up {len(renderer_registry)} registered renderers")

        for renderer_id, renderer_data in renderer_registry.items():
            try:
                # Clean up matplotlib renderer resources
                if (
                    "matplotlib_renderer" in renderer_data
                    and renderer_data["matplotlib_renderer"]
                ):
                    renderer_data["matplotlib_renderer"].cleanup_resources()

                # Clean up RGB renderer caches
                if (
                    "rgb_renderer" in renderer_data
                    and renderer_data["rgb_renderer"]
                    and hasattr(renderer_data["rgb_renderer"], "cleanup_resources")
                ):
                    renderer_data["rgb_renderer"].cleanup_resources()

                _logger.debug(f"Cleaned up renderer: {renderer_id}")

            except Exception as e:
                _logger.warning(f"Cleanup failed for renderer {renderer_id}: {e}")

        # Clear the registry
        renderer_registry.clear()

        # Force garbage collection
        gc.collect()

        _logger.info("Renderer cleanup completed")

    # Register atexit handler for automatic cleanup at program termination
    atexit.register(cleanup_all_renderers)

    # Set up signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        """Handle shutdown signals gracefully."""
        _logger.info(f"Received signal {signum}, cleaning up renderers")
        cleanup_all_renderers()
        sys.exit(0)

    with contextlib.suppress(AttributeError, OSError):
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    _logger.debug("Cleanup handlers registered successfully")


def _warn_about_limitations(
    capabilities: Dict[str, Any],
    requested_mode: Optional[str] = None,
    include_recommendations: bool = True,
) -> None:
    """
    Internal function to issue appropriate warnings about system limitations,
    missing dependencies, performance constraints, and platform-specific rendering
    capabilities for user awareness and troubleshooting guidance.

    Args:
        capabilities: System capability assessment from detect_rendering_capabilities
        requested_mode: Optional specific rendering mode being requested
        include_recommendations: Whether to include configuration recommendations

    Returns:
        None: Issues warnings and logs system limitations with guidance
    """
    # Analyze capabilities for missing dependencies and limitations
    if not capabilities.get("numpy_available", False):
        warnings.warn(
            "NumPy not available - RGB array rendering disabled. "
            "Install numpy>=2.1.0 for full rendering support.",
            UserWarning,
            stacklevel=2,
        )

    # Issue matplotlib backend warnings if interactive backends unavailable
    if capabilities.get("matplotlib_available", False):
        available_backends = capabilities.get("matplotlib_backends", [])
        interactive_backends = [
            b for b in available_backends if b not in ["Agg", "svg", "pdf"]
        ]

        if not interactive_backends:
            warnings.warn(
                "No interactive matplotlib backends available - human mode will use non-interactive display. "
                "Install tkinter or Qt for full interactive visualization.",
                UserWarning,
                stacklevel=2,
            )

        if not capabilities.get("display_available", True):
            _logger.info(
                "Running in headless environment - matplotlib configured for non-interactive use"
            )
    else:
        warnings.warn(
            "Matplotlib not available - human mode visualization disabled. "
            "Install matplotlib>=3.9.0 for interactive rendering support.",
            UserWarning,
            stacklevel=2,
        )

    # Warn about performance limitations if system doesn't meet targets
    performance_metrics = capabilities.get("performance_characteristics", {})
    numpy_rating = performance_metrics.get("numpy_performance_rating", "unknown")

    if numpy_rating in ["acceptable", "slow"]:
        warnings.warn(
            f"NumPy performance rating: {numpy_rating}. "
            "Consider optimizing system or reducing grid size for better performance.",
            PerformanceWarning,
            stacklevel=2,
        )

    # Provide platform-specific warnings for Windows community support
    if sys.platform == "win32":
        platform_support = capabilities.get("platform_support", {})
        if not platform_support.get("full_support", True):
            _logger.warning(
                "Running on Windows with community support - "
                "some features may have limited testing coverage"
            )

    # Issue accessibility warnings if high contrast support unavailable
    color_support = capabilities.get("color_scheme_support", {})
    if color_support.get("colormap_support_rating") == "limited":
        _logger.warning(
            "Limited colormap support detected - some accessibility features may be unavailable"
        )

    # Include configuration recommendations if requested
    if include_recommendations:
        if recommendations := capabilities.get("recommendations", []):
            _logger.info("Configuration recommendations:")
            for i, recommendation in enumerate(recommendations, 1):
                _logger.info(f"  {i}. {recommendation}")


# Performance warning class for system capability issues
class PerformanceWarning(UserWarning):
    """Warning category for performance-related system limitations."""

    pass


# Register cleanup handlers on module import
_register_cleanup_handlers(_RENDERER_REGISTRY)

# Issue system capability warnings on import
try:
    _system_capabilities = detect_rendering_capabilities(
        test_matplotlib_backends=True,
        test_performance_characteristics=False,
        generate_recommendations=False,
    )
    _warn_about_limitations(_system_capabilities, include_recommendations=False)
except Exception as e:
    _logger.warning(f"Initial capability detection failed: {e}")

# Log module initialization
_logger.info(
    f"Rendering module initialized: "
    f"Registry capacity={_MAX_CONCURRENT_RENDERERS}, "
    f"Memory threshold={_MEMORY_WARNING_THRESHOLD_MB}MB"
)

# Comprehensive public interface exports for rendering pipeline
__all__ = [
    # Core renderer classes and interfaces
    "BaseRenderer",  # Abstract base renderer defining consistent interface
    "RenderContext",  # Immutable rendering context with environment state
    "RenderingMetrics",  # Performance metrics tracking for rendering operations
    # Concrete renderer implementations
    "NumpyRGBRenderer",  # High-performance RGB array renderer with optimization
    "MatplotlibRenderer",  # Interactive matplotlib renderer for human mode
    "MatplotlibBackendManager",  # Backend management for cross-platform compatibility
    # Color scheme management system
    "ColorSchemeManager",  # Central color scheme management with caching
    "CustomColorScheme",  # Custom color scheme configuration with validation
    "PredefinedScheme",  # Enumeration of predefined accessibility-compliant schemes
    # Factory functions for renderer creation
    "create_rgb_renderer",  # Factory for optimized NumpyRGBRenderer instances
    "create_matplotlib_renderer",  # Factory for configured matplotlib renderer
    "create_dual_mode_renderer",  # Factory for dual-mode rendering with fallback support
    "create_render_context",  # Factory for validated rendering context creation
    # Color scheme factory functions
    "create_color_scheme",  # Factory for custom color schemes with validation
    "get_default_scheme",  # Factory for default color scheme with standard colors
    # Validation and utility functions
    "validate_renderer_config",  # Comprehensive renderer configuration validation
    "validate_rendering_parameters",  # Rendering parameter validation with error reporting
    "validate_rgb_array_output",  # RGB array quality assurance and format compliance
    # System capability detection
    "detect_rendering_capabilities",  # System capability assessment and optimization recommendations
    "detect_matplotlib_capabilities",  # Matplotlib backend availability and performance assessment
    # Core types and enumerations
    "RenderMode",  # Rendering mode enumeration for dual-mode support
    "GridSize",  # Grid dimension representation for configuration
    # Utility functions for high-performance operations
    "generate_rgb_array_fast",  # High-performance RGB array generation with minimal overhead
    # Exception classes for error handling
    "ValidationError",  # Renderer configuration validation exception
    "PerformanceWarning",  # Performance-related system limitation warning
]
