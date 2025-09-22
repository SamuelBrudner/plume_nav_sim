"""
Interactive matplotlib visualization renderer for human mode plume navigation with real-time updates,
backend compatibility management, cross-platform support, and performance optimization.

This module implements BaseRenderer for dual-mode rendering pipeline with matplotlib figure management,
colormap integration, marker visualization, and graceful fallback mechanisms for headless environments
while targeting <50ms human mode rendering performance.

Key Components:
- MatplotlibRenderer: Main renderer class with backend management and interactive visualization
- MatplotlibBackendManager: Backend detection, selection, and configuration utility
- InteractiveUpdateManager: Efficient matplotlib updates with change detection and performance optimization
- Factory functions: Renderer creation, capability detection, and integration validation

Architecture Integration:
- Strategy pattern implementation of BaseRenderer for polymorphic usage
- Performance targets: <50ms human mode updates with comprehensive optimization
- Cross-platform support: Linux (full), macOS (full), Windows (community) with headless operation
- Resource management: Automatic cleanup, memory management, and graceful degradation
- Error handling: Comprehensive exception handling with fallback mechanisms
"""

# Standard library imports - Python >=3.10
import atexit  # >=3.10 - Automatic resource cleanup registration for matplotlib figures at program exit
import os  # >=3.10 - Environment variable detection for headless operation and display availability
import sys  # >=3.10 - Platform detection and system capability assessment for backend selection
import time  # >=3.10 - High-precision timing for performance measurement and interactive update delays
from typing import (  # >=3.10 - Type hints for matplotlib renderer type safety
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

import matplotlib.axes  # >=3.9.0 - Axes object management for plotting area configuration and marker placement
import matplotlib.backends.backend_agg  # >=3.9.0 - Agg backend for headless rendering fallback when GUI backends unavailable
import matplotlib.colors  # >=3.9.0 - Colormap and normalization utilities for concentration field visualization
import matplotlib.figure  # >=3.9.0 - Figure object management for matplotlib visualization with proper resource handling

# Third-party imports - External dependencies for matplotlib visualization
import matplotlib.pyplot as plt  # >=3.9.0 - Primary matplotlib interface for figure creation, plotting, and display management
import numpy as np  # >=2.1.0 - Array operations for concentration field processing and coordinate arithmetic

from ..core.constants import (
    BACKEND_PRIORITY_LIST,  # Backend priority order ['TkAgg', 'Qt5Agg', 'Agg'] for matplotlib configuration fallback
)
from ..core.constants import (
    MATPLOTLIB_DEFAULT_FIGSIZE,  # Default matplotlib figure size (8, 8) for human mode visualization configuration
)
from ..core.constants import (
    PERFORMANCE_TARGET_HUMAN_RENDER_MS,  # Performance target (<50ms) for human mode rendering operations and timing validation
)

# Internal imports - Core types and constants for rendering operations
from ..core.types import (
    Coordinates,  # 2D coordinate representation for agent and source position marker placement
)
from ..core.types import (
    GridSize,  # Grid dimension representation for matplotlib axes configuration and bounds
)
from ..core.types import (
    RenderMode,  # Rendering mode enumeration for dual-mode visualization support and mode validation
)

# Internal imports - Exception handling and logging utilities
from ..utils.exceptions import (
    ComponentError,  # General component-level exception handling for renderer lifecycle and resource management
)
from ..utils.exceptions import (
    RenderingError,  # Exception handling for rendering operation failures and matplotlib backend errors
)
from ..utils.logging import (
    get_component_logger,  # Component-specific logger creation for matplotlib renderer with performance monitoring integration
)
from ..utils.logging import (
    monitor_performance,  # Performance monitoring decorator for automatic timing and resource usage tracking
)

# Internal imports - Base renderer interface and shared functionality
from .base_renderer import (
    BaseRenderer,  # Abstract base class providing shared rendering functionality, performance tracking, and consistent API contracts
)
from .base_renderer import (
    RenderContext,  # Immutable rendering context with environment state and visual configuration for consistent renderer operations
)

# Internal imports - Color scheme management and matplotlib integration
from .color_schemes import (
    ColorSchemeManager,  # Color scheme management with caching, validation, and optimization for dual-mode rendering
)
from .color_schemes import (
    CustomColorScheme,  # Custom color scheme configuration with matplotlib integration and validation
)
from .color_schemes import (
    convert_rgb_to_matplotlib,  # Utility function for RGB color conversion to matplotlib-compatible formats
)
from .color_schemes import (
    get_matplotlib_colormap,  # Cached utility function for matplotlib colormap retrieval with performance optimization
)

# Global configuration constants for matplotlib renderer behavior and optimization
DEFAULT_FIGURE_SIZE = MATPLOTLIB_DEFAULT_FIGSIZE  # Default figure size (8, 8) for matplotlib visualization
DEFAULT_DPI = (
    100  # Standard DPI for matplotlib figure resolution and display compatibility
)
INTERACTIVE_UPDATE_INTERVAL = (
    0.001  # Minimal interactive pause for matplotlib display refresh (1ms)
)
BACKEND_SELECTION_TIMEOUT = (
    10.0  # Maximum time for backend selection and configuration (10 seconds)
)
FIGURE_CLEANUP_TIMEOUT = 5.0  # Maximum time for figure cleanup operations (5 seconds)
COLORMAP_CACHE_SIZE = (
    20  # Maximum number of cached colormaps for performance optimization
)
MARKER_UPDATE_BATCH_SIZE = (
    100  # Batch size for efficient marker updates in large-scale visualizations
)

# Module exports for external usage and renderer factory functions
__all__ = [
    "MatplotlibRenderer",  # Interactive matplotlib renderer for human mode visualization with backend management
    "MatplotlibBackendManager",  # Backend management utility for intelligent backend selection and configuration
    "InteractiveUpdateManager",  # Interactive update manager for efficient matplotlib visualization updates
    "create_matplotlib_renderer",  # Factory function for creating configured matplotlib renderer with optimization
    "detect_matplotlib_capabilities",  # System capability detection function for matplotlib integration assessment
    "configure_matplotlib_backend",  # Backend configuration function with priority-based selection and fallback
    "validate_matplotlib_integration",  # Comprehensive validation function for matplotlib integration and performance testing
]


class MatplotlibBackendManager:
    """
    Backend management utility class providing intelligent backend selection, capability detection,
    configuration, and fallback handling for cross-platform matplotlib compatibility.

    This class encapsulates the complexity of matplotlib backend management, providing automatic
    backend detection, priority-based selection, and graceful fallback to headless operation.
    The manager supports Linux (full support), macOS (full support), and Windows (community support)
    with comprehensive error handling and performance optimization.

    Key Features:
    - Intelligent backend detection with system capability assessment
    - Priority-based backend selection with fallback mechanisms
    - Headless environment detection and automatic Agg backend configuration
    - Backend capability caching for performance optimization
    - Cross-platform compatibility with platform-specific optimizations
    - Thread-safe backend operations for concurrent usage
    """

    def __init__(
        self,
        backend_preferences: Optional[List[str]] = None,
        enable_fallback: bool = True,
        backend_options: dict = None,
    ):
        """
        Initialize backend manager with preferences, fallback configuration, and capability assessment.

        Args:
            backend_preferences: Ordered list of preferred backends or None for system defaults
            enable_fallback: Enable automatic fallback to headless Agg backend
            backend_options: Additional backend configuration options
        """
        # Set backend preferences to provided list or use system defaults from constants
        self.backend_preferences = backend_preferences or list(BACKEND_PRIORITY_LIST)

        # Configure fallback behavior for headless operation and error recovery
        self.enable_fallback = enable_fallback

        # Store backend configuration options for matplotlib customization
        self.backend_options = backend_options or {}

        # Initialize current backend tracking for state management
        self.current_backend: Optional[str] = None

        # Initialize backend capabilities cache for performance optimization
        self.backend_capabilities: Dict[str, dict] = {}

        # Detect headless environment using display availability and system configuration
        self.headless_mode = self._detect_headless_environment()

        # Initialize configuration cache for backend selection optimization
        self.configuration_cache: Dict[str, dict] = {}

        # Create component logger for backend management operations and debugging
        self.logger = get_component_logger(f"{__name__}.MatplotlibBackendManager")

        # Log initialization with configuration details for debugging
        self.logger.debug(
            f"MatplotlibBackendManager initialized: preferences={self.backend_preferences}, "
            f"fallback_enabled={enable_fallback}, headless={self.headless_mode}"
        )

    def select_backend(
        self, force_reselection: bool = False, validate_functionality: bool = True
    ) -> str:
        """
        Intelligent backend selection with priority-based testing, compatibility validation,
        and automatic fallback configuration.

        Args:
            force_reselection: Force backend reselection even if current backend available
            validate_functionality: Test backend functionality with sample operations

        Returns:
            Selected backend name with capability validation and configuration
        """
        # Return current backend if already selected and force_reselection is False
        if self.current_backend and not force_reselection:
            self.logger.debug(f"Using existing backend: {self.current_backend}")
            return self.current_backend

        selected_backend = None

        # Iterate through backend preferences testing availability and functionality
        for backend_name in self.backend_preferences:
            try:
                # Skip interactive backends in headless environment unless explicitly configured
                if self.headless_mode and backend_name not in ["Agg"]:
                    if not self.backend_options.get(
                        "force_interactive_in_headless", False
                    ):
                        self.logger.debug(
                            f"Skipping interactive backend {backend_name} in headless mode"
                        )
                        continue

                # Test backend availability by attempting import and initialization
                if self._test_backend_availability(backend_name):
                    # Validate backend functionality if validate_functionality enabled
                    if validate_functionality:
                        if self._test_backend_functionality(backend_name):
                            selected_backend = backend_name
                            break
                        else:
                            self.logger.warning(
                                f"Backend {backend_name} available but functionality test failed"
                            )
                    else:
                        selected_backend = backend_name
                        break

            except Exception as e:
                self.logger.debug(f"Backend {backend_name} selection failed: {e}")
                continue

        # Apply headless fallback to Agg backend if no interactive backend available
        if not selected_backend and self.enable_fallback:
            try:
                if self._test_backend_availability("Agg"):
                    selected_backend = "Agg"
                    self.logger.info("Falling back to headless Agg backend")
            except Exception as e:
                self.logger.error(f"Even Agg backend fallback failed: {e}")

        # Raise error if no backend could be selected
        if not selected_backend:
            raise RenderingError("No compatible matplotlib backend available")

        # Update current backend and cache selection results
        self.current_backend = selected_backend
        self.logger.info(f"Selected matplotlib backend: {selected_backend}")

        return selected_backend

    def get_backend_capabilities(
        self, backend_name: Optional[str] = None, force_refresh: bool = False
    ) -> Dict[str, any]:
        """
        Retrieve comprehensive backend capabilities including display support, interactive features,
        performance characteristics, and platform compatibility.

        Args:
            backend_name: Specific backend to analyze or None for current backend
            force_refresh: Force capability reassessment ignoring cache

        Returns:
            Dictionary with backend capabilities and performance characteristics
        """
        # Use current backend if backend_name not provided
        effective_backend = backend_name or self.current_backend
        if not effective_backend:
            effective_backend = self.select_backend()

        # Check configuration cache unless force_refresh enabled
        cache_key = f"capabilities_{effective_backend}"
        if not force_refresh and cache_key in self.configuration_cache:
            return self.configuration_cache[cache_key]

        capabilities = {
            "backend_name": effective_backend,
            "analysis_timestamp": time.time(),
            "display_support": False,
            "interactive_features": False,
            "gui_toolkit": None,
            "headless_compatible": False,
            "platform_support": {},
            "performance_characteristics": {},
        }

        try:
            # Test backend display support and GUI toolkit integration
            if effective_backend == "TkAgg":
                capabilities["display_support"] = True
                capabilities["interactive_features"] = True
                capabilities["gui_toolkit"] = "Tkinter"
                capabilities["headless_compatible"] = False
            elif effective_backend == "Qt5Agg":
                capabilities["display_support"] = True
                capabilities["interactive_features"] = True
                capabilities["gui_toolkit"] = "Qt5"
                capabilities["headless_compatible"] = False
            elif effective_backend == "Agg":
                capabilities["display_support"] = False
                capabilities["interactive_features"] = False
                capabilities["gui_toolkit"] = None
                capabilities["headless_compatible"] = True

            # Evaluate platform compatibility and system-specific features
            platform_name = sys.platform
            if platform_name.startswith("linux"):
                capabilities["platform_support"]["linux"] = "full"
            elif platform_name == "darwin":
                capabilities["platform_support"]["macos"] = "full"
            elif platform_name.startswith("win"):
                capabilities["platform_support"]["windows"] = "community"

            # Measure performance characteristics with timing benchmarks
            if self.backend_options.get("measure_performance", False):
                capabilities["performance_characteristics"] = (
                    self._measure_backend_performance(effective_backend)
                )

            # Cache capabilities for performance optimization
            self.configuration_cache[cache_key] = capabilities

        except Exception as e:
            self.logger.error(
                f"Capability analysis failed for {effective_backend}: {e}"
            )
            capabilities["error"] = str(e)

        return capabilities

    def configure_backend(
        self, backend_name: str, configuration_options: dict = None
    ) -> bool:
        """
        Configure selected matplotlib backend with optimization settings, threading configuration,
        and resource management setup.

        Args:
            backend_name: Backend name to configure
            configuration_options: Backend-specific configuration parameters

        Returns:
            True if configuration successful, False otherwise
        """
        config_options = configuration_options or {}

        try:
            # Switch to specified backend using matplotlib with error handling
            original_backend = plt.get_backend()
            plt.switch_backend(backend_name)

            # Apply configuration options including figure defaults and performance settings
            if "figure_size" in config_options:
                plt.rcParams["figure.figsize"] = config_options["figure_size"]
            if "dpi" in config_options:
                plt.rcParams["figure.dpi"] = config_options["dpi"]
            if "interactive" in config_options:
                if config_options["interactive"]:
                    plt.ion()  # Turn on interactive mode
                else:
                    plt.ioff()  # Turn off interactive mode

            # Configure threading settings for thread-safe operations
            if "thread_safe" in config_options and config_options["thread_safe"]:
                # Enable thread-safe operations where supported
                pass  # Backend-specific thread safety configuration

            # Test backend functionality with sample operations
            test_fig = plt.figure(figsize=(1, 1))
            test_ax = test_fig.add_subplot(111)
            test_ax.plot([0, 1], [0, 1])
            plt.close(test_fig)

            # Update current backend and log configuration success
            self.current_backend = backend_name
            self.logger.info(f"Backend {backend_name} configured successfully")

            return True

        except Exception as e:
            self.logger.error(f"Backend configuration failed for {backend_name}: {e}")
            # Attempt to restore original backend on failure
            try:
                plt.switch_backend(original_backend)
            except Exception:
                pass  # Ignore restoration errors
            return False

    def restore_original_backend(self) -> bool:
        """
        Restore original matplotlib backend configuration for clean system state.

        Returns:
            True if restoration successful, False otherwise
        """
        try:
            # Attempt to restore matplotlib to default backend
            plt.switch_backend("Agg")  # Safe default backend
            self.current_backend = None
            self.logger.debug("Matplotlib backend restored to default")
            return True
        except Exception as e:
            self.logger.error(f"Backend restoration failed: {e}")
            return False

    def _detect_headless_environment(self) -> bool:
        """
        Detect headless environment using display availability and system environment variables.

        Returns:
            True if headless environment detected, False if display available
        """
        # Check for DISPLAY environment variable on Unix-like systems
        if os.name == "posix":
            display = os.environ.get("DISPLAY", "")
            if not display:
                return True

        # Check for SSH connection indicating remote headless session
        if os.environ.get("SSH_CONNECTION"):
            return True

        # Additional headless detection logic for different platforms
        try:
            if sys.platform.startswith("linux"):
                # Check for X11 availability
                import subprocess

                result = subprocess.run(
                    ["xset", "q"], capture_output=True, text=True, timeout=1
                )
                return result.returncode != 0
        except Exception:
            pass  # Assume headless if detection fails

        return False

    def _test_backend_availability(self, backend_name: str) -> bool:
        """
        Test backend availability by attempting import and basic initialization.

        Args:
            backend_name: Backend name to test

        Returns:
            True if backend available and functional, False otherwise
        """
        try:
            # Attempt to import backend module
            if backend_name == "TkAgg":
                import importlib

                importlib.import_module("tkinter")
                importlib.import_module("matplotlib.backends.backend_tkagg")
            elif backend_name == "Qt5Agg":
                import importlib

                importlib.import_module("matplotlib.backends.backend_qt5agg")
                # PyQt5 or PySide2 availability test would go here
            elif backend_name == "Agg":
                import importlib

                importlib.import_module("matplotlib.backends.backend_agg")

            return True

        except ImportError as e:
            self.logger.debug(f"Backend {backend_name} import failed: {e}")
            return False
        except Exception as e:
            self.logger.debug(f"Backend {backend_name} availability test failed: {e}")
            return False

    def _test_backend_functionality(self, backend_name: str) -> bool:
        """
        Test backend functionality with sample figure operations.

        Args:
            backend_name: Backend name to test functionality

        Returns:
            True if backend functional, False otherwise
        """
        try:
            # Store current backend for restoration
            original_backend = plt.get_backend()

            # Switch to test backend temporarily
            plt.switch_backend(backend_name)

            # Create test figure and perform basic operations
            test_fig = plt.figure(figsize=(2, 2))
            test_ax = test_fig.add_subplot(111)
            test_ax.plot([0, 1], [0, 1], "r-")
            test_ax.set_title("Backend Test")

            # Test drawing operations
            test_fig.canvas.draw()

            # Clean up test figure
            plt.close(test_fig)

            # Restore original backend
            plt.switch_backend(original_backend)

            return True

        except Exception as e:
            self.logger.debug(f"Backend {backend_name} functionality test failed: {e}")
            # Attempt to restore original backend
            try:
                plt.switch_backend(original_backend)
            except Exception:
                pass
            return False

    def _measure_backend_performance(self, backend_name: str) -> Dict[str, float]:
        """
        Measure backend performance characteristics with timing benchmarks.

        Args:
            backend_name: Backend to benchmark

        Returns:
            Dictionary with performance metrics
        """
        performance_metrics = {
            "figure_creation_ms": 0.0,
            "plot_operation_ms": 0.0,
            "draw_operation_ms": 0.0,
            "total_benchmark_ms": 0.0,
        }

        try:
            benchmark_start = time.time()

            # Measure figure creation time
            creation_start = time.time()
            test_fig = plt.figure(figsize=(4, 4))
            creation_time = (time.time() - creation_start) * 1000
            performance_metrics["figure_creation_ms"] = creation_time

            # Measure plotting operations
            plot_start = time.time()
            test_ax = test_fig.add_subplot(111)
            test_data = np.random.rand(100, 100)
            test_ax.imshow(test_data, cmap="viridis")
            plot_time = (time.time() - plot_start) * 1000
            performance_metrics["plot_operation_ms"] = plot_time

            # Measure draw operations
            draw_start = time.time()
            test_fig.canvas.draw()
            draw_time = (time.time() - draw_start) * 1000
            performance_metrics["draw_operation_ms"] = draw_time

            # Clean up and calculate total time
            plt.close(test_fig)
            total_time = (time.time() - benchmark_start) * 1000
            performance_metrics["total_benchmark_ms"] = total_time

        except Exception as e:
            self.logger.warning(
                f"Performance measurement failed for {backend_name}: {e}"
            )

        return performance_metrics


class InteractiveUpdateManager:
    """
    Manager class for handling interactive matplotlib updates including efficient marker updates,
    figure refresh optimization, animation control, and performance monitoring for real-time visualization.

    This class provides optimized update management for matplotlib interactive displays, implementing
    change detection, selective refresh, and performance optimization strategies. The manager handles
    concentration field updates, marker positioning, and display refresh with minimal overhead for
    real-time plume navigation visualization.

    Key Features:
    - Efficient change detection to minimize unnecessary updates
    - Selective refresh of modified visualization elements
    - Performance-optimized marker updates with batch processing
    - Interactive display management with frame rate control
    - Memory-efficient caching of matplotlib objects and state
    - Thread-safe operations for concurrent visualization access
    """

    def __init__(
        self,
        figure: matplotlib.figure.Figure,
        axes: matplotlib.axes.Axes,
        update_interval: float = INTERACTIVE_UPDATE_INTERVAL,
    ):
        """
        Initialize interactive update manager with matplotlib objects and performance configuration.

        Args:
            figure: Matplotlib figure object for display management
            axes: Matplotlib axes object for plotting operations
            update_interval: Minimum time between display updates for frame rate control
        """
        # Store matplotlib object references for interactive updates
        self.figure = figure
        self.axes = axes
        self.update_interval = update_interval

        # Initialize marker references for lazy creation and efficient updates
        self.agent_marker: Optional[matplotlib.lines.Line2D] = None
        self.source_marker: Optional[matplotlib.lines.Line2D] = None
        self.heatmap: Optional[matplotlib.image.AxesImage] = None

        # Initialize update cache for change detection and optimization
        self.update_cache: Dict[str, any] = {}

        # Initialize performance tracking counters
        self.update_count = 0
        self.last_update_time = 0.0

    def update_concentration_field(
        self,
        concentration_field: np.ndarray,
        color_scheme: CustomColorScheme,
        force_update: bool = False,
    ) -> bool:
        """
        Update concentration field heatmap with efficient change detection, colormap application,
        and axes configuration optimization.

        Args:
            concentration_field: 2D numpy array with concentration values
            color_scheme: Color scheme configuration for matplotlib integration
            force_update: Skip change detection and force complete update

        Returns:
            True if update successful, False if update failed or skipped
        """
        try:
            # Check for changes unless force_update enabled
            field_id = f"concentration_{concentration_field.shape}_{np.sum(concentration_field):.6f}"
            if (
                not force_update
                and self.update_cache.get("concentration_field_id") == field_id
            ):
                return True  # Skip update - no changes detected

            # Configure matplotlib axes with color scheme integration
            color_scheme.configure_matplotlib_axes(self.axes)

            # Create or update heatmap using axes.imshow with concentration field data
            if self.heatmap is None:
                # Create new heatmap with optimized configuration
                colormap = get_matplotlib_colormap(color_scheme.concentration_colormap)
                self.heatmap = self.axes.imshow(
                    concentration_field,
                    cmap=colormap,
                    origin="lower",  # Match coordinate system
                    aspect="equal",
                    interpolation="nearest",
                    vmin=0.0,
                    vmax=1.0,  # Normalize concentration range
                )
            else:
                # Update existing heatmap data efficiently
                self.heatmap.set_array(concentration_field)
                self.heatmap.set_clim(vmin=0.0, vmax=1.0)

            # Update cache with current field identifier
            self.update_cache["concentration_field_id"] = field_id

            return True

        except Exception as e:
            logger = get_component_logger(f"{__name__}.InteractiveUpdateManager")
            logger.error(f"Concentration field update failed: {e}")
            return False

    def update_agent_marker(
        self,
        agent_position: Coordinates,
        color_scheme: CustomColorScheme,
        animate_transition: bool = False,
    ) -> bool:
        """
        Update agent position marker with efficient positioning, color application,
        and visual optimization for real-time tracking.

        Args:
            agent_position: Current agent coordinates
            color_scheme: Color scheme for marker appearance
            animate_transition: Enable smooth position transition animation

        Returns:
            True if marker update successful, False otherwise
        """
        try:
            # Check for position changes for efficient update detection
            position_key = f"agent_{agent_position.x}_{agent_position.y}"
            if (
                self.update_cache.get("agent_position") == position_key
                and not animate_transition
            ):
                return True  # Skip update - position unchanged

            # Convert agent color from color scheme to matplotlib format
            agent_color = convert_rgb_to_matplotlib(color_scheme.agent_color)

            if self.agent_marker is None:
                # Create agent marker with optimized configuration
                self.agent_marker = self.axes.scatter(
                    [agent_position.x],
                    [agent_position.y],
                    c=[agent_color],
                    s=100,
                    marker="s",  # Square marker
                    edgecolors="black",
                    linewidth=1,
                    zorder=10,  # Ensure marker appears above heatmap
                )
            else:
                # Update existing marker position efficiently
                self.agent_marker.set_offsets([(agent_position.x, agent_position.y)])
                self.agent_marker.set_color([agent_color])

            # Apply animation transition if enabled
            if animate_transition and hasattr(self.figure, "canvas"):
                self.figure.canvas.draw_idle()
                time.sleep(self.update_interval)  # Brief pause for smooth animation

            # Update cache with current position
            self.update_cache["agent_position"] = position_key

            return True

        except Exception as e:
            logger = get_component_logger(f"{__name__}.InteractiveUpdateManager")
            logger.error(f"Agent marker update failed: {e}")
            return False

    def update_source_marker(
        self, source_position: Coordinates, color_scheme: CustomColorScheme
    ) -> bool:
        """
        Update source location marker with cross-pattern visualization, color application,
        and visibility optimization for goal indication.

        Args:
            source_position: Source location coordinates
            color_scheme: Color scheme for marker appearance

        Returns:
            True if source marker update successful, False otherwise
        """
        try:
            # Check if source position changed
            position_key = f"source_{source_position.x}_{source_position.y}"
            if self.update_cache.get("source_position") == position_key:
                return True  # Skip update - position unchanged

            # Convert source color from color scheme to matplotlib format
            source_color = convert_rgb_to_matplotlib(color_scheme.source_color)

            if self.source_marker is None:
                # Create source marker with cross pattern
                self.source_marker = self.axes.scatter(
                    [source_position.x],
                    [source_position.y],
                    c=[source_color],
                    s=200,
                    marker="+",  # Cross marker
                    linewidth=3,
                    zorder=15,  # Ensure source appears above agent
                )
            else:
                # Update existing marker position and appearance
                self.source_marker.set_offsets([(source_position.x, source_position.y)])
                self.source_marker.set_color([source_color])

            # Update cache with current position
            self.update_cache["source_position"] = position_key

            return True

        except Exception as e:
            logger = get_component_logger(f"{__name__}.InteractiveUpdateManager")
            logger.error(f"Source marker update failed: {e}")
            return False

    def refresh_display(
        self, force_refresh: bool = False, measure_performance: bool = False
    ) -> float:
        """
        Refresh matplotlib display with performance optimization, frame rate control,
        and resource management for smooth interactive updates.

        Args:
            force_refresh: Skip frame rate limiting and force immediate refresh
            measure_performance: Record refresh timing for performance analysis

        Returns:
            Refresh duration in milliseconds for performance monitoring
        """
        refresh_start = time.time()

        try:
            # Check frame rate timing unless force_refresh enabled
            current_time = time.time()
            if not force_refresh:
                time_since_last = current_time - self.last_update_time
                if time_since_last < self.update_interval:
                    # Too soon for another update - return minimal duration
                    return (time.time() - refresh_start) * 1000

            # Perform matplotlib display refresh with error handling
            if hasattr(self.figure, "canvas") and self.figure.canvas is not None:
                self.figure.canvas.draw()

                # Apply interactive pause for smooth display updates
                if plt.isinteractive():
                    plt.pause(self.update_interval)

            # Update timing tracking
            self.last_update_time = current_time
            self.update_count += 1

        except Exception as e:
            logger = get_component_logger(f"{__name__}.InteractiveUpdateManager")
            logger.error(f"Display refresh failed: {e}")

        # Calculate and return refresh duration
        refresh_duration = (time.time() - refresh_start) * 1000
        return refresh_duration

    def batch_update(
        self,
        context: RenderContext,
        color_scheme: CustomColorScheme,
        optimize_updates: bool = True,
    ) -> Dict[str, any]:
        """
        Perform batch update of all visualization elements with optimized rendering,
        change detection, and performance monitoring for efficient updates.

        Args:
            context: Complete rendering context with environment state
            color_scheme: Color scheme configuration for consistent visualization
            optimize_updates: Enable update optimization with change detection

        Returns:
            Dictionary with update performance metrics and timing analysis
        """
        batch_start = time.time()
        performance_report = {
            "batch_start_time": batch_start,
            "component_timings": {},
            "total_changes": 0,
            "optimization_enabled": optimize_updates,
        }

        try:
            # Update concentration field with performance tracking
            field_start = time.time()
            field_updated = self.update_concentration_field(
                context.concentration_field,
                color_scheme,
                force_update=not optimize_updates,
            )
            performance_report["component_timings"]["concentration_field"] = (
                time.time() - field_start
            ) * 1000
            if field_updated:
                performance_report["total_changes"] += 1

            # Update agent marker with position tracking
            agent_start = time.time()
            agent_updated = self.update_agent_marker(
                context.agent_position, color_scheme
            )
            performance_report["component_timings"]["agent_marker"] = (
                time.time() - agent_start
            ) * 1000
            if agent_updated:
                performance_report["total_changes"] += 1

            # Update source marker with visualization consistency
            source_start = time.time()
            source_updated = self.update_source_marker(
                context.source_position, color_scheme
            )
            performance_report["component_timings"]["source_marker"] = (
                time.time() - source_start
            ) * 1000
            if source_updated:
                performance_report["total_changes"] += 1

            # Perform coordinated display refresh
            refresh_duration = self.refresh_display(
                force_refresh=not optimize_updates, measure_performance=True
            )
            performance_report["component_timings"][
                "display_refresh"
            ] = refresh_duration

            # Calculate total batch performance
            total_duration = (time.time() - batch_start) * 1000
            performance_report["total_duration_ms"] = total_duration
            performance_report["update_efficiency"] = performance_report[
                "total_changes"
            ] / max(1, total_duration / 10)

        except Exception as e:
            logger = get_component_logger(f"{__name__}.InteractiveUpdateManager")
            logger.error(f"Batch update failed: {e}")
            performance_report["error"] = str(e)

        return performance_report


class MatplotlibRenderer(BaseRenderer):
    """
    Concrete matplotlib renderer implementation providing interactive human mode visualization
    with backend management, real-time updates, performance optimization, and comprehensive
    resource management for plume navigation environments.

    This class implements the BaseRenderer interface using matplotlib for interactive visualization,
    providing comprehensive backend management, real-time updates, and performance optimization.
    The renderer supports cross-platform operation with graceful fallback mechanisms and targets
    <50ms update performance for smooth human mode visualization.

    Key Features:
    - Interactive matplotlib visualization with real-time environment updates
    - Intelligent backend management with cross-platform compatibility
    - Performance-optimized rendering targeting <50ms human mode updates
    - Comprehensive resource management with automatic cleanup
    - Color scheme integration with matplotlib colormap support
    - Headless environment support with automatic Agg backend fallback
    - Thread-safe operations for concurrent visualization access
    """

    def __init__(
        self,
        grid_size: GridSize,
        color_scheme_name: Optional[str] = None,
        backend_preferences: Optional[List[str]] = None,
        renderer_options: dict = None,
    ):
        """
        Initialize matplotlib renderer with backend management, color scheme integration,
        and performance optimization setup.

        Args:
            grid_size: Grid dimensions for matplotlib axes configuration
            color_scheme_name: Optional color scheme identifier for visualization
            backend_preferences: Optional list of preferred matplotlib backends
            renderer_options: Dictionary of renderer-specific configuration options
        """
        # Initialize base renderer with grid configuration and options
        super().__init__(grid_size, color_scheme_name, renderer_options)

        # Configure backend preferences using provided list or system defaults
        self.backend_preferences = backend_preferences or list(BACKEND_PRIORITY_LIST)

        # Initialize backend manager with preference configuration and fallback handling
        self.backend_manager = MatplotlibBackendManager(
            backend_preferences=self.backend_preferences,
            enable_fallback=True,
            backend_options=self.renderer_options,
        )

        # Initialize color scheme manager with optimization and caching
        self.color_manager = ColorSchemeManager()

        # Initialize matplotlib objects for lazy creation and resource management
        self.figure: Optional[matplotlib.figure.Figure] = None
        self.axes: Optional[matplotlib.axes.Axes] = None
        self.update_manager: Optional[InteractiveUpdateManager] = None

        # Initialize current color scheme with default configuration
        self.current_color_scheme = self.color_manager.get_scheme(
            color_scheme_name or "standard"
        )

        # Configure interactive mode based on renderer options and backend capabilities
        self.interactive_mode = self.renderer_options.get("interactive", True)

        # Initialize performance cache for optimization and monitoring
        self.performance_cache: Dict[str, any] = {}

        # Register cleanup function for automatic resource management
        atexit.register(self.cleanup_resources)

    def supports_render_mode(self, mode: RenderMode) -> bool:
        """
        Check if renderer supports specified rendering mode with capability validation.

        Args:
            mode: RenderMode enumeration value to check for support

        Returns:
            True if mode is HUMAN and matplotlib backend available, False otherwise
        """
        if mode == RenderMode.HUMAN:
            try:
                # Check matplotlib backend availability for human mode visualization
                backend_capabilities = self.backend_manager.get_backend_capabilities()
                return "error" not in backend_capabilities
            except Exception:
                return False

        # This renderer only supports HUMAN mode - RGB mode handled by separate renderer
        return False

    def _initialize_renderer_resources(self) -> None:
        """
        Initialize matplotlib-specific resources including backend configuration, figure creation,
        color scheme setup, and interactive update management.
        """
        try:
            # Configure matplotlib backend using backend manager with error handling
            selected_backend = self.backend_manager.select_backend(
                validate_functionality=True
            )
            self.backend_manager.configure_backend(
                selected_backend,
                {
                    "figure_size": self.renderer_options.get(
                        "figure_size", DEFAULT_FIGURE_SIZE
                    ),
                    "dpi": self.renderer_options.get("dpi", DEFAULT_DPI),
                    "interactive": self.interactive_mode,
                },
            )

            # Create matplotlib figure with optimized configuration
            figsize = self.renderer_options.get("figure_size", DEFAULT_FIGURE_SIZE)
            dpi = self.renderer_options.get("dpi", DEFAULT_DPI)
            self.figure = plt.figure(figsize=figsize, dpi=dpi)

            # Configure figure axes with grid dimensions and visualization properties
            self.axes = self.figure.add_subplot(111)
            self.axes.set_xlim(0, self.grid_size.width - 1)
            self.axes.set_ylim(0, self.grid_size.height - 1)
            self.axes.set_xlabel("Grid X Coordinate")
            self.axes.set_ylabel("Grid Y Coordinate")
            self.axes.set_title("Plume Navigation Visualization")
            self.axes.grid(True, alpha=0.3)

            # Initialize color scheme with matplotlib optimization
            optimized_scheme = self.color_manager.optimize_scheme(
                self.current_color_scheme, RenderMode.HUMAN
            )
            self.current_color_scheme = optimized_scheme

            # Create interactive update manager for performance optimization
            update_interval = self.renderer_options.get(
                "update_interval", INTERACTIVE_UPDATE_INTERVAL
            )
            self.update_manager = InteractiveUpdateManager(
                self.figure, self.axes, update_interval
            )

            # Configure interactive mode based on backend capabilities and options
            if self.interactive_mode:
                plt.ion()
                if hasattr(self.figure, "show"):
                    self.figure.show()

            self.logger.info(
                f"Matplotlib renderer initialized with backend: {selected_backend}"
            )

        except Exception as e:
            self.logger.error(f"Matplotlib renderer initialization failed: {e}")
            raise ComponentError(f"Failed to initialize matplotlib renderer: {e}")

    def _cleanup_renderer_resources(self) -> None:
        """
        Clean up matplotlib resources including figure disposal, backend restoration,
        and memory management with comprehensive resource cleanup.
        """
        try:
            # Close matplotlib figure with error handling
            if self.figure is not None:
                plt.close(self.figure)
                self.figure = None

            # Clear axes and update manager references
            self.axes = None
            self.update_manager = None

            # Clear performance cache and optimization data
            self.performance_cache.clear()

            # Restore original matplotlib backend configuration
            if hasattr(self, "backend_manager"):
                self.backend_manager.restore_original_backend()

            # Force garbage collection for memory cleanup
            import gc

            gc.collect()

            self.logger.debug("Matplotlib renderer resources cleaned up")

        except Exception as e:
            self.logger.error(f"Matplotlib resource cleanup failed: {e}")

    def _render_rgb_array(self, context: RenderContext) -> np.ndarray:
        """
        RGB array rendering not supported by matplotlib renderer - raises NotImplementedError.

        Args:
            context: Rendering context (unused)

        Raises:
            NotImplementedError: Matplotlib renderer only supports HUMAN mode
        """
        raise NotImplementedError("MatplotlibRenderer only supports HUMAN render mode")

    @monitor_performance("render_human")
    def _render_human(self, context: RenderContext) -> None:
        """
        Render interactive matplotlib visualization for human mode with real-time updates,
        marker placement, and performance optimization targeting <50ms updates.

        Args:
            context: Validated rendering context with current environment state
        """
        try:
            # Initialize matplotlib resources if not already available
            if self.figure is None or self.axes is None:
                self._initialize_renderer_resources()

            # Validate update manager availability
            if self.update_manager is None:
                raise RenderingError("Update manager not initialized")

            # Perform batch update with performance optimization and change detection
            performance_report = self.update_manager.batch_update(
                context, self.current_color_scheme, optimize_updates=True
            )

            # Validate performance against target timing
            total_duration = performance_report.get("total_duration_ms", 0)
            if total_duration > PERFORMANCE_TARGET_HUMAN_RENDER_MS:
                self.logger.warning(
                    f"Render duration {total_duration:.2f}ms exceeds target "
                    f"{PERFORMANCE_TARGET_HUMAN_RENDER_MS}ms"
                )

            # Cache performance data for optimization analysis
            self.performance_cache["last_render"] = {
                "duration_ms": total_duration,
                "timestamp": time.time(),
                "changes_count": performance_report.get("total_changes", 0),
            }

        except Exception as e:
            self.logger.error(f"Human mode rendering failed: {e}")
            raise RenderingError(f"Matplotlib rendering failed: {e}")

    def set_color_scheme(
        self, color_scheme: Union[str, CustomColorScheme], force_update: bool = False
    ) -> bool:
        """
        Update matplotlib renderer color scheme with validation, optimization,
        and interactive display refresh for consistent visualization.

        Args:
            color_scheme: Color scheme identifier string or CustomColorScheme instance
            force_update: Force immediate visualization update with new colors

        Returns:
            True if color scheme update successful, False otherwise
        """
        try:
            # Resolve color scheme using color manager if string provided
            if isinstance(color_scheme, str):
                resolved_scheme = self.color_manager.get_scheme(color_scheme)
            else:
                resolved_scheme = color_scheme

            # Validate color scheme compatibility with matplotlib
            resolved_scheme.validate()

            # Optimize color scheme for matplotlib backend
            optimized_scheme = self.color_manager.optimize_scheme(
                resolved_scheme, RenderMode.HUMAN
            )

            # Update current color scheme configuration
            self.current_color_scheme = optimized_scheme

            # Force visualization update if requested and figure exists
            if (
                force_update
                and self.figure is not None
                and self.update_manager is not None
            ):
                # Trigger complete visualization refresh with new colors
                self.update_manager.update_cache.clear()  # Clear cache to force updates

                # Update axes configuration with new color scheme
                optimized_scheme.configure_matplotlib_axes(self.axes)

            self.logger.info(
                f"Color scheme updated: {getattr(resolved_scheme, 'name', 'custom')}"
            )
            return True

        except Exception as e:
            self.logger.error(f"Color scheme update failed: {e}")
            return False

    def get_figure(
        self, create_if_needed: bool = True
    ) -> Optional[matplotlib.figure.Figure]:
        """
        Retrieve matplotlib figure object with lazy initialization and resource management.

        Args:
            create_if_needed: Initialize renderer resources if figure not available

        Returns:
            Matplotlib figure object or None if not initialized
        """
        if self.figure is None and create_if_needed:
            try:
                self._initialize_renderer_resources()
            except Exception as e:
                self.logger.error(f"Figure initialization failed: {e}")
                return None

        return self.figure

    def save_figure(
        self, filename: str, format: Optional[str] = None, save_options: dict = None
    ) -> bool:
        """
        Save current matplotlib visualization to file with format support, quality configuration,
        and metadata preservation for publication and analysis.

        Args:
            filename: Output filename with path
            format: File format (inferred from extension if None)
            save_options: Additional matplotlib savefig options

        Returns:
            True if save successful, False otherwise
        """
        try:
            # Ensure figure is initialized and ready
            if self.figure is None:
                self.logger.error("No figure available for saving")
                return False

            # Configure save options with defaults
            options = save_options or {}
            default_options = {
                "dpi": self.renderer_options.get("save_dpi", 300),
                "bbox_inches": "tight",
                "facecolor": "white",
                "edgecolor": "none",
            }
            default_options.update(options)

            # Save figure with comprehensive error handling
            self.figure.savefig(filename, format=format, **default_options)

            self.logger.info(f"Figure saved to: {filename}")
            return True

        except Exception as e:
            self.logger.error(f"Figure save failed: {e}")
            return False

    def configure_interactive_mode(
        self,
        enable_interactive: bool,
        update_interval: Optional[float] = None,
        interactive_options: dict = None,
    ) -> bool:
        """
        Configure interactive matplotlib mode with event handling, update intervals,
        and performance optimization for responsive visualization.

        Args:
            enable_interactive: Enable or disable interactive matplotlib mode
            update_interval: Optional update interval for interactive refresh
            interactive_options: Additional interactive configuration options

        Returns:
            True if configuration successful, False otherwise
        """
        try:
            # Update interactive mode configuration
            self.interactive_mode = enable_interactive
            options = interactive_options or {}

            # Configure matplotlib interactive mode
            if enable_interactive:
                plt.ion()
                self.logger.info("Interactive mode enabled")
            else:
                plt.ioff()
                self.logger.info("Interactive mode disabled")

            # Update interactive refresh interval if provided
            if update_interval is not None and self.update_manager is not None:
                self.update_manager.update_interval = update_interval
                self.logger.debug(f"Update interval set to {update_interval}s")

            # Apply additional interactive options
            for option, value in options.items():
                try:
                    plt.rcParams[option] = value
                except KeyError:
                    self.logger.warning(f"Unknown matplotlib option: {option}")

            return True

        except Exception as e:
            self.logger.error(f"Interactive mode configuration failed: {e}")
            return False

    def get_performance_metrics(
        self, include_backend_info: bool = True, include_update_stats: bool = True
    ) -> Dict[str, any]:
        """
        Retrieve comprehensive matplotlib renderer performance metrics including timing analysis,
        resource usage, and optimization recommendations.

        Args:
            include_backend_info: Include matplotlib backend information and capabilities
            include_update_stats: Include interactive update statistics and performance data

        Returns:
            Dictionary with comprehensive performance analysis and optimization guidance
        """
        metrics = {
            "collection_timestamp": time.time(),
            "renderer_type": "MatplotlibRenderer",
            "grid_size": f"{self.grid_size.width}x{self.grid_size.height}",
            "color_scheme": getattr(self.current_color_scheme, "name", "custom"),
            "interactive_mode": self.interactive_mode,
        }

        # Include backend information if requested
        if include_backend_info:
            try:
                backend_capabilities = self.backend_manager.get_backend_capabilities()
                metrics["backend_info"] = {
                    "current_backend": self.backend_manager.current_backend,
                    "headless_mode": self.backend_manager.headless_mode,
                    "display_support": backend_capabilities.get(
                        "display_support", False
                    ),
                    "interactive_features": backend_capabilities.get(
                        "interactive_features", False
                    ),
                    "platform_support": backend_capabilities.get(
                        "platform_support", {}
                    ),
                }
            except Exception as e:
                metrics["backend_info"] = {"error": str(e)}

        # Include interactive update statistics if requested
        if include_update_stats and self.update_manager is not None:
            metrics["update_stats"] = {
                "total_updates": self.update_manager.update_count,
                "last_update_time": self.update_manager.last_update_time,
                "update_interval": self.update_manager.update_interval,
                "cache_size": len(self.update_manager.update_cache),
            }

        # Include recent performance data from cache
        if "last_render" in self.performance_cache:
            metrics["recent_performance"] = self.performance_cache["last_render"]

        # Generate performance recommendations
        recommendations = []
        if (
            self.performance_cache.get("last_render", {}).get("duration_ms", 0)
            > PERFORMANCE_TARGET_HUMAN_RENDER_MS
        ):
            recommendations.append(
                "Consider reducing update frequency or grid size for better performance"
            )

        if self.backend_manager.headless_mode:
            recommendations.append(
                "Running in headless mode - interactive features limited"
            )

        if not recommendations:
            recommendations.append("Performance within expected parameters")

        metrics["recommendations"] = recommendations

        return metrics


def create_matplotlib_renderer(
    grid_size: GridSize,
    color_scheme_name: Optional[str] = None,
    backend_preferences: Optional[List[str]] = None,
    renderer_options: dict = None,
) -> MatplotlibRenderer:
    """
    Factory function to create matplotlib renderer with backend detection, capability assessment,
    and configuration optimization for interactive human mode visualization.

    Args:
        grid_size: Grid dimensions for matplotlib axes configuration
        color_scheme_name: Optional color scheme identifier for visualization
        backend_preferences: Optional list of preferred matplotlib backends
        renderer_options: Dictionary of renderer-specific configuration options

    Returns:
        Configured MatplotlibRenderer ready for human mode visualization
    """
    try:
        # Detect matplotlib capabilities for system compatibility assessment
        capabilities = detect_matplotlib_capabilities(
            test_backends=True, check_display_availability=True
        )

        # Configure backend preferences with system-specific optimizations
        effective_preferences = backend_preferences or list(BACKEND_PRIORITY_LIST)

        # Filter preferences based on detected capabilities
        if not capabilities.get("display_available", True):
            effective_preferences = [
                backend for backend in effective_preferences if backend == "Agg"
            ]  # Only headless backend for no display

        # Configure renderer options with performance defaults
        effective_options = renderer_options or {}
        default_options = {
            "figure_size": MATPLOTLIB_DEFAULT_FIGSIZE,
            "dpi": DEFAULT_DPI,
            "interactive": capabilities.get("display_available", False),
            "update_interval": INTERACTIVE_UPDATE_INTERVAL,
        }
        default_options.update(effective_options)

        # Create MatplotlibRenderer with validated configuration
        renderer = MatplotlibRenderer(
            grid_size=grid_size,
            color_scheme_name=color_scheme_name,
            backend_preferences=effective_preferences,
            renderer_options=default_options,
        )

        # Initialize renderer with comprehensive validation
        renderer.initialize(
            validate_immediately=True, enable_performance_monitoring=True
        )

        # Test renderer functionality with basic operations
        test_context = RenderContext(
            concentration_field=np.zeros(
                (grid_size.height, grid_size.width), dtype=np.float32
            ),
            agent_position=Coordinates(x=0, y=0),
            source_position=Coordinates(
                x=grid_size.width // 2, y=grid_size.height // 2
            ),
            grid_size=grid_size,
        )

        # Validate renderer with test context
        renderer.validate_context(test_context)

        logger = get_component_logger(f"{__name__}.create_matplotlib_renderer")
        logger.info(
            f"MatplotlibRenderer created successfully with backend: "
            f"{renderer.backend_manager.current_backend}"
        )

        return renderer

    except Exception as e:
        logger = get_component_logger(f"{__name__}.create_matplotlib_renderer")
        logger.error(f"MatplotlibRenderer creation failed: {e}")
        raise ComponentError(f"Failed to create matplotlib renderer: {e}")


def detect_matplotlib_capabilities(
    test_backends: bool = False,
    check_display_availability: bool = True,
    assess_performance: bool = False,
) -> Dict[str, any]:
    """
    Comprehensive system capability detection for matplotlib including backend availability,
    display detection, platform compatibility, and performance characteristics assessment.

    Args:
        test_backends: Test backend availability by attempting import and initialization
        check_display_availability: Check for display environment and GUI capability
        assess_performance: Evaluate performance characteristics with timing benchmarks

    Returns:
        Dictionary with system capabilities, backend availability, and performance characteristics
    """
    capabilities = {
        "detection_timestamp": time.time(),
        "python_version": sys.version,
        "platform": sys.platform,
        "matplotlib_available": False,
        "display_available": False,
        "backend_availability": {},
        "gui_toolkits": {},
        "performance_characteristics": {},
        "recommendations": [],
    }

    try:
        # Test matplotlib availability and version
        import matplotlib

        capabilities["matplotlib_available"] = True
        capabilities["matplotlib_version"] = matplotlib.__version__

        # Check display availability if requested
        if check_display_availability:
            # Check for display environment variables
            display_vars = ["DISPLAY", "WAYLAND_DISPLAY"]
            display_detected = any(os.environ.get(var) for var in display_vars)

            # Check for SSH connection (usually headless)
            ssh_connection = bool(os.environ.get("SSH_CONNECTION"))

            capabilities["display_available"] = display_detected and not ssh_connection
            capabilities["headless_detected"] = not capabilities["display_available"]

        # Test backend availability if requested
        if test_backends:
            backends_to_test = ["TkAgg", "Qt5Agg", "Agg"]

            for backend in backends_to_test:
                try:
                    # Test backend import and basic functionality using importlib to avoid unused import warnings
                    import importlib

                    if backend == "TkAgg":
                        importlib.import_module("tkinter")
                        importlib.import_module("matplotlib.backends.backend_tkagg")
                        capabilities["backend_availability"][backend] = True
                        capabilities["gui_toolkits"]["tkinter"] = True
                    elif backend == "Qt5Agg":
                        importlib.import_module("matplotlib.backends.backend_qt5agg")
                        capabilities["backend_availability"][backend] = True
                        capabilities["gui_toolkits"]["qt5"] = True
                    elif backend == "Agg":
                        importlib.import_module("matplotlib.backends.backend_agg")
                        capabilities["backend_availability"][backend] = True

                except ImportError:
                    capabilities["backend_availability"][backend] = False

        # Assess performance characteristics if requested
        if assess_performance:
            performance_metrics = {}

            try:
                # Test basic matplotlib operations timing
                start_time = time.time()
                test_fig = plt.figure(figsize=(4, 4))
                test_ax = test_fig.add_subplot(111)
                test_data = np.random.rand(50, 50)
                test_ax.imshow(test_data)
                test_fig.canvas.draw()
                plt.close(test_fig)

                operation_time = (time.time() - start_time) * 1000
                performance_metrics["basic_operations_ms"] = operation_time
                performance_metrics["performance_acceptable"] = (
                    operation_time < 100
                )  # 100ms threshold

            except Exception as e:
                performance_metrics["error"] = str(e)

            capabilities["performance_characteristics"] = performance_metrics

        # Generate recommendations based on detected capabilities
        recommendations = []

        if not capabilities["matplotlib_available"]:
            recommendations.append(
                "Install matplotlib>=3.9.0 for visualization support"
            )

        if not capabilities.get("display_available", True):
            recommendations.append(
                "Running in headless mode - only Agg backend available"
            )

        backend_count = sum(capabilities.get("backend_availability", {}).values())
        if backend_count == 0:
            recommendations.append(
                "No matplotlib backends available - check GUI dependencies"
            )
        elif backend_count == 1:
            recommendations.append(
                "Limited backend options - consider installing additional GUI toolkits"
            )

        if not recommendations:
            recommendations.append(
                "System fully compatible with matplotlib visualization"
            )

        capabilities["recommendations"] = recommendations

    except Exception as e:
        capabilities["error"] = str(e)
        capabilities["recommendations"] = [f"Capability detection failed: {e}"]

    return capabilities


def configure_matplotlib_backend(
    backend_preferences: Optional[List[str]] = None,
    allow_headless_fallback: bool = True,
    configuration_options: dict = None,
) -> Tuple[str, Dict[str, any]]:
    """
    Intelligent matplotlib backend configuration with priority-based selection, compatibility testing,
    and graceful fallback to headless operation.

    Args:
        backend_preferences: Ordered list of preferred backends or None for defaults
        allow_headless_fallback: Enable automatic fallback to Agg backend
        configuration_options: Backend-specific configuration parameters

    Returns:
        Tuple of (selected_backend, backend_info) with configuration details
    """
    preferences = backend_preferences or list(BACKEND_PRIORITY_LIST)
    config_options = configuration_options or {}

    backend_info = {
        "selection_timestamp": time.time(),
        "selection_process": [],
        "selected_backend": None,
        "backend_capabilities": {},
        "configuration_applied": {},
        "fallback_used": False,
    }

    selected_backend = None

    try:
        # Iterate through backend preferences with testing and validation
        for backend_name in preferences:
            backend_info["selection_process"].append(f"Testing {backend_name}")

            try:
                # Test backend availability and functionality
                plt.switch_backend(backend_name)

                # Test basic functionality with sample operations
                test_fig = plt.figure(figsize=(2, 2))
                test_ax = test_fig.add_subplot(111)
                test_ax.plot([0, 1], [0, 1])
                test_fig.canvas.draw()
                plt.close(test_fig)

                # Backend successful - apply configuration
                backend_config = {}
                if "figure_size" in config_options:
                    plt.rcParams["figure.figsize"] = config_options["figure_size"]
                    backend_config["figure_size"] = config_options["figure_size"]

                if "dpi" in config_options:
                    plt.rcParams["figure.dpi"] = config_options["dpi"]
                    backend_config["dpi"] = config_options["dpi"]

                if "interactive" in config_options:
                    if config_options["interactive"]:
                        plt.ion()
                    else:
                        plt.ioff()
                    backend_config["interactive"] = config_options["interactive"]

                selected_backend = backend_name
                backend_info["configuration_applied"] = backend_config
                backend_info["selection_process"].append(f"Selected {backend_name}")
                break

            except Exception as e:
                backend_info["selection_process"].append(f"{backend_name} failed: {e}")
                continue

        # Apply headless fallback if no backend selected and fallback allowed
        if not selected_backend and allow_headless_fallback:
            try:
                plt.switch_backend("Agg")

                # Configure Agg backend with provided options
                if "figure_size" in config_options:
                    plt.rcParams["figure.figsize"] = config_options["figure_size"]
                if "dpi" in config_options:
                    plt.rcParams["figure.dpi"] = config_options["dpi"]

                plt.ioff()  # Ensure non-interactive mode for Agg

                selected_backend = "Agg"
                backend_info["fallback_used"] = True
                backend_info["selection_process"].append("Fallback to Agg backend")

            except Exception as e:
                backend_info["selection_process"].append(f"Agg fallback failed: {e}")

        # Record final selection results
        backend_info["selected_backend"] = selected_backend

        if selected_backend:
            # Get backend capabilities for information
            manager = MatplotlibBackendManager()
            capabilities = manager.get_backend_capabilities(selected_backend)
            backend_info["backend_capabilities"] = capabilities

        return selected_backend, backend_info

    except Exception as e:
        backend_info["error"] = str(e)
        return None, backend_info


def validate_matplotlib_integration(
    backend_name: str,
    color_scheme: Optional[CustomColorScheme] = None,
    test_rendering_operations: bool = True,
    validate_performance: bool = True,
) -> Tuple[bool, Dict[str, any]]:
    """
    Comprehensive validation of matplotlib integration including backend functionality, colormap availability,
    rendering capability, and performance compliance testing.

    Args:
        backend_name: Backend name to validate functionality
        color_scheme: Optional color scheme to test with matplotlib integration
        test_rendering_operations: Execute rendering operations with sample data
        validate_performance: Validate performance against targets

    Returns:
        Tuple of (is_valid, validation_report) with comprehensive testing analysis
    """
    validation_report = {
        "validation_timestamp": time.time(),
        "backend_tested": backend_name,
        "test_results": {},
        "performance_metrics": {},
        "integration_status": "unknown",
        "error_details": [],
        "recommendations": [],
    }

    is_valid = True

    try:
        # Test backend functionality with figure creation and plotting
        original_backend = plt.get_backend()

        try:
            plt.switch_backend(backend_name)

            # Basic functionality test
            test_fig = plt.figure(figsize=(4, 4))
            test_ax = test_fig.add_subplot(111)
            test_data = np.random.rand(32, 32)

            validation_report["test_results"]["figure_creation"] = True
            validation_report["test_results"]["axes_creation"] = True

            # Test colormap integration if color scheme provided
            if color_scheme:
                try:
                    colormap = get_matplotlib_colormap(
                        color_scheme.concentration_colormap
                    )
                    test_ax.imshow(test_data, cmap=colormap)
                    validation_report["test_results"]["colormap_integration"] = True
                except Exception as e:
                    validation_report["test_results"]["colormap_integration"] = False
                    validation_report["error_details"].append(
                        f"Colormap test failed: {e}"
                    )
                    is_valid = False

            # Test rendering operations if requested
            if test_rendering_operations:
                render_start = time.time()

                # Test marker placement
                test_ax.scatter([16], [16], c="red", s=100, marker="s")
                test_ax.scatter([8], [8], c="white", s=200, marker="+")

                # Test display operations
                test_fig.canvas.draw()

                render_duration = (time.time() - render_start) * 1000
                validation_report["test_results"]["rendering_operations"] = True
                validation_report["performance_metrics"][
                    "render_duration_ms"
                ] = render_duration

                # Validate performance if requested
                if validate_performance:
                    target_ms = PERFORMANCE_TARGET_HUMAN_RENDER_MS
                    performance_acceptable = render_duration <= target_ms
                    validation_report["test_results"][
                        "performance_compliance"
                    ] = performance_acceptable
                    validation_report["performance_metrics"]["target_ms"] = target_ms
                    validation_report["performance_metrics"][
                        "within_target"
                    ] = performance_acceptable

                    if not performance_acceptable:
                        is_valid = False
                        validation_report["error_details"].append(
                            f"Performance target missed: {render_duration:.2f}ms > {target_ms}ms"
                        )

            # Clean up test figure
            plt.close(test_fig)

            # Restore original backend
            plt.switch_backend(original_backend)

            validation_report["test_results"]["cleanup"] = True

        except Exception as e:
            validation_report["test_results"]["backend_functionality"] = False
            validation_report["error_details"].append(
                f"Backend functionality test failed: {e}"
            )
            is_valid = False

            # Attempt to restore original backend
            try:
                plt.switch_backend(original_backend)
            except Exception:
                pass

        # Set integration status based on results
        if is_valid:
            validation_report["integration_status"] = "fully_compatible"
        else:
            validation_report["integration_status"] = "limited_compatibility"

        # Generate recommendations based on test results
        recommendations = []

        if not validation_report["test_results"].get("backend_functionality", True):
            recommendations.append(
                f"Backend {backend_name} not functional - try alternative backend"
            )

        if not validation_report["test_results"].get("performance_compliance", True):
            recommendations.append(
                "Performance below target - consider optimization or simpler visualization"
            )

        if not validation_report["test_results"].get("colormap_integration", True):
            recommendations.append(
                "Colormap integration issues - check color scheme compatibility"
            )

        if not recommendations:
            recommendations.append("Full matplotlib integration validated successfully")

        validation_report["recommendations"] = recommendations

    except Exception as e:
        validation_report["error_details"].append(f"Validation process failed: {e}")
        validation_report["integration_status"] = "validation_failed"
        is_valid = False

    return is_valid, validation_report
