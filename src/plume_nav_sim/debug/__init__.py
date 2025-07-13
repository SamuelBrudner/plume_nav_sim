"""
Debug Module Base Initialization for Plume Navigation Simulation.

This module provides unified access to all debugging functionality including interactive GUI viewer,
CLI utilities, and session management. Implements hierarchical backend fallback (PySide6 → Streamlit → console)
with graceful degradation when optional dependencies are unavailable. Integrates with existing logging
patterns and maintains zero-overhead design when debugging is disabled.

Key Features:
- **Hierarchical Backend Fallback**: Automatic selection from PySide6 → Streamlit → console
- **Graceful Degradation**: Optional dependency handling with fallback implementations
- **Unified Access**: Centralized entry point for all debug functionality
- **Zero-Overhead Design**: No performance impact when debugging is disabled
- **Session Management**: Comprehensive debug session tracking and correlation
- **Collaborative Debugging**: Support for shared debugging sessions per Section 7.6.4.2

Architecture Components:
- DebugGUI: Main debug interface with backend selection and session management
- Backend Detection: Runtime availability checking for PySide6 and Streamlit
- CLI Integration: Command-line debug utilities and session management
- Session Tracking: Correlation context and collaborative debugging support
- Performance Monitoring: Zero-overhead design with optional performance tracking

Examples:
    Basic debug GUI launch with automatic backend selection:
        >>> from plume_nav_sim.debug import launch_viewer, DebugGUI
        >>> if __availability__['gui']:
        ...     debug_gui = launch_viewer(env=my_env)
        ...     debug_gui.start_session()
        
    Manual backend selection:
        >>> from plume_nav_sim.debug import DebugGUI
        >>> debug_gui = DebugGUI(backend='qt')
        >>> debug_gui.configure_backend(window_size=(1400, 900))
        >>> debug_gui.start_session()
        
    Initial state visualization:
        >>> from plume_nav_sim.debug import plot_initial_state
        >>> if __availability__['gui']:
        ...     fig = plot_initial_state(env, source=my_source, 
        ...                              agent_positions=start_positions)
        
    CLI debug utilities:
        >>> from plume_nav_sim.debug import debug_group
        >>> # Use via command line:
        >>> # python -m plume_nav_sim.debug.cli launch-viewer --backend qt
        
    Availability checking:
        >>> from plume_nav_sim.debug import __availability__
        >>> print(f"GUI available: {__availability__['gui']}")
        >>> print(f"PySide6 available: {__availability__['pyside6']}")
        >>> print(f"Streamlit available: {__availability__['streamlit']}")
"""

import warnings
import importlib
from typing import Dict, Any, Optional, List, Union

# Enhanced logging integration
from plume_nav_sim.utils.logging_setup import get_logger

# Initialize module logger
logger = get_logger(__name__)

# Backend availability detection with graceful fallback
def _detect_backend_availability() -> Dict[str, bool]:
    """
    Detect which GUI backends are available for debug viewer implementation.
    
    Implements hierarchical backend detection per Section 7.1.5.1 Backend Selection Strategy,
    checking for PySide6, Streamlit, and console fallback availability.
    
    Returns:
        Dict mapping backend names to availability status
    """
    availability = {
        'pyside6': False,
        'streamlit': False,
        'console': True  # Always available as fallback
    }
    
    # Check PySide6 availability for Qt backend
    try:
        spec = importlib.util.find_spec('PySide6')
        if spec is not None:
            availability['pyside6'] = True
            logger.debug("PySide6 backend available for Qt-based debug GUI")
    except (ImportError, AttributeError, ModuleNotFoundError):
        logger.debug("PySide6 backend not available")
    
    # Check Streamlit availability for web backend  
    try:
        spec = importlib.util.find_spec('streamlit')
        if spec is not None:
            availability['streamlit'] = True
            logger.debug("Streamlit backend available for web-based debug GUI")
    except (ImportError, AttributeError, ModuleNotFoundError):
        logger.debug("Streamlit backend not available")
    
    return availability

# Detect backend availability at module import
_backend_availability = _detect_backend_availability()

# Debug GUI imports with graceful degradation
try:
    from plume_nav_sim.debug.gui import (
        DebugGUI as _DebugGUI,
        DebugSession,
        DebugConfig,
        plot_initial_state as _plot_initial_state,
        launch_viewer as _launch_viewer
    )
    GUI_AVAILABLE = True
    logger.debug("Debug GUI functionality imported successfully")
except ImportError as e:
    logger.warning(f"Debug GUI functionality not available: {e}")
    GUI_AVAILABLE = False
    
    # Provide fallback implementations
    class _DebugGUI:
        """Fallback debug GUI class when GUI functionality is not available."""
        
        def __init__(self, backend: str = 'console', **kwargs):
            """Initialize fallback debug GUI."""
            self.backend = backend
            warnings.warn(
                "Debug GUI functionality not available. Install PySide6 or Streamlit for full functionality.",
                UserWarning
            )
        
        def start_session(self):
            """Fallback session start."""
            logger.info("Debug session started (console mode)")
            return self
        
        def step_through(self):
            """Fallback step through."""
            logger.info("Debug step executed (console mode)")
            return True
        
        def export_screenshots(self, output_dir: str = './debug_exports'):
            """Fallback screenshot export."""
            logger.warning("Screenshot export not available in console mode")
            return None
        
        def configure_backend(self, **kwargs):
            """Fallback backend configuration."""
            logger.info(f"Backend configuration not available in console mode: {kwargs}")
    
    def _plot_initial_state(*args, **kwargs):
        """Fallback initial state plotting."""
        warnings.warn(
            "plot_initial_state not available. Install matplotlib and GUI backend for visualization.",
            UserWarning
        )
        logger.warning("Initial state plotting not available - GUI dependencies missing")
        return None
    
    def _launch_viewer(*args, **kwargs):
        """Fallback viewer launch."""
        warnings.warn(
            "Debug viewer not available. Install PySide6 or Streamlit for interactive debugging.",
            UserWarning
        )
        logger.warning("Debug viewer launch not available - GUI dependencies missing")
        return _DebugGUI(backend='console')
    
    # Create fallback classes for missing dependencies
    class DebugSession:
        """Fallback debug session class."""
        def __init__(self, *args, **kwargs):
            logger.info("Debug session created (console mode)")
    
    class DebugConfig:
        """Fallback debug config class."""
        def __init__(self, *args, **kwargs):
            logger.info("Debug config created (console mode)")

# CLI debug utilities import with graceful degradation
try:
    from plume_nav_sim.debug.cli import debug_group as _debug_group
    CLI_AVAILABLE = True
    logger.debug("Debug CLI functionality imported successfully")
except ImportError as e:
    logger.warning(f"Debug CLI functionality not available: {e}")
    CLI_AVAILABLE = False
    
    # Provide fallback CLI implementation
    def _debug_group(*args, **kwargs):
        """Fallback debug CLI group."""
        warnings.warn(
            "Debug CLI functionality not available. Install Click and Rich for full CLI debugging.",
            UserWarning
        )
        logger.warning("Debug CLI not available - CLI dependencies missing")


class DebugGUI:
    """
    Main debug GUI interface with automatic backend selection and unified access.
    
    Provides hierarchical backend fallback (PySide6 → Streamlit → console) with
    comprehensive session management and zero-overhead design when disabled.
    
    This class serves as the primary entry point for all debug GUI functionality,
    automatically selecting the best available backend and providing graceful
    degradation when optional dependencies are unavailable.
    """
    
    def __init__(self, backend: str = 'auto', **kwargs):
        """
        Initialize debug GUI with automatic backend selection.
        
        Args:
            backend: Backend preference ('qt', 'streamlit', 'auto', 'console')
            **kwargs: Additional configuration parameters passed to backend
        """
        self.requested_backend = backend
        self.effective_backend = self._select_backend(backend)
        self.kwargs = kwargs
        
        # Create backend implementation
        self._impl = self._create_implementation()
        
        logger.info(
            f"Debug GUI initialized with backend: {self.effective_backend}",
            extra={
                "requested_backend": backend,
                "effective_backend": self.effective_backend,
                "gui_available": GUI_AVAILABLE,
                "backend_availability": _backend_availability
            }
        )
    
    def _select_backend(self, requested: str) -> str:
        """
        Select effective backend based on availability and request.
        
        Implements hierarchical fallback strategy per Section 7.1.5.1.
        
        Args:
            requested: Requested backend name
            
        Returns:
            Effective backend name that will be used
        """
        if requested == 'auto':
            # Hierarchical fallback: PySide6 → Streamlit → console
            if _backend_availability['pyside6']:
                return 'qt'
            elif _backend_availability['streamlit']:
                return 'streamlit'
            else:
                return 'console'
        elif requested == 'qt':
            if _backend_availability['pyside6']:
                return 'qt'
            else:
                warnings.warn(
                    "PySide6 not available, falling back to console mode",
                    UserWarning
                )
                return 'console'
        elif requested == 'streamlit':
            if _backend_availability['streamlit']:
                return 'streamlit'
            else:
                warnings.warn(
                    "Streamlit not available, falling back to console mode",
                    UserWarning
                )
                return 'console'
        elif requested == 'console':
            return 'console'
        else:
            warnings.warn(
                f"Unknown backend '{requested}', falling back to auto selection",
                UserWarning
            )
            return self._select_backend('auto')
    
    def _create_implementation(self):
        """Create backend-specific implementation."""
        if GUI_AVAILABLE and self.effective_backend != 'console':
            return _DebugGUI(backend=self.effective_backend, **self.kwargs)
        else:
            return _DebugGUI(backend='console', **self.kwargs)
    
    def start_session(self):
        """
        Start debug session with backend-appropriate initialization.
        
        Returns:
            Self for method chaining
        """
        try:
            result = self._impl.start_session()
            logger.info(
                "Debug session started successfully",
                extra={
                    "backend": self.effective_backend,
                    "session_type": "interactive" if self.effective_backend != 'console' else "console"
                }
            )
            return result
        except Exception as e:
            logger.error(f"Failed to start debug session: {e}")
            warnings.warn(f"Debug session start failed: {e}", UserWarning)
            return self
    
    def step_through(self):
        """
        Perform single step debugging with backend-appropriate implementation.
        
        Returns:
            True if step was successful, False otherwise
        """
        try:
            result = self._impl.step_through()
            logger.debug("Debug step executed", extra={"backend": self.effective_backend})
            return result
        except Exception as e:
            logger.error(f"Debug step failed: {e}")
            return False
    
    def export_screenshots(self, output_dir: str = './debug_exports'):
        """
        Export screenshots with backend-appropriate implementation.
        
        Args:
            output_dir: Output directory for screenshots
            
        Returns:
            Path to exported screenshot or None if failed/unavailable
        """
        try:
            result = self._impl.export_screenshots(output_dir)
            if result:
                logger.info(
                    f"Screenshot exported successfully: {result}",
                    extra={"output_path": result, "backend": self.effective_backend}
                )
            else:
                logger.warning("Screenshot export failed or not available")
            return result
        except Exception as e:
            logger.error(f"Screenshot export failed: {e}")
            return None
    
    def configure_backend(self, **kwargs):
        """
        Configure backend-specific settings.
        
        Args:
            **kwargs: Backend-specific configuration parameters
        """
        try:
            self._impl.configure_backend(**kwargs)
            logger.debug(
                "Backend configured",
                extra={"backend": self.effective_backend, "config": kwargs}
            )
        except Exception as e:
            logger.error(f"Backend configuration failed: {e}")


def plot_initial_state(*args, **kwargs):
    """
    Plot source location, domain boundaries, and agent starting positions.
    
    Provides publication-quality visualization of the initial simulation state,
    showing spatial relationships between odor sources, navigation domain,
    and agent starting positions for research documentation and debugging.
    
    This function implements graceful degradation when visualization dependencies
    are unavailable, issuing appropriate warnings via the logging system.
    
    Args:
        *args: Positional arguments passed to underlying plot function
        **kwargs: Keyword arguments passed to underlying plot function
        
    Returns:
        matplotlib Figure object or None if visualization unavailable
        
    Examples:
        >>> if __availability__['gui']:
        ...     fig = plot_initial_state(env, source=my_source)
    """
    try:
        result = _plot_initial_state(*args, **kwargs)
        if result is not None:
            logger.debug("Initial state plot created successfully")
        return result
    except Exception as e:
        logger.error(f"Initial state plotting failed: {e}")
        warnings.warn(f"Initial state plotting failed: {e}", UserWarning)
        return None


def launch_viewer(env=None, backend: str = 'auto', **kwargs):
    """
    Launch debug viewer with automatic configuration and session management.
    
    Provides convenient factory function for creating and launching debug GUI
    with sensible defaults, automatic backend selection, and comprehensive
    error handling with graceful degradation.
    
    Args:
        env: Optional environment instance to debug
        backend: Backend selection ('qt', 'streamlit', 'auto', 'console')
        **kwargs: Additional configuration parameters
        
    Returns:
        DebugGUI instance ready for interaction
        
    Examples:
        >>> debug_gui = launch_viewer(env=my_env)
        >>> debug_gui.start_session()
        
        >>> debug_gui = launch_viewer(backend='qt', window_size=(1400, 900))
    """
    try:
        result = _launch_viewer(env=env, backend=backend, **kwargs)
        logger.info(
            "Debug viewer launched successfully",
            extra={
                "backend": backend,
                "has_environment": env is not None,
                "config_params": list(kwargs.keys())
            }
        )
        return result
    except Exception as e:
        logger.error(f"Debug viewer launch failed: {e}")
        warnings.warn(f"Debug viewer launch failed: {e}", UserWarning)
        # Return fallback console implementation
        return DebugGUI(backend='console', **kwargs)


# CLI debug group access
debug_group = _debug_group


# Comprehensive availability information for conditional imports
__availability__ = {
    'gui': GUI_AVAILABLE,
    'cli': CLI_AVAILABLE,
    'pyside6': _backend_availability['pyside6'],
    'streamlit': _backend_availability['streamlit']
}


# Log module initialization status
logger.info(
    "Debug module initialized",
    extra={
        "module": "plume_nav_sim.debug",
        "availability": __availability__,
        "backend_support": _backend_availability,
        "hierarchical_fallback": "enabled"
    }
)


# Public API exports
__all__ = [
    # Main debug interface
    'DebugGUI',
    
    # Utility functions
    'plot_initial_state',
    'launch_viewer',
    
    # CLI access
    'debug_group',
    
    # Availability information
    '__availability__'
]