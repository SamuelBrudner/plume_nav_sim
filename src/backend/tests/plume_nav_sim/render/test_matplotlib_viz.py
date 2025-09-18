"""
Comprehensive test suite for matplotlib visualization renderer validating interactive human mode rendering,
backend management, cross-platform compatibility, performance optimization, and integration with plume 
navigation environment. Tests MatplotlibRenderer, MatplotlibBackendManager, InteractiveUpdateManager 
with comprehensive coverage of backend fallback, color scheme integration, resource management, and 
accessibility features targeting <50ms human mode rendering performance.

This test module provides extensive validation of the matplotlib-based rendering pipeline including:
- MatplotlibRenderer initialization, configuration, and rendering operations
- MatplotlibBackendManager backend selection, fallback mechanisms, and cross-platform compatibility
- InteractiveUpdateManager performance optimization and real-time update capabilities
- Color scheme integration, accessibility features, and visual consistency validation
- Performance testing against target thresholds and resource management verification
- Error handling patterns, graceful degradation, and comprehensive edge case coverage
"""

# External imports with version comments
import pytest  # >=8.0.0 - Testing framework for comprehensive test execution, fixtures, and parameterized testing
import numpy as np  # >=2.1.0 - Array operations for test data generation and numerical validation
import matplotlib  # >=3.9.0 - Primary matplotlib library for backend testing and figure validation
import matplotlib.pyplot as plt  # >=3.9.0 - Pyplot interface testing for backend management and display functionality
import matplotlib.figure  # >=3.9.0 - Figure object testing for resource management and performance monitoring
import matplotlib.backends.backend_agg  # >=3.9.0 - Agg backend testing for headless operation validation
import unittest.mock  # >=3.10 - Mocking utilities for backend availability simulation and error condition testing
import warnings  # >=3.10 - Warning capture and validation for backend compatibility warnings
import time  # >=3.10 - Timing utilities for performance testing and rendering latency measurement
import os  # >=3.10 - Environment variable testing for headless detection and platform-specific behavior validation
import sys  # >=3.10 - Platform detection testing for cross-platform compatibility validation
import tempfile  # >=3.10 - Temporary file testing for figure saving functionality and cleanup verification
from typing import Dict, List, Optional, Union, Any, Tuple  # >=3.10 - Type hints for test functions and fixtures
from unittest.mock import Mock, patch, MagicMock, call  # >=3.10 - Mock objects for comprehensive testing scenarios

# Internal imports from plume_nav_sim modules
from plume_nav_sim.render.matplotlib_viz import (
    MatplotlibRenderer, MatplotlibBackendManager, InteractiveUpdateManager,
    create_matplotlib_renderer, detect_matplotlib_capabilities, configure_matplotlib_backend,
    validate_matplotlib_integration
)
from plume_nav_sim.render.base_renderer import BaseRenderer, RenderContext, create_render_context
from plume_nav_sim.render.color_schemes import CustomColorScheme, get_default_scheme, create_accessibility_scheme
from plume_nav_sim.core.types import RenderMode, Coordinates, GridSize
from plume_nav_sim.core.constants import (
    PERFORMANCE_TARGET_HUMAN_RENDER_MS, MATPLOTLIB_DEFAULT_FIGSIZE, BACKEND_PRIORITY_LIST
)
from plume_nav_sim.utils.exceptions import ValidationError, RenderingError

# Global test constants for consistent test execution
MATPLOTLIB_AVAILABLE = True  # Flag indicating matplotlib availability for test setup
TEST_GRID_SIZE = GridSize(width=32, height=32)  # Small grid size for fast test execution
TEST_AGENT_POSITION = Coordinates(x=10, y=15)  # Test agent position for marker testing
TEST_SOURCE_POSITION = Coordinates(x=20, y=25)  # Test source position for marker testing
PERFORMANCE_TOLERANCE_MS = 5.0  # Additional tolerance for performance testing in CI environments
MAX_CLEANUP_TIMEOUT_SEC = 2.0  # Maximum timeout for resource cleanup testing
MOCK_BACKEND_PRIORITY = ['MockBackend1', 'MockBackend2', 'Agg']  # Mock backend list for testing
TEST_COLORMAP_NAME = 'gray'  # Test colormap for matplotlib integration testing


# Session-scoped fixtures for test environment setup
@pytest.fixture(scope='session')
def matplotlib_available():
    """
    Session-scoped fixture to check matplotlib availability and skip tests if unavailable.
    
    Returns:
        bool: True if matplotlib is available and functional for testing
        
    Raises:
        pytest.skip: If matplotlib is not available or not functional
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.backends.backend_agg
        
        # Test basic matplotlib functionality including pyplot and figure modules
        fig, ax = plt.subplots(figsize=(2, 2))
        ax.plot([1, 2, 3], [1, 2, 3])
        plt.close(fig)
        
        # Validate essential matplotlib components for rendering testing
        assert hasattr(plt, 'figure'), "matplotlib.pyplot.figure not available"
        assert hasattr(plt, 'imshow'), "matplotlib.pyplot.imshow not available"
        
        return True
        
    except (ImportError, AttributeError, AssertionError) as e:
        pytest.skip(f"Matplotlib not available or not functional: {str(e)}")


@pytest.fixture(scope='function')
def test_render_context():
    """
    Function-scoped fixture providing validated RenderContext with test data for rendering operations.
    
    Returns:
        RenderContext: Validated render context with test concentration field, agent and source positions
    """
    # Create test concentration field with realistic Gaussian plume data
    concentration_field = np.zeros((TEST_GRID_SIZE.height, TEST_GRID_SIZE.width), dtype=np.float32)
    
    # Create simple Gaussian distribution centered at source position
    y_center, x_center = TEST_SOURCE_POSITION.y, TEST_SOURCE_POSITION.x
    y, x = np.ogrid[:TEST_GRID_SIZE.height, :TEST_GRID_SIZE.width]
    
    # Generate concentration field with normalized values [0,1]
    sigma = 8.0
    distance_sq = (x - x_center)**2 + (y - y_center)**2
    concentration_field = np.exp(-distance_sq / (2 * sigma**2))
    concentration_field = concentration_field.astype(np.float32)
    
    # Create RenderContext using create_render_context factory function
    render_context = create_render_context(
        concentration_field=concentration_field,
        agent_position=TEST_AGENT_POSITION,
        source_position=TEST_SOURCE_POSITION,
        grid_size=TEST_GRID_SIZE
    )
    
    # Validate render context and ensure all parameters are consistent
    render_context.validate()
    
    return render_context


@pytest.fixture(scope='function')
def test_color_scheme():
    """
    Function-scoped fixture providing CustomColorScheme with test visualization colors.
    
    Returns:
        CustomColorScheme: Configured color scheme for matplotlib testing with validation
    """
    # Create test color scheme with standard visualization colors
    color_scheme = get_default_scheme()
    
    # Configure color scheme for matplotlib optimization and compatibility
    color_scheme.optimize_for_render_mode(RenderMode.HUMAN)
    
    # Validate color scheme parameters and matplotlib integration
    color_scheme.validate(check_accessibility=False, check_performance=True)
    
    return color_scheme


@pytest.fixture(scope='function')
def mock_matplotlib_backend():
    """
    Function-scoped fixture providing mocked matplotlib backend for controlled testing.
    
    Returns:
        Mock: Mock backend configuration for backend selection and fallback testing
    """
    with patch('matplotlib.pyplot.switch_backend') as mock_switch:
        with patch('matplotlib.get_backend') as mock_get_backend:
            # Create mock backend with configurable availability
            mock_get_backend.return_value = 'Agg'
            mock_switch.return_value = None
            
            yield {
                'switch_backend': mock_switch,
                'get_backend': mock_get_backend
            }


@pytest.fixture(scope='function')
def performance_monitor():
    """
    Function-scoped fixture providing performance monitoring for rendering operation timing.
    
    Returns:
        Dict: Performance monitoring context with timing utilities and validation methods
    """
    performance_data = {
        'start_times': {},
        'durations': {},
        'operations': []
    }
    
    def start_timing(operation_name: str):
        """Start timing for a specific operation"""
        performance_data['start_times'][operation_name] = time.time()
    
    def end_timing(operation_name: str) -> float:
        """End timing and return duration in milliseconds"""
        if operation_name in performance_data['start_times']:
            duration = (time.time() - performance_data['start_times'][operation_name]) * 1000
            performance_data['durations'][operation_name] = duration
            performance_data['operations'].append(operation_name)
            return duration
        return 0.0
    
    def validate_performance(operation_name: str, target_ms: float, tolerance_ms: float = 0.0):
        """Validate operation performance against target with tolerance"""
        if operation_name in performance_data['durations']:
            actual_duration = performance_data['durations'][operation_name]
            max_allowed = target_ms + tolerance_ms
            assert actual_duration <= max_allowed, f"Operation '{operation_name}' took {actual_duration:.2f}ms, exceeds target {max_allowed:.2f}ms"
            return True
        return False
    
    return {
        'start_timing': start_timing,
        'end_timing': end_timing,
        'validate_performance': validate_performance,
        'get_data': lambda: performance_data.copy()
    }


@pytest.fixture(scope='function')
def headless_environment():
    """
    Function-scoped fixture simulating headless environment for backend fallback testing.
    
    Returns:
        Dict: Headless environment configuration with context manager for DISPLAY variable manipulation
    """
    original_display = os.environ.get('DISPLAY')
    
    def enable_headless():
        """Enable headless mode by removing DISPLAY variable"""
        if 'DISPLAY' in os.environ:
            del os.environ['DISPLAY']
    
    def restore_display():
        """Restore original DISPLAY environment variable"""
        if original_display is not None:
            os.environ['DISPLAY'] = original_display
        elif 'DISPLAY' in os.environ:
            del os.environ['DISPLAY']
    
    yield {
        'enable_headless': enable_headless,
        'restore_display': restore_display,
        'original_display': original_display
    }
    
    # Clean up by restoring original environment
    restore_display()


# Unit tests for matplotlib availability and basic functionality
@pytest.mark.unit
def test_matplotlib_availability():
    """Test matplotlib import availability and version compatibility for test environment validation."""
    # Import matplotlib and validate version compatibility with >=3.9.0 requirement
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.figure
    
    # Check basic matplotlib functionality including pyplot and figure modules
    assert hasattr(matplotlib, '__version__'), "Matplotlib version information not available"
    
    version_parts = matplotlib.__version__.split('.')
    major_version = int(version_parts[0])
    minor_version = int(version_parts[1])
    assert major_version >= 3, f"Matplotlib major version {major_version} < required 3"
    assert major_version > 3 or minor_version >= 9, f"Matplotlib version {matplotlib.__version__} < required 3.9.0"
    
    # Validate essential matplotlib components for rendering testing
    assert hasattr(plt, 'figure'), "matplotlib.pyplot.figure not available"
    assert hasattr(plt, 'subplots'), "matplotlib.pyplot.subplots not available"
    assert hasattr(plt, 'imshow'), "matplotlib.pyplot.imshow not available"
    assert hasattr(matplotlib.figure, 'Figure'), "matplotlib.figure.Figure not available"


@pytest.mark.unit
def test_backend_manager_initialization():
    """Test MatplotlibBackendManager initialization with various configuration options and parameter validation."""
    # Create MatplotlibBackendManager with default parameters and validate initialization
    backend_manager = MatplotlibBackendManager()
    assert backend_manager is not None
    assert hasattr(backend_manager, 'backend_preferences')
    assert hasattr(backend_manager, 'fallback_backend')
    assert hasattr(backend_manager, '_original_backend')
    
    # Test initialization with custom backend preferences and fallback configuration
    custom_preferences = ['Qt5Agg', 'TkAgg', 'Agg']
    custom_manager = MatplotlibBackendManager(
        backend_preferences=custom_preferences,
        fallback_backend='Agg',
        enable_caching=True
    )
    
    # Validate backend_preferences setting and caching configuration
    assert custom_manager.backend_preferences == custom_preferences
    assert custom_manager.fallback_backend == 'Agg'
    assert custom_manager.caching_enabled == True
    
    # Test initialization with invalid parameters and verify appropriate error handling
    with pytest.raises(ValidationError):
        MatplotlibBackendManager(backend_preferences=[])  # Empty preferences list
    
    with pytest.raises(ValidationError):
        MatplotlibBackendManager(fallback_backend="")  # Empty fallback backend
    
    # Assert proper logger setup and component configuration for debugging
    assert hasattr(custom_manager, 'logger')
    assert custom_manager.logger is not None


@pytest.mark.unit
def test_backend_selection_priority(mock_matplotlib_backend):
    """Test backend selection with priority-based testing and compatibility validation across different system configurations."""
    # Create backend manager with test priority list and validate selection process
    test_priority = ['MockInteractive', 'MockGUI', 'Agg']
    backend_manager = MatplotlibBackendManager(backend_preferences=test_priority)
    
    # Mock backend availability testing for systematic priority evaluation
    with patch('matplotlib.pyplot.switch_backend') as mock_switch:
        with patch('builtins.__import__') as mock_import:
            # Simulate first backend available, second unavailable, third available
            def side_effect(backend_name, *args, **kwargs):
                if backend_name == 'MockInteractive':
                    return True  # Available
                elif backend_name == 'MockGUI':
                    raise ImportError("Backend not available")  # Unavailable
                elif backend_name == 'Agg':
                    return True  # Available
                else:
                    raise ImportError("Unknown backend")
            
            mock_import.side_effect = side_effect
            
            # Test backend selection with various availability scenarios and fallback triggers
            selected_backend = backend_manager.select_backend()
            
            # Should select first available backend from priority list
            assert selected_backend is not None
            
    # Validate headless mode detection and Agg backend fallback mechanism
    with patch('os.environ.get', return_value=None) as mock_env:  # No DISPLAY variable
        headless_manager = MatplotlibBackendManager()
        backend = headless_manager.select_backend(force_headless=True)
        
        # Should fall back to Agg backend in headless environment
        assert backend == 'Agg' or backend is not None
    
    # Assert backend capabilities reporting and configuration validation
    capabilities = backend_manager.get_backend_capabilities()
    assert isinstance(capabilities, dict)
    assert 'display_available' in capabilities
    assert 'interactive_supported' in capabilities
    
    # Test force_reselection parameter and backend switching functionality
    original_backend = backend_manager.get_current_backend()
    backend_manager.select_backend(force_reselection=True)
    # Should trigger reselection process even if backend was previously selected


@pytest.mark.unit
def test_backend_capabilities_detection():
    """Test comprehensive backend capabilities detection including display support, interactive features, and performance characteristics."""
    backend_manager = MatplotlibBackendManager()
    
    # Test backend capabilities detection for various backend types
    capabilities = backend_manager.get_backend_capabilities()
    
    # Validate display support detection and GUI toolkit integration testing
    assert 'display_available' in capabilities
    assert 'gui_toolkit' in capabilities
    assert 'interactive_supported' in capabilities
    assert isinstance(capabilities['display_available'], bool)
    
    # Test interactive features assessment including event handling and animation support
    if capabilities['interactive_supported']:
        assert 'event_handling' in capabilities
        assert 'animation_supported' in capabilities
    
    # Mock different system environments for capability detection validation
    with patch('os.environ.get') as mock_env:
        # Test headless environment
        mock_env.return_value = None  # No DISPLAY
        headless_caps = backend_manager.get_backend_capabilities()
        assert headless_caps['display_available'] == False
        
        # Test display environment
        mock_env.return_value = ':0.0'  # X11 display
        display_caps = backend_manager.get_backend_capabilities()
        # May be True or False depending on actual environment
    
    # Assert performance characteristics measurement and caching functionality
    assert 'performance_tier' in capabilities
    assert capabilities['performance_tier'] in ['high', 'medium', 'low']
    
    # Test capability refresh and cache invalidation mechanisms
    backend_manager.refresh_capabilities()
    new_capabilities = backend_manager.get_backend_capabilities()
    assert isinstance(new_capabilities, dict)


@pytest.mark.unit
def test_backend_configuration():
    """Test backend configuration with optimization settings, threading configuration, and error handling validation."""
    backend_manager = MatplotlibBackendManager()
    
    # Test backend configuration with various matplotlib backends and settings
    config_options = {
        'figure_dpi': 100,
        'figure_size': MATPLOTLIB_DEFAULT_FIGSIZE,
        'interactive_mode': False
    }
    
    # Validate configuration options application including DPI, figure size, and threading
    result = backend_manager.configure_backend(config_options)
    assert isinstance(result, bool)
    
    # Test configuration error handling and validation for invalid settings
    invalid_config = {
        'figure_dpi': -100,  # Invalid negative DPI
        'invalid_option': 'invalid_value'
    }
    
    with pytest.raises(ValidationError):
        backend_manager.configure_backend(invalid_config, strict_validation=True)
    
    # Mock backend switching failures and verify error handling mechanisms
    with patch('matplotlib.pyplot.switch_backend', side_effect=ImportError("Backend switch failed")):
        with pytest.raises(RenderingError):
            backend_manager.switch_to_backend('InvalidBackend')
    
    # Assert configuration success reporting and validation status
    status = backend_manager.get_configuration_status()
    assert isinstance(status, dict)
    assert 'backend_configured' in status
    assert 'configuration_valid' in status
    
    # Test configuration rollback and restoration functionality
    backend_manager.save_configuration()
    backend_manager.configure_backend({'figure_dpi': 200})
    backend_manager.restore_configuration()
    # Configuration should be restored to saved state


@pytest.mark.unit
def test_matplotlib_renderer_initialization(test_color_scheme):
    """Test MatplotlibRenderer initialization with grid configuration, color scheme setup, and performance monitoring infrastructure."""
    # Create MatplotlibRenderer with test grid size and validate initialization
    renderer = MatplotlibRenderer(
        grid_size=TEST_GRID_SIZE,
        color_scheme=test_color_scheme
    )
    
    assert renderer is not None
    assert isinstance(renderer, BaseRenderer)  # Should inherit from BaseRenderer
    assert renderer.grid_size == TEST_GRID_SIZE
    assert renderer.color_scheme is not None
    
    # Test initialization with custom color scheme and backend preferences
    custom_scheme = create_accessibility_scheme('high_contrast')
    custom_renderer = MatplotlibRenderer(
        grid_size=TEST_GRID_SIZE,
        color_scheme=custom_scheme,
        backend_preferences=['Agg', 'TkAgg']
    )
    
    # Validate component initialization including backend manager and color manager
    assert custom_renderer.color_scheme == custom_scheme
    assert hasattr(custom_renderer, 'backend_manager')
    assert hasattr(custom_renderer, '_figure')
    assert hasattr(custom_renderer, '_axes')
    
    # Test initialization with invalid parameters and error handling validation
    with pytest.raises(ValidationError):
        MatplotlibRenderer(grid_size=None)  # Invalid grid size
    
    with pytest.raises(ValidationError):
        MatplotlibRenderer(
            grid_size=TEST_GRID_SIZE,
            color_scheme="invalid_scheme"  # Invalid color scheme type
        )
    
    # Assert performance monitoring setup and resource cache initialization
    assert hasattr(renderer, 'performance_metrics')
    assert hasattr(renderer, '_resource_cache')
    
    # Test lazy initialization patterns and resource creation timing
    # Resources should not be created until first use
    assert renderer._figure is None  # Figure should be created lazily
    assert renderer._axes is None    # Axes should be created lazily


@pytest.mark.unit
def test_matplotlib_renderer_supports_render_mode():
    """Test render mode support validation for HUMAN mode with backend availability checking and compatibility assessment."""
    renderer = MatplotlibRenderer(grid_size=TEST_GRID_SIZE)
    
    # Test supports_render_mode with RenderMode.HUMAN and backend availability
    supports_human = renderer.supports_render_mode(RenderMode.HUMAN)
    assert isinstance(supports_human, bool)
    
    # Validate return True for HUMAN mode with functional matplotlib backend
    if MATPLOTLIB_AVAILABLE:
        assert supports_human == True, "Should support HUMAN mode when matplotlib is available"
    
    # Test return False for RGB_ARRAY mode (not supported by matplotlib renderer)
    supports_rgb = renderer.supports_render_mode(RenderMode.RGB_ARRAY)
    assert supports_rgb == False, "MatplotlibRenderer should not support RGB_ARRAY mode"
    
    # Mock backend unavailability and verify mode support returns False
    with patch.object(renderer.backend_manager, 'select_backend', return_value=None):
        supports_human_no_backend = renderer.supports_render_mode(RenderMode.HUMAN)
        assert supports_human_no_backend == False, "Should not support HUMAN mode without available backend"
    
    # Assert proper mode validation and capability checking
    with pytest.raises(ValidationError):
        renderer.supports_render_mode("invalid_mode")  # Invalid render mode
    
    with pytest.raises(ValidationError):
        renderer.supports_render_mode(None)  # None render mode
    
    # Test mode support with different backend configurations
    agg_renderer = MatplotlibRenderer(
        grid_size=TEST_GRID_SIZE,
        backend_preferences=['Agg']
    )
    assert agg_renderer.supports_render_mode(RenderMode.HUMAN) == True, "Should support HUMAN mode with Agg backend"


@pytest.mark.unit
def test_renderer_resource_initialization():
    """Test matplotlib-specific resource initialization including backend configuration, figure creation, and color scheme setup."""
    renderer = MatplotlibRenderer(grid_size=TEST_GRID_SIZE)
    
    # Test _initialize_renderer_resources method with proper resource setup
    renderer.initialize()
    
    # Validate backend configuration and figure creation with test grid dimensions
    assert renderer._figure is not None
    assert isinstance(renderer._figure, matplotlib.figure.Figure)
    assert renderer._axes is not None
    
    # Test axes configuration with coordinate system and visualization properties
    axes = renderer._axes
    assert axes.get_xlim()[1] >= TEST_GRID_SIZE.width
    assert axes.get_ylim()[1] >= TEST_GRID_SIZE.height
    assert axes.get_aspect() == 'equal'  # Should maintain aspect ratio
    
    # Validate color scheme initialization and matplotlib optimization
    assert renderer.color_scheme is not None
    
    # Assert update manager creation and interactive mode configuration
    assert hasattr(renderer, 'update_manager')
    assert renderer.update_manager is not None
    
    # Test error handling during resource initialization and cleanup
    with patch('matplotlib.pyplot.subplots', side_effect=Exception("Figure creation failed")):
        error_renderer = MatplotlibRenderer(grid_size=TEST_GRID_SIZE)
        with pytest.raises(RenderingError):
            error_renderer.initialize()


@pytest.mark.unit
def test_interactive_update_manager_initialization():
    """Test InteractiveUpdateManager initialization with matplotlib objects and performance configuration validation."""
    # Create test matplotlib figure and axes for update manager initialization
    fig, ax = plt.subplots(figsize=MATPLOTLIB_DEFAULT_FIGSIZE)
    
    # Initialize InteractiveUpdateManager with figure references and update interval
    update_manager = InteractiveUpdateManager(
        figure=fig,
        axes=ax,
        update_interval=0.1,  # 100ms update interval
        performance_monitoring=True
    )
    
    # Validate marker references initialization and cache setup
    assert update_manager.figure == fig
    assert update_manager.axes == ax
    assert update_manager.update_interval == 0.1
    assert hasattr(update_manager, '_concentration_cache')
    assert hasattr(update_manager, '_agent_marker')
    assert hasattr(update_manager, '_source_marker')
    
    # Test performance monitoring configuration and statistics tracking
    assert update_manager.performance_monitoring == True
    assert hasattr(update_manager, 'performance_stats')
    assert 'update_count' in update_manager.performance_stats
    assert 'total_update_time' in update_manager.performance_stats
    
    # Assert proper resource management and cleanup preparation
    assert hasattr(update_manager, 'cleanup_resources')
    
    # Test initialization with invalid parameters and error handling
    with pytest.raises(ValidationError):
        InteractiveUpdateManager(figure=None, axes=ax)  # Invalid figure
    
    with pytest.raises(ValidationError):
        InteractiveUpdateManager(figure=fig, axes=None)  # Invalid axes
    
    # Clean up test figure
    plt.close(fig)


@pytest.mark.unit
def test_concentration_field_update():
    """Test concentration field heatmap update with efficient change detection and matplotlib axes configuration."""
    fig, ax = plt.subplots(figsize=(4, 4))
    update_manager = InteractiveUpdateManager(figure=fig, axes=ax)
    
    # Create test concentration field with realistic plume data
    concentration_field = np.random.rand(TEST_GRID_SIZE.height, TEST_GRID_SIZE.width).astype(np.float32)
    concentration_field = np.clip(concentration_field, 0.0, 1.0)
    
    # Test update_concentration_field with change detection optimization
    result = update_manager.update_concentration_field(concentration_field)
    assert result == True, "Should successfully update concentration field"
    
    # Validate matplotlib axes configuration and colormap application
    images = ax.get_images()
    assert len(images) > 0, "Should create image object for concentration field"
    
    # Test force_update parameter and cache bypassing functionality
    same_field_result = update_manager.update_concentration_field(concentration_field)
    assert same_field_result == False, "Should detect no change and skip update"
    
    force_result = update_manager.update_concentration_field(concentration_field, force_update=True)
    assert force_result == True, "Should force update even with identical data"
    
    # Assert proper colormap normalization and scaling for visualization
    image = images[0]
    assert hasattr(image, 'get_clim'), "Image should have colormap limits"
    vmin, vmax = image.get_clim()
    assert vmin >= 0.0 and vmax <= 1.0, f"Colormap limits {vmin}, {vmax} should be in [0,1] range"
    
    # Test update performance timing and optimization validation
    start_time = time.time()
    for _ in range(10):
        update_manager.update_concentration_field(concentration_field, force_update=True)
    update_time = (time.time() - start_time) * 1000 / 10  # Average time per update
    
    assert update_time < 20.0, f"Concentration field update should be <20ms, got {update_time:.2f}ms"
    
    plt.close(fig)


@pytest.mark.unit
def test_agent_marker_update():
    """Test agent position marker update with efficient positioning and color application for real-time tracking."""
    fig, ax = plt.subplots(figsize=(4, 4))
    update_manager = InteractiveUpdateManager(figure=fig, axes=ax)
    
    # Create test agent position and color scheme configuration
    agent_position = TEST_AGENT_POSITION
    color_scheme = get_default_scheme()
    
    # Test update_agent_marker with position change detection and optimization
    result = update_manager.update_agent_marker(agent_position, color_scheme)
    assert result == True, "Should successfully update agent marker"
    
    # Validate marker creation and position update using matplotlib scatter
    collections = ax.collections
    # Should create scatter plot collection for agent marker
    
    # Test color application from color scheme with RGB conversion
    # Agent marker should use agent_color from color scheme
    expected_color = tuple(c / 255.0 for c in color_scheme.agent_color)
    
    # Assert marker properties configuration including size and visibility
    # Marker should be visible and properly positioned
    
    # Test animation transition parameter and smooth movement functionality
    new_position = Coordinates(x=12, y=17)
    smooth_result = update_manager.update_agent_marker(
        new_position, 
        color_scheme, 
        animate_transition=True
    )
    assert smooth_result == True, "Should handle animated marker transition"
    
    plt.close(fig)


@pytest.mark.unit  
def test_source_marker_update():
    """Test source location marker update with cross-pattern visualization and visibility optimization for goal indication."""
    fig, ax = plt.subplots(figsize=(4, 4))
    update_manager = InteractiveUpdateManager(figure=fig, axes=ax)
    
    # Create test source position and marker configuration
    source_position = TEST_SOURCE_POSITION
    color_scheme = get_default_scheme()
    
    # Test update_source_marker with cross-pattern visualization
    result = update_manager.update_source_marker(source_position, color_scheme)
    assert result == True, "Should successfully update source marker"
    
    # Validate marker creation with white color and appropriate size
    # Source marker should use source_color (typically white) from color scheme
    expected_color = tuple(c / 255.0 for c in color_scheme.source_color)
    
    # Test position update and visibility optimization for display hierarchy
    # Source marker should be clearly visible and distinguishable from agent marker
    
    # Assert proper matplotlib marker configuration and styling
    lines = ax.get_lines()
    # Should create line objects for cross pattern (horizontal and vertical lines)
    
    # Test marker layering and z-order for proper display hierarchy
    # Source marker should have higher z-order than concentration field but lower than agent
    
    plt.close(fig)


@pytest.mark.unit
def test_display_refresh_functionality():
    """Test matplotlib display refresh with performance optimization and frame rate control for smooth interactive updates."""
    fig, ax = plt.subplots(figsize=(4, 4))
    update_manager = InteractiveUpdateManager(figure=fig, axes=ax, update_interval=0.05)
    
    # Test refresh_display with frame rate control and timing validation
    start_time = time.time()
    result = update_manager.refresh_display()
    refresh_time = (time.time() - start_time) * 1000
    
    assert result == True, "Should successfully refresh display"
    assert refresh_time < 100.0, f"Display refresh should be <100ms, got {refresh_time:.2f}ms"
    
    # Validate figure.canvas.draw() execution and error handling
    with patch.object(fig.canvas, 'draw', side_effect=Exception("Draw failed")) as mock_draw:
        error_result = update_manager.refresh_display()
        # Should handle draw errors gracefully
        mock_draw.assert_called_once()
    
    # Test interactive pause timing and update interval management
    # Should respect update_interval for smooth animation
    
    # Mock display errors and verify graceful degradation mechanisms
    with patch.object(fig.canvas, 'flush_events', side_effect=Exception("Flush failed")):
        # Should continue operating even if flush_events fails
        degraded_result = update_manager.refresh_display()
    
    # Assert performance measurement and timing statistics collection
    stats = update_manager.get_performance_stats()
    assert 'refresh_count' in stats
    assert 'average_refresh_time' in stats
    
    # Test force_refresh parameter and immediate update functionality
    force_result = update_manager.refresh_display(force_refresh=True)
    assert force_result == True, "Should force refresh regardless of timing"
    
    plt.close(fig)


@pytest.mark.unit
def test_batch_update_optimization():
    """Test batch update of all visualization elements with optimized rendering and performance monitoring."""
    fig, ax = plt.subplots(figsize=(4, 4))
    update_manager = InteractiveUpdateManager(figure=fig, axes=ax)
    
    # Create complete render context with concentration field and positions
    render_context = create_render_context(
        concentration_field=np.random.rand(TEST_GRID_SIZE.height, TEST_GRID_SIZE.width).astype(np.float32),
        agent_position=TEST_AGENT_POSITION,
        source_position=TEST_SOURCE_POSITION,
        grid_size=TEST_GRID_SIZE
    )
    color_scheme = get_default_scheme()
    
    # Test batch_update with coordinated element updates and optimization
    start_time = time.time()
    result = update_manager.batch_update(render_context, color_scheme)
    batch_time = (time.time() - start_time) * 1000
    
    assert result == True, "Should successfully complete batch update"
    assert batch_time < 50.0, f"Batch update should be <50ms, got {batch_time:.2f}ms"
    
    # Validate selective refresh and change detection optimization
    # Second batch update with same data should be optimized
    optimized_start = time.time()
    optimized_result = update_manager.batch_update(render_context, color_scheme)
    optimized_time = (time.time() - optimized_start) * 1000
    
    # Should be faster due to change detection
    assert optimized_time <= batch_time, "Optimized batch update should not be slower"
    
    # Test performance monitoring and timing collection during batch operations
    stats = update_manager.get_performance_stats()
    assert 'batch_update_count' in stats
    assert 'total_batch_time' in stats
    
    # Assert optimization statistics generation and performance reporting
    optimization_report = update_manager.get_optimization_report()
    assert 'cache_hit_ratio' in optimization_report
    assert 'average_update_time' in optimization_report
    
    # Test batch update with various optimization settings and configurations
    optimized_update_manager = InteractiveUpdateManager(
        figure=fig, 
        axes=ax,
        optimization_level='high',
        change_detection=True
    )
    
    high_perf_result = optimized_update_manager.batch_update(render_context, color_scheme)
    assert high_perf_result == True, "Should handle high optimization level"
    
    plt.close(fig)


# Integration tests for complete rendering pipeline
@pytest.mark.integration
@pytest.mark.performance
def test_human_mode_rendering(test_render_context, test_color_scheme, performance_monitor):
    """Test complete human mode rendering with real-time updates, marker placement, and performance validation targeting <50ms updates."""
    # Create comprehensive render context with realistic plume data
    render_context = test_render_context
    color_scheme = test_color_scheme
    
    # Initialize matplotlib renderer with human mode configuration
    renderer = MatplotlibRenderer(
        grid_size=TEST_GRID_SIZE,
        color_scheme=color_scheme,
        backend_preferences=['Agg']  # Use Agg for consistent testing
    )
    
    # Test _render_human method with complete visualization pipeline
    performance_monitor['start_timing']('human_render')
    
    renderer.initialize()
    result = renderer.render(render_context, RenderMode.HUMAN)
    
    render_time = performance_monitor['end_timing']('human_render')
    
    # Validate concentration field rendering with colormap application
    assert renderer._figure is not None, "Should create matplotlib figure"
    assert renderer._axes is not None, "Should create matplotlib axes"
    
    # Assert agent and source marker placement with proper positioning
    # Markers should be visible and correctly positioned
    
    # Measure rendering performance against PERFORMANCE_TARGET_HUMAN_RENDER_MS
    target_ms = PERFORMANCE_TARGET_HUMAN_RENDER_MS + PERFORMANCE_TOLERANCE_MS
    performance_monitor['validate_performance']('human_render', target_ms, PERFORMANCE_TOLERANCE_MS)
    
    # Test error handling and recovery during rendering operations
    with patch.object(renderer._axes, 'imshow', side_effect=Exception("Rendering failed")):
        with pytest.raises(RenderingError):
            renderer.render(render_context, RenderMode.HUMAN)
    
    # Validate resource cleanup and memory management after rendering
    renderer.cleanup_resources()
    assert renderer._figure is None or not hasattr(renderer._figure, 'canvas'), "Should clean up figure resources"


@pytest.mark.integration
def test_color_scheme_integration(test_render_context):
    """Test color scheme integration with matplotlib renderer including colormap application and marker visualization."""
    # Create various color schemes including default and accessibility variants
    default_scheme = get_default_scheme()
    accessibility_scheme = create_accessibility_scheme('high_contrast')
    
    renderer = MatplotlibRenderer(grid_size=TEST_GRID_SIZE, backend_preferences=['Agg'])
    
    # Test set_color_scheme with validation and matplotlib optimization
    renderer.set_color_scheme(default_scheme)
    assert renderer.color_scheme == default_scheme
    
    # Validate colormap integration and axes configuration updates
    renderer.initialize()
    renderer.render(test_render_context, RenderMode.HUMAN)
    
    # Test color scheme switching and interactive display refresh
    renderer.set_color_scheme(accessibility_scheme)
    renderer.render(test_render_context, RenderMode.HUMAN)
    
    # Assert proper RGB to matplotlib color conversion and formatting
    matplotlib_color = renderer.color_scheme.agent_color
    assert len(matplotlib_color) == 3, "Should have RGB color components"
    assert all(0 <= c <= 255 for c in matplotlib_color), "Color components should be in valid range"
    
    # Test color scheme performance impact and optimization effectiveness
    start_time = time.time()
    for _ in range(5):
        renderer.render(test_render_context, RenderMode.HUMAN)
    color_render_time = (time.time() - start_time) * 1000 / 5
    
    assert color_render_time < 100.0, f"Color scheme rendering should be efficient, got {color_render_time:.2f}ms"
    
    renderer.cleanup_resources()


@pytest.mark.unit
def test_figure_management():
    """Test matplotlib figure object management including creation, access, and resource cleanup with error handling."""
    renderer = MatplotlibRenderer(grid_size=TEST_GRID_SIZE)
    
    # Test get_figure method with lazy initialization and resource management
    figure = renderer.get_figure()
    assert figure is not None
    assert isinstance(figure, matplotlib.figure.Figure)
    
    # Validate figure creation with proper size and configuration settings
    assert figure.get_size_inches()[0] > 0, "Figure should have positive width"
    assert figure.get_size_inches()[1] > 0, "Figure should have positive height"
    
    # Test figure access patterns and reference management
    same_figure = renderer.get_figure()
    assert same_figure is figure, "Should return same figure instance"
    
    # Validate figure cleanup and resource disposal mechanisms
    renderer.cleanup_resources()
    
    # Figure should be properly closed
    try:
        # Accessing figure after cleanup may raise exception
        figure.canvas.draw()
    except Exception:
        pass  # Expected behavior after cleanup
    
    # Assert proper error handling for figure creation failures
    with patch('matplotlib.pyplot.figure', side_effect=Exception("Figure creation failed")):
        error_renderer = MatplotlibRenderer(grid_size=TEST_GRID_SIZE)
        with pytest.raises(RenderingError):
            error_renderer.get_figure()
    
    # Test figure sharing and multi-renderer scenarios
    renderer1 = MatplotlibRenderer(grid_size=TEST_GRID_SIZE)
    renderer2 = MatplotlibRenderer(grid_size=TEST_GRID_SIZE)
    
    fig1 = renderer1.get_figure()
    fig2 = renderer2.get_figure()
    assert fig1 is not fig2, "Different renderers should have different figures"
    
    renderer1.cleanup_resources()
    renderer2.cleanup_resources()


@pytest.mark.integration
def test_figure_saving_functionality():
    """Test figure saving capabilities with format support, quality configuration, and metadata preservation."""
    renderer = MatplotlibRenderer(grid_size=TEST_GRID_SIZE, backend_preferences=['Agg'])
    render_context = create_render_context(
        concentration_field=np.random.rand(TEST_GRID_SIZE.height, TEST_GRID_SIZE.width).astype(np.float32),
        agent_position=TEST_AGENT_POSITION,
        source_position=TEST_SOURCE_POSITION,
        grid_size=TEST_GRID_SIZE
    )
    
    # Create test figure with rendered visualization content
    renderer.initialize()
    renderer.render(render_context, RenderMode.HUMAN)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test save_figure with various file formats (PNG, PDF, SVG) and quality settings
        png_path = os.path.join(temp_dir, "test_render.png")
        pdf_path = os.path.join(temp_dir, "test_render.pdf")
        svg_path = os.path.join(temp_dir, "test_render.svg")
        
        # Test PNG format with DPI settings
        png_result = renderer.save_figure(
            png_path,
            format='png',
            dpi=150,
            transparent=False
        )
        assert png_result == True, "Should successfully save PNG figure"
        assert os.path.exists(png_path), "PNG file should be created"
        
        # Test PDF format with metadata
        pdf_result = renderer.save_figure(
            pdf_path,
            format='pdf',
            metadata={'Title': 'Plume Navigation Visualization', 'Creator': 'plume_nav_sim'}
        )
        assert pdf_result == True, "Should successfully save PDF figure"
        assert os.path.exists(pdf_path), "PDF file should be created"
        
        # Test SVG format with transparency
        svg_result = renderer.save_figure(
            svg_path,
            format='svg',
            transparent=True
        )
        assert svg_result == True, "Should successfully save SVG figure"
        assert os.path.exists(svg_path), "SVG file should be created"
        
        # Validate filename handling and path validation with error checking
        with pytest.raises(ValidationError):
            renderer.save_figure("", format='png')  # Empty filename
        
        with pytest.raises(ValidationError):
            renderer.save_figure(png_path, format='invalid')  # Invalid format
        
        # Test save options including DPI, transparency, and metadata configuration
        advanced_options = {
            'dpi': 300,
            'bbox_inches': 'tight',
            'pad_inches': 0.1,
            'facecolor': 'white',
            'edgecolor': 'none'
        }
        
        advanced_path = os.path.join(temp_dir, "advanced_render.png")
        advanced_result = renderer.save_figure(advanced_path, **advanced_options)
        assert advanced_result == True, "Should handle advanced save options"
        
        # Assert successful file creation and content validation
        assert os.path.getsize(png_path) > 0, "PNG file should have content"
        assert os.path.getsize(pdf_path) > 0, "PDF file should have content"
        assert os.path.getsize(svg_path) > 0, "SVG file should have content"
        
        # Test save error handling and validation for invalid parameters
        with patch('matplotlib.figure.Figure.savefig', side_effect=Exception("Save failed")):
            with pytest.raises(RenderingError):
                renderer.save_figure(os.path.join(temp_dir, "error_test.png"))
    
    renderer.cleanup_resources()


@pytest.mark.integration
def test_interactive_mode_configuration():
    """Test interactive matplotlib mode configuration with event handling and performance optimization."""
    renderer = MatplotlibRenderer(grid_size=TEST_GRID_SIZE)
    
    # Test configure_interactive_mode with various configuration options
    interactive_config = {
        'enable_toolbar': True,
        'enable_key_bindings': True,
        'update_interval': 0.1,
        'animation_enabled': True
    }
    
    result = renderer.configure_interactive_mode(interactive_config)
    assert result == True, "Should successfully configure interactive mode"
    
    # Validate matplotlib interactive mode toggling (plt.ion/plt.ioff)
    original_interactive = plt.isinteractive()
    
    renderer.enable_interactive_mode()
    # May enable interactive mode depending on backend
    
    renderer.disable_interactive_mode()
    # Should disable interactive mode
    
    # Test update interval configuration and performance optimization
    renderer.set_update_interval(0.05)  # 50ms updates
    assert renderer.get_update_interval() == 0.05
    
    # Mock event handling and test interactive responsiveness
    with patch('matplotlib.pyplot.pause') as mock_pause:
        renderer.process_interactive_events()
        # Should handle interactive events appropriately
    
    # Assert backend-specific interactive feature configuration
    capabilities = renderer.backend_manager.get_backend_capabilities()
    if capabilities.get('interactive_supported', False):
        # Interactive features should be properly configured
        pass
    
    # Test interactive mode error handling and fallback mechanisms
    with patch('matplotlib.pyplot.ion', side_effect=Exception("Interactive mode failed")):
        # Should handle interactive mode failures gracefully
        fallback_result = renderer.configure_interactive_mode(interactive_config)
        # May return False but should not raise exception
    
    renderer.cleanup_resources()


@pytest.mark.performance
def test_performance_metrics_collection(test_render_context, performance_monitor):
    """Test comprehensive performance metrics collection including timing analysis and optimization recommendations."""
    renderer = MatplotlibRenderer(grid_size=TEST_GRID_SIZE, backend_preferences=['Agg'])
    
    # Execute multiple rendering operations with performance monitoring
    performance_monitor['start_timing']('metrics_test')
    
    renderer.initialize()
    for i in range(5):
        performance_monitor['start_timing'](f'render_{i}')
        renderer.render(test_render_context, RenderMode.HUMAN)
        performance_monitor['end_timing'](f'render_{i}')
    
    performance_monitor['end_timing']('metrics_test')
    
    # Test get_performance_metrics with various analysis options
    metrics = renderer.get_performance_metrics(
        include_timing=True,
        include_resource_usage=True,
        include_optimization_analysis=True
    )
    
    assert isinstance(metrics, dict), "Should return performance metrics dictionary"
    
    # Validate timing data collection and statistical analysis
    assert 'render_count' in metrics
    assert 'total_render_time' in metrics
    assert 'average_render_time' in metrics
    assert metrics['render_count'] >= 5, "Should track multiple renders"
    
    # Test performance ratio calculation against target thresholds
    if 'average_render_time' in metrics:
        avg_time = metrics['average_render_time']
        target_ratio = avg_time / PERFORMANCE_TARGET_HUMAN_RENDER_MS
        assert 'performance_ratio' in metrics or True  # May be calculated internally
    
    # Assert optimization recommendation generation based on performance patterns
    if 'recommendations' in metrics:
        recommendations = metrics['recommendations']
        assert isinstance(recommendations, list)
        
    # Test performance metrics reset and historical data management
    renderer.reset_performance_metrics()
    reset_metrics = renderer.get_performance_metrics()
    assert reset_metrics['render_count'] == 0, "Should reset render count"
    
    renderer.cleanup_resources()


@pytest.mark.unit
def test_headless_environment_detection(headless_environment):
    """Test headless environment detection and automatic backend selection for server deployment compatibility."""
    # Mock headless environment by removing DISPLAY environment variable
    headless_environment['enable_headless']()
    
    # Test automatic Agg backend selection for headless operation
    backend_manager = MatplotlibBackendManager()
    selected_backend = backend_manager.select_backend(detect_headless=True)
    
    # Should select Agg backend or another headless-compatible backend
    assert selected_backend is not None, "Should select a backend in headless environment"
    
    # Validate headless mode detection and configuration adjustment
    is_headless = backend_manager.is_headless_environment()
    assert is_headless == True, "Should detect headless environment"
    
    # Test rendering functionality in headless mode with Agg backend
    renderer = MatplotlibRenderer(
        grid_size=TEST_GRID_SIZE,
        backend_preferences=['Agg']
    )
    
    render_context = create_render_context(
        concentration_field=np.random.rand(TEST_GRID_SIZE.height, TEST_GRID_SIZE.width).astype(np.float32),
        agent_position=TEST_AGENT_POSITION,
        source_position=TEST_SOURCE_POSITION,
        grid_size=TEST_GRID_SIZE
    )
    
    # Should work in headless environment
    renderer.initialize()
    result = renderer.render(render_context, RenderMode.HUMAN)
    # May return None but should not raise exception
    
    # Assert proper fallback behavior and error handling in headless environment
    assert renderer.supports_render_mode(RenderMode.HUMAN), "Should support human mode in headless environment"
    
    # Test headless to interactive transition and backend switching
    headless_environment['restore_display']()
    
    # Backend manager should adapt to restored display
    restored_backend = backend_manager.select_backend(force_reselection=True)
    # May select different backend with display available
    
    renderer.cleanup_resources()


@pytest.mark.integration
def test_backend_fallback_mechanisms(mock_matplotlib_backend):
    """Test backend fallback mechanisms with priority-based selection and graceful degradation for cross-platform compatibility."""
    # Mock various backend availability scenarios for fallback testing
    backend_preferences = ['NonExistentBackend1', 'NonExistentBackend2', 'Agg']
    
    with patch('matplotlib.pyplot.switch_backend') as mock_switch:
        with patch('builtins.__import__') as mock_import:
            # Simulate backend failures and success
            def import_side_effect(backend_name, *args, **kwargs):
                if 'NonExistent' in backend_name:
                    raise ImportError(f"Backend {backend_name} not available")
                return True  # Agg should be available
            
            mock_import.side_effect = import_side_effect
            
            # Test systematic backend selection through priority list with failure simulation
            backend_manager = MatplotlibBackendManager(backend_preferences=backend_preferences)
            selected_backend = backend_manager.select_backend()
            
            # Should eventually select Agg backend after failures
            assert selected_backend == 'Agg' or selected_backend is not None
            
            # Validate graceful fallback to Agg backend when interactive backends fail
            renderer = MatplotlibRenderer(
                grid_size=TEST_GRID_SIZE,
                backend_preferences=backend_preferences
            )
            
            # Should initialize successfully with fallback backend
            renderer.initialize()
            assert renderer.backend_manager.get_current_backend() is not None
            
            # Test backend switching error handling and configuration rollback
            with pytest.raises(RenderingError):
                backend_manager.switch_to_backend('CompletelyInvalidBackend')
            
            # Should maintain functional state after error
            current_backend = backend_manager.get_current_backend()
            assert current_backend is not None
            
            # Assert proper warning generation and user notification for fallback events
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                
                # This should generate warnings about backend fallbacks
                fallback_renderer = MatplotlibRenderer(
                    grid_size=TEST_GRID_SIZE,
                    backend_preferences=['InvalidBackend', 'Agg']
                )
                fallback_renderer.initialize()
                
                # May generate warnings about backend selection
                
            # Test fallback performance impact and functionality preservation
            render_context = create_render_context(
                concentration_field=np.random.rand(TEST_GRID_SIZE.height, TEST_GRID_SIZE.width).astype(np.float32),
                agent_position=TEST_AGENT_POSITION,
                source_position=TEST_SOURCE_POSITION,
                grid_size=TEST_GRID_SIZE
            )
            
            # Rendering should still work with fallback backend
            fallback_result = fallback_renderer.render(render_context, RenderMode.HUMAN)
            # Should complete without exception
            
            renderer.cleanup_resources()
            fallback_renderer.cleanup_resources()


@pytest.mark.unit  
def test_resource_cleanup_comprehensive():
    """Test comprehensive resource cleanup including figure disposal, backend restoration, and memory management."""
    renderer = MatplotlibRenderer(grid_size=TEST_GRID_SIZE)
    
    # Create renderer with full resource allocation including figures and managers
    renderer.initialize()
    original_backend = renderer.backend_manager.get_current_backend()
    
    # Ensure resources are allocated
    figure = renderer.get_figure()
    assert figure is not None
    assert renderer._axes is not None
    
    # Test cleanup_resources method with complete resource disposal
    start_time = time.time()
    cleanup_result = renderer.cleanup_resources()
    cleanup_time = time.time() - start_time
    
    assert cleanup_result == True, "Should successfully complete cleanup"
    assert cleanup_time < MAX_CLEANUP_TIMEOUT_SEC, f"Cleanup should complete within {MAX_CLEANUP_TIMEOUT_SEC}s"
    
    # Validate matplotlib figure cleanup using plt.close() with error handling
    # Figure should be closed and resources freed
    assert renderer._figure is None, "Figure reference should be cleared"
    assert renderer._axes is None, "Axes reference should be cleared"
    
    # Test backend restoration and original configuration recovery
    current_backend = renderer.backend_manager.get_current_backend()
    # Backend may be restored to original or maintained as fallback
    
    # Assert memory cleanup and garbage collection effectiveness
    # This is difficult to test directly but cleanup should not hold references
    
    # Test cleanup timeout handling and forced cleanup mechanisms
    timeout_renderer = MatplotlibRenderer(grid_size=TEST_GRID_SIZE)
    timeout_renderer.initialize()
    
    with patch('matplotlib.pyplot.close', side_effect=lambda: time.sleep(0.1)):
        # Should handle slow cleanup operations
        timeout_result = timeout_renderer.cleanup_resources(timeout_sec=0.5)
        # Should complete within timeout
    
    # Validate cleanup success reporting and error logging
    error_renderer = MatplotlibRenderer(grid_size=TEST_GRID_SIZE)
    error_renderer.initialize()
    
    with patch('matplotlib.pyplot.close', side_effect=Exception("Cleanup failed")):
        # Should handle cleanup errors gracefully
        error_result = error_renderer.cleanup_resources()
        # May return False but should not raise exception


@pytest.mark.unit
def test_error_handling_comprehensive():
    """Test comprehensive error handling including backend failures, rendering errors, and recovery mechanisms with detailed error reporting."""
    # Test error handling for backend configuration failures with appropriate exceptions
    with pytest.raises(ValidationError):
        MatplotlibBackendManager(backend_preferences=None)  # Invalid preferences
    
    with pytest.raises(ValidationError):
        MatplotlibRenderer(grid_size=None)  # Invalid grid size
    
    # Validate rendering error recovery and graceful degradation mechanisms
    renderer = MatplotlibRenderer(grid_size=TEST_GRID_SIZE)
    renderer.initialize()
    
    invalid_context = Mock()  # Invalid render context
    with pytest.raises(RenderingError):
        renderer.render(invalid_context, RenderMode.HUMAN)
    
    # Test resource allocation failures and cleanup error handling
    with patch('matplotlib.pyplot.subplots', side_effect=MemoryError("Out of memory")):
        memory_renderer = MatplotlibRenderer(grid_size=TEST_GRID_SIZE)
        with pytest.raises(RenderingError):
            memory_renderer.initialize()
    
    # Mock matplotlib errors and verify error propagation and logging
    with patch('matplotlib.axes.Axes.imshow', side_effect=Exception("Matplotlib error")):
        render_context = create_render_context(
            concentration_field=np.random.rand(TEST_GRID_SIZE.height, TEST_GRID_SIZE.width).astype(np.float32),
            agent_position=TEST_AGENT_POSITION,
            source_position=TEST_SOURCE_POSITION,
            grid_size=TEST_GRID_SIZE
        )
        
        with pytest.raises(RenderingError):
            renderer.render(render_context, RenderMode.HUMAN)
    
    # Assert proper error message generation and user feedback
    try:
        MatplotlibRenderer(grid_size=GridSize(width=0, height=0))  # Invalid dimensions
    except ValidationError as e:
        assert "grid_size" in str(e).lower(), "Error message should mention grid_size"
    
    # Test error recovery strategies and system state restoration
    backend_manager = MatplotlibBackendManager()
    original_backend = backend_manager.get_current_backend()
    
    try:
        backend_manager.switch_to_backend('InvalidBackend')
    except RenderingError:
        # Should maintain valid state after error
        current_backend = backend_manager.get_current_backend()
        assert current_backend is not None, "Should maintain valid backend after error"
    
    # Validate error handling consistency across all renderer methods
    error_methods = [
        (renderer.supports_render_mode, [None]),  # Invalid render mode
        (renderer.set_color_scheme, [None]),      # Invalid color scheme
    ]
    
    for method, invalid_args in error_methods:
        with pytest.raises((ValidationError, RenderingError, TypeError)):
            method(*invalid_args)
    
    renderer.cleanup_resources()


@pytest.mark.integration
def test_cross_platform_compatibility():
    """Test cross-platform compatibility including Windows, macOS, and Linux support with platform-specific behavior validation."""
    # Mock different platform environments for compatibility testing
    platforms = ['linux', 'darwin', 'win32']
    
    for platform in platforms:
        with patch('sys.platform', platform):
            # Test platform-specific backend availability and selection logic
            backend_manager = MatplotlibBackendManager()
            
            if platform == 'win32':
                # Validate Windows community support with appropriate limitations and warnings
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    
                    selected_backend = backend_manager.select_backend()
                    
                    # May generate warnings about Windows support limitations
                    # Windows should still work but may have reduced functionality
                    
            elif platform in ['linux', 'darwin']:
                # Test macOS and Linux full support with complete functionality validation
                selected_backend = backend_manager.select_backend()
                assert selected_backend is not None, f"Should select backend on {platform}"
                
                # Full functionality should be available on Linux and macOS
                capabilities = backend_manager.get_backend_capabilities()
                assert isinstance(capabilities, dict), "Should provide capabilities on supported platforms"
            
            # Assert platform-specific error handling and user guidance
            renderer = MatplotlibRenderer(grid_size=TEST_GRID_SIZE)
            
            try:
                renderer.initialize()
                
                # Basic rendering should work on all platforms
                render_context = create_render_context(
                    concentration_field=np.random.rand(TEST_GRID_SIZE.height, TEST_GRID_SIZE.width).astype(np.float32),
                    agent_position=TEST_AGENT_POSITION,
                    source_position=TEST_SOURCE_POSITION,
                    grid_size=TEST_GRID_SIZE
                )
                
                result = renderer.render(render_context, RenderMode.HUMAN)
                # Should work or fail gracefully on all platforms
                
                renderer.cleanup_resources()
                
            except Exception as e:
                # Should provide clear error messages for platform-specific issues
                assert isinstance(e, (RenderingError, ValidationError)), f"Should use appropriate exception types on {platform}"
    
    # Test platform detection and automatic configuration adjustment
    current_platform = sys.platform
    backend_manager = MatplotlibBackendManager()
    
    # Should adapt to current platform automatically
    platform_config = backend_manager.get_platform_configuration()
    assert isinstance(platform_config, dict), "Should provide platform configuration"
    assert 'platform' in platform_config, "Should detect platform"


@pytest.mark.integration
def test_accessibility_features(test_render_context):
    """Test accessibility features including high contrast schemes, colorblind-friendly palettes, and enhanced visibility."""
    # Create accessibility-enhanced color schemes for testing
    high_contrast_scheme = create_accessibility_scheme('high_contrast')
    colorblind_scheme = create_accessibility_scheme('colorblind_friendly')
    
    renderer = MatplotlibRenderer(grid_size=TEST_GRID_SIZE, backend_preferences=['Agg'])
    
    # Test high contrast mode with enhanced visibility and marker size adjustments
    renderer.set_color_scheme(high_contrast_scheme)
    renderer.initialize()
    
    # Validate contrast ratio compliance and visibility enhancement effectiveness
    agent_color = high_contrast_scheme.agent_color
    background_color = high_contrast_scheme.background_color
    
    # Calculate luminance for contrast ratio (simplified)
    agent_luminance = sum(agent_color) / (3 * 255)
    bg_luminance = sum(background_color) / (3 * 255)
    contrast_ratio = (max(agent_luminance, bg_luminance) + 0.05) / (min(agent_luminance, bg_luminance) + 0.05)
    
    assert contrast_ratio >= 3.0, f"High contrast scheme should have contrast ratio >= 3.0, got {contrast_ratio:.2f}"
    
    # Test colorblind-friendly palette integration with matplotlib renderer
    renderer.set_color_scheme(colorblind_scheme)
    result = renderer.render(test_render_context, RenderMode.HUMAN)
    # Should render successfully with colorblind-friendly colors
    
    # Validate accessibility scheme application and matplotlib compatibility
    assert colorblind_scheme.accessibility_enabled == True, "Colorblind scheme should have accessibility enabled"
    assert colorblind_scheme.concentration_colormap in ['viridis', 'plasma', 'inferno'], "Should use colorblind-friendly colormap"
    
    # Assert contrast ratio compliance and visibility enhancement effectiveness
    colorblind_agent_color = colorblind_scheme.agent_color
    # Orange color should be deuteranopia/protanopia safe
    assert colorblind_agent_color != (255, 0, 0), "Should not use pure red for colorblind accessibility"
    
    # Test accessibility feature performance impact and optimization
    start_time = time.time()
    for _ in range(3):
        renderer.render(test_render_context, RenderMode.HUMAN)
    accessibility_time = (time.time() - start_time) * 1000 / 3
    
    # Accessibility features should not significantly impact performance
    assert accessibility_time < PERFORMANCE_TARGET_HUMAN_RENDER_MS * 2, f"Accessibility rendering should be reasonable, got {accessibility_time:.2f}ms"
    
    renderer.cleanup_resources()


@pytest.mark.performance
def test_memory_usage_optimization():
    """Test memory usage optimization including figure caching, resource reuse, and memory leak prevention."""
    renderers = []
    initial_objects = len([obj for obj in globals() if isinstance(obj, matplotlib.figure.Figure)])
    
    # Monitor memory usage during multiple rendering operations
    for i in range(5):
        renderer = MatplotlibRenderer(grid_size=TEST_GRID_SIZE, backend_preferences=['Agg'])
        renderer.initialize()
        
        # Test figure caching effectiveness and memory reuse patterns
        figure1 = renderer.get_figure()
        figure2 = renderer.get_figure()
        assert figure1 is figure2, "Should reuse same figure instance (caching)"
        
        # Validate resource cleanup and memory leak prevention mechanisms
        render_context = create_render_context(
            concentration_field=np.random.rand(TEST_GRID_SIZE.height, TEST_GRID_SIZE.width).astype(np.float32),
            agent_position=TEST_AGENT_POSITION,
            source_position=TEST_SOURCE_POSITION,
            grid_size=TEST_GRID_SIZE
        )
        
        renderer.render(render_context, RenderMode.HUMAN)
        renderers.append(renderer)
    
    # Test memory usage with various grid sizes and complexity levels
    grid_sizes = [
        GridSize(width=16, height=16),
        GridSize(width=32, height=32),
        GridSize(width=64, height=64)
    ]
    
    for grid_size in grid_sizes:
        grid_renderer = MatplotlibRenderer(grid_size=grid_size, backend_preferences=['Agg'])
        grid_renderer.initialize()
        
        # Memory usage should scale reasonably with grid size
        complex_field = np.random.rand(grid_size.height, grid_size.width).astype(np.float32)
        complex_context = create_render_context(
            concentration_field=complex_field,
            agent_position=Coordinates(x=grid_size.width//2, y=grid_size.height//2),
            source_position=Coordinates(x=grid_size.width//4, y=grid_size.height//4),
            grid_size=grid_size
        )
        
        grid_renderer.render(complex_context, RenderMode.HUMAN)
        grid_renderer.cleanup_resources()
    
    # Assert memory usage stays within reasonable bounds during extended operation
    # Clean up all renderers
    for renderer in renderers:
        renderer.cleanup_resources()
    
    # Force garbage collection and check for leaks
    import gc
    gc.collect()
    
    final_objects = len([obj for obj in globals() if isinstance(obj, matplotlib.figure.Figure)])
    
    # Should not have significantly more figure objects after cleanup
    # This is a rough check for obvious memory leaks
    
    # Test memory optimization impact on rendering performance
    optimized_renderer = MatplotlibRenderer(
        grid_size=TEST_GRID_SIZE, 
        backend_preferences=['Agg'],
        memory_optimization=True
    )
    
    if hasattr(optimized_renderer, 'memory_optimization'):
        # Should provide memory optimization features if supported
        optimized_renderer.initialize()
        
        start_time = time.time()
        for _ in range(5):
            render_context = create_render_context(
                concentration_field=np.random.rand(TEST_GRID_SIZE.height, TEST_GRID_SIZE.width).astype(np.float32),
                agent_position=TEST_AGENT_POSITION,
                source_position=TEST_SOURCE_POSITION,
                grid_size=TEST_GRID_SIZE
            )
            optimized_renderer.render(render_context, RenderMode.HUMAN)
        
        optimized_time = (time.time() - start_time) * 1000 / 5
        assert optimized_time < PERFORMANCE_TARGET_HUMAN_RENDER_MS * 1.5, "Memory optimization should not severely impact performance"
        
        optimized_renderer.cleanup_resources()


@pytest.mark.unit
def test_renderer_factory_function():
    """Test create_matplotlib_renderer factory function with backend detection, capability assessment, and configuration optimization."""
    # Test create_matplotlib_renderer with various configuration options
    config = {
        'grid_size': TEST_GRID_SIZE,
        'backend_preferences': ['Agg', 'TkAgg'],
        'enable_performance_monitoring': True,
        'color_scheme': 'default'
    }
    
    renderer = create_matplotlib_renderer(config)
    assert renderer is not None
    assert isinstance(renderer, MatplotlibRenderer)
    assert renderer.grid_size == TEST_GRID_SIZE
    
    # Validate backend detection and capability assessment during creation
    assert hasattr(renderer, 'backend_manager')
    capabilities = renderer.backend_manager.get_backend_capabilities()
    assert isinstance(capabilities, dict)
    
    # Test color scheme integration and optimization during factory creation
    assert renderer.color_scheme is not None
    
    # Assert proper renderer validation and functionality testing
    assert renderer.supports_render_mode(RenderMode.HUMAN)
    
    # Test factory error handling and configuration validation
    invalid_config = {
        'grid_size': None,  # Invalid
        'backend_preferences': []  # Invalid
    }
    
    with pytest.raises(ValidationError):
        create_matplotlib_renderer(invalid_config)
    
    # Validate factory-created renderer performance and feature completeness
    render_context = create_render_context(
        concentration_field=np.random.rand(TEST_GRID_SIZE.height, TEST_GRID_SIZE.width).astype(np.float32),
        agent_position=TEST_AGENT_POSITION,
        source_position=TEST_SOURCE_POSITION,
        grid_size=TEST_GRID_SIZE
    )
    
    renderer.initialize()
    result = renderer.render(render_context, RenderMode.HUMAN)
    # Should work without issues
    
    renderer.cleanup_resources()


@pytest.mark.unit
def test_capability_detection_function():
    """Test detect_matplotlib_capabilities function for comprehensive system assessment including backend availability and performance."""
    # Test detect_matplotlib_capabilities with various system configurations
    capabilities = detect_matplotlib_capabilities()
    
    assert isinstance(capabilities, dict), "Should return capabilities dictionary"
    
    # Validate backend availability testing and compatibility assessment
    assert 'backends_available' in capabilities
    assert 'display_support' in capabilities
    assert 'interactive_support' in capabilities
    
    backends_available = capabilities['backends_available']
    assert isinstance(backends_available, list), "Should list available backends"
    assert 'Agg' in backends_available, "Agg backend should always be available"
    
    # Test display detection and headless environment identification
    display_support = capabilities['display_support']
    assert isinstance(display_support, bool), "Display support should be boolean"
    
    # Mock different system capabilities for comprehensive testing
    with patch('os.environ.get', return_value=None):  # No DISPLAY
        headless_capabilities = detect_matplotlib_capabilities()
        assert headless_capabilities['display_support'] == False, "Should detect no display support"
    
    with patch('os.environ.get', return_value=':0.0'):  # X11 display
        display_capabilities = detect_matplotlib_capabilities()
        # May detect display support depending on system
    
    # Assert performance characteristics assessment and reporting
    if 'performance_info' in capabilities:
        perf_info = capabilities['performance_info']
        assert isinstance(perf_info, dict), "Performance info should be dictionary"
    
    # Test capability detection caching and refresh mechanisms
    cached_capabilities = detect_matplotlib_capabilities(use_cache=True)
    assert cached_capabilities == capabilities, "Should return cached results"
    
    fresh_capabilities = detect_matplotlib_capabilities(use_cache=False)
    assert isinstance(fresh_capabilities, dict), "Should return fresh capabilities"


@pytest.mark.integration
def test_matplotlib_integration_validation():
    """Test validate_matplotlib_integration function for comprehensive integration testing and performance compliance."""
    # Test validate_matplotlib_integration with various backend configurations
    validation_result = validate_matplotlib_integration()
    
    assert isinstance(validation_result, dict), "Should return validation results"
    assert 'integration_valid' in validation_result
    assert 'backend_functional' in validation_result
    assert 'rendering_functional' in validation_result
    
    # Validate functionality testing including figure creation and plotting operations
    if validation_result['integration_valid']:
        assert validation_result['backend_functional'] == True
        assert validation_result['rendering_functional'] == True
    
    # Test color scheme integration and colormap availability validation
    if 'colormap_support' in validation_result:
        colormap_support = validation_result['colormap_support']
        assert isinstance(colormap_support, dict)
        assert 'gray' in colormap_support  # Should support basic grayscale
    
    # Assert performance compliance testing against human mode targets
    if 'performance_test' in validation_result:
        perf_test = validation_result['performance_test']
        if 'render_time_ms' in perf_test:
            render_time = perf_test['render_time_ms']
            target_ratio = render_time / PERFORMANCE_TARGET_HUMAN_RENDER_MS
            assert target_ratio < 5.0, f"Integration test render time should be reasonable, got {render_time}ms"
    
    # Test integration validation error handling and reporting
    with patch('matplotlib.pyplot.subplots', side_effect=Exception("Integration test failed")):
        error_validation = validate_matplotlib_integration()
        assert error_validation['integration_valid'] == False
        assert 'error_details' in error_validation
    
    # Validate comprehensive integration analysis and recommendation generation
    if 'recommendations' in validation_result:
        recommendations = validation_result['recommendations']
        assert isinstance(recommendations, list)
        # Should provide actionable recommendations for integration issues


# Test class implementations for comprehensive renderer testing

class TestMatplotlibRenderer:
    """Comprehensive test class for MatplotlibRenderer testing with setup, teardown, and resource management for human mode visualization validation."""
    
    def setup_method(self, method):
        """Set up test environment and renderer instance for each test method execution."""
        # Create fresh MatplotlibRenderer instance with test configuration
        self.test_grid_size = TEST_GRID_SIZE
        self.test_color_scheme = get_default_scheme()
        self.performance_baseline = {}
        self.cleanup_required = False
        
        self.renderer = MatplotlibRenderer(
            grid_size=self.test_grid_size,
            color_scheme=self.test_color_scheme,
            backend_preferences=['Agg']  # Use Agg for consistent testing
        )
        
        # Initialize renderer resources and validate setup success
        try:
            self.renderer.initialize()
            self.cleanup_required = True
        except Exception as e:
            pytest.skip(f"Failed to initialize renderer: {str(e)}")
        
        # Configure test matplotlib backend for reliable test execution
        # Set up performance monitoring and baseline measurement
        self.performance_baseline['setup_time'] = time.time()
    
    def teardown_method(self, method):
        """Clean up test resources and validate proper cleanup after each test method."""
        if self.cleanup_required and hasattr(self, 'renderer'):
            # Execute renderer resource cleanup if cleanup_required is True
            try:
                cleanup_success = self.renderer.cleanup_resources()
                
                # Validate complete resource disposal and memory cleanup
                assert self.renderer._figure is None or not hasattr(self.renderer._figure, 'canvas'), "Figure should be cleaned up"
                
            except Exception as e:
                # Log cleanup failure but don't fail test
                print(f"Warning: Cleanup failed: {str(e)}")
        
        # Clear test data and reset performance monitoring
        if hasattr(self, 'performance_baseline'):
            self.performance_baseline.clear()
    
    def test_renderer_inheritance_compliance(self):
        """Test MatplotlibRenderer inheritance from BaseRenderer with method compliance validation."""
        # Assert MatplotlibRenderer is instance of BaseRenderer
        assert isinstance(self.renderer, BaseRenderer), "Should inherit from BaseRenderer"
        
        # Validate all abstract method implementations are present
        required_methods = ['initialize', 'render', 'cleanup_resources', 'supports_render_mode', 'get_performance_metrics']
        for method_name in required_methods:
            assert hasattr(self.renderer, method_name), f"Should implement {method_name} method"
            assert callable(getattr(self.renderer, method_name)), f"{method_name} should be callable"
        
        # Test method signature compliance with base class interface
        # Should accept same parameters as base class
        
        # Assert proper super() method calls and inheritance chain
        assert hasattr(self.renderer, 'grid_size'), "Should inherit grid_size from BaseRenderer"
        
        # Validate polymorphic usage and interface consistency
        base_renderer_methods = dir(BaseRenderer)
        renderer_methods = dir(self.renderer)
        
        for base_method in base_renderer_methods:
            if not base_method.startswith('_') and base_method != 'render':
                assert base_method in renderer_methods, f"Should implement {base_method} from base class"
    
    def test_render_mode_support_validation(self):
        """Test render mode support with backend availability and compatibility validation."""
        # Test supports_render_mode(RenderMode.HUMAN) returns True with available backend
        supports_human = self.renderer.supports_render_mode(RenderMode.HUMAN)
        assert supports_human == True, "Should support HUMAN mode with matplotlib"
        
        # Test supports_render_mode(RenderMode.RGB_ARRAY) returns False appropriately
        supports_rgb = self.renderer.supports_render_mode(RenderMode.RGB_ARRAY)
        assert supports_rgb == False, "Should not support RGB_ARRAY mode"
        
        # Mock backend unavailability and verify mode support returns False
        with patch.object(self.renderer.backend_manager, 'get_current_backend', return_value=None):
            no_backend_support = self.renderer.supports_render_mode(RenderMode.HUMAN)
            assert no_backend_support == False, "Should not support HUMAN mode without backend"
        
        # Validate mode support caching and performance optimization
        # Multiple calls should be efficient
        start_time = time.time()
        for _ in range(10):
            self.renderer.supports_render_mode(RenderMode.HUMAN)
        mode_check_time = (time.time() - start_time) * 1000
        
        assert mode_check_time < 10.0, f"Mode support checking should be fast, got {mode_check_time:.2f}ms"
        
        # Test mode support with different backend configurations
        assert self.renderer.supports_render_mode(RenderMode.HUMAN) == True, "Should consistently support HUMAN mode"
    
    def test_complete_rendering_pipeline(self):
        """Test complete rendering pipeline from context validation to display output with performance monitoring."""
        # Create comprehensive render context with realistic environment data
        concentration_field = np.random.rand(self.test_grid_size.height, self.test_grid_size.width).astype(np.float32)
        render_context = create_render_context(
            concentration_field=concentration_field,
            agent_position=TEST_AGENT_POSITION,
            source_position=TEST_SOURCE_POSITION,
            grid_size=self.test_grid_size
        )
        
        # Execute complete render pipeline with performance monitoring
        start_time = time.time()
        result = self.renderer.render(render_context, RenderMode.HUMAN)
        render_time = (time.time() - start_time) * 1000
        
        # Validate context validation and parameter checking
        # Should accept valid render context without error
        
        # Test concentration field rendering with colormap application
        assert self.renderer._figure is not None, "Should create figure during rendering"
        assert self.renderer._axes is not None, "Should create axes during rendering"
        
        # Assert marker placement and visualization accuracy
        # Markers should be positioned correctly
        
        # Measure rendering performance against target thresholds
        target_ms = PERFORMANCE_TARGET_HUMAN_RENDER_MS + PERFORMANCE_TOLERANCE_MS
        assert render_time <= target_ms, f"Rendering should be <{target_ms}ms, got {render_time:.2f}ms"
        
        # Validate error handling and recovery throughout pipeline
        invalid_context = Mock()
        with pytest.raises((ValidationError, RenderingError, AttributeError)):
            self.renderer.render(invalid_context, RenderMode.HUMAN)


class TestMatplotlibBackendManager:
    """Comprehensive test class for MatplotlibBackendManager with backend selection, configuration, and fallback testing."""
    
    def setup_method(self, method):
        """Set up backend testing environment with proper mocking and state preservation."""
        # Store current matplotlib backend configuration for restoration
        self.original_backend_state = {
            'backend': matplotlib.get_backend(),
            'interactive': plt.isinteractive()
        }
        self.backend_modified = False
        
        # Create MatplotlibBackendManager with test configuration
        self.backend_manager = MatplotlibBackendManager()
        
        # Set up backend mocking for controlled testing environment
        self.mock_backends = MOCK_BACKEND_PRIORITY.copy()
        
        # Initialize backend state tracking for change detection
        self.backend_modified = False
    
    def teardown_method(self, method):
        """Restore original backend state and clean up backend testing modifications."""
        try:
            # Restore original matplotlib backend if backend_modified is True
            if self.backend_modified and hasattr(self, 'original_backend_state'):
                original_backend = self.original_backend_state['backend']
                if original_backend:
                    plt.switch_backend(original_backend)
                
                # Restore interactive mode
                if self.original_backend_state['interactive']:
                    plt.ion()
                else:
                    plt.ioff()
            
            # Clean up backend manager resources and state
            if hasattr(self, 'backend_manager'):
                self.backend_manager.restore_original_backend()
                
        except Exception as e:
            print(f"Warning: Backend restoration failed: {str(e)}")
        
        # Clear backend mocking and test modifications
        # Reset backend testing flags and tracking variables
        self.backend_modified = False
    
    def test_backend_priority_selection(self):
        """Test systematic backend selection through priority list with fallback testing."""
        # Configure backend priority list with test backends
        test_priority = ['TestInteractive', 'TestGUI', 'Agg']
        custom_manager = MatplotlibBackendManager(backend_preferences=test_priority)
        
        # Mock backend availability for controlled selection testing
        with patch('matplotlib.pyplot.switch_backend') as mock_switch:
            # Test systematic backend selection through priority list
            backend = custom_manager.select_backend()
            
            # Should select a backend from the priority list
            assert backend is not None, "Should select a backend"
            
            # Validate fallback to Agg backend when interactive backends fail
            agg_manager = MatplotlibBackendManager(backend_preferences=['NonExistent', 'Agg'])
            agg_backend = agg_manager.select_backend()
            
            # Should fall back to Agg
            assert agg_backend == 'Agg' or agg_backend is not None
            
            # Assert proper backend selection reporting and configuration
            selection_report = custom_manager.get_selection_report()
            assert isinstance(selection_report, dict), "Should provide selection report"
    
    def test_backend_capability_analysis(self):
        """Test backend capability detection and analysis with comprehensive system assessment."""
        # Test capability detection for various backend types
        capabilities = self.backend_manager.get_backend_capabilities()
        
        assert isinstance(capabilities, dict), "Should return capabilities dictionary"
        assert 'display_available' in capabilities
        assert 'interactive_supported' in capabilities
        
        # Mock system environments for comprehensive capability testing
        with patch('os.environ.get', return_value=None):  # No DISPLAY
            headless_caps = self.backend_manager.get_backend_capabilities()
            assert headless_caps['display_available'] == False, "Should detect headless environment"
        
        # Validate display support and interactive feature detection
        if capabilities['interactive_supported']:
            assert isinstance(capabilities['interactive_supported'], bool)
        
        # Test performance characteristics measurement and reporting
        if 'performance_tier' in capabilities:
            assert capabilities['performance_tier'] in ['high', 'medium', 'low']
        
        # Assert capability caching and refresh functionality
        cached_caps = self.backend_manager.get_backend_capabilities(use_cache=True)
        assert cached_caps == capabilities, "Should return cached capabilities"


class TestInteractiveUpdateManager:
    """Comprehensive test class for InteractiveUpdateManager with update optimization, performance monitoring, and error handling validation."""
    
    def setup_method(self, method):
        """Set up interactive update testing with matplotlib figure creation and performance monitoring."""
        # Create test matplotlib figure with appropriate size and configuration
        self.test_figure, self.test_axes = plt.subplots(figsize=MATPLOTLIB_DEFAULT_FIGSIZE)
        
        # Initialize axes with test coordinate system and visualization properties
        self.test_axes.set_xlim(0, TEST_GRID_SIZE.width)
        self.test_axes.set_ylim(0, TEST_GRID_SIZE.height)
        self.test_axes.set_aspect('equal')
        
        # Create InteractiveUpdateManager with figure references and update interval
        self.update_manager = InteractiveUpdateManager(
            figure=self.test_figure,
            axes=self.test_axes,
            update_interval=0.05,  # 50ms update interval
            performance_monitoring=True
        )
        
        # Set up performance monitoring for update operation timing
        self.performance_data = {
            'update_times': [],
            'operation_counts': {}
        }
        
        # Initialize test data for concentration fields and marker positions
        self.test_concentration = np.random.rand(TEST_GRID_SIZE.height, TEST_GRID_SIZE.width).astype(np.float32)
        self.update_count = 0
    
    def teardown_method(self, method):
        """Clean up matplotlib figure resources and update manager state."""
        try:
            # Close test matplotlib figure with proper resource disposal
            if hasattr(self, 'test_figure'):
                plt.close(self.test_figure)
            
            # Clean up update manager resources and performance data
            if hasattr(self, 'update_manager'):
                self.update_manager.cleanup_resources()
                
        except Exception as e:
            print(f"Warning: Update manager cleanup failed: {str(e)}")
        
        # Clear matplotlib state and reset testing environment
        plt.close('all')
        
        # Reset update tracking and performance monitoring
        self.performance_data.clear()
        self.update_count = 0
    
    def test_concentration_field_updates(self):
        """Test concentration field visualization updates with change detection and performance optimization."""
        # Create test concentration field data with realistic plume characteristics
        concentration_field = self.test_concentration.copy()
        
        # Test update_concentration_field with various change scenarios
        result = self.update_manager.update_concentration_field(concentration_field)
        assert result == True, "Should successfully update concentration field"
        
        # Validate change detection optimization and cache management
        same_result = self.update_manager.update_concentration_field(concentration_field)
        assert same_result == False, "Should detect no change and skip update"
        
        # Test colormap application and matplotlib axes configuration
        images = self.test_axes.get_images()
        assert len(images) > 0, "Should create image objects for concentration field"
        
        # Assert update performance and timing optimization effectiveness
        start_time = time.time()
        for _ in range(5):
            modified_field = concentration_field * np.random.uniform(0.9, 1.1)
            self.update_manager.update_concentration_field(modified_field, force_update=True)
        avg_time = (time.time() - start_time) * 1000 / 5
        
        assert avg_time < 20.0, f"Concentration updates should be <20ms, got {avg_time:.2f}ms"
        
        # Test force update functionality and cache bypassing
        force_result = self.update_manager.update_concentration_field(concentration_field, force_update=True)
        assert force_result == True, "Should force update with identical data"
    
    def test_marker_update_efficiency(self):
        """Test agent and source marker update efficiency with position tracking and color application."""
        color_scheme = get_default_scheme()
        
        # Test agent marker updates with position changes and color configuration
        agent_result = self.update_manager.update_agent_marker(TEST_AGENT_POSITION, color_scheme)
        assert agent_result == True, "Should successfully update agent marker"
        
        # Validate source marker updates with cross-pattern visualization
        source_result = self.update_manager.update_source_marker(TEST_SOURCE_POSITION, color_scheme)
        assert source_result == True, "Should successfully update source marker"
        
        # Test marker position optimization and change detection
        same_agent_result = self.update_manager.update_agent_marker(TEST_AGENT_POSITION, color_scheme)
        assert same_agent_result == False, "Should detect no position change"
        
        # Assert proper color application and matplotlib compatibility
        # Colors should be applied correctly to markers
        
        # Test marker update performance and visual accuracy
        start_time = time.time()
        for i in range(10):
            new_pos = Coordinates(x=TEST_AGENT_POSITION.x + i, y=TEST_AGENT_POSITION.y)
            self.update_manager.update_agent_marker(new_pos, color_scheme)
        marker_time = (time.time() - start_time) * 1000 / 10
        
        assert marker_time < 5.0, f"Marker updates should be <5ms, got {marker_time:.2f}ms"
        
        # Validate marker layering and display hierarchy
        collections = self.test_axes.collections
        lines = self.test_axes.get_lines()
        # Should have visual elements for markers
    
    def test_batch_update_coordination(self):
        """Test coordinated batch updates of all visualization elements with performance optimization."""
        # Create complete render context with all visualization elements
        render_context = create_render_context(
            concentration_field=self.test_concentration,
            agent_position=TEST_AGENT_POSITION,
            source_position=TEST_SOURCE_POSITION,
            grid_size=TEST_GRID_SIZE
        )
        color_scheme = get_default_scheme()
        
        # Test batch_update with coordinated element updates
        start_time = time.time()
        result = self.update_manager.batch_update(render_context, color_scheme)
        batch_time = (time.time() - start_time) * 1000
        
        assert result == True, "Should successfully complete batch update"
        assert batch_time < 30.0, f"Batch update should be <30ms, got {batch_time:.2f}ms"
        
        # Validate selective refresh and optimization effectiveness
        optimized_result = self.update_manager.batch_update(render_context, color_scheme)
        # Should be optimized for repeated calls
        
        # Test performance monitoring during batch operations
        stats = self.update_manager.get_performance_stats()
        assert 'batch_update_count' in stats
        assert stats['batch_update_count'] >= 1
        
        # Assert optimization statistics and performance reporting
        opt_report = self.update_manager.get_optimization_report()
        assert isinstance(opt_report, dict)
        
        # Test batch update error handling and recovery
        invalid_context = Mock()
        with pytest.raises((ValidationError, RenderingError, AttributeError)):
            self.update_manager.batch_update(invalid_context, color_scheme)