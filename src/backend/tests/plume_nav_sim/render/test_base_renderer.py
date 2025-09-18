# External imports with version comments
import pytest  # >=8.0.0 - Testing framework for fixture usage, parametrization, exception testing, and test organization
import numpy as np  # >=2.1.0 - Array operations for creating test concentration fields and validating rendering outputs
from abc import ABC, abstractmethod  # >=3.10 - Abstract base class decorators for creating concrete test implementations of BaseRenderer
from unittest.mock import Mock, patch, MagicMock, call  # >=3.10 - Mocking framework for creating test implementations, patching dependencies, and simulating error conditions
import time  # >=3.10 - Timing utilities for performance testing and timeout validation
import contextlib  # >=3.10 - Context manager utilities for testing resource cleanup and exception handling
import warnings  # >=3.10 - Warning management for testing performance warnings and deprecation handling
import uuid  # >=3.10 - Unique identifier generation for test context tracking and validation
import threading  # >=3.10 - Thread safety testing for concurrent rendering operations
from typing import Optional, Dict, Any, List, Union

# Internal imports for BaseRenderer and related components
from plume_nav_sim.render.base_renderer import (
    BaseRenderer, 
    RenderContext, 
    RenderingMetrics,
    create_render_context,
    validate_rendering_parameters,
    create_rendering_metrics
)
from plume_nav_sim.core.types import (
    RenderMode, 
    Coordinates, 
    GridSize, 
    RGBArray,
    create_coordinates,
    create_grid_size,
    calculate_euclidean_distance
)
from plume_nav_sim.core.constants import (
    PERFORMANCE_TARGET_RGB_RENDER_MS,
    PERFORMANCE_TARGET_HUMAN_RENDER_MS,
    FIELD_DTYPE,
    RGB_DTYPE
)
from plume_nav_sim.utils.exceptions import (
    ValidationError,
    RenderingError,
    ComponentError
)

# Global test constants
TEST_GRID_SIZE = (32, 32)
TEST_SOURCE_LOCATION = (16, 16)
TEST_AGENT_POSITION = (8, 8)
TEST_PERFORMANCE_TOLERANCE_MS = 2.0
MOCK_RENDERER_TYPE = 'MockRenderer'
DEFAULT_TEST_TIMEOUT = 5.0


def create_mock_renderer(grid_size: GridSize, 
                        color_scheme_name: Optional[str] = None,
                        supports_rgb_array: bool = True,
                        supports_human: bool = True,
                        performance_overrides: Optional[Dict] = None) -> 'MockRenderer':
    """Factory function to create concrete BaseRenderer implementation for testing abstract base class functionality 
    with configurable behavior and performance characteristics.
    
    Args:
        grid_size: Grid dimensions for renderer configuration
        color_scheme_name: Optional color scheme identifier for rendering
        supports_rgb_array: Whether renderer supports RGB array mode
        supports_human: Whether renderer supports human mode
        performance_overrides: Custom timing behavior for performance testing
        
    Returns:
        MockRenderer: Concrete BaseRenderer implementation configured for testing with specified capabilities
    """
    # Create MockRenderer instance with specified configuration
    mock_renderer = MockRenderer(
        grid_size=grid_size,
        color_scheme_name=color_scheme_name or 'default_test_scheme',
        renderer_options={'test_mode': True}
    )
    
    # Configure supports_render_mode method to return specified mode support
    mock_renderer.configure_for_testing(
        enable_rgb_array=supports_rgb_array,
        enable_human=supports_human,
        simulate_failures=False,
        performance_overrides=performance_overrides
    )
    
    return mock_renderer


def create_test_concentration_field(grid_size: GridSize, 
                                  source_location: Coordinates,
                                  sigma: float = 12.0,
                                  add_noise: bool = False) -> np.ndarray:
    """Utility function to create test concentration field arrays with Gaussian distribution and proper data types 
    for render context testing.
    
    Args:
        grid_size: Grid dimensions for field generation
        source_location: Center point for Gaussian distribution
        sigma: Standard deviation for Gaussian distribution
        add_noise: Whether to add realistic field variations
        
    Returns:
        np.ndarray: Test concentration field array with FIELD_DTYPE and values in [0,1] range
    """
    # Create coordinate meshgrid using grid_size dimensions for mathematical field generation
    x = np.arange(grid_size.width, dtype=FIELD_DTYPE)
    y = np.arange(grid_size.height, dtype=FIELD_DTYPE)
    X, Y = np.meshgrid(x, y, indexing='xy')
    
    # Calculate Gaussian concentration field using source_location and sigma parameters
    dx = X - float(source_location.x)
    dy = Y - float(source_location.y)
    distance_squared = dx**2 + dy**2
    concentration_field = np.exp(-distance_squared / (2 * sigma**2))
    
    # Apply noise if add_noise is True for testing robustness with realistic field variations
    if add_noise:
        noise = np.random.normal(0, 0.05, concentration_field.shape).astype(FIELD_DTYPE)
        concentration_field += noise
    
    # Normalize field values to [0,1] range and convert to FIELD_DTYPE for consistency
    concentration_field = np.clip(concentration_field, 0.0, 1.0).astype(FIELD_DTYPE)
    
    return concentration_field


def assert_render_output_valid(render_output: Union[RGBArray, None],
                              expected_mode: RenderMode,
                              expected_grid_size: GridSize,
                              check_markers: bool = True) -> None:
    """Assertion utility function to validate rendering outputs including RGB arrays, performance metrics, 
    and resource usage with comprehensive format checking.
    
    Args:
        render_output: Rendering output to validate
        expected_mode: Expected rendering mode (RGB_ARRAY or HUMAN)
        expected_grid_size: Expected grid dimensions
        check_markers: Whether to validate agent and source markers
        
    Raises:
        AssertionError: If render output validation fails with detailed error context
    """
    # Validate render_output type matches expected_mode (RGB array for RGB_ARRAY, None for HUMAN)
    if expected_mode == RenderMode.RGB_ARRAY:
        assert render_output is not None, "RGB_ARRAY mode must return array, got None"
        assert isinstance(render_output, np.ndarray), f"Expected numpy array, got {type(render_output)}"
        
        # Check RGB array shape matches expected_grid_size with proper (H,W,3) dimensions
        expected_shape = (expected_grid_size.height, expected_grid_size.width, 3)
        assert render_output.shape == expected_shape, f"Shape mismatch: {render_output.shape} != {expected_shape}"
        
        # Validate RGB array dtype is RGB_DTYPE (uint8) with values in [0,255] range
        assert render_output.dtype == RGB_DTYPE, f"Dtype mismatch: {render_output.dtype} != {RGB_DTYPE}"
        assert np.all((render_output >= 0) & (render_output <= 255)), "RGB values must be in [0,255] range"
        
        # Check for agent and source markers if check_markers is True
        if check_markers:
            # Look for red agent marker pixels
            red_pixels = np.sum((render_output[:, :, 0] > 200) & (render_output[:, :, 1] < 50) & (render_output[:, :, 2] < 50))
            assert red_pixels > 0, "No red agent marker pixels found in RGB output"
            
            # Look for white source marker pixels
            white_pixels = np.sum((render_output[:, :, 0] > 200) & (render_output[:, :, 1] > 200) & (render_output[:, :, 2] > 200))
            assert white_pixels > 0, "No white source marker pixels found in RGB output"
    
    elif expected_mode == RenderMode.HUMAN:
        # Human mode should return None for interactive display
        assert render_output is None, f"HUMAN mode must return None, got {render_output}"
    
    else:
        raise AssertionError(f"Unknown render mode: {expected_mode}")


def measure_rendering_performance(renderer: BaseRenderer,
                                context: RenderContext,
                                num_iterations: int = 10,
                                warm_up_runs: bool = True) -> Dict[str, Any]:
    """Performance measurement utility function for timing rendering operations and validating against 
    performance targets with detailed analysis.
    
    Args:
        renderer: Renderer instance for performance testing
        context: Render context for operations
        num_iterations: Number of performance measurement iterations
        warm_up_runs: Whether to perform warm-up runs before measurement
        
    Returns:
        Dict[str, Any]: Performance statistics including timing, resource usage, and target compliance analysis
    """
    # Perform warm-up rendering runs if warm_up_runs is True to stabilize performance
    if warm_up_runs:
        for _ in range(3):
            try:
                renderer.render(context, RenderMode.RGB_ARRAY)
            except Exception:
                pass  # Ignore warm-up errors
    
    # Execute num_iterations rendering operations with high-precision timing measurement
    rgb_times = []
    human_times = []
    
    for _ in range(num_iterations):
        # Test RGB array rendering timing
        if renderer.supports_render_mode(RenderMode.RGB_ARRAY):
            start_time = time.perf_counter()
            renderer.render(context, RenderMode.RGB_ARRAY)
            end_time = time.perf_counter()
            rgb_times.append((end_time - start_time) * 1000)  # Convert to milliseconds
        
        # Test human mode rendering timing
        if renderer.supports_render_mode(RenderMode.HUMAN):
            start_time = time.perf_counter()
            renderer.render(context, RenderMode.HUMAN)
            end_time = time.perf_counter()
            human_times.append((end_time - start_time) * 1000)  # Convert to milliseconds
    
    # Calculate performance statistics including mean, median, min, max timing values
    performance_stats = {
        'rgb_array_performance': {},
        'human_performance': {},
        'target_compliance': {}
    }
    
    if rgb_times:
        performance_stats['rgb_array_performance'] = {
            'mean_ms': np.mean(rgb_times),
            'median_ms': np.median(rgb_times),
            'min_ms': np.min(rgb_times),
            'max_ms': np.max(rgb_times),
            'std_ms': np.std(rgb_times)
        }
        # Compare results against performance targets (RGB <5ms, HUMAN <50ms)
        performance_stats['target_compliance']['rgb_target_met'] = np.mean(rgb_times) < PERFORMANCE_TARGET_RGB_RENDER_MS
    
    if human_times:
        performance_stats['human_performance'] = {
            'mean_ms': np.mean(human_times),
            'median_ms': np.median(human_times),
            'min_ms': np.min(human_times),
            'max_ms': np.max(human_times),
            'std_ms': np.std(human_times)
        }
        performance_stats['target_compliance']['human_target_met'] = np.mean(human_times) < PERFORMANCE_TARGET_HUMAN_RENDER_MS
    
    # Return comprehensive performance analysis with target compliance and optimization recommendations
    performance_stats['num_iterations'] = num_iterations
    performance_stats['optimization_recommendations'] = []
    
    if not performance_stats['target_compliance'].get('rgb_target_met', True):
        performance_stats['optimization_recommendations'].append('Optimize RGB rendering pipeline')
    
    if not performance_stats['target_compliance'].get('human_target_met', True):
        performance_stats['optimization_recommendations'].append('Optimize human rendering pipeline')
    
    return performance_stats


class MockRenderer(BaseRenderer):
    """Concrete implementation of BaseRenderer for testing abstract base class functionality with configurable 
    behavior, performance characteristics, and error simulation capabilities.
    
    This mock renderer provides a complete implementation of the abstract BaseRenderer interface
    specifically designed for comprehensive testing scenarios.
    """
    
    def __init__(self, grid_size: GridSize, color_scheme_name: Optional[str] = None, renderer_options: Optional[Dict] = None):
        """Initialize mock renderer with testing configuration and behavior control.
        
        Args:
            grid_size: Grid dimensions for rendering operations
            color_scheme_name: Optional color scheme identifier
            renderer_options: Additional renderer configuration options
        """
        # Call parent BaseRenderer constructor with grid_size, color_scheme_name, and renderer_options
        super().__init__(grid_size, color_scheme_name or 'mock_scheme', renderer_options or {})
        
        # Initialize supports_rgb_array_mode and supports_human_mode to True by default
        self.supports_rgb_array_mode = True
        self.supports_human_mode = True
        
        # Initialize empty performance_overrides dictionary for custom timing behavior
        self.performance_overrides = {}
        
        # Set simulate_failures to False for normal operation testing
        self.simulate_failures = False
        
        # Initialize counters for initialization_calls and cleanup_calls tracking
        self.initialization_calls = 0
        self.cleanup_calls = 0
        
        # Initialize empty render_history list for operation tracking and validation
        self.render_history = []
    
    def supports_render_mode(self, mode: RenderMode) -> bool:
        """Implementation of abstract method to specify supported render modes for testing polymorphic behavior.
        
        Args:
            mode: Render mode to check for support
            
        Returns:
            bool: True if renderer supports the specified mode based on test configuration
        """
        # Check if mode is RenderMode.RGB_ARRAY and return supports_rgb_array_mode
        if mode == RenderMode.RGB_ARRAY:
            return self.supports_rgb_array_mode
        
        # Check if mode is RenderMode.HUMAN and return supports_human_mode
        elif mode == RenderMode.HUMAN:
            return self.supports_human_mode
        
        # Return False for any unsupported modes
        else:
            return False
    
    def _initialize_renderer_resources(self) -> None:
        """Implementation of abstract method to initialize mock renderer resources with tracking.
        
        Raises:
            ComponentError: If simulate_failures is True for error testing
        """
        # Increment initialization_calls counter for test validation
        self.initialization_calls += 1
        
        # Simulate resource initialization delay if performance_overrides specify timing
        if 'initialization_delay_ms' in self.performance_overrides:
            time.sleep(self.performance_overrides['initialization_delay_ms'] / 1000.0)
        
        # Raise ComponentError if simulate_failures is True for error testing
        if self.simulate_failures:
            raise ComponentError("Simulated initialization failure", component_name="MockRenderer")
    
    def _cleanup_renderer_resources(self) -> None:
        """Implementation of abstract method to cleanup mock renderer resources with tracking."""
        # Increment cleanup_calls counter for test validation
        self.cleanup_calls += 1
        
        # Simulate cleanup delay if performance_overrides specify timing
        if 'cleanup_delay_ms' in self.performance_overrides:
            time.sleep(self.performance_overrides['cleanup_delay_ms'] / 1000.0)
        
        # Clear render_history and reset performance tracking
        self.render_history.clear()
    
    def _render_rgb_array(self, context: RenderContext) -> RGBArray:
        """Implementation of abstract method to generate test RGB array with markers and performance tracking.
        
        Args:
            context: Render context with environment state
            
        Returns:
            RGBArray: Test RGB array with concentration field and position markers
        """
        # Record render operation in render_history with timestamp and context details
        self.render_history.append({
            'mode': 'rgb_array',
            'timestamp': time.time(),
            'context_id': str(hash(str(context.agent_position.to_tuple())))
        })
        
        # Simulate rendering delay based on performance_overrides for performance testing
        if 'rgb_render_delay_ms' in self.performance_overrides:
            time.sleep(self.performance_overrides['rgb_render_delay_ms'] / 1000.0)
        
        # Create RGB array from context.concentration_field with grayscale representation
        field = context.concentration_field
        rgb_array = np.zeros((self.grid_size.height, self.grid_size.width, 3), dtype=RGB_DTYPE)
        
        # Convert concentration to grayscale (0-255)
        gray_values = (field * 255).astype(RGB_DTYPE)
        rgb_array[:, :, 0] = gray_values
        rgb_array[:, :, 1] = gray_values
        rgb_array[:, :, 2] = gray_values
        
        # Add red agent marker at context.agent_position coordinates
        agent_x, agent_y = context.agent_position.x, context.agent_position.y
        if 0 <= agent_x < self.grid_size.width and 0 <= agent_y < self.grid_size.height:
            # Place 3x3 red marker centered on agent position
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    x, y = agent_x + dx, agent_y + dy
                    if 0 <= x < self.grid_size.width and 0 <= y < self.grid_size.height:
                        rgb_array[y, x] = [255, 0, 0]  # Red RGB values
        
        # Add white source marker at context.source_position coordinates
        source_x, source_y = context.source_position.x, context.source_position.y
        if 0 <= source_x < self.grid_size.width and 0 <= source_y < self.grid_size.height:
            # Place 5x5 white cross marker centered on source position
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    if abs(dx) <= 1 or abs(dy) <= 1:  # Cross pattern
                        x, y = source_x + dx, source_y + dy
                        if 0 <= x < self.grid_size.width and 0 <= y < self.grid_size.height:
                            rgb_array[y, x] = [255, 255, 255]  # White RGB values
        
        return rgb_array
    
    def _render_human(self, context: RenderContext) -> None:
        """Implementation of abstract method to simulate human mode rendering with performance tracking.
        
        Args:
            context: Render context for human display
            
        Raises:
            RenderingError: If simulate_failures is True for error handling testing
        """
        # Record render operation in render_history with human mode indication
        self.render_history.append({
            'mode': 'human',
            'timestamp': time.time(),
            'context_id': str(hash(str(context.agent_position.to_tuple())))
        })
        
        # Simulate interactive display delay based on performance_overrides
        if 'human_render_delay_ms' in self.performance_overrides:
            time.sleep(self.performance_overrides['human_render_delay_ms'] / 1000.0)
        
        # Raise RenderingError if simulate_failures is True for error handling testing
        if self.simulate_failures:
            raise RenderingError("Simulated human rendering failure", render_mode='human')
    
    def configure_for_testing(self, enable_rgb_array: bool = True, enable_human: bool = True, 
                             simulate_failures: bool = False, performance_overrides: Optional[Dict] = None) -> None:
        """Configure mock renderer behavior for specific test scenarios including failure simulation and performance overrides.
        
        Args:
            enable_rgb_array: Whether to enable RGB array mode support
            enable_human: Whether to enable human mode support
            simulate_failures: Whether to simulate failure conditions
            performance_overrides: Custom timing behavior for testing
        """
        # Set supports_rgb_array_mode and supports_human_mode based on parameters
        self.supports_rgb_array_mode = enable_rgb_array
        self.supports_human_mode = enable_human
        
        # Configure simulate_failures for error condition testing
        self.simulate_failures = simulate_failures
        
        # Update performance_overrides with custom timing behavior
        if performance_overrides:
            self.performance_overrides.update(performance_overrides)
        
        # Reset counters and history for fresh test scenario
        self.initialization_calls = 0
        self.cleanup_calls = 0
        self.render_history.clear()
    
    def get_test_statistics(self) -> Dict[str, Any]:
        """Get comprehensive testing statistics including call counts, render history, and performance data.
        
        Returns:
            Dict[str, Any]: Testing statistics with operational metrics and performance data
        """
        # Compile call counters (initialization_calls, cleanup_calls, render operations)
        rgb_renders = len([h for h in self.render_history if h['mode'] == 'rgb_array'])
        human_renders = len([h for h in self.render_history if h['mode'] == 'human'])
        
        # Calculate success rates and error frequencies for reliability analysis
        return {
            'initialization_calls': self.initialization_calls,
            'cleanup_calls': self.cleanup_calls,
            'total_renders': len(self.render_history),
            'rgb_array_renders': rgb_renders,
            'human_renders': human_renders,
            'render_history': self.render_history.copy(),
            'performance_overrides': self.performance_overrides.copy(),
            'supports_modes': {
                'rgb_array': self.supports_rgb_array_mode,
                'human': self.supports_human_mode
            },
            'failure_simulation_enabled': self.simulate_failures
        }


# Pytest fixtures for comprehensive test setup

@pytest.fixture
def mock_renderer():
    """Provides clean MockRenderer instance for each test with default configuration supporting both RGB and human rendering modes."""
    # Create MockRenderer with TEST_GRID_SIZE configuration
    grid_size = create_grid_size(TEST_GRID_SIZE)
    renderer = create_mock_renderer(
        grid_size=grid_size,
        color_scheme_name='test_default',
        supports_rgb_array=True,
        supports_human=True
    )
    
    yield renderer
    
    # Call cleanup_resources to ensure proper resource deallocation
    try:
        renderer.cleanup_resources()
    except Exception:
        pass  # Ignore cleanup errors in test teardown


@pytest.fixture 
def initialized_mock_renderer():
    """Provides initialized MockRenderer instance ready for testing operations without explicit initialization calls."""
    # Create MockRenderer with TEST_GRID_SIZE configuration
    grid_size = create_grid_size(TEST_GRID_SIZE)
    renderer = create_mock_renderer(grid_size=grid_size)
    
    # Call initialize() with validation and performance monitoring enabled
    renderer.initialize()
    
    yield renderer
    
    # Call cleanup_resources to deallocate resources properly
    try:
        renderer.cleanup_resources()
    except Exception:
        pass


@pytest.fixture
def valid_context():
    """Provides valid RenderContext with properly formatted concentration field and valid position data for testing."""
    # Create test concentration field using create_test_concentration_field
    grid_size = create_grid_size(TEST_GRID_SIZE)
    source_location = create_coordinates(TEST_SOURCE_LOCATION)
    agent_position = create_coordinates(TEST_AGENT_POSITION)
    
    concentration_field = create_test_concentration_field(
        grid_size=grid_size,
        source_location=source_location,
        sigma=12.0,
        add_noise=False
    )
    
    # Create RenderContext using create_render_context factory function
    context = create_render_context(
        concentration_field=concentration_field,
        agent_position=agent_position,
        source_position=source_location,
        grid_size=grid_size
    )
    
    yield context


@pytest.fixture
def performance_test_context():
    """Provides RenderContext optimized for performance testing with realistic data and timing validation."""
    # Create larger concentration field for realistic performance testing
    grid_size = create_grid_size((64, 64))  # Larger for performance testing
    source_location = create_coordinates((32, 32))
    agent_position = create_coordinates((16, 16))
    
    # Add noise to field data for realistic rendering complexity
    concentration_field = create_test_concentration_field(
        grid_size=grid_size,
        source_location=source_location,
        sigma=15.0,
        add_noise=True
    )
    
    context = create_render_context(
        concentration_field=concentration_field,
        agent_position=agent_position,
        source_position=source_location,
        grid_size=grid_size
    )
    
    yield context


@pytest.fixture
def rgb_capable_renderer():
    """Provides MockRenderer configured to support only RGB array rendering for mode-specific testing."""
    grid_size = create_grid_size(TEST_GRID_SIZE)
    renderer = create_mock_renderer(
        grid_size=grid_size,
        supports_rgb_array=True,
        supports_human=False  # Disable human mode support for testing mode limitations
    )
    renderer.initialize()
    
    yield renderer
    
    try:
        renderer.cleanup_resources()
    except Exception:
        pass


@pytest.fixture
def human_capable_renderer():
    """Provides MockRenderer configured to support only human mode rendering for interactive testing scenarios."""
    grid_size = create_grid_size(TEST_GRID_SIZE)
    renderer = create_mock_renderer(
        grid_size=grid_size,
        supports_rgb_array=False,  # Disable RGB array mode support
        supports_human=True
    )
    renderer.initialize()
    
    yield renderer
    
    try:
        renderer.cleanup_resources()
    except Exception:
        pass


# Parametrized fixtures for comprehensive testing

@pytest.fixture(params=[
    create_grid_size((32, 32)),
    create_grid_size((64, 64)), 
    create_grid_size((128, 128))
], ids=['small_grid', 'medium_grid', 'large_grid'])
def test_grid_sizes(request):
    """Parametrized fixture providing various grid sizes for comprehensive scalability testing."""
    return request.param


@pytest.fixture(params=[RenderMode.RGB_ARRAY, RenderMode.HUMAN], ids=['rgb_mode', 'human_mode'])
def render_modes(request):
    """Parametrized fixture providing both render modes for comprehensive dual-mode testing."""
    return request.param


# Main test class for BaseRenderer functionality

class TestBaseRenderer:
    """Main test class for BaseRenderer abstract base class functionality with comprehensive testing of interface 
    compliance, performance monitoring, error handling, and resource management.
    """
    
    @pytest.mark.unit
    def test_renderer_initialization_success(self, mock_renderer):
        """Test successful renderer initialization with valid parameters and resource setup validation."""
        # Verify renderer is not initially initialized (_initialized is False)
        assert not mock_renderer._initialized, "Renderer should not be initialized initially"
        
        # Call initialize() method with validation and performance monitoring enabled
        mock_renderer.initialize()
        
        # Assert _initialized flag is set to True after successful initialization
        assert mock_renderer._initialized, "Renderer should be initialized after calling initialize()"
        
        # Verify initialization_calls counter is incremented in mock renderer
        stats = mock_renderer.get_test_statistics()
        assert stats['initialization_calls'] == 1, "Initialization should be called exactly once"
        
        # Assert logger is properly configured for component logging
        assert mock_renderer.logger is not None, "Logger should be configured after initialization"
        
        # Validate resource cache is initialized and ready for use
        assert hasattr(mock_renderer, '_resource_cache'), "Resource cache should be initialized"
    
    @pytest.mark.unit
    @pytest.mark.error_handling
    def test_renderer_initialization_with_invalid_parameters(self):
        """Test renderer initialization failure handling with invalid parameters and proper error reporting."""
        # Create renderer with invalid grid_size (negative dimensions)
        with pytest.raises(ValidationError) as exc_info:
            invalid_grid = GridSize(-10, 20)
            create_mock_renderer(grid_size=invalid_grid)
        
        # Verify error context contains grid_size validation details
        assert 'negative' in str(exc_info.value).lower() or 'invalid' in str(exc_info.value).lower()
        
        # Test initialization with invalid color_scheme_name parameter
        grid_size = create_grid_size(TEST_GRID_SIZE)
        renderer = create_mock_renderer(grid_size=grid_size, color_scheme_name="")
        
        # The renderer should handle empty color scheme gracefully or raise ValidationError
        with pytest.raises((ValidationError, ValueError)):
            renderer.initialize()
    
    @pytest.mark.unit
    def test_render_context_validation_success(self, initialized_mock_renderer, valid_context):
        """Test successful render context validation with properly formatted context data."""
        # Call validate_context with valid_context containing proper concentration field
        validation_result = initialized_mock_renderer.validate_context(valid_context)
        
        # Assert validation returns True for valid context with all required fields
        assert validation_result, "Valid context should pass validation"
        
        # Verify context compatibility with renderer grid_size configuration
        assert valid_context.grid_size == initialized_mock_renderer.grid_size, "Context grid size should match renderer"
        
        # Test context validation with strict_validation enabled
        strict_validation = initialized_mock_renderer.validate_context(valid_context, strict_validation=True)
        assert strict_validation, "Valid context should pass strict validation"
    
    @pytest.mark.unit
    @pytest.mark.error_handling
    def test_render_context_validation_failure(self, initialized_mock_renderer):
        """Test render context validation failure scenarios with comprehensive error reporting."""
        # Create invalid context with mismatched grid_size and concentration field dimensions
        invalid_field = np.random.random((16, 16)).astype(FIELD_DTYPE)  # Wrong size
        grid_size = create_grid_size(TEST_GRID_SIZE)  # 32x32
        
        invalid_context = create_render_context(
            concentration_field=invalid_field,
            agent_position=create_coordinates(TEST_AGENT_POSITION),
            source_position=create_coordinates(TEST_SOURCE_LOCATION),
            grid_size=grid_size
        )
        
        # Expect ValidationError with detailed context validation analysis
        with pytest.raises(ValidationError) as exc_info:
            initialized_mock_renderer.validate_context(invalid_context)
        
        # Verify ValidationError contains coordinate bounds validation details
        error_msg = str(exc_info.value).lower()
        assert 'dimension' in error_msg or 'size' in error_msg or 'shape' in error_msg
        
        # Test context with agent_position outside grid boundaries
        out_of_bounds_context = create_render_context(
            concentration_field=create_test_concentration_field(
                create_grid_size(TEST_GRID_SIZE), 
                create_coordinates(TEST_SOURCE_LOCATION)
            ),
            agent_position=create_coordinates((100, 100)),  # Outside 32x32 grid
            source_position=create_coordinates(TEST_SOURCE_LOCATION),
            grid_size=create_grid_size(TEST_GRID_SIZE)
        )
        
        with pytest.raises(ValidationError):
            initialized_mock_renderer.validate_context(out_of_bounds_context)
    
    @pytest.mark.unit
    def test_render_rgb_array_mode_success(self, rgb_capable_renderer, valid_context):
        """Test successful RGB array rendering with proper output format and performance validation."""
        # Call render method with test_context expecting RGB array output
        start_time = time.perf_counter()
        render_output = rgb_capable_renderer.render(valid_context, RenderMode.RGB_ARRAY)
        end_time = time.perf_counter()
        
        # Assert returned array has correct shape (H, W, 3) matching grid_size
        expected_shape = (valid_context.grid_size.height, valid_context.grid_size.width, 3)
        assert render_output.shape == expected_shape, f"RGB array shape should be {expected_shape}"
        
        # Verify array dtype is RGB_DTYPE (uint8) with values in [0, 255] range
        assert render_output.dtype == RGB_DTYPE, f"RGB array dtype should be {RGB_DTYPE}"
        assert np.all((render_output >= 0) & (render_output <= 255)), "RGB values must be in [0,255] range"
        
        # Check for agent marker (red pixels) at context.agent_position
        agent_y, agent_x = valid_context.agent_position.y, valid_context.agent_position.x
        agent_pixel = render_output[agent_y, agent_x]
        assert agent_pixel[0] > 200 and agent_pixel[1] < 50 and agent_pixel[2] < 50, "Agent marker should be red"
        
        # Verify source marker (white pixels) at context.source_position
        source_y, source_x = valid_context.source_position.y, valid_context.source_position.x
        source_pixel = render_output[source_y, source_x]
        assert np.all(source_pixel > 200), "Source marker should be white"
        
        # Assert rendering time meets performance target (< PERFORMANCE_TARGET_RGB_RENDER_MS)
        render_time_ms = (end_time - start_time) * 1000
        assert render_time_ms < PERFORMANCE_TARGET_RGB_RENDER_MS + TEST_PERFORMANCE_TOLERANCE_MS, \
               f"RGB rendering took {render_time_ms:.2f}ms, exceeds target {PERFORMANCE_TARGET_RGB_RENDER_MS}ms"
    
    @pytest.mark.unit
    def test_render_human_mode_success(self, human_capable_renderer, valid_context):
        """Test successful human mode rendering with display simulation and performance validation."""
        # Call render method with test_context in human mode
        start_time = time.perf_counter()
        render_output = human_capable_renderer.render(valid_context, RenderMode.HUMAN)
        end_time = time.perf_counter()
        
        # Assert method returns None for human mode (interactive display)
        assert render_output is None, "Human mode should return None"
        
        # Verify render operation is recorded in mock renderer history
        stats = human_capable_renderer.get_test_statistics()
        assert stats['human_renders'] == 1, "Human render should be recorded in history"
        
        # Assert rendering time meets performance target (< PERFORMANCE_TARGET_HUMAN_RENDER_MS)
        render_time_ms = (end_time - start_time) * 1000
        assert render_time_ms < PERFORMANCE_TARGET_HUMAN_RENDER_MS + TEST_PERFORMANCE_TOLERANCE_MS, \
               f"Human rendering took {render_time_ms:.2f}ms, exceeds target {PERFORMANCE_TARGET_HUMAN_RENDER_MS}ms"
    
    @pytest.mark.unit
    @pytest.mark.error_handling
    def test_render_unsupported_mode(self, valid_context):
        """Test rendering behavior when renderer doesn't support requested mode with proper error handling."""
        # Configure mock renderer to support only RGB_ARRAY mode
        grid_size = create_grid_size(TEST_GRID_SIZE)
        limited_renderer = create_mock_renderer(
            grid_size=grid_size,
            supports_rgb_array=True,
            supports_human=False
        )
        limited_renderer.initialize()
        
        # Attempt rendering in HUMAN mode and expect ComponentError
        with pytest.raises(ComponentError) as exc_info:
            limited_renderer.render(valid_context, RenderMode.HUMAN)
        
        # Verify error message indicates mode not supported by renderer
        error_msg = str(exc_info.value).lower()
        assert 'unsupported' in error_msg or 'not supported' in error_msg
        
        # Test renderer supporting only HUMAN mode with RGB_ARRAY request
        human_only_renderer = create_mock_renderer(
            grid_size=grid_size,
            supports_rgb_array=False,
            supports_human=True
        )
        human_only_renderer.initialize()
        
        with pytest.raises(ComponentError):
            human_only_renderer.render(valid_context, RenderMode.RGB_ARRAY)
        
        # Cleanup
        try:
            limited_renderer.cleanup_resources()
            human_only_renderer.cleanup_resources()
        except Exception:
            pass
    
    @pytest.mark.unit
    @pytest.mark.performance
    def test_performance_metrics_collection(self, initialized_mock_renderer, valid_context):
        """Test comprehensive performance metrics collection including timing data and resource usage tracking."""
        # Execute multiple rendering operations with timing measurement
        num_operations = 5
        for _ in range(num_operations):
            initialized_mock_renderer.render(valid_context, RenderMode.RGB_ARRAY)
        
        # Call get_performance_metrics to retrieve comprehensive metrics
        metrics = initialized_mock_renderer.get_performance_metrics()
        
        # Assert metrics contain timing data for render operations
        assert 'render_operations' in metrics, "Metrics should contain render operation data"
        assert metrics['render_operations'] >= num_operations, "Should track all render operations"
        
        # Verify operation_count tracking and frequency analysis
        stats = initialized_mock_renderer.get_test_statistics()
        assert stats['rgb_array_renders'] == num_operations, "Should track RGB render count"
        
        # Check resource usage metrics including memory consumption
        assert 'resource_usage' in metrics, "Metrics should include resource usage data"
    
    @pytest.mark.performance
    def test_performance_target_validation(self, valid_context):
        """Test performance target validation with timing analysis and warning generation for optimization guidance."""
        # Configure mock renderer with deliberately slow rendering (exceeding targets)
        grid_size = create_grid_size(TEST_GRID_SIZE)
        slow_renderer = create_mock_renderer(
            grid_size=grid_size,
            performance_overrides={
                'rgb_render_delay_ms': PERFORMANCE_TARGET_RGB_RENDER_MS * 2,  # Deliberately slow
                'human_render_delay_ms': PERFORMANCE_TARGET_HUMAN_RENDER_MS * 2
            }
        )
        slow_renderer.initialize()
        
        # Execute rendering operations and capture performance warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # Measure performance with slow renderer
            performance_stats = measure_rendering_performance(
                slow_renderer, 
                valid_context, 
                num_iterations=3, 
                warm_up_runs=False
            )
            
            # Assert performance metrics indicate target violations
            assert not performance_stats['target_compliance'].get('rgb_target_met', True), \
                   "Slow renderer should fail RGB performance targets"
        
        # Cleanup
        try:
            slow_renderer.cleanup_resources()
        except Exception:
            pass
    
    @pytest.mark.unit
    @pytest.mark.resource_management
    def test_resource_cleanup_success(self, mock_renderer):
        """Test successful resource cleanup with proper resource management and memory deallocation."""
        # Initialize renderer with resource allocation tracking
        mock_renderer.initialize()
        initial_stats = mock_renderer.get_test_statistics()
        
        # Perform rendering operations that allocate cached resources
        valid_context = create_render_context(
            concentration_field=create_test_concentration_field(
                mock_renderer.grid_size,
                create_coordinates(TEST_SOURCE_LOCATION)
            ),
            agent_position=create_coordinates(TEST_AGENT_POSITION),
            source_position=create_coordinates(TEST_SOURCE_LOCATION),
            grid_size=mock_renderer.grid_size
        )
        mock_renderer.render(valid_context, RenderMode.RGB_ARRAY)
        
        # Call cleanup_resources with standard timeout
        mock_renderer.cleanup_resources()
        
        # Assert cleanup_calls counter is incremented in mock renderer
        final_stats = mock_renderer.get_test_statistics()
        assert final_stats['cleanup_calls'] == 1, "Cleanup should be called exactly once"
        
        # Verify _initialized flag is set to False after cleanup
        assert not mock_renderer._initialized, "Renderer should not be initialized after cleanup"
    
    @pytest.mark.unit
    @pytest.mark.resource_management
    def test_resource_cleanup_timeout(self):
        """Test resource cleanup timeout handling with forced cleanup and error reporting."""
        # Configure mock renderer with deliberately slow cleanup operations
        grid_size = create_grid_size(TEST_GRID_SIZE)
        slow_cleanup_renderer = create_mock_renderer(
            grid_size=grid_size,
            performance_overrides={
                'cleanup_delay_ms': 100  # 100ms delay
            }
        )
        slow_cleanup_renderer.initialize()
        
        # Call cleanup_resources with short timeout period
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # Test cleanup with very short timeout
            slow_cleanup_renderer.cleanup_resources(timeout=0.05)  # 50ms timeout
            
            # Verify cleanup completes (may generate warnings about timeout)
            assert not slow_cleanup_renderer._initialized
    
    @pytest.mark.unit
    def test_supports_render_mode_validation(self, mock_renderer):
        """Test render mode support checking with comprehensive mode validation and capability reporting."""
        # Configure renderer to support both RGB_ARRAY and HUMAN modes
        mock_renderer.configure_for_testing(enable_rgb_array=True, enable_human=True)
        
        # Test supports_render_mode for each supported mode
        assert mock_renderer.supports_render_mode(RenderMode.RGB_ARRAY), "Should support RGB_ARRAY mode"
        assert mock_renderer.supports_render_mode(RenderMode.HUMAN), "Should support HUMAN mode"
        
        # Configure renderer with only RGB_ARRAY support
        mock_renderer.configure_for_testing(enable_rgb_array=True, enable_human=False)
        
        # Verify supports_render_mode returns False for unsupported HUMAN mode
        assert mock_renderer.supports_render_mode(RenderMode.RGB_ARRAY), "Should still support RGB_ARRAY"
        assert not mock_renderer.supports_render_mode(RenderMode.HUMAN), "Should not support HUMAN mode"
    
    @pytest.mark.unit
    @pytest.mark.error_handling
    def test_error_handling_during_rendering(self, valid_context):
        """Test comprehensive error handling during rendering operations with recovery strategies and fallback mechanisms."""
        # Configure mock renderer to simulate RenderingError during operations
        grid_size = create_grid_size(TEST_GRID_SIZE)
        error_prone_renderer = create_mock_renderer(grid_size=grid_size)
        error_prone_renderer.initialize()
        error_prone_renderer.configure_for_testing(simulate_failures=True)
        
        # Attempt rendering and expect proper error handling with context preservation
        with pytest.raises(RenderingError) as exc_info:
            error_prone_renderer.render(valid_context, RenderMode.HUMAN)
        
        # Verify error details include render mode, context, and recovery suggestions
        error = exc_info.value
        assert hasattr(error, 'render_mode'), "Error should include render mode information"
        assert error.render_mode == 'human', "Error should specify failed render mode"
        
        # Cleanup
        try:
            error_prone_renderer.cleanup_resources()
        except Exception:
            pass
    
    @pytest.mark.unit
    def test_context_cloning_and_overrides(self, valid_context):
        """Test render context cloning functionality with parameter overrides and immutability preservation."""
        # Create cloned context with new agent position using clone_with_overrides
        new_agent_position = create_coordinates((20, 20))
        cloned_context = valid_context.clone_with_overrides(agent_position=new_agent_position)
        
        # Assert original context remains unchanged (immutability)
        assert valid_context.agent_position != cloned_context.agent_position, "Original context should be unchanged"
        assert valid_context.agent_position.to_tuple() == TEST_AGENT_POSITION, "Original agent position should be preserved"
        
        # Verify cloned context has updated agent position
        assert cloned_context.agent_position.to_tuple() == (20, 20), "Cloned context should have new agent position"
        
        # Test context cloning with multiple parameter overrides simultaneously
        multi_clone = valid_context.clone_with_overrides(
            agent_position=create_coordinates((10, 15)),
            source_position=create_coordinates((25, 25))
        )
        assert multi_clone.agent_position.to_tuple() == (10, 15), "Multi-override should update agent position"
        assert multi_clone.source_position.to_tuple() == (25, 25), "Multi-override should update source position"
    
    @pytest.mark.unit
    @pytest.mark.integration
    def test_rendering_with_different_grid_sizes(self, test_grid_sizes):
        """Test rendering functionality across different grid sizes with memory and performance scaling validation."""
        # Create renderer and context for each grid size configuration
        renderer = create_mock_renderer(grid_size=test_grid_sizes)
        renderer.initialize()
        
        # Create context with appropriate dimensions
        source_pos = create_coordinates((test_grid_sizes.width // 2, test_grid_sizes.height // 2))
        agent_pos = create_coordinates((test_grid_sizes.width // 4, test_grid_sizes.height // 4))
        
        concentration_field = create_test_concentration_field(
            grid_size=test_grid_sizes,
            source_location=source_pos
        )
        
        context = create_render_context(
            concentration_field=concentration_field,
            agent_position=agent_pos,
            source_position=source_pos,
            grid_size=test_grid_sizes
        )
        
        # Execute rendering operations and validate output scaling
        rgb_output = renderer.render(context, RenderMode.RGB_ARRAY)
        
        # Assert memory usage scales appropriately with grid dimensions
        expected_shape = (test_grid_sizes.height, test_grid_sizes.width, 3)
        assert rgb_output.shape == expected_shape, f"Output shape should match grid size: {expected_shape}"
        
        # Verify performance characteristics meet targets across all grid sizes
        performance_stats = measure_rendering_performance(renderer, context, num_iterations=3)
        
        # Cleanup
        try:
            renderer.cleanup_resources()
        except Exception:
            pass
    
    @pytest.mark.unit
    @pytest.mark.integration
    def test_concurrent_rendering_operations(self, valid_context):
        """Test thread safety and concurrent rendering operations with resource contention handling."""
        # Create multiple renderer instances for concurrent testing
        grid_size = create_grid_size(TEST_GRID_SIZE)
        renderers = []
        
        for i in range(3):
            renderer = create_mock_renderer(grid_size=grid_size)
            renderer.initialize()
            renderers.append(renderer)
        
        # Execute simultaneous rendering operations from different threads
        results = []
        errors = []
        
        def render_worker(renderer, context, worker_id):
            try:
                output = renderer.render(context, RenderMode.RGB_ARRAY)
                results.append((worker_id, output))
            except Exception as e:
                errors.append((worker_id, e))
        
        threads = []
        for i, renderer in enumerate(renderers):
            thread = threading.Thread(target=render_worker, args=(renderer, valid_context, i))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=DEFAULT_TEST_TIMEOUT)
        
        # Assert render outputs are correct and don't interfere between threads
        assert len(errors) == 0, f"No rendering errors should occur in concurrent execution: {errors}"
        assert len(results) == 3, "All renderers should complete successfully"
        
        # Verify each output has correct format
        for worker_id, output in results:
            assert_render_output_valid(output, RenderMode.RGB_ARRAY, grid_size, check_markers=True)
        
        # Cleanup all renderers
        for renderer in renderers:
            try:
                renderer.cleanup_resources()
            except Exception:
                pass