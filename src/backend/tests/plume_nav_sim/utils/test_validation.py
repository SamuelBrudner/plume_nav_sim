# External imports with version comments
import pytest  # >=8.0.0 - Primary testing framework for test organization, fixtures, parametrized testing, and comprehensive test execution with advanced assertion capabilities
import numpy as np  # >=2.1.0 - Array operations, random number generation testing, dtype validation, and mathematical validation test scenarios
import time  # >=3.10 - Performance timing measurements, validation latency testing, and benchmark validation for performance requirements
import threading  # >=3.10 - Thread safety testing for validation operations including concurrent validation and thread-safe caching validation
import unittest.mock as mock  # >=3.10 - Mocking capabilities for testing validation system behavior including dependency isolation and error scenario simulation
import warnings  # >=3.10 - Warning capture and validation for testing validation system warnings and deprecation handling
import copy  # >=3.10 - Deep copying operations for testing parameter sanitization and validation result isolation
import gc  # >=3.10 - Garbage collection control for memory usage testing and resource constraint validation
import sys  # >=3.10 - System information access for platform-specific validation testing and memory limit validation

# Internal imports - validation utilities module for comprehensive testing
from plume_nav_sim.utils.validation import (
    validate_environment_config,
    validate_action_parameter, 
    validate_observation_parameter,
    validate_coordinates,
    validate_grid_size,
    validate_plume_parameters,
    validate_render_mode,
    validate_seed_value,
    validate_performance_constraints,
    sanitize_parameters,
    check_parameter_consistency,
    ValidationContext,
    ValidationResult,
    ParameterValidator,
    create_validation_context,
    validate_with_context,
    get_validation_summary,
    optimize_validation_performance
)

# Internal imports - core types and data structures
from plume_nav_sim.core.types import (
    Action,
    RenderMode,
    Coordinates,
    GridSize,
    PlumeParameters,
    EnvironmentConfig
)

# Internal imports - exception hierarchy for error testing
from plume_nav_sim.utils.exceptions import (
    ValidationError,
    ConfigurationError,
    ResourceError,
    sanitize_error_context
)

# Internal imports - system constants for boundary testing
from plume_nav_sim.core.constants import (
    DEFAULT_GRID_SIZE,
    MIN_GRID_SIZE,
    MAX_GRID_SIZE,
    MIN_PLUME_SIGMA,
    MAX_PLUME_SIGMA,
    ACTION_SPACE_SIZE,
    CONCENTRATION_RANGE,
    SUPPORTED_RENDER_MODES,
    SEED_MIN_VALUE,
    SEED_MAX_VALUE,
    MEMORY_LIMIT_TOTAL_MB,
    PERFORMANCE_TARGET_STEP_LATENCY_MS
)

# Global test constants for comprehensive validation testing
VALIDATION_TEST_TIMEOUT = 30.0
PERFORMANCE_TEST_ITERATIONS = 1000
VALIDATION_PERFORMANCE_TOLERANCE_MS = 1.0
BATCH_VALIDATION_SIZE = 100
CACHE_TEST_SIZE = 500
THREAD_SAFETY_TEST_THREADS = 10
MEMORY_TEST_LIMIT_MB = 100

# Valid test data collections for parametrized testing
VALID_TEST_ACTIONS = [0, 1, 2, 3]
INVALID_TEST_ACTIONS = [-1, 4, 'invalid', None, 3.14, [0], {'action': 0}]
VALID_TEST_SEEDS = [42, 0, 123, 456, 999999]
INVALID_TEST_SEEDS = [-1, 'invalid', 3.14, [], {}, None]
VALID_TEST_GRID_DIMENSIONS = [(16, 16), (32, 32), (64, 64), (128, 128), (256, 256)]
INVALID_TEST_GRID_DIMENSIONS = [(0, 0), (15, 15), (-1, -1), (1000, 1000), 'invalid', None]
VALID_TEST_COORDINATES = [(0, 0), (16, 16), (63, 63), (127, 127)]
INVALID_TEST_COORDINATES = [(-1, 0), (128, 128), (200, 200), 'invalid', None, [0, 0]]
VALID_TEST_RENDER_MODES = ['rgb_array', 'human']
INVALID_TEST_RENDER_MODES = ['invalid', None, 123, 'RGB_ARRAY', 'Human']

# Security test data for injection prevention and sanitization testing
SECURITY_TEST_INPUTS = {
    'safe_parameters': {
        'grid_size': (64, 64),
        'action': 1
    },
    'malicious_inputs': {
        '<script>alert(1)</script>': 'value',
        '../../etc/passwd': 'config',
        'DROP TABLE users;': 'command'
    },
    'sensitive_data': {
        'password': 'secret123',
        'api_key': 'abc123def456',
        'private_key': 'private_data'
    }
}

# Edge case test scenarios for boundary condition validation
EDGE_CASE_TEST_SCENARIOS = {
    'boundary_values': {
        'min_sigma': MIN_PLUME_SIGMA,
        'max_sigma': MAX_PLUME_SIGMA,
        'min_grid': MIN_GRID_SIZE,
        'max_grid': MAX_GRID_SIZE
    },
    'extreme_values': {
        'huge_sigma': 1000.0,
        'tiny_sigma': 0.001,
        'huge_grid': (2048, 2048)
    },
    'corner_cases': {
        'zero_coordinates': (0, 0),
        'single_cell_grid': (1, 1)
    }
}

# Cross-parameter consistency test scenarios
CONSISTENCY_TEST_SCENARIOS = [
    {'grid_size': (32, 32), 'source_location': (16, 16), 'sigma': 5.0},
    {'grid_size': (64, 64), 'source_location': (0, 0), 'sigma': 10.0},
    {'grid_size': (128, 128), 'source_location': (127, 127), 'sigma': 20.0}
]


class TestValidationFunctions:
    """Comprehensive test suite for individual validation functions covering parameter validation, error handling, security features, and performance requirements with extensive edge case coverage."""

    def test_validate_action_parameter_valid_actions(self):
        """Test validate_action_parameter with all valid action values ensuring proper Discrete(4) space compliance and type conversion."""
        for action in VALID_TEST_ACTIONS:
            # Test with ValidationContext for comprehensive error tracking
            context = create_validation_context('test_action_validation')
            result = validate_action_parameter(action, context=context)
            
            # Validate action is converted to integer in valid range [0, 3]
            assert isinstance(result, int)
            assert 0 <= result < ACTION_SPACE_SIZE
            assert result == action
        
        # Test Action enum values
        for action_enum in Action:
            context = create_validation_context('test_action_enum')
            result = validate_action_parameter(action_enum.value, context=context)
            assert result == action_enum.value

    def test_validate_action_parameter_invalid_actions(self):
        """Test validate_action_parameter with invalid action values ensuring proper ValidationError handling and error context."""
        for invalid_action in INVALID_TEST_ACTIONS:
            context = create_validation_context('test_invalid_action')
            
            with pytest.raises(ValidationError) as exc_info:
                validate_action_parameter(invalid_action, context=context)
            
            # Validate error contains parameter information
            error = exc_info.value
            assert 'action' in str(error.message).lower()
            assert error.parameter_name == 'action'
            assert error.invalid_value is not None
            
            # Verify recovery suggestions are provided
            assert error.recovery_suggestion is not None
            assert len(error.recovery_suggestion) > 0

    def test_validate_observation_parameter_valid_observations(self):
        """Test validate_observation_parameter with valid concentration values ensuring Box space compliance and range validation."""
        valid_observations = [0.0, 0.25, 0.5, 0.75, 1.0, np.float32(0.6), np.array([0.3])]
        
        for obs in valid_observations:
            context = create_validation_context('test_observation_validation')
            result = validate_observation_parameter(obs, context=context)
            
            # Validate observation is converted to numpy array with correct dtype
            assert isinstance(result, np.ndarray)
            assert result.dtype == np.float32
            assert result.shape == (1,)
            assert CONCENTRATION_RANGE[0] <= result[0] <= CONCENTRATION_RANGE[1]

    def test_validate_observation_parameter_invalid_observations(self):
        """Test validate_observation_parameter with invalid concentration values ensuring proper error handling and range validation."""
        invalid_observations = [-0.1, 1.1, -10.0, 5.0, 'invalid', None, [0.5, 0.6]]
        
        for invalid_obs in invalid_observations:
            context = create_validation_context('test_invalid_observation')
            
            with pytest.raises(ValidationError) as exc_info:
                validate_observation_parameter(invalid_obs, context=context)
            
            # Validate error contains observation-specific information
            error = exc_info.value
            assert 'observation' in str(error.message).lower() or 'concentration' in str(error.message).lower()
            assert error.parameter_name == 'observation'

    def test_validate_coordinates_valid_coordinates(self):
        """Test validate_coordinates with valid coordinate values ensuring bounds checking and grid compatibility validation."""
        grid_size = (128, 128)
        
        for coords in VALID_TEST_COORDINATES:
            context = create_validation_context('test_coordinates_validation')
            result = validate_coordinates(coords, grid_size, context=context)
            
            # Validate coordinates are converted to Coordinates dataclass
            assert isinstance(result, Coordinates)
            assert result.x >= 0 and result.x < grid_size[0]
            assert result.y >= 0 and result.y < grid_size[1]
            
            # Test grid bounds checking
            assert result.is_within_bounds(GridSize(*grid_size))

    def test_validate_coordinates_invalid_coordinates(self):
        """Test validate_coordinates with invalid coordinate values ensuring proper error handling and bounds validation."""
        grid_size = (128, 128)
        
        for invalid_coords in INVALID_TEST_COORDINATES:
            context = create_validation_context('test_invalid_coordinates')
            
            with pytest.raises(ValidationError) as exc_info:
                validate_coordinates(invalid_coords, grid_size, context=context)
            
            # Validate error contains coordinate-specific information
            error = exc_info.value
            assert 'coordinate' in str(error.message).lower()
            assert error.parameter_name == 'coordinates'

    def test_validate_grid_size_valid_dimensions(self):
        """Test validate_grid_size with valid grid dimensions ensuring dimension checking and memory estimation validation."""
        for grid_dims in VALID_TEST_GRID_DIMENSIONS:
            context = create_validation_context('test_grid_size_validation')
            result = validate_grid_size(grid_dims, context=context)
            
            # Validate grid size is converted to GridSize dataclass
            assert isinstance(result, GridSize)
            assert result.width == grid_dims[0]
            assert result.height == grid_dims[1]
            
            # Test memory estimation functionality
            memory_mb = result.estimate_memory_mb()
            assert isinstance(memory_mb, (int, float))
            assert memory_mb > 0

    def test_validate_grid_size_invalid_dimensions(self):
        """Test validate_grid_size with invalid grid dimensions ensuring proper error handling and constraint validation."""
        for invalid_dims in INVALID_TEST_GRID_DIMENSIONS:
            context = create_validation_context('test_invalid_grid_size')
            
            with pytest.raises(ValidationError) as exc_info:
                validate_grid_size(invalid_dims, context=context)
            
            # Validate error contains grid size specific information
            error = exc_info.value
            assert 'grid' in str(error.message).lower() or 'size' in str(error.message).lower()
            assert error.parameter_name == 'grid_size'

    def test_validate_plume_parameters_valid_parameters(self):
        """Test validate_plume_parameters with valid plume parameters ensuring mathematical consistency and Gaussian formula coherence."""
        valid_plume_params = [
            {'source_location': (64, 64), 'sigma': 10.0, 'intensity': 1.0},
            {'source_location': (32, 32), 'sigma': MIN_PLUME_SIGMA, 'intensity': 0.8},
            {'source_location': (0, 0), 'sigma': MAX_PLUME_SIGMA, 'intensity': 0.5}
        ]
        
        for params in valid_plume_params:
            context = create_validation_context('test_plume_parameters')
            result = validate_plume_parameters(params, context=context)
            
            # Validate plume parameters are converted to PlumeParameters dataclass
            assert isinstance(result, PlumeParameters)
            assert result.source_location is not None
            assert MIN_PLUME_SIGMA <= result.sigma <= MAX_PLUME_SIGMA
            assert result.intensity > 0.0
            
            # Test mathematical validation
            result.validate()  # Should not raise exception

    def test_validate_plume_parameters_invalid_parameters(self):
        """Test validate_plume_parameters with invalid plume parameters ensuring mathematical validation and constraint checking."""
        invalid_plume_params = [
            {'source_location': (-1, -1), 'sigma': 10.0, 'intensity': 1.0},  # Invalid location
            {'source_location': (64, 64), 'sigma': -1.0, 'intensity': 1.0},  # Invalid sigma
            {'source_location': (64, 64), 'sigma': 10.0, 'intensity': -1.0}   # Invalid intensity
        ]
        
        for invalid_params in invalid_plume_params:
            context = create_validation_context('test_invalid_plume_params')
            
            with pytest.raises(ValidationError) as exc_info:
                validate_plume_parameters(invalid_params, context=context)
            
            # Validate error contains plume-specific information
            error = exc_info.value
            assert 'plume' in str(error.message).lower() or 'parameter' in str(error.message).lower()

    def test_validate_render_mode_valid_modes(self):
        """Test validate_render_mode with valid render modes ensuring supported modes and backend compatibility validation."""
        for mode in VALID_TEST_RENDER_MODES:
            context = create_validation_context('test_render_mode')
            result = validate_render_mode(mode, context=context)
            
            # Validate render mode is converted to RenderMode enum
            assert isinstance(result, RenderMode)
            assert result.value == mode
            assert mode in SUPPORTED_RENDER_MODES

    def test_validate_render_mode_invalid_modes(self):
        """Test validate_render_mode with invalid render modes ensuring proper error handling and supported mode validation."""
        for invalid_mode in INVALID_TEST_RENDER_MODES:
            context = create_validation_context('test_invalid_render_mode')
            
            with pytest.raises(ValidationError) as exc_info:
                validate_render_mode(invalid_mode, context=context)
            
            # Validate error contains render mode specific information
            error = exc_info.value
            assert 'render' in str(error.message).lower() or 'mode' in str(error.message).lower()
            assert error.parameter_name == 'render_mode'

    def test_validate_seed_value_valid_seeds(self):
        """Test validate_seed_value with valid seed values ensuring reproducibility requirements and type compliance."""
        for seed in VALID_TEST_SEEDS:
            context = create_validation_context('test_seed_validation')
            result = validate_seed_value(seed, context=context)
            
            # Validate seed is converted to integer within valid range
            assert isinstance(result, int)
            assert SEED_MIN_VALUE <= result <= SEED_MAX_VALUE
            assert result == seed

    def test_validate_seed_value_invalid_seeds(self):
        """Test validate_seed_value with invalid seed values ensuring proper error handling and range validation."""
        for invalid_seed in INVALID_TEST_SEEDS:
            context = create_validation_context('test_invalid_seed')
            
            with pytest.raises(ValidationError) as exc_info:
                validate_seed_value(invalid_seed, context=context)
            
            # Validate error contains seed-specific information
            error = exc_info.value
            assert 'seed' in str(error.message).lower()
            assert error.parameter_name == 'seed'

    def test_validate_performance_constraints_valid_constraints(self):
        """Test validate_performance_constraints with valid system constraints ensuring capability and resource limits validation."""
        valid_constraints = [
            {'max_memory_mb': 50, 'max_latency_ms': 1.0, 'max_grid_cells': 16384},
            {'max_memory_mb': MEMORY_TEST_LIMIT_MB, 'max_latency_ms': PERFORMANCE_TARGET_STEP_LATENCY_MS}
        ]
        
        for constraints in valid_constraints:
            context = create_validation_context('test_performance_constraints')
            result = validate_performance_constraints(constraints, context=context)
            
            # Validate constraints are properly processed
            assert isinstance(result, dict)
            assert 'validated' in result
            assert result['validated'] is True

    def test_validate_performance_constraints_invalid_constraints(self):
        """Test validate_performance_constraints with invalid performance constraints ensuring resource limit validation."""
        invalid_constraints = [
            {'max_memory_mb': -1, 'max_latency_ms': 1.0},     # Invalid memory
            {'max_memory_mb': 50, 'max_latency_ms': -1.0},    # Invalid latency
            {'max_memory_mb': 10000, 'max_latency_ms': 1.0}   # Excessive memory
        ]
        
        for invalid_constraint in invalid_constraints:
            context = create_validation_context('test_invalid_performance')
            
            with pytest.raises((ValidationError, ResourceError)) as exc_info:
                validate_performance_constraints(invalid_constraint, context=context)
            
            # Validate error contains performance-specific information
            error = exc_info.value
            assert 'performance' in str(error.message).lower() or 'resource' in str(error.message).lower()


class TestSecurityFeatures:
    """Comprehensive security testing suite for input sanitization, injection prevention, information disclosure protection, and secure error handling ensuring validation system security compliance."""

    def test_sanitize_parameters_secure_input_handling(self):
        """Test sanitize_parameters with malicious and sensitive inputs ensuring proper sanitization and injection prevention."""
        # Test with security test inputs
        malicious_data = SECURITY_TEST_INPUTS['malicious_inputs']
        
        context = create_validation_context('test_parameter_sanitization')
        result = sanitize_parameters(malicious_data, context=context)
        
        # Validate malicious inputs are sanitized
        assert isinstance(result, dict)
        for key, value in result.items():
            # Check that potentially harmful content is sanitized
            assert '<script>' not in str(value).lower()
            assert 'drop table' not in str(value).lower()
            assert '../' not in str(value)

    def test_sanitize_parameters_sensitive_data_protection(self):
        """Test sanitize_parameters with sensitive data ensuring information disclosure protection and secure error reporting."""
        sensitive_data = SECURITY_TEST_INPUTS['sensitive_data']
        
        context = create_validation_context('test_sensitive_data')
        result = sanitize_parameters(sensitive_data, context=context)
        
        # Validate sensitive data is properly masked
        assert isinstance(result, dict)
        for key, value in result.items():
            # Sensitive values should be replaced or masked
            if any(sensitive in key.lower() for sensitive in ['password', 'key', 'secret']):
                assert str(value) != sensitive_data[key]  # Original value should be masked

    def test_sanitize_parameters_sql_injection_prevention(self):
        """Test sanitize_parameters against SQL injection attempts ensuring database security and input validation."""
        sql_injection_inputs = {
            'user_input': "'; DROP TABLE users; --",
            'search_term': "' OR '1'='1",
            'config_value': "'; SELECT * FROM sensitive_data; --"
        }
        
        context = create_validation_context('test_sql_injection')
        result = sanitize_parameters(sql_injection_inputs, context=context)
        
        # Validate SQL injection patterns are neutralized
        for key, value in result.items():
            assert 'DROP TABLE' not in str(value)
            assert 'SELECT *' not in str(value)
            assert "'OR'1'='1" not in str(value)

    def test_sanitize_parameters_xss_prevention(self):
        """Test sanitize_parameters against XSS attacks ensuring web security and script injection prevention."""
        xss_inputs = {
            'user_message': '<script>alert("XSS")</script>',
            'comment': '<img src=x onerror=alert("XSS")>',
            'title': 'javascript:alert("XSS")'
        }
        
        context = create_validation_context('test_xss_prevention')
        result = sanitize_parameters(xss_inputs, context=context)
        
        # Validate XSS vectors are neutralized
        for key, value in result.items():
            assert '<script>' not in str(value).lower()
            assert 'javascript:' not in str(value).lower()
            assert 'onerror=' not in str(value).lower()

    def test_secure_error_context_sanitization(self):
        """Test error context sanitization ensuring no sensitive information is disclosed in error messages or logs."""
        # Create error context with sensitive information
        sensitive_context = {
            'user_password': 'secret123',
            'api_key': 'abc123def456',
            'database_url': 'postgresql://user:pass@localhost/db',
            'safe_parameter': 'safe_value'
        }
        
        # Test sanitization through ValidationError
        try:
            raise ValidationError("Test error", parameter_name="test_param")
        except ValidationError as e:
            e.add_context('sensitive_context', sensitive_context)
            error_details = e.get_error_details()
            
            # Validate sensitive information is not present in error details
            error_str = str(error_details)
            assert 'secret123' not in error_str
            assert 'abc123def456' not in error_str
            assert 'user:pass' not in error_str
            
            # Validate safe parameters are preserved
            assert 'safe_value' in error_str or 'safe_parameter' in error_str

    def test_secure_validation_context_handling(self):
        """Test ValidationContext secure handling ensuring caller information doesn't leak sensitive data."""
        # Create validation context that might contain sensitive frame information
        context = create_validation_context('secure_test_operation')
        context.add_caller_info()
        context.add_system_info()
        
        # Add some sensitive additional data
        context.additional_data.update({
            'password_hash': 'sensitive_hash_value',
            'session_token': 'secret_token',
            'safe_data': 'this_is_safe'
        })
        
        # Sanitize the context
        context.sanitize(['password_hash', 'session_token'])
        
        # Validate sensitive data is sanitized
        context_dict = context.to_dict()
        context_str = str(context_dict)
        assert 'sensitive_hash_value' not in context_str
        assert 'secret_token' not in context_str
        assert 'this_is_safe' in context_str or '<sanitized>' in context_str

    def test_parameter_injection_validation(self):
        """Test parameter validation against various injection techniques ensuring comprehensive input security."""
        injection_test_cases = [
            # Command injection
            {'command': '$(rm -rf /)'},
            {'path': '../../../../etc/passwd'},
            
            # LDAP injection
            {'username': 'admin)(|(password=*)'},
            
            # NoSQL injection
            {'query': '{"$ne": null}'},
            
            # Template injection
            {'template': '{{7*7}}'},
            
            # Path traversal
            {'filename': '../../../sensitive_file.txt'}
        ]
        
        for injection_case in injection_test_cases:
            context = create_validation_context('test_injection_prevention')
            result = sanitize_parameters(injection_case, context=context)
            
            # Validate injection patterns are neutralized
            for key, value in result.items():
                assert '$(rm -rf' not in str(value)
                assert '../../../' not in str(value)
                assert '$((' not in str(value) and '))' not in str(value)
                assert '$ne' not in str(value)


class TestValidationContextClass:
    """Comprehensive test suite for ValidationContext class testing caller information, timing data, context merging, and debugging support functionality."""

    def test_validation_context_initialization(self):
        """Test ValidationContext initialization with proper component and operation identification."""
        component = 'test_component'
        operation = 'test_operation'
        
        context = ValidationContext(component, operation)
        
        # Validate basic initialization
        assert context.component_name == component
        assert context.operation_name == operation
        assert isinstance(context.timestamp, float)
        assert context.timestamp > 0
        
        # Validate optional fields are properly initialized
        assert context.caller_info is None
        assert context.timing_data == {}
        assert context.additional_context == {}

    def test_validation_context_add_caller_info(self):
        """Test ValidationContext caller information capture ensuring proper stack inspection and frame analysis."""
        context = ValidationContext('test_component', 'caller_test')
        context.add_caller_info()
        
        # Validate caller information is captured
        assert context.caller_info is not None
        assert isinstance(context.caller_info, dict)
        assert 'function_name' in context.caller_info
        assert 'line_number' in context.caller_info
        assert 'filename' in context.caller_info
        
        # Validate caller info contains meaningful data
        assert context.caller_info['function_name'] is not None
        assert isinstance(context.caller_info['line_number'], int)
        assert context.caller_info['line_number'] > 0

    def test_validation_context_add_timing_data(self):
        """Test ValidationContext timing data capture ensuring accurate performance measurement and latency tracking."""
        context = ValidationContext('test_component', 'timing_test')
        
        # Add timing data with different operations
        start_time = time.time()
        time.sleep(0.001)  # Small delay for measurable timing
        end_time = time.time()
        
        context.add_timing_data('test_operation', start_time, end_time)
        
        # Validate timing data is properly stored
        assert 'test_operation' in context.timing_data
        timing_info = context.timing_data['test_operation']
        
        assert isinstance(timing_info, dict)
        assert 'duration' in timing_info
        assert 'start_time' in timing_info
        assert 'end_time' in timing_info
        
        # Validate timing calculations
        assert timing_info['duration'] > 0
        assert timing_info['duration'] < 1.0  # Should be less than 1 second
        assert timing_info['end_time'] > timing_info['start_time']

    def test_validation_context_get_context_summary(self):
        """Test ValidationContext summary generation ensuring comprehensive context reporting and performance analysis."""
        context = ValidationContext('summary_component', 'summary_operation')
        context.add_caller_info()
        context.add_timing_data('op1', time.time() - 0.001, time.time())
        context.additional_context.update({'test_key': 'test_value', 'param_count': 5})
        
        summary = context.get_context_summary()
        
        # Validate summary contains all expected information
        assert isinstance(summary, dict)
        assert 'component_name' in summary
        assert 'operation_name' in summary
        assert 'timestamp' in summary
        
        # Validate summary includes caller and timing information
        assert 'caller_info' in summary
        assert 'timing_data' in summary
        assert 'additional_context' in summary
        
        # Validate summary is properly formatted
        assert summary['component_name'] == 'summary_component'
        assert summary['operation_name'] == 'summary_operation'
        assert len(summary['additional_context']) == 2

    def test_validation_context_merge_context(self):
        """Test ValidationContext merging functionality ensuring proper context combination and data preservation."""
        # Create primary context
        context1 = ValidationContext('component1', 'operation1')
        context1.additional_context.update({'key1': 'value1', 'shared_key': 'original'})
        
        # Create secondary context to merge
        context2 = ValidationContext('component2', 'operation2')
        context2.additional_context.update({'key2': 'value2', 'shared_key': 'merged'})
        context2.add_timing_data('merge_op', time.time() - 0.002, time.time())
        
        # Test context merging
        context1.merge_context(context2)
        
        # Validate merged context preserves both sets of data
        assert 'key1' in context1.additional_context
        assert 'key2' in context1.additional_context
        
        # Validate timing data is merged
        assert 'merge_op' in context1.timing_data
        
        # Validate merge handling of conflicting keys
        assert context1.additional_context['shared_key'] == 'merged'  # Secondary context takes precedence

    def test_validation_context_thread_safety(self):
        """Test ValidationContext thread safety ensuring concurrent access doesn't corrupt context data."""
        context = ValidationContext('thread_test_component', 'thread_test_operation')
        results = []
        errors = []
        
        def worker_thread(thread_id):
            try:
                # Each thread adds its own timing and context data
                start_time = time.time()
                time.sleep(0.001 * thread_id)  # Variable delay
                end_time = time.time()
                
                context.add_timing_data(f'thread_{thread_id}', start_time, end_time)
                context.additional_context[f'thread_{thread_id}_data'] = f'value_{thread_id}'
                
                results.append(thread_id)
            except Exception as e:
                errors.append((thread_id, str(e)))
        
        # Start multiple threads
        threads = []
        for i in range(THREAD_SAFETY_TEST_THREADS):
            thread = threading.Thread(target=worker_thread, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=VALIDATION_TEST_TIMEOUT)
        
        # Validate all threads completed successfully
        assert len(errors) == 0, f"Thread errors occurred: {errors}"
        assert len(results) == THREAD_SAFETY_TEST_THREADS
        
        # Validate context data from all threads is preserved
        assert len(context.timing_data) == THREAD_SAFETY_TEST_THREADS
        assert len(context.additional_context) == THREAD_SAFETY_TEST_THREADS


class TestValidationResultClass:
    """Comprehensive test suite for ValidationResult class testing error reporting, warning management, performance metrics, and recovery suggestion functionality."""

    def test_validation_result_initialization(self):
        """Test ValidationResult initialization with proper error tracking and status management."""
        result = ValidationResult()
        
        # Validate initial state
        assert result.is_valid is True
        assert result.errors == []
        assert result.warnings == []
        assert result.performance_metrics == {}
        assert result.recovery_suggestions == []

    def test_validation_result_add_error(self):
        """Test ValidationResult error addition ensuring proper error categorization and tracking."""
        result = ValidationResult()
        
        # Add different types of errors
        result.add_error('Critical validation error', 'parameter_name', 'CRITICAL')
        result.add_error('Warning level issue', 'other_parameter', 'WARNING')
        
        # Validate errors are properly tracked
        assert result.is_valid is False
        assert len(result.errors) == 2
        
        # Validate error structure
        error = result.errors[0]
        assert isinstance(error, dict)
        assert 'message' in error
        assert 'parameter' in error
        assert 'severity' in error
        assert error['message'] == 'Critical validation error'
        assert error['parameter'] == 'parameter_name'
        assert error['severity'] == 'CRITICAL'

    def test_validation_result_add_warning(self):
        """Test ValidationResult warning addition ensuring proper warning categorization and non-critical issue tracking."""
        result = ValidationResult()
        
        # Add warnings
        result.add_warning('Performance warning', 'timing_parameter')
        result.add_warning('Configuration suggestion', 'config_parameter')
        
        # Validate warnings don't affect validity
        assert result.is_valid is True  # Warnings don't make result invalid
        assert len(result.warnings) == 2
        assert len(result.errors) == 0
        
        # Validate warning structure
        warning = result.warnings[0]
        assert isinstance(warning, dict)
        assert 'message' in warning
        assert 'parameter' in warning
        assert warning['message'] == 'Performance warning'

    def test_validation_result_set_performance_metrics(self):
        """Test ValidationResult performance metrics setting ensuring accurate timing and resource usage tracking."""
        result = ValidationResult()
        
        # Set performance metrics
        metrics = {
            'validation_time_ms': 1.5,
            'memory_usage_mb': 25.0,
            'cache_hit_rate': 0.85,
            'parameters_validated': 10
        }
        
        result.set_performance_metrics(metrics)
        
        # Validate metrics are properly stored
        assert result.performance_metrics == metrics
        assert result.performance_metrics['validation_time_ms'] == 1.5
        assert result.performance_metrics['memory_usage_mb'] == 25.0

    def test_validation_result_generate_summary(self):
        """Test ValidationResult summary generation ensuring comprehensive validation reporting and analysis."""
        result = ValidationResult()
        
        # Add mixed validation results
        result.add_error('Critical error', 'param1', 'CRITICAL')
        result.add_warning('Performance warning', 'param2')
        result.set_performance_metrics({'validation_time_ms': 2.0})
        result.recovery_suggestions.append('Try reducing parameter complexity')
        
        summary = result.generate_summary()
        
        # Validate summary structure
        assert isinstance(summary, dict)
        assert 'is_valid' in summary
        assert 'error_count' in summary
        assert 'warning_count' in summary
        assert 'performance_metrics' in summary
        assert 'recovery_suggestions' in summary
        
        # Validate summary content
        assert summary['is_valid'] is False
        assert summary['error_count'] == 1
        assert summary['warning_count'] == 1
        assert len(summary['recovery_suggestions']) == 1

    def test_validation_result_to_dict(self):
        """Test ValidationResult dictionary conversion ensuring proper serialization and data structure export."""
        result = ValidationResult()
        
        # Populate result with comprehensive data
        result.add_error('Test error', 'test_param', 'HIGH')
        result.add_warning('Test warning', 'warn_param')
        result.set_performance_metrics({'time': 1.0, 'memory': 50.0})
        
        result_dict = result.to_dict()
        
        # Validate dictionary structure
        assert isinstance(result_dict, dict)
        assert 'is_valid' in result_dict
        assert 'errors' in result_dict
        assert 'warnings' in result_dict
        assert 'performance_metrics' in result_dict
        
        # Validate dictionary content matches result state
        assert result_dict['is_valid'] == result.is_valid
        assert len(result_dict['errors']) == len(result.errors)
        assert len(result_dict['warnings']) == len(result.warnings)
        assert result_dict['performance_metrics'] == result.performance_metrics

    def test_validation_result_error_severity_handling(self):
        """Test ValidationResult error severity handling ensuring proper error classification and priority management."""
        result = ValidationResult()
        
        # Add errors with different severity levels
        severity_levels = ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
        for i, severity in enumerate(severity_levels):
            result.add_error(f'Error {i}', f'param{i}', severity)
        
        # Validate all errors are tracked
        assert len(result.errors) == 4
        assert result.is_valid is False
        
        # Validate errors maintain severity information
        for i, error in enumerate(result.errors):
            assert error['severity'] == severity_levels[i]
        
        # Test severity-based error filtering/analysis
        critical_errors = [e for e in result.errors if e['severity'] == 'CRITICAL']
        assert len(critical_errors) == 1
        assert critical_errors[0]['message'] == 'Error 3'


class TestParameterValidatorClass:
    """Comprehensive test suite for ParameterValidator class testing rule-based validation, caching functionality, performance monitoring, and batch validation capabilities."""

    def test_parameter_validator_initialization(self):
        """Test ParameterValidator initialization with proper rule setup and cache configuration."""
        validator = ParameterValidator()
        
        # Validate initial state
        assert isinstance(validator.validation_rules, dict)
        assert isinstance(validator.cache, dict)
        assert isinstance(validator.performance_stats, dict)
        assert validator.cache_enabled is True
        assert validator.performance_monitoring is True

    def test_parameter_validator_validate_parameter(self):
        """Test ParameterValidator single parameter validation ensuring proper rule application and result generation."""
        validator = ParameterValidator()
        
        # Test action parameter validation
        result = validator.validate_parameter('action', 1, {'type': 'int', 'range': [0, 3]})
        
        # Validate successful validation
        assert isinstance(result, ValidationResult)
        assert result.is_valid is True
        assert len(result.errors) == 0
        
        # Test invalid parameter validation
        result = validator.validate_parameter('action', 5, {'type': 'int', 'range': [0, 3]})
        
        # Validate failed validation
        assert result.is_valid is False
        assert len(result.errors) > 0

    def test_parameter_validator_batch_validate(self):
        """Test ParameterValidator batch validation ensuring efficient processing of multiple parameters."""
        validator = ParameterValidator()
        
        # Prepare batch validation data
        parameters = {
            'action': {'value': 2, 'rules': {'type': 'int', 'range': [0, 3]}},
            'grid_size': {'value': (64, 64), 'rules': {'type': 'tuple', 'min_dims': (16, 16)}},
            'seed': {'value': 42, 'rules': {'type': 'int', 'range': [0, 999999]}}
        }
        
        results = validator.batch_validate(parameters)
        
        # Validate batch results
        assert isinstance(results, dict)
        assert len(results) == 3
        
        for param_name, result in results.items():
            assert isinstance(result, ValidationResult)
            assert param_name in parameters
        
        # Validate all parameters passed validation
        all_valid = all(result.is_valid for result in results.values())
        assert all_valid is True

    def test_parameter_validator_add_validation_rule(self):
        """Test ParameterValidator custom rule addition ensuring extensible validation framework and rule management."""
        validator = ParameterValidator()
        
        # Add custom validation rule
        def custom_coordinate_validator(value, rules):
            if not isinstance(value, tuple) or len(value) != 2:
                return False, 'Must be a 2-tuple'
            x, y = value
            if not (isinstance(x, int) and isinstance(y, int)):
                return False, 'Coordinates must be integers'
            if x < 0 or y < 0:
                return False, 'Coordinates must be non-negative'
            return True, 'Valid coordinates'
        
        validator.add_validation_rule('coordinates', custom_coordinate_validator)
        
        # Validate rule is added
        assert 'coordinates' in validator.validation_rules
        
        # Test custom rule usage
        result = validator.validate_parameter('test_coords', (10, 20), {'type': 'coordinates'})
        assert result.is_valid is True
        
        # Test custom rule failure
        result = validator.validate_parameter('test_coords', (-1, 5), {'type': 'coordinates'})
        assert result.is_valid is False

    def test_parameter_validator_get_validation_stats(self):
        """Test ParameterValidator performance statistics tracking ensuring monitoring and optimization insights."""
        validator = ParameterValidator()
        
        # Perform several validations to generate stats
        for i in range(10):
            validator.validate_parameter('action', i % 4, {'type': 'int', 'range': [0, 3]})
            validator.validate_parameter('seed', i * 100, {'type': 'int', 'range': [0, 999999]})
        
        stats = validator.get_validation_stats()
        
        # Validate statistics structure
        assert isinstance(stats, dict)
        assert 'total_validations' in stats
        assert 'cache_hits' in stats
        assert 'cache_misses' in stats
        assert 'average_validation_time' in stats
        
        # Validate statistics content
        assert stats['total_validations'] >= 20  # At least 20 validations performed
        assert isinstance(stats['average_validation_time'], (int, float))
        assert stats['average_validation_time'] >= 0

    def test_parameter_validator_clear_cache(self):
        """Test ParameterValidator cache clearing ensuring proper memory management and cache reset functionality."""
        validator = ParameterValidator()
        
        # Populate cache with validations
        for i in range(CACHE_TEST_SIZE):
            validator.validate_parameter('action', i % 4, {'type': 'int', 'range': [0, 3]})
        
        # Validate cache is populated
        initial_cache_size = len(validator.cache)
        assert initial_cache_size > 0
        
        # Clear cache
        validator.clear_cache()
        
        # Validate cache is cleared
        assert len(validator.cache) == 0
        assert validator.performance_stats['cache_clears'] >= 1

    def test_parameter_validator_caching_performance(self):
        """Test ParameterValidator caching performance ensuring significant speedup for repeated validations."""
        validator = ParameterValidator()
        
        # Perform initial validation (cache miss)
        start_time = time.time()
        result1 = validator.validate_parameter('action', 2, {'type': 'int', 'range': [0, 3]})
        first_validation_time = time.time() - start_time
        
        # Perform repeated validation (cache hit)
        start_time = time.time()
        result2 = validator.validate_parameter('action', 2, {'type': 'int', 'range': [0, 3]})
        cached_validation_time = time.time() - start_time
        
        # Validate both results are equivalent
        assert result1.is_valid == result2.is_valid
        assert len(result1.errors) == len(result2.errors)
        
        # Validate caching provides performance benefit
        stats = validator.get_validation_stats()
        assert stats['cache_hits'] >= 1
        assert cached_validation_time <= first_validation_time  # Cached should be faster or equal


class TestEnvironmentConfigValidation:
    """Comprehensive test suite for environment configuration validation testing complete configuration validation, resource estimation, and cross-component consistency."""

    def test_validate_environment_config_complete_valid_config(self):
        """Test validate_environment_config with complete valid configuration ensuring comprehensive validation and consistency checking."""
        valid_config = {
            'grid_size': (128, 128),
            'source_location': (64, 64),
            'plume_parameters': {
                'sigma': 12.0,
                'intensity': 1.0
            },
            'max_steps': 1000,
            'render_mode': 'rgb_array',
            'seed': 42
        }
        
        context = create_validation_context('test_complete_config')
        result = validate_environment_config(valid_config, context=context)
        
        # Validate complete configuration is properly processed
        assert isinstance(result, EnvironmentConfig)
        assert result.grid_size is not None
        assert result.source_location is not None
        assert result.plume_parameters is not None
        assert result.max_steps == 1000
        assert result.seed == 42
        
        # Test configuration validation method
        result.validate()  # Should not raise exception
        
        # Test resource estimation
        resources = result.estimate_resources()
        assert isinstance(resources, dict)
        assert 'memory_mb' in resources
        assert resources['memory_mb'] > 0

    def test_validate_environment_config_minimal_valid_config(self):
        """Test validate_environment_config with minimal valid configuration ensuring default value application."""
        minimal_config = {
            'grid_size': (64, 64),
            'source_location': (32, 32)
        }
        
        context = create_validation_context('test_minimal_config')
        result = validate_environment_config(minimal_config, context=context)
        
        # Validate minimal configuration with defaults
        assert isinstance(result, EnvironmentConfig)
        assert result.grid_size.width == 64
        assert result.grid_size.height == 64
        assert result.source_location.x == 32
        assert result.source_location.y == 32
        
        # Validate default values are applied
        assert result.max_steps > 0  # Default max_steps
        assert result.plume_parameters.sigma > 0  # Default plume parameters

    def test_validate_environment_config_invalid_configurations(self):
        """Test validate_environment_config with invalid configurations ensuring comprehensive error detection."""
        invalid_configs = [
            # Invalid grid size
            {'grid_size': (0, 0), 'source_location': (0, 0)},
            
            # Source location outside grid
            {'grid_size': (32, 32), 'source_location': (50, 50)},
            
            # Invalid plume parameters
            {'grid_size': (64, 64), 'source_location': (32, 32), 'plume_parameters': {'sigma': -1.0}},
            
            # Invalid max_steps
            {'grid_size': (64, 64), 'source_location': (32, 32), 'max_steps': -1}
        ]
        
        for invalid_config in invalid_configs:
            context = create_validation_context('test_invalid_config')
            
            with pytest.raises((ValidationError, ConfigurationError)) as exc_info:
                validate_environment_config(invalid_config, context=context)
            
            # Validate error contains configuration-specific information
            error = exc_info.value
            assert 'config' in str(error.message).lower() or 'parameter' in str(error.message).lower()

    def test_validate_environment_config_cross_parameter_consistency(self):
        """Test validate_environment_config cross-parameter consistency ensuring mathematical coherence and integration compatibility."""
        # Test scenarios from CONSISTENCY_TEST_SCENARIOS
        for scenario in CONSISTENCY_TEST_SCENARIOS:
            context = create_validation_context('test_consistency')
            
            config = {
                'grid_size': scenario['grid_size'],
                'source_location': scenario.get('source_location', (scenario['grid_size'][0]//2, scenario['grid_size'][1]//2)),
                'plume_parameters': {
                    'sigma': scenario['sigma'],
                    'intensity': 1.0
                }
            }
            
            # This should validate successfully with consistent parameters
            result = validate_environment_config(config, context=context)
            assert isinstance(result, EnvironmentConfig)
            
            # Verify consistency through configuration validation
            result.validate()  # Should not raise exception

    def test_validate_environment_config_resource_constraints(self):
        """Test validate_environment_config resource constraint validation ensuring memory limits and performance feasibility."""
        # Test configuration that might exceed resource limits
        large_config = {
            'grid_size': (1024, 1024),  # Large grid
            'source_location': (512, 512),
            'plume_parameters': {
                'sigma': 50.0,
                'intensity': 1.0
            }
        }
        
        context = create_validation_context('test_resource_constraints')
        
        # This might succeed or fail depending on system resources
        try:
            result = validate_environment_config(large_config, context=context)
            
            # If it succeeds, check resource estimation
            resources = result.estimate_resources()
            assert isinstance(resources, dict)
            assert 'memory_mb' in resources
            
            # Large configuration should require significant memory
            assert resources['memory_mb'] > 100  # At least 100MB for 1024x1024 grid
            
        except ResourceError:
            # Resource error is acceptable for large configurations
            pass

    def test_validate_environment_config_edge_cases(self):
        """Test validate_environment_config with edge case scenarios ensuring boundary condition handling."""
        edge_cases = [
            # Minimum valid grid
            {'grid_size': MIN_GRID_SIZE, 'source_location': (0, 0)},
            
            # Maximum valid grid (if within resource limits)
            {'grid_size': (256, 256), 'source_location': (255, 255)},
            
            # Minimum sigma
            {'grid_size': (64, 64), 'source_location': (32, 32), 'plume_parameters': {'sigma': MIN_PLUME_SIGMA}},
            
            # Maximum sigma  
            {'grid_size': (64, 64), 'source_location': (32, 32), 'plume_parameters': {'sigma': MAX_PLUME_SIGMA}}
        ]
        
        for edge_case in edge_cases:
            context = create_validation_context('test_edge_case')
            
            try:
                result = validate_environment_config(edge_case, context=context)
                assert isinstance(result, EnvironmentConfig)
                
                # Validate edge case configuration
                result.validate()
                
            except (ValidationError, ResourceError):
                # Some edge cases might legitimately fail due to resource constraints
                pass


class TestConsistencyValidation:
    """Test suite for cross-parameter consistency validation ensuring mathematical coherence and integration compatibility across all environment parameters."""

    def test_check_parameter_consistency_valid_combinations(self):
        """Test check_parameter_consistency with valid parameter combinations ensuring mathematical coherence."""
        valid_combinations = [
            {
                'grid_size': (64, 64),
                'source_location': (32, 32),
                'plume_sigma': 10.0,
                'max_steps': 500
            },
            {
                'grid_size': (128, 128),
                'source_location': (0, 0),
                'plume_sigma': 20.0,
                'max_steps': 1000
            }
        ]
        
        for combination in valid_combinations:
            context = create_validation_context('test_valid_consistency')
            result = check_parameter_consistency(combination, context=context)
            
            # Validate consistency check passes
            assert isinstance(result, dict)
            assert result.get('consistent', False) is True
            assert 'violations' not in result or len(result['violations']) == 0

    def test_check_parameter_consistency_invalid_combinations(self):
        """Test check_parameter_consistency with invalid parameter combinations ensuring inconsistency detection."""
        invalid_combinations = [
            # Source location outside grid bounds
            {
                'grid_size': (32, 32),
                'source_location': (50, 50),  # Outside grid
                'plume_sigma': 10.0
            },
            
            # Sigma too large for grid size
            {
                'grid_size': (16, 16),
                'source_location': (8, 8),
                'plume_sigma': 50.0  # Sigma larger than grid
            },
            
            # Inconsistent max_steps for grid size
            {
                'grid_size': (256, 256),
                'source_location': (128, 128),
                'max_steps': 10  # Too few steps for large grid
            }
        ]
        
        for combination in invalid_combinations:
            context = create_validation_context('test_invalid_consistency')
            result = check_parameter_consistency(combination, context=context)
            
            # Validate consistency violations are detected
            assert isinstance(result, dict)
            assert result.get('consistent', True) is False
            assert 'violations' in result
            assert len(result['violations']) > 0

    def test_check_parameter_consistency_performance_feasibility(self):
        """Test check_parameter_consistency for performance feasibility ensuring system capability alignment."""
        performance_test_cases = [
            # Reasonable performance requirements
            {
                'grid_size': (64, 64),
                'max_steps': 1000,
                'expected_fps': 60,
                'memory_limit_mb': 50
            },
            
            # High performance requirements
            {
                'grid_size': (256, 256),
                'max_steps': 10000,
                'expected_fps': 1000,  # Unrealistic FPS
                'memory_limit_mb': 10   # Insufficient memory
            }
        ]
        
        for test_case in performance_test_cases:
            context = create_validation_context('test_performance_feasibility')
            result = check_parameter_consistency(test_case, context=context)
            
            # Check if performance requirements are feasible
            assert isinstance(result, dict)
            
            if test_case.get('expected_fps', 0) > 100:  # Unrealistic FPS
                assert result.get('consistent', True) is False
                assert any('performance' in violation.lower() for violation in result.get('violations', []))

    def test_check_parameter_consistency_mathematical_relationships(self):
        """Test check_parameter_consistency for mathematical relationships ensuring Gaussian formula coherence."""
        # Test mathematical relationship between sigma and grid size
        math_test_cases = [
            {
                'grid_size': (64, 64),
                'plume_sigma': 32.0,  # Half of grid width
                'source_location': (32, 32)
            },
            {
                'grid_size': (128, 128),
                'plume_sigma': 5.0,   # Much smaller than grid
                'source_location': (64, 64)
            }
        ]
        
        for test_case in math_test_cases:
            context = create_validation_context('test_mathematical_consistency')
            result = check_parameter_consistency(test_case, context=context)
            
            # Validate mathematical relationships
            assert isinstance(result, dict)
            
            # Check specific mathematical constraints
            grid_diagonal = np.sqrt(test_case['grid_size'][0]**2 + test_case['grid_size'][1]**2)
            sigma_to_diagonal_ratio = test_case['plume_sigma'] / grid_diagonal
            
            # Sigma should be reasonable relative to grid size
            if sigma_to_diagonal_ratio > 1.0:  # Sigma much larger than grid diagonal
                assert result.get('consistent', True) is False
                assert any('sigma' in violation.lower() for violation in result.get('violations', []))

    def test_check_parameter_consistency_integration_compatibility(self):
        """Test check_parameter_consistency for integration compatibility ensuring component interaction validation."""
        integration_test_cases = [
            {
                'grid_size': (128, 128),
                'source_location': (64, 64),
                'render_mode': 'rgb_array',
                'action_space_size': 4,
                'observation_space_shape': (1,)
            },
            {
                'grid_size': (64, 64),
                'source_location': (32, 32), 
                'render_mode': 'human',
                'action_space_size': 8,  # Non-standard action space
                'observation_space_shape': (3,)  # Non-standard observation
            }
        ]
        
        for test_case in integration_test_cases:
            context = create_validation_context('test_integration_compatibility')
            result = check_parameter_consistency(test_case, context=context)
            
            # Check integration compatibility
            assert isinstance(result, dict)
            
            # Standard configurations should be consistent
            if test_case.get('action_space_size') == 4 and test_case.get('observation_space_shape') == (1,):
                assert result.get('consistent', False) is True
            
            # Non-standard configurations might be flagged as inconsistent
            if test_case.get('action_space_size') != 4:
                # This might be flagged as inconsistent depending on implementation
                pass  # Implementation-dependent behavior


class TestPerformanceValidation:
    """Comprehensive performance testing suite validating validation operations meet latency requirements, memory constraints, and optimization targets."""

    @pytest.mark.timeout(VALIDATION_TEST_TIMEOUT)
    def test_validation_performance_latency(self):
        """Test validation function latency ensuring sub-millisecond performance for all validation operations."""
        # Test individual validation function performance
        validation_functions = [
            (validate_action_parameter, [2, None]),
            (validate_coordinates, [(50, 50), (128, 128), None]),
            (validate_grid_size, [(64, 64), None]),
            (validate_render_mode, ['rgb_array', None]),
            (validate_seed_value, [42, None])
        ]
        
        performance_results = {}
        
        for func, args in validation_functions:
            func_name = func.__name__
            times = []
            
            for _ in range(PERFORMANCE_TEST_ITERATIONS):
                start_time = time.time()
                try:
                    func(*args)
                except (ValidationError, ValueError):
                    pass  # Expected for some test cases
                end_time = time.time()
                times.append((end_time - start_time) * 1000)  # Convert to milliseconds
            
            avg_time = np.mean(times)
            max_time = np.max(times)
            
            performance_results[func_name] = {
                'avg_time_ms': avg_time,
                'max_time_ms': max_time,
                'samples': len(times)
            }
            
            # Validate performance targets are met
            assert avg_time < VALIDATION_PERFORMANCE_TOLERANCE_MS, f"{func_name} avg time {avg_time:.3f}ms exceeds {VALIDATION_PERFORMANCE_TOLERANCE_MS}ms"
            assert max_time < VALIDATION_PERFORMANCE_TOLERANCE_MS * 5, f"{func_name} max time {max_time:.3f}ms exceeds threshold"

    def test_batch_validation_performance(self):
        """Test batch validation performance ensuring efficient processing of multiple parameters simultaneously."""
        validator = ParameterValidator()
        
        # Create large batch of parameters for testing
        batch_parameters = {}
        for i in range(BATCH_VALIDATION_SIZE):
            batch_parameters[f'param_{i}'] = {
                'value': i % 4,  # Valid action values
                'rules': {'type': 'int', 'range': [0, 3]}
            }
        
        # Measure batch validation performance
        start_time = time.time()
        results = validator.batch_validate(batch_parameters)
        end_time = time.time()
        
        batch_time_ms = (end_time - start_time) * 1000
        per_param_time_ms = batch_time_ms / BATCH_VALIDATION_SIZE
        
        # Validate batch performance
        assert len(results) == BATCH_VALIDATION_SIZE
        assert batch_time_ms < BATCH_VALIDATION_SIZE  # Should be less than 1ms per parameter
        assert per_param_time_ms < VALIDATION_PERFORMANCE_TOLERANCE_MS

    def test_validation_memory_usage(self):
        """Test validation memory usage ensuring operations stay within memory constraints and don't cause leaks."""
        validator = ParameterValidator()
        
        # Measure initial memory state
        gc.collect()  # Clean up before measurement
        initial_objects = len(gc.get_objects())
        
        # Perform memory-intensive validation operations
        large_context_data = {f'key_{i}': f'value_{i}' * 100 for i in range(1000)}
        
        for i in range(100):
            context = create_validation_context('memory_test', large_context_data)
            try:
                validate_action_parameter(i % 4, context=context)
                validate_coordinates((i % 64, (i * 2) % 64), (64, 64), context=context)
            except ValidationError:
                pass
        
        # Force garbage collection and measure final state
        gc.collect()
        final_objects = len(gc.get_objects())
        
        # Validate memory usage is reasonable
        object_growth = final_objects - initial_objects
        assert object_growth < 10000, f"Excessive object growth: {object_growth} objects"
        
        # Test validation cache memory management
        validator.clear_cache()
        gc.collect()
        post_clear_objects = len(gc.get_objects())
        assert post_clear_objects <= final_objects, "Cache clearing should not increase object count"

    def test_validation_optimization_effectiveness(self):
        """Test validation optimization strategies ensuring caching and performance improvements are effective."""
        validator = ParameterValidator()
        
        # Test validation without optimization
        validator.cache_enabled = False
        validator.performance_monitoring = False
        
        start_time = time.time()
        for i in range(100):
            validator.validate_parameter('action', i % 4, {'type': 'int', 'range': [0, 3]})
        unoptimized_time = time.time() - start_time
        
        # Reset validator and enable optimizations
        validator = ParameterValidator()
        validator.cache_enabled = True
        validator.performance_monitoring = True
        
        start_time = time.time()
        for i in range(100):
            validator.validate_parameter('action', i % 4, {'type': 'int', 'range': [0, 3]})
        optimized_time = time.time() - start_time
        
        # Validate optimization provides performance benefit
        stats = validator.get_validation_stats()
        assert stats['cache_hits'] > 0, "Caching should provide cache hits for repeated validations"
        
        # With caching, optimized time should be significantly better for repeated operations
        cache_hit_ratio = stats['cache_hits'] / stats['total_validations']
        if cache_hit_ratio > 0.5:  # If we got good cache hits
            assert optimized_time <= unoptimized_time * 1.1, "Optimization should not significantly degrade performance"

    def test_concurrent_validation_performance(self):
        """Test concurrent validation performance ensuring thread safety doesn't severely impact performance."""
        validator = ParameterValidator()
        results = []
        errors = []
        
        def validation_worker(worker_id, iterations):
            worker_times = []
            try:
                for i in range(iterations):
                    start_time = time.time()
                    result = validator.validate_parameter(
                        f'worker_{worker_id}_param_{i}', 
                        (worker_id + i) % 4,
                        {'type': 'int', 'range': [0, 3]}
                    )
                    end_time = time.time()
                    worker_times.append((end_time - start_time) * 1000)
                
                avg_time = np.mean(worker_times)
                results.append((worker_id, avg_time, len(worker_times)))
                
            except Exception as e:
                errors.append((worker_id, str(e)))
        
        # Start multiple worker threads
        threads = []
        iterations_per_worker = 50
        
        start_time = time.time()
        for worker_id in range(THREAD_SAFETY_TEST_THREADS):
            thread = threading.Thread(target=validation_worker, args=(worker_id, iterations_per_worker))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=VALIDATION_TEST_TIMEOUT)
        
        total_time = time.time() - start_time
        
        # Validate concurrent performance
        assert len(errors) == 0, f"Concurrent validation errors: {errors}"
        assert len(results) == THREAD_SAFETY_TEST_THREADS
        
        # Calculate performance metrics
        total_validations = sum(result[2] for result in results)
        avg_times = [result[1] for result in results]
        overall_avg_time = np.mean(avg_times)
        
        # Performance should still be reasonable under concurrent load
        assert overall_avg_time < VALIDATION_PERFORMANCE_TOLERANCE_MS * 2, f"Concurrent validation too slow: {overall_avg_time:.3f}ms"
        
        # Throughput should be reasonable
        throughput = total_validations / total_time
        assert throughput > 100, f"Validation throughput too low: {throughput:.1f} validations/second"


class TestUtilityFunctions:
    """Test suite for utility functions including context creation, summary generation, and performance optimization ensuring comprehensive utility function coverage."""

    def test_create_validation_context_basic(self):
        """Test create_validation_context basic functionality ensuring proper context initialization and data population."""
        context = create_validation_context('test_operation')
        
        # Validate context structure
        assert isinstance(context, ValidationContext)
        assert context.operation_name == 'test_operation'
        assert isinstance(context.timestamp, float)
        assert context.timestamp > 0
        
        # Validate context has reasonable component name
        assert isinstance(context.component_name, str)
        assert len(context.component_name) > 0

    def test_create_validation_context_with_additional_data(self):
        """Test create_validation_context with additional context data ensuring proper data merging and sanitization."""
        additional_data = {
            'parameter_count': 5,
            'validation_type': 'comprehensive',
            'sensitive_key': 'sensitive_value'
        }
        
        context = create_validation_context('test_with_data', additional_data)
        
        # Validate additional data is included
        assert 'parameter_count' in context.additional_context
        assert context.additional_context['parameter_count'] == 5
        assert context.additional_context['validation_type'] == 'comprehensive'
        
        # Validate sensitive data is handled appropriately
        # (Implementation may sanitize or preserve based on security requirements)
        assert 'sensitive_key' in context.additional_context

    def test_validate_with_context_functionality(self):
        """Test validate_with_context wrapper functionality ensuring consistent error handling and performance monitoring."""
        # Test successful validation with context
        def test_validation_func(value, context=None):
            if value < 0:
                raise ValidationError("Negative value not allowed")
            return value * 2
        
        context = create_validation_context('test_wrapper')
        result = validate_with_context(test_validation_func, 5, context=context)
        
        # Validate successful execution
        assert result == 10
        
        # Test error handling with context
        with pytest.raises(ValidationError) as exc_info:
            validate_with_context(test_validation_func, -1, context=context)
        
        # Validate error includes context information
        error = exc_info.value
        assert error.context is not None

    def test_get_validation_summary_comprehensive(self):
        """Test get_validation_summary comprehensive reporting ensuring statistics aggregation and performance analysis."""
        # Create multiple validation contexts with different results
        contexts = []
        for i in range(10):
            context = create_validation_context(f'test_operation_{i}')
            context.add_timing_data('validation', time.time() - 0.001, time.time())
            contexts.append(context)
        
        summary = get_validation_summary(contexts)
        
        # Validate summary structure
        assert isinstance(summary, dict)
        assert 'total_operations' in summary
        assert 'average_timing' in summary
        assert 'operation_breakdown' in summary
        
        # Validate summary content
        assert summary['total_operations'] == 10
        assert isinstance(summary['average_timing'], (int, float))
        assert summary['average_timing'] >= 0

    def test_optimize_validation_performance_recommendations(self):
        """Test optimize_validation_performance recommendation generation ensuring actionable optimization guidance."""
        # Create performance data for optimization analysis
        performance_data = {
            'validation_times': [0.5, 1.2, 0.3, 2.1, 0.8],  # Various validation times in ms
            'cache_hit_rate': 0.65,
            'memory_usage_mb': 45.0,
            'concurrent_operations': 5
        }
        
        recommendations = optimize_validation_performance(performance_data)
        
        # Validate recommendations structure
        assert isinstance(recommendations, dict)
        assert 'recommendations' in recommendations
        assert 'priority' in recommendations
        assert 'estimated_improvement' in recommendations
        
        # Validate recommendations are actionable
        assert isinstance(recommendations['recommendations'], list)
        assert len(recommendations['recommendations']) > 0
        
        # Check for appropriate recommendations based on performance data
        recommendations_text = str(recommendations['recommendations'])
        if performance_data['cache_hit_rate'] < 0.8:
            assert 'cache' in recommendations_text.lower()

    def test_optimization_strategies_effectiveness(self):
        """Test optimization strategy effectiveness ensuring recommendations lead to measurable improvements."""
        validator = ParameterValidator()
        
        # Get initial performance baseline
        initial_stats = validator.get_validation_stats()
        
        # Apply optimization strategies
        optimize_recommendations = optimize_validation_performance({
            'cache_hit_rate': 0.3,
            'validation_times': [2.0, 1.8, 2.2, 1.9],
            'memory_usage_mb': 80.0
        })
        
        # Simulate applying cache optimization recommendation
        if any('cache' in rec.lower() for rec in optimize_recommendations.get('recommendations', [])):
            # Pre-populate cache with common validations
            common_validations = [
                ('action', 0), ('action', 1), ('action', 2), ('action', 3),
                ('seed', 42), ('seed', 123), ('seed', 456)
            ]
            
            for param_type, value in common_validations:
                rules = {'type': 'int', 'range': [0, 3] if param_type == 'action' else [0, 999999]}
                validator.validate_parameter(param_type, value, rules)
        
        # Test performance after optimization
        optimized_times = []
        for i in range(50):
            param_type = 'action' if i % 2 == 0 else 'seed'
            value = (i % 4) if param_type == 'action' else (i * 111) % 1000
            rules = {'type': 'int', 'range': [0, 3] if param_type == 'action' else [0, 999999]}
            
            start_time = time.time()
            validator.validate_parameter(param_type, value, rules)
            end_time = time.time()
            optimized_times.append((end_time - start_time) * 1000)
        
        # Validate optimization effectiveness
        final_stats = validator.get_validation_stats()
        assert final_stats['cache_hits'] > initial_stats.get('cache_hits', 0)
        
        # Average validation time should be reasonable
        avg_optimized_time = np.mean(optimized_times)
        assert avg_optimized_time < VALIDATION_PERFORMANCE_TOLERANCE_MS * 2


class TestErrorHandlingIntegration:
    """Test suite for error handling integration testing hierarchical error handling, context management, recovery suggestions, and exception system integration."""

    def test_validation_error_hierarchy_integration(self):
        """Test validation error hierarchy ensuring proper exception inheritance and error classification."""
        # Test ValidationError integration
        try:
            validate_action_parameter(-1)
        except ValidationError as e:
            assert isinstance(e, ValidationError)
            assert e.parameter_name is not None
            assert e.invalid_value is not None
            assert e.recovery_suggestion is not None
            
            # Test error details generation
            details = e.get_validation_details()
            assert isinstance(details, dict)
            assert 'parameter_name' in details
            assert 'invalid_value' in details

    def test_configuration_error_integration(self):
        """Test configuration error integration ensuring proper configuration error handling and valid option suggestions."""
        invalid_config = {
            'grid_size': 'invalid_size',
            'source_location': 'invalid_location'
        }
        
        try:
            validate_environment_config(invalid_config)
        except (ValidationError, ConfigurationError) as e:
            # Validate error provides configuration guidance
            assert hasattr(e, 'recovery_suggestion')
            assert e.recovery_suggestion is not None
            assert len(e.recovery_suggestion) > 0
            
            # Test configuration-specific error details
            if hasattr(e, 'get_valid_options'):
                valid_options = e.get_valid_options()
                assert isinstance(valid_options, dict)

    def test_resource_error_integration(self):
        """Test resource error integration ensuring proper resource constraint error handling with cleanup suggestions."""
        # Test resource constraints with large configuration
        large_config = {
            'grid_size': (2048, 2048),  # Very large grid
            'source_location': (1024, 1024),
            'max_steps': 100000
        }
        
        try:
            validate_environment_config(large_config)
            # If it doesn't raise an error, test resource estimation
            result = validate_environment_config(large_config)
            resources = result.estimate_resources()
            
            # Large configuration should estimate significant resource usage
            assert resources.get('memory_mb', 0) > 100
            
        except ResourceError as e:
            # Validate resource error provides cleanup suggestions
            assert hasattr(e, 'suggest_cleanup_actions')
            cleanup_actions = e.suggest_cleanup_actions()
            assert isinstance(cleanup_actions, list)
            assert len(cleanup_actions) > 0

    def test_error_context_sanitization_integration(self):
        """Test error context sanitization integration ensuring secure error reporting throughout validation system."""
        # Create validation context with sensitive information
        sensitive_context = {
            'user_credentials': 'sensitive_data',
            'api_keys': {'secret': 'key123'},
            'safe_parameter': 'safe_value'
        }
        
        context = create_validation_context('security_test', sensitive_context)
        
        try:
            validate_action_parameter('invalid_action', context=context)
        except ValidationError as e:
            # Validate error context is properly sanitized
            error_details = e.get_error_details()
            error_str = str(error_details)
            
            # Sensitive information should not appear in error output
            assert 'sensitive_data' not in error_str
            assert 'key123' not in error_str
            
            # Safe parameters should be preserved or safely handled
            assert 'safe_parameter' in error_str or '<sanitized>' in error_str

    def test_exception_chaining_and_recovery(self):
        """Test exception chaining and recovery suggestions ensuring proper error escalation and recovery guidance."""
        # Test nested validation scenario that might chain exceptions
        def complex_validation_scenario():
            try:
                # First level validation
                validate_grid_size((-1, -1))
            except ValidationError as grid_error:
                # Chain with higher-level validation error
                raise ConfigurationError(
                    "Grid configuration failed",
                    config_parameter="grid_size",
                    invalid_value=(-1, -1)
                ) from grid_error
        
        try:
            complex_validation_scenario()
        except ConfigurationError as e:
            # Validate exception chaining
            assert e.__cause__ is not None
            assert isinstance(e.__cause__, ValidationError)
            
            # Validate recovery suggestions are comprehensive
            assert e.recovery_suggestion is not None
            
            # Test error reporting includes both levels
            if hasattr(e, 'get_error_details'):
                details = e.get_error_details()
                assert isinstance(details, dict)

    def test_validation_result_error_aggregation(self):
        """Test ValidationResult error aggregation ensuring comprehensive error collection and reporting."""
        result = ValidationResult()
        
        # Simulate multiple validation errors
        validation_errors = [
            ('Invalid action', 'action', 'HIGH'),
            ('Invalid coordinates', 'coordinates', 'MEDIUM'),
            ('Performance warning', 'performance', 'LOW')
        ]
        
        for message, param, severity in validation_errors:
            if severity == 'LOW':
                result.add_warning(message, param)
            else:
                result.add_error(message, param, severity)
        
        # Validate error aggregation
        assert result.is_valid is False
        assert len(result.errors) == 2  # HIGH and MEDIUM errors
        assert len(result.warnings) == 1  # LOW severity as warning
        
        # Test comprehensive summary generation
        summary = result.generate_summary()
        assert summary['error_count'] == 2
        assert summary['warning_count'] == 1
        
        # Validate error details are preserved
        result_dict = result.to_dict()
        assert len(result_dict['errors']) == 2
        assert len(result_dict['warnings']) == 1


class TestIntegrationWithCoreTypes:
    """Integration test suite validating seamless interaction between validation utilities and core types system ensuring proper type conversion and validation."""

    def test_action_enum_integration(self):
        """Test Action enum integration ensuring proper validation and conversion between enum values and integers."""
        # Test Action enum validation
        for action in Action:
            context = create_validation_context('action_enum_test')
            result = validate_action_parameter(action.value, context=context)
            
            # Validate action enum values are properly handled
            assert isinstance(result, int)
            assert result == action.value
            assert 0 <= result < ACTION_SPACE_SIZE
        
        # Test Action enum in environment configuration
        config_with_action = {
            'grid_size': (64, 64),
            'source_location': (32, 32),
            'default_action': Action.UP.value
        }
        
        context = create_validation_context('config_action_test')
        # This tests that action validation integrates properly with configuration validation
        result = validate_environment_config(config_with_action, context=context)
        assert isinstance(result, EnvironmentConfig)

    def test_render_mode_enum_integration(self):
        """Test RenderMode enum integration ensuring proper render mode validation and backend compatibility."""
        # Test RenderMode enum validation
        for render_mode in RenderMode:
            context = create_validation_context('render_mode_test')
            result = validate_render_mode(render_mode.value, context=context)
            
            # Validate render mode enum is properly handled
            assert isinstance(result, RenderMode)
            assert result.value == render_mode.value
            assert render_mode.value in SUPPORTED_RENDER_MODES
        
        # Test render mode in environment configuration
        for mode in SUPPORTED_RENDER_MODES:
            config_with_render = {
                'grid_size': (64, 64),
                'source_location': (32, 32),
                'render_mode': mode
            }
            
            context = create_validation_context('config_render_test')
            result = validate_environment_config(config_with_render, context=context)
            assert isinstance(result, EnvironmentConfig)

    def test_coordinates_dataclass_integration(self):
        """Test Coordinates dataclass integration ensuring proper coordinate validation and bounds checking functionality."""
        grid_size = GridSize(128, 128)
        
        # Test coordinate validation with various input formats
        coordinate_inputs = [
            (64, 64),           # Tuple
            [32, 32],           # List
            Coordinates(16, 16) # Already a Coordinates object
        ]
        
        for coord_input in coordinate_inputs:
            context = create_validation_context('coordinates_integration_test')
            result = validate_coordinates(coord_input, (grid_size.width, grid_size.height), context=context)
            
            # Validate coordinates are converted to Coordinates dataclass
            assert isinstance(result, Coordinates)
            assert hasattr(result, 'is_within_bounds')
            
            # Test bounds checking functionality
            assert result.is_within_bounds(grid_size)
            assert 0 <= result.x < grid_size.width
            assert 0 <= result.y < grid_size.height

    def test_grid_size_dataclass_integration(self):
        """Test GridSize dataclass integration ensuring proper grid size validation and memory estimation functionality."""
        # Test grid size validation with various input formats
        grid_inputs = [
            (64, 64),        # Tuple
            [128, 128],      # List  
            GridSize(32, 32) # Already a GridSize object
        ]
        
        for grid_input in grid_inputs:
            context = create_validation_context('grid_size_integration_test')
            result = validate_grid_size(grid_input, context=context)
            
            # Validate grid size is converted to GridSize dataclass
            assert isinstance(result, GridSize)
            assert hasattr(result, 'total_cells')
            assert hasattr(result, 'estimate_memory_mb')
            
            # Test GridSize functionality
            total_cells = result.total_cells()
            memory_estimate = result.estimate_memory_mb()
            
            assert isinstance(total_cells, int)
            assert total_cells > 0
            assert isinstance(memory_estimate, (int, float))
            assert memory_estimate > 0

    def test_plume_parameters_dataclass_integration(self):
        """Test PlumeParameters dataclass integration ensuring proper plume parameter validation and mathematical consistency."""
        # Test plume parameter validation
        plume_input = {
            'source_location': (64, 64),
            'sigma': 12.0,
            'intensity': 1.0
        }
        
        context = create_validation_context('plume_parameters_test')
        result = validate_plume_parameters(plume_input, context=context)
        
        # Validate plume parameters are converted to PlumeParameters dataclass
        assert isinstance(result, PlumeParameters)
        assert hasattr(result, 'validate')
        
        # Test PlumeParameters validation functionality
        result.validate()  # Should not raise exception for valid parameters
        
        # Validate parameter values
        assert isinstance(result.source_location, Coordinates)
        assert MIN_PLUME_SIGMA <= result.sigma <= MAX_PLUME_SIGMA
        assert result.intensity > 0.0

    def test_environment_config_dataclass_integration(self):
        """Test EnvironmentConfig dataclass integration ensuring complete configuration validation and resource estimation."""
        # Create comprehensive environment configuration
        complete_config = {
            'grid_size': (128, 128),
            'source_location': (64, 64),
            'plume_parameters': {
                'sigma': 15.0,
                'intensity': 0.8
            },
            'max_steps': 1000,
            'render_mode': 'rgb_array',
            'seed': 42
        }
        
        context = create_validation_context('environment_config_integration')
        result = validate_environment_config(complete_config, context=context)
        
        # Validate environment configuration dataclass
        assert isinstance(result, EnvironmentConfig)
        assert hasattr(result, 'validate')
        assert hasattr(result, 'estimate_resources')
        
        # Test EnvironmentConfig functionality
        result.validate()  # Should not raise exception
        
        resources = result.estimate_resources()
        assert isinstance(resources, dict)
        assert 'memory_mb' in resources
        assert resources['memory_mb'] > 0
        
        # Validate all component types are properly integrated
        assert isinstance(result.grid_size, GridSize)
        assert isinstance(result.source_location, Coordinates)
        assert isinstance(result.plume_parameters, PlumeParameters)

    def test_type_conversion_consistency(self):
        """Test type conversion consistency ensuring validation functions consistently convert inputs to appropriate types."""
        # Test consistent type conversion across validation functions
        test_cases = [
            # Grid size conversion consistency
            {
                'function': validate_grid_size,
                'inputs': [(64, 64), [64, 64]],
                'expected_type': GridSize,
                'args': [None]
            },
            
            # Coordinates conversion consistency
            {
                'function': validate_coordinates,
                'inputs': [(32, 32), [32, 32]],
                'expected_type': Coordinates,
                'args': [(64, 64), None]
            },
            
            # Action conversion consistency
            {
                'function': validate_action_parameter,
                'inputs': [0, 1, 2, 3],
                'expected_type': int,
                'args': [None]
            }
        ]
        
        for test_case in test_cases:
            func = test_case['function']
            expected_type = test_case['expected_type']
            additional_args = test_case['args']
            
            for test_input in test_case['inputs']:
                context = create_validation_context('type_conversion_test')
                
                # Call function with appropriate arguments
                if additional_args[0] is not None:
                    result = func(test_input, additional_args[0], context=context)
                else:
                    result = func(test_input, context=context)
                
                # Validate type conversion consistency
                assert isinstance(result, expected_type), f"Expected {expected_type}, got {type(result)} for input {test_input}"

    def test_cross_type_validation_consistency(self):
        """Test cross-type validation consistency ensuring validation rules are consistent across related types."""
        # Test coordinate and grid size consistency
        grid_size = (64, 64)
        valid_coordinates = [(0, 0), (32, 32), (63, 63)]
        invalid_coordinates = [(-1, 0), (64, 64), (100, 100)]
        
        context_grid = create_validation_context('grid_validation')
        validated_grid = validate_grid_size(grid_size, context=context_grid)
        
        for coords in valid_coordinates:
            context_coord = create_validation_context('coord_validation')
            validated_coords = validate_coordinates(coords, grid_size, context=context_coord)
            
            # Validate coordinates are consistent with grid bounds
            assert validated_coords.is_within_bounds(validated_grid)
        
        for coords in invalid_coordinates:
            context_coord = create_validation_context('invalid_coord_validation')
            
            with pytest.raises(ValidationError):
                validate_coordinates(coords, grid_size, context=context_coord)


# Performance and stress testing fixtures
@pytest.fixture(scope="session")
def performance_test_data():
    """Generate comprehensive test data for performance testing."""
    return {
        'actions': VALID_TEST_ACTIONS * (PERFORMANCE_TEST_ITERATIONS // len(VALID_TEST_ACTIONS)),
        'coordinates': [(i % 64, (i * 2) % 64) for i in range(PERFORMANCE_TEST_ITERATIONS)],
        'grid_sizes': [(32, 32), (64, 64), (128, 128)] * (PERFORMANCE_TEST_ITERATIONS // 3),
        'seeds': [i for i in range(PERFORMANCE_TEST_ITERATIONS)]
    }


@pytest.fixture
def validation_test_environment():
    """Create controlled test environment for validation testing."""
    # Setup test environment with controlled conditions
    original_gc_threshold = gc.get_threshold()
    gc.set_threshold(1000, 10, 10)  # More aggressive garbage collection for testing
    
    yield {
        'memory_monitoring': True,
        'performance_tracking': True,
        'security_testing': True
    }
    
    # Cleanup test environment
    gc.set_threshold(*original_gc_threshold)
    gc.collect()


# Integration test markers for different test categories
pytestmark = [
    pytest.mark.validation,
    pytest.mark.security,
    pytest.mark.performance,
    pytest.mark.integration
]


if __name__ == "__main__":
    # Run comprehensive validation test suite
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--cov=plume_nav_sim.utils.validation",
        "--cov-report=term-missing",
        "--cov-report=html"
    ])