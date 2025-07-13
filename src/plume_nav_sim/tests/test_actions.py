"""
Comprehensive pytest test module for ActionInterfaceProtocol implementations.

This module provides exhaustive validation of ActionInterfaceProtocol implementations including
Continuous2DAction and CardinalDiscreteAction, ensuring standardized action processing across
different RL frameworks and navigation strategies. The test suite validates protocol compliance,
action space translation, bounds checking, Gymnasium compatibility, and vectorized multi-agent
action processing with performance benchmarks.

Key Components Tested:
- ActionInterfaceProtocol: Core protocol interface defining standardized action processing methods
- Continuous2DAction: Continuous 2D navigation control with velocity commands
- CardinalDiscreteAction: Discrete directional movement (N, S, E, W, NE, NW, SE, SW, Stop)
- create_action_interface: Factory function for configuration-driven action interface creation

Test Coverage Areas:
- Protocol compliance and interface implementation validation
- Action translation accuracy and bounds checking functionality
- Gymnasium action space compatibility and RL framework integration
- Vectorized action processing for multi-agent scenarios with performance benchmarks
- Performance requirements validation (≤33ms step latency with 100 agents)
- Configuration-driven instantiation via Hydra integration patterns
- Error handling and edge case management
- Backward compatibility with legacy action patterns

Performance Requirements:
- Action translation: <0.1ms per agent for minimal control overhead
- Validation: <0.05ms per agent for constraint checking
- Multi-agent scaling: ≤33ms step latency with 100 concurrent agents
- Memory efficiency: <100 bytes per action for structured representations

Examples:
    Basic ActionInterfaceProtocol compliance testing:
    >>> def test_protocol_compliance():
    ...     action_interface = Continuous2DAction(max_velocity=2.0)
    ...     assert isinstance(action_interface, ActionInterfaceProtocol)
    ...     assert hasattr(action_interface, 'translate_action')
    ...     assert hasattr(action_interface, 'validate_action')
    ...     assert hasattr(action_interface, 'get_action_space')

    Continuous action interface testing:
    >>> def test_continuous_action_translation():
    ...     action_interface = Continuous2DAction(max_velocity=2.0, max_angular_velocity=45.0)
    ...     action = np.array([1.5, 15.0])
    ...     nav_command = action_interface.translate_action(action)
    ...     assert nav_command['linear_velocity'] == 1.5
    ...     assert nav_command['angular_velocity'] == 15.0

    Discrete action interface testing:
    >>> def test_discrete_action_translation():
    ...     action_interface = CardinalDiscreteAction(speed=1.0)
    ...     action = 2  # East direction
    ...     nav_command = action_interface.translate_action(action)
    ...     assert nav_command['direction'] == 'EAST'
    ...     assert nav_command['linear_velocity'] == 1.0

    Performance validation testing:
    >>> def test_action_performance():
    ...     action_interface = Continuous2DAction()
    ...     actions = [np.array([1.0, 0.0]) for _ in range(100)]
    ...     start_time = time.perf_counter()
    ...     for action in actions:
    ...         action_interface.translate_action(action)
    ...     duration = time.perf_counter() - start_time
    ...     assert duration < 0.033  # 33ms requirement for 100 agents

Dependencies Integration:
- pytest ≥7.4.0: Testing framework with fixtures, parameterization, and markers
- numpy ≥1.26.0: Numerical operations and array validation for action processing
- gymnasium 0.29.x: Action space compatibility and RL framework integration
- time: High-resolution performance measurement for latency validation
- typing: Static type checking support for test method signatures
- unittest.mock: Mocking utilities for isolated component testing
"""

from __future__ import annotations
import time
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
from unittest.mock import patch, MagicMock

import pytest
import numpy as np
from gymnasium import spaces

# Import ActionInterfaceProtocol and implementations
from src.plume_nav_sim.core.protocols import ActionInterfaceProtocol
from src.plume_nav_sim.core.actions import (
    Continuous2DAction,
    CardinalDiscreteAction,
    create_action_interface,
    list_available_action_types,
    validate_action_config,
    get_action_space_info
)


# Test configuration constants
NUMERICAL_PRECISION_TOLERANCE = 1e-6
PERFORMANCE_LATENCY_THRESHOLD_MS = 33  # ≤33ms per step with 100 agents
SINGLE_ACTION_LATENCY_THRESHOLD_MS = 0.1  # <0.1ms per agent for translation
VALIDATION_LATENCY_THRESHOLD_MS = 0.05  # <0.05ms per agent for validation
MEMORY_EFFICIENCY_THRESHOLD_BYTES = 100  # <100 bytes per action


class TestActionInterfaceProtocol:
    """
    Test suite for ActionInterfaceProtocol compliance and interface validation.
    
    This test class validates that ActionInterfaceProtocol implementations correctly
    define the standardized action processing interface with proper method signatures,
    protocol inheritance, and interface compliance across different action types.
    """

    def test_protocol_compliance(self):
        """
        Test that ActionInterfaceProtocol defines the correct interface methods.
        
        Validates that the protocol includes all required methods with proper
        signatures for standardized action processing across implementations.
        """
        # Verify protocol has required methods
        assert hasattr(ActionInterfaceProtocol, 'translate_action')
        assert hasattr(ActionInterfaceProtocol, 'validate_action')
        assert hasattr(ActionInterfaceProtocol, 'get_action_space')
        
        # Verify methods are callable (protocol methods)
        assert callable(getattr(ActionInterfaceProtocol, 'translate_action', None))
        assert callable(getattr(ActionInterfaceProtocol, 'validate_action', None))
        assert callable(getattr(ActionInterfaceProtocol, 'get_action_space', None))

    def test_interface_implementation(self):
        """
        Test that concrete implementations properly implement ActionInterfaceProtocol.
        
        Validates that Continuous2DAction and CardinalDiscreteAction classes
        implement the protocol interface and can be used interchangeably.
        """
        # Test Continuous2DAction implements protocol
        continuous_action = Continuous2DAction(max_velocity=2.0, max_angular_velocity=45.0)
        assert isinstance(continuous_action, ActionInterfaceProtocol)
        
        # Test CardinalDiscreteAction implements protocol
        discrete_action = CardinalDiscreteAction(speed=1.0, use_8_directions=True)
        assert isinstance(discrete_action, ActionInterfaceProtocol)
        
        # Test both have required methods
        for action_interface in [continuous_action, discrete_action]:
            assert hasattr(action_interface, 'translate_action')
            assert hasattr(action_interface, 'validate_action')
            assert hasattr(action_interface, 'get_action_space')
            assert callable(action_interface.translate_action)
            assert callable(action_interface.validate_action)
            assert callable(action_interface.get_action_space)

    def test_method_signatures(self):
        """
        Test that protocol method signatures are correctly implemented.
        
        Validates that implementations provide methods with compatible signatures
        for translate_action, validate_action, and get_action_space.
        """
        continuous_action = Continuous2DAction()
        discrete_action = CardinalDiscreteAction()
        
        # Test translate_action signature compatibility
        continuous_result = continuous_action.translate_action(np.array([1.0, 0.0]))
        assert isinstance(continuous_result, dict)
        assert 'linear_velocity' in continuous_result
        assert 'angular_velocity' in continuous_result
        
        discrete_result = discrete_action.translate_action(1)  # North direction
        assert isinstance(discrete_result, dict)
        assert 'linear_velocity' in discrete_result
        assert 'angular_velocity' in discrete_result
        
        # Test validate_action signature compatibility
        validated_continuous = continuous_action.validate_action(np.array([2.0, 30.0]))
        assert isinstance(validated_continuous, np.ndarray)
        
        validated_discrete = discrete_action.validate_action(5)
        assert isinstance(validated_discrete, (int, np.integer))
        
        # Test get_action_space signature compatibility
        continuous_space = continuous_action.get_action_space()
        discrete_space = discrete_action.get_action_space()
        
        if continuous_space is not None:
            assert isinstance(continuous_space, spaces.Space)
        if discrete_space is not None:
            assert isinstance(discrete_space, spaces.Space)

    def test_protocol_inheritance(self):
        """
        Test that protocol inheritance works correctly for type checking.
        
        Validates that isinstance checks work properly and that the protocol
        can be used for type annotations and runtime validation.
        """
        # Test that implementations are recognized as protocol instances
        continuous_action = Continuous2DAction()
        discrete_action = CardinalDiscreteAction()
        
        assert isinstance(continuous_action, ActionInterfaceProtocol)
        assert isinstance(discrete_action, ActionInterfaceProtocol)
        
        # Test that a list of different implementations can be typed consistently
        action_interfaces: List[ActionInterfaceProtocol] = [
            continuous_action,
            discrete_action
        ]
        
        # Validate all can be used through protocol interface
        for action_interface in action_interfaces:
            # Protocol methods should work
            if hasattr(action_interface, 'get_action_space'):
                space = action_interface.get_action_space()
                if space is not None:
                    assert isinstance(space, spaces.Space)


class TestContinuous2DAction:
    """
    Test suite for Continuous2DAction implementation.
    
    This test class validates continuous 2D action interface functionality including
    action translation, velocity bounds validation, action space creation, clipping
    behavior, and configuration options for continuous navigation control.
    """

    def test_continuous_action_translation(self):
        """
        Test continuous action translation to navigation commands.
        
        Validates that continuous actions are properly translated to navigation
        commands with correct linear and angular velocity values.
        """
        action_interface = Continuous2DAction(
            max_velocity=2.0,
            max_angular_velocity=45.0,
            min_velocity=-1.0,
            min_angular_velocity=-45.0
        )
        
        # Test standard 2D action translation
        action = np.array([1.5, 20.0])
        result = action_interface.translate_action(action)
        
        assert isinstance(result, dict)
        assert result['linear_velocity'] == 1.5
        assert result['angular_velocity'] == 20.0
        assert result['action_type'] == 'continuous_2d'
        
        # Test single component action (linear only)
        action = np.array([1.0])
        result = action_interface.translate_action(action)
        
        assert result['linear_velocity'] == 1.0
        assert result['angular_velocity'] == 0.0
        assert result['action_type'] == 'continuous_2d'
        
        # Test scalar action
        action = np.array(0.5)  # Scalar
        result = action_interface.translate_action(action)
        
        assert result['linear_velocity'] == 0.5
        assert result['angular_velocity'] == 0.0
        assert result['action_type'] == 'continuous_2d'

    def test_velocity_bounds_validation(self):
        """
        Test velocity bounds validation and clipping behavior.
        
        Validates that actions are properly clipped to configured bounds
        and that invalid values are handled correctly.
        """
        action_interface = Continuous2DAction(
            max_velocity=2.0,
            max_angular_velocity=45.0,
            min_velocity=-1.0,
            min_angular_velocity=-30.0
        )
        
        # Test action within bounds
        action = np.array([1.5, 20.0])
        validated = action_interface.validate_action(action)
        np.testing.assert_allclose(validated, [1.5, 20.0], atol=NUMERICAL_PRECISION_TOLERANCE)
        
        # Test action exceeding bounds (should be clipped)
        action = np.array([3.0, 60.0])
        validated = action_interface.validate_action(action)
        np.testing.assert_allclose(validated, [2.0, 45.0], atol=NUMERICAL_PRECISION_TOLERANCE)
        
        # Test action below bounds (should be clipped)
        action = np.array([-2.0, -50.0])
        validated = action_interface.validate_action(action)
        np.testing.assert_allclose(validated, [-1.0, -30.0], atol=NUMERICAL_PRECISION_TOLERANCE)
        
        # Test invalid values (NaN, inf)
        action = np.array([np.nan, np.inf])
        validated = action_interface.validate_action(action)
        np.testing.assert_allclose(validated, [0.0, 0.0], atol=NUMERICAL_PRECISION_TOLERANCE)

    def test_action_space_creation(self):
        """
        Test Gymnasium action space creation for continuous actions.
        
        Validates that the action space is properly created with correct bounds,
        shape, and dtype for RL framework compatibility.
        """
        action_interface = Continuous2DAction(
            max_velocity=2.0,
            max_angular_velocity=45.0,
            min_velocity=-1.0,
            min_angular_velocity=-30.0
        )
        
        action_space = action_interface.get_action_space()
        
        # Skip test if Gymnasium not available
        if action_space is None:
            pytest.skip("Gymnasium not available")
        
        assert isinstance(action_space, spaces.Box)
        assert action_space.shape == (2,)
        assert action_space.dtype == np.float32
        
        # Test bounds
        np.testing.assert_allclose(action_space.low, [-1.0, -30.0], atol=NUMERICAL_PRECISION_TOLERANCE)
        np.testing.assert_allclose(action_space.high, [2.0, 45.0], atol=NUMERICAL_PRECISION_TOLERANCE)
        
        # Test that actions can be sampled from space
        sample_action = action_space.sample()
        assert action_space.contains(sample_action)
        
        # Test that validated actions are within space
        validated = action_interface.validate_action(sample_action)
        assert action_space.contains(validated)

    def test_action_clipping(self):
        """
        Test action clipping behavior for out-of-bounds actions.
        
        Validates that actions exceeding bounds are properly clipped while
        maintaining valid navigation command structure.
        """
        action_interface = Continuous2DAction(max_velocity=1.0, max_angular_velocity=30.0)
        
        # Test clipping in translate_action (should use validate_action internally)
        extreme_action = np.array([5.0, 90.0])
        result = action_interface.translate_action(extreme_action)
        
        assert result['linear_velocity'] <= 1.0
        assert result['angular_velocity'] <= 30.0
        assert result['linear_velocity'] >= -1.0  # Default min_velocity
        assert result['angular_velocity'] >= -30.0  # Default min_angular_velocity
        
        # Test translation preserves clipped values
        clipped_action = np.array([1.0, 30.0])  # At bounds
        result = action_interface.translate_action(clipped_action)
        
        assert result['linear_velocity'] == 1.0
        assert result['angular_velocity'] == 30.0

    def test_configuration_options(self):
        """
        Test configuration options and dynamic bounds adjustment.
        
        Validates that bounds can be adjusted dynamically and that
        configuration changes are properly validated.
        """
        action_interface = Continuous2DAction(max_velocity=2.0, max_angular_velocity=45.0)
        
        # Test initial configuration
        assert action_interface.get_max_velocity() == 2.0
        assert action_interface.get_max_angular_velocity() == 45.0
        
        # Test bounds adjustment
        action_interface.set_bounds(max_velocity=3.0, max_angular_velocity=60.0)
        assert action_interface.get_max_velocity() == 3.0
        assert action_interface.get_max_angular_velocity() == 60.0
        
        # Test partial bounds update
        action_interface.set_bounds(max_velocity=2.5)
        assert action_interface.get_max_velocity() == 2.5
        assert action_interface.get_max_angular_velocity() == 60.0  # Unchanged
        
        # Test invalid bounds (min >= max)
        with pytest.raises(ValueError):
            action_interface.set_bounds(min_velocity=1.0, max_velocity=0.5)
        
        # Test action space updates after bounds change
        new_space = action_interface.get_action_space()
        if new_space is not None:
            assert new_space.high[0] == 2.5  # Updated max_velocity


class TestCardinalDiscreteAction:
    """
    Test suite for CardinalDiscreteAction implementation.
    
    This test class validates discrete directional action interface functionality
    including action translation, mapping validation, cardinal directions, speed
    configuration, and action space bounds for discrete navigation control.
    """

    def test_discrete_action_translation(self):
        """
        Test discrete action translation to navigation commands.
        
        Validates that discrete action indices are properly translated to
        navigation commands with correct velocity vectors and directions.
        """
        # Test 8-direction mode with stay action
        action_interface = CardinalDiscreteAction(
            speed=1.0,
            use_8_directions=True,
            include_stay_action=True
        )
        
        # Test stay action (typically index 0)
        result = action_interface.translate_action(0)
        assert result['direction'] == 'STAY'
        assert result['linear_velocity'] == 0.0
        assert result['angular_velocity'] == 0.0
        assert result['action_type'] == 'cardinal_discrete'
        
        # Test cardinal directions (North should be index 1 with stay enabled)
        result = action_interface.translate_action(1)
        assert result['direction'] == 'NORTH'
        assert result['linear_velocity'] == 1.0
        assert result['angular_velocity'] == 0.0
        assert result['velocity_y'] == -1.0  # North is negative y
        
        # Test diagonal direction (should have normalized speed)
        result = action_interface.translate_action(5)  # Should be Northeast
        assert result['direction'] == 'NORTHEAST'
        assert abs(result['linear_velocity'] - 1.0/np.sqrt(2)) < NUMERICAL_PRECISION_TOLERANCE
        assert result['angular_velocity'] == 0.0

    def test_action_mapping_validation(self):
        """
        Test action mapping validation and consistency.
        
        Validates that action mappings are consistent, indices are valid,
        and direction names match expected values.
        """
        action_interface = CardinalDiscreteAction(speed=1.0, use_8_directions=True)
        
        # Get action mapping
        mapping = action_interface.get_action_mapping()
        available_actions = action_interface.get_available_actions()
        
        assert len(mapping) == len(available_actions)
        assert len(available_actions) == action_interface.get_num_actions()
        
        # Test all actions are valid
        for action_idx in available_actions:
            assert action_idx in mapping
            direction = action_interface.get_direction_for_action(action_idx)
            assert direction == mapping[action_idx]
            
            # Test translation works for each action
            result = action_interface.translate_action(action_idx)
            assert result['direction'] == direction
            assert 'linear_velocity' in result
            assert 'angular_velocity' in result

    def test_cardinal_directions(self):
        """
        Test cardinal direction mappings and velocity vectors.
        
        Validates that cardinal directions (N, S, E, W) have correct
        velocity vectors and movement behavior.
        """
        action_interface = CardinalDiscreteAction(speed=2.0, use_8_directions=False, include_stay_action=False)
        
        # Expected cardinal directions and their velocity vectors
        expected_directions = {
            'NORTH': (0.0, -2.0),   # Negative y (up)
            'SOUTH': (0.0, 2.0),    # Positive y (down)
            'EAST': (2.0, 0.0),     # Positive x (right)
            'WEST': (-2.0, 0.0),    # Negative x (left)
        }
        
        mapping = action_interface.get_action_mapping()
        
        for action_idx, direction in mapping.items():
            if direction in expected_directions:
                result = action_interface.translate_action(action_idx)
                expected_vx, expected_vy = expected_directions[direction]
                
                assert abs(result['velocity_x'] - expected_vx) < NUMERICAL_PRECISION_TOLERANCE
                assert abs(result['velocity_y'] - expected_vy) < NUMERICAL_PRECISION_TOLERANCE
                assert abs(result['linear_velocity'] - 2.0) < NUMERICAL_PRECISION_TOLERANCE

    def test_speed_configuration(self):
        """
        Test speed configuration and dynamic adjustment.
        
        Validates that movement speed can be configured and adjusted
        dynamically while maintaining direction relationships.
        """
        action_interface = CardinalDiscreteAction(speed=1.0)
        
        # Test initial speed
        assert action_interface.get_speed() == 1.0
        
        # Test movement with initial speed
        result = action_interface.translate_action(1)  # North
        assert abs(result['linear_velocity'] - 1.0) < NUMERICAL_PRECISION_TOLERANCE
        
        # Test speed adjustment
        action_interface.set_speed(2.5)
        assert action_interface.get_speed() == 2.5
        
        # Test movement with new speed
        result = action_interface.translate_action(1)  # North
        assert abs(result['linear_velocity'] - 2.5) < NUMERICAL_PRECISION_TOLERANCE
        
        # Test diagonal speed adjustment (should be normalized)
        if action_interface._use_8_directions:
            diagonal_result = action_interface.translate_action(5)  # Assuming Northeast
            expected_diagonal_speed = 2.5 / np.sqrt(2)
            assert abs(diagonal_result['linear_velocity'] - expected_diagonal_speed) < NUMERICAL_PRECISION_TOLERANCE
        
        # Test invalid speed
        with pytest.raises(ValueError):
            action_interface.set_speed(-1.0)
        
        with pytest.raises(ValueError):
            action_interface.set_speed(0.0)

    def test_action_space_bounds(self):
        """
        Test action space bounds for discrete actions.
        
        Validates that the discrete action space has correct bounds
        and that all actions are within the valid range.
        """
        # Test 4-direction with stay
        action_interface = CardinalDiscreteAction(
            speed=1.0,
            use_8_directions=False,
            include_stay_action=True
        )
        
        action_space = action_interface.get_action_space()
        
        # Skip test if Gymnasium not available
        if action_space is None:
            pytest.skip("Gymnasium not available")
        
        assert isinstance(action_space, spaces.Discrete)
        assert action_space.n == 5  # STAY + 4 cardinal directions
        
        # Test all actions are valid
        for action_idx in range(action_space.n):
            assert action_space.contains(action_idx)
            # Should be able to translate without error
            result = action_interface.translate_action(action_idx)
            assert isinstance(result, dict)
        
        # Test 8-direction with stay
        action_interface_8 = CardinalDiscreteAction(
            speed=1.0,
            use_8_directions=True,
            include_stay_action=True
        )
        
        action_space_8 = action_interface_8.get_action_space()
        if action_space_8 is not None:
            assert action_space_8.n == 9  # STAY + 8 directions
        
        # Test 8-direction without stay
        action_interface_no_stay = CardinalDiscreteAction(
            speed=1.0,
            use_8_directions=True,
            include_stay_action=False
        )
        
        action_space_no_stay = action_interface_no_stay.get_action_space()
        if action_space_no_stay is not None:
            assert action_space_no_stay.n == 8  # 8 directions only


class TestGymnasiumCompatibility:
    """
    Test suite for Gymnasium action space compatibility.
    
    This test class validates that action interfaces properly integrate with
    Gymnasium RL framework including action space compliance, sampling,
    containment, dtype validation, and framework integration.
    """

    def test_action_space_compliance(self):
        """
        Test that action spaces comply with Gymnasium standards.
        
        Validates that action spaces follow Gymnasium conventions for
        shape, dtype, bounds, and API compatibility.
        """
        continuous_action = Continuous2DAction(max_velocity=2.0, max_angular_velocity=45.0)
        discrete_action = CardinalDiscreteAction(speed=1.0)
        
        continuous_space = continuous_action.get_action_space()
        discrete_space = discrete_action.get_action_space()
        
        # Skip test if Gymnasium not available
        if continuous_space is None or discrete_space is None:
            pytest.skip("Gymnasium not available")
        
        # Test continuous space compliance
        assert isinstance(continuous_space, spaces.Box)
        assert len(continuous_space.shape) == 1  # Should be 1D
        assert continuous_space.shape[0] == 2    # [linear_velocity, angular_velocity]
        assert continuous_space.dtype == np.float32
        assert hasattr(continuous_space, 'low')
        assert hasattr(continuous_space, 'high')
        assert hasattr(continuous_space, 'sample')
        assert hasattr(continuous_space, 'contains')
        
        # Test discrete space compliance
        assert isinstance(discrete_space, spaces.Discrete)
        assert hasattr(discrete_space, 'n')
        assert discrete_space.n > 0
        assert hasattr(discrete_space, 'sample')
        assert hasattr(discrete_space, 'contains')

    def test_space_sampling(self):
        """
        Test action space sampling functionality.
        
        Validates that action spaces can generate valid samples and that
        these samples can be processed by the action interfaces.
        """
        continuous_action = Continuous2DAction()
        discrete_action = CardinalDiscreteAction()
        
        continuous_space = continuous_action.get_action_space()
        discrete_space = discrete_action.get_action_space()
        
        # Skip test if Gymnasium not available
        if continuous_space is None or discrete_space is None:
            pytest.skip("Gymnasium not available")
        
        # Test continuous space sampling
        for _ in range(10):  # Multiple samples to test randomness
            sample = continuous_space.sample()
            assert continuous_space.contains(sample)
            
            # Sample should be translatable
            result = continuous_action.translate_action(sample)
            assert isinstance(result, dict)
            assert 'linear_velocity' in result
            assert 'angular_velocity' in result
        
        # Test discrete space sampling
        for _ in range(10):
            sample = discrete_space.sample()
            assert discrete_space.contains(sample)
            
            # Sample should be translatable
            result = discrete_action.translate_action(sample)
            assert isinstance(result, dict)
            assert 'direction' in result
            assert 'linear_velocity' in result

    def test_space_containment(self):
        """
        Test action space containment validation.
        
        Validates that containment checks work correctly for valid and
        invalid actions across different action space types.
        """
        continuous_action = Continuous2DAction(max_velocity=2.0, max_angular_velocity=45.0)
        discrete_action = CardinalDiscreteAction(speed=1.0)
        
        continuous_space = continuous_action.get_action_space()
        discrete_space = discrete_action.get_action_space()
        
        # Skip test if Gymnasium not available
        if continuous_space is None or discrete_space is None:
            pytest.skip("Gymnasium not available")
        
        # Test continuous space containment
        valid_continuous = np.array([1.0, 20.0], dtype=np.float32)
        invalid_continuous = np.array([5.0, 100.0], dtype=np.float32)
        
        assert continuous_space.contains(valid_continuous)
        assert not continuous_space.contains(invalid_continuous)
        
        # Test discrete space containment
        valid_discrete = 0
        invalid_discrete = discrete_space.n + 10
        
        assert discrete_space.contains(valid_discrete)
        assert not discrete_space.contains(invalid_discrete)

    def test_action_dtype_validation(self):
        """
        Test action data type validation and conversion.
        
        Validates that action interfaces properly handle different input
        data types and convert them appropriately for processing.
        """
        continuous_action = Continuous2DAction()
        discrete_action = CardinalDiscreteAction()
        
        # Test continuous action with different dtypes
        float64_action = np.array([1.0, 0.5], dtype=np.float64)
        float32_action = np.array([1.0, 0.5], dtype=np.float32)
        int_action = np.array([1, 0], dtype=np.int32)
        
        for action in [float64_action, float32_action, int_action]:
            result = continuous_action.translate_action(action)
            assert isinstance(result, dict)
            assert isinstance(result['linear_velocity'], float)
            assert isinstance(result['angular_velocity'], float)
        
        # Test discrete action with different types
        int_action = 1
        numpy_int = np.int32(1)
        numpy_array = np.array([1])
        
        for action in [int_action, numpy_int, numpy_array]:
            result = discrete_action.translate_action(action)
            assert isinstance(result, dict)
            assert 'direction' in result

    def test_rl_framework_integration(self):
        """
        Test integration patterns with RL frameworks.
        
        Validates that action interfaces work correctly in typical RL
        framework usage patterns including policy sampling and training.
        """
        # Simulate RL framework usage pattern
        continuous_action = Continuous2DAction(max_velocity=2.0, max_angular_velocity=45.0)
        discrete_action = CardinalDiscreteAction(speed=1.0)
        
        action_spaces = []
        continuous_space = continuous_action.get_action_space()
        discrete_space = discrete_action.get_action_space()
        
        if continuous_space is not None:
            action_spaces.append((continuous_action, continuous_space))
        if discrete_space is not None:
            action_spaces.append((discrete_action, discrete_space))
        
        # Skip test if Gymnasium not available
        if not action_spaces:
            pytest.skip("Gymnasium not available")
        
        # Test typical RL workflow
        for action_interface, action_space in action_spaces:
            # 1. Sample action from policy (simulated)
            sampled_action = action_space.sample()
            
            # 2. Validate action is in space
            assert action_space.contains(sampled_action)
            
            # 3. Translate action for environment
            nav_command = action_interface.translate_action(sampled_action)
            assert isinstance(nav_command, dict)
            
            # 4. Validate translated action
            validated_action = action_interface.validate_action(sampled_action)
            assert action_space.contains(validated_action)
            
            # 5. Re-translate validated action (should be consistent)
            validated_nav_command = action_interface.translate_action(validated_action)
            assert isinstance(validated_nav_command, dict)


class TestVectorizedActionProcessing:
    """
    Test suite for vectorized multi-agent action processing.
    
    This test class validates vectorized action translation, validation, and
    processing for multi-agent scenarios with performance benchmarks and
    concurrent action handling validation.
    """

    def test_multi_agent_action_translation(self):
        """
        Test vectorized action translation for multiple agents.
        
        Validates that action interfaces can process multiple actions
        simultaneously with consistent results and performance.
        """
        continuous_action = Continuous2DAction()
        discrete_action = CardinalDiscreteAction()
        
        num_agents = 10
        
        # Test continuous multi-agent actions
        continuous_actions = [np.array([1.0, float(i * 5)]) for i in range(num_agents)]
        continuous_results = []
        
        for action in continuous_actions:
            result = continuous_action.translate_action(action)
            continuous_results.append(result)
        
        assert len(continuous_results) == num_agents
        for i, result in enumerate(continuous_results):
            assert result['linear_velocity'] == 1.0
            assert result['angular_velocity'] == float(i * 5)
        
        # Test discrete multi-agent actions
        discrete_actions = list(range(min(num_agents, discrete_action.get_num_actions())))
        discrete_results = []
        
        for action in discrete_actions:
            result = discrete_action.translate_action(action)
            discrete_results.append(result)
        
        assert len(discrete_results) == len(discrete_actions)
        for result in discrete_results:
            assert 'direction' in result
            assert 'linear_velocity' in result

    def test_vectorized_validation(self):
        """
        Test vectorized action validation for multiple agents.
        
        Validates that validation works efficiently for multiple actions
        with consistent bounds checking and constraint enforcement.
        """
        action_interface = Continuous2DAction(max_velocity=2.0, max_angular_velocity=45.0)
        
        # Test batch validation
        actions = [
            np.array([1.0, 20.0]),    # Valid
            np.array([3.0, 60.0]),    # Exceeds bounds
            np.array([-0.5, -10.0]),  # Valid
            np.array([np.nan, 0.0]),  # Invalid value
        ]
        
        validated_actions = []
        for action in actions:
            validated = action_interface.validate_action(action)
            validated_actions.append(validated)
        
        # Check validation results
        assert len(validated_actions) == len(actions)
        
        # First action should be unchanged
        np.testing.assert_allclose(validated_actions[0], [1.0, 20.0], atol=NUMERICAL_PRECISION_TOLERANCE)
        
        # Second action should be clipped
        np.testing.assert_allclose(validated_actions[1], [2.0, 45.0], atol=NUMERICAL_PRECISION_TOLERANCE)
        
        # Third action should be unchanged
        np.testing.assert_allclose(validated_actions[2], [-0.5, -10.0], atol=NUMERICAL_PRECISION_TOLERANCE)
        
        # Fourth action should replace NaN with 0
        assert np.isfinite(validated_actions[3]).all()

    def test_batch_action_processing(self):
        """
        Test batch action processing efficiency and consistency.
        
        Validates that processing multiple actions maintains consistency
        and provides expected performance characteristics.
        """
        continuous_action = Continuous2DAction()
        discrete_action = CardinalDiscreteAction()
        
        batch_size = 50
        
        # Test continuous batch processing
        continuous_actions = [np.array([np.random.uniform(-1, 1), np.random.uniform(-30, 30)]) 
                            for _ in range(batch_size)]
        
        start_time = time.perf_counter()
        continuous_results = [continuous_action.translate_action(action) for action in continuous_actions]
        continuous_duration = time.perf_counter() - start_time
        
        assert len(continuous_results) == batch_size
        assert continuous_duration < 0.1  # Should be fast for 50 actions
        
        # Test discrete batch processing
        discrete_actions = [np.random.randint(0, discrete_action.get_num_actions()) 
                          for _ in range(batch_size)]
        
        start_time = time.perf_counter()
        discrete_results = [discrete_action.translate_action(action) for action in discrete_actions]
        discrete_duration = time.perf_counter() - start_time
        
        assert len(discrete_results) == batch_size
        assert discrete_duration < 0.1  # Should be fast for 50 actions

    def test_concurrent_action_handling(self):
        """
        Test concurrent action handling and thread safety.
        
        Validates that action interfaces can handle concurrent access
        patterns without race conditions or inconsistent results.
        """
        import threading
        import concurrent.futures
        
        action_interface = Continuous2DAction()
        num_threads = 5
        actions_per_thread = 20
        
        def process_actions(thread_id):
            """Process actions in a thread."""
            results = []
            for i in range(actions_per_thread):
                action = np.array([float(thread_id), float(i)])
                result = action_interface.translate_action(action)
                results.append(result)
            return results
        
        # Test concurrent execution
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(process_actions, i) for i in range(num_threads)]
            all_results = []
            
            for future in concurrent.futures.as_completed(futures):
                thread_results = future.result()
                all_results.extend(thread_results)
        
        # Validate results
        assert len(all_results) == num_threads * actions_per_thread
        for result in all_results:
            assert isinstance(result, dict)
            assert 'linear_velocity' in result
            assert 'angular_velocity' in result

    def test_vectorized_performance(self):
        """
        Test vectorized action processing performance requirements.
        
        Validates that vectorized operations meet performance requirements
        for multi-agent scenarios with 100 concurrent agents.
        """
        continuous_action = Continuous2DAction()
        discrete_action = CardinalDiscreteAction()
        
        num_agents = 100
        
        # Test continuous action performance
        continuous_actions = [np.array([np.random.uniform(-1, 1), np.random.uniform(-30, 30)]) 
                            for _ in range(num_agents)]
        
        start_time = time.perf_counter()
        for action in continuous_actions:
            continuous_action.translate_action(action)
        continuous_duration = time.perf_counter() - start_time
        
        # Should meet ≤33ms requirement for 100 agents
        assert continuous_duration <= PERFORMANCE_LATENCY_THRESHOLD_MS / 1000.0
        
        # Test discrete action performance
        discrete_actions = [np.random.randint(0, discrete_action.get_num_actions()) 
                          for _ in range(num_agents)]
        
        start_time = time.perf_counter()
        for action in discrete_actions:
            discrete_action.translate_action(action)
        discrete_duration = time.perf_counter() - start_time
        
        # Should meet ≤33ms requirement for 100 agents
        assert discrete_duration <= PERFORMANCE_LATENCY_THRESHOLD_MS / 1000.0


class TestActionPerformance:
    """
    Test suite for action interface performance validation.
    
    This test class validates performance requirements including translation
    latency, validation overhead, action space creation performance, and
    memory efficiency for action processing operations.
    """

    def test_translation_latency(self):
        """
        Test action translation latency requirements.
        
        Validates that action translation meets <0.1ms per agent requirement
        for minimal control overhead in simulation loops.
        """
        continuous_action = Continuous2DAction()
        discrete_action = CardinalDiscreteAction()
        
        # Test continuous action translation latency
        action = np.array([1.0, 15.0])
        
        start_time = time.perf_counter()
        for _ in range(100):  # Multiple iterations for accurate measurement
            continuous_action.translate_action(action)
        duration = time.perf_counter() - start_time
        
        avg_latency = duration / 100
        assert avg_latency < SINGLE_ACTION_LATENCY_THRESHOLD_MS / 1000.0
        
        # Test discrete action translation latency
        action = 1  # North direction
        
        start_time = time.perf_counter()
        for _ in range(100):
            discrete_action.translate_action(action)
        duration = time.perf_counter() - start_time
        
        avg_latency = duration / 100
        assert avg_latency < SINGLE_ACTION_LATENCY_THRESHOLD_MS / 1000.0

    def test_100_agent_performance(self):
        """
        Test performance with 100 concurrent agents.
        
        Validates that action processing meets ≤33ms step latency requirement
        with 100 agents for real-time simulation capability.
        """
        continuous_action = Continuous2DAction()
        discrete_action = CardinalDiscreteAction()
        
        num_agents = 100
        
        # Generate actions for 100 agents
        continuous_actions = [np.array([np.random.uniform(-1, 1), np.random.uniform(-30, 30)]) 
                            for _ in range(num_agents)]
        discrete_actions = [np.random.randint(0, discrete_action.get_num_actions()) 
                          for _ in range(num_agents)]
        
        # Test continuous action performance
        start_time = time.perf_counter()
        for action in continuous_actions:
            continuous_action.translate_action(action)
            continuous_action.validate_action(action)
        continuous_duration = time.perf_counter() - start_time
        
        assert continuous_duration <= PERFORMANCE_LATENCY_THRESHOLD_MS / 1000.0
        
        # Test discrete action performance
        start_time = time.perf_counter()
        for action in discrete_actions:
            discrete_action.translate_action(action)
            discrete_action.validate_action(action)
        discrete_duration = time.perf_counter() - start_time
        
        assert discrete_duration <= PERFORMANCE_LATENCY_THRESHOLD_MS / 1000.0

    def test_validation_overhead(self):
        """
        Test action validation overhead requirements.
        
        Validates that validation meets <0.05ms per agent requirement
        for efficient constraint checking in simulation loops.
        """
        continuous_action = Continuous2DAction()
        discrete_action = CardinalDiscreteAction()
        
        # Test continuous validation overhead
        action = np.array([2.5, 50.0])  # Out of bounds to trigger validation
        
        start_time = time.perf_counter()
        for _ in range(100):
            continuous_action.validate_action(action)
        duration = time.perf_counter() - start_time
        
        avg_latency = duration / 100
        assert avg_latency < VALIDATION_LATENCY_THRESHOLD_MS / 1000.0
        
        # Test discrete validation overhead
        action = 999  # Invalid action to trigger validation
        
        start_time = time.perf_counter()
        for _ in range(100):
            discrete_action.validate_action(action)
        duration = time.perf_counter() - start_time
        
        avg_latency = duration / 100
        assert avg_latency < VALIDATION_LATENCY_THRESHOLD_MS / 1000.0

    def test_action_space_creation_performance(self):
        """
        Test action space creation performance.
        
        Validates that action space creation is efficient since it's
        typically called during environment initialization.
        """
        # Test multiple action space creations (simulating multiple environments)
        start_time = time.perf_counter()
        
        for _ in range(50):
            continuous_action = Continuous2DAction()
            continuous_action.get_action_space()
            
            discrete_action = CardinalDiscreteAction()
            discrete_action.get_action_space()
        
        duration = time.perf_counter() - start_time
        
        # Should be fast even for multiple environments
        assert duration < 1.0  # 1 second for 50 environments

    def test_memory_efficiency(self):
        """
        Test memory efficiency requirements for action representations.
        
        Validates that action processing maintains <100 bytes per action
        for structured representations to support large-scale simulations.
        """
        import sys
        
        continuous_action = Continuous2DAction()
        discrete_action = CardinalDiscreteAction()
        
        # Test memory usage of action translation results
        action = np.array([1.0, 15.0])
        result = continuous_action.translate_action(action)
        
        # Estimate memory usage of result dictionary
        result_size = sys.getsizeof(result)
        for key, value in result.items():
            result_size += sys.getsizeof(key) + sys.getsizeof(value)
        
        assert result_size < MEMORY_EFFICIENCY_THRESHOLD_BYTES
        
        # Test discrete action memory usage
        discrete_result = discrete_action.translate_action(1)
        discrete_size = sys.getsizeof(discrete_result)
        for key, value in discrete_result.items():
            discrete_size += sys.getsizeof(key) + sys.getsizeof(value)
        
        assert discrete_size < MEMORY_EFFICIENCY_THRESHOLD_BYTES


class TestActionFactory:
    """
    Test suite for action interface factory functionality.
    
    This test class validates the create_action_interface factory function
    including instantiation, configuration validation, Hydra integration,
    and runtime action space selection capabilities.
    """

    def test_factory_instantiation(self):
        """
        Test factory-based action interface instantiation.
        
        Validates that the factory function correctly creates action interfaces
        from configuration dictionaries with proper parameter handling.
        """
        # Test Continuous2D creation
        continuous_config = {
            'type': 'Continuous2D',
            'max_velocity': 2.5,
            'max_angular_velocity': 60.0,
            'min_velocity': -1.5,
            'min_angular_velocity': -30.0
        }
        
        continuous_interface = create_action_interface(continuous_config)
        assert isinstance(continuous_interface, Continuous2DAction)
        assert isinstance(continuous_interface, ActionInterfaceProtocol)
        assert continuous_interface.get_max_velocity() == 2.5
        assert continuous_interface.get_max_angular_velocity() == 60.0
        
        # Test CardinalDiscrete creation
        discrete_config = {
            'type': 'CardinalDiscrete',
            'speed': 1.5,
            'use_8_directions': False,
            'include_stay_action': True
        }
        
        discrete_interface = create_action_interface(discrete_config)
        assert isinstance(discrete_interface, CardinalDiscreteAction)
        assert isinstance(discrete_interface, ActionInterfaceProtocol)
        assert discrete_interface.get_speed() == 1.5
        assert discrete_interface.get_num_actions() == 5  # 4 directions + stay

    def test_configuration_validation(self):
        """
        Test configuration validation in factory function.
        
        Validates that invalid configurations are properly detected and
        that appropriate error messages are provided for debugging.
        """
        # Test missing type field
        with pytest.raises(KeyError):
            create_action_interface({})
        
        # Test invalid type
        with pytest.raises(ValueError, match="Unknown action interface type"):
            create_action_interface({'type': 'InvalidType'})
        
        # Test invalid parameters for Continuous2D
        with pytest.raises(ValueError):
            create_action_interface({
                'type': 'Continuous2D',
                'max_velocity': -1.0  # Invalid negative max velocity
            })
        
        # Test invalid parameters for CardinalDiscrete
        with pytest.raises(ValueError):
            create_action_interface({
                'type': 'CardinalDiscrete',
                'speed': -1.0  # Invalid negative speed
            })
        
        # Test non-dict configuration
        with pytest.raises(TypeError):
            create_action_interface("not_a_dict")

    def test_hydra_integration(self):
        """
        Test Hydra configuration integration patterns.
        
        Validates that factory function works with Hydra-style configuration
        patterns and OmegaConf DictConfig objects.
        """
        try:
            from omegaconf import DictConfig, OmegaConf
            
            # Test with OmegaConf DictConfig
            hydra_config = OmegaConf.create({
                'type': 'Continuous2D',
                'max_velocity': 3.0,
                'max_angular_velocity': 90.0
            })
            
            action_interface = create_action_interface(hydra_config)
            assert isinstance(action_interface, Continuous2DAction)
            assert action_interface.get_max_velocity() == 3.0
            
        except ImportError:
            pytest.skip("OmegaConf not available")

    def test_runtime_action_space_selection(self):
        """
        Test runtime action space selection and configuration.
        
        Validates that different action interfaces can be selected and
        configured at runtime based on experimental requirements.
        """
        # Test available action types
        available_types = list_available_action_types()
        assert 'Continuous2D' in available_types
        assert 'CardinalDiscrete' in available_types
        
        # Test configuration validation utility
        valid_config = {'type': 'Continuous2D', 'max_velocity': 2.0}
        invalid_config = {'type': 'InvalidType'}
        
        assert validate_action_config(valid_config) == True
        assert validate_action_config(invalid_config) == False
        
        # Test action space info utility
        continuous_interface = create_action_interface(valid_config)
        info = get_action_space_info(continuous_interface)
        
        assert info['type'] == 'continuous'
        assert info['interface_class'] == 'Continuous2DAction'
        assert 'max_velocity' in info

    def test_factory_error_handling(self):
        """
        Test factory error handling and graceful degradation.
        
        Validates that factory function provides clear error messages
        and handles edge cases appropriately.
        """
        # Test with defaults for optional parameters
        minimal_continuous_config = {'type': 'Continuous2D'}
        continuous_interface = create_action_interface(minimal_continuous_config)
        assert isinstance(continuous_interface, Continuous2DAction)
        
        minimal_discrete_config = {'type': 'CardinalDiscrete'}
        discrete_interface = create_action_interface(minimal_discrete_config)
        assert isinstance(discrete_interface, CardinalDiscreteAction)
        
        # Test parameter type conversion
        string_params_config = {
            'type': 'Continuous2D',
            'max_velocity': '2.0',  # String that can be converted to float
            'max_angular_velocity': '45.0'
        }
        
        interface = create_action_interface(string_params_config)
        assert isinstance(interface, Continuous2DAction)
        assert interface.get_max_velocity() == 2.0


# Standalone test functions for additional validation

def test_action_interface_backwards_compatibility():
    """
    Test backward compatibility with legacy action patterns.
    
    Validates that action interfaces maintain compatibility with existing
    navigation patterns and legacy action processing workflows.
    """
    # Test that old-style action processing still works
    continuous_action = Continuous2DAction()
    
    # Test various input formats that might exist in legacy code
    legacy_formats = [
        [1.0, 15.0],           # Python list
        (1.0, 15.0),           # Tuple
        np.array([1.0, 15.0]), # NumPy array
        np.array(1.0),         # Scalar numpy array
        1.0                    # Python float (should work for single component)
    ]
    
    for legacy_action in legacy_formats:
        try:
            result = continuous_action.translate_action(legacy_action)
            assert isinstance(result, dict)
            assert 'linear_velocity' in result
            assert 'angular_velocity' in result
        except (ValueError, TypeError):
            # Some formats may not be supported, which is acceptable
            pass


def test_action_error_handling_and_validation():
    """
    Test comprehensive error handling and edge case validation.
    
    Validates that action interfaces properly handle invalid inputs,
    edge cases, and error conditions with appropriate error messages.
    """
    continuous_action = Continuous2DAction()
    discrete_action = CardinalDiscreteAction()
    
    # Test invalid action shapes for continuous
    with pytest.raises(ValueError):
        continuous_action.translate_action(np.array([1.0, 2.0, 3.0]))  # Too many dimensions
    
    # Test invalid action indices for discrete
    with pytest.raises(ValueError):
        discrete_action.translate_action(-1)  # Negative index
    
    with pytest.raises(ValueError):
        discrete_action.translate_action(discrete_action.get_num_actions() + 10)  # Too high
    
    # Test edge case values
    edge_values = [
        np.array([np.inf, 0.0]),     # Infinity
        np.array([np.nan, 0.0]),     # NaN
        np.array([1e10, 1e10]),      # Very large values
        np.array([-1e10, -1e10]),    # Very negative values
    ]
    
    for edge_action in edge_values:
        # Should not crash, should handle gracefully
        try:
            validated = continuous_action.validate_action(edge_action)
            assert np.isfinite(validated).all()  # Should produce finite values
        except Exception as e:
            pytest.fail(f"Action validation failed to handle edge case {edge_action}: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])