"""Tests for the navigator module."""

import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import from the core module instead
from odor_plume_nav.core.navigator import SimpleNavigator


def test_simple_navigator_initialization():
    """Test that SimpleNavigator can be initialized with orientation and speed."""
    # Create a navigator with default parameters
    navigator = SimpleNavigator()
    
    # Default values should be set
    assert navigator.orientation == 0.0
    assert navigator.speed == 0.0
    
    # Create a navigator with custom parameters
    custom_navigator = SimpleNavigator(orientation=45.0, speed=0.5)
    
    # Custom values should be set
    assert custom_navigator.orientation == 45.0
    assert custom_navigator.speed == 0.5


def test_simple_navigator_set_orientation():
    """Test that orientation can be set and is normalized properly."""
    navigator = SimpleNavigator()
    
    # Test setting orientation in degrees
    navigator.set_orientation(90.0)
    assert navigator.orientation == 90.0
    
    # Test normalization of angles > 360
    navigator.set_orientation(450.0)
    assert navigator.orientation == 90.0
    
    # Test normalization of negative angles
    navigator.set_orientation(-90.0)
    assert navigator.orientation == 270.0


def test_simple_navigator_set_speed():
    """Test that speed can be set with proper constraints."""
    navigator = SimpleNavigator(max_speed=1.0)
    
    # Test setting valid speed
    navigator.set_speed(0.5)
    assert navigator.speed == 0.5
    
    # Test setting speed above max_speed
    navigator.set_speed(2.0)
    assert navigator.speed == 1.0  # Should be capped at max_speed
    
    # Test setting negative speed
    navigator.set_speed(-0.5)
    assert navigator.speed == 0.0  # Should be capped at 0


def test_simple_navigator_move():
    """Test that the navigator can calculate movement vectors."""
    navigator = SimpleNavigator(orientation=0.0, speed=1.0)
    
    # At 0 degrees, movement should be along positive x-axis
    movement = navigator.get_movement_vector()
    assert np.isclose(movement[0], 1.0)
    assert np.isclose(movement[1], 0.0)
    
    # Change orientation to 90 degrees (positive y-axis)
    navigator.set_orientation(90.0)
    movement = navigator.get_movement_vector()
    assert np.isclose(movement[0], 0.0)
    assert np.isclose(movement[1], 1.0)
    
    # Change orientation to 45 degrees
    navigator.set_orientation(45.0)
    movement = navigator.get_movement_vector()
    assert np.isclose(movement[0], 0.7071, atol=1e-4)
    assert np.isclose(movement[1], 0.7071, atol=1e-4)
    
    # Change speed to 0.5
    navigator.set_speed(0.5)
    movement = navigator.get_movement_vector()
    assert np.isclose(movement[0], 0.3536, atol=1e-4)
    assert np.isclose(movement[1], 0.3536, atol=1e-4)


def test_simple_navigator_update():
    """Test that the navigator can update its position."""
    # Starting at position (0, 0) with orientation 0 and speed 1.0
    navigator = SimpleNavigator(orientation=0.0, speed=1.0, position=(0.0, 0.0))

    position = update_and_verify_position(navigator, 1.0, 0.0)
    # Change orientation to 90 degrees and update again
    navigator.set_orientation(90.0)
    position = update_and_verify_position(navigator, 1.0, 1.0)
    position = update_and_verify_position(navigator, 0.5, 1.5)


def update_and_verify_position(navigator, dt, expected_y):
    """Update navigator position and verify x=1.0 and y=expected_y."""
    # Move for the specified time
    navigator.update(dt=dt)
    result = navigator.get_position()
    assert np.isclose(result[0], 1.0)
    assert np.isclose(result[1], expected_y)

    return result
