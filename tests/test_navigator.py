"""Tests for the navigator functionality."""

import pytest
import numpy as np
from numpy.testing import assert_allclose
from unittest.mock import MagicMock

from odor_plume_nav.core.navigator import Navigator


def test_navigator_initialization():
    """Test that Navigator can be initialized with orientation and speed."""
    # Create a navigator with default parameters
    navigator = Navigator.single()
    
    # Default values should be set
    assert navigator.orientations[0] == 0.0
    assert navigator.speeds[0] == 0.0
    
    # Create a navigator with custom parameters
    custom_navigator = Navigator.single(orientation=45.0, speed=0.5)
    
    # Custom values should be set
    assert custom_navigator.orientations[0] == 45.0
    assert custom_navigator.speeds[0] == 0.5


def test_navigator_orientation():
    """Test that orientation is normalized properly during step."""
    # Create navigators with different orientations
    navigator_90 = Navigator.single(orientation=90.0)
    assert navigator_90.orientations[0] == 90.0
    
    # Test normalization of angles during step
    navigator_450 = Navigator.single(orientation=450.0)
    # Initial value is not automatically normalized
    assert navigator_450.orientations[0] == 450.0
    
    # Normalization happens during step
    navigator_450.step(np.zeros((100, 100)))
    assert navigator_450.orientations[0] == 90.0
    
    # Test normalization of negative angles during step
    navigator_neg90 = Navigator.single(orientation=-90.0)
    # Initial value is not automatically normalized
    assert navigator_neg90.orientations[0] == -90.0
    
    # Normalization happens during step
    navigator_neg90.step(np.zeros((100, 100)))
    assert navigator_neg90.orientations[0] == 270.0


def test_navigator_speed():
    """Test speed values in the controller."""
    # Test setting valid speed
    navigator_half = Navigator.single(speed=0.5, max_speed=1.0)
    assert navigator_half.speeds[0] == 0.5
    
    # Test that the controller accepts speeds above max_speed
    # In the new implementation, speeds are not automatically capped
    navigator_max = Navigator.single(speed=2.0, max_speed=1.0)
    assert navigator_max.speeds[0] == 2.0
    
    # After a step, the speed is still not capped
    navigator_max.step(np.zeros((100, 100)))
    assert navigator_max.speeds[0] == 2.0
    
    # Verify that the movement uses the actual speed value
    dist_moved = np.linalg.norm(navigator_max.positions[0])
    assert np.isclose(dist_moved, 2.0, atol=1e-4)


def test_navigator_movement():
    """Test that the navigator calculates correct movement."""
    # At 0 degrees with speed 1.0, should move along positive x-axis
    navigator = Navigator.single(orientation=0.0, speed=1.0, position=(0.0, 0.0))
    
    # Step the simulation to apply movement
    navigator.step(np.zeros((100, 100)))
    new_pos = navigator.positions[0]
    
    # Should have moved along positive x-axis
    assert np.isclose(new_pos[0], 1.0)
    assert np.isclose(new_pos[1], 0.0)
    
    # Reset and test movement at 90 degrees
    navigator = Navigator.single(orientation=90.0, speed=1.0, position=(0.0, 0.0))
    navigator.step(np.zeros((100, 100)))
    new_pos = navigator.positions[0]
    
    # Should have moved along positive y-axis
    assert np.isclose(new_pos[0], 0.0)
    assert np.isclose(new_pos[1], 1.0)
    
    # Reset and test movement at 45 degrees with speed 0.5
    navigator = Navigator.single(orientation=45.0, speed=0.5, position=(0.0, 0.0))
    navigator.step(np.zeros((100, 100)))
    new_pos = navigator.positions[0]
    
    # Should have moved at 45-degree angle
    assert np.isclose(new_pos[0], 0.3536, atol=1e-4)
    assert np.isclose(new_pos[1], 0.3536, atol=1e-4)


def test_navigator_update():
    """Test that the navigator can update its position over multiple steps."""
    # Starting at position (0, 0) with orientation 0 and speed 1.0
    navigator = Navigator.single(orientation=0.0, speed=1.0, position=(0.0, 0.0))
    env = np.zeros((100, 100))

    # Move for 1 second along x-axis
    navigator.step(env)
    pos = navigator.positions[0]
    assert np.isclose(pos[0], 1.0)
    assert np.isclose(pos[1], 0.0)
    
    # Change orientation to 90 degrees and update again
    # We need to create a new navigator since we can't directly set orientation
    navigator = Navigator.single(orientation=90.0, speed=1.0, position=(1.0, 0.0))
    navigator.step(env)
    pos = navigator.positions[0]
    assert np.isclose(pos[0], 1.0)
    assert np.isclose(pos[1], 1.0)
    
    # One more step
    navigator.step(env)
    pos = navigator.positions[0]
    assert np.isclose(pos[0], 1.0)
    assert np.isclose(pos[1], 2.0)
