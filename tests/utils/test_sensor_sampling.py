"""Tests for sensor sampling utility functions."""

import pytest
import numpy as np

from odor_plume_nav.core.navigator import Navigator
from odor_plume_nav.utils import (
    calculate_sensor_positions,
    sample_odor_at_sensors,
)


def test_calculate_sensor_positions_single_agent():
    """Test calculating sensor positions for a single agent."""
    # Create a navigator with a known position and orientation
    navigator = Navigator(position=(50, 60), orientation=90)  # facing "north"
    
    # Calculate sensor positions with default parameters (2 sensors, 45-degree angle)
    sensor_positions = calculate_sensor_positions(navigator)
    
    # Should return an array of shape (1, 2, 2) - (num_agents, num_sensors, x/y)
    assert sensor_positions.shape == (1, 2, 2)
    
    # Check that positions are calculated correctly
    # With orientation 90 and sensor angle 45, sensors should be at +/- 22.5 degrees
    # Sensor 0 should be at angle 90 - 22.5 = 67.5 degrees from horizontal
    # Sensor 1 should be at angle 90 + 22.5 = 112.5 degrees from horizontal
    
    # Check approximately using np.isclose to handle floating point
    # Sensor 0: expect ~(50 + 5*cos(67.5°), 60 + 5*sin(67.5°))
    assert np.isclose(sensor_positions[0, 0, 0], 50 + 5 * np.cos(np.radians(67.5)), atol=0.1)
    assert np.isclose(sensor_positions[0, 0, 1], 60 + 5 * np.sin(np.radians(67.5)), atol=0.1)
    
    # Sensor 1: expect ~(50 + 5*cos(112.5°), 60 + 5*sin(112.5°))
    assert np.isclose(sensor_positions[0, 1, 0], 50 + 5 * np.cos(np.radians(112.5)), atol=0.1)
    assert np.isclose(sensor_positions[0, 1, 1], 60 + 5 * np.sin(np.radians(112.5)), atol=0.1)


def test_calculate_sensor_positions_multi_agent():
    """Test calculating sensor positions for multiple agents."""
    # Create a navigator with multiple agents
    positions = [(10, 20), (30, 40), (50, 60)]
    orientations = [0, 90, 180]  # facing right, up, left
    navigator = Navigator(positions=positions, orientations=orientations)
    
    # Calculate sensor positions with custom parameters
    sensor_positions = calculate_sensor_positions(
        navigator, 
        sensor_distance=10.0,
        sensor_angle=30.0,
        num_sensors=3
    )
    
    # Should return array of shape (3, 3, 2) - (num_agents, num_sensors, x/y)
    assert sensor_positions.shape == (3, 3, 2)
    
    # For 3 sensors, we expect one in the center and one on each side
    # Check the center sensor for each agent (should be directly ahead)
    # Agent 0: orientation 0 degrees, center sensor at (10+10, 20)
    assert np.isclose(sensor_positions[0, 1, 0], 20, atol=0.1)
    assert np.isclose(sensor_positions[0, 1, 1], 20, atol=0.1)
    
    # Agent 1: orientation 90 degrees, center sensor at (30, 40+10)
    assert np.isclose(sensor_positions[1, 1, 0], 30, atol=0.1)
    assert np.isclose(sensor_positions[1, 1, 1], 50, atol=0.1)
    
    # Agent 2: orientation 180 degrees, center sensor at (50-10, 60)
    assert np.isclose(sensor_positions[2, 1, 0], 40, atol=0.1)
    assert np.isclose(sensor_positions[2, 1, 1], 60, atol=0.1)


def test_sample_odor_at_sensors_single_agent():
    """Test sampling odor at sensor positions for a single agent."""
    # Create a navigator with a known position
    navigator = Navigator(position=(5, 5))
    
    # Create a test environment with a known odor pattern
    # Higher values in the center, decreasing toward edges
    env_size = 20
    env_array = np.zeros((env_size, env_size), dtype=np.uint8)
    for y in range(env_size):
        for x in range(env_size):
            # Distance from center
            dist = np.sqrt((x - env_size//2)**2 + (y - env_size//2)**2)
            # Odor decreases with distance from center
            env_array[y, x] = max(0, int(255 * (1 - dist / (env_size//2))))
    
    # Sample with 2 sensors
    odor_values = sample_odor_at_sensors(navigator, env_array)
    
    # Should return array of shape (1, 2) - (num_agents, num_sensors)
    assert isinstance(odor_values, np.ndarray)
    assert odor_values.shape == (1, 2)
    
    # Sample with 4 sensors, different distance
    odor_values = sample_odor_at_sensors(
        navigator, 
        env_array,
        sensor_distance=3.0,
        sensor_angle=30.0,
        num_sensors=4
    )
    
    # Should return array of shape (1, 4) - (num_agents, num_sensors)
    assert odor_values.shape == (1, 4)


def test_sample_odor_at_sensors_multi_agent():
    """Test sampling odor at sensor positions for multiple agents."""
    # Create a navigator with multiple agents at different positions
    positions = [(5, 5), (10, 10), (15, 15)]
    navigator = Navigator(positions=positions)
    
    # Create a test environment with a known odor pattern
    # Higher values in the center, decreasing toward edges
    env_size = 20
    env_array = np.zeros((env_size, env_size), dtype=np.uint8)
    for y in range(env_size):
        for x in range(env_size):
            # Distance from center
            dist = np.sqrt((x - env_size//2)**2 + (y - env_size//2)**2)
            # Odor decreases with distance from center
            env_array[y, x] = max(0, int(255 * (1 - dist / (env_size//2))))
    
    # Sample with multiple sensors
    odor_values = sample_odor_at_sensors(
        navigator, 
        env_array,
        num_sensors=3
    )
    
    # Should return array of shape (3, 3) - (num_agents, num_sensors)
    assert odor_values.shape == (3, 3)
    
    # Check that values are normalized to [0, 1] range
    assert np.all(odor_values >= 0)
    assert np.all(odor_values <= 1)


def test_navigator_sample_multiple_sensors():
    """Test the Navigator.sample_multiple_sensors method."""
    # Create a single agent navigator
    navigator_single = Navigator(position=(5, 5))
    
    # Create a multi-agent navigator
    positions = [(5, 5), (10, 10)]
    navigator_multi = Navigator(positions=positions)
    
    # Create a test environment
    env_size = 20
    env_array = np.zeros((env_size, env_size), dtype=np.uint8)
    for y in range(env_size):
        for x in range(env_size):
            # Distance from center
            dist = np.sqrt((x - env_size//2)**2 + (y - env_size//2)**2)
            # Odor decreases with distance from center
            env_array[y, x] = max(0, int(255 * (1 - dist / (env_size//2))))
    
    # Test single agent - should return a list
    readings_single = navigator_single.sample_multiple_sensors(env_array)
    assert isinstance(readings_single, list)
    assert len(readings_single) == 2  # default num_sensors
    
    # Test multi-agent - should return an array
    readings_multi = navigator_multi.sample_multiple_sensors(env_array)
    assert isinstance(readings_multi, np.ndarray)
    assert readings_multi.shape == (2, 2)  # (num_agents, num_sensors)
    
    # Test with custom parameters
    readings_custom = navigator_single.sample_multiple_sensors(
        env_array,
        sensor_distance=2.0,
        sensor_angle=60.0,
        num_sensors=3
    )
    assert isinstance(readings_custom, list)
    assert len(readings_custom) == 3  # custom num_sensors


def test_out_of_bounds_sensors():
    """Test handling of sensors that fall outside the environment boundaries."""
    # Create a navigator at the edge of the environment
    navigator = Navigator(position=(0, 0))  # at the corner
    
    # Create a small environment
    env_size = 10
    env_array = np.ones((env_size, env_size), dtype=np.uint8) * 128
    
    # With default sensor positions, at least one sensor should be out of bounds
    odor_values = sample_odor_at_sensors(navigator, env_array)
    
    # The result should still have the correct shape
    assert odor_values.shape == (1, 2)
    
    # At least one sensor should read zero (out of bounds)
    assert np.any(odor_values == 0)
