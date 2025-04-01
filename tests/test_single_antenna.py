"""Tests for single antenna odor sensing functionality."""

import pytest
import numpy as np
from numpy.testing import assert_allclose
from unittest.mock import MagicMock

from odor_plume_nav.navigator import SimpleNavigator


class TestSingleAntennaSensing:
    """Tests for the single antenna odor sensing functionality."""
    
    def test_read_single_antenna_odor_simple_environment(self):
        """Test reading odor value from a simple environment at navigator position."""
        # Create a simple 5x5 environment with a hotspot
        environment = np.zeros((5, 5), dtype=np.float32)
        hotspot_position = (2, 3)  # x=2, y=3
        hotspot_value = 0.75
        environment[hotspot_position[1], hotspot_position[0]] = hotspot_value  # Note: numpy indexing is [y, x]
        
        # Create navigator at the same position as the hotspot
        navigator = SimpleNavigator(position=hotspot_position)
        
        # Read the odor at the navigator's position
        odor_value = navigator.read_single_antenna_odor(environment)
        
        # Use numpy assertion with tolerance for floating-point comparison
        assert_allclose(odor_value, hotspot_value, rtol=1e-5)
    
    def test_read_single_antenna_odor_gradient_position1(self):
        """Test reading odor value from a gradient at position (3,3)."""
        # Create a 10x10 environment with a gradient
        x, y = np.meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 10))
        # Create a Gaussian-like distribution centered at (0.7, 0.6)
        sigma = 0.2
        environment = np.exp(-((x-0.7)**2 + (y-0.6)**2) / (2*sigma**2))
        
        # Position and navigator for this test
        position = (3, 3)
        navigator = SimpleNavigator(position=position)
        
        # Read the odor at the navigator's position
        odor_value = navigator.read_single_antenna_odor(environment)
        
        # Check that the odor value matches the environment at (x, y)
        # Note: numpy indexing is [y, x]
        expected_value = environment[position[1], position[0]]  
        assert_allclose(odor_value, expected_value, rtol=1e-5)
    
    def test_read_single_antenna_odor_gradient_position2(self):
        """Test reading odor value from a gradient at position (7,6)."""
        # Create a 10x10 environment with a gradient
        x, y = np.meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 10))
        # Create a Gaussian-like distribution centered at (0.7, 0.6)
        sigma = 0.2
        environment = np.exp(-((x-0.7)**2 + (y-0.6)**2) / (2*sigma**2))
        
        # Position and navigator for this test
        position = (7, 6)
        navigator = SimpleNavigator(position=position)
        
        # Read the odor at the navigator's position
        odor_value = navigator.read_single_antenna_odor(environment)
        
        # Check that the odor value matches the environment at (x, y)
        # Note: numpy indexing is [y, x]
        expected_value = environment[position[1], position[0]]  
        assert_allclose(odor_value, expected_value, rtol=1e-5)
    
    def test_read_single_antenna_odor_gradient_position3(self):
        """Test reading odor value from a gradient at position (9,9)."""
        # Create a 10x10 environment with a gradient
        x, y = np.meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 10))
        # Create a Gaussian-like distribution centered at (0.7, 0.6)
        sigma = 0.2
        environment = np.exp(-((x-0.7)**2 + (y-0.6)**2) / (2*sigma**2))
        
        # Position and navigator for this test
        position = (9, 9)
        navigator = SimpleNavigator(position=position)
        
        # Read the odor at the navigator's position
        odor_value = navigator.read_single_antenna_odor(environment)
        
        # Check that the odor value matches the environment at (x, y)
        # Note: numpy indexing is [y, x]
        expected_value = environment[position[1], position[0]]  
        assert_allclose(odor_value, expected_value, rtol=1e-5)
    
    def test_read_single_antenna_odor_video_plume_position1(self):
        """Test reading odor from a mock video plume at position (2,3)."""
        # Create a mock VideoPlume with a current_frame
        mock_plume = MagicMock()
        mock_frame = np.zeros((10, 10), dtype=np.uint8)
        
        # Set up some odor values in the frame
        mock_frame[3, 2] = 150  # Note: numpy indexing is [y, x]
        mock_frame[7, 8] = 200
        mock_plume.current_frame = mock_frame
        
        # Create navigator and test
        position = (2, 3)  # Position with value 150
        expected_value = 150/255
        
        navigator = SimpleNavigator(position=position)
        odor_value = navigator.read_single_antenna_odor(mock_plume)
        
        # Check that the odor value matches the normalized frame value
        assert_allclose(odor_value, expected_value, rtol=1e-5)
    
    def test_read_single_antenna_odor_video_plume_position2(self):
        """Test reading odor from a mock video plume at position (8,7)."""
        # Create a mock VideoPlume with a current_frame
        mock_plume = MagicMock()
        mock_frame = np.zeros((10, 10), dtype=np.uint8)
        
        # Set up some odor values in the frame
        mock_frame[3, 2] = 150  # Note: numpy indexing is [y, x]
        mock_frame[7, 8] = 200
        mock_plume.current_frame = mock_frame
        
        # Create navigator and test
        position = (8, 7)  # Position with value 200
        expected_value = 200/255
        
        navigator = SimpleNavigator(position=position)
        odor_value = navigator.read_single_antenna_odor(mock_plume)
        
        # Check that the odor value matches the normalized frame value
        assert_allclose(odor_value, expected_value, rtol=1e-5)
    
    def test_read_single_antenna_odor_video_plume_position3(self):
        """Test reading odor from a mock video plume at position (5,5)."""
        # Create a mock VideoPlume with a current_frame
        mock_plume = MagicMock()
        mock_frame = np.zeros((10, 10), dtype=np.uint8)
        
        # Set up some odor values in the frame
        mock_frame[3, 2] = 150  # Note: numpy indexing is [y, x]
        mock_frame[7, 8] = 200
        mock_plume.current_frame = mock_frame
        
        # Create navigator and test
        position = (5, 5)  # Position with value 0
        expected_value = 0
        
        navigator = SimpleNavigator(position=position)
        odor_value = navigator.read_single_antenna_odor(mock_plume)
        
        # Check that the odor value matches the normalized frame value
        assert_allclose(odor_value, expected_value, rtol=1e-5)
    
    def test_read_single_antenna_odor_out_of_bounds_left(self):
        """Test reading odor when navigator is outside environment bounds (left)."""
        # Create a simple environment
        environment = np.zeros((5, 5), dtype=np.float32)
        
        # Test position outside left
        position = (-1, 2)
        
        # Create navigator at this out-of-bounds position
        navigator = SimpleNavigator(position=position)
        
        # Read the odor at the navigator's position
        odor_value = navigator.read_single_antenna_odor(environment)
        
        # Should return 0 for out-of-bounds positions
        assert odor_value == 0
    
    def test_read_single_antenna_odor_out_of_bounds_top(self):
        """Test reading odor when navigator is outside environment bounds (top)."""
        # Create a simple environment
        environment = np.zeros((5, 5), dtype=np.float32)
        
        # Test position outside top
        position = (2, -1)
        
        # Create navigator at this out-of-bounds position
        navigator = SimpleNavigator(position=position)
        
        # Read the odor at the navigator's position
        odor_value = navigator.read_single_antenna_odor(environment)
        
        # Should return 0 for out-of-bounds positions
        assert odor_value == 0
    
    def test_read_single_antenna_odor_out_of_bounds_right(self):
        """Test reading odor when navigator is outside environment bounds (right)."""
        # Create a simple environment
        environment = np.zeros((5, 5), dtype=np.float32)
        
        # Test position outside right
        position = (5, 2)
        
        # Create navigator at this out-of-bounds position
        navigator = SimpleNavigator(position=position)
        
        # Read the odor at the navigator's position
        odor_value = navigator.read_single_antenna_odor(environment)
        
        # Should return 0 for out-of-bounds positions
        assert odor_value == 0
    
    def test_read_single_antenna_odor_out_of_bounds_bottom(self):
        """Test reading odor when navigator is outside environment bounds (bottom)."""
        # Create a simple environment
        environment = np.zeros((5, 5), dtype=np.float32)
        
        # Test position outside bottom
        position = (2, 5)
        
        # Create navigator at this out-of-bounds position
        navigator = SimpleNavigator(position=position)
        
        # Read the odor at the navigator's position
        odor_value = navigator.read_single_antenna_odor(environment)
        
        # Should return 0 for out-of-bounds positions
        assert odor_value == 0
    
    def test_read_single_antenna_odor_out_of_bounds_far(self):
        """Test reading odor when navigator is far outside environment bounds."""
        # Create a simple environment
        environment = np.zeros((5, 5), dtype=np.float32)
        
        # Test position far outside
        position = (10, 10)
        
        # Create navigator at this out-of-bounds position
        navigator = SimpleNavigator(position=position)
        
        # Read the odor at the navigator's position
        odor_value = navigator.read_single_antenna_odor(environment)
        
        # Should return 0 for out-of-bounds positions
        assert odor_value == 0
