"""Tests for the angular velocity functionality in the Navigator class."""

import numpy as np
import pytest
from odor_plume_nav.navigator import Navigator


class TestNavigatorAngularVelocity:
    """Test cases for angular velocity functionality in Navigator."""

    @staticmethod
    def create_navigator_with_angular_velocity(orientation=0.0, angular_velocity=30.0):
        """Create a single-agent navigator with specified orientation and angular velocity."""
        navigator = Navigator(orientation=orientation)
        navigator.angular_velocities = np.array([angular_velocity])
        return navigator
    
    @staticmethod
    def create_multi_agent_navigator(orientations, angular_velocities):
        """Create a multi-agent navigator with specified orientations and angular velocities."""
        navigator = Navigator(orientations=orientations)
        navigator.angular_velocities = angular_velocities
        return navigator
    
    @staticmethod
    def assert_orientation_close(actual, expected, tolerance=1e-5):
        """Assert that orientation values are close, accounting for floating-point precision."""
        assert np.isclose(actual, expected, atol=tolerance)

    def test_single_agent_angular_velocity(self):
        """Test orientation updates for a single agent with angular velocity."""
        # Create a navigator with initial orientation 0
        navigator = self.create_navigator_with_angular_velocity(orientation=0.0, angular_velocity=30.0)
        
        # Update with dt=1
        navigator.update(dt=1.0)
        self.assert_orientation_close(navigator.orientation, 30.0)
        
        # Update again with dt=1
        navigator.update(dt=1.0)
        self.assert_orientation_close(navigator.orientation, 60.0)
        
        # Update with dt=2
        navigator.update(dt=2.0)
        self.assert_orientation_close(navigator.orientation, 120.0)
        
        # Testing modulo 360
        navigator.set_orientation(350.0)
        navigator.update(dt=1.0)
        self.assert_orientation_close(navigator.orientation, 20.0)  # 350 + 30 = 380, 380 % 360 = 20

    def test_multi_agent_angular_velocity(self):
        """Test orientation updates for multiple agents with different angular velocities."""
        # Create a navigator with multiple agents and different initial orientations
        orientations = np.array([0.0, 90.0, 180.0])
        angular_velocities = np.array([10.0, 20.0, 30.0])
        navigator = self.create_multi_agent_navigator(orientations, angular_velocities)
        
        # Update with dt=1
        navigator.update(dt=1.0)
        self.assert_orientation_close(navigator.orientations[0], 10.0)
        self.assert_orientation_close(navigator.orientations[1], 110.0)
        self.assert_orientation_close(navigator.orientations[2], 210.0)
        
        # Update with dt=2
        navigator.update(dt=2.0)
        self.assert_orientation_close(navigator.orientations[0], 30.0)  # 10 + (10 * 2) = 30
        self.assert_orientation_close(navigator.orientations[1], 150.0)  # 110 + (20 * 2) = 150
        self.assert_orientation_close(navigator.orientations[2], 270.0)  # 210 + (30 * 2) = 270

    def test_negative_angular_velocity(self):
        """Test orientation updates with negative angular velocity (turning right)."""
        # Create a navigator with initial orientation 180
        navigator = self.create_navigator_with_angular_velocity(orientation=180.0, angular_velocity=-45.0)
        
        # Update with dt=1
        navigator.update(dt=1.0)
        self.assert_orientation_close(navigator.orientation, 135.0)  # 180 - 45 = 135
        
        # Update with dt=3
        navigator.update(dt=3.0)
        self.assert_orientation_close(navigator.orientation, 0.0)  # 135 - (45 * 3) = 0

    def test_large_angular_change(self):
        """Test orientation updates with large changes (> 360 degrees)."""
        # Create a navigator with initial orientation 0
        navigator = self.create_navigator_with_angular_velocity(orientation=0.0, angular_velocity=180.0)
        
        # Update with dt=3 (should wrap around 360)
        navigator.update(dt=3.0)
        self.assert_orientation_close(navigator.orientation, 180.0)  # 0 + (180 * 3) = 540, 540 % 360 = 180
        
        # Test with negative large change
        navigator.set_orientation(0.0)
        navigator.angular_velocities = np.array([-180.0])
        navigator.update(dt=3.0)
        self.assert_orientation_close(navigator.orientation, 180.0)  # 0 - (180 * 3) = -540, -540 % 360 = 180

    def test_set_angular_velocity(self):
        """Test setting angular velocity for a single agent."""
        navigator = Navigator()
        
        # Test setting angular velocity
        navigator.set_angular_velocity(45.0)
        self.assert_orientation_close(navigator.angular_velocities[0], 45.0)
        
        # Test setting angular velocity for a specific agent in multi-agent setup
        navigator = Navigator(num_agents=3)
        navigator.set_angular_velocity_at(1, 30.0)
        self.assert_orientation_close(navigator.angular_velocities[1], 30.0)
        
        # Test setting angular velocities for all agents
        angular_velocities = np.array([10.0, 20.0, 30.0])
        navigator.set_angular_velocities(angular_velocities)
        assert np.allclose(navigator.angular_velocities, angular_velocities)

    def test_initialization_with_angular_velocity(self):
        """Test initialization with angular velocity parameter."""
        # Single agent
        navigator = Navigator(orientation=0.0, angular_velocity=15.0)
        self.assert_orientation_close(navigator.angular_velocities[0], 15.0)
        
        # Multiple agents
        angular_velocities = np.array([5.0, 10.0, 15.0])
        navigator = Navigator(
            num_agents=3,
            orientations=np.zeros(3),
            angular_velocities=angular_velocities
        )
        assert np.allclose(navigator.angular_velocities, angular_velocities)
