"""Tests for the angular velocity functionality in the Navigator class."""

import numpy as np
import pytest
from odor_plume_nav.core.navigator import Navigator


class TestNavigatorAngularVelocity:
    """Test cases for angular velocity functionality in Navigator."""

    @staticmethod
    def create_navigator_with_angular_velocity(orientation=0.0, angular_velocity=30.0):
        """Create a single-agent navigator with specified orientation and angular velocity."""
        # In the protocol-based architecture, we use Navigator.single()
        navigator = Navigator.single(orientation=orientation)
        # Access angular_velocities via controller
        controller = navigator._controller
        controller._angular_velocity[0] = angular_velocity
        return navigator
    
    @staticmethod
    def create_multi_agent_navigator(orientations, angular_velocities):
        """Create a multi-agent navigator with specified orientations and angular velocities."""
        # In the protocol-based architecture, we use Navigator.multi()
        # We need to create positions since it's a required parameter
        num_agents = len(orientations)
        positions = np.zeros((num_agents, 2))  # Default positions at origin
        
        navigator = Navigator.multi(positions=positions, orientations=orientations)
        # Access angular_velocities via controller
        controller = navigator._controller
        controller._angular_velocities = angular_velocities
        return navigator
    
    @staticmethod
    def assert_orientation_close(actual, expected, tolerance=1e-5):
        """Assert that orientation values are close, accounting for floating-point precision."""
        assert np.isclose(actual, expected, atol=tolerance)

    def test_single_agent_angular_velocity(self):
        """Test orientation updates for a single agent with angular velocity."""
        # Create a navigator with initial orientation 0
        navigator = self.create_navigator_with_angular_velocity(orientation=0.0, angular_velocity=30.0)

        # In protocol-based architecture, we use step() instead of update()
        # and we access properties with array indexing

        # Create an environment array for step()
        env = np.zeros((10, 10))

        # Take a step (equivalent to dt=1 in old architecture)
        navigator.step(env)
        self.assert_orientation_close(navigator.orientations[0], 30.0)

        # Take another step
        navigator.step(env)
        self.assert_orientation_close(navigator.orientations[0], 60.0)

        controller = self._extracted_from_test_large_angular_change_21(
            navigator, env, 120.0, 350.0
        )
        navigator.step(env)
        self.assert_orientation_close(navigator.orientations[0], 20.0)  # 350 + 30 = 380, 380 % 360 = 20

    def test_multi_agent_angular_velocity(self):
        """Test orientation updates for multiple agents with different angular velocities."""
        # Create a navigator with multiple agents and different initial orientations
        orientations = np.array([0.0, 90.0, 180.0])
        angular_velocities = np.array([10.0, 20.0, 30.0])
        navigator = self.create_multi_agent_navigator(orientations, angular_velocities)

        # Create an environment array for step()
        env = np.zeros((10, 10))

        # Take a step (equivalent to dt=1 in old architecture)
        navigator.step(env)
        self.assert_orientation_close(navigator.orientations[0], 10.0)
        self.assert_orientation_close(navigator.orientations[1], 110.0)
        self.assert_orientation_close(navigator.orientations[2], 210.0)

        self._extracted_from_test_large_angular_change_21(navigator, env, 30.0)
        self.assert_orientation_close(navigator.orientations[1], 150.0)  # 110 + (20 * 2) = 150
        self.assert_orientation_close(navigator.orientations[2], 270.0)  # 210 + (30 * 2) = 270

    def test_negative_angular_velocity(self):
        """Test orientation updates with negative angular velocity (turning right)."""
        # Create a navigator with initial orientation 180
        navigator = self.create_navigator_with_angular_velocity(orientation=180.0, angular_velocity=-45.0)

        # Create an environment array for step()
        env = np.zeros((10, 10))

        # Take a step (equivalent to dt=1 in old architecture)
        navigator.step(env)
        self.assert_orientation_close(navigator.orientations[0], 135.0)  # 180 - 45 = 135

        # Take three more steps (equivalent to dt=3 in old architecture)
        navigator.step(env)
        self._extracted_from_test_large_angular_change_21(navigator, env, 0.0)

    def test_large_angular_change(self):
        """Test orientation updates with large changes (> 360 degrees)."""
        # Create a navigator with initial orientation 0
        navigator = self.create_navigator_with_angular_velocity(orientation=0.0, angular_velocity=180.0)

        # Create an environment array for step()
        env = np.zeros((10, 10))

        # Take three steps (equivalent to dt=3 in old architecture)
        navigator.step(env)
        controller = self._extracted_from_test_large_angular_change_21(
            navigator, env, 180.0, 0.0
        )
        controller._angular_velocity[0] = -180.0

        # Take three steps
        navigator.step(env)
        self._extracted_from_test_large_angular_change_21(navigator, env, 180.0)

    # TODO Rename this here and in `test_single_agent_angular_velocity`, `test_multi_agent_angular_velocity`, `test_negative_angular_velocity` and `test_large_angular_change`
    def _extracted_from_test_large_angular_change_21(self, navigator, env, arg2, arg3):
        self._extracted_from_test_large_angular_change_21(navigator, env, arg2)
        result = navigator._controller
        result._orientation[0] = arg3
        return result

    # TODO Rename this here and in `test_single_agent_angular_velocity`, `test_multi_agent_angular_velocity`, `test_negative_angular_velocity` and `test_large_angular_change`
    def _extracted_from_test_large_angular_change_21(self, navigator, env, arg2):
        navigator.step(env)
        navigator.step(env)
        self.assert_orientation_close(navigator.orientations[0], arg2)

    def test_set_angular_velocity(self):
        """Test setting angular velocity for a single agent."""
        navigator = Navigator.single()
        
        # Test setting angular velocity
        navigator._controller._angular_velocity[0] = 45.0
        self.assert_orientation_close(navigator._controller._angular_velocity[0], 45.0)
        
        # Test setting angular velocity for a specific agent in multi-agent setup
        # Create positions required for multi-agent
        positions = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
        navigator = Navigator.multi(positions=positions)
        controller = navigator._controller
        controller._angular_velocities[1] = 30.0
        self.assert_orientation_close(controller._angular_velocities[1], 30.0)
        
        # Test setting angular velocities for all agents
        angular_velocities = np.array([10.0, 20.0, 30.0])
        controller._angular_velocities = angular_velocities
        assert np.allclose(controller._angular_velocities, angular_velocities)

    def test_initialization_with_angular_velocity(self):
        """Test initialization with angular velocity parameter."""
        # Single agent
        navigator = Navigator.single(orientation=0.0, angular_velocity=15.0)
        self.assert_orientation_close(navigator._controller._angular_velocity[0], 15.0)
        
        # Multiple agents
        angular_velocities = np.array([5.0, 10.0, 15.0])
        # Create positions required for multi-agent
        positions = np.zeros((3, 2))
        navigator = Navigator.multi(
            positions=positions,
            orientations=np.zeros(3),
            angular_velocities=angular_velocities
        )
        assert np.allclose(navigator._controller._angular_velocities, angular_velocities)
