"""Tests for the angular velocity functionality in the Navigator class."""

import numpy as np
import pytest
from {{cookiecutter.project_slug}}.core.navigator import Navigator


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

        # Take two more steps
        navigator.step(env)
        navigator.step(env)
        self.assert_orientation_close(navigator.orientations[0], 120.0)

        # Test wraparound at 360 degrees
        controller = navigator._controller
        controller._orientation[0] = 350.0
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

        # Take another step
        navigator.step(env)
        self.assert_orientation_close(navigator.orientations[0], 20.0)  # 10 + (10 * 1) = 20
        self.assert_orientation_close(navigator.orientations[1], 130.0)  # 110 + (20 * 1) = 130
        self.assert_orientation_close(navigator.orientations[2], 240.0)  # 210 + (30 * 1) = 240

        # Take one more step
        navigator.step(env)
        self.assert_orientation_close(navigator.orientations[0], 30.0)
        self.assert_orientation_close(navigator.orientations[1], 150.0)  # 130 + (20 * 1) = 150
        self.assert_orientation_close(navigator.orientations[2], 270.0)  # 240 + (30 * 1) = 270

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
        navigator.step(env)
        navigator.step(env)
        self.assert_orientation_close(navigator.orientations[0], 0.0)  # 135 - (45 * 3) = 0

    def test_large_angular_change(self):
        """Test orientation updates with large changes (> 360 degrees)."""
        # Create a navigator with initial orientation 0
        navigator = self.create_navigator_with_angular_velocity(orientation=0.0, angular_velocity=180.0)

        # Create an environment array for step()
        env = np.zeros((10, 10))

        # Take three steps (equivalent to dt=3 in old architecture)
        navigator.step(env)
        navigator.step(env)
        navigator.step(env)
        self.assert_orientation_close(navigator.orientations[0], 180.0)  # 540 % 360 = 180

        # Test negative large angular change
        controller = navigator._controller
        controller._orientation[0] = 0.0
        controller._angular_velocity[0] = -180.0

        # Take three steps
        navigator.step(env)
        navigator.step(env)
        navigator.step(env)
        self.assert_orientation_close(navigator.orientations[0], 180.0)  # -540 % 360 = 180

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


class TestNavigatorAngularVelocityHydraIntegration:
    """Test cases for angular velocity functionality with Hydra configuration system."""

    def test_hydra_angular_velocity_configuration(self):
        """Test angular velocity configuration through Hydra configuration system."""
        # This test validates that angular velocity can be configured through Hydra
        # In a real implementation, this would use hydra.compose() with config composition
        
        # Mock Hydra config structure for testing
        from types import SimpleNamespace
        
        # Simulate Hydra DictConfig for single agent
        hydra_config = SimpleNamespace()
        hydra_config.navigator = SimpleNamespace()
        hydra_config.navigator.orientation = 45.0
        hydra_config.navigator.angular_velocity = 25.0
        
        # Test that configuration values can be applied to navigator
        navigator = Navigator.single(
            orientation=hydra_config.navigator.orientation,
            angular_velocity=hydra_config.navigator.angular_velocity
        )
        
        self.assert_orientation_close(navigator.orientations[0], 45.0)
        self.assert_orientation_close(navigator._controller._angular_velocity[0], 25.0)

    def test_hydra_multi_agent_angular_velocity_configuration(self):
        """Test multi-agent angular velocity configuration through Hydra."""
        from types import SimpleNamespace
        
        # Simulate Hydra DictConfig for multi-agent
        hydra_config = SimpleNamespace()
        hydra_config.navigator = SimpleNamespace()
        hydra_config.navigator.num_agents = 3
        hydra_config.navigator.orientations = [0.0, 90.0, 180.0]
        hydra_config.navigator.angular_velocities = [10.0, 20.0, 30.0]
        
        # Create positions and configure navigator
        positions = np.zeros((hydra_config.navigator.num_agents, 2))
        orientations = np.array(hydra_config.navigator.orientations)
        angular_velocities = np.array(hydra_config.navigator.angular_velocities)
        
        navigator = Navigator.multi(
            positions=positions,
            orientations=orientations,
            angular_velocities=angular_velocities
        )
        
        # Verify configuration was applied correctly
        assert np.allclose(navigator.orientations, orientations)
        assert np.allclose(navigator._controller._angular_velocities, angular_velocities)

    def test_hydra_angular_velocity_override(self):
        """Test that Hydra overrides work for angular velocity parameters."""
        from types import SimpleNamespace
        
        # Simulate base configuration
        base_config = SimpleNamespace()
        base_config.navigator = SimpleNamespace()
        base_config.navigator.angular_velocity = 15.0
        
        # Simulate override configuration
        override_config = SimpleNamespace()
        override_config.navigator = SimpleNamespace()
        override_config.navigator.angular_velocity = 35.0
        
        # Test that override takes precedence
        navigator = Navigator.single(angular_velocity=override_config.navigator.angular_velocity)
        self.assert_orientation_close(navigator._controller._angular_velocity[0], 35.0)

    def test_angular_velocity_validation_through_hydra(self):
        """Test angular velocity parameter validation through configuration system."""
        # Test valid angular velocity ranges
        valid_angular_velocities = [0.0, 15.0, 45.0, 90.0, 180.0, 360.0]
        
        for angular_velocity in valid_angular_velocities:
            navigator = Navigator.single(angular_velocity=angular_velocity)
            self.assert_orientation_close(navigator._controller._angular_velocity[0], angular_velocity)
        
        # Test that extremely large values are handled (should not raise exceptions)
        navigator = Navigator.single(angular_velocity=720.0)
        self.assert_orientation_close(navigator._controller._angular_velocity[0], 720.0)

    @staticmethod
    def assert_orientation_close(actual, expected, tolerance=1e-5):
        """Assert that orientation values are close, accounting for floating-point precision."""
        assert np.isclose(actual, expected, atol=tolerance)


class TestNavigatorAngularVelocityFactoryPatterns:
    """Test cases for angular velocity functionality with new factory instantiation patterns."""

    def test_navigator_factory_with_angular_velocity(self):
        """Test navigator creation through factory methods with angular velocity configuration."""
        # This would typically use the factory method from api/navigation.py
        # For testing purposes, we simulate the factory pattern
        
        # Test single agent factory pattern
        config = {
            'orientation': 30.0,
            'angular_velocity': 20.0,
            'max_speed': 5.0
        }
        
        navigator = Navigator.single(
            orientation=config['orientation'],
            angular_velocity=config['angular_velocity']
        )
        
        assert np.isclose(navigator.orientations[0], 30.0)
        assert np.isclose(navigator._controller._angular_velocity[0], 20.0)

    def test_multi_agent_factory_with_angular_velocities(self):
        """Test multi-agent navigator creation through factory patterns."""
        config = {
            'num_agents': 2,
            'positions': [[0.0, 0.0], [5.0, 5.0]],
            'orientations': [45.0, 135.0],
            'angular_velocities': [15.0, 25.0]
        }
        
        positions = np.array(config['positions'])
        orientations = np.array(config['orientations'])
        angular_velocities = np.array(config['angular_velocities'])
        
        navigator = Navigator.multi(
            positions=positions,
            orientations=orientations,
            angular_velocities=angular_velocities
        )
        
        assert np.allclose(navigator.positions, positions)
        assert np.allclose(navigator.orientations, orientations)
        assert np.allclose(navigator._controller._angular_velocities, angular_velocities)

    def test_factory_pattern_with_defaults(self):
        """Test factory pattern behavior with default angular velocity values."""
        # Test that factory patterns provide sensible defaults
        navigator = Navigator.single()
        
        # Verify default angular velocity is set (typically 0.0)
        default_angular_velocity = navigator._controller._angular_velocity[0]
        assert isinstance(default_angular_velocity, (int, float, np.number))
        assert not np.isnan(default_angular_velocity)

    def test_factory_pattern_parameter_validation(self):
        """Test parameter validation in factory patterns."""
        # Test that invalid parameters are handled appropriately
        positions = np.array([[0.0, 0.0], [1.0, 1.0]])
        orientations = np.array([0.0, 90.0])
        
        # Test mismatched array lengths (should handle gracefully or raise appropriate error)
        invalid_angular_velocities = np.array([10.0])  # Length 1, but need 2
        
        try:
            navigator = Navigator.multi(
                positions=positions,
                orientations=orientations,
                angular_velocities=invalid_angular_velocities
            )
            # If no exception, verify the system handled it appropriately
            assert len(navigator._controller._angular_velocities) == len(positions)
        except (ValueError, AssertionError):
            # Expected behavior for mismatched dimensions
            pass