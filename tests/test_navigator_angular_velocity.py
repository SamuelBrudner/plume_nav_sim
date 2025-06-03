"""Tests for the angular velocity functionality in the Navigator class.

This module provides comprehensive testing for angular velocity configuration and behavior
in the enhanced cookiecutter-based navigation system with Hydra configuration support.

The tests validate angular velocity handling across single-agent and multi-agent scenarios,
ensuring proper integration with the new Hydra configuration management system and
maintaining compatibility with the NavigatorProtocol interface.
"""

import numpy as np
import pytest
from hydra import compose, initialize
from omegaconf import DictConfig
from {{cookiecutter.project_slug}}.core.navigator import NavigatorProtocol


class TestNavigatorAngularVelocity:
    """Test cases for angular velocity functionality in Navigator implementations.
    
    This test class validates angular velocity configuration and behavior across
    the enhanced navigation system, including Hydra configuration integration,
    protocol compliance, and mathematical accuracy of angular velocity updates.
    
    Test Coverage:
    - Single-agent angular velocity configuration and updates
    - Multi-agent angular velocity management
    - Hydra configuration-driven angular velocity settings
    - Edge cases including negative velocities and large angular changes
    - Protocol compliance for angular velocity properties
    """

    @pytest.fixture
    def hydra_config_fixture(self):
        """Provides Hydra configuration for navigator testing.
        
        Creates a test-specific Hydra configuration composition with angular
        velocity parameters for comprehensive configuration-driven testing.
        
        Returns
        -------
        DictConfig
            Hydra configuration object with navigator and angular velocity settings
            
        Notes
        -----
        This fixture initializes Hydra with test configuration and provides
        structured configuration access for navigator instantiation patterns.
        """
        with initialize(version_base=None, config_path="../conf"):
            cfg = compose(
                config_name="config",
                overrides=[
                    "navigator.type=single",
                    "navigator.orientation=0.0",
                    "navigator.angular_velocity=30.0",
                    "navigator.max_speed=10.0"
                ]
            )
            return cfg

    @pytest.fixture
    def multi_agent_hydra_config(self):
        """Provides multi-agent Hydra configuration for testing.
        
        Creates Hydra configuration for multi-agent scenarios with multiple
        angular velocity settings and comprehensive parameter validation.
        
        Returns
        -------
        DictConfig
            Multi-agent configuration with angular velocity arrays
        """
        with initialize(version_base=None, config_path="../conf"):
            cfg = compose(
                config_name="config",
                overrides=[
                    "navigator.type=multi",
                    "navigator.num_agents=3",
                    "navigator.orientations=[0.0,90.0,180.0]",
                    "navigator.angular_velocities=[10.0,20.0,30.0]",
                    "navigator.max_speeds=[5.0,10.0,15.0]"
                ]
            )
            return cfg

    @staticmethod
    def create_navigator_with_angular_velocity(orientation=0.0, angular_velocity=30.0):
        """Create a single-agent navigator with specified orientation and angular velocity.
        
        This factory method creates a navigator instance with specific angular velocity
        configuration using the new protocol-based architecture and Hydra integration.
        
        Parameters
        ----------
        orientation : float, optional
            Initial orientation in degrees, by default 0.0
        angular_velocity : float, optional
            Angular velocity in degrees per timestep, by default 30.0
            
        Returns
        -------
        NavigatorProtocol
            Navigator instance configured with specified parameters
            
        Notes
        -----
        Uses the new factory pattern with Hydra configuration composition
        for consistent navigator instantiation across test scenarios.
        """
        # Create configuration for single agent navigator
        with initialize(version_base=None, config_path="../conf"):
            cfg = compose(
                config_name="config",
                overrides=[
                    "navigator.type=single",
                    f"navigator.orientation={orientation}",
                    f"navigator.angular_velocity={angular_velocity}"
                ]
            )
        
        # Import the factory method for navigator creation
        from {{cookiecutter.project_slug}}.api.navigation import create_navigator
        navigator = create_navigator(cfg.navigator)
        
        return navigator
    
    @staticmethod
    def create_multi_agent_navigator(orientations, angular_velocities):
        """Create a multi-agent navigator with specified orientations and angular velocities.
        
        Factory method for creating multi-agent navigators with individual angular
        velocity settings using Hydra configuration and protocol-based architecture.
        
        Parameters
        ----------
        orientations : array_like
            Initial orientations for each agent in degrees
        angular_velocities : array_like
            Angular velocities for each agent in degrees per timestep
            
        Returns
        -------
        NavigatorProtocol
            Multi-agent navigator configured with specified parameters
            
        Notes
        -----
        Creates positions automatically and validates that orientations and
        angular_velocities arrays have consistent dimensions.
        """
        num_agents = len(orientations)
        
        # Create configuration for multi-agent navigator
        with initialize(version_base=None, config_path="../conf"):
            cfg = compose(
                config_name="config",
                overrides=[
                    "navigator.type=multi",
                    f"navigator.num_agents={num_agents}",
                    f"navigator.orientations={list(orientations)}",
                    f"navigator.angular_velocities={list(angular_velocities)}"
                ]
            )
        
        # Import the factory method for navigator creation
        from {{cookiecutter.project_slug}}.api.navigation import create_navigator
        navigator = create_navigator(cfg.navigator)
        
        return navigator
    
    @staticmethod
    def assert_orientation_close(actual, expected, tolerance=1e-5):
        """Assert that orientation values are close, accounting for floating-point precision.
        
        Specialized assertion for orientation comparisons with appropriate numerical
        tolerance for scientific computing accuracy requirements.
        
        Parameters
        ----------
        actual : float or array_like
            Actual orientation value(s) to test
        expected : float or array_like
            Expected orientation value(s)
        tolerance : float, optional
            Absolute tolerance for comparison, by default 1e-5
            
        Notes
        -----
        Uses numpy.isclose for robust floating-point comparison with
        scientific computing precision standards.
        """
        assert np.allclose(actual, expected, atol=tolerance)

    def test_single_agent_angular_velocity(self):
        """Test orientation updates for a single agent with angular velocity.
        
        Validates basic angular velocity functionality including orientation updates
        over multiple timesteps and proper angle normalization for single-agent scenarios.
        """
        # Create a navigator with initial orientation 0 and angular velocity 30
        navigator = self.create_navigator_with_angular_velocity(orientation=0.0, angular_velocity=30.0)
        
        # Verify initial state
        assert isinstance(navigator, NavigatorProtocol)
        self.assert_orientation_close(navigator.orientations[0], 0.0)
        self.assert_orientation_close(navigator.angular_velocities[0], 30.0)

        # Create an environment array for step()
        env = np.zeros((10, 10))

        # Take a step (dt=1 equivalent)
        navigator.step(env)
        self.assert_orientation_close(navigator.orientations[0], 30.0)

        # Take another step
        navigator.step(env)
        self.assert_orientation_close(navigator.orientations[0], 60.0)

        # Take additional steps to test angle normalization
        for _ in range(10):
            navigator.step(env)
        # After 12 total steps: 0 + (30 * 12) = 360, normalized to 0
        self.assert_orientation_close(navigator.orientations[0], 0.0)

    def test_multi_agent_angular_velocity(self):
        """Test orientation updates for multiple agents with different angular velocities.
        
        Validates multi-agent angular velocity functionality ensuring independent
        agent updates and proper vectorized operations across agent populations.
        """
        # Create a navigator with multiple agents and different initial orientations
        orientations = np.array([0.0, 90.0, 180.0])
        angular_velocities = np.array([10.0, 20.0, 30.0])
        navigator = self.create_multi_agent_navigator(orientations, angular_velocities)

        # Verify initial state
        assert navigator.num_agents == 3
        self.assert_orientation_close(navigator.orientations, orientations)
        self.assert_orientation_close(navigator.angular_velocities, angular_velocities)

        # Create an environment array for step()
        env = np.zeros((10, 10))

        # Take a step (dt=1 equivalent)
        navigator.step(env)
        expected_orientations = np.array([10.0, 110.0, 210.0])
        self.assert_orientation_close(navigator.orientations, expected_orientations)

        # Take another step
        navigator.step(env)
        expected_orientations = np.array([20.0, 130.0, 240.0])
        self.assert_orientation_close(navigator.orientations, expected_orientations)

    def test_negative_angular_velocity(self):
        """Test orientation updates with negative angular velocity (clockwise rotation).
        
        Validates proper handling of negative angular velocities representing
        clockwise rotation and ensures correct angle normalization for negative values.
        """
        # Create a navigator with initial orientation 180 and negative angular velocity
        navigator = self.create_navigator_with_angular_velocity(orientation=180.0, angular_velocity=-45.0)

        # Create an environment array for step()
        env = np.zeros((10, 10))

        # Take a step
        navigator.step(env)
        self.assert_orientation_close(navigator.orientations[0], 135.0)  # 180 - 45 = 135

        # Take another step
        navigator.step(env)
        self.assert_orientation_close(navigator.orientations[0], 90.0)   # 135 - 45 = 90

        # Take more steps to test negative angle normalization
        navigator.step(env)
        navigator.step(env)
        navigator.step(env)  # Should be at 180 - (45 * 5) = -45, normalized to 315
        self.assert_orientation_close(navigator.orientations[0], 315.0)

    def test_large_angular_change(self):
        """Test orientation updates with large angular changes (> 360 degrees).
        
        Validates proper angle normalization for large angular velocity values
        and ensures mathematical consistency across multiple rotation cycles.
        """
        # Create a navigator with large angular velocity
        navigator = self.create_navigator_with_angular_velocity(orientation=0.0, angular_velocity=180.0)

        # Create an environment array for step()
        env = np.zeros((10, 10))

        # Take steps to accumulate large angular changes
        navigator.step(env)
        self.assert_orientation_close(navigator.orientations[0], 180.0)

        navigator.step(env)
        self.assert_orientation_close(navigator.orientations[0], 0.0)  # 360 normalized to 0

        navigator.step(env)
        self.assert_orientation_close(navigator.orientations[0], 180.0)

    def test_hydra_configuration_angular_velocity(self, hydra_config_fixture):
        """Test angular velocity configuration through Hydra configuration system.
        
        Validates integration of angular velocity parameters with Hydra configuration
        management including parameter composition and override capabilities.
        
        Parameters
        ----------
        hydra_config_fixture : DictConfig
            Hydra configuration fixture with angular velocity settings
        """
        # Import the factory method for navigator creation
        from {{cookiecutter.project_slug}}.api.navigation import create_navigator
        
        # Create navigator from Hydra configuration
        navigator = create_navigator(hydra_config_fixture.navigator)
        
        # Verify configuration was applied correctly
        assert isinstance(navigator, NavigatorProtocol)
        self.assert_orientation_close(navigator.orientations[0], 0.0)
        self.assert_orientation_close(navigator.angular_velocities[0], 30.0)
        
        # Test step behavior with configured angular velocity
        env = np.zeros((10, 10))
        navigator.step(env)
        self.assert_orientation_close(navigator.orientations[0], 30.0)

    def test_hydra_multi_agent_angular_velocity(self, multi_agent_hydra_config):
        """Test multi-agent angular velocity configuration through Hydra.
        
        Validates Hydra configuration support for multi-agent scenarios with
        individual angular velocity specifications and parameter validation.
        
        Parameters
        ----------
        multi_agent_hydra_config : DictConfig
            Multi-agent Hydra configuration with angular velocity arrays
        """
        # Import the factory method for navigator creation
        from {{cookiecutter.project_slug}}.api.navigation import create_navigator
        
        # Create multi-agent navigator from Hydra configuration
        navigator = create_navigator(multi_agent_hydra_config.navigator)
        
        # Verify configuration was applied correctly
        assert navigator.num_agents == 3
        expected_orientations = np.array([0.0, 90.0, 180.0])
        expected_angular_velocities = np.array([10.0, 20.0, 30.0])
        
        self.assert_orientation_close(navigator.orientations, expected_orientations)
        self.assert_orientation_close(navigator.angular_velocities, expected_angular_velocities)

    def test_hydra_configuration_overrides(self):
        """Test Hydra configuration override capabilities for angular velocity.
        
        Validates dynamic parameter modification through Hydra override syntax
        ensuring flexible experiment configuration and parameter exploration.
        """
        # Test configuration override through compose
        with initialize(version_base=None, config_path="../conf"):
            cfg = compose(
                config_name="config",
                overrides=[
                    "navigator.type=single",
                    "navigator.angular_velocity=75.0",  # Override default value
                    "navigator.orientation=45.0"
                ]
            )
        
        # Import the factory method for navigator creation
        from {{cookiecutter.project_slug}}.api.navigation import create_navigator
        
        # Create navigator with overridden configuration
        navigator = create_navigator(cfg.navigator)
        
        # Verify overrides were applied
        self.assert_orientation_close(navigator.orientations[0], 45.0)
        self.assert_orientation_close(navigator.angular_velocities[0], 75.0)

    def test_angular_velocity_property_access(self):
        """Test angular velocity property access through NavigatorProtocol.
        
        Validates that angular velocity properties conform to the NavigatorProtocol
        interface and provide consistent data access patterns.
        """
        # Create navigator with known angular velocity
        navigator = self.create_navigator_with_angular_velocity(angular_velocity=25.0)
        
        # Test property access
        angular_velocities = navigator.angular_velocities
        assert isinstance(angular_velocities, np.ndarray)
        assert angular_velocities.shape == (1,)  # Single agent
        self.assert_orientation_close(angular_velocities[0], 25.0)
        
        # Test multi-agent property access
        multi_navigator = self.create_multi_agent_navigator(
            orientations=[0.0, 90.0], 
            angular_velocities=[15.0, 35.0]
        )
        
        multi_angular_velocities = multi_navigator.angular_velocities
        assert isinstance(multi_angular_velocities, np.ndarray)
        assert multi_angular_velocities.shape == (2,)  # Two agents
        expected_values = np.array([15.0, 35.0])
        self.assert_orientation_close(multi_angular_velocities, expected_values)

    def test_angular_velocity_edge_cases(self):
        """Test angular velocity edge cases and boundary conditions.
        
        Validates robust handling of edge cases including zero angular velocity,
        very small values, and boundary conditions for numerical stability.
        """
        # Test zero angular velocity
        navigator_zero = self.create_navigator_with_angular_velocity(angular_velocity=0.0)
        env = np.zeros((10, 10))
        
        initial_orientation = navigator_zero.orientations[0]
        navigator_zero.step(env)
        # Orientation should remain unchanged with zero angular velocity
        self.assert_orientation_close(navigator_zero.orientations[0], initial_orientation)
        
        # Test very small angular velocity
        navigator_small = self.create_navigator_with_angular_velocity(angular_velocity=1e-6)
        navigator_small.step(env)
        self.assert_orientation_close(navigator_small.orientations[0], 1e-6, tolerance=1e-8)
        
        # Test very large angular velocity
        navigator_large = self.create_navigator_with_angular_velocity(angular_velocity=720.0)
        navigator_large.step(env)
        # 720 degrees should normalize to 0 (720 % 360 = 0)
        self.assert_orientation_close(navigator_large.orientations[0], 0.0)

    def test_angular_velocity_consistency_across_resets(self):
        """Test angular velocity consistency across navigator reset operations.
        
        Validates that angular velocity configuration persists correctly through
        navigator reset operations and maintains consistent behavior patterns.
        """
        # Create navigator with specific angular velocity
        navigator = self.create_navigator_with_angular_velocity(
            orientation=90.0, 
            angular_velocity=20.0
        )
        
        # Take some steps to change orientation
        env = np.zeros((10, 10))
        navigator.step(env)
        navigator.step(env)
        
        # Verify orientation changed
        assert not np.isclose(navigator.orientations[0], 90.0)
        
        # Reset navigator
        navigator.reset(orientations=np.array([90.0]))
        
        # Verify angular velocity preserved but orientation reset
        self.assert_orientation_close(navigator.orientations[0], 90.0)
        self.assert_orientation_close(navigator.angular_velocities[0], 20.0)
        
        # Verify behavior consistency after reset
        navigator.step(env)
        self.assert_orientation_close(navigator.orientations[0], 110.0)  # 90 + 20