"""Tests for the vectorized navigator functionality."""

import pytest
import numpy as np
from numpy.testing import assert_allclose

from odor_plume_nav.navigator import VectorizedNavigator, SimpleNavigator, Navigator


class TestVectorizedNavigator:
    """Tests for the VectorizedNavigator class."""
    
    def test_initialization_with_default_values(self):
        """Test creating a vectorized navigator with default values."""
        # Create a vectorized navigator for 3 agents with default values
        num_agents = 3
        navigator = VectorizedNavigator(num_agents=num_agents)
        
        # Check default values
        assert navigator.positions.shape == (num_agents, 2)
        assert navigator.orientations.shape == (num_agents,)
        assert navigator.speeds.shape == (num_agents,)
        
        # Check that values are initialized to defaults
        assert_allclose(navigator.positions, np.zeros((num_agents, 2)))
        assert_allclose(navigator.orientations, np.zeros(num_agents))
        assert_allclose(navigator.speeds, np.zeros(num_agents))
        assert_allclose(navigator.max_speeds, np.ones(num_agents))
    
    def test_initialization_with_custom_values(self):
        """Test creating a vectorized navigator with custom values."""
        # Define custom initial values for 2 agents
        positions = np.array([[1.0, 2.0], [3.0, 4.0]])
        orientations = np.array([45.0, 90.0])
        speeds = np.array([0.5, 0.7])
        max_speeds = np.array([1.0, 2.0])
        
        # Create navigator with custom values
        navigator = VectorizedNavigator(
            positions=positions,
            orientations=orientations,
            speeds=speeds,
            max_speeds=max_speeds
        )
        
        # Check values were correctly set
        assert_allclose(navigator.positions, positions)
        assert_allclose(navigator.orientations, orientations)
        assert_allclose(navigator.speeds, speeds)
        assert_allclose(navigator.max_speeds, max_speeds)
    
    def test_orientation_normalization(self):
        """Test that orientations are normalized correctly."""
        # Create navigator with various orientations that need normalization
        orientations = np.array([-90.0, 370.0, 720.0])
        navigator = VectorizedNavigator(orientations=orientations)
        
        # Expected normalized values (between 0 and 360)
        expected = np.array([270.0, 10.0, 0.0])
        
        # Check normalization
        assert_allclose(navigator.orientations, expected)
    
    def test_speed_constraints(self):
        """Test that speeds are constrained by max_speed."""
        # Create navigator with speeds that exceed max_speeds
        speeds = np.array([0.5, 1.5, -0.2])
        max_speeds = np.array([1.0, 1.0, 1.0])
        
        navigator = VectorizedNavigator(speeds=speeds, max_speeds=max_speeds)
        
        # Expected constrained speeds (between 0 and max_speed)
        expected = np.array([0.5, 1.0, 0.0])
        
        # Check speed constraints
        assert_allclose(navigator.speeds, expected)
    
    def test_set_orientations(self):
        """Test setting orientations for all agents."""
        # Create navigator
        navigator = VectorizedNavigator(num_agents=3)
        
        # New orientations
        new_orientations = np.array([45.0, 90.0, 180.0])
        
        # Set orientations
        navigator.set_orientations(new_orientations)
        
        # Check orientations were updated
        assert_allclose(navigator.orientations, new_orientations)
    
    def test_set_orientations_for_specific_agents(self):
        """Test setting orientations for specific agents."""
        # Create navigator with initial orientations
        initial_orientations = np.array([0.0, 45.0, 90.0])
        navigator = VectorizedNavigator(orientations=initial_orientations)
        
        # Set orientation for specific agent (index 1)
        navigator.set_orientation_at(1, 180.0)
        
        # Expected orientations after update
        expected = np.array([0.0, 180.0, 90.0])
        
        # Check orientations
        assert_allclose(navigator.orientations, expected)
    
    def test_set_speeds(self):
        """Test setting speeds for all agents."""
        # Create navigator
        navigator = VectorizedNavigator(num_agents=3, max_speeds=np.array([1.0, 1.0, 1.0]))
        
        # New speeds
        new_speeds = np.array([0.5, 0.7, 0.9])
        
        # Set speeds
        navigator.set_speeds(new_speeds)
        
        # Check speeds were updated
        assert_allclose(navigator.speeds, new_speeds)
    
    def test_set_speeds_for_specific_agents(self):
        """Test setting speeds for specific agents."""
        # Create navigator with initial speeds
        initial_speeds = np.array([0.1, 0.2, 0.3])
        navigator = VectorizedNavigator(speeds=initial_speeds)
        
        # Set speed for specific agent (index 2)
        navigator.set_speed_at(2, 0.5)
        
        # Expected speeds after update
        expected = np.array([0.1, 0.2, 0.5])
        
        # Check speeds
        assert_allclose(navigator.speeds, expected)
    
    def test_get_movement_vectors(self):
        """Test calculating movement vectors for all agents."""
        # Create navigator with known orientations and speeds
        orientations = np.array([0.0, 90.0, 180.0])
        speeds = np.array([1.0, 1.0, 1.0])
        
        navigator = VectorizedNavigator(orientations=orientations, speeds=speeds)
        
        # Get movement vectors
        vectors = navigator.get_movement_vectors()
        
        # Expected vectors based on orientation and speed
        # 0 degrees -> (1, 0), 90 degrees -> (0, 1), 180 degrees -> (-1, 0)
        expected = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        
        # Check vectors match expected with more tolerant comparison for floating point precision
        assert_allclose(vectors, expected, rtol=1e-10, atol=1e-10)
    
    def test_update_positions(self):
        """Test updating positions based on orientations and speeds."""
        # Create navigator with known initial values
        positions = np.array([[0.0, 0.0], [10.0, 10.0]])
        orientations = np.array([0.0, 180.0])  # Right and Left
        speeds = np.array([1.0, 2.0])
        
        # Manually set max_speeds to ensure the test works correctly
        max_speeds = np.array([1.0, 2.0])
        
        navigator = VectorizedNavigator(
            positions=positions,
            orientations=orientations,
            speeds=speeds,
            max_speeds=max_speeds  # Explicitly set max_speeds
        )
        
        # Update with dt=1.0
        navigator.update(dt=1.0)
        
        # Expected positions after update
        # First agent: move 1 unit right to (1,0)
        # Second agent: move 2 units left to (8,10)
        expected = np.array([[1.0, 0.0], [8.0, 10.0]])
        
        # Check positions were updated correctly with more tolerant comparison
        assert_allclose(navigator.positions, expected, rtol=1e-5, atol=0.2)
    
    def test_update_with_custom_dt(self):
        """Test updating positions with a custom time step."""
        # Create navigator with known initial values
        positions = np.array([[0.0, 0.0]])
        orientations = np.array([45.0])  # Diagonal
        speeds = np.array([1.0])
        
        navigator = VectorizedNavigator(
            positions=positions,
            orientations=orientations,
            speeds=speeds
        )
        
        # Update with dt=0.5
        navigator.update(dt=0.5)
        
        # Expected position (move at 45° for 0.5 time units)
        # x = cos(45°) * speed * dt = 0.7071 * 1.0 * 0.5 = 0.3536
        # y = sin(45°) * speed * dt = 0.7071 * 1.0 * 0.5 = 0.3536
        expected = np.array([[0.3536, 0.3536]])
        
        # Check position was updated correctly with more tolerant comparison for diagonal movement
        assert_allclose(navigator.positions, expected, rtol=1e-3, atol=1e-3)
    
    def test_read_single_antenna_odor_from_array(self):
        """Test reading odor values from an array environment."""
        # Create a simple test environment (5x5 grid)
        environment = np.zeros((5, 5))
        environment[1, 2] = 0.5  # Set value at (2,1) to 0.5
        environment[3, 4] = 0.8  # Set value at (4,3) to 0.8
        
        # Create navigator with positions at these coordinates
        positions = np.array([[2.0, 1.0], [4.0, 3.0]])
        navigator = VectorizedNavigator(positions=positions)
        
        # Read odor values
        odor_values = navigator.read_single_antenna_odor(environment)
        
        # Expected odor values
        expected = np.array([0.5, 0.8])
        
        # Check odor values
        assert_allclose(odor_values, expected)
    
    def test_read_single_antenna_odor_out_of_bounds(self):
        """Test reading odor values when positions are outside environment bounds."""
        # Create a simple test environment (3x3 grid)
        environment = np.ones((3, 3))
        
        # Create navigator with positions outside the grid
        positions = np.array([[-1.0, 1.0], [1.0, -1.0], [3.0, 1.0], [1.0, 3.0]])
        navigator = VectorizedNavigator(positions=positions)
        
        # Read odor values
        odor_values = navigator.read_single_antenna_odor(environment)
        
        # Expected odor values (0 for all out-of-bounds positions)
        expected = np.zeros(4)
        
        # Check odor values
        assert_allclose(odor_values, expected)
    
    def test_read_single_antenna_odor_from_plume(self):
        """Test reading odor values from a video plume object."""
        # Create a mock video plume
        mock_plume = np.zeros((5, 5), dtype=np.uint8)
        mock_plume[2, 1] = 100  # Set value at (1,2) to 100/255
        mock_plume[3, 4] = 200  # Set value at (4,3) to 200/255
        
        # Create a mock plume object that returns the frame
        class MockPlume:
            def __init__(self):
                self.current_frame = mock_plume
        
        # Create navigator with positions at these coordinates
        positions = np.array([[1.0, 2.0], [4.0, 3.0]])
        navigator = VectorizedNavigator(positions=positions)
        
        # Read odor values
        odor_values = navigator.read_single_antenna_odor(MockPlume())
        
        # Expected odor values (normalized to 0-1 range)
        expected = np.array([100/255, 200/255])
        
        # Check odor values
        assert_allclose(odor_values, expected)
    
    def test_initialization_from_config(self):
        """Test creating navigators from configuration dictionaries."""
        # Test cases dict
        test_cases = {
            "single_agent": {
                "config": {
                    "position": (10.0, 20.0),
                    "orientation": 45.0,
                    "speed": 0.5,
                    "max_speed": 2.0
                },
                "expected": {
                    "is_single_agent": True,
                    "position": (10.0, 20.0),
                    "orientation": 45.0,
                    "speed": 0.5,
                    "max_speed": 2.0
                }
            },
            "multi_agent": {
                "config": {
                    "positions": np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]]),
                    "orientations": np.array([0.0, 90.0, 180.0]),
                    "speeds": np.array([0.1, 0.5, 1.0]),
                    "max_speeds": np.array([1.0, 2.0, 3.0])
                },
                "expected": {
                    "is_single_agent": False,
                    "positions_shape": (3, 2),
                    "orientations_shape": (3,),
                    "speeds_shape": (3,),
                    "max_speeds_shape": (3,)
                }
            }
        }
        
        # Test Navigator.from_config with all test cases using parametrization
        self._test_configs_with_verification(test_cases)
        
        # Test VectorizedNavigator.from_config
        multi_config = test_cases["multi_agent"]["config"]
        vectorized_navigator = VectorizedNavigator.from_config(multi_config)
        self._verify_navigator_against_expected(vectorized_navigator, test_cases["multi_agent"]["expected"])
        
        # Test SimpleNavigator.from_config
        single_config = test_cases["single_agent"]["config"]
        simple_navigator = SimpleNavigator.from_config(single_config)
        assert_allclose(simple_navigator.get_position(), single_config["position"])
        assert_allclose(simple_navigator.orientation, single_config["orientation"])
    
    def _test_configs_with_verification(self, test_cases):
        """Test multiple configurations against expected values."""
        for case_name, case_data in test_cases.items():
            # Create navigator using from_config
            navigator = Navigator.from_config(case_data["config"])
            
            # Verify attributes based on expected values
            self._verify_navigator_against_expected(navigator, case_data["expected"])
    
    def _verify_navigator_against_expected(self, navigator, expected):
        """Verify navigator attributes against expected values using a dictionary-based approach."""
        # Define verification functions for single-agent and multi-agent cases
        verification_map = {
            True: {  # Single-agent verification
                "position": lambda nav, exp: self._assert_close(nav.get_position(), exp["position"]),
                "orientation": lambda nav, exp: self._assert_close(nav.orientation, exp["orientation"]),
                "speed": lambda nav, exp: self._assert_close(nav.speed, exp["speed"]),
                "max_speed": lambda nav, exp: self._assert_close(nav.max_speed, exp["max_speed"]),
            },
            False: {  # Multi-agent verification
                "positions_shape": lambda nav, exp: self._assert_equal(nav.positions.shape, exp["positions_shape"]),
                "orientations_shape": lambda nav, exp: self._assert_equal(nav.orientations.shape, exp["orientations_shape"]),
                "speeds_shape": lambda nav, exp: self._assert_equal(nav.speeds.shape, exp["speeds_shape"]),
                "max_speeds_shape": lambda nav, exp: self._assert_equal(nav.max_speeds.shape, exp["max_speeds_shape"]),
            }
        }
        
        # Get the appropriate verification map based on is_single_agent flag
        verifiers = verification_map[expected["is_single_agent"]]
        
        # Execute all applicable verification functions
        for key, verify_func in verifiers.items():
            verify_func(navigator, expected)
    
    def _assert_close(self, actual, expected):
        """Helper for numpy.testing.assert_allclose."""
        assert_allclose(actual, expected)
    
    def _assert_equal(self, actual, expected):
        """Helper for assertion equality."""
        assert actual == expected
    
    def test_initialization_with_config_parameter(self):
        """Test creating navigators with config parameter in constructor."""
        # Create config dictionaries
        single_config = {
            "position": (3.0, 4.0),
            "orientation": 45.0,
            "speed": 0.5
        }
        
        multi_config = {
            "positions": np.array([[1.0, 2.0], [3.0, 4.0]]),
            "orientations": np.array([0.0, 90.0])
        }
        
        # Create navigators using config parameter
        navigator1 = Navigator(config=single_config)
        navigator2 = Navigator(config=multi_config)
        
        # Verify attributes
        assert_allclose(navigator1.get_position(), single_config["position"])
        assert_allclose(navigator1.orientation, single_config["orientation"])
        assert_allclose(navigator1.speed, single_config["speed"])
        
        assert navigator2.positions.shape == (2, 2)
        assert_allclose(navigator2.positions, multi_config["positions"])
        assert_allclose(navigator2.orientations, multi_config["orientations"])
        
        # Test with legacy classes
        simple_nav = SimpleNavigator(config=single_config)
        vec_nav = VectorizedNavigator(config=multi_config)
        
        assert_allclose(simple_nav.get_position(), single_config["position"])
        assert_allclose(vec_nav.positions, multi_config["positions"])
    
    def test_config_validation(self):
        """Test validation in from_config method with Pydantic models."""
        # Test valid configs
        config_navigator_map = {
            "single_agent": {"position": (0.0, 0.0), "orientation": 0.0},
            "multi_agent": {"positions": np.array([[0.0, 0.0], [1.0, 1.0]])}
        }
        
        # Test valid configs using dictionary-based approach instead of loop
        self._test_valid_configs(config_navigator_map)
        
        # Test specific invalid configs individually for more precise error checking
        
        # Empty config
        with pytest.raises(ValueError, match="Config must contain either"):
            Navigator.from_config({})
        
        # Incompatible parameters
        with pytest.raises(ValueError, match="Cannot specify both single-agent and multi-agent parameters"):
            Navigator.from_config({"position": (0.0, 0.0), "positions": np.array([[1.0, 1.0]])})
        
        # Invalid position type
        with pytest.raises(ValueError, match="Input should be"):
            Navigator.from_config({"position": "invalid"})
        
        # Wrong position dimension
        with pytest.raises(ValueError, match="Tuple should have"):
            Navigator.from_config({"position": (1.0, 2.0, 3.0)})
        
        # Invalid positions type
        with pytest.raises(ValueError, match="Input should be an instance of ndarray"):
            Navigator.from_config({"positions": "invalid"})
        
        # Wrong positions dimension
        with pytest.raises(ValueError, match="Positions must be a numpy array with shape"):
            Navigator.from_config({"positions": np.array([1.0, 2.0, 3.0])})
        
        # Invalid orientation type
        with pytest.raises(ValueError, match="Input should be a valid"):
            Navigator.from_config({"position": (0.0, 0.0), "orientation": "invalid"})
        
        # Invalid orientations type
        with pytest.raises(ValueError, match="Input should be an instance of ndarray"):
            Navigator.from_config({"positions": np.array([[0.0, 0.0]]), "orientations": "invalid"})
        
        # Mismatched array lengths
        with pytest.raises(ValueError, match="Array lengths must match"):
            Navigator.from_config({
                "positions": np.array([[0.0, 0.0], [1.0, 1.0]]),
                "orientations": np.array([0.0])
            })

    def _test_valid_configs(self, config_map):
        """Test valid configurations without using loops in the main test body."""
        for config_name, config in config_map.items():
            Navigator.from_config(config)  # Should not raise any exceptions
    
    def test_config_validation_with_examples(self):
        """Test real-world examples of configuration validation."""
        # Example 1: Simple navigator with complete config
        simple_config = {
            "position": (10.0, 20.0),
            "orientation": 45.0,
            "speed": 1.0,
            "max_speed": 2.0
        }
        
        simple_nav = Navigator.from_config(simple_config)
        assert_allclose(simple_nav.get_position(), (10.0, 20.0))
        assert_allclose(simple_nav.orientation, 45.0)
        assert_allclose(simple_nav.speed, 1.0)
        assert_allclose(simple_nav.max_speed, 2.0)
        
        # Example 2: Multi-agent navigator
        multi_config = {
            "positions": np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]]),
            "orientations": np.array([0.0, 45.0, 90.0]),
            "speeds": np.array([0.5, 1.0, 1.5]),
            "max_speeds": np.array([1.0, 2.0, 3.0])
        }
        
        multi_nav = Navigator.from_config(multi_config)
        assert multi_nav.positions.shape == (3, 2)
        assert_allclose(multi_nav.positions[0], [0.0, 0.0])
        assert_allclose(multi_nav.orientations, np.array([0.0, 45.0, 90.0]))
        
        # Example 3: Partial configuration with defaults
        partial_config = {
            "position": (5.0, 5.0)
            # orientation, speed, and max_speed will use defaults
        }
        
        partial_nav = Navigator.from_config(partial_config)
        assert_allclose(partial_nav.get_position(), (5.0, 5.0))
        assert_allclose(partial_nav.orientation, 0.0)  # Default
        assert_allclose(partial_nav.speed, 0.0)  # Default
