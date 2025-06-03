"""Tests for the vectorized navigator functionality with updated package structure and Hydra configuration integration."""

import pytest
import numpy as np
from numpy.testing import assert_allclose
from omegaconf import DictConfig, OmegaConf

from {{cookiecutter.project_slug}}.core.navigator import Navigator
from {{cookiecutter.project_slug}}.api.navigation import create_navigator


# Create a mock plume for testing
mock_plume = np.zeros((100, 100))
mock_plume[40:60, 40:60] = 1.0  # Create a region with non-zero values


@pytest.fixture
def basic_single_agent_config():
    """Fixture providing basic single-agent Hydra configuration."""
    return OmegaConf.create({
        "type": "single",
        "position": [0.0, 0.0],
        "orientation": 0.0,
        "speed": 0.0,
        "max_speed": 1.0,
        "angular_velocity": 0.0
    })


@pytest.fixture
def basic_multi_agent_config():
    """Fixture providing basic multi-agent Hydra configuration."""
    return OmegaConf.create({
        "type": "multi",
        "positions": [[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]],
        "orientations": [0.0, 45.0, 90.0],
        "speeds": [0.0, 0.0, 0.0],
        "max_speeds": [1.0, 1.0, 1.0],
        "angular_velocities": [0.0, 0.0, 0.0]
    })


@pytest.fixture
def custom_multi_agent_config():
    """Fixture providing custom multi-agent configuration for testing."""
    return OmegaConf.create({
        "type": "multi",
        "positions": [[1.0, 2.0], [3.0, 4.0]],
        "orientations": [45.0, 90.0],
        "speeds": [0.5, 0.7],
        "max_speeds": [1.0, 2.0],
        "angular_velocities": [0.1, 0.2]
    })


def create_test_navigator_from_config(config: DictConfig):
    """Helper function to create navigator from configuration for testing."""
    if config.type == "single":
        return Navigator.single(
            position=config.get("position", (0.0, 0.0)),
            orientation=config.get("orientation", 0.0),
            speed=config.get("speed", 0.0),
            max_speed=config.get("max_speed", 1.0)
        )
    elif config.type == "multi":
        return Navigator.multi(
            positions=np.array(config.get("positions", [[0.0, 0.0]])),
            orientations=np.array(config.get("orientations", [0.0])),
            speeds=np.array(config.get("speeds", [0.0])),
            max_speeds=np.array(config.get("max_speeds", [1.0]))
        )
    else:
        raise ValueError(f"Unknown navigator type: {config.type}")


class TestHydraConfigurationIntegration:
    """Tests for Hydra configuration integration with navigator instantiation."""
    
    def test_single_agent_from_hydra_config(self, basic_single_agent_config):
        """Test creating single-agent navigator from Hydra configuration."""
        navigator = create_test_navigator_from_config(basic_single_agent_config)
        
        # Verify the navigator was created correctly
        assert navigator.num_agents == 1
        assert_allclose(navigator.positions[0], [0.0, 0.0])
        assert_allclose(navigator.orientations[0], 0.0)
        assert_allclose(navigator.speeds[0], 0.0)
        assert_allclose(navigator.max_speeds[0], 1.0)
    
    def test_multi_agent_from_hydra_config(self, basic_multi_agent_config):
        """Test creating multi-agent navigator from Hydra configuration."""
        navigator = create_test_navigator_from_config(basic_multi_agent_config)
        
        # Verify the navigator was created correctly
        assert navigator.num_agents == 3
        expected_positions = np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]])
        expected_orientations = np.array([0.0, 45.0, 90.0])
        
        assert_allclose(navigator.positions, expected_positions)
        assert_allclose(navigator.orientations, expected_orientations)
        assert_allclose(navigator.speeds, np.zeros(3))
        assert_allclose(navigator.max_speeds, np.ones(3))
    
    def test_custom_multi_agent_from_hydra_config(self, custom_multi_agent_config):
        """Test creating multi-agent navigator with custom Hydra configuration."""
        navigator = create_test_navigator_from_config(custom_multi_agent_config)
        
        # Verify custom parameters were applied
        assert navigator.num_agents == 2
        expected_positions = np.array([[1.0, 2.0], [3.0, 4.0]])
        expected_orientations = np.array([45.0, 90.0])
        expected_speeds = np.array([0.5, 0.7])
        expected_max_speeds = np.array([1.0, 2.0])
        
        assert_allclose(navigator.positions, expected_positions)
        assert_allclose(navigator.orientations, expected_orientations)
        assert_allclose(navigator.speeds, expected_speeds)
        assert_allclose(navigator.max_speeds, expected_max_speeds)
    
    def test_hydra_config_override_patterns(self):
        """Test dynamic configuration override patterns using OmegaConf."""
        base_config = OmegaConf.create({
            "type": "multi",
            "positions": [[0.0, 0.0], [5.0, 5.0]],
            "orientations": [0.0, 90.0],
            "speeds": [0.5, 0.5],
            "max_speeds": [1.0, 1.0]
        })
        
        # Test configuration override
        override_config = OmegaConf.merge(base_config, OmegaConf.create({
            "speeds": [1.0, 1.5],
            "max_speeds": [2.0, 3.0]
        }))
        
        navigator = create_test_navigator_from_config(override_config)
        
        # Verify overrides were applied
        assert_allclose(navigator.speeds, [1.0, 1.5])
        assert_allclose(navigator.max_speeds, [2.0, 3.0])
        # Original values should be preserved
        assert_allclose(navigator.positions, [[0.0, 0.0], [5.0, 5.0]])
        assert_allclose(navigator.orientations, [0.0, 90.0])


class TestNavigatorInstantiationPatterns:
    """Tests for new navigator instantiation patterns with updated API."""
    
    def test_factory_method_single_agent(self):
        """Test single-agent factory method instantiation."""
        navigator = Navigator.single(
            position=(10.0, 15.0),
            orientation=45.0,
            speed=0.8,
            max_speed=2.0
        )
        
        assert navigator.num_agents == 1
        assert_allclose(navigator.positions[0], [10.0, 15.0])
        assert_allclose(navigator.orientations[0], 45.0)
        assert_allclose(navigator.speeds[0], 0.8)
        assert_allclose(navigator.max_speeds[0], 2.0)
    
    def test_factory_method_multi_agent(self):
        """Test multi-agent factory method instantiation."""
        positions = np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]])
        orientations = np.array([0.0, 90.0, 180.0])
        speeds = np.array([0.5, 1.0, 1.5])
        max_speeds = np.array([1.0, 2.0, 3.0])
        
        navigator = Navigator.multi(
            positions=positions,
            orientations=orientations,
            speeds=speeds,
            max_speeds=max_speeds
        )
        
        assert navigator.num_agents == 3
        assert_allclose(navigator.positions, positions)
        assert_allclose(navigator.orientations, orientations)
        assert_allclose(navigator.speeds, speeds)
        assert_allclose(navigator.max_speeds, max_speeds)
    
    def test_factory_method_with_partial_parameters(self):
        """Test factory methods with partial parameter specification."""
        # Test single agent with minimal parameters
        navigator_single = Navigator.single(position=(5.0, 7.0))
        
        assert_allclose(navigator_single.positions[0], [5.0, 7.0])
        assert_allclose(navigator_single.orientations[0], 0.0)  # Default
        assert_allclose(navigator_single.speeds[0], 0.0)  # Default
        
        # Test multi-agent with minimal parameters
        positions = np.array([[1.0, 2.0], [3.0, 4.0]])
        navigator_multi = Navigator.multi(positions=positions)
        
        assert navigator_multi.num_agents == 2
        assert_allclose(navigator_multi.positions, positions)
        assert_allclose(navigator_multi.orientations, [0.0, 0.0])  # Default
        assert_allclose(navigator_multi.speeds, [0.0, 0.0])  # Default
        assert_allclose(navigator_multi.max_speeds, [1.0, 1.0])  # Default
    
    def test_vectorized_operations_compatibility(self):
        """Test that new instantiation patterns maintain vectorized operation compatibility."""
        # Create navigator using new factory method
        positions = np.array([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]])
        orientations = np.array([0.0, 90.0, 45.0])
        speeds = np.array([1.0, 0.5, 0.7])
        
        navigator = Navigator.multi(
            positions=positions,
            orientations=orientations,
            speeds=speeds
        )
        
        # Test vectorized odor sampling
        odor_readings = navigator.read_single_antenna_odor(mock_plume)
        
        # Verify vectorized operation returns correct shape
        assert odor_readings.shape == (3,)
        # Verify readings match expected based on positions in mock_plume
        # Position [10,10] is outside odor region, [20,20] and [30,30] are outside too
        expected_readings = np.array([0.0, 0.0, 0.0])
        assert_allclose(odor_readings, expected_readings)
        
        # Test vectorized step operation
        initial_positions = navigator.positions.copy()
        navigator.step(mock_plume)
        
        # Verify positions changed (movement occurred)
        movement_vectors = navigator.positions - initial_positions
        # All agents should have moved (non-zero movement)
        assert np.any(movement_vectors != 0.0)


class TestResearchWorkflowIntegration:
    """Tests for navigator integration in research workflows with Hydra configuration."""
    
    def test_parameter_sweep_simulation(self):
        """Test parameter sweep scenarios common in research workflows."""
        # Simulate parameter sweep over different speed configurations
        speed_values = [0.5, 1.0, 1.5, 2.0]
        navigators = []
        
        for speed in speed_values:
            config = OmegaConf.create({
                "type": "multi",
                "positions": [[10.0, 10.0], [20.0, 20.0]],
                "orientations": [0.0, 90.0],
                "speeds": [speed, speed],
                "max_speeds": [speed * 2, speed * 2]
            })
            
            navigator = create_test_navigator_from_config(config)
            navigators.append(navigator)
        
        # Verify each navigator was configured with the correct speed
        for i, navigator in enumerate(navigators):
            expected_speed = speed_values[i]
            assert_allclose(navigator.speeds, [expected_speed, expected_speed])
            assert_allclose(navigator.max_speeds, [expected_speed * 2, expected_speed * 2])
    
    def test_multi_run_configuration_patterns(self):
        """Test configuration patterns for multi-run experiments."""
        # Base configuration template
        base_template = {
            "type": "multi",
            "positions": [[0.0, 0.0], [50.0, 50.0]],
            "orientations": [0.0, 180.0]
        }
        
        # Different experiment variations
        experiment_configs = [
            {"speeds": [0.5, 0.5], "max_speeds": [1.0, 1.0]},  # Low speed
            {"speeds": [1.0, 1.0], "max_speeds": [2.0, 2.0]},  # Medium speed
            {"speeds": [1.5, 1.5], "max_speeds": [3.0, 3.0]},  # High speed
        ]
        
        results = []
        for exp_config in experiment_configs:
            # Merge base template with experiment-specific parameters
            full_config = OmegaConf.merge(
                OmegaConf.create(base_template),
                OmegaConf.create(exp_config)
            )
            
            navigator = create_test_navigator_from_config(full_config)
            
            # Simulate one step of navigation
            initial_positions = navigator.positions.copy()
            navigator.step(mock_plume)
            final_positions = navigator.positions.copy()
            
            # Calculate movement distance
            movement = np.linalg.norm(final_positions - initial_positions, axis=1)
            results.append({
                "config": exp_config,
                "movement": movement.mean()
            })
        
        # Verify that higher speeds result in larger movements
        movements = [r["movement"] for r in results]
        assert movements[1] > movements[0]  # Medium > Low
        assert movements[2] > movements[1]  # High > Medium
    
    def test_configuration_inheritance_patterns(self):
        """Test configuration inheritance and override patterns."""
        # Parent configuration
        parent_config = OmegaConf.create({
            "type": "multi",
            "positions": [[0.0, 0.0], [10.0, 10.0]],
            "orientations": [0.0, 90.0],
            "speeds": [0.5, 0.5],
            "max_speeds": [1.0, 1.0]
        })
        
        # Child configurations that inherit and override
        child_configs = [
            OmegaConf.create({"speeds": [1.0, 1.0]}),  # Override speeds only
            OmegaConf.create({"orientations": [45.0, 135.0]}),  # Override orientations only
            OmegaConf.create({  # Override multiple parameters
                "speeds": [0.8, 1.2],
                "max_speeds": [2.0, 3.0]
            })
        ]
        
        # Test inheritance for each child configuration
        for i, child_config in enumerate(child_configs):
            merged_config = OmegaConf.merge(parent_config, child_config)
            navigator = create_test_navigator_from_config(merged_config)
            
            # All should inherit base positions
            assert_allclose(navigator.positions, [[0.0, 0.0], [10.0, 10.0]])
            
            # Check specific overrides
            if i == 0:  # Speed override
                assert_allclose(navigator.speeds, [1.0, 1.0])
                assert_allclose(navigator.orientations, [0.0, 90.0])  # Inherited
            elif i == 1:  # Orientation override
                assert_allclose(navigator.orientations, [45.0, 135.0])
                assert_allclose(navigator.speeds, [0.5, 0.5])  # Inherited
            elif i == 2:  # Multiple overrides
                assert_allclose(navigator.speeds, [0.8, 1.2])
                assert_allclose(navigator.max_speeds, [2.0, 3.0])
                assert_allclose(navigator.orientations, [0.0, 90.0])  # Inherited
    
    def test_batch_experiment_execution(self):
        """Test batch experiment execution patterns."""
        # Create a batch of experiment configurations
        batch_configs = []
        
        # Generate 5 different agent configurations
        for i in range(5):
            config = OmegaConf.create({
                "type": "multi",
                "positions": [[i * 10.0, i * 10.0], [(i + 1) * 10.0, (i + 1) * 10.0]],
                "orientations": [i * 30.0, (i + 1) * 30.0],
                "speeds": [0.5 + i * 0.1, 0.5 + i * 0.1],
                "max_speeds": [1.0 + i * 0.2, 1.0 + i * 0.2]
            })
            batch_configs.append(config)
        
        # Execute batch processing
        batch_results = []
        for i, config in enumerate(batch_configs):
            navigator = create_test_navigator_from_config(config)
            
            # Run a short simulation
            trajectory = []
            for step in range(3):
                trajectory.append(navigator.positions.copy())
                navigator.step(mock_plume)
            
            batch_results.append({
                "experiment_id": i,
                "trajectory": trajectory,
                "final_positions": navigator.positions.copy()
            })
        
        # Verify batch processing worked correctly
        assert len(batch_results) == 5
        
        # Each experiment should have different final positions
        final_positions_list = [r["final_positions"] for r in batch_results]
        for i in range(len(final_positions_list) - 1):
            for j in range(i + 1, len(final_positions_list)):
                # No two experiments should have identical final positions
                assert not np.allclose(final_positions_list[i], final_positions_list[j])


class TestMultiAgentNavigator:
    """Tests for the multi-agent Navigator functionality."""
    
    def test_initialization_with_default_values(self):
        """Test creating a vectorized navigator with default values."""
        # Create a vectorized navigator for 3 agents with default values
        num_agents = 3
        positions = np.zeros((num_agents, 2))
        navigator = Navigator.multi(positions=positions)
        
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
        navigator = Navigator.multi(
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
        positions = np.zeros((3, 2))  # Need to provide positions for multi-agent
        navigator = Navigator.multi(positions=positions, orientations=orientations)
        
        # Expected normalized values (between 0 and 360)
        expected = np.array([270.0, 10.0, 0.0])
        
        # We need to run a step to trigger normalization in the new architecture
        navigator.step(np.zeros((100, 100)))
        
        # Check normalization
        assert_allclose(navigator.orientations, expected)
    
    def test_speed_constraints(self):
        """Test that speeds affect movement proportionally."""
        # Create navigator with different speeds
        speeds = np.array([0.5, 1.0, 1.5])
        positions = np.zeros((3, 2))  # Need to provide positions for multi-agent
        orientations = np.zeros(3)  # All agents moving along positive x-axis
        
        # Create navigator with these parameters
        navigator = Navigator.multi(
            positions=positions,
            orientations=orientations,
            speeds=speeds
        )
        
        # Store initial positions to compare with after movement
        initial_positions = navigator.positions.copy()
        
        # Take a step
        navigator.step(np.zeros((100, 100)))
        
        # Calculate the actual movement distances
        movement_vectors = navigator.positions - initial_positions
        
        # For agents moving along x-axis, we'll check the x-coordinate movement
        movement_x = movement_vectors[:, 0]
        
        # The agent with speed 1.0 should move at the reference speed
        # Agents with speeds 0.5 and 1.5 should move at 0.5x and 1.5x respectively
        # (we're testing relative movement, not constraint enforcement)
        reference_distance = movement_x[1]  # Distance moved by agent with speed 1.0
        
        # Check that the ratios of movement match the ratios of speeds
        assert_allclose(movement_x[0] / reference_distance, 0.5, atol=1e-5)  # First agent moves at 0.5x speed
        assert_allclose(movement_x[2] / reference_distance, 1.5, atol=1e-5)  # Third agent moves at 1.5x speed
    
    def test_read_single_antenna_odor_from_array(self):
        """Test reading odor values from an array environment."""
        # Create a navigator with agents both in and out of the non-zero odor region
        positions = np.array([
            [45, 45],   # Inside the non-zero region
            [20, 20],   # Outside the non-zero region
            [75, 75]    # Outside the non-zero region
        ])
        
        navigator = Navigator.multi(positions=positions)
        
        # Sample odor at the current positions
        odor_readings = navigator.read_single_antenna_odor(mock_plume)
        
        # Expected readings: 1.0 for agent in non-zero region, 0.0 for others
        expected = np.array([1.0, 0.0, 0.0])
        
        # Check odor readings match expected
        assert_allclose(odor_readings, expected)
    
    def test_read_single_antenna_odor_out_of_bounds(self):
        """Test reading odor values when positions are outside environment bounds."""
        # Create a navigator with agents positioned outside the environment bounds
        positions = np.array([
            [-10, -10],     # Negative coordinates (out of bounds)
            [50, 50],       # Inside bounds
            [200, 200]      # Beyond environment bounds
        ])
        
        navigator = Navigator.multi(positions=positions)
        
        # Sample odor at the current positions (out-of-bounds should return 0.0)
        odor_readings = navigator.read_single_antenna_odor(mock_plume)
        
        # Expected readings: 0.0 for out-of-bounds agents, 1.0 for in-bounds agent in non-zero region
        expected = np.array([0.0, 1.0, 0.0])
        
        # Check odor readings match expected
        assert_allclose(odor_readings, expected)
    
    def test_config_validation_with_new_patterns(self):
        """Test handling of different input configurations with new instantiation patterns."""
        # Test that the Navigator handles single-agent configuration correctly
        single_nav = Navigator.single(
            position=(1.0, 2.0),
            orientation=45.0,
            speed=0.5
        )
        assert_allclose(single_nav.positions[0], [1.0, 2.0])
        assert_allclose(single_nav.orientations[0], 45.0)
        assert_allclose(single_nav.speeds[0], 0.5)
        
        # Test that the Navigator handles multi-agent configuration correctly
        multi_nav = Navigator.multi(
            positions=np.array([[0.0, 0.0], [1.0, 1.0]]),
            orientations=np.array([0.0, 90.0]),
            speeds=np.array([0.5, 1.0])
        )
        assert multi_nav.positions.shape == (2, 2)
        assert multi_nav.orientations.shape == (2,)
        assert multi_nav.speeds.shape == (2,)
        
        # Test default value handling - if orientations not provided, should default to zeros
        positions_only_nav = Navigator.multi(
            positions=np.array([[0.0, 0.0], [1.0, 1.0]])
        )
        assert_allclose(positions_only_nav.orientations, np.zeros(2))
        
        # Test handling of different array length configurations
        # If speeds are provided for multi-agent, they should match positions length
        positions = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        speeds = np.array([0.5, 1.0, 1.5])
        matched_nav = Navigator.multi(
            positions=positions,
            speeds=speeds
        )
        assert matched_nav.positions.shape[0] == matched_nav.speeds.shape[0]
        assert_allclose(matched_nav.speeds, speeds)
    
    def test_hydra_config_integration_examples(self):
        """Test real-world examples of Hydra configuration integration."""
        # Example 1: Single agent with Hydra DictConfig
        hydra_single_config = OmegaConf.create({
            "position": [7.0, 8.0],
            "orientation": 90.0,
            "speed": 0.8,
            "max_speed": 2.0
        })
        
        hydra_single_nav = Navigator.single(
            position=tuple(hydra_single_config.position),
            orientation=hydra_single_config.orientation,
            speed=hydra_single_config.speed,
            max_speed=hydra_single_config.max_speed
        )
        
        assert_allclose(hydra_single_nav.positions[0], [7.0, 8.0])
        assert_allclose(hydra_single_nav.orientations[0], 90.0)
        assert_allclose(hydra_single_nav.speeds[0], 0.8)
        assert_allclose(hydra_single_nav.max_speeds[0], 2.0)
        
        # Example 2: Multi-agent with Hydra DictConfig
        hydra_multi_config = OmegaConf.create({
            "positions": [[1.0, 1.0], [2.0, 2.0]],
            "orientations": [45.0, 135.0],
            "speeds": [0.3, 0.6],
            "max_speeds": [1.5, 2.5]
        })
        
        hydra_multi_nav = Navigator.multi(
            positions=np.array(hydra_multi_config.positions),
            orientations=np.array(hydra_multi_config.orientations),
            speeds=np.array(hydra_multi_config.speeds),
            max_speeds=np.array(hydra_multi_config.max_speeds)
        )
        
        assert_allclose(hydra_multi_nav.positions, np.array([[1.0, 1.0], [2.0, 2.0]]))
        assert_allclose(hydra_multi_nav.orientations, np.array([45.0, 135.0]))
        assert_allclose(hydra_multi_nav.speeds, np.array([0.3, 0.6]))
        assert_allclose(hydra_multi_nav.max_speeds, np.array([1.5, 2.5]))
        
        # Example 3: Configuration composition with OmegaConf.merge
        base_config = OmegaConf.create({
            "positions": [[0.0, 0.0], [5.0, 5.0]],
            "orientations": [0.0, 90.0],
            "speeds": [0.2, 0.2]
        })
        
        override_config = OmegaConf.create({
            "speeds": [0.8, 1.2],
            "max_speeds": [2.0, 3.0]
        })
        
        merged_config = OmegaConf.merge(base_config, override_config)
        
        composed_nav = Navigator.multi(
            positions=np.array(merged_config.positions),
            orientations=np.array(merged_config.orientations),
            speeds=np.array(merged_config.speeds),
            max_speeds=np.array(merged_config.max_speeds)
        )
        
        # Verify merge worked correctly
        assert_allclose(composed_nav.speeds, [0.8, 1.2])  # Override applied
        assert_allclose(composed_nav.max_speeds, [2.0, 3.0])  # Override applied
        assert_allclose(composed_nav.positions, [[0.0, 0.0], [5.0, 5.0]])  # Base preserved
        assert_allclose(composed_nav.orientations, [0.0, 90.0])  # Base preserved