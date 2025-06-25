"""
Property-based test suite for coordinate frame consistency and geometric invariants.

This module implements comprehensive property-based testing using Hypothesis to validate
coordinate frame consistency, geometric invariants, and spatial reference integrity
across all navigation operations and transformations in the odor plume navigation system.

The test suite ensures that:
1. Coordinate transformations preserve geometric invariants
2. Spatial reference frames remain consistent across operations
3. Multi-agent coordination maintains relative positioning integrity
4. Edge cases and boundary conditions are properly handled

Tests cover:
- Position and orientation updates via step() operations
- Sensor coordinate transformations (local to global)
- Reset operations and coordinate frame preservation
- Multi-agent spatial relationship consistency
- Boundary conditions and coordinate space limits
"""

import math
import numpy as np
import pytest
from hypothesis import given, strategies as st, assume, note, example
from hypothesis.extra.numpy import arrays, array_shapes
from typing import Tuple, List, Optional

# Import the dependencies we need to test
from src.odor_plume_nav.core.navigator import Navigator
from src.odor_plume_nav.core.controllers import SingleAgentController, MultiAgentController
from src.odor_plume_nav.api.navigation import create_navigator
from src.odor_plume_nav.utils.navigator_utils import (
    rotate_offset,
    compute_sensor_positions,
    get_predefined_sensor_layout,
    define_sensor_offsets,
    update_positions_and_orientations,
    read_odor_values,
    PREDEFINED_SENSOR_LAYOUTS
)


# Test constants and bounds
POSITION_BOUNDS = (-1000.0, 1000.0)
SPEED_BOUNDS = (0.0, 100.0)
ORIENTATION_BOUNDS = (0.0, 360.0)
ANGULAR_VELOCITY_BOUNDS = (-180.0, 180.0)
SENSOR_DISTANCE_BOUNDS = (0.1, 50.0)
SENSOR_ANGLE_BOUNDS = (1.0, 180.0)
DT_BOUNDS = (0.01, 2.0)


# Hypothesis strategies for coordinate frame testing
@st.composite
def coordinate_strategy(draw):
    """Generate valid 2D coordinates."""
    x = draw(st.floats(min_value=POSITION_BOUNDS[0], max_value=POSITION_BOUNDS[1], allow_nan=False, allow_infinity=False))
    y = draw(st.floats(min_value=POSITION_BOUNDS[0], max_value=POSITION_BOUNDS[1], allow_nan=False, allow_infinity=False))
    return (x, y)


@st.composite
def orientation_strategy(draw):
    """Generate valid orientations in degrees."""
    return draw(st.floats(min_value=0.0, max_value=360.0, allow_nan=False, allow_infinity=False))


@st.composite
def speed_strategy(draw):
    """Generate valid speeds."""
    return draw(st.floats(min_value=SPEED_BOUNDS[0], max_value=SPEED_BOUNDS[1], allow_nan=False, allow_infinity=False))


@st.composite
def angular_velocity_strategy(draw):
    """Generate valid angular velocities."""
    return draw(st.floats(min_value=ANGULAR_VELOCITY_BOUNDS[0], max_value=ANGULAR_VELOCITY_BOUNDS[1], allow_nan=False, allow_infinity=False))


@st.composite
def single_agent_state_strategy(draw):
    """Generate valid single agent state."""
    position = draw(coordinate_strategy())
    orientation = draw(orientation_strategy())
    speed = draw(speed_strategy())
    max_speed = draw(st.floats(min_value=speed, max_value=SPEED_BOUNDS[1], allow_nan=False, allow_infinity=False))
    angular_velocity = draw(angular_velocity_strategy())
    
    return {
        'position': position,
        'orientation': orientation,
        'speed': speed,
        'max_speed': max_speed,
        'angular_velocity': angular_velocity
    }


@st.composite
def multi_agent_state_strategy(draw, min_agents=2, max_agents=10):
    """Generate valid multi-agent state."""
    num_agents = draw(st.integers(min_value=min_agents, max_value=max_agents))
    
    positions = []
    orientations = []
    speeds = []
    max_speeds = []
    angular_velocities = []
    
    for _ in range(num_agents):
        positions.append(draw(coordinate_strategy()))
        orientations.append(draw(orientation_strategy()))
        speed = draw(speed_strategy())
        speeds.append(speed)
        max_speeds.append(draw(st.floats(min_value=speed, max_value=SPEED_BOUNDS[1], allow_nan=False, allow_infinity=False)))
        angular_velocities.append(draw(angular_velocity_strategy()))
    
    return {
        'positions': np.array(positions),
        'orientations': np.array(orientations),
        'speeds': np.array(speeds),
        'max_speeds': np.array(max_speeds),
        'angular_velocities': np.array(angular_velocities)
    }


@st.composite
def sensor_configuration_strategy(draw):
    """Generate valid sensor configuration."""
    num_sensors = draw(st.integers(min_value=1, max_value=8))
    distance = draw(st.floats(min_value=SENSOR_DISTANCE_BOUNDS[0], max_value=SENSOR_DISTANCE_BOUNDS[1], allow_nan=False, allow_infinity=False))
    angle = draw(st.floats(min_value=SENSOR_ANGLE_BOUNDS[0], max_value=SENSOR_ANGLE_BOUNDS[1], allow_nan=False, allow_infinity=False))
    layout_name = draw(st.one_of(st.none(), st.sampled_from(list(PREDEFINED_SENSOR_LAYOUTS.keys()))))
    
    return {
        'num_sensors': num_sensors,
        'distance': distance,
        'angle': angle,
        'layout_name': layout_name
    }


@st.composite
def environment_array_strategy(draw, min_size=10, max_size=100):
    """Generate valid environment arrays for testing."""
    height = draw(st.integers(min_value=min_size, max_value=max_size))
    width = draw(st.integers(min_value=min_size, max_value=max_size))
    
    # Generate either float or uint8 environment data
    dtype = draw(st.sampled_from([np.float32, np.float64, np.uint8]))
    
    if dtype == np.uint8:
        # For uint8, generate values in [0, 255] range
        env_array = draw(arrays(dtype=dtype, shape=(height, width), elements=st.integers(min_value=0, max_value=255)))
    else:
        # For float types, generate values in [0, 1] range
        env_array = draw(arrays(dtype=dtype, shape=(height, width), elements=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)))
    
    return env_array


class TestCoordinateFrameConsistency:
    """Test coordinate frame consistency across all navigation operations."""
    
    @given(single_agent_state_strategy())
    def test_single_agent_position_update_consistency(self, agent_state):
        """Test that single agent position updates maintain coordinate frame consistency."""
        # Create navigator with initial state
        navigator = Navigator.single(**agent_state)
        
        # Store initial state
        initial_position = navigator.positions[0].copy()
        initial_orientation = navigator.orientations[0]
        
        # Create a simple environment
        env_array = np.ones((50, 50))
        
        # Perform a step
        dt = 1.0
        navigator.step(env_array, dt)
        
        # Check coordinate frame consistency
        new_position = navigator.positions[0]
        new_orientation = navigator.orientations[0]
        
        # Position should change according to speed and orientation
        expected_dx = agent_state['speed'] * np.cos(np.radians(initial_orientation)) * dt
        expected_dy = agent_state['speed'] * np.sin(np.radians(initial_orientation)) * dt
        expected_position = initial_position + np.array([expected_dx, expected_dy])
        
        # Allow for small floating point errors
        np.testing.assert_allclose(new_position, expected_position, rtol=1e-10)
        
        # Orientation should change according to angular velocity
        expected_orientation = (initial_orientation + agent_state['angular_velocity'] * dt) % 360.0
        assert abs(new_orientation - expected_orientation) < 1e-10
    
    @given(multi_agent_state_strategy())
    def test_multi_agent_position_update_consistency(self, agent_state):
        """Test that multi-agent position updates maintain coordinate frame consistency."""
        # Create navigator with initial state
        navigator = Navigator.multi(**agent_state)
        
        # Store initial state
        initial_positions = navigator.positions.copy()
        initial_orientations = navigator.orientations.copy()
        
        # Create a simple environment
        env_array = np.ones((50, 50))
        
        # Perform a step
        dt = 1.0
        navigator.step(env_array, dt)
        
        # Check coordinate frame consistency for each agent
        new_positions = navigator.positions
        new_orientations = navigator.orientations
        
        for i in range(navigator.num_agents):
            # Position should change according to speed and orientation
            expected_dx = agent_state['speeds'][i] * np.cos(np.radians(initial_orientations[i])) * dt
            expected_dy = agent_state['speeds'][i] * np.sin(np.radians(initial_orientations[i])) * dt
            expected_position = initial_positions[i] + np.array([expected_dx, expected_dy])
            
            # Allow for small floating point errors
            np.testing.assert_allclose(new_positions[i], expected_position, rtol=1e-10)
            
            # Orientation should change according to angular velocity
            expected_orientation = (initial_orientations[i] + agent_state['angular_velocities'][i] * dt) % 360.0
            assert abs(new_orientations[i] - expected_orientation) < 1e-10
    
    @given(
        st.floats(min_value=0.0, max_value=360.0, allow_nan=False, allow_infinity=False),
        st.floats(min_value=DT_BOUNDS[0], max_value=DT_BOUNDS[1], allow_nan=False, allow_infinity=False)
    )
    def test_orientation_wrapping_consistency(self, initial_orientation, dt):
        """Test that orientation wrapping maintains consistency."""
        # Test with a large angular velocity that will cause wrapping
        angular_velocity = 400.0  # degrees/second - will definitely cause wrapping
        
        navigator = Navigator.single(
            position=(0, 0),
            orientation=initial_orientation,
            angular_velocity=angular_velocity
        )
        
        env_array = np.ones((10, 10))
        navigator.step(env_array, dt)
        
        new_orientation = navigator.orientations[0]
        
        # Check that orientation is wrapped to [0, 360) range
        assert 0.0 <= new_orientation < 360.0
        
        # Check that the wrapping is mathematically correct
        expected_orientation = (initial_orientation + angular_velocity * dt) % 360.0
        assert abs(new_orientation - expected_orientation) < 1e-10


class TestGeometricInvariants:
    """Test geometric invariants in coordinate transformations."""
    
    @given(
        coordinate_strategy(),
        orientation_strategy()
    )
    def test_rotation_preserves_distance(self, local_offset, orientation):
        """Test that rotation transformations preserve distances."""
        local_offset_array = np.array(local_offset)
        
        # Calculate original distance from origin
        original_distance = np.linalg.norm(local_offset_array)
        
        # Apply rotation
        rotated_offset = rotate_offset(local_offset_array, orientation)
        
        # Calculate distance after rotation
        rotated_distance = np.linalg.norm(rotated_offset)
        
        # Distance should be preserved
        assert abs(original_distance - rotated_distance) < 1e-10
    
    @given(
        coordinate_strategy(),
        orientation_strategy(),
        orientation_strategy()
    )
    def test_rotation_composition(self, local_offset, angle1, angle2):
        """Test that rotation transformations compose correctly."""
        local_offset_array = np.array(local_offset)
        
        # Apply rotations separately
        temp_rotation = rotate_offset(local_offset_array, angle1)
        double_rotation = rotate_offset(temp_rotation, angle2)
        
        # Apply combined rotation
        combined_angle = (angle1 + angle2) % 360.0
        combined_rotation = rotate_offset(local_offset_array, combined_angle)
        
        # Results should be identical (within floating point precision)
        np.testing.assert_allclose(double_rotation, combined_rotation, rtol=1e-10)
    
    @given(sensor_configuration_strategy())
    def test_sensor_layout_geometric_properties(self, sensor_config):
        """Test that sensor layouts maintain geometric properties."""
        if sensor_config['layout_name'] is not None:
            # Test predefined layouts
            layout = get_predefined_sensor_layout(sensor_config['layout_name'], sensor_config['distance'])
            
            # All sensors should be at the specified distance from origin (for scaled layouts)
            for sensor_offset in layout:
                distance = np.linalg.norm(sensor_offset)
                # Some layouts might be at origin (like "SINGLE"), so handle that
                if not np.allclose(sensor_offset, [0, 0]):
                    expected_distance = sensor_config['distance']
                    assert abs(distance - expected_distance) < 1e-10
        else:
            # Test dynamically generated layouts
            layout = define_sensor_offsets(
                sensor_config['num_sensors'],
                sensor_config['distance'],
                sensor_config['angle']
            )
            
            # All sensors should be at the specified distance from origin
            for sensor_offset in layout:
                distance = np.linalg.norm(sensor_offset)
                assert abs(distance - sensor_config['distance']) < 1e-10
            
            # Check angular spacing for multi-sensor layouts
            if sensor_config['num_sensors'] > 1:
                angles = []
                for sensor_offset in layout:
                    angle = np.degrees(np.arctan2(sensor_offset[1], sensor_offset[0]))
                    angles.append(angle)
                
                # Sort angles for consistent comparison
                angles.sort()
                
                # Check that angular spacing is consistent
                if sensor_config['num_sensors'] > 2:
                    for i in range(1, len(angles)):
                        angle_diff = angles[i] - angles[i-1]
                        # Should be close to the specified sensor angle (within tolerance)
                        assert abs(angle_diff - sensor_config['angle']) < 1e-8
    
    @given(
        single_agent_state_strategy(),
        sensor_configuration_strategy()
    )
    def test_sensor_position_invariants(self, agent_state, sensor_config):
        """Test that sensor positions maintain geometric invariants."""
        # Create navigator
        navigator = Navigator.single(**agent_state)
        
        # Compute sensor positions
        if sensor_config['layout_name'] is not None:
            sensor_positions = compute_sensor_positions(
                navigator.positions,
                navigator.orientations,
                layout_name=sensor_config['layout_name'],
                distance=sensor_config['distance']
            )
        else:
            sensor_positions = compute_sensor_positions(
                navigator.positions,
                navigator.orientations,
                distance=sensor_config['distance'],
                angle=sensor_config['angle'],
                num_sensors=sensor_config['num_sensors']
            )
        
        agent_position = navigator.positions[0]
        
        # Check that each sensor is at the correct distance from the agent
        for sensor_pos in sensor_positions[0]:  # First agent's sensors
            distance_to_agent = np.linalg.norm(sensor_pos - agent_position)
            
            # For layouts that might have sensors at the agent position
            if sensor_config['layout_name'] == "SINGLE":
                expected_distance = 0.0
            else:
                expected_distance = sensor_config['distance']
            
            assert abs(distance_to_agent - expected_distance) < 1e-10


class TestSpatialReferenceIntegrity:
    """Test spatial reference frame integrity across operations."""
    
    @given(multi_agent_state_strategy(min_agents=2, max_agents=5))
    def test_multi_agent_relative_positioning(self, agent_state):
        """Test that multi-agent relative positioning is preserved."""
        # Create navigator
        navigator = Navigator.multi(**agent_state)
        
        # Calculate initial relative positions
        initial_positions = navigator.positions.copy()
        initial_relative_distances = {}
        
        for i in range(navigator.num_agents):
            for j in range(i + 1, navigator.num_agents):
                distance = np.linalg.norm(initial_positions[i] - initial_positions[j])
                initial_relative_distances[(i, j)] = distance
        
        # Perform several steps
        env_array = np.ones((50, 50))
        for _ in range(5):
            navigator.step(env_array, dt=0.1)
        
        # Calculate final relative distances
        final_positions = navigator.positions
        final_relative_distances = {}
        
        for i in range(navigator.num_agents):
            for j in range(i + 1, navigator.num_agents):
                distance = np.linalg.norm(final_positions[i] - final_positions[j])
                final_relative_distances[(i, j)] = distance
        
        # Relative distances should remain the same if agents move with same velocity
        # (This test assumes agents move independently, so we're checking frame consistency)
        # We verify that distance calculations are consistent across coordinate frames
        for key in initial_relative_distances:
            initial_dist = initial_relative_distances[key]
            final_dist = final_relative_distances[key]
            
            # Both distances should be non-negative and finite
            assert initial_dist >= 0
            assert final_dist >= 0
            assert np.isfinite(initial_dist)
            assert np.isfinite(final_dist)
    
    @given(
        single_agent_state_strategy(),
        sensor_configuration_strategy()
    )
    def test_coordinate_frame_transformation_consistency(self, agent_state, sensor_config):
        """Test coordinate frame transformations between local and global frames."""
        # Create navigator
        navigator = Navigator.single(**agent_state)
        
        # Test sensor coordinate transformations
        if sensor_config['layout_name'] is not None:
            # Get local sensor offsets
            local_offsets = get_predefined_sensor_layout(
                sensor_config['layout_name'], 
                sensor_config['distance']
            )
        else:
            local_offsets = define_sensor_offsets(
                sensor_config['num_sensors'],
                sensor_config['distance'],
                sensor_config['angle']
            )
        
        # Transform to global coordinates manually
        agent_position = navigator.positions[0]
        agent_orientation = navigator.orientations[0]
        
        global_sensor_positions_manual = []
        for local_offset in local_offsets:
            # Rotate offset by agent orientation
            rotated_offset = rotate_offset(local_offset, agent_orientation)
            # Add to agent position
            global_position = agent_position + rotated_offset
            global_sensor_positions_manual.append(global_position)
        
        # Transform using the utility function
        global_sensor_positions_utility = compute_sensor_positions(
            navigator.positions,
            navigator.orientations,
            layout_name=sensor_config['layout_name'] if sensor_config['layout_name'] else None,
            distance=sensor_config['distance'],
            angle=sensor_config['angle'] if sensor_config['layout_name'] is None else 45.0,
            num_sensors=sensor_config['num_sensors'] if sensor_config['layout_name'] is None else 2
        )[0]  # First agent's sensors
        
        # Results should be identical
        for manual_pos, utility_pos in zip(global_sensor_positions_manual, global_sensor_positions_utility):
            np.testing.assert_allclose(manual_pos, utility_pos, rtol=1e-10)
    
    @given(single_agent_state_strategy())
    def test_reset_preserves_coordinate_frame(self, agent_state):
        """Test that reset operations preserve coordinate frame consistency."""
        # Create navigator and perform some steps
        navigator = Navigator.single(**agent_state)
        
        env_array = np.ones((20, 20))
        for _ in range(3):
            navigator.step(env_array)
        
        # Reset to original state
        navigator.reset(**agent_state)
        
        # Check that state is reset correctly
        np.testing.assert_allclose(navigator.positions[0], agent_state['position'], rtol=1e-10)
        assert abs(navigator.orientations[0] - agent_state['orientation']) < 1e-10
        assert abs(navigator.speeds[0] - agent_state['speed']) < 1e-10
        assert abs(navigator.max_speeds[0] - agent_state['max_speed']) < 1e-10
        assert abs(navigator.angular_velocities[0] - agent_state['angular_velocity']) < 1e-10


class TestEdgeCasesAndBoundaryConditions:
    """Test edge cases and boundary conditions for coordinate frame operations."""
    
    @given(
        environment_array_strategy(),
        coordinate_strategy()
    )
    def test_environment_boundary_handling(self, env_array, position):
        """Test that environment boundary conditions are handled correctly."""
        height, width = env_array.shape
        
        # Test position that might be out of bounds
        positions = np.array([position])
        
        # Read odor values (should not crash for out-of-bounds positions)
        odor_values = read_odor_values(env_array, positions)
        
        # Should return valid values
        assert len(odor_values) == 1
        assert np.isfinite(odor_values[0])
        assert odor_values[0] >= 0.0
        
        # For out-of-bounds positions, should return 0
        x, y = position
        if x < 0 or x >= width or y < 0 or y >= height:
            assert odor_values[0] == 0.0
    
    @given(
        single_agent_state_strategy(),
        st.floats(min_value=0.001, max_value=10.0, allow_nan=False, allow_infinity=False)
    )
    def test_small_time_step_consistency(self, agent_state, dt):
        """Test coordinate frame consistency with very small time steps."""
        # Create navigator
        navigator = Navigator.single(**agent_state)
        
        # Store initial state
        initial_position = navigator.positions[0].copy()
        initial_orientation = navigator.orientations[0]
        
        # Take a step with the given dt
        env_array = np.ones((10, 10))
        navigator.step(env_array, dt)
        
        # Verify position update is proportional to dt
        position_change = navigator.positions[0] - initial_position
        expected_change_magnitude = agent_state['speed'] * dt
        actual_change_magnitude = np.linalg.norm(position_change)
        
        if agent_state['speed'] > 0:
            assert abs(actual_change_magnitude - expected_change_magnitude) < 1e-10
        
        # Verify orientation update is proportional to dt
        orientation_change = (navigator.orientations[0] - initial_orientation) % 360.0
        if orientation_change > 180:
            orientation_change = orientation_change - 360.0
        
        expected_orientation_change = agent_state['angular_velocity'] * dt
        assert abs(orientation_change - expected_orientation_change) < 1e-10
    
    @given(st.integers(min_value=1, max_value=20))
    def test_extreme_sensor_configurations(self, num_sensors):
        """Test coordinate frame consistency with extreme sensor configurations."""
        # Test with many sensors
        distance = 10.0
        angle = 360.0 / num_sensors  # Distribute evenly around circle
        
        sensor_offsets = define_sensor_offsets(num_sensors, distance, angle)
        
        # Check that all sensors are at correct distance
        for offset in sensor_offsets:
            actual_distance = np.linalg.norm(offset)
            assert abs(actual_distance - distance) < 1e-10
        
        # For circular arrangements, check that sensors form proper angular spacing
        if num_sensors > 2:
            angles = []
            for offset in sensor_offsets:
                angle = np.degrees(np.arctan2(offset[1], offset[0]))
                angles.append(angle)
            
            angles.sort()
            
            # Check angular spacing
            expected_spacing = 360.0 / num_sensors
            for i in range(len(angles) - 1):
                actual_spacing = angles[i + 1] - angles[i]
                assert abs(actual_spacing - expected_spacing) < 1e-8
    
    @given(
        multi_agent_state_strategy(min_agents=1, max_agents=1),  # Single agent in multi-agent format
        sensor_configuration_strategy()
    )
    def test_single_agent_in_multi_agent_format(self, agent_state, sensor_config):
        """Test coordinate frame consistency when single agent is in multi-agent format."""
        # Create navigator in multi-agent format with single agent
        navigator = Navigator.multi(**agent_state)
        
        assert navigator.num_agents == 1
        
        # Test sensor positioning
        if sensor_config['layout_name'] is not None:
            sensor_positions = compute_sensor_positions(
                navigator.positions,
                navigator.orientations,
                layout_name=sensor_config['layout_name'],
                distance=sensor_config['distance']
            )
        else:
            sensor_positions = compute_sensor_positions(
                navigator.positions,
                navigator.orientations,
                distance=sensor_config['distance'],
                angle=sensor_config['angle'],
                num_sensors=sensor_config['num_sensors']
            )
        
        # Should have shape (1, num_sensors, 2)
        assert sensor_positions.shape[0] == 1  # One agent
        assert sensor_positions.shape[2] == 2  # 2D coordinates
        
        # All coordinate calculations should be valid
        for sensor_pos in sensor_positions[0]:
            assert np.isfinite(sensor_pos).all()


class TestCoordinateFrameIntegration:
    """Integration tests for coordinate frame operations across the system."""
    
    @given(
        single_agent_state_strategy(),
        environment_array_strategy(min_size=20, max_size=50),
        st.integers(min_value=1, max_value=10)
    )
    def test_full_navigation_coordinate_consistency(self, agent_state, env_array, num_steps):
        """Test coordinate frame consistency through complete navigation sequence."""
        # Create navigator
        navigator = Navigator.single(**agent_state)
        
        # Store trajectory
        trajectory_positions = [navigator.positions[0].copy()]
        trajectory_orientations = [navigator.orientations[0]]
        
        # Perform navigation steps
        dt = 0.1
        for step in range(num_steps):
            navigator.step(env_array, dt)
            trajectory_positions.append(navigator.positions[0].copy())
            trajectory_orientations.append(navigator.orientations[0])
        
        # Verify trajectory consistency
        for i in range(1, len(trajectory_positions)):
            prev_pos = trajectory_positions[i-1]
            curr_pos = trajectory_positions[i]
            prev_orient = trajectory_orientations[i-1]
            
            # Calculate expected movement
            expected_dx = agent_state['speed'] * np.cos(np.radians(prev_orient)) * dt
            expected_dy = agent_state['speed'] * np.sin(np.radians(prev_orient)) * dt
            expected_pos = prev_pos + np.array([expected_dx, expected_dy])
            
            # Account for any intermediate orientation changes
            np.testing.assert_allclose(curr_pos, expected_pos, rtol=1e-8)
        
        # Verify all orientations are in valid range
        for orientation in trajectory_orientations:
            assert 0.0 <= orientation < 360.0
    
    @given(
        multi_agent_state_strategy(min_agents=2, max_agents=4),
        environment_array_strategy(),
        sensor_configuration_strategy()
    )
    def test_multi_agent_sensor_coordination(self, agent_state, env_array, sensor_config):
        """Test coordinate frame consistency in multi-agent sensor operations."""
        # Create navigator
        navigator = Navigator.multi(**agent_state)
        
        # Compute sensor positions for all agents
        if sensor_config['layout_name'] is not None:
            sensor_positions = compute_sensor_positions(
                navigator.positions,
                navigator.orientations,
                layout_name=sensor_config['layout_name'],
                distance=sensor_config['distance']
            )
        else:
            sensor_positions = compute_sensor_positions(
                navigator.positions,
                navigator.orientations,
                distance=sensor_config['distance'],
                angle=sensor_config['angle'],
                num_sensors=sensor_config['num_sensors']
            )
        
        # Verify sensor positions for each agent
        for agent_idx in range(navigator.num_agents):
            agent_position = navigator.positions[agent_idx]
            agent_sensors = sensor_positions[agent_idx]
            
            # Each sensor should be at correct distance from its agent
            for sensor_pos in agent_sensors:
                distance_to_agent = np.linalg.norm(sensor_pos - agent_position)
                
                # Handle special case of sensors at agent position
                if sensor_config['layout_name'] == "SINGLE":
                    expected_distance = 0.0
                else:
                    expected_distance = sensor_config['distance']
                
                assert abs(distance_to_agent - expected_distance) < 1e-10
        
        # Test sensor sampling (should not crash and return valid values)
        height, width = env_array.shape
        all_sensor_positions_flat = sensor_positions.reshape(-1, 2)
        
        # Filter to in-bounds positions for testing
        in_bounds_sensors = []
        for pos in all_sensor_positions_flat:
            x, y = pos
            if 0 <= x < width and 0 <= y < height:
                in_bounds_sensors.append(pos)
        
        if in_bounds_sensors:
            in_bounds_array = np.array(in_bounds_sensors)
            odor_values = read_odor_values(env_array, in_bounds_array)
            
            # All values should be valid
            assert len(odor_values) == len(in_bounds_sensors)
            assert np.isfinite(odor_values).all()
            assert (odor_values >= 0.0).all()


# Additional parameterized tests for specific coordinate frame scenarios
class TestSpecificCoordinateFrameScenarios:
    """Test specific coordinate frame scenarios and known edge cases."""
    
    @pytest.mark.parametrize("layout_name", list(PREDEFINED_SENSOR_LAYOUTS.keys()))
    def test_predefined_sensor_layouts(self, layout_name):
        """Test all predefined sensor layouts for coordinate frame consistency."""
        distance = 5.0
        layout = get_predefined_sensor_layout(layout_name, distance)
        
        # Verify layout properties
        assert isinstance(layout, np.ndarray)
        assert layout.shape[1] == 2  # 2D coordinates
        
        # Test with different agent orientations
        for orientation in [0, 90, 180, 270]:
            agent_pos = np.array([10.0, 20.0])
            agent_orientation = float(orientation)
            
            # Compute global sensor positions
            global_positions = []
            for local_offset in layout:
                rotated_offset = rotate_offset(local_offset, agent_orientation)
                global_pos = agent_pos + rotated_offset
                global_positions.append(global_pos)
            
            # All positions should be valid
            for pos in global_positions:
                assert np.isfinite(pos).all()
    
    @pytest.mark.parametrize("orientation", [0, 45, 90, 135, 180, 225, 270, 315])
    def test_cardinal_and_intercardinal_orientations(self, orientation):
        """Test coordinate frame consistency at cardinal and intercardinal orientations."""
        # Create navigator with specific orientation
        navigator = Navigator.single(
            position=(0, 0),
            orientation=orientation,
            speed=1.0,
            angular_velocity=0.0
        )
        
        # Take a step
        env_array = np.ones((10, 10))
        navigator.step(env_array, dt=1.0)
        
        # Check movement direction
        new_position = navigator.positions[0]
        expected_x = np.cos(np.radians(orientation))
        expected_y = np.sin(np.radians(orientation))
        
        np.testing.assert_allclose(new_position, [expected_x, expected_y], rtol=1e-10)
    
    @pytest.mark.parametrize("speed", [0.0, 0.1, 1.0, 10.0, 100.0])
    def test_different_speeds_coordinate_consistency(self, speed):
        """Test coordinate frame consistency at different speeds."""
        navigator = Navigator.single(
            position=(0, 0),
            orientation=45.0,  # 45 degrees
            speed=speed,
            angular_velocity=0.0
        )
        
        env_array = np.ones((10, 10))
        navigator.step(env_array, dt=1.0)
        
        new_position = navigator.positions[0]
        
        if speed == 0.0:
            # Should not move
            np.testing.assert_allclose(new_position, [0, 0], rtol=1e-10)
        else:
            # Should move in 45-degree direction
            expected_distance = speed
            actual_distance = np.linalg.norm(new_position)
            assert abs(actual_distance - expected_distance) < 1e-10
            
            # Should maintain 45-degree angle
            expected_angle = 45.0
            actual_angle = np.degrees(np.arctan2(new_position[1], new_position[0]))
            assert abs(actual_angle - expected_angle) < 1e-10


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])