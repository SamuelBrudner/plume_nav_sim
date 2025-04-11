"""Tests for navigator utility functions."""

import pytest
import numpy as np

from odor_plume_nav.utils import (
    normalize_array_parameter,
    create_navigator_from_params,
)


def test_normalize_array_parameter_none():
    """Test that None parameter returns None."""
    result = normalize_array_parameter(None, 3)
    assert result is None


def test_normalize_array_parameter_scalar():
    """Test converting a scalar value to array of desired length."""
    result = _extracted_from_test_normalize_array_parameter_scalar_4(5, 3)
    assert np.array_equal(result, np.array([5, 5, 5]))

    result = _extracted_from_test_normalize_array_parameter_scalar_4(2.5, 4)
    assert np.array_equal(result, np.array([2.5, 2.5, 2.5, 2.5]))


# TODO Rename this here and in `test_normalize_array_parameter_scalar`
def _extracted_from_test_normalize_array_parameter_scalar_4(arg0, arg1):
    # Test with integer
    result = normalize_array_parameter(arg0, arg1)
    assert isinstance(result, np.ndarray)
    assert result.shape == (arg1, )
    return result


def test_normalize_array_parameter_list():
    """Test converting a list to a numpy array."""
    result = normalize_array_parameter([1, 2, 3], 3)
    assert isinstance(result, np.ndarray)
    assert result.shape == (3,)
    assert np.array_equal(result, np.array([1, 2, 3]))


def test_normalize_array_parameter_ndarray():
    """Test that a numpy array is returned unchanged."""
    input_array = np.array([4, 5, 6])
    result = normalize_array_parameter(input_array, 3)
    assert isinstance(result, np.ndarray)
    assert result.shape == (3,)
    assert np.array_equal(result, input_array)
    # Verify it's the same object (unchanged)
    assert result is input_array


def test_create_navigator_from_params_single_agent():
    """Test creating a navigator from parameters for a single agent."""
    # Test with tuple position
    navigator = create_navigator_from_params(
        positions=(5, 10),
        orientations=30,
        speeds=1.5,
        max_speeds=3.0
    )
    
    assert navigator._single_agent is True
    assert navigator.get_position() == (5, 10)
    assert navigator.orientation == 30
    assert navigator.speed == 1.5
    assert navigator.max_speed == 3.0


def test_create_navigator_from_params_multi_agent():
    """Test creating a navigator from parameters for multiple agents."""
    # Test with list of positions
    positions = [(1, 2), (3, 4), (5, 6)]
    orientations = [10, 20, 30]
    speeds = [0.1, 0.2, 0.3]
    max_speeds = [1.0, 2.0, 3.0]
    
    navigator = create_navigator_from_params(
        positions=positions,
        orientations=orientations,
        speeds=speeds,
        max_speeds=max_speeds
    )
    
    assert navigator._single_agent is False
    assert navigator.num_agents == 3
    assert np.allclose(navigator.positions[0], (1, 2))
    assert np.allclose(navigator.positions[1], (3, 4))
    assert np.allclose(navigator.positions[2], (5, 6))
    assert np.array_equal(navigator.orientations, np.array(orientations))
    assert np.array_equal(navigator.speeds, np.array(speeds))
    assert np.array_equal(navigator.max_speeds, np.array(max_speeds))


def test_create_navigator_from_params_mixed_types():
    """Test creating a navigator with mixed parameter types."""
    positions = [(1, 2), (3, 4), (5, 6)]
    
    # Scalar values for multi-agent parameters should be broadcasted
    navigator = create_navigator_from_params(
        positions=positions,
        orientations=45,  # scalar
        speeds=[0.1, 0.2, 0.3],  # list
        max_speeds=np.array([1.0, 2.0, 3.0])  # numpy array
    )
    
    assert navigator._single_agent is False
    assert navigator.num_agents == 3
    assert np.array_equal(navigator.orientations, np.array([45, 45, 45]))
    assert np.array_equal(navigator.speeds, np.array([0.1, 0.2, 0.3]))
    assert np.array_equal(navigator.max_speeds, np.array([1.0, 2.0, 3.0]))
