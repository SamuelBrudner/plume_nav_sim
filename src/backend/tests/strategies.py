"""
Hypothesis strategies for property-based testing.

This module provides reusable strategies for generating test data
that conforms to the system's type contracts.

Contract References:
- src/backend/contracts/core_types.md
- src/backend/contracts/*_interface.md
"""

from typing import Any, Dict

import numpy as np
from hypothesis import strategies as st

from plume_nav_sim.core.geometry import Coordinates, GridSize
from plume_nav_sim.core.state import AgentState

# ==============================================================================
# Core Type Strategies
# ==============================================================================


@st.composite
def coordinates_strategy(
    draw,
    min_x: int = 0,
    max_x: int = 127,
    min_y: int = 0,
    max_y: int = 127,
) -> Coordinates:
    """Generate random Coordinates within specified bounds.

    Args:
        draw: Hypothesis draw function
        min_x: Minimum x coordinate (default: 0)
        max_x: Maximum x coordinate (default: 127)
        min_y: Minimum y coordinate (default: 0)
        max_y: Maximum y coordinate (default: 127)

    Returns:
        Random Coordinates instance

    Contract: core_types.md - Coordinates accepts any integers
    """
    x = draw(st.integers(min_value=min_x, max_value=max_x))
    y = draw(st.integers(min_value=min_y, max_value=max_y))
    return Coordinates(x=x, y=y)


@st.composite
def grid_size_strategy(
    draw,
    min_width: int = 1,
    max_width: int = 256,
    min_height: int = 1,
    max_height: int = 256,
) -> GridSize:
    """Generate random GridSize within specified bounds.

    Args:
        draw: Hypothesis draw function
        min_width: Minimum grid width (default: 1)
        max_width: Maximum grid width (default: 256)
        min_height: Minimum grid height (default: 1)
        max_height: Maximum grid height (default: 256)

    Returns:
        Random GridSize instance

    Contract: core_types.md - GridSize requires positive dimensions
    """
    width = draw(st.integers(min_value=min_width, max_value=max_width))
    height = draw(st.integers(min_value=min_height, max_value=max_height))
    return GridSize(width=width, height=height)


@st.composite
def agent_state_strategy(
    draw,
    grid: GridSize | None = None,
    min_step_count: int = 0,
    max_step_count: int = 100,
) -> AgentState:
    """Generate random AgentState.

    Args:
        draw: Hypothesis draw function
        grid: Optional GridSize to constrain position (default: 128x128)
        min_step_count: Minimum step count (default: 0)
        max_step_count: Maximum step count (default: 100)

    Returns:
        Random AgentState instance

    Contract: core_types.md - AgentState specification
    """
    if grid is None:
        grid = GridSize(width=128, height=128)

    # Generate position within grid
    position = draw(
        coordinates_strategy(
            min_x=0,
            max_x=grid.width - 1,
            min_y=0,
            max_y=grid.height - 1,
        )
    )

    # Generate orientation [0, 360)
    orientation = draw(st.floats(min_value=0.0, max_value=359.999))

    # Generate other fields
    step_count = draw(st.integers(min_value=min_step_count, max_value=max_step_count))
    total_reward = draw(st.floats(min_value=0.0, max_value=1000.0))
    goal_reached = draw(st.booleans())

    return AgentState(
        position=position,
        orientation=orientation,
        step_count=step_count,
        total_reward=total_reward,
        goal_reached=goal_reached,
    )


@st.composite
def valid_position_for_grid_strategy(draw, grid: GridSize) -> Coordinates:
    """Generate random position that is guaranteed within grid bounds.

    Args:
        draw: Hypothesis draw function
        grid: GridSize defining bounds

    Returns:
        Coordinates within grid bounds
    """
    x = draw(st.integers(min_value=0, max_value=grid.width - 1))
    y = draw(st.integers(min_value=0, max_value=grid.height - 1))
    return Coordinates(x=x, y=y)


# ==============================================================================
# Action Strategies
# ==============================================================================


@st.composite
def discrete_action_strategy(draw, n_actions: int = 4) -> int:
    """Generate random discrete action.

    Args:
        draw: Hypothesis draw function
        n_actions: Number of actions in space (default: 4 for cardinal directions)

    Returns:
        Random action integer in [0, n_actions)
    """
    return draw(st.integers(min_value=0, max_value=n_actions - 1))


@st.composite
def continuous_action_strategy(
    draw,
    dims: int = 2,
    min_value: float = -1.0,
    max_value: float = 1.0,
) -> np.ndarray:
    """Generate random continuous action vector.

    Args:
        draw: Hypothesis draw function
        dims: Action dimensionality (default: 2 for x,y velocity)
        min_value: Minimum action value (default: -1.0)
        max_value: Maximum action value (default: 1.0)

    Returns:
        Random action as numpy array
    """
    actions = [
        draw(st.floats(min_value=min_value, max_value=max_value)) for _ in range(dims)
    ]
    return np.array(actions, dtype=np.float32)


# ==============================================================================
# Environment State Strategies
# ==============================================================================


@st.composite
def env_state_strategy(
    draw,
    grid: GridSize | None = None,
    include_plume_field: bool = True,
) -> Dict[str, Any]:
    """Generate random environment state dictionary.

    Args:
        draw: Hypothesis draw function
        grid: Optional GridSize (default: 32x32 for reasonable test performance)
        include_plume_field: Whether to include plume field (default: True)

    Returns:
        Environment state dictionary with required keys

    Contract: observation_model_interface.md - env_state pattern

    Note:
        Uses smaller default grid (32x32) to keep Hypothesis examples reasonable.
    """
    if grid is None:
        # Use smaller grid for hypothesis tests
        grid = GridSize(width=32, height=32)

    agent_state = draw(agent_state_strategy(grid=grid))
    time_step = draw(st.integers(min_value=0, max_value=1000))

    env_state: Dict[str, Any] = {
        "agent_state": agent_state,
        "grid_size": grid,
        "time_step": time_step,
    }

    if include_plume_field:
        # Generate plume field directly as numpy array (more efficient)
        # Use just_() for small grids to generate faster
        if grid.total_cells() <= 1024:  # 32x32
            plume_array = np.random.rand(grid.height, grid.width).astype(np.float32)
            env_state["plume_field"] = plume_array
        else:
            # For larger grids, use a simpler pattern
            plume_array = np.zeros((grid.height, grid.width), dtype=np.float32)
            env_state["plume_field"] = plume_array

    return env_state


# ==============================================================================
# Plume/Field Strategies
# ==============================================================================


@st.composite
def concentration_field_strategy(
    draw,
    grid: GridSize | None = None,
    min_concentration: float = 0.0,
    max_concentration: float = 1.0,
) -> np.ndarray:
    """Generate random concentration field.

    Args:
        draw: Hypothesis draw function
        grid: Optional GridSize (default: 32x32)
        min_concentration: Minimum concentration (default: 0.0)
        max_concentration: Maximum concentration (default: 1.0)

    Returns:
        Concentration field as 2D numpy array
    """
    if grid is None:
        grid = GridSize(width=32, height=32)

    # Generate random field
    field = np.array(
        [
            draw(st.floats(min_value=min_concentration, max_value=max_concentration))
            for _ in range(grid.total_cells())
        ],
        dtype=np.float32,
    )

    return field.reshape(grid.height, grid.width)


# ==============================================================================
# Orientation Strategies
# ==============================================================================


@st.composite
def orientation_strategy(draw) -> float:
    """Generate random orientation in degrees [0, 360).

    Args:
        draw: Hypothesis draw function

    Returns:
        Random orientation in degrees

    Contract: core_types.md - AgentState orientation convention
    """
    return draw(st.floats(min_value=0.0, max_value=359.999))


@st.composite
def cardinal_orientation_strategy(draw) -> float:
    """Generate random cardinal direction (0, 90, 180, 270).

    Args:
        draw: Hypothesis draw function

    Returns:
        Cardinal orientation (East, North, West, South)
    """
    return draw(st.sampled_from([0.0, 90.0, 180.0, 270.0]))


# ==============================================================================
# Helper Functions
# ==============================================================================


def assume_position_in_grid(position: Coordinates, grid: GridSize) -> bool:
    """Helper for hypothesis.assume() - checks if position is in grid.

    Args:
        position: Coordinates to check
        grid: GridSize defining bounds

    Returns:
        True if position is within grid, False otherwise
    """
    return position.is_within_bounds(grid)


def assume_finite(*values: float) -> bool:
    """Helper for hypothesis.assume() - checks if all values are finite.

    Args:
        *values: Float values to check

    Returns:
        True if all values are finite (not NaN, not inf)
    """
    return all(np.isfinite(v) for v in values)


__all__ = [
    # Core types
    "coordinates_strategy",
    "grid_size_strategy",
    "agent_state_strategy",
    "valid_position_for_grid_strategy",
    # Actions
    "discrete_action_strategy",
    "continuous_action_strategy",
    # Environment
    "env_state_strategy",
    "concentration_field_strategy",
    # Orientation
    "orientation_strategy",
    "cardinal_orientation_strategy",
    # Helpers
    "assume_position_in_grid",
    "assume_finite",
]
