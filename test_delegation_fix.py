#!/usr/bin/env python3
"""Quick test to verify the delegation test fix."""
import sys

sys.path.insert(0, "src/backend")

import numpy as np

from plume_nav_sim.actions import DiscreteGridActions
from plume_nav_sim.core.geometry import Coordinates, GridSize
from plume_nav_sim.envs import ComponentBasedEnvironment
from plume_nav_sim.observations import ConcentrationSensor
from plume_nav_sim.plume.concentration_field import ConcentrationField
from plume_nav_sim.rewards import SparseGoalReward

# Set up environment
grid_size = GridSize(width=64, height=64)
goal_location = Coordinates(50, 50)

# Create concentration field
field = ConcentrationField(grid_size=grid_size, enable_caching=False)
x = np.arange(grid_size.width)
y = np.arange(grid_size.height)
xx, yy = np.meshgrid(x, y)
sigma = 10.0
dx = xx - goal_location.x
dy = yy - goal_location.y
field_array = np.exp(-(dx**2 + dy**2) / (2 * sigma**2))
field.field_array = field_array.astype(np.float32)
field.is_generated = True

# Create environment
env = ComponentBasedEnvironment(
    action_processor=DiscreteGridActions(step_size=1),
    observation_model=ConcentrationSensor(),
    reward_function=SparseGoalReward(goal_position=goal_location, goal_radius=5.0),
    concentration_field=field,
    grid_size=grid_size,
    max_steps=100,
    goal_location=goal_location,
    goal_radius=5.0,
)

# Test with seed
env.reset(seed=42)
initial_pos = env._agent_state.position
print(f"Initial position: {initial_pos}")

# Choose action based on position
if initial_pos.x >= 62:
    action = 3  # LEFT
    print("Agent near right edge, using LEFT action")
else:
    action = 1  # RIGHT
    print("Using RIGHT action")

# Take step
obs, reward, terminated, truncated, info = env.step(action)
new_pos = env._agent_state.position
print(f"New position: {new_pos}")
print(f"Position changed: {new_pos != initial_pos}")

assert new_pos != initial_pos, f"Position should change! {initial_pos} -> {new_pos}"
print("✅ Test passed!")
