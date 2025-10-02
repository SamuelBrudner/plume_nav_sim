"""
Reward function implementations for plume navigation.

This package provides various reward function strategies:
- SparseGoalReward: Binary reward at goal
- DenseNavigationReward: Distance-based continuous reward
- ConcentrationReward: Gradient following (TODO - future phase)

All implementations satisfy the RewardFunction protocol.

Contract: reward_function_interface.md
"""

from plume_nav_sim.rewards.dense_navigation import DenseNavigationReward
from plume_nav_sim.rewards.sparse_goal import SparseGoalReward

__all__ = [
    "SparseGoalReward",
    "DenseNavigationReward",
]
