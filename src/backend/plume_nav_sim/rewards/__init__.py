"""
Reward Function Implementations.

This package provides concrete implementations of the RewardFunction protocol
for computing rewards based on agent state and environment conditions.

Available Reward Functions:
    - SparseGoalReward: Binary reward (1.0 at goal, 0.0 elsewhere)
    - StepPenaltyReward: Goal bonus with per-step time penalty

Contract: src/backend/contracts/reward_function_interface.md
"""

from plume_nav_sim.rewards.sparse_goal import SparseGoalReward
from plume_nav_sim.rewards.step_penalty import StepPenaltyReward

__all__ = [
    "SparseGoalReward",
    "StepPenaltyReward",
]
