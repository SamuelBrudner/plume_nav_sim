"""
Action Processor Implementations.

This package provides concrete implementations of the ActionProcessor protocol
for processing agent actions and computing new agent states.

Available Action Processors:
    - DiscreteGridActions: 4-direction cardinal movement
    - OrientedGridActions: 3-action surge/turn control with orientation

Contract: src/backend/contracts/action_processor_interface.md
"""

from plume_nav_sim.actions.discrete_grid import DiscreteGridActions
from plume_nav_sim.actions.oriented_grid import OrientedGridActions

__all__ = [
    "DiscreteGridActions",
    "OrientedGridActions",
]
