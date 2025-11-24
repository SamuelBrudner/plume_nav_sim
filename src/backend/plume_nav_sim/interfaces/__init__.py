"""
Protocol interfaces for pluggable components in plume navigation simulation.

This module defines the core protocols that enable dependency injection
and component swapping without environment modification.

Contract References:
- reward_function_interface.md
- observation_model_interface.md
- action_processor_interface.md
- component_interfaces.md
"""

from .action import ActionProcessor, ActionType
from .fields import ScalarField, VectorField
from .observation import ObservationModel
from .policy import Policy
from .reward import RewardFunction

__all__ = [
    "RewardFunction",
    "ObservationModel",
    "ActionProcessor",
    "ActionType",
    "Policy",
    "ScalarField",
    "VectorField",
]
