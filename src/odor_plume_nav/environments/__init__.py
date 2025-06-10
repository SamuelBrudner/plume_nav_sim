"""
Environments module for odor plume navigation.

This module contains environment classes for simulation, including VideoPlume
and new Gymnasium-compliant environments for reinforcement learning integration.
"""

# Core environment - always available
from odor_plume_nav.environments.video_plume import VideoPlume

# List of all available exports - will be updated based on successful imports
__all__ = [
    "VideoPlume",
]

# Conditional imports for reinforcement learning features
# These require gymnasium and related RL dependencies
try:
    from odor_plume_nav.environments.gymnasium_env import GymnasiumEnv
    __all__.append("GymnasiumEnv")
except ImportError:
    # RL dependencies not available - GymnasiumEnv not available
    GymnasiumEnv = None

try:
    from odor_plume_nav.environments.spaces import (
        create_action_space,
        create_observation_space,
        ActionSpaceConfig,
        ObservationSpaceConfig,
    )
    __all__.extend([
        "create_action_space",
        "create_observation_space", 
        "ActionSpaceConfig",
        "ObservationSpaceConfig",
    ])
except ImportError:
    # Spaces module not available
    create_action_space = None
    create_observation_space = None
    ActionSpaceConfig = None
    ObservationSpaceConfig = None

try:
    from odor_plume_nav.environments.wrappers import (
        NormalizationWrapper,
        FrameStackWrapper,
        ClippingWrapper,
        RewardShapingWrapper,
    )
    __all__.extend([
        "NormalizationWrapper",
        "FrameStackWrapper",
        "ClippingWrapper", 
        "RewardShapingWrapper",
    ])
except ImportError:
    # Wrappers module not available
    NormalizationWrapper = None
    FrameStackWrapper = None
    ClippingWrapper = None
    RewardShapingWrapper = None