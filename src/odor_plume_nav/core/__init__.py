"""
Core modules for odor plume navigation.

This module contains the core classes and functions for agent navigation.
"""

# Export navigator and simulation from this module
from .legacy_navigator import Navigator as LegacyNavigator
from .legacy_navigator import VectorizedNavigator
from .navigator import Navigator
from .controllers import SingleAgentController, MultiAgentController
from .protocols import NavigatorProtocol
from .simulation import run_simulation

__all__ = [
    'LegacyNavigator',
    'VectorizedNavigator',
    'Navigator',
    'SingleAgentController',
    'MultiAgentController',
    'NavigatorProtocol',
    'run_simulation',
]
