"""
Core package initialization module for plume_nav_sim providing centralized access to all core
components including constants, types, state management, episode management, action processing,
reward calculation, and boundary enforcement with comprehensive API exposure for Gymnasium-compatible
reinforcement learning environment implementation.

This module serves as the unified entry point for the plume_nav_sim core package, exposing a
comprehensive API that includes:
- System-wide constants and configuration values
- Complete type system with enums, data classes, and factory functions
- State management components for agent and episode lifecycle
- Episode orchestration and coordination services
- Action processing and validation framework
- Reward calculation and termination logic
- Boundary enforcement and position validation

The module follows enterprise-grade patterns with comprehensive error handling, performance
optimization, type safety, and extensive documentation to support production-ready reinforcement
learning environment implementations compatible with the Gymnasium framework.

Key Features:
- Modular component architecture with clean separation of concerns
- Type-safe interfaces with comprehensive validation and error handling
- Performance-optimized core operations targeting <1ms step latency
- Extensive factory functions for proper component initialization
- Cross-component coordination and dependency injection support
- Comprehensive logging and monitoring integration
- Reproducibility framework with seeding utilities
- Gymnasium API compliance with standard RL ecosystem integration
"""

# Action validation, movement calculation, and boundary enforcement integration
from .constants import *  # System-wide constants for environment configuration, performance targets, and component coordination

# Expose only the lightweight core types needed by environments at import time
from .enums import Action, RenderMode
from .geometry import Coordinates, GridSize
from .types import EnvironmentConfig, create_coordinates, create_grid_size
from .typing import ActionType, ObservationType, RewardType, InfoType

# Comprehensive module exports providing complete core package API
__all__ = [
    "PACKAGE_NAME",
    "PACKAGE_VERSION",
    "DEFAULT_GRID_SIZE",
    "DEFAULT_SOURCE_LOCATION",
    "DEFAULT_MAX_STEPS",
    "DEFAULT_GOAL_RADIUS",
    "Action",
    "RenderMode",
    "Coordinates",
    "GridSize",
    "EnvironmentConfig",
    "ActionType",
    "ObservationType",
    "RewardType",
    "InfoType",
    "create_coordinates",
    "create_grid_size",
]
