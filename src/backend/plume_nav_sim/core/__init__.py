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

# Core system constants providing configuration defaults and performance targets
from .constants import *  # System-wide constants for environment configuration, performance targets, and component coordination

# Complete type system with enums, data classes, factory functions, and type aliases
from .types import *  # Comprehensive type definitions, data classes, and factory functions for environment implementation

# Central state management for agent state, episode lifecycle, and component synchronization
from .state_manager import (
    StateManager,        # Central state management class coordinating agent state, episode lifecycle, and component synchronization
    StateManagerConfig,  # Configuration data class for state manager with comprehensive validation and parameter management
    create_state_manager # Factory function for creating properly configured StateManager with component coordination and validation
)

# Episode management orchestrator for complete episode lifecycle coordination
from .episode_manager import (
    EpisodeManager,        # Central episode management orchestrator coordinating complete episode lifecycle with component integration
    EpisodeManagerConfig,  # Configuration data class for episode manager with comprehensive validation and component coordination  
    create_episode_manager # Factory function for creating properly configured EpisodeManager with validation and component integration
)

# Action validation, movement calculation, and boundary enforcement integration
from .action_processor import (
    ActionProcessor # Action validation, movement calculation, and boundary enforcement integration
)

# Goal-based reward calculation with distance analysis and termination logic
from .reward_calculator import (
    RewardCalculator # Goal-based reward calculation with distance analysis and termination logic
)

# Boundary constraint enforcement for agent position validation and movement limits
from .boundary_enforcer import (
    BoundaryEnforcer # Boundary constraint enforcement for agent position validation and movement limits
)

# Comprehensive module exports providing complete core package API
__all__ = [
    # System constants for environment configuration and performance optimization
    'PACKAGE_NAME',          # Package identifier constant for system identification and environment registration
    'PACKAGE_VERSION',       # Version identifier following semantic versioning for compatibility tracking and validation
    'DEFAULT_GRID_SIZE',     # Default environment grid dimensions (128, 128) for standard environment configuration
    'DEFAULT_SOURCE_LOCATION', # Default plume source location (64, 64) at grid center for balanced navigation
    'DEFAULT_MAX_STEPS',     # Default maximum episode steps (1000) for truncation and training efficiency
    'DEFAULT_GOAL_RADIUS',   # Default goal detection radius (0) requiring exact source location for termination
    
    # Core action and rendering enumerations with utility methods
    'Action',               # Discrete action enumeration for agent movement with vector calculation and direction analysis
    'RenderMode',           # Enumeration for dual-mode rendering with output format analysis and display requirements
    
    # Fundamental coordinate and spatial data structures
    'Coordinates',          # Immutable 2D coordinate representation with distance calculations and movement operations
    'GridSize',             # Immutable grid dimension representation with memory estimation and performance analysis
    
    # Agent and episode state management data structures
    'AgentState',           # Mutable agent state tracking with position, rewards, and trajectory analysis
    'EpisodeState',         # Comprehensive episode state management with tracking and analysis capabilities
    'EnvironmentConfig',    # Complete environment configuration with validation and resource estimation
    'PlumeParameters',      # Plume model configuration with mathematical validation and consistency checking
    'StateSnapshot',        # Immutable state snapshot for debugging, analysis, and reproducibility
    'PerformanceMetrics',   # Performance metrics collection with trend analysis and optimization recommendations
    
    # Type aliases for Gymnasium API compatibility and flexible parameter validation
    'ActionType',           # Type alias for Union[Action, int] providing flexible action parameter validation
    'ObservationType',      # Type alias for numpy.ndarray representing observation values from environment
    'RewardType',           # Type alias for float representing reward values from environment step
    'InfoType',             # Type alias for dict representing info dictionary returned by environment step
    
    # Core component classes for environment implementation and coordination
    'StateManager',         # Central state management class coordinating agent state, episode lifecycle, and component synchronization
    'StateManagerConfig',   # Configuration data class for state manager with comprehensive validation and parameter management
    'EpisodeManager',       # Central episode management orchestrator coordinating complete episode lifecycle with component integration
    'EpisodeManagerConfig', # Configuration data class for episode manager with comprehensive validation and component coordination
    'ActionProcessor',      # Action validation, movement calculation, and boundary enforcement integration
    'RewardCalculator',     # Goal-based reward calculation with distance analysis and termination logic
    'BoundaryEnforcer',     # Boundary constraint enforcement for agent position validation and movement limits
    
    # Factory functions for creating validated data structures and components
    'create_coordinates',    # Factory function for creating validated Coordinates from various input formats with bounds checking
    'create_grid_size',      # Factory function for creating validated GridSize with memory and performance validation
    'create_agent_state',    # Factory function for creating initialized AgentState for episode start with performance tracking
    'create_episode_state',  # Factory function for creating initialized EpisodeState with tracking and history setup
    'create_environment_config', # Factory function for creating complete EnvironmentConfig with comprehensive validation
    'create_state_manager',  # Factory function for creating properly configured StateManager with component coordination and validation
    'create_episode_manager', # Factory function for creating properly configured EpisodeManager with validation and component integration
    
    # Validation utilities for type-safe parameter checking and error handling
    'validate_action',       # Type-safe action validation ensuring action is valid Action enum or integer in discrete action space
    'validate_coordinates',  # Comprehensive coordinate validation with type checking, bounds validation, and grid compatibility
    'validate_grid_size'     # Grid size validation with dimension checking, memory estimation, and performance feasibility analysis
]