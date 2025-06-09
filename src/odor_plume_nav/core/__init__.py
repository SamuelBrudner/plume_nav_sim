"""
Core modules for odor plume navigation with integrated application-layer functionality.

This module provides the unified public API for odor plume navigation, combining
core simulation engine components with enhanced configuration-driven instantiation
patterns. The integrated architecture merges domain-layer navigation logic with
application-layer factory methods and configuration management capabilities.

Core Navigation Components:
    - Navigator: Core navigation class supporting single and multi-agent scenarios
    - NavigatorProtocol: Protocol interface defining navigator behavior contracts
    - SingleAgentController, MultiAgentController: Enhanced controller implementations
    
Enhanced Factory Patterns:
    - NavigatorFactory: Configuration-driven navigator instantiation with Hydra integration
    - Factory methods for both single-agent and multi-agent scenarios
    - Type-safe parameter validation and structured configuration support
    
Simulation and Orchestration:
    - run_simulation: Comprehensive simulation runner with performance monitoring
    - Real-time performance tracking and optimization (≥30 FPS requirement)
    - Context-managed resource lifecycle for video streams and databases

Integration Features:
    - Unified package structure combining existing and templated capabilities
    - Backward compatibility with existing import patterns
    - Enhanced error handling and structured logging with loguru integration
    - Hydra configuration system integration for ML pipeline compatibility

Example Usage:
    Basic navigation with enhanced controllers:
    
    >>> from odor_plume_nav.core import Navigator, SingleAgentController
    >>> navigator = Navigator.single(position=(10, 20), speed=1.5)
    >>> controller = SingleAgentController(position=(10, 20), speed=1.5)
    
    Configuration-driven instantiation:
    
    >>> from odor_plume_nav.core import NavigatorFactory
    >>> factory = NavigatorFactory()
    >>> navigator = factory.from_config(config_dict)
    
    Simulation orchestration:
    
    >>> from odor_plume_nav.core import run_simulation
    >>> results = run_simulation(navigator, environment, steps=1000)

Architecture Notes:
    This module serves as the stable facade API aggregating components from the
    unified package structure. Changes to this interface require coordination
    with downstream CI import checks and dependent modules that assume the
    consolidated core API.
"""

# Export navigator and simulation from this module
from .navigator import Navigator
from .controllers import SingleAgentController, MultiAgentController
from .protocols import NavigatorProtocol, NavigatorFactory
from .simulation import run_simulation

__all__ = [
    'Navigator',
    'NavigatorFactory',
    'SingleAgentController',
    'MultiAgentController',
    'NavigatorProtocol',
    'run_simulation',
]
