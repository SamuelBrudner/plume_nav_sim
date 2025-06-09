"""Enhanced Navigator module providing unified navigation capabilities.

This module serves as the primary entry point for odor plume navigation functionality,
combining the core Navigator class with enhanced factory-based instantiation patterns
for configuration-driven component creation.

Features:
    - Navigator: Core navigation class supporting single and multi-agent scenarios
    - NavigatorFactory: Factory class for Hydra-based configuration instantiation
    - Enhanced factory patterns for unified navigator creation
    - Configuration-driven component instantiation support

Example:
    Basic navigation usage:
    
    >>> from odor_plume_nav.core.navigator import Navigator
    >>> navigator = Navigator.single(position=(10, 20), speed=1.5)
    
    Configuration-driven instantiation:
    
    >>> from odor_plume_nav.core.navigator import NavigatorFactory
    >>> factory = NavigatorFactory()
    >>> navigator = factory.from_config(config_dict)

Integration:
    This module integrates both domain-layer navigation patterns and application-layer
    factory methods to support the unified package architecture. It provides backward
    compatibility with existing Navigator imports while enabling new Hydra-based
    configuration workflows.
"""
from odor_plume_nav.domain.navigator import Navigator
from odor_plume_nav.core.protocols import NavigatorFactory

__all__ = ["Navigator", "NavigatorFactory"]
