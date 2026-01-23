"""
Comprehensive API reference documentation module for plume_nav_sim providing complete documentation
of all public interfaces, classes, functions, and usage patterns. Serves as the authoritative reference
for researchers, developers, and users with detailed examples, configuration options, integration patterns,
and cross-component relationships.

This module implements interactive documentation generation, code examples, and comprehensive usage
guides for all system components with focus on research workflow integration and scientific reproducibility.
"""

import inspect  # >=3.10 - Runtime introspection for automatic documentation generation
import json  # >=3.10 - JSON serialization for configuration examples and structured output
import textwrap  # >=3.10 - Text formatting utilities for generating readable documentation
from pathlib import Path  # >=3.10 - Path handling for documentation file operations
from typing import (  # >=3.10 - Type annotations for comprehensive API documentation
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)

# External imports with version comments
import gymnasium as gym  # >=0.29.0 - Reinforcement learning environment framework for API compatibility documentation
import matplotlib.pyplot as plt  # >=3.9.0 - Visualization examples and human rendering documentation
import numpy as np  # >=2.1.0 - Mathematical operations and array handling for data structure documentation

# Internal imports - Constants and configuration
from ..plume_nav_sim.core.constants import (
    DEFAULT_GRID_SIZE,
    DEFAULT_SOURCE_LOCATION,
    get_default_environment_constants,
)

# Internal imports - Core types and data structures
from ..plume_nav_sim.core.types import (
    Action,
    AgentState,
    Coordinates,
    GridSize,
    create_coordinates,
    create_grid_size,
)
from ..plume_nav_sim.envs.config_types import EnvironmentConfig, create_environment_config

# Internal imports - Core environment and registration system
from ..plume_nav_sim.envs.plume_search_env import (
    PlumeSearchEnv,
    create_plume_search_env,
)

# Internal imports - Plume model and mathematical components
from ..plume_nav_sim.plume.static_gaussian import (
    StaticGaussianPlume,
    calculate_gaussian_concentration,
    create_static_gaussian_plume,
)
from ..plume_nav_sim.registration.register import (
    ENV_ID,
    get_registration_info,
    is_registered,
    register_env,
    unregister_env,
)

# Internal imports - Rendering and visualization
from ..plume_nav_sim.render.numpy_rgb import NumpyRGBRenderer, create_rgb_renderer

# Internal imports - Seeding and reproducibility utilities
from ..plume_nav_sim.utils.seeding import (
    ReproducibilityTracker,
    SeedManager,
    create_seeded_rng,
    validate_seed,
)

# Global documentation configuration
API_VERSION = '1.0.0'
DOCUMENTATION_FORMAT = 'markdown'
INCLUDE_EXAMPLES = True
INCLUDE_PERFORMANCE_NOTES = True
EXAMPLE_OUTPUT_DIR = 'examples'
_documentation_cache = {}

# Documentation template constants
SECTION_SEPARATOR = "\n" + "="*80 + "\n"
SUBSECTION_SEPARATOR = "\n" + "-"*60 + "\n"
CODE_BLOCK_SEPARATOR = "```python\n{}\n```"

__all__ = [
    "inspect",
    "json",
    "textwrap",
    "Path",
    "Any",
    "Callable",
    "Dict",
    "List",
    "Optional",
    "Tuple",
    "Type",
    "Union",
    "gym",
    "plt",
    "np",
    "DEFAULT_GRID_SIZE",
    "DEFAULT_SOURCE_LOCATION",
    "get_default_environment_constants",
    "Action",
    "AgentState",
    "Coordinates",
    "EnvironmentConfig",
    "GridSize",
    "create_coordinates",
    "create_environment_config",
    "create_grid_size",
    "PlumeSearchEnv",
    "create_plume_search_env",
    "StaticGaussianPlume",
    "calculate_gaussian_concentration",
    "create_static_gaussian_plume",
    "ENV_ID",
    "get_registration_info",
    "is_registered",
    "register_env",
    "unregister_env",
    "NumpyRGBRenderer",
    "create_rgb_renderer",
    "ReproducibilityTracker",
    "SeedManager",
    "create_seeded_rng",
    "validate_seed",
]
