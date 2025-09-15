"""
Comprehensive API reference documentation module for plume_nav_sim providing complete documentation 
of all public interfaces, classes, functions, and usage patterns. Serves as the authoritative reference 
for researchers, developers, and users with detailed examples, configuration options, integration patterns, 
and cross-component relationships.

This module implements interactive documentation generation, code examples, and comprehensive usage 
guides for all system components with focus on research workflow integration and scientific reproducibility.
"""

# External imports with version comments
import gymnasium as gym  # >=0.29.0 - Reinforcement learning environment framework for API compatibility documentation
import numpy as np  # >=2.1.0 - Mathematical operations and array handling for data structure documentation
import matplotlib.pyplot as plt  # >=3.9.0 - Visualization examples and human rendering documentation
from typing import Dict, List, Any, Optional, Union, Tuple, Type, Callable  # >=3.10 - Type annotations for comprehensive API documentation
import inspect  # >=3.10 - Runtime introspection for automatic documentation generation
import json  # >=3.10 - JSON serialization for configuration examples and structured output
from pathlib import Path  # >=3.10 - Path handling for documentation file operations
import textwrap  # >=3.10 - Text formatting utilities for generating readable documentation

# Internal imports - Core environment and registration system
from ..plume_nav_sim.envs.plume_search_env import (
    PlumeSearchEnv, create_plume_search_env
)
from ..plume_nav_sim.registration.register import (
    register_env, unregister_env, is_registered, get_registration_info, ENV_ID
)

# Internal imports - Plume model and mathematical components
from ..plume_nav_sim.plume.static_gaussian import (
    StaticGaussianPlume, create_static_gaussian_plume, calculate_gaussian_concentration
)

# Internal imports - Rendering and visualization
from ..plume_nav_sim.render.numpy_rgb import (
    NumpyRGBRenderer, create_rgb_renderer
)

# Internal imports - Seeding and reproducibility utilities
from ..plume_nav_sim.utils.seeding import (
    SeedManager, ReproducibilityTracker, validate_seed, create_seeded_rng
)

# Internal imports - Core types and data structures
from ..plume_nav_sim.core.types import (
    Action, Coordinates, GridSize, AgentState, EpisodeState, PlumeParameters,
    EnvironmentConfig, create_coordinates, create_grid_size, create_environment_config
)

# Internal imports - Constants and configuration
from ..plume_nav_sim.core.constants import (
    DEFAULT_GRID_SIZE, DEFAULT_SOURCE_LOCATION, get_default_environment_constants,
    get_plume_model_constants, get_action_space_constants, get_rendering_constants
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
CODE_BLOCK_SEPARATOR = "```python\n{}\n