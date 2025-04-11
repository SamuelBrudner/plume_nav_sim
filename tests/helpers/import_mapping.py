"""
Target import mapping for the odor_plume_nav package.

This module defines the target import locations for all classes 
in the refactored module structure. Tests will use this mapping 
to enforce imports from the correct locations.
"""

from typing import Dict

# Map of class names to their target module paths
TARGET_IMPORT_MAPPING: Dict[str, str] = {
    # Navigator classes
    "Navigator": "odor_plume_nav.core.navigator",
    "SimpleNavigator": "odor_plume_nav.core.navigator",
    "VectorizedNavigator": "odor_plume_nav.core.navigator",
    
    # Environment classes
    "VideoPlume": "odor_plume_nav.environments.video_plume",
    
    # Configuration classes
    "load_config": "odor_plume_nav.config.utils",
    "save_config": "odor_plume_nav.config.utils",
    "validate_config": "odor_plume_nav.config.utils",
    "update_config": "odor_plume_nav.config.utils",
    "deep_update": "odor_plume_nav.config.utils",
    "ConfigValidationError": "odor_plume_nav.config.validator",
    
    # Factory functions
    "create_navigator_from_config": "odor_plume_nav.core.navigator_factory",
    "create_video_plume_from_config": "odor_plume_nav.environments.video_plume_factory",
    
    # Simulation
    "Simulation": "odor_plume_nav.core.simulation",
    "run_simulation": "odor_plume_nav.core.simulation",
    
    # Visualization
    "visualize_trajectory": "odor_plume_nav.visualization.trajectory",
}

# List of files that should be migrated to the new location
FILES_TO_MIGRATE = [
    # Root module files that need to be migrated to submodules
    ("navigator.py", "core/navigator.py"),
    ("config_utils.py", "config/utils.py"),
    ("config_validator.py", "config/validator.py"), 
    ("video_plume.py", "environments/video_plume.py"),
    ("simulation.py", "core/simulation.py"),
    ("navigator_factory.py", "core/navigator_factory.py"),
    ("video_plume_factory.py", "environments/video_plume_factory.py"),
    ("visualization.py", "visualization/base.py"),
]

# Legacy import paths
LEGACY_IMPORT_PATHS = {
    "odor_plume_nav.navigator": "odor_plume_nav.core.navigator",
    "odor_plume_nav.config_utils": "odor_plume_nav.config.utils",
    "odor_plume_nav.config_validator": "odor_plume_nav.config.validator",
    "odor_plume_nav.video_plume": "odor_plume_nav.environments.video_plume",
    "odor_plume_nav.simulation": "odor_plume_nav.core.simulation",
    "odor_plume_nav.navigator_factory": "odor_plume_nav.core.navigator_factory",
    "odor_plume_nav.video_plume_factory": "odor_plume_nav.environments.video_plume_factory",
    "odor_plume_nav.visualization": "odor_plume_nav.visualization.base",
}
