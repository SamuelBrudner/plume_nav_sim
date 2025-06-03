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

# Legacy import paths mapped to new cookiecutter template structure
LEGACY_IMPORT_PATHS = {
    # Legacy odor_plume_nav imports to new {{cookiecutter.project_slug}} structure
    "odor_plume_nav": "{{cookiecutter.project_slug}}",
    "odor_plume_nav.api": "{{cookiecutter.project_slug}}.api.navigation",
    "odor_plume_nav.navigator": "{{cookiecutter.project_slug}}.core.navigator",
    "odor_plume_nav.domain.models": "{{cookiecutter.project_slug}}.config.schemas",
    "odor_plume_nav.domain.navigator_protocol": "{{cookiecutter.project_slug}}.core.navigator",
    "odor_plume_nav.domain.sensor_strategies": "{{cookiecutter.project_slug}}.core.sensors",
    "odor_plume_nav.services.navigation.single_agent_controller": "{{cookiecutter.project_slug}}.core.controllers",
    "odor_plume_nav.services.navigation.multi_agent_controller": "{{cookiecutter.project_slug}}.core.controllers",
    "odor_plume_nav.services.config_loader": "{{cookiecutter.project_slug}}.config.schemas",
    "odor_plume_nav.services.navigator_factory": "{{cookiecutter.project_slug}}.api.navigation",
    "odor_plume_nav.services.simulation_runner": "{{cookiecutter.project_slug}}.api.navigation",
    "odor_plume_nav.services.video_plume_factory": "{{cookiecutter.project_slug}}.api.navigation",
    "odor_plume_nav.adapters.video_plume_opencv": "{{cookiecutter.project_slug}}.data.video_plume",
    "odor_plume_nav.interfaces.api": "{{cookiecutter.project_slug}}.api.navigation",
    "odor_plume_nav.interfaces.visualization.simulation_visualization": "{{cookiecutter.project_slug}}.utils.visualization",
    "odor_plume_nav.interfaces.visualization.trajectory_plots": "{{cookiecutter.project_slug}}.utils.visualization",
    "odor_plume_nav.utils.logging_setup": "{{cookiecutter.project_slug}}.utils.logging",
    
    # Direct legacy mappings for common imports
    "odor_plume_nav.config_utils": "{{cookiecutter.project_slug}}.config.schemas",
    "odor_plume_nav.config_validator": "{{cookiecutter.project_slug}}.config.schemas",
    "odor_plume_nav.video_plume": "{{cookiecutter.project_slug}}.data.video_plume",
    "odor_plume_nav.simulation": "{{cookiecutter.project_slug}}.api.navigation",
    "odor_plume_nav.navigator_factory": "{{cookiecutter.project_slug}}.api.navigation",
    "odor_plume_nav.video_plume_factory": "{{cookiecutter.project_slug}}.api.navigation",
    "odor_plume_nav.visualization": "{{cookiecutter.project_slug}}.utils.visualization",
    "odor_plume_nav.environments.video_plume": "{{cookiecutter.project_slug}}.data.video_plume",
    "odor_plume_nav.core.navigator": "{{cookiecutter.project_slug}}.core.navigator",
    "odor_plume_nav.core.simulation": "{{cookiecutter.project_slug}}.api.navigation",
    "odor_plume_nav.core.navigator_factory": "{{cookiecutter.project_slug}}.api.navigation",
    "odor_plume_nav.environments.video_plume_factory": "{{cookiecutter.project_slug}}.api.navigation",
    "odor_plume_nav.visualization.base": "{{cookiecutter.project_slug}}.utils.visualization",
    "odor_plume_nav.visualization.trajectory": "{{cookiecutter.project_slug}}.utils.visualization",
}
