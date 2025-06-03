"""
Target import mapping for the {{cookiecutter.project_slug}} package.

This module defines the target import locations for all classes 
in the refactored cookiecutter template structure. Tests will use this mapping 
to enforce imports from the correct locations per Section 0.2.1 technical reorganization.

The mapping reflects the migration from the legacy odor_plume_nav structure to the
standardized cookiecutter template with enhanced modularity for Kedro integration,
reinforcement learning frameworks, and machine learning analysis workflows.
"""

from typing import Dict

# Map of class names to their target module paths in cookiecutter template structure
TARGET_IMPORT_MAPPING: Dict[str, str] = {
    # Core navigation classes and protocols
    "NavigatorProtocol": "{{cookiecutter.project_slug}}.core.navigator",
    "Navigator": "{{cookiecutter.project_slug}}.core.navigator", 
    "SimpleNavigator": "{{cookiecutter.project_slug}}.core.navigator",
    "VectorizedNavigator": "{{cookiecutter.project_slug}}.core.navigator",
    
    # Navigation controllers (migrated from services/navigation/)
    "SingleAgentController": "{{cookiecutter.project_slug}}.core.controllers",
    "MultiAgentController": "{{cookiecutter.project_slug}}.core.controllers",
    
    # Sensor strategies
    "SensorProtocol": "{{cookiecutter.project_slug}}.core.sensors",
    "SingleSensorConfig": "{{cookiecutter.project_slug}}.core.sensors",
    "MultiSensorConfig": "{{cookiecutter.project_slug}}.core.sensors",
    
    # Data processing classes (migrated from adapters/environments)
    "VideoPlume": "{{cookiecutter.project_slug}}.data.video_plume",
    
    # Configuration schemas and validation (migrated from domain/models)
    "NavigatorConfig": "{{cookiecutter.project_slug}}.config.schemas",
    "SingleAgentConfig": "{{cookiecutter.project_slug}}.config.schemas",
    "MultiAgentConfig": "{{cookiecutter.project_slug}}.config.schemas",
    "VideoConfig": "{{cookiecutter.project_slug}}.config.schemas",
    "SimulationConfig": "{{cookiecutter.project_slug}}.config.schemas",
    "ConfigValidationError": "{{cookiecutter.project_slug}}.config.schemas",
    
    # API factory functions and simulation orchestration
    "create_navigator": "{{cookiecutter.project_slug}}.api.navigation",
    "create_video_plume": "{{cookiecutter.project_slug}}.api.navigation", 
    "run_plume_simulation": "{{cookiecutter.project_slug}}.api.navigation",
    "create_navigator_from_config": "{{cookiecutter.project_slug}}.api.navigation",
    "create_video_plume_from_config": "{{cookiecutter.project_slug}}.api.navigation",
    
    # Consolidated visualization components
    "SimulationVisualization": "{{cookiecutter.project_slug}}.utils.visualization",
    "visualize_trajectory": "{{cookiecutter.project_slug}}.utils.visualization",
    "animate_plume_navigation": "{{cookiecutter.project_slug}}.utils.visualization",
    "export_animation": "{{cookiecutter.project_slug}}.utils.visualization",
    
    # New utility components
    "set_global_seed": "{{cookiecutter.project_slug}}.utils.seed_manager",
    "get_random_state": "{{cookiecutter.project_slug}}.utils.seed_manager", 
    "SeedManager": "{{cookiecutter.project_slug}}.utils.seed_manager",
    
    # CLI interface components
    "main": "{{cookiecutter.project_slug}}.cli.main",
    "run_simulation_cli": "{{cookiecutter.project_slug}}.cli.main",
    "validate_config_cli": "{{cookiecutter.project_slug}}.cli.main",
    
    # Database session management (new infrastructure component)
    "DatabaseSessionManager": "{{cookiecutter.project_slug}}.db.session",
    "get_session": "{{cookiecutter.project_slug}}.db.session",
    "create_engine": "{{cookiecutter.project_slug}}.db.session",
}

# List of files that should be migrated to the new cookiecutter template structure
# Based on Section 0.2.1 comprehensive mapping table
FILES_TO_MIGRATE = [
    # Core navigation components
    ("src/odor_plume_nav/api.py", "src/{{cookiecutter.project_slug}}/api/navigation.py"),
    ("src/odor_plume_nav/domain/navigator_protocol.py", "src/{{cookiecutter.project_slug}}/core/navigator.py"),
    ("src/odor_plume_nav/domain/sensor_strategies.py", "src/{{cookiecutter.project_slug}}/core/sensors.py"),
    ("src/odor_plume_nav/services/navigation/single_agent_controller.py", "src/{{cookiecutter.project_slug}}/core/controllers.py"),
    ("src/odor_plume_nav/services/navigation/multi_agent_controller.py", "src/{{cookiecutter.project_slug}}/core/controllers.py"),
    
    # Data processing components
    ("src/odor_plume_nav/adapters/video_plume_opencv.py", "src/{{cookiecutter.project_slug}}/data/video_plume.py"),
    
    # Configuration and validation
    ("src/odor_plume_nav/domain/models.py", "src/{{cookiecutter.project_slug}}/config/schemas.py"),
    ("src/odor_plume_nav/services/config_loader.py", "conf/base.yaml"),
    
    # Simulation orchestration  
    ("src/odor_plume_nav/services/simulation_runner.py", "src/{{cookiecutter.project_slug}}/api/navigation.py"),
    ("src/odor_plume_nav/services/navigator_factory.py", "src/{{cookiecutter.project_slug}}/api/navigation.py"),
    ("src/odor_plume_nav/services/video_plume_factory.py", "src/{{cookiecutter.project_slug}}/api/navigation.py"),
    
    # Visualization components consolidation
    ("src/odor_plume_nav/interfaces/visualization/simulation_visualization.py", "src/{{cookiecutter.project_slug}}/utils/visualization.py"),
    ("src/odor_plume_nav/interfaces/visualization/trajectory_plots.py", "src/{{cookiecutter.project_slug}}/utils/visualization.py"),
    
    # Utility components
    ("src/odor_plume_nav/utils/logging_setup.py", "src/{{cookiecutter.project_slug}}/utils/logging.py"),
    
    # Configuration files migration
    ("configs/default.yaml", "conf/base.yaml"),
    ("configs/example_user_config.yaml", "conf/config.yaml"),
    
    # Example notebooks migration
    ("examples/agent_visualization_demo.py", "notebooks/demos/agent_visualization_demo.ipynb"),
    
    # New components (created during refactoring)
    ("", "src/{{cookiecutter.project_slug}}/cli/main.py"),
    ("", "src/{{cookiecutter.project_slug}}/utils/seed_manager.py"),
    ("", "src/{{cookiecutter.project_slug}}/db/session.py"),
]

# Legacy import paths mapping from old odor_plume_nav structure to new cookiecutter template
LEGACY_IMPORT_PATHS = {
    # Root level legacy imports to new core structure
    "odor_plume_nav.api": "{{cookiecutter.project_slug}}.api.navigation",
    "odor_plume_nav.navigator": "{{cookiecutter.project_slug}}.core.navigator",
    "odor_plume_nav.video_plume": "{{cookiecutter.project_slug}}.data.video_plume",
    
    # Domain module migrations
    "odor_plume_nav.domain.models": "{{cookiecutter.project_slug}}.config.schemas",
    "odor_plume_nav.domain.navigator_protocol": "{{cookiecutter.project_slug}}.core.navigator", 
    "odor_plume_nav.domain.sensor_strategies": "{{cookiecutter.project_slug}}.core.sensors",
    
    # Services module reorganization
    "odor_plume_nav.services.navigation.single_agent_controller": "{{cookiecutter.project_slug}}.core.controllers",
    "odor_plume_nav.services.navigation.multi_agent_controller": "{{cookiecutter.project_slug}}.core.controllers",
    "odor_plume_nav.services.config_loader": "{{cookiecutter.project_slug}}.config.schemas",
    "odor_plume_nav.services.navigator_factory": "{{cookiecutter.project_slug}}.api.navigation",
    "odor_plume_nav.services.simulation_runner": "{{cookiecutter.project_slug}}.api.navigation",
    "odor_plume_nav.services.video_plume_factory": "{{cookiecutter.project_slug}}.api.navigation",
    
    # Adapters to data module
    "odor_plume_nav.adapters.video_plume_opencv": "{{cookiecutter.project_slug}}.data.video_plume",
    
    # Interfaces reorganization
    "odor_plume_nav.interfaces.api": "{{cookiecutter.project_slug}}.api.navigation",
    "odor_plume_nav.interfaces.visualization.simulation_visualization": "{{cookiecutter.project_slug}}.utils.visualization",
    "odor_plume_nav.interfaces.visualization.trajectory_plots": "{{cookiecutter.project_slug}}.utils.visualization",
    
    # Utilities migration
    "odor_plume_nav.utils.logging_setup": "{{cookiecutter.project_slug}}.utils.logging",
    
    # Consolidated legacy paths for backwards compatibility
    "odor_plume_nav.config_utils": "{{cookiecutter.project_slug}}.config.schemas",
    "odor_plume_nav.config_validator": "{{cookiecutter.project_slug}}.config.schemas", 
    "odor_plume_nav.simulation": "{{cookiecutter.project_slug}}.api.navigation",
    "odor_plume_nav.visualization": "{{cookiecutter.project_slug}}.utils.visualization",
}