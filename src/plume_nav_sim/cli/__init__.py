"""
Command-line interface module for plume_nav_sim.

This module provides console script entry points for the plume navigation simulation library.
"""

from .main import (
    main, 
    train_main,
    cli,
    run,
    config, 
    visualize,
    batch,
    CLIError,
    ConfigValidationError,
    _setup_cli_logging,
    _validate_hydra_availability, 
    _validate_configuration,
    _export_config_documentation,
    _measure_performance,
    _safe_config_access,
    _CLI_CONFIG,
    get_cli_version,
    is_cli_available,
    validate_cli_environment,
    register_command,
    extend_cli,
    run_command,
    get_available_commands,
    CLI_CONFIG
)

__version__ = "1.0.0"
__all__ = [
    "main", 
    "train_main",
    "cli",
    "run",
    "config", 
    "visualize",
    "batch",
    "CLIError",
    "ConfigValidationError",
    "_setup_cli_logging",
    "_validate_hydra_availability", 
    "_validate_configuration",
    "_export_config_documentation",
    "_measure_performance",
    "_safe_config_access",
    "_CLI_CONFIG",
    "get_cli_version",
    "is_cli_available",
    "validate_cli_environment",
    "register_command",
    "extend_cli",
    "run_command",
    "get_available_commands",
    "CLI_CONFIG"
]