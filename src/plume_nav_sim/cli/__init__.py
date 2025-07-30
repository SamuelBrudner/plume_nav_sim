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
    train,
    CLIError,
    ConfigValidationError,
    ConfigurationError,
    SimulationError,
    _setup_cli_logging,
    _validate_hydra_availability, 
    _measure_performance,
    _safe_config_access,
    _CLI_CONFIG,
    HYDRA_AVAILABLE,
    GYMNASIUM_AVAILABLE,
    SB3_AVAILABLE
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
    "train",
    "CLIError",
    "ConfigValidationError",
    "ConfigurationError",
    "SimulationError",
    "_setup_cli_logging",
    "_validate_hydra_availability", 
    "_measure_performance",
    "_safe_config_access",
    "_CLI_CONFIG",
    "HYDRA_AVAILABLE",
    "GYMNASIUM_AVAILABLE",
    "SB3_AVAILABLE"
]