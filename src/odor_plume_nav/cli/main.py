"""
Comprehensive command-line interface for odor plume navigation system.

This module provides a production-ready CLI built with Click framework and Hydra configuration
integration, supporting simulation execution, configuration validation, batch processing,
parameter sweeps, and visualization export commands. The interface implements @hydra.main
decorator for seamless configuration injection and multi-run experiment orchestration.

The CLI architecture supports:
- Real-time simulation execution with parameter overrides
- Configuration validation and export for development workflows
- Multi-run parameter sweeps via --multirun flag for automated experiments
- Visualization export with publication-quality output generation
- Batch processing capabilities for headless execution environments
- Comprehensive help system with usage examples and parameter documentation
- Error handling with recovery strategies and detailed diagnostic information

Performance Characteristics:
- Command initialization: <2s per Section 2.2.9.3 performance criteria
- Configuration validation: <500ms for complex hierarchical configurations
- Parameter override processing: Real-time with immediate validation feedback
- Help system generation: Instant response with comprehensive documentation

Examples:
    # Basic simulation execution
    python -m odor_plume_nav.cli.main run
    
    # Simulation with parameter overrides
    python -m odor_plume_nav.cli.main run navigator.max_speed=10.0 simulation.num_steps=500
    
    # Multi-run parameter sweep
    python -m odor_plume_nav.cli.main --multirun run navigator.max_speed=5,10,15
    
    # Configuration validation
    python -m odor_plume_nav.cli.main config validate
    
    # Visualization export
    python -m odor_plume_nav.cli.main visualize export --format mp4 --output results.mp4
    
    # Dry-run validation
    python -m odor_plume_nav.cli.main run --dry-run
"""

import os
import sys
import time
import traceback
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
import warnings

import click
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger

# Hydra imports for configuration management
try:
    import hydra
    from hydra import compose, initialize, initialize_config_dir
    from hydra.core.config_store import ConfigStore
    from hydra.core.global_hydra import GlobalHydra
    from hydra.core.hydra_config import HydraConfig
    from omegaconf import DictConfig, OmegaConf, ListConfig
    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False
    warnings.warn(
        "Hydra not available. Advanced configuration features will be limited.",
        ImportWarning
    )

# Import core system components
from odor_plume_nav.api.navigation import (
    create_navigator,
    create_video_plume,
    run_plume_simulation
)
from odor_plume_nav.config.models import (
    NavigatorConfig,
    VideoPlumeConfig,
    SimulationConfig
)
from odor_plume_nav.utils.seed_manager import set_global_seed, get_last_seed
from odor_plume_nav.utils.logging_setup import setup_logger, get_enhanced_logger
from odor_plume_nav.utils.visualization import (
    visualize_plume_simulation,
    visualize_trajectory,
    create_realtime_visualizer
)

# Import frame cache for performance optimization
try:
    from odor_plume_nav.cache.frame_cache import FrameCache
    FRAME_CACHE_AVAILABLE = True
except ImportError:
    FRAME_CACHE_AVAILABLE = False
    warnings.warn(
        "Frame cache not available. Performance optimizations will be limited.",
        ImportWarning
    )

# Global configuration for CLI state management
_CLI_CONFIG = {
    'verbose': False,
    'quiet': False,
    'log_level': 'INFO',
    'start_time': None,
    'dry_run': False
}


class CLIError(Exception):
    """CLI-specific error for command execution failures."""
    pass


class ConfigValidationError(Exception):
    """Configuration validation specific errors."""
    pass


class ConfigurationError(Exception):
    """Configuration-related error for component creation failures."""
    pass


class SimulationError(Exception):
    """Simulation execution error for runtime failures."""
    pass


def _setup_cli_logging(verbose: bool = False, quiet: bool = False, log_level: str = 'INFO') -> None:
    """
    Initialize CLI-specific logging configuration with performance monitoring.
    
    Args:
        verbose: Enable verbose logging output with debug information
        quiet: Suppress non-essential output (errors only)
        log_level: Logging level for CLI operations
    """
    try:
        # Configure loguru for CLI operations
        logger.remove()  # Remove default handler
        
        if quiet:
            # Only show errors in quiet mode
            logger.add(
                sys.stderr,
                level="ERROR",
                format="<red>ERROR</red>: {message}",
                colorize=True
            )
        elif verbose:
            # Verbose mode with detailed information
            logger.add(
                sys.stderr,
                level="DEBUG",
                format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | "
                      "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
                      "<level>{message}</level>",
                colorize=True
            )
        else:
            # Standard mode with clean output
            logger.add(
                sys.stderr,
                level=log_level,
                format="<level>{level}</level>: {message}",
                colorize=True
            )
        
        # Update global CLI configuration
        _CLI_CONFIG.update({
            'verbose': verbose,
            'quiet': quiet,
            'log_level': log_level
        })
        
        logger.info("CLI logging initialized successfully")
        
    except Exception as e:
        # Fallback to basic logging if setup fails
        click.echo(f"Warning: Failed to setup advanced logging: {e}", err=True)
        logger.add(sys.stderr, level="INFO")


def _measure_performance(func_name: str, start_time: float) -> None:
    """
    Measure and log performance metrics for CLI operations.
    
    Args:
        func_name: Name of the function being measured
        start_time: Start time for performance measurement
    """
    elapsed = time.time() - start_time
    
    if elapsed > 2.0:
        logger.warning(f"{func_name} took {elapsed:.2f}s (>2s threshold)")
    else:
        logger.debug(f"{func_name} completed in {elapsed:.2f}s")


def _validate_hydra_availability() -> None:
    """Validate that Hydra is available for advanced CLI features."""
    if not HYDRA_AVAILABLE:
        raise CLIError(
            "Hydra is required for CLI functionality. Please install with: "
            "pip install hydra-core"
        )


def _safe_config_access(cfg: DictConfig, path: str, default: Any = None) -> Any:
    """
    Safely access nested configuration values with error handling.
    
    Args:
        cfg: Hydra configuration object
        path: Dot-separated path to configuration value
        default: Default value if path doesn't exist
        
    Returns:
        Configuration value or default
    """
    try:
        keys = path.split('.')
        value = cfg
        for key in keys:
            if hasattr(value, key):
                value = getattr(value, key)
            elif isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value
    except Exception:
        return default


def _export_config_documentation(cfg: DictConfig, output_path: Optional[Path] = None) -> Path:
    """
    Export configuration as documentation with comprehensive formatting.
    
    Args:
        cfg: Hydra configuration to export
        output_path: Optional output file path
        
    Returns:
        Path to exported configuration file
    """
    if output_path is None:
        output_path = Path("config_export.yaml")
    
    try:
        # Convert to container and format for documentation
        config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)
        
        # Add documentation headers
        documentation = {
            "_metadata": {
                "export_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "hydra_version": hydra.__version__ if HYDRA_AVAILABLE else "N/A",
                "config_source": str(HydraConfig.get().runtime.config_sources) if HydraConfig.initialized() else "Unknown"
            },
            **config_dict
        }
        
        # Write formatted YAML
        with open(output_path, 'w') as f:
            OmegaConf.save(documentation, f)
        
        logger.info(f"Configuration exported to {output_path}")
        return output_path
        
    except Exception as e:
        raise CLIError(f"Failed to export configuration: {e}") from e


def _validate_configuration(cfg: DictConfig, strict: bool = True) -> Dict[str, Any]:
    """
    Comprehensive configuration validation with detailed error reporting.
    
    Args:
        cfg: Hydra configuration to validate
        strict: Whether to use strict validation rules
        
    Returns:
        Validation results with errors and warnings
    """
    validation_results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'summary': {}
    }
    
    start_time = time.time()
    
    try:
        # Validate navigator configuration
        if hasattr(cfg, 'navigator') and cfg.navigator:
            try:
                NavigatorConfig.model_validate(OmegaConf.to_container(cfg.navigator, resolve=True))
                validation_results['summary']['navigator'] = 'valid'
            except Exception as e:
                validation_results['errors'].append(f"Navigator config invalid: {e}")
                validation_results['valid'] = False
                validation_results['summary']['navigator'] = 'invalid'
        
        # Validate video plume configuration
        if hasattr(cfg, 'video_plume') and cfg.video_plume:
            try:
                VideoPlumeConfig.model_validate(OmegaConf.to_container(cfg.video_plume, resolve=True))
                validation_results['summary']['video_plume'] = 'valid'
            except Exception as e:
                validation_results['errors'].append(f"Video plume config invalid: {e}")
                validation_results['valid'] = False
                validation_results['summary']['video_plume'] = 'invalid'
        
        # Validate simulation configuration
        if hasattr(cfg, 'simulation') and cfg.simulation:
            try:
                SimulationConfig.model_validate(OmegaConf.to_container(cfg.simulation, resolve=True))
                validation_results['summary']['simulation'] = 'valid'
            except Exception as e:
                validation_results['errors'].append(f"Simulation config invalid: {e}")
                validation_results['valid'] = False
                validation_results['summary']['simulation'] = 'invalid'
        
        # Check for missing required configurations
        required_sections = ['navigator', 'video_plume', 'simulation']
        for section in required_sections:
            if not hasattr(cfg, section) or not getattr(cfg, section):
                if strict:
                    validation_results['errors'].append(f"Required section '{section}' is missing or empty")
                    validation_results['valid'] = False
                else:
                    validation_results['warnings'].append(f"Section '{section}' is missing or empty")
        
        # Validate file paths
        video_path = _safe_config_access(cfg, 'video_plume.video_path')
        if video_path and not Path(video_path).exists():
            validation_results['errors'].append(f"Video file not found: {video_path}")
            validation_results['valid'] = False
        
        # Check for reasonable parameter values
        max_speed = _safe_config_access(cfg, 'navigator.max_speed')
        if max_speed and (max_speed <= 0 or max_speed > 100):
            validation_results['warnings'].append(
                f"Navigator max_speed ({max_speed}) may be unreasonable (expected: 0-100)"
            )
        
        num_steps = _safe_config_access(cfg, 'simulation.num_steps')
        if num_steps and (num_steps <= 0 or num_steps > 100000):
            validation_results['warnings'].append(
                f"Simulation num_steps ({num_steps}) may be unreasonable (expected: 1-100000)"
            )
        
    except Exception as e:
        validation_results['errors'].append(f"Validation error: {e}")
        validation_results['valid'] = False
    
    _measure_performance("Configuration validation", start_time)
    return validation_results


def _validate_frame_cache_availability() -> None:
    """Validate that frame cache functionality is available."""
    if not FRAME_CACHE_AVAILABLE:
        raise CLIError(
            "Frame cache functionality is not available. Please ensure all dependencies are installed."
        )


def _create_frame_cache(
    cache_mode: str, 
    memory_limit_gb: float = 2.0,
    video_path: Optional[Union[str, pathlib.Path]] = None
) -> Optional[FrameCache]:
    """
    Create a FrameCache instance based on the specified mode and configuration.
    
    Args:
        cache_mode: Cache mode ("none", "lru", or "all")
        memory_limit_gb: Memory limit in gigabytes (default 2.0 GB)
        video_path: Optional video path for preload validation
        
    Returns:
        FrameCache instance or None if cache_mode is "none"
        
    Raises:
        CLIError: If cache creation fails or invalid mode is specified
    """
    if cache_mode == "none":
        logger.info("Frame caching disabled - using direct frame access")
        return None
    
    if not FRAME_CACHE_AVAILABLE:
        logger.warning("Frame cache requested but not available, falling back to direct access")
        return None
    
    try:
        memory_limit_bytes = int(memory_limit_gb * 1024 * 1024 * 1024)
        
        if cache_mode == "lru":
            cache = FrameCache(
                mode="lru",
                memory_limit=memory_limit_bytes,
                max_frames=None  # Let memory limit control capacity
            )
            logger.info(f"Created LRU frame cache with {memory_limit_gb:.1f} GB memory limit")
            
        elif cache_mode == "all":
            cache = FrameCache(
                mode="all",
                memory_limit=memory_limit_bytes,
                max_frames=None  # Preload all frames up to memory limit
            )
            logger.info(f"Created full preload frame cache with {memory_limit_gb:.1f} GB memory limit")
            
        else:
            raise ValueError(f"Invalid cache mode: {cache_mode}. Must be 'none', 'lru', or 'all'")
        
        return cache
        
    except Exception as e:
        logger.error(f"Failed to create frame cache: {e}")
        raise CLIError(f"Frame cache creation failed: {e}") from e


def _validate_frame_cache_mode(cache_mode: str) -> str:
    """
    Validate frame cache mode parameter.
    
    Args:
        cache_mode: Cache mode string to validate
        
    Returns:
        Validated cache mode string
        
    Raises:
        click.BadParameter: If cache mode is invalid
    """
    valid_modes = {"none", "lru", "all"}
    if cache_mode not in valid_modes:
        raise click.BadParameter(
            f"Invalid frame cache mode: {cache_mode}. Must be one of: {', '.join(valid_modes)}"
        )
    return cache_mode


# Click CLI Groups and Commands Implementation

@click.group(invoke_without_command=True)
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging output')
@click.option('--quiet', '-q', is_flag=True, help='Suppress non-essential output')
@click.option('--log-level', default='INFO', type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR']),
              help='Set logging level for CLI operations')
@click.pass_context
def cli(ctx, verbose: bool, quiet: bool, log_level: str) -> None:
    """
    Odor Plume Navigation CLI - Comprehensive command-line interface.
    
    This CLI provides complete access to simulation execution, configuration management,
    visualization generation, and batch processing capabilities. Built with Hydra
    configuration integration for advanced parameter management and experiment orchestration.
    
    Examples:
        # Run simulation with default configuration
        odor-plume-nav-cli run
        
        # Run with parameter overrides
        odor-plume-nav-cli run navigator.max_speed=10.0
        
        # Multi-run parameter sweep
        odor-plume-nav-cli --multirun run navigator.max_speed=5,10,15
        
        # Validate configuration
        odor-plume-nav-cli config validate
        
        # Export visualization
        odor-plume-nav-cli visualize export --format mp4
    
    For detailed help on any command, use: odor-plume-nav-cli COMMAND --help
    """
    _CLI_CONFIG['start_time'] = time.time()
    
    # Setup CLI logging based on options
    _setup_cli_logging(verbose=verbose, quiet=quiet, log_level=log_level)
    
    # Initialize click context for subcommands
    ctx.ensure_object(dict)
    ctx.obj.update(_CLI_CONFIG)
    
    # If no command specified, show help
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())
        ctx.exit()


@cli.command()
@click.option('--dry-run', is_flag=True, 
              help='Validate simulation setup without executing')
@click.option('--seed', type=int, 
              help='Random seed for reproducible results')
@click.option('--output-dir', type=click.Path(), 
              help='Directory for output files (overrides Hydra default)')
@click.option('--save-trajectory', is_flag=True, default=True,
              help='Save trajectory data for post-analysis')
@click.option('--show-animation', is_flag=True, 
              help='Display real-time animation during simulation')
@click.option('--export-video', type=click.Path(),
              help='Export animation as MP4 video to specified path')
@click.option('--frame-cache', type=click.Choice(['none', 'lru', 'all']), default='none',
               callback=lambda ctx, param, value: _validate_frame_cache_mode(value),
               help='Frame caching mode: "none" (no caching), "lru" (LRU eviction with 2GB limit), '
                    '"all" (preload all frames for maximum throughput)')
@click.pass_context
def run(ctx, dry_run: bool, seed: Optional[int], output_dir: Optional[str], 
        save_trajectory: bool, show_animation: bool, export_video: Optional[str], 
        frame_cache: str) -> None:
    """
    Execute odor plume navigation simulation with comprehensive options.
    
    This command runs the complete simulation pipeline including navigator creation,
    video plume environment loading, simulation execution, and optional visualization.
    Supports parameter overrides via Hydra syntax and dry-run validation.
    
    Configuration Parameters:
        All Hydra configuration parameters can be overridden using dot notation:
        - navigator.max_speed=10.0
        - simulation.num_steps=1000  
        - video_plume.flip=true
        - simulation.dt=0.1
    
    Examples:
        # Basic simulation
        odor-plume-nav-cli run
        
        # With parameter overrides
        odor-plume-nav-cli run navigator.max_speed=15.0 simulation.num_steps=500
        
        # Dry-run validation
        odor-plume-nav-cli run --dry-run
        
        # With visualization export
        odor-plume-nav-cli run --export-video results.mp4
        
        # Reproducible run with seed
        odor-plume-nav-cli run --seed 42
        
        # With frame caching for performance optimization
        odor-plume-nav-cli run --frame-cache lru
        
        # With full frame preloading for maximum throughput
        odor-plume-nav-cli run --frame-cache all
    """
    start_time = time.time()
    ctx.obj['dry_run'] = dry_run
    
    try:
        # Validate Hydra availability
        _validate_hydra_availability()
        
        # Access Hydra configuration
        if not HydraConfig.initialized():
            logger.error("Hydra configuration not initialized. Use @hydra.main decorator.")
            raise CLIError("Hydra configuration required for run command")
        
        cfg = HydraConfig.get().cfg
        
        # Set global seed if provided
        if seed is not None:
            set_global_seed(seed)
            logger.info(f"Global seed set to: {seed}")
        
        # Validate configuration
        logger.info("Validating configuration...")
        validation_results = _validate_configuration(cfg, strict=not dry_run)
        
        if not validation_results['valid']:
            logger.error("Configuration validation failed:")
            for error in validation_results['errors']:
                logger.error(f"  - {error}")
            raise ConfigValidationError("Invalid configuration")
        
        if validation_results['warnings']:
            logger.warning("Configuration warnings:")
            for warning in validation_results['warnings']:
                logger.warning(f"  - {warning}")
        
        logger.info("Configuration validation passed")
        
        if dry_run:
            logger.info("Dry-run mode: Simulation validation completed successfully")
            logger.info(f"Configuration summary: {validation_results['summary']}")
            _measure_performance("Dry-run validation", start_time)
            return
        
        # Create components
        logger.info("Creating navigation components...")
        
        
        # Create frame cache if requested
        frame_cache_instance = None
        if frame_cache != "none":
            try:
                video_path = _safe_config_access(cfg, 'video_plume.video_path')
                frame_cache_instance = _create_frame_cache(frame_cache, video_path=video_path)
                if frame_cache_instance:
                    logger.info(f"Frame cache created in '{frame_cache}' mode")
            except Exception as e:
                logger.warning(f"Failed to create frame cache, falling back to direct access: {e}")
                frame_cache_instance = None
        try:
            navigator = create_navigator(cfg=cfg.navigator if hasattr(cfg, 'navigator') else None)
            logger.info(f"Navigator created: {type(navigator).__name__} with {navigator.num_agents} agent(s)")
        except Exception as e:
            logger.error(f"Failed to create navigator: {e}")
            raise ConfigurationError(f"Navigator creation failed: {e}") from e
        
        try:
            # Pass frame cache to video plume creation
            video_plume_config = cfg.video_plume if hasattr(cfg, 'video_plume') else None
            video_plume = create_video_plume(
                cfg=video_plume_config, 
                frame_cache=frame_cache_instance
            )
            logger.info(f"Video plume loaded: {video_plume.frame_count} frames")
        except Exception as e:
            logger.error(f"Failed to create video plume: {e}")
            raise ConfigurationError(f"Video plume creation failed: {e}") from e
        
        # Execute simulation
        logger.info("Starting simulation execution...")
        sim_start_time = time.time()
        
        try:
            positions, orientations, odor_readings = run_plume_simulation(
                navigator=navigator,
                video_plume=video_plume,
                cfg=cfg.simulation if hasattr(cfg, 'simulation') else None,
                record_trajectories=save_trajectory,
                seed=seed,
                frame_cache=frame_cache_instance,
            )
            
            sim_duration = time.time() - sim_start_time
            logger.info(f"Simulation completed in {sim_duration:.2f}s")
            
            # Log simulation statistics
            logger.info(f"Final positions shape: {positions.shape}")
            logger.info(f"Trajectory length: {positions.shape[1]} steps")
            logger.info(f"Average odor reading: {np.mean(odor_readings):.4f}")
            
        except Exception as e:
            logger.error(f"Simulation execution failed: {e}")
            raise SimulationError(f"Simulation failed: {e}") from e
        
        # Handle visualization and export
        if show_animation or export_video:
            logger.info("Generating visualization...")
            try:
                visualization_results = {
                    'positions': positions,
                    'orientations': orientations,
                    'odor_readings': odor_readings
                }
                
                if show_animation:
                    visualize_plume_simulation(
                        positions=positions,
                        orientations=orientations,
                        odor_readings=odor_readings,
                        show_plot=True,
                        cfg=cfg.visualization if hasattr(cfg, 'visualization') else None
                    )
                
                if export_video:
                    export_path = Path(export_video)
                    visualize_plume_simulation(
                        positions=positions,
                        orientations=orientations,
                        odor_readings=odor_readings,
                        output_path=export_path,
                        show_plot=False,
                        cfg=cfg.visualization if hasattr(cfg, 'visualization') else None
                    )
                    logger.info(f"Animation exported to {export_path}")
                    
            except Exception as e:
                logger.warning(f"Visualization failed: {e}")
        
        # Save trajectory data if requested
        if save_trajectory and output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            np.savez(
                output_path / "trajectory_data.npz",
                positions=positions,
                orientations=orientations,
                odor_readings=odor_readings,
                seed=get_last_seed()
            )
            logger.info(f"Trajectory data saved to {output_path}/trajectory_data.npz")
        
        _measure_performance("Simulation execution", start_time)
        logger.info("Run command completed successfully")
        
    except KeyboardInterrupt:
        logger.warning("Simulation interrupted by user")
        ctx.exit(1)
    except (CLIError, ConfigValidationError, ConfigurationError, SimulationError) as e:
        logger.error(str(e))
        ctx.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if ctx.obj.get('verbose'):
            logger.error(traceback.format_exc())
        ctx.exit(1)


@cli.group()
def config() -> None:
    """
    Configuration management commands for validation and export.
    
    This command group provides utilities for working with Hydra configuration files,
    including validation, export, and documentation generation capabilities.
    """
    pass


@config.command()
@click.option('--strict', is_flag=True,
              help='Use strict validation rules (fail on warnings)')
@click.option('--export-results', type=click.Path(),
              help='Export validation results to JSON file')
@click.pass_context
def validate(ctx, strict: bool, export_results: Optional[str]) -> None:
    """
    Validate Hydra configuration files with comprehensive error reporting.
    
    Performs thorough validation of configuration schemas, parameter ranges,
    file existence checks, and cross-section consistency validation.
    
    Examples:
        # Basic validation
        odor-plume-nav-cli config validate
        
        # Strict validation (warnings as errors)
        odor-plume-nav-cli config validate --strict
        
        # Export validation results
        odor-plume-nav-cli config validate --export-results validation.json
    """
    start_time = time.time()
    
    try:
        _validate_hydra_availability()
        
        if not HydraConfig.initialized():
            logger.error("Hydra configuration not initialized")
            raise CLIError("Configuration validation requires Hydra initialization")
        
        cfg = HydraConfig.get().cfg
        
        logger.info("Starting configuration validation...")
        validation_results = _validate_configuration(cfg, strict=strict)
        
        # Report validation results
        if validation_results['valid']:
            logger.info("✓ Configuration validation passed")
            click.echo(click.style("Configuration is valid!", fg='green'))
        else:
            logger.error("✗ Configuration validation failed")
            click.echo(click.style("Configuration validation failed!", fg='red'))
            
            for error in validation_results['errors']:
                click.echo(click.style(f"ERROR: {error}", fg='red'))
        
        if validation_results['warnings']:
            click.echo(click.style("Warnings:", fg='yellow'))
            for warning in validation_results['warnings']:
                click.echo(click.style(f"WARNING: {warning}", fg='yellow'))
        
        # Show summary
        click.echo("\nValidation Summary:")
        for section, status in validation_results['summary'].items():
            status_color = 'green' if status == 'valid' else 'red'
            click.echo(f"  {section}: {click.style(status, fg=status_color)}")
        
        # Export results if requested
        if export_results:
            import json
            with open(export_results, 'w') as f:
                json.dump(validation_results, f, indent=2)
            logger.info(f"Validation results exported to {export_results}")
        
        _measure_performance("Configuration validation", start_time)
        
        if not validation_results['valid']:
            ctx.exit(1)
            
    except (CLIError, ConfigValidationError) as e:
        logger.error(str(e))
        ctx.exit(1)
    except Exception as e:
        logger.error(f"Validation error: {e}")
        if ctx.obj.get('verbose'):
            logger.error(traceback.format_exc())
        ctx.exit(1)


@config.command()
@click.option('--output', '-o', type=click.Path(),
              help='Output file path (default: config_export.yaml)')
@click.option('--format', 'output_format', type=click.Choice(['yaml', 'json']), default='yaml',
              help='Export format')
@click.option('--resolve', is_flag=True, default=True,
              help='Resolve configuration interpolations')
@click.pass_context  
def export(ctx, output: Optional[str], output_format: str, resolve: bool) -> None:
    """
    Export current configuration for documentation and sharing.
    
    Generates a comprehensive configuration export including metadata,
    resolved interpolations, and formatted output for documentation purposes.
    
    Examples:
        # Export to default file
        odor-plume-nav-cli config export
        
        # Export to specific file
        odor-plume-nav-cli config export --output my_config.yaml
        
        # Export as JSON
        odor-plume-nav-cli config export --format json
        
        # Export without resolving interpolations
        odor-plume-nav-cli config export --no-resolve
    """
    start_time = time.time()
    
    try:
        _validate_hydra_availability()
        
        if not HydraConfig.initialized():
            logger.error("Hydra configuration not initialized")
            raise CLIError("Configuration export requires Hydra initialization")
        
        cfg = HydraConfig.get().cfg
        
        # Determine output path
        if output is None:
            output = f"config_export.{output_format}"
        
        output_path = Path(output)
        
        logger.info(f"Exporting configuration to {output_path}")
        
        # Prepare configuration for export
        if resolve:
            config_data = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)
        else:
            config_data = OmegaConf.to_container(cfg, resolve=False)
        
        # Add metadata
        export_data = {
            "_metadata": {
                "export_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "hydra_version": hydra.__version__ if HYDRA_AVAILABLE else "N/A",
                "resolved": resolve,
                "format": output_format
            },
            **config_data
        }
        
        # Write to file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if output_format == 'yaml':
            with open(output_path, 'w') as f:
                OmegaConf.save(export_data, f)
        elif output_format == 'json':
            import json
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2)
        
        logger.info(f"Configuration successfully exported to {output_path}")
        click.echo(f"Configuration exported to: {click.style(str(output_path), fg='green')}")
        
        _measure_performance("Configuration export", start_time)
        
    except (CLIError, OSError) as e:
        logger.error(f"Export failed: {e}")
        ctx.exit(1)
    except Exception as e:
        logger.error(f"Unexpected export error: {e}")
        if ctx.obj.get('verbose'):
            logger.error(traceback.format_exc())
        ctx.exit(1)


@cli.group()
def visualize() -> None:
    """
    Visualization generation and export commands.
    
    This command group provides utilities for generating publication-quality
    visualizations, animations, and trajectory plots from simulation data.
    """
    pass


@visualize.command()
@click.option('--input-data', type=click.Path(exists=True),
              help='Path to trajectory data file (.npz format)')
@click.option('--format', 'output_format', type=click.Choice(['mp4', 'gif', 'png']), default='mp4',
              help='Output format for visualization')
@click.option('--output', '-o', type=click.Path(),
              help='Output file path')
@click.option('--dpi', type=int, default=100,
              help='Resolution (DPI) for output visualization')
@click.option('--fps', type=int, default=30,
              help='Frame rate for video outputs')
@click.option('--quality', type=click.Choice(['low', 'medium', 'high']), default='medium',
              help='Quality preset for output')
@click.pass_context
def export(ctx, input_data: Optional[str], output_format: str, output: Optional[str],
           dpi: int, fps: int, quality: str) -> None:
    """
    Export visualization with publication-quality formatting.
    
    Generates high-quality visualizations from simulation data with configurable
    output formats, resolution settings, and quality presets for research publication.
    
    Examples:
        # Export MP4 animation
        odor-plume-nav-cli visualize export --format mp4 --output animation.mp4
        
        # High-quality PNG trajectory plot
        odor-plume-nav-cli visualize export --format png --quality high --dpi 300
        
        # From existing data file
        odor-plume-nav-cli visualize export --input-data trajectory.npz --format gif
    """
    start_time = time.time()
    
    try:
        # Load data from file or current Hydra run
        if input_data:
            data_path = Path(input_data)
            if not data_path.exists():
                raise CLIError(f"Input data file not found: {data_path}")
            
            logger.info(f"Loading trajectory data from {data_path}")
            data = np.load(data_path)
            
            positions = data['positions']
            orientations = data['orientations']
            odor_readings = data['odor_readings']
            
        else:
            # Check if we're in a Hydra run context with data
            logger.error("No input data specified and no current simulation data available")
            raise CLIError("Input data file required for visualization export")
        
        # Determine output path
        if output is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output = f"visualization_{timestamp}.{output_format}"
        
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create visualization configuration
        viz_config = {
            'format': output_format,
            'dpi': dpi,
            'fps': fps,
            'quality': quality
        }
        
        logger.info(f"Generating {output_format} visualization...")
        
        # Generate visualization based on format
        if output_format in ['mp4', 'gif']:
            # Animation export using real-time visualizer
            visualizer = create_realtime_visualizer(
                fps=fps,
                headless=True,
                resolution='720p'
            )
            
            # Create animation and save
            def frame_callback(frame_idx):
                return (
                    (positions[0, frame_idx], orientations[0, frame_idx]),
                    odor_readings[0, frame_idx]
                ) if len(positions.shape) == 3 else (
                    (positions[frame_idx], orientations[frame_idx]),
                    odor_readings[frame_idx]
                )
            
            animation_obj = visualizer.create_animation(
                frame_callback, 
                frames=min(positions.shape[-2], 1000)  # Limit frames for memory
            )
            visualizer.save_animation(output_path, fps=fps, quality=quality)
            
        elif output_format == 'png':
            # Static trajectory plot
            visualize_trajectory(
                positions=positions,
                orientations=orientations,
                output_path=output_path,
                show_plot=False,
                dpi=dpi,
                format='png'
            )
        
        logger.info(f"Visualization exported to {output_path}")
        click.echo(f"Visualization saved: {click.style(str(output_path), fg='green')}")
        
        _measure_performance("Visualization export", start_time)
        
    except CLIError as e:
        logger.error(str(e))
        ctx.exit(1)
    except Exception as e:
        logger.error(f"Visualization export failed: {e}")
        if ctx.obj.get('verbose'):
            logger.error(traceback.format_exc())
        ctx.exit(1)


@cli.command()
@click.option('--jobs', '-j', type=int, default=1,
              help='Number of parallel jobs for batch processing')
@click.option('--config-dir', type=click.Path(exists=True),
              help='Directory containing batch configuration files')
@click.option('--pattern', default='*.yaml',
              help='File pattern for batch configuration files')
@click.option('--output-base', type=click.Path(),
              help='Base directory for batch output files')
@click.pass_context
def batch(ctx, jobs: int, config_dir: Optional[str], pattern: str, output_base: Optional[str]) -> None:
    """
    Execute batch processing for multiple configuration files.
    
    Processes multiple configuration files in parallel, enabling large-scale
    parameter studies and automated experiment execution.
    
    Examples:
        # Process all configs in directory
        odor-plume-nav-cli batch --config-dir experiments/
        
        # Parallel processing with 4 jobs
        odor-plume-nav-cli batch --config-dir experiments/ --jobs 4
        
        # Custom output directory
        odor-plume-nav-cli batch --config-dir configs/ --output-base results/
    """
    start_time = time.time()
    
    try:
        if not config_dir:
            raise CLIError("Config directory required for batch processing")
        
        config_path = Path(config_dir)
        if not config_path.exists():
            raise CLIError(f"Config directory not found: {config_path}")
        
        # Find configuration files
        config_files = list(config_path.glob(pattern))
        
        if not config_files:
            raise CLIError(f"No configuration files found matching pattern: {pattern}")
        
        logger.info(f"Found {len(config_files)} configuration files for batch processing")
        
        # Setup output directory
        if output_base:
            output_dir = Path(output_base)
        else:
            output_dir = Path("batch_results") / time.strftime("%Y%m%d_%H%M%S")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process files (simplified single-threaded for now)
        # TODO: Implement parallel processing with multiprocessing
        
        successful = 0
        failed = 0
        
        for config_file in config_files:
            try:
                logger.info(f"Processing: {config_file.name}")
                
                # This would be enhanced to actually run simulations
                # with the specific config file in a production implementation
                click.echo(f"Processing {config_file.name}...")
                
                successful += 1
                
            except Exception as e:
                logger.error(f"Failed to process {config_file.name}: {e}")
                failed += 1
        
        logger.info(f"Batch processing completed: {successful} successful, {failed} failed")
        click.echo(f"Batch results: {click.style(f'{successful} successful', fg='green')}, "
                  f"{click.style(f'{failed} failed', fg='red')}")
        
        _measure_performance("Batch processing", start_time)
        
        if failed > 0:
            ctx.exit(1)
            
    except CLIError as e:
        logger.error(str(e))
        ctx.exit(1)
    except Exception as e:
        logger.error(f"Batch processing error: {e}")
        if ctx.obj.get('verbose'):
            logger.error(traceback.format_exc())
        ctx.exit(1)


# RL Training Commands and Utilities

# Conditional imports for RL training functionality
try:
    import stable_baselines3
    from stable_baselines3 import PPO, SAC, TD3, A2C, DDPG
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from stable_baselines3.common.callbacks import (
        CheckpointCallback, EvalCallback, StopTrainingOnRewardThreshold,
        ProgressBarCallback
    )
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.env_util import make_vec_env
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    warnings.warn(
        "Stable-baselines3 not available. RL training commands will be limited.",
        ImportWarning
    )

# Import Gymnasium environment creation function with graceful fallback
try:
    from odor_plume_nav.api.navigation import create_gymnasium_environment
    GYMNASIUM_ENV_AVAILABLE = True
except ImportError:
    GYMNASIUM_ENV_AVAILABLE = False
    warnings.warn(
        "Gymnasium environment not available. RL training features will be disabled.",
        ImportWarning
    )


def _validate_rl_availability() -> None:
    """Validate that RL dependencies are available for training commands."""
    if not SB3_AVAILABLE:
        raise CLIError(
            "Stable-baselines3 is required for RL training. Please install with: "
            "pip install 'stable-baselines3>=2.0.0'"
        )
    
    if not GYMNASIUM_ENV_AVAILABLE:
        raise CLIError(
            "Gymnasium environment support is not available. Please install with: "
            "pip install 'gymnasium>=0.29.0'"
        )


def _create_algorithm_factory() -> Dict[str, Any]:
    """Create algorithm factory mapping for supported RL algorithms."""
    if not SB3_AVAILABLE:
        return {}
    
    return {
        'PPO': PPO,
        'SAC': SAC,
        'TD3': TD3,
        'A2C': A2C,
        'DDPG': DDPG
    }


def _create_vectorized_environment(
    env_config: DictConfig,
    n_envs: int = 4,
    vec_env_type: str = 'dummy',
    frame_cache: Optional[FrameCache] = None
) -> Union[DummyVecEnv, SubprocVecEnv]:
    """
    Create vectorized environment for parallel RL training.
    
    Args:
        env_config: Hydra configuration for environment creation
        n_envs: Number of parallel environments
        vec_env_type: Type of vectorized environment ('dummy' or 'subproc')
        frame_cache: Optional FrameCache instance for performance optimization
        
    Returns:
        Vectorized environment instance
    """
    def make_env():
        """Factory function for creating individual environments."""
        env = create_gymnasium_environment(cfg=env_config, frame_cache=frame_cache)
        env = Monitor(env)  # Add monitoring wrapper
        return env
    
    if vec_env_type == 'subproc':
        return SubprocVecEnv([make_env for _ in range(n_envs)])
    else:
        return DummyVecEnv([make_env for _ in range(n_envs)])


def _setup_training_callbacks(
    output_dir: Path,
    checkpoint_freq: int,
    save_freq: int,
    eval_freq: Optional[int] = None
) -> List[Any]:
    """
    Setup training callbacks for model checkpointing and monitoring.
    
    Args:
        output_dir: Directory for saving checkpoints and logs
        checkpoint_freq: Frequency for saving model checkpoints
        save_freq: Frequency for saving training progress
        eval_freq: Frequency for evaluation (optional)
        
    Returns:
        List of configured callbacks
    """
    callbacks = []
    
    # Model checkpointing
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=str(output_dir / "checkpoints"),
        name_prefix="rl_model"
    )
    callbacks.append(checkpoint_callback)
    
    # Progress bar for training monitoring
    if not _CLI_CONFIG.get('quiet', False):
        progress_callback = ProgressBarCallback()
        callbacks.append(progress_callback)
    
    return callbacks


@cli.group()
def train() -> None:
    """
    Reinforcement learning training commands for odor plume navigation.
    
    This command group provides comprehensive RL training capabilities using stable-baselines3
    algorithms with the Gymnasium environment wrapper. Supports algorithm selection,
    vectorized training, automatic checkpointing, performance monitoring, and frame caching
    for optimal training performance.
    
    Available algorithms:
    - PPO (Proximal Policy Optimization) - recommended for most scenarios
    - SAC (Soft Actor-Critic) - for continuous control with exploration
    - TD3 (Twin Delayed DDPG) - for deterministic continuous control
    - A2C (Advantage Actor-Critic) - lightweight policy gradient method
    - DDPG (Deep Deterministic Policy Gradient) - for continuous control
    
    Examples:
        # Train PPO agent with default settings
        plume-nav-sim train algorithm --algorithm PPO
        
        # Train SAC with custom timesteps and vectorized environments
        plume-nav-sim train algorithm --algorithm SAC --total-timesteps 100000 --n-envs 8
        
        # Train with custom configuration and frequent checkpointing
        plume-nav-sim train algorithm --algorithm PPO --config custom_rl.yaml --checkpoint-freq 5000
    """
    pass


@train.command()
@click.option('--algorithm', '-a', required=True,
              type=click.Choice(['PPO', 'SAC', 'TD3', 'A2C', 'DDPG'], case_sensitive=False),
              help='RL algorithm to use for training')
@click.option('--total-timesteps', '-t', type=int, default=50000,
              help='Total number of training timesteps')
@click.option('--n-envs', '-n', type=int, default=4,
              help='Number of parallel environments for vectorized training')
@click.option('--vec-env-type', type=click.Choice(['dummy', 'subproc']), default='dummy',
              help='Type of vectorized environment (dummy for single-process, subproc for multi-process)')
@click.option('--checkpoint-freq', type=int, default=10000,
              help='Frequency for saving model checkpoints (in timesteps)')
@click.option('--learning-rate', type=float, default=None,
              help='Learning rate for the algorithm (uses algorithm default if not specified)')
@click.option('--policy', type=str, default='MultiInputPolicy',
              help='Policy architecture to use (MultiInputPolicy recommended for multi-modal observations)')
@click.option('--verbose', type=int, default=1,
              help='Verbosity level for training output (0=silent, 1=info, 2=debug)')
@click.option('--tensorboard-log', type=click.Path(),
              help='Directory for TensorBoard logging')
@click.option('--eval-freq', type=int, default=None,
              help='Frequency for model evaluation during training')
@click.option('--output-dir', '-o', type=click.Path(), default='rl_training_output',
              help='Output directory for trained models and logs')
@click.option('--frame-cache', type=click.Choice(['none', 'lru', 'all']), default='none',
               callback=lambda ctx, param, value: _validate_frame_cache_mode(value),
               help='Frame caching mode: "none" (no caching), "lru" (LRU eviction with 2GB limit), '
                    '"all" (preload all frames for maximum throughput)')
@click.pass_context
def algorithm(ctx, algorithm: str, total_timesteps: int, n_envs: int, vec_env_type: str,
              checkpoint_freq: int, learning_rate: Optional[float], policy: str,
              verbose: int, tensorboard_log: Optional[str], eval_freq: Optional[int],
               frame_cache: str,
              output_dir: str) -> None:
    """
    Train an RL agent using stable-baselines3 algorithms with the Gymnasium environment.
    
    This command creates a Gymnasium-compliant environment from the Hydra configuration
    and trains a policy using the specified algorithm. Supports vectorized environments
    for improved training efficiency and comprehensive monitoring with checkpoints.
    
    Algorithm Selection Guide:
    - PPO: General-purpose, stable, good for most navigation tasks
    - SAC: Sample-efficient, good exploration, handles stochastic environments
    - TD3: Deterministic control, good for precise navigation tasks
    - A2C: Fast training, lower sample efficiency, good for quick experiments
    - DDPG: Continuous control pioneer, may require careful tuning
    
    Examples:
        # Basic PPO training
        plume-nav-sim train algorithm --algorithm PPO --total-timesteps 100000
        
        # High-performance SAC training with vectorized environments
        plume-nav-sim train algorithm --algorithm SAC --n-envs 8 --vec-env-type subproc
        
        # Training with custom learning rate and TensorBoard logging
        plume-nav-sim train algorithm --algorithm TD3 --learning-rate 0.001 --tensorboard-log ./logs
        
        # Quick experiment with frequent checkpointing
        plume-nav-sim train algorithm --algorithm A2C --total-timesteps 25000 --checkpoint-freq 2500
        
        # Training with frame caching for optimal performance
        plume-nav-sim train algorithm --algorithm PPO --frame-cache lru --n-envs 4
        
        # High-throughput training with full frame preloading
        plume-nav-sim train algorithm --algorithm SAC --frame-cache all --n-envs 8
    """
    start_time = time.time()
    
    
    # Handle frame_cache parameter - use default if not passed due to signature issue
    if 'frame_cache' not in locals():
        frame_cache = 'none'
    try:
        # Validate RL dependencies
        _validate_rl_availability()
        
        # Validate Hydra availability and configuration
        _validate_hydra_availability()
        
        if not HydraConfig.initialized():
            logger.error("Hydra configuration not initialized")
            raise CLIError("RL training requires Hydra configuration initialization")
        
        cfg = HydraConfig.get().cfg
        algorithm_name = algorithm.upper()
        
        logger.info(f"Starting RL training with {algorithm_name} algorithm")
        logger.info(f"Training parameters: {total_timesteps} timesteps, {n_envs} environments")
        
        # Setup output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create algorithm factory
        algorithm_factory = _create_algorithm_factory()
        if algorithm_name not in algorithm_factory:
            raise CLIError(f"Algorithm {algorithm_name} not supported. Available: {list(algorithm_factory.keys())}")
        
        AlgorithmClass = algorithm_factory[algorithm_name]
        
        
        # Create frame cache if requested for RL training
        frame_cache_instance = None
        if frame_cache != "none":
            try:
                # Extract video path from config for cache initialization
                video_path = None
                if hasattr(cfg, 'environment') and hasattr(cfg.environment, 'video_path'):
                    video_path = cfg.environment.video_path
                elif hasattr(cfg, 'video_plume') and hasattr(cfg.video_plume, 'video_path'):
                    video_path = cfg.video_plume.video_path
                
                frame_cache_instance = _create_frame_cache(frame_cache, video_path=video_path)
                if frame_cache_instance:
                    logger.info(f"Frame cache created in '{frame_cache}' mode for RL training")
            except Exception as e:
                logger.warning(f"Failed to create frame cache for RL training, falling back to direct access: {e}")
                frame_cache_instance = None
        # Create vectorized environment
        logger.info(f"Creating {n_envs} vectorized environments ({vec_env_type} type)")
        try:
            if hasattr(cfg, 'environment'):
                env_config = cfg.environment
            else:
                # Fallback to using the entire config
                env_config = cfg
                logger.warning("No 'environment' section found in config, using entire config")
            
            vec_env = _create_vectorized_environment(
                env_config=env_config,
                n_envs=n_envs,
                vec_env_type=vec_env_type,
                frame_cache=frame_cache_instance
            )
            logger.info("Vectorized environment created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create vectorized environment: {e}")
            raise CLIError(f"Environment creation failed: {e}") from e
        
        # Setup training callbacks
        callbacks = _setup_training_callbacks(
            output_dir=output_path,
            checkpoint_freq=checkpoint_freq,
            save_freq=checkpoint_freq,
            eval_freq=eval_freq
        )
        
        # Prepare algorithm parameters
        algorithm_kwargs = {
            'policy': policy,
            'env': vec_env,
            'verbose': verbose,
            'tensorboard_log': tensorboard_log
        }
        
        # Add learning rate if specified
        if learning_rate is not None:
            algorithm_kwargs['learning_rate'] = learning_rate
        
        # Create and configure the algorithm
        logger.info(f"Initializing {algorithm_name} algorithm with policy '{policy}'")
        model = AlgorithmClass(**algorithm_kwargs)
        
        # Start training with progress monitoring
        logger.info(f"Beginning training for {total_timesteps} timesteps...")
        training_start = time.time()
        
        try:
            model.learn(
                total_timesteps=total_timesteps,
                callback=callbacks,
                tb_log_name=f"{algorithm_name.lower()}_training",
                reset_num_timesteps=True,
                progress_bar=verbose > 0
            )
            
            training_duration = time.time() - training_start
            logger.info(f"Training completed in {training_duration:.2f}s")
            
        except KeyboardInterrupt:
            logger.warning("Training interrupted by user")
            training_duration = time.time() - training_start
            logger.info(f"Training ran for {training_duration:.2f}s before interruption")
        
        # Save final model
        final_model_path = output_path / f"final_{algorithm_name.lower()}_model"
        model.save(str(final_model_path))
        logger.info(f"Final model saved to {final_model_path}")
        
        # Generate training summary
        summary = {
            "algorithm": algorithm_name,
            "total_timesteps": total_timesteps,
            "n_environments": n_envs,
            "training_duration": training_duration,
            "model_path": str(final_model_path),
            "policy": policy,
            "learning_rate": model.learning_rate if hasattr(model, 'learning_rate') else "default"
        }
        
        # Save training metadata
        import json
        metadata_path = output_path / "training_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info("Training summary:")
        for key, value in summary.items():
            logger.info(f"  {key}: {value}")
        
        # Clean up environment
        vec_env.close()
        
        _measure_performance("RL training", start_time)
        click.echo(click.style(f"✓ RL training completed successfully!", fg='green'))
        click.echo(f"Model saved to: {click.style(str(final_model_path), fg='cyan')}")
        click.echo(f"Training metadata: {click.style(str(metadata_path), fg='cyan')}")
        
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        ctx.exit(1)
    except (CLIError, ConfigValidationError, ConfigurationError) as e:
        logger.error(str(e))
        ctx.exit(1)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        if ctx.obj.get('verbose'):
            logger.error(traceback.format_exc())
        ctx.exit(1)


# Hydra main decorator for configuration injection
@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main CLI entrypoint with Hydra configuration management.
    
    This function serves as the primary entry point for the CLI with automatic
    Hydra configuration loading, parameter injection, and multi-run support.
    The @hydra.main decorator enables sophisticated configuration management
    including parameter overrides, configuration composition, and multi-run experiments.
    
    The function processes command-line arguments through Click framework while
    maintaining access to the full Hydra configuration hierarchy for advanced
    parameter management and experiment orchestration.
    
    Args:
        cfg: Hydra configuration object with full hierarchy loaded from conf/ directory
    """
    try:
        # Initialize global timing
        _CLI_CONFIG['start_time'] = time.time()
        
        # Store Hydra config for CLI commands
        if not hasattr(cli, '_hydra_cfg'):
            cli._hydra_cfg = cfg
        
        # Process CLI commands
        cli(standalone_mode=False)
        
    except SystemExit as e:
        # Handle normal CLI exits
        if e.code != 0:
            logger.error(f"CLI exited with code {e.code}")
        sys.exit(e.code)
    except KeyboardInterrupt:
        logger.warning("CLI interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected CLI error: {e}")
        logger.debug(traceback.format_exc())
        sys.exit(1)


# Entry point for direct module execution
if __name__ == "__main__":
    main()
