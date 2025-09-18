"""
Default configuration module for the plume_nav_sim package.

This module defines the core data model for all simulation configurations,
including the main `CompleteConfig` class and its components. It provides a
centralized, validated structure for environment, performance, and system settings.
"""

# External imports
from dataclasses import dataclass, field
import copy
from typing import Union, Optional, Dict, List, Tuple, Any

# Internal imports from plume_nav_sim core modules
from plume_nav_sim.core.constants import (
    DEFAULT_GRID_SIZE, DEFAULT_SOURCE_LOCATION, DEFAULT_PLUME_SIGMA, DEFAULT_GOAL_RADIUS,
    PERFORMANCE_TARGET_STEP_LATENCY_MS, MEMORY_LIMIT_TOTAL_MB
)
from plume_nav_sim.utils.exceptions import ConfigurationError, ResourceError
from plume_nav_sim.core.types import EnvironmentConfig as CoreEnvironmentConfig, create_environment_config

__all__ = [
    'CompleteConfig',
    'EnvironmentConfig',
    'PerformanceConfig',
    'RenderConfig',
    'PlumeConfig',
    'get_complete_default_config',
    'get_default_environment_config',
    'get_default_plume_config',
    'get_default_render_config',
    'get_default_performance_config',
    'validate_configuration_compatibility',
    'merge_configurations',
    'create_config_from_dict'
]


@dataclass
class CompleteConfig:
    """Defines the complete, unified configuration for a simulation environment.

    This data class aggregates all parameters required to define and run a simulation,
    including environment geometry, plume characteristics, performance targets, and
    reproducibility settings. It serves as the single source of truth for an
    environment's configuration.
    """

    # Core configuration components
    grid_size: Tuple[int, int] = DEFAULT_GRID_SIZE
    source_location: Tuple[int, int] = DEFAULT_SOURCE_LOCATION
    plume_sigma: float = DEFAULT_PLUME_SIGMA
    goal_radius: float = DEFAULT_GOAL_RADIUS
    max_steps: int = 1000
    render_mode: str = 'rgb_array'

    # Performance and system configuration
    enable_validation: bool = True
    enable_performance_monitoring: bool = False
    memory_limit_mb: float = MEMORY_LIMIT_TOTAL_MB
    step_latency_target_ms: float = PERFORMANCE_TARGET_STEP_LATENCY_MS

    # Testing and reproducibility configuration
    random_seed: Optional[int] = None
    deterministic_mode: bool = False

    # Advanced configuration options
    advanced_options: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def clone_with_overrides(self, overrides: Optional[Dict[str, Any]] = None, **keyword_overrides: Any) -> 'CompleteConfig':
        """Create deep copy of configuration with parameter overrides applied.

        Args:
            overrides: Dictionary of parameter overrides to apply

        Returns:
            CompleteConfig: New configuration instance with overrides applied
        """
        new_config = copy.deepcopy(self)
        combined_overrides: Dict[str, Any] = {}
        if overrides:
            combined_overrides.update(overrides)
        if keyword_overrides:
            combined_overrides.update(keyword_overrides)

        for key, value in combined_overrides.items():
            if hasattr(new_config, key):
                setattr(new_config, key, value)
            else:
                new_config.advanced_options[key] = value
        new_config.metadata['last_modified'] = 'clone_with_overrides'
        new_config.metadata['override_keys'] = list(combined_overrides.keys())
        return new_config

    def validate_all(self, strict_mode: bool = False) -> bool:
        """Comprehensive validation of all configuration parameters.

        Args:
            strict_mode: Enable strict validation with additional checks

        Returns:
            bool: True if configuration is valid, raises ConfigurationError if invalid

        Raises:
            ConfigurationError: If configuration validation fails
        """
        validation_errors = []
        if not isinstance(self.grid_size, (tuple, list)) or len(self.grid_size) != 2:
            validation_errors.append("Grid size must be tuple[int, int]")
        elif not all(isinstance(x, int) and x > 0 for x in self.grid_size):
            validation_errors.append("Grid size must contain positive integers")
        if not isinstance(self.source_location, (tuple, list)) or len(self.source_location) != 2:
            validation_errors.append("Source location must be tuple[int, int]")
        elif (self.source_location[0] >= self.grid_size[0] or
              self.source_location[1] >= self.grid_size[1]):
            validation_errors.append("Source location must be within grid bounds")
        if not isinstance(self.plume_sigma, (int, float)) or self.plume_sigma <= 0:
            validation_errors.append("Plume sigma must be positive number")
        if not isinstance(self.max_steps, int) or self.max_steps <= 0:
            validation_errors.append("Max steps must be positive integer")
        if self.render_mode not in ['rgb_array', 'human']:
            validation_errors.append("Render mode must be 'rgb_array' or 'human'")
        if strict_mode:
            grid_cells = self.grid_size[0] * self.grid_size[1]
            estimated_memory = (grid_cells * 4) / (1024 * 1024)
            if estimated_memory > self.memory_limit_mb:
                validation_errors.append(f"Estimated memory {estimated_memory:.1f}MB exceeds limit {self.memory_limit_mb}MB")
        if validation_errors:
            raise ConfigurationError(
                f"Configuration validation failed: {'; '.join(validation_errors)}",
                config_parameter="complete_config",
                invalid_value="multiple_parameters"
            )
        return True

# Re-export canonical EnvironmentConfig from core.types to avoid duplication
EnvironmentConfig = CoreEnvironmentConfig

# Backwards compatible alias used by historical tests
PlumeConfig = CompleteConfig

@dataclass
class PerformanceConfig:
    """Configuration class for performance targets."""

    step_latency_target_ms: float = PERFORMANCE_TARGET_STEP_LATENCY_MS
    render_latency_target_ms: float = 5.0
    memory_limit_mb: float = MEMORY_LIMIT_TOTAL_MB
    enable_profiling: bool = False

    def get_target_for_operation(self, operation_name: str) -> float:
        """Get performance target for specific operation."""
        operation_targets = {
            'step': self.step_latency_target_ms,
            'render': self.render_latency_target_ms,
            'reset': 10.0,
            'plume_generation': 10.0
        }
        return operation_targets.get(operation_name, 1.0)


@dataclass
class RenderConfig:
    """Minimal renderer configuration used by the tests."""

    render_mode: str = 'rgb_array'
    allow_human_mode: bool = False

    def validate(self) -> bool:
        if self.render_mode not in ('rgb_array', 'human'):
            raise ConfigurationError(
                f"Unsupported render_mode: {self.render_mode}",
                config_parameter='render_mode',
                invalid_value=self.render_mode,
                expected_format="'rgb_array' or 'human'"
            )
        return True

def get_complete_default_config() -> CompleteConfig:
    """Factory function for creating a default complete configuration.

    Returns:
        CompleteConfig: A default complete configuration with standard parameters.
    """
    return CompleteConfig()


def get_default_environment_config() -> EnvironmentConfig:
    return EnvironmentConfig()


def get_default_plume_config() -> PlumeConfig:
    return PlumeConfig()


def get_default_render_config() -> RenderConfig:
    return RenderConfig()


def get_default_performance_config() -> PerformanceConfig:
    return PerformanceConfig()


def validate_configuration_compatibility(environment: EnvironmentConfig,
                                         render: RenderConfig) -> bool:
    if render.render_mode == 'human' and not environment.enable_rendering:
        raise ConfigurationError(
            'Human rendering requires enable_rendering=True',
            config_parameter='render_mode',
            invalid_value=render.render_mode,
            expected_format='enable_rendering must be True'
        )
    render.validate()
    return True


def merge_configurations(base: CompleteConfig, overrides: Dict[str, Any]) -> CompleteConfig:
    return base.clone_with_overrides(overrides)


def create_config_from_dict(config_dict: Dict[str, Any]) -> CompleteConfig:
    return CompleteConfig(**config_dict)
