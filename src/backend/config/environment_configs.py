"""
Environment-specific configuration module providing preset configurations, research scenarios,
performance benchmarks, and specialized environment setups for the plume_nav_sim system.

This module serves as the central registry for named environment configurations including
small-scale testing, research scenarios, performance benchmarks, and production-ready
configurations with intelligent parameter selection and validation.
"""

import copy  # >=3.10

# External imports
from dataclasses import dataclass, field  # >=3.10
from typing import Any, Dict, List, Optional, Tuple, Union  # >=3.10

# Internal imports from plume_nav_sim core
from plume_nav_sim.core.constants import (
    DEFAULT_GRID_SIZE,
    DEFAULT_MAX_STEPS,
    DEFAULT_PLUME_SIGMA,
    DEFAULT_SOURCE_LOCATION,
    TESTING_CONSTANTS,
)
from plume_nav_sim.core.geometry import Coordinates, GridSize
from plume_nav_sim.core.types import PlumeParameters

# Try to import from default_config - may not exist in early development phases
try:
    from .default_config import CompleteConfig, get_complete_default_config

    DEFAULT_CONFIG_AVAILABLE = True
except ImportError:
    # Handle case where default_config.py doesn't exist yet
    DEFAULT_CONFIG_AVAILABLE = False

    # Create minimal placeholder classes for development
    class CompleteConfig:
        """Placeholder for CompleteConfig when default_config.py not available."""

        def __init__(self, **kwargs):
            self.environment_config = kwargs
            self.grid_size = kwargs.get("grid_size", DEFAULT_GRID_SIZE)
            self.source_location = kwargs.get(
                "source_location", DEFAULT_SOURCE_LOCATION
            )
            self.plume_sigma = kwargs.get("plume_sigma", DEFAULT_PLUME_SIGMA)
            self.max_steps = kwargs.get("max_steps", DEFAULT_MAX_STEPS)

        def clone_with_overrides(self, **overrides):
            """Create copy with parameter overrides."""
            new_config = copy.deepcopy(self.environment_config)
            new_config.update(overrides)
            return CompleteConfig(**new_config)

        def validate_all(self):
            """Basic validation placeholder."""
            return True

    def get_complete_default_config():
        """Placeholder factory function."""
        return CompleteConfig(
            grid_size=DEFAULT_GRID_SIZE,
            source_location=DEFAULT_SOURCE_LOCATION,
            plume_sigma=DEFAULT_PLUME_SIGMA,
            max_steps=DEFAULT_MAX_STEPS,
            render_mode=RenderMode.RGB_ARRAY,
        )


@dataclass(frozen=True)
class PresetMetadata:
    """
    Data class containing comprehensive metadata for configuration presets including
    descriptions, requirements, use cases, and technical specifications for preset
    discovery and selection.

    This class provides essential information for researchers to understand and
    select appropriate environment configurations for their specific research needs.
    """

    name: str
    category: str
    description: str
    use_cases: List[str]
    technical_specs: Dict[str, Any]
    requirements: Dict[str, Any] = field(default_factory=dict)
    performance_characteristics: Dict[str, Any] = field(default_factory=dict)
    version: str = field(default="1.0.0")

    def __post_init__(self):
        """Initialize computed fields after dataclass creation."""
        # Set default requirements if not provided
        if not self.requirements:
            object.__setattr__(
                self,
                "requirements",
                {
                    "memory_mb": self._estimate_memory_requirements(),
                    "cpu_cores": 1,
                    "python_version": ">=3.10",
                },
            )

        # Set default performance characteristics if not provided
        if not self.performance_characteristics:
            object.__setattr__(
                self,
                "performance_characteristics",
                {
                    "step_latency_ms": self._estimate_step_latency(),
                    "reset_time_ms": self._estimate_reset_time(),
                    "rendering_time_ms": self._estimate_rendering_time(),
                },
            )

    def _estimate_memory_requirements(self) -> float:
        """Estimate memory requirements based on grid size."""
        grid_size = self.technical_specs.get("grid_size", DEFAULT_GRID_SIZE)
        if isinstance(grid_size, (list, tuple)):
            width, height = grid_size
            # Estimate: grid cells * 4 bytes (float32) + overhead
            return (width * height * 4) / (1024 * 1024) + 10  # MB
        return 50.0  # Default estimate

    def _estimate_step_latency(self) -> float:
        """Estimate step execution latency."""
        grid_size = self.technical_specs.get("grid_size", DEFAULT_GRID_SIZE)
        if isinstance(grid_size, (list, tuple)):
            total_cells = grid_size[0] * grid_size[1]
            # Base latency + scaling factor
            return 0.1 + (total_cells / 100000.0)  # ms
        return 1.0  # Default estimate

    def _estimate_reset_time(self) -> float:
        """Estimate environment reset time."""
        return self._estimate_step_latency() * 10  # Reset is typically ~10x step time

    def _estimate_rendering_time(self) -> float:
        """Estimate rendering time based on grid size."""
        grid_size = self.technical_specs.get("grid_size", DEFAULT_GRID_SIZE)
        if isinstance(grid_size, (list, tuple)):
            total_cells = grid_size[0] * grid_size[1]
            return 1.0 + (total_cells / 50000.0)  # ms
        return 5.0  # Default estimate

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts preset metadata to dictionary for serialization and external display.

        Returns:
            dict: Dictionary representation of preset metadata with all fields
        """
        return {
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "use_cases": self.use_cases.copy(),
            "technical_specs": self.technical_specs.copy(),
            "requirements": self.requirements.copy(),
            "performance_characteristics": self.performance_characteristics.copy(),
            "version": self.version,
        }

    def matches_filter(self, filter_criteria: Dict[str, Any]) -> bool:
        """
        Checks if preset metadata matches specified filter criteria for preset discovery.

        Args:
            filter_criteria: Dictionary with filter conditions

        Returns:
            bool: True if preset matches all filter criteria, False otherwise
        """
        # Check category filter
        if "category" in filter_criteria:
            if self.category.lower() != filter_criteria["category"].lower():
                return False

        # Check use case filters
        if "use_cases" in filter_criteria:
            required_use_cases = filter_criteria["use_cases"]
            if isinstance(required_use_cases, str):
                required_use_cases = [required_use_cases]

            if not any(
                uc.lower() in [existing.lower() for existing in self.use_cases]
                for uc in required_use_cases
            ):
                return False

        # Check technical specifications
        if "min_grid_size" in filter_criteria:
            grid_size = self.technical_specs.get("grid_size", (0, 0))
            min_size = filter_criteria["min_grid_size"]
            if grid_size[0] < min_size[0] or grid_size[1] < min_size[1]:
                return False

        # Check performance requirements
        if "max_memory_mb" in filter_criteria:
            memory_req = self.requirements.get("memory_mb", float("inf"))
            if memory_req > filter_criteria["max_memory_mb"]:
                return False

        return True

    def estimate_resource_requirements(self) -> Dict[str, Any]:
        """
        Estimates system resource requirements for preset configuration.

        Returns:
            dict: Resource requirement estimates including memory, CPU, and timing
        """
        grid_size = self.technical_specs.get("grid_size", DEFAULT_GRID_SIZE)

        # Calculate detailed resource estimates
        if isinstance(grid_size, (list, tuple)):
            width, height = grid_size
            total_cells = width * height
        else:
            total_cells = 16384  # Default 128x128

        return {
            "memory_requirements": {
                "base_mb": 10,
                "plume_field_mb": (total_cells * 4) / (1024 * 1024),
                "rendering_buffer_mb": 5,
                "total_estimated_mb": self.requirements.get("memory_mb", 50),
            },
            "cpu_requirements": {
                "min_cores": 1,
                "recommended_cores": 2,
                "cpu_intensity": "low" if total_cells < 50000 else "medium",
            },
            "timing_estimates": {
                "initialization_ms": self.performance_characteristics.get(
                    "reset_time_ms", 10
                ),
                "step_latency_ms": self.performance_characteristics.get(
                    "step_latency_ms", 1
                ),
                "render_time_ms": self.performance_characteristics.get(
                    "rendering_time_ms", 5
                ),
            },
        }


class ConfigurationRegistry:
    """
    Registry class for managing named configuration presets with discovery, validation,
    registration, and metadata management capabilities for centralized preset administration.

    This class serves as the central hub for all environment configuration management,
    providing thread-safe operations and comprehensive preset lifecycle management.
    """

    def __init__(self, registry_name: Optional[str] = None):
        """
        Initialize configuration registry with preset storage, metadata management,
        and category tracking.

        Args:
            registry_name: Optional name for the registry instance
        """
        self.name = registry_name or "EnvironmentConfigRegistry"
        self._presets: Dict[str, CompleteConfig] = {}
        self._metadata: Dict[str, PresetMetadata] = {}
        self._categories: set = set()
        self._registry_info = {
            "created_at": self._get_timestamp(),
            "version": "1.0.0",
            "preset_count": 0,
        }

        # Register built-in presets during initialization
        self._register_builtin_presets()

    def _get_timestamp(self) -> str:
        """Get current timestamp for registry operations."""
        import datetime

        return datetime.datetime.now().isoformat()

    def _register_builtin_presets(self):
        """Register built-in environment presets during registry initialization."""
        # Small-scale testing configuration (32x32)
        small_config = get_complete_default_config().clone_with_overrides(
            grid_size=(32, 32), source_location=(16, 16), max_steps=200
        )
        small_metadata = PresetMetadata(
            name="small_testing",
            category="testing",
            description="Small-scale testing environment (32×32) for rapid development and debugging",
            use_cases=["unit testing", "algorithm debugging", "rapid prototyping"],
            technical_specs={
                "grid_size": (32, 32),
                "source_location": (16, 16),
                "max_steps": 200,
                "plume_sigma": DEFAULT_PLUME_SIGMA,
            },
        )
        self.register_preset("small_testing", small_config, small_metadata)

        # Standard research configuration (128x128)
        standard_config = get_complete_default_config()
        standard_metadata = PresetMetadata(
            name="standard_research",
            category="research",
            description="Standard research environment (128×128) for algorithm development and comparison",
            use_cases=[
                "algorithm development",
                "performance comparison",
                "research studies",
            ],
            technical_specs={
                "grid_size": DEFAULT_GRID_SIZE,
                "source_location": DEFAULT_SOURCE_LOCATION,
                "max_steps": DEFAULT_MAX_STEPS,
                "plume_sigma": DEFAULT_PLUME_SIGMA,
            },
        )
        self.register_preset("standard_research", standard_config, standard_metadata)

        # Large-scale benchmark configuration (256x256)
        benchmark_config = get_complete_default_config().clone_with_overrides(
            grid_size=(256, 256), source_location=(128, 128), max_steps=2000
        )
        benchmark_metadata = PresetMetadata(
            name="large_benchmark",
            category="benchmark",
            description="Large-scale benchmark environment (256×256) for performance validation",
            use_cases=[
                "performance benchmarking",
                "scalability testing",
                "system validation",
            ],
            technical_specs={
                "grid_size": (256, 256),
                "source_location": (128, 128),
                "max_steps": 2000,
                "plume_sigma": DEFAULT_PLUME_SIGMA,
            },
        )
        self.register_preset("large_benchmark", benchmark_config, benchmark_metadata)

        # Fast testing configuration
        fast_config = get_complete_default_config().clone_with_overrides(
            grid_size=(16, 16), source_location=(8, 8), max_steps=50
        )
        fast_metadata = PresetMetadata(
            name="fast_testing",
            category="testing",
            description="Ultra-fast testing environment (16×16) for CI/CD and automated testing",
            use_cases=["automated testing", "CI/CD pipelines", "quick validation"],
            technical_specs={
                "grid_size": (16, 16),
                "source_location": (8, 8),
                "max_steps": 50,
                "plume_sigma": DEFAULT_PLUME_SIGMA,
            },
        )
        self.register_preset("fast_testing", fast_config, fast_metadata)

    def register_preset(
        self, preset_name: str, config: CompleteConfig, metadata: PresetMetadata
    ) -> bool:
        """
        Registers new preset configuration with metadata validation and uniqueness checking.

        Args:
            preset_name: Unique name for the preset
            config: Complete configuration object
            metadata: Preset metadata with descriptions and specifications

        Returns:
            bool: True if preset registered successfully, False if name conflicts exist

        Raises:
            ValueError: If preset name is invalid or configuration validation fails
        """
        # Validate preset name
        if not preset_name or not isinstance(preset_name, str):
            raise ValueError("Preset name must be a non-empty string")

        if preset_name in self._presets:
            return False  # Name conflict

        # Validate configuration
        if not config.validate_all():
            raise ValueError(
                f"Configuration validation failed for preset '{preset_name}'"
            )

        # Validate metadata consistency
        if metadata.name != preset_name:
            raise ValueError("Metadata name must match preset name")

        # Store preset and metadata
        self._presets[preset_name] = config
        self._metadata[preset_name] = metadata
        self._categories.add(metadata.category)

        # Update registry info
        self._registry_info["preset_count"] = len(self._presets)
        self._registry_info["last_updated"] = self._get_timestamp()

        return True

    def get_preset(self, preset_name: str) -> CompleteConfig:
        """
        Retrieves preset configuration by name with validation and cloning for safe use.

        Args:
            preset_name: Name of the preset to retrieve

        Returns:
            CompleteConfig: Deep copy of preset configuration for safe modification

        Raises:
            KeyError: If preset name not found in registry
        """
        if preset_name not in self._presets:
            available_presets = ", ".join(self._presets.keys())
            raise KeyError(
                f"Preset '{preset_name}' not found. Available presets: {available_presets}"
            )

        # Return deep copy to prevent modification of stored preset
        config = self._presets[preset_name]
        return copy.deepcopy(config)

    def list_presets(self, category_filter: Optional[str] = None) -> List[str]:
        """
        Returns list of all registered preset names with optional category filtering.

        Args:
            category_filter: Optional category to filter presets by

        Returns:
            List[str]: List of preset names matching filter criteria
        """
        preset_names = list(self._presets.keys())

        if category_filter:
            filtered_names = []
            for name in preset_names:
                metadata = self._metadata[name]
                if metadata.category.lower() == category_filter.lower():
                    filtered_names.append(name)
            preset_names = filtered_names

        return sorted(preset_names)

    def get_metadata(self, preset_name: str) -> PresetMetadata:
        """
        Retrieves preset metadata by name with comprehensive preset information.

        Args:
            preset_name: Name of the preset to get metadata for

        Returns:
            PresetMetadata: Metadata object containing preset information and specifications

        Raises:
            KeyError: If preset name not found in metadata registry
        """
        if preset_name not in self._metadata:
            available_presets = ", ".join(self._metadata.keys())
            raise KeyError(
                f"Metadata for preset '{preset_name}' not found. Available: {available_presets}"
            )

        return self._metadata[preset_name]

    def list_categories(self) -> List[str]:
        """
        Returns list of all available preset categories for filtering and organization.

        Returns:
            List[str]: Sorted list of all preset categories in registry
        """
        return sorted(list(self._categories))

    def search_presets(
        self, search_term: str, case_sensitive: bool = False
    ) -> List[str]:
        """
        Searches presets by keyword matching in names, descriptions, and use cases.

        Args:
            search_term: Term to search for in preset information
            case_sensitive: Whether to perform case-sensitive search

        Returns:
            List[str]: List of preset names matching search criteria
        """
        if not case_sensitive:
            search_term = search_term.lower()

        matching_presets = []

        for preset_name, metadata in self._metadata.items():
            # Search in preset name
            name_to_search = preset_name if case_sensitive else preset_name.lower()
            if search_term in name_to_search:
                matching_presets.append(preset_name)
                continue

            # Search in description
            desc_to_search = (
                metadata.description if case_sensitive else metadata.description.lower()
            )
            if search_term in desc_to_search:
                matching_presets.append(preset_name)
                continue

            # Search in use cases
            for use_case in metadata.use_cases:
                use_case_to_search = use_case if case_sensitive else use_case.lower()
                if search_term in use_case_to_search:
                    matching_presets.append(preset_name)
                    break

        # Sort by relevance (exact name matches first, then others)
        exact_matches = [
            name for name in matching_presets if search_term in name.lower()
        ]
        other_matches = [name for name in matching_presets if name not in exact_matches]

        return sorted(exact_matches) + sorted(other_matches)

    def validate_registry(
        self, strict_mode: bool = False
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Validates all presets in registry for consistency and correctness with comprehensive reporting.

        Args:
            strict_mode: Whether to enable strict validation rules

        Returns:
            tuple: Tuple of (is_valid: bool, validation_report: dict)
        """
        validation_report = {
            "is_valid": True,
            "total_presets": len(self._presets),
            "validation_errors": [],
            "validation_warnings": [],
            "preset_status": {},
        }

        for preset_name in self._presets.keys():
            preset_status = {"valid": True, "errors": [], "warnings": []}

            try:
                # Validate configuration
                config = self._presets[preset_name]
                if not config.validate_all():
                    preset_status["valid"] = False
                    preset_status["errors"].append("Configuration validation failed")

                # Validate metadata exists
                if preset_name not in self._metadata:
                    preset_status["valid"] = False
                    preset_status["errors"].append("Missing metadata")
                else:
                    metadata = self._metadata[preset_name]

                    # Check metadata consistency
                    if metadata.name != preset_name:
                        preset_status["valid"] = False
                        preset_status["errors"].append("Metadata name mismatch")

                    # Strict mode checks
                    if strict_mode:
                        if not metadata.description:
                            preset_status["warnings"].append("Empty description")

                        if not metadata.use_cases:
                            preset_status["warnings"].append("No use cases specified")

            except Exception as e:
                preset_status["valid"] = False
                preset_status["errors"].append(f"Validation exception: {str(e)}")

            validation_report["preset_status"][preset_name] = preset_status

            if not preset_status["valid"]:
                validation_report["is_valid"] = False
                validation_report["validation_errors"].extend(
                    [f"{preset_name}: {error}" for error in preset_status["errors"]]
                )

            validation_report["validation_warnings"].extend(
                [f"{preset_name}: {warning}" for warning in preset_status["warnings"]]
            )

        return validation_report["is_valid"], validation_report


# Global registry instance for environment presets
ENVIRONMENT_REGISTRY = ConfigurationRegistry("GlobalEnvironmentRegistry")


# Factory Functions for Creating Configurations


def create_preset_config(
    preset_name: str, overrides: Optional[Dict[str, Any]] = None, validate: bool = True
) -> CompleteConfig:
    """
    Factory function to create environment configuration from named preset with parameter
    validation and intelligent defaults for common research scenarios.

    Args:
        preset_name: Name of the preset configuration to use
        overrides: Optional parameter overrides to apply
        validate: Whether to validate complete configuration

    Returns:
        CompleteConfig: Complete environment configuration created from preset with validated parameters

    Raises:
        KeyError: If preset name not found in registry
        ValueError: If configuration validation fails
    """
    # Validate preset name exists in registry
    if preset_name not in ENVIRONMENT_REGISTRY._presets:
        available_presets = ", ".join(ENVIRONMENT_REGISTRY.list_presets())
        raise KeyError(
            f"Preset '{preset_name}' not found. Available presets: {available_presets}"
        )

    # Retrieve base preset configuration
    base_config = ENVIRONMENT_REGISTRY.get_preset(preset_name)

    # Apply user-specified parameter overrides
    if overrides:
        config = base_config.clone_with_overrides(**overrides)
    else:
        config = base_config

    # Validate complete configuration if requested
    if validate and not config.validate_all():
        raise ValueError(
            f"Configuration validation failed for preset '{preset_name}' with overrides"
        )

    return config


def create_research_scenario(
    scenario_name: str,
    grid_size: Optional[int] = None,
    plume_complexity: Optional[float] = None,
    research_params: Optional[Dict[str, Any]] = None,
) -> CompleteConfig:
    """
    Creates specialized research scenario configuration with academic parameters, reproducible
    settings, and research-optimized defaults for scientific studies.

    Args:
        scenario_name: Name identifier for the research scenario
        grid_size: Optional grid dimension (creates square grid)
        plume_complexity: Optional plume dispersion complexity factor
        research_params: Optional research-specific parameter overrides

    Returns:
        CompleteConfig: Research-optimized configuration with academic parameters and reproducible settings
    """
    # Research scenario mapping
    research_scenarios = {
        "simple_navigation": {
            "grid_size": (64, 64),
            "source_location": (32, 32),
            "plume_sigma": 8.0,
            "max_steps": 500,
        },
        "complex_plume": {
            "grid_size": (128, 128),
            "source_location": (96, 32),
            "plume_sigma": 15.0,
            "max_steps": 1000,
        },
        "large_environment": {
            "grid_size": (256, 256),
            "source_location": (192, 64),
            "plume_sigma": 20.0,
            "max_steps": 2000,
        },
        "corner_source": {
            "grid_size": (128, 128),
            "source_location": (16, 16),
            "plume_sigma": DEFAULT_PLUME_SIGMA,
            "max_steps": 1500,
        },
    }

    # Validate scenario name
    if scenario_name not in research_scenarios:
        available_scenarios = ", ".join(research_scenarios.keys())
        raise ValueError(
            f"Unknown research scenario '{scenario_name}'. Available: {available_scenarios}"
        )

    # Get base scenario parameters
    scenario_params = research_scenarios[scenario_name].copy()

    # Apply grid size override if provided
    if grid_size:
        scenario_params["grid_size"] = (grid_size, grid_size)
        # Adjust source location to center of new grid
        scenario_params["source_location"] = (grid_size // 2, grid_size // 2)

    # Apply plume complexity factor
    if plume_complexity:
        base_sigma = scenario_params.get("plume_sigma", DEFAULT_PLUME_SIGMA)
        scenario_params["plume_sigma"] = base_sigma * plume_complexity

    # Apply additional research parameters
    if research_params:
        scenario_params.update(research_params)

    # Create configuration with research-optimized settings
    base_config = get_complete_default_config()
    research_config = base_config.clone_with_overrides(**scenario_params)

    # Validate research configuration
    if not research_config.validate_all():
        raise ValueError(f"Research scenario '{scenario_name}' failed validation")

    return research_config


def create_benchmark_config(
    benchmark_type: str,
    target_grid_size: Optional[GridSize] = None,
    performance_targets: Optional[Dict[str, Any]] = None,
    enable_profiling: bool = False,
) -> CompleteConfig:
    """
    Creates performance benchmark configuration with specific timing targets, resource constraints,
    and optimization settings for system performance validation.

    Args:
        benchmark_type: Type of benchmark ('speed', 'memory', 'scalability', 'accuracy')
        target_grid_size: Optional grid size for benchmark complexity
        performance_targets: Optional performance target specifications
        enable_profiling: Whether to enable performance profiling settings

    Returns:
        CompleteConfig: Benchmark-optimized configuration with performance targets and profiling settings
    """
    # Benchmark type configurations
    benchmark_configs = {
        "speed": {
            "grid_size": (64, 64),
            "source_location": (32, 32),
            "max_steps": 100,
            "render_mode": RenderMode.RGB_ARRAY,
            "plume_sigma": 10.0,
        },
        "memory": {
            "grid_size": (512, 512),
            "source_location": (256, 256),
            "max_steps": 50,
            "render_mode": RenderMode.RGB_ARRAY,
            "plume_sigma": DEFAULT_PLUME_SIGMA,
        },
        "scalability": {
            "grid_size": (256, 256),
            "source_location": (128, 128),
            "max_steps": 1000,
            "render_mode": RenderMode.RGB_ARRAY,
            "plume_sigma": DEFAULT_PLUME_SIGMA,
        },
        "accuracy": {
            "grid_size": (128, 128),
            "source_location": (64, 64),
            "max_steps": 2000,
            "render_mode": RenderMode.RGB_ARRAY,
            "plume_sigma": DEFAULT_PLUME_SIGMA,
        },
    }

    # Validate benchmark type
    if benchmark_type not in benchmark_configs:
        available_types = ", ".join(benchmark_configs.keys())
        raise ValueError(
            f"Unknown benchmark type '{benchmark_type}'. Available: {available_types}"
        )

    # Get base benchmark parameters
    benchmark_params = benchmark_configs[benchmark_type].copy()

    # Apply target grid size if provided
    if target_grid_size:
        if hasattr(target_grid_size, "to_tuple"):
            benchmark_params["grid_size"] = target_grid_size.to_tuple()
        else:
            benchmark_params["grid_size"] = target_grid_size

        # Adjust source location for new grid size
        grid_w, grid_h = benchmark_params["grid_size"]
        benchmark_params["source_location"] = (grid_w // 2, grid_h // 2)

    # Apply performance targets
    if performance_targets:
        benchmark_params.update(performance_targets)

    # Enable profiling settings if requested
    if enable_profiling:
        benchmark_params["enable_profiling"] = True
        benchmark_params["collect_timing_data"] = True

    # Create benchmark configuration
    base_config = get_complete_default_config()
    benchmark_config = base_config.clone_with_overrides(**benchmark_params)

    # Validate benchmark configuration
    if not benchmark_config.validate_all():
        raise ValueError(
            f"Benchmark configuration '{benchmark_type}' failed validation"
        )

    return benchmark_config


def create_testing_config(
    test_type: str,
    seed: Optional[int] = None,
    grid_size: Optional[GridSize] = None,
    fast_mode: bool = False,
) -> CompleteConfig:
    """
    Creates testing environment configuration with controlled parameters, deterministic behavior,
    and validation-friendly settings for automated testing.

    Args:
        test_type: Type of test configuration ('unit', 'integration', 'performance', 'regression')
        seed: Optional deterministic seed for reproducible tests
        grid_size: Optional grid size override for test complexity
        fast_mode: Whether to enable fast test execution settings

    Returns:
        CompleteConfig: Testing-optimized configuration with deterministic behavior and controlled parameters
    """
    # Apply testing constants from TESTING_CONSTANTS
    test_params = TESTING_CONSTANTS.copy() if TESTING_CONSTANTS else {}

    # Test type specific configurations
    test_type_configs = {
        "unit": {
            "grid_size": (16, 16),
            "source_location": (8, 8),
            "max_steps": 50,
            "plume_sigma": 5.0,
        },
        "integration": {
            "grid_size": (32, 32),
            "source_location": (16, 16),
            "max_steps": 100,
            "plume_sigma": DEFAULT_PLUME_SIGMA,
        },
        "performance": {
            "grid_size": (128, 128),
            "source_location": (64, 64),
            "max_steps": 1000,
            "plume_sigma": DEFAULT_PLUME_SIGMA,
        },
        "regression": {
            "grid_size": (64, 64),
            "source_location": (32, 32),
            "max_steps": 200,
            "plume_sigma": DEFAULT_PLUME_SIGMA,
        },
    }

    # Validate test type
    if test_type not in test_type_configs:
        available_types = ", ".join(test_type_configs.keys())
        raise ValueError(
            f"Unknown test type '{test_type}'. Available: {available_types}"
        )

    # Merge test parameters
    test_params.update(test_type_configs[test_type])

    # Set deterministic seed if provided
    if seed is not None:
        test_params["seed"] = seed

    # Apply grid size override
    if grid_size:
        if hasattr(grid_size, "to_tuple"):
            test_params["grid_size"] = grid_size.to_tuple()
        else:
            test_params["grid_size"] = grid_size

    # Fast mode optimizations
    if fast_mode:
        test_params["grid_size"] = (8, 8)
        test_params["source_location"] = (4, 4)
        test_params["max_steps"] = 25
        test_params["render_mode"] = (
            RenderMode.RGB_ARRAY
        )  # No human rendering in fast mode

    # Ensure RGB array rendering for automated testing
    test_params.setdefault("render_mode", RenderMode.RGB_ARRAY)

    # Create testing configuration
    base_config = get_complete_default_config()
    testing_config = base_config.clone_with_overrides(**test_params)

    return testing_config


def get_available_presets(
    category_filter: Optional[str] = None, include_metadata: bool = False
) -> Dict[str, PresetMetadata]:
    """
    Returns comprehensive list of all available configuration presets with metadata,
    descriptions, and usage recommendations for user discovery.

    Args:
        category_filter: Optional category to filter presets by
        include_metadata: Whether to include full metadata objects

    Returns:
        Dict[str, PresetMetadata]: Dictionary mapping preset names to metadata objects with descriptions and specifications
    """
    # Get preset names with optional filtering
    preset_names = ENVIRONMENT_REGISTRY.list_presets(category_filter)

    # Build result dictionary
    available_presets = {}

    for preset_name in preset_names:
        if include_metadata:
            available_presets[preset_name] = ENVIRONMENT_REGISTRY.get_metadata(
                preset_name
            )
        else:
            # Return basic info without full metadata objects
            metadata = ENVIRONMENT_REGISTRY.get_metadata(preset_name)
            available_presets[preset_name] = {
                "category": metadata.category,
                "description": metadata.description,
                "use_cases": metadata.use_cases,
                "grid_size": metadata.technical_specs.get("grid_size"),
            }

    return available_presets


def validate_preset_name(
    preset_name: str, suggest_alternatives: bool = False
) -> Tuple[bool, str, List[str]]:
    """
    Validates preset name exists in registry and returns validation status with error details
    and suggestions for invalid names.

    Args:
        preset_name: Name of preset to validate
        suggest_alternatives: Whether to suggest similar preset names if validation fails

    Returns:
        tuple: Tuple of (is_valid: bool, validation_message: str, suggestions: List[str])
    """
    # Check if preset exists
    if preset_name in ENVIRONMENT_REGISTRY._presets:
        return True, f"Preset '{preset_name}' is valid and available", []

    # Preset not found - generate helpful error message
    error_message = f"Preset '{preset_name}' not found in registry"
    suggestions = []

    if suggest_alternatives:
        # Find similar preset names using simple string similarity
        available_presets = ENVIRONMENT_REGISTRY.list_presets()

        # Simple similarity based on common substrings
        for available_preset in available_presets:
            # Check for partial matches
            if (
                preset_name.lower() in available_preset.lower()
                or available_preset.lower() in preset_name.lower()
            ):
                suggestions.append(available_preset)

        # If no partial matches, suggest by category similarity
        if not suggestions:
            # Try to infer category from preset name
            if "test" in preset_name.lower():
                suggestions.extend(
                    [p for p in available_presets if "test" in p.lower()]
                )
            elif "bench" in preset_name.lower():
                suggestions.extend(
                    [p for p in available_presets if "bench" in p.lower()]
                )
            elif "research" in preset_name.lower():
                suggestions.extend(
                    [p for p in available_presets if "research" in p.lower()]
                )

        # Limit suggestions to avoid overwhelming output
        suggestions = suggestions[:5]

        if suggestions:
            error_message += f". Did you mean one of: {', '.join(suggestions)}?"
        else:
            error_message += f". Available presets: {', '.join(available_presets[:10])}"
            if len(available_presets) > 10:
                error_message += f" (and {len(available_presets) - 10} more)"

    return False, error_message, suggestions


def create_custom_scenario(
    scenario_name: str,
    environment_params: Dict[str, Any],
    plume_params: Dict[str, Any],
    render_params: Optional[Dict[str, Any]] = None,
    performance_params: Optional[Dict[str, Any]] = None,
) -> CompleteConfig:
    """
    Creates custom environment configuration with user-specified parameters, intelligent defaults,
    and comprehensive validation for specialized use cases.

    Args:
        scenario_name: Unique name for the custom scenario
        environment_params: Environment-specific parameters (grid_size, source_location, etc.)
        plume_params: Plume model parameters (sigma, concentration scaling, etc.)
        render_params: Optional rendering parameters
        performance_params: Optional performance and optimization parameters

    Returns:
        CompleteConfig: Custom configuration with user-specified parameters and intelligent defaults

    Raises:
        ValueError: If scenario name invalid or parameters fail validation
    """
    # Validate scenario name
    if not scenario_name or not isinstance(scenario_name, str):
        raise ValueError("Scenario name must be a non-empty string")

    # Start with default configuration
    base_config = get_complete_default_config()
    custom_params = {}

    # Apply environment parameters with validation
    if environment_params:
        # Validate grid size
        if "grid_size" in environment_params:
            grid_size = environment_params["grid_size"]
            if isinstance(grid_size, (list, tuple)) and len(grid_size) == 2:
                if grid_size[0] > 0 and grid_size[1] > 0:
                    custom_params["grid_size"] = tuple(grid_size)
                else:
                    raise ValueError("Grid dimensions must be positive integers")
            else:
                raise ValueError(
                    "Grid size must be a tuple or list of two positive integers"
                )

        # Validate source location
        if "source_location" in environment_params:
            source_loc = environment_params["source_location"]
            if isinstance(source_loc, (list, tuple)) and len(source_loc) == 2:
                custom_params["source_location"] = tuple(source_loc)
            else:
                raise ValueError(
                    "Source location must be a tuple or list of two coordinates"
                )

        # Apply other environment parameters
        for param, value in environment_params.items():
            if param not in ["grid_size", "source_location"]:
                custom_params[param] = value

    # Apply plume parameters with validation
    if plume_params:
        if "plume_sigma" in plume_params:
            sigma = plume_params["plume_sigma"]
            if sigma > 0:
                custom_params["plume_sigma"] = float(sigma)
            else:
                raise ValueError("Plume sigma must be positive")

        # Apply other plume parameters
        for param, value in plume_params.items():
            if param != "plume_sigma":
                custom_params[param] = value

    # Apply render parameters
    if render_params:
        custom_params.update(render_params)

    # Apply performance parameters
    if performance_params:
        custom_params.update(performance_params)

    # Create custom configuration
    custom_config = base_config.clone_with_overrides(**custom_params)

    # Validate complete custom configuration
    if not custom_config.validate_all():
        raise ValueError(f"Custom scenario '{scenario_name}' failed validation")

    return custom_config


# Export all public interface components
__all__ = [
    # Factory functions for creating configurations
    "create_preset_config",
    "create_research_scenario",
    "create_benchmark_config",
    "create_testing_config",
    "create_custom_scenario",
    # Utility functions for discovery and validation
    "get_available_presets",
    "validate_preset_name",
    # Core classes for configuration management
    "PresetMetadata",
    "ConfigurationRegistry",
    # Global registry instance
    "ENVIRONMENT_REGISTRY",
]
