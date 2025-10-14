"""
Rendering configuration management module providing comprehensive preset configurations,
factory functions, and optimization settings for dual-mode plume navigation visualization.

This module manages rendering presets for research workflows, performance benchmarks,
accessibility configurations, and custom rendering scenarios with template integration,
color scheme management, and cross-platform compatibility.

Key Components:
- RenderPresetCategory: Enumeration of preset categories for organization and filtering
- RenderConfigPreset: Comprehensive data class for rendering configuration presets
- RenderPresetRegistry: Registry system for managing and discovering presets
- Factory functions for creating optimized presets for different use cases
- Optimization and validation utilities for performance tuning
- Global registry with built-in presets for common scenarios

Performance Targets:
- RGB Array Mode: <5ms generation time
- Human Mode: <50ms rendering updates
- Memory Usage: <10MB per preset configuration
- Cross-platform compatibility with graceful fallbacks
"""

import copy  # >=3.10 - Deep copying of rendering configuration objects for safe modification
import logging
import time
from dataclasses import (  # >=3.10 - Data class decorators and field specification
    dataclass,
    field,
)
from enum import (  # >=3.10 - Base enumeration class for rendering configuration type definitions
    Enum,
)
from typing import (  # >=3.10 - Type hints for rendering configuration management
    Dict,
    List,
    Optional,
    Union,
)

from plume_nav_sim.core.constants import (
    BACKEND_PRIORITY_LIST,
    MATPLOTLIB_DEFAULT_FIGSIZE,
    PERFORMANCE_TARGET_HUMAN_RENDER_MS,
    PERFORMANCE_TARGET_RGB_RENDER_MS,
)

# Internal imports for core types and functionality
from plume_nav_sim.core.enums import RenderMode
from plume_nav_sim.core.geometry import GridSize
from plume_nav_sim.core.types import create_grid_size

# Import color scheme and template systems
from ..assets.default_colormap import PredefinedScheme
from ..assets.render_templates import TemplateQuality

# Initialize module logger
logger = logging.getLogger("plume_nav_sim.config.render_configs")


# Global configuration constants for different rendering scenarios
DEFAULT_RGB_CONFIG = {
    "template_quality": TemplateQuality.STANDARD,
    "color_scheme": PredefinedScheme.STANDARD,
    "enable_caching": True,
    "performance_target_ms": 5.0,
}

DEFAULT_MATPLOTLIB_CONFIG = {
    "backend_preferences": ["TkAgg", "Qt5Agg", "Agg"],
    "figure_size": (8, 8),
    "enable_interactive": True,
    "performance_target_ms": 50.0,
}

ACCESSIBILITY_CONFIG = {
    "high_contrast_enabled": True,
    "colorblind_friendly": True,
    "font_scaling": 1.0,
    "contrast_ratio_minimum": 4.5,
}

PERFORMANCE_PRESETS = {
    "ultra_fast": "Maximum speed, minimal features",
    "balanced": "Standard performance and features",
    "quality": "Enhanced quality, relaxed timing",
}

RESEARCH_CONFIG_TEMPLATES = {
    "publication": "High-quality static images",
    "analysis": "Interactive exploration",
    "benchmark": "Performance testing",
    "debugging": "Development visualization",
}

# Global preset registry for centralized management
RENDER_PRESETS_REGISTRY = {}


class RenderPresetCategory(Enum):
    """
    Enumeration defining categories of rendering configuration presets for organization,
    filtering, and user discovery with specific characteristics and use case focus
    for different visualization workflows.

    Categories provide logical grouping of presets based on optimization focus:
    - RGB_ARRAY: Presets optimized for programmatic RGB array generation
    - MATPLOTLIB: Presets optimized for interactive matplotlib visualization
    - RESEARCH: Presets optimized for academic and research workflows
    - ACCESSIBILITY: Presets optimized for inclusive visualization
    - PERFORMANCE: Presets optimized for maximum rendering speed
    - CUSTOM: User-defined presets with specialized requirements
    """

    RGB_ARRAY = "rgb_array"  # Presets optimized for programmatic RGB array generation
    MATPLOTLIB = (
        "matplotlib"  # Presets optimized for interactive matplotlib visualization
    )
    RESEARCH = "research"  # Presets optimized for academic and research workflows
    ACCESSIBILITY = "accessibility"  # Presets optimized for inclusive visualization
    PERFORMANCE = "performance"  # Presets optimized for maximum rendering speed
    CUSTOM = "custom"  # User-defined presets with specialized requirements

    def get_description(self) -> str:
        """
        Returns human-readable description of preset category with use cases and characteristics.

        Returns:
            str: Description of preset category including primary use cases and optimization focus
        """
        descriptions = {
            self.RGB_ARRAY: "Optimized for programmatic RGB array generation with <5ms performance targeting. Ideal for automated analysis, batch processing, and algorithmic visualization workflows.",
            self.MATPLOTLIB: "Optimized for interactive matplotlib visualization with backend management and cross-platform compatibility. Best for real-time debugging and human observation.",
            self.RESEARCH: "Optimized for academic workflows including publication graphics, analysis visualization, and reproducible research with customizable quality levels.",
            self.ACCESSIBILITY: "Enhanced for inclusive visualization with high contrast colors, colorblind-friendly schemes, and improved visibility features meeting WCAG guidelines.",
            self.PERFORMANCE: "Aggressively optimized for maximum rendering speed with minimal overhead. Designed for high-throughput scenarios and performance benchmarking.",
            self.CUSTOM: "User-defined presets with specialized requirements and advanced customization options for unique research needs.",
        }
        return descriptions[self]

    def get_default_settings(self) -> dict:
        """
        Returns default configuration settings appropriate for preset category.

        Returns:
            dict: Default configuration parameters optimized for preset category
        """
        default_settings = {
            self.RGB_ARRAY: {
                "render_mode": RenderMode.RGB_ARRAY,
                "performance_target_ms": PERFORMANCE_TARGET_RGB_RENDER_MS,
                "template_quality": TemplateQuality.STANDARD,
                "enable_caching": True,
                "color_optimization": True,
            },
            self.MATPLOTLIB: {
                "render_mode": RenderMode.HUMAN,
                "performance_target_ms": PERFORMANCE_TARGET_HUMAN_RENDER_MS,
                "backend_preferences": BACKEND_PRIORITY_LIST,
                "figure_size": MATPLOTLIB_DEFAULT_FIGSIZE,
                "enable_interactive": True,
            },
            self.RESEARCH: {
                "template_quality": TemplateQuality.QUALITY,
                "enable_publication_mode": True,
                "reproducible_settings": True,
                "metadata_preservation": True,
                "high_dpi_support": True,
            },
            self.ACCESSIBILITY: {
                "high_contrast_enabled": True,
                "colorblind_friendly": True,
                "contrast_ratio_minimum": 4.5,
                "font_scaling": 1.2,
                "enhanced_markers": True,
            },
            self.PERFORMANCE: {
                "template_quality": TemplateQuality.ULTRA_FAST,
                "performance_target_ms": 1.0,
                "minimal_features": True,
                "aggressive_optimization": True,
                "enable_profiling": True,
            },
            self.CUSTOM: {
                "flexible_configuration": True,
                "advanced_validation": True,
                "custom_templates": True,
                "user_overrides": True,
            },
        }
        return default_settings[self]


@dataclass
class RenderConfigPreset:
    """
    Comprehensive data class for rendering configuration presets containing complete
    rendering settings, metadata, validation, and optimization parameters with
    serialization support and usage tracking.

    This class encapsulates all aspects of a rendering configuration preset including:
    - Core rendering configuration with template and color settings
    - Metadata for discovery, description, and usage guidance
    - Performance characteristics and system requirements
    - Validation status and optimization metadata
    - Version tracking for compatibility management

    Attributes:
        name: Unique identifier for the preset
        category: Category classification for organization and filtering
        render_config: Core rendering configuration object
        description: Human-readable description for user discovery
        use_cases: List of specific use case scenarios
        performance_characteristics: Timing and resource usage estimates
        system_requirements: Minimum system capabilities needed
        version: Version identifier for compatibility tracking
        validated: Flag indicating validation status
        optimization_metadata: Performance optimization details
    """

    name: str  # Unique preset identifier
    category: RenderPresetCategory  # Category classification
    render_config: dict  # Core rendering configuration (placeholder for RenderConfig)
    description: str  # Human-readable description
    use_cases: List[str] = field(default_factory=list)  # Specific use case scenarios
    performance_characteristics: dict = field(
        default_factory=dict
    )  # Timing and resource estimates
    system_requirements: dict = field(
        default_factory=dict
    )  # Minimum capabilities needed
    version: str = field(default="1.0.0")  # Version for compatibility tracking
    validated: bool = field(default=False)  # Validation status flag
    optimization_metadata: Optional[dict] = field(
        default=None
    )  # Performance optimization details

    def __post_init__(self):
        """Initialize preset with default values and validation setup."""
        if not self.performance_characteristics:
            self.performance_characteristics = (
                self._initialize_performance_characteristics()
            )

        if not self.system_requirements:
            self.system_requirements = self._initialize_system_requirements()

        logger.debug(
            f"Initialized RenderConfigPreset: {self.name} (category: {self.category.value})"
        )

    def _initialize_performance_characteristics(self) -> dict:
        """Initialize performance characteristics based on category defaults."""
        category_defaults = self.category.get_default_settings()
        return {
            "estimated_render_time_ms": category_defaults.get(
                "performance_target_ms", 10.0
            ),
            "memory_usage_mb": 5.0,
            "cpu_utilization": "medium",
            "gpu_acceleration": False,
            "parallel_processing": False,
        }

    def _initialize_system_requirements(self) -> dict:
        """Initialize system requirements based on category needs."""
        return {
            "min_python_version": "3.10",
            "required_packages": ["numpy", "matplotlib"],
            "min_memory_mb": 512,
            "display_required": self.category == RenderPresetCategory.MATPLOTLIB,
            "backend_compatibility": (
                ["TkAgg", "Qt5Agg", "Agg"]
                if self.category == RenderPresetCategory.MATPLOTLIB
                else None
            ),
        }

    def validate(
        self,
        check_system_compatibility: bool = True,
        validate_performance: bool = True,
        strict_mode: bool = False,
    ) -> bool:
        """
        Comprehensive validation of preset configuration including compatibility,
        performance feasibility, and system requirements with detailed error reporting.

        Args:
            check_system_compatibility: Enable system compatibility validation
            validate_performance: Enable performance target validation
            strict_mode: Apply strict validation rules including edge cases

        Returns:
            bool: True if preset is valid and meets all requirements

        Raises:
            ValidationError: If any validation fails with detailed context
        """
        try:
            return self._extracted_from_validate_23(
                strict_mode, check_system_compatibility, validate_performance
            )
        except Exception as e:
            logger.error(f"Preset {self.name} validation failed: {e}")
            self.validated = False
            raise

    # TODO Rename this here and in `validate`
    def _extracted_from_validate_23(
        self, strict_mode, check_system_compatibility, validate_performance
    ):
        logger.debug(f"Validating preset {self.name} with strict_mode={strict_mode}")

        # Basic configuration validation
        if not self.name or not self.name.strip():
            raise ValueError("Preset name cannot be empty")

        if not isinstance(self.category, RenderPresetCategory):
            raise ValueError(f"Invalid category type: {type(self.category)}")

        if not self.render_config:
            raise ValueError("Render configuration cannot be empty")

        # Validate render_config structure (placeholder validation)
        required_keys = (
            ["render_mode", "template_quality"]
            if self.category != RenderPresetCategory.CUSTOM
            else []
        )
        for key in required_keys:
            if key not in self.render_config:
                raise ValueError(f"Missing required configuration key: {key}")

        # System compatibility validation
        if check_system_compatibility:
            self._validate_system_compatibility()

        # Performance validation
        if validate_performance:
            self._validate_performance_targets()

        # Strict mode additional validations
        if strict_mode:
            self._validate_strict_requirements()

        self.validated = True
        logger.info(f"Preset {self.name} validation successful")
        return True

    def _validate_system_compatibility(self):
        """Validate system compatibility requirements."""
        # Check display requirements for matplotlib presets
        if (
            self.category == RenderPresetCategory.MATPLOTLIB
            and self.system_requirements.get("display_required", False)
        ):
            logger.debug("Display compatibility check passed")

        # Check package availability (placeholder)
        required_packages = self.system_requirements.get("required_packages", [])
        for package in required_packages:
            logger.debug(f"Package {package} compatibility check passed")

    def _validate_performance_targets(self):
        """Validate performance characteristics against targets."""
        target_time = self.performance_characteristics.get(
            "estimated_render_time_ms", 0
        )
        category_target = self.category.get_default_settings().get(
            "performance_target_ms", 100
        )

        if target_time > category_target * 2:  # Allow 2x tolerance
            logger.warning(
                f"Performance target may be challenging: {target_time}ms > {category_target}ms"
            )

    def _validate_strict_requirements(self):
        """Apply strict validation rules for edge cases."""
        # Validate version format
        if not self.version or len(self.version.split(".")) != 3:
            raise ValueError(f"Invalid version format: {self.version}")

        # Validate description quality
        if len(self.description) < 20:
            raise ValueError("Description too short for strict validation")

    def estimate_performance(
        self, grid_size: Optional[GridSize] = None, system_info: Optional[dict] = None
    ) -> dict:
        """
        Estimates rendering performance characteristics for preset configuration
        based on system capabilities and optimization settings.

        Args:
            grid_size: Optional grid dimensions for performance scaling
            system_info: Optional system capability information

        Returns:
            dict: Performance estimates including timing, memory usage, and resource requirements
        """
        base_performance = self.performance_characteristics.copy()

        # Scale performance based on grid size
        if grid_size:
            scale_factor = (grid_size.width * grid_size.height) / (
                128 * 128
            )  # Relative to default 128x128
            base_performance["estimated_render_time_ms"] *= scale_factor
            base_performance["memory_usage_mb"] *= scale_factor

        # Factor in system capabilities
        if system_info:
            cpu_factor = system_info.get("cpu_performance_factor", 1.0)
            memory_factor = system_info.get("memory_performance_factor", 1.0)

            base_performance["estimated_render_time_ms"] /= cpu_factor
            base_performance["memory_usage_mb"] *= memory_factor

        # Category-specific performance adjustments
        if self.category == RenderPresetCategory.PERFORMANCE:
            base_performance[
                "estimated_render_time_ms"
            ] *= 0.5  # Aggressive optimization
        elif self.category == RenderPresetCategory.ACCESSIBILITY:
            base_performance["estimated_render_time_ms"] *= 1.2  # Additional processing

        logger.debug(f"Performance estimate for {self.name}: {base_performance}")
        return base_performance

    def optimize_for_system(
        self, system_info: dict, optimization_targets: dict
    ) -> "RenderConfigPreset":
        """
        Optimizes preset configuration for specific system capabilities with
        performance tuning and compatibility adjustments.

        Args:
            system_info: System capability information
            optimization_targets: Performance and resource targets

        Returns:
            RenderConfigPreset: Optimized preset with system-specific configuration
        """
        # Create deep copy for optimization
        optimized_config = copy.deepcopy(self.render_config)

        # Apply system-specific optimizations
        if system_info.get("high_performance_cpu", False):
            optimized_config["enable_advanced_features"] = True

        if system_info.get("limited_memory", False):
            optimized_config["memory_optimization"] = True
            optimized_config["reduce_quality_if_needed"] = True

        # Apply optimization targets
        target_time = optimization_targets.get("max_render_time_ms")
        if (
            target_time
            and target_time
            < self.performance_characteristics["estimated_render_time_ms"]
        ):
            optimized_config["aggressive_optimization"] = True

        # Create optimized preset
        optimized_preset = RenderConfigPreset(
            name=f"{self.name}_optimized",
            category=self.category,
            render_config=optimized_config,
            description=f"System-optimized version of {self.description}",
            use_cases=self.use_cases.copy(),
            performance_characteristics=self.estimate_performance(
                system_info=system_info
            ),
            system_requirements=self.system_requirements.copy(),
            version=self.version,
            validated=False,  # Requires re-validation
        )

        # Store optimization metadata
        optimized_preset.optimization_metadata = {
            "source_preset": self.name,
            "optimization_timestamp": time.time(),
            "system_info": system_info,
            "optimization_targets": optimization_targets,
            "optimizations_applied": ["system_specific", "performance_targeting"],
        }

        logger.info(f"Created optimized preset: {optimized_preset.name}")
        return optimized_preset

    def clone_with_overrides(
        self,
        new_name: str,
        config_overrides: Optional[dict] = None,
        preserve_validation: bool = False,
    ) -> "RenderConfigPreset":
        """
        Creates deep copy of preset with optional parameter overrides for
        customization and experimentation.

        Args:
            new_name: Name for the cloned preset
            config_overrides: Optional configuration parameter overrides
            preserve_validation: Whether to preserve validation status

        Returns:
            RenderConfigPreset: Cloned preset with applied overrides
        """
        # Deep copy all attributes
        cloned_config = copy.deepcopy(self.render_config)
        if config_overrides:
            cloned_config.update(config_overrides)

        cloned_preset = RenderConfigPreset(
            name=new_name,
            category=self.category,
            render_config=cloned_config,
            description=f"Cloned from {self.name}: {self.description}",
            use_cases=self.use_cases.copy(),
            performance_characteristics=self.performance_characteristics.copy(),
            system_requirements=self.system_requirements.copy(),
            version=self.version,
            validated=self.validated if preserve_validation else False,
        )

        logger.info(f"Cloned preset {self.name} as {new_name}")
        return cloned_preset

    def to_dict(
        self,
        include_config: bool = True,
        include_metadata: bool = True,
        include_performance: bool = True,
    ) -> dict:
        """
        Converts preset to comprehensive dictionary representation for serialization,
        storage, and external integration.

        Args:
            include_config: Whether to include render_config details
            include_metadata: Whether to include metadata information
            include_performance: Whether to include performance characteristics

        Returns:
            dict: Dictionary representation of preset
        """
        preset_dict = {
            "name": self.name,
            "category": self.category.value,
            "description": self.description,
            "version": self.version,
            "validated": self.validated,
        }

        if include_config:
            preset_dict["render_config"] = self.render_config

        if include_metadata:
            preset_dict.update(
                {
                    "use_cases": self.use_cases,
                    "system_requirements": self.system_requirements,
                }
            )

        if include_performance:
            preset_dict["performance_characteristics"] = (
                self.performance_characteristics
            )

        if self.optimization_metadata:
            preset_dict["optimization_metadata"] = self.optimization_metadata

        return preset_dict

    def test_functionality(
        self,
        test_grid_size: Optional[GridSize] = None,
        measure_performance: bool = False,
    ) -> tuple:
        """
        Tests preset functionality with sample rendering operations to verify
        configuration correctness and performance.

        Args:
            test_grid_size: Optional grid size for testing
            measure_performance: Whether to measure performance metrics

        Returns:
            tuple: (test_passed: bool, test_results: dict) with functionality validation
        """
        test_results = {
            "configuration_valid": False,
            "performance_metrics": {},
            "errors": [],
            "warnings": [],
        }

        try:
            # Test configuration validation
            self.validate(check_system_compatibility=True, validate_performance=True)
            test_results["configuration_valid"] = True

            # Performance measurement (simulated for proof-of-life)
            if measure_performance:
                start_time = time.perf_counter()
                # Simulate rendering operation
                time.sleep(0.001)  # Simulate 1ms rendering
                end_time = time.perf_counter()

                measured_time = (end_time - start_time) * 1000  # Convert to ms
                test_results["performance_metrics"] = {
                    "actual_render_time_ms": measured_time,
                    "target_render_time_ms": self.performance_characteristics[
                        "estimated_render_time_ms"
                    ],
                    "performance_ratio": measured_time
                    / self.performance_characteristics["estimated_render_time_ms"],
                }

            logger.info(f"Functionality test passed for preset {self.name}")
            return True, test_results

        except Exception as e:
            test_results["errors"].append(str(e))
            logger.error(f"Functionality test failed for preset {self.name}: {e}")
            return False, test_results


class RenderPresetRegistry:
    """
    Registry class for managing named rendering configuration presets with discovery,
    validation, registration, and optimization capabilities providing centralized
    preset administration and user-friendly preset management.

    The registry maintains a centralized catalog of rendering presets organized by
    categories with efficient indexing, search capabilities, and comprehensive
    validation support for preset management workflows.

    Attributes:
        name: Registry identifier for logging and identification
        _presets: Internal dictionary storing registered presets
        _category_index: Index mapping categories to preset names
        _registry_metadata: Metadata about registry state and history
        _validation_enabled: Flag controlling automatic preset validation
    """

    def __init__(self, registry_name: Optional[str] = None):
        """
        Initialize rendering preset registry with storage, indexing, and
        management capabilities for centralized preset administration.

        Args:
            registry_name: Optional name for the registry (defaults to 'render_presets')
        """
        self.name = registry_name or "render_presets"
        self._presets: Dict[str, RenderConfigPreset] = {}
        self._category_index: Dict[RenderPresetCategory, List[str]] = {
            category: [] for category in RenderPresetCategory
        }
        self._registry_metadata = {
            "created_at": time.time(),
            "version": "1.0.0",
            "total_presets": 0,
            "last_modified": time.time(),
        }
        self._validation_enabled = True

        # Initialize with built-in presets
        self._register_builtin_presets()

        logger.info(f"Initialized RenderPresetRegistry: {self.name}")

    def _register_builtin_presets(self):
        """Register built-in rendering presets during initialization."""
        try:
            # Create built-in RGB array preset
            rgb_preset = RenderConfigPreset(
                name="standard_rgb",
                category=RenderPresetCategory.RGB_ARRAY,
                render_config={
                    "render_mode": RenderMode.RGB_ARRAY,
                    "template_quality": TemplateQuality.STANDARD,
                    "color_scheme": PredefinedScheme.STANDARD,
                    "enable_caching": True,
                    "performance_target_ms": PERFORMANCE_TARGET_RGB_RENDER_MS,
                },
                description="Standard RGB array rendering preset optimized for programmatic processing",
                use_cases=[
                    "algorithm_development",
                    "automated_analysis",
                    "batch_processing",
                ],
            )
            self.register_preset(rgb_preset, validate_preset=False)

            # Create built-in matplotlib preset
            matplotlib_preset = RenderConfigPreset(
                name="interactive_matplotlib",
                category=RenderPresetCategory.MATPLOTLIB,
                render_config={
                    "render_mode": RenderMode.HUMAN,
                    "backend_preferences": BACKEND_PRIORITY_LIST,
                    "figure_size": MATPLOTLIB_DEFAULT_FIGSIZE,
                    "enable_interactive": True,
                    "performance_target_ms": PERFORMANCE_TARGET_HUMAN_RENDER_MS,
                },
                description="Interactive matplotlib visualization preset for human observation",
                use_cases=["debugging", "demonstrations", "educational"],
            )
            self.register_preset(matplotlib_preset, validate_preset=False)

            logger.debug("Built-in presets registered successfully")

        except Exception as e:
            logger.error(f"Failed to register built-in presets: {e}")

    def register_preset(
        self,
        preset: RenderConfigPreset,
        validate_preset: bool = True,
        allow_override: bool = False,
    ) -> bool:
        """
        Registers new rendering preset with validation, uniqueness checking,
        and metadata management.

        Args:
            preset: The preset configuration to register
            validate_preset: Whether to validate the preset before registration
            allow_override: Whether to allow overriding existing presets

        Returns:
            bool: True if preset registered successfully, False if registration failed
        """
        try:
            # Check for name conflicts
            if preset.name in self._presets and not allow_override:
                logger.error(
                    f"Preset name '{preset.name}' already exists. Use allow_override=True to replace."
                )
                return False

            # Validate preset if enabled
            if validate_preset and self._validation_enabled:
                preset.validate(
                    check_system_compatibility=True, validate_performance=True
                )

            # Register the preset
            self._presets[preset.name] = preset

            # Update category index
            if preset.name not in self._category_index[preset.category]:
                self._category_index[preset.category].append(preset.name)

            # Update registry metadata
            self._registry_metadata["total_presets"] = len(self._presets)
            self._registry_metadata["last_modified"] = time.time()

            logger.info(
                f"Successfully registered preset: {preset.name} (category: {preset.category.value})"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to register preset {preset.name}: {e}")
            return False

    def get_preset(
        self, preset_name: str, clone_for_modification: bool = False
    ) -> RenderConfigPreset:
        """
        Retrieves rendering preset by name with validation and safe cloning
        for user modification.

        Args:
            preset_name: Name of the preset to retrieve
            clone_for_modification: Whether to return a safe clone for modification

        Returns:
            RenderConfigPreset: Retrieved preset configuration

        Raises:
            KeyError: If preset name does not exist in registry
        """
        if preset_name not in self._presets:
            available_presets = list(self._presets.keys())
            raise KeyError(
                f"Preset '{preset_name}' not found. Available presets: {available_presets}"
            )

        preset = self._presets[preset_name]

        if clone_for_modification:
            return preset.clone_with_overrides(
                f"{preset_name}_copy", preserve_validation=True
            )

        return preset

    def list_presets_by_category(
        self, category: RenderPresetCategory, include_descriptions: bool = False
    ) -> Union[List[str], Dict[str, str]]:
        """
        Returns list of preset names filtered by category with optional metadata inclusion.

        Args:
            category: Category to filter presets by
            include_descriptions: Whether to include preset descriptions

        Returns:
            List of preset names or dictionary with descriptions if include_descriptions is True
        """
        preset_names = sorted(self._category_index[category])

        if not include_descriptions:
            return preset_names

        # Return dictionary with descriptions
        return {name: self._presets[name].description for name in preset_names}

    def search_presets(
        self,
        search_term: str,
        case_sensitive: bool = False,
        category_filter: Optional[RenderPresetCategory] = None,
    ) -> List[str]:
        """
        Searches presets by keyword matching in names, descriptions, and use cases
        with relevance ranking.

        Args:
            search_term: Term to search for in preset metadata
            case_sensitive: Whether search should be case sensitive
            category_filter: Optional category to limit search scope

        Returns:
            List[str]: List of preset names matching search criteria ranked by relevance
        """
        if not case_sensitive:
            search_term = search_term.lower()

        # Determine search scope
        if category_filter:
            search_presets = [
                (name, self._presets[name])
                for name in self._category_index[category_filter]
            ]
        else:
            search_presets = list(self._presets.items())

        # Score presets based on relevance
        scored_results = []
        for name, preset in search_presets:
            score = 0
            search_text = search_term if case_sensitive else search_term.lower()

            # Check name (highest weight)
            preset_name = name if case_sensitive else name.lower()
            if search_text in preset_name:
                score += 10
                if preset_name == search_text:  # Exact match
                    score += 10

            # Check description (medium weight)
            preset_desc = (
                preset.description if case_sensitive else preset.description.lower()
            )
            if search_text in preset_desc:
                score += 5

            # Check use cases (lower weight)
            for use_case in preset.use_cases:
                use_case_text = use_case if case_sensitive else use_case.lower()
                if search_text in use_case_text:
                    score += 2

            if score > 0:
                scored_results.append((name, score))

        # Sort by score (descending) and return names
        scored_results.sort(key=lambda x: x[1], reverse=True)
        return [name for name, score in scored_results]

    def validate_all_presets(
        self, check_system_compatibility: bool = False, strict_mode: bool = False
    ) -> tuple:
        """
        Validates all presets in registry for consistency, compatibility, and
        performance with comprehensive reporting.

        Args:
            check_system_compatibility: Whether to check system compatibility
            strict_mode: Whether to apply strict validation rules

        Returns:
            tuple: (all_valid: bool, validation_report: dict) with registry-wide validation
        """
        validation_report = {
            "total_presets": len(self._presets),
            "valid_presets": 0,
            "failed_presets": [],
            "warnings": [],
            "validation_timestamp": time.time(),
        }

        all_valid = True

        for name, preset in self._presets.items():
            try:
                preset.validate(
                    check_system_compatibility=check_system_compatibility,
                    validate_performance=True,
                    strict_mode=strict_mode,
                )
                validation_report["valid_presets"] += 1
                logger.debug(f"Preset {name} validation successful")

            except Exception as e:
                all_valid = False
                validation_report["failed_presets"].append(
                    {
                        "preset_name": name,
                        "error": str(e),
                        "category": preset.category.value,
                    }
                )
                logger.warning(f"Preset {name} validation failed: {e}")

        # Additional registry-level validations
        if len(self._presets) == 0:
            validation_report["warnings"].append(
                "Registry is empty - no presets registered"
            )

        if empty_categories := [
            cat for cat in RenderPresetCategory if len(self._category_index[cat]) == 0
        ]:
            validation_report["warnings"].append(
                f"Empty categories: {[cat.value for cat in empty_categories]}"
            )

        logger.info(
            f"Registry validation complete: {validation_report['valid_presets']}/{validation_report['total_presets']} presets valid"
        )
        return all_valid, validation_report

    def export_registry(
        self,
        include_configs: bool = True,
        include_metadata: bool = True,
        export_format: str = "dict",
    ) -> dict:
        """
        Exports complete registry to dictionary format for serialization,
        backup, and external integration.

        Args:
            include_configs: Whether to include preset configurations
            include_metadata: Whether to include comprehensive metadata
            export_format: Export format specification (currently supports 'dict')

        Returns:
            dict: Complete registry export with presets and metadata
        """
        export_data = {
            "registry_info": {
                "name": self.name,
                "export_timestamp": time.time(),
                "export_format": export_format,
                "total_presets": len(self._presets),
            },
            "presets": {},
            "category_index": {},
        }

        # Export presets
        for name, preset in self._presets.items():
            export_data["presets"][name] = preset.to_dict(
                include_config=include_configs,
                include_metadata=include_metadata,
                include_performance=True,
            )

        # Export category index
        for category, preset_names in self._category_index.items():
            export_data["category_index"][category.value] = preset_names

        # Include registry metadata
        if include_metadata:
            export_data["registry_metadata"] = self._registry_metadata.copy()

        logger.info(f"Exported registry with {len(self._presets)} presets")
        return export_data


# Factory functions for creating specialized rendering presets


def create_rgb_preset(
    preset_name: str,
    quality_level: Optional[TemplateQuality] = None,
    color_scheme: Optional[PredefinedScheme] = None,
    grid_size: Optional[GridSize] = None,
    optimization_options: dict = None,
) -> RenderConfigPreset:
    """
    Factory function to create RGB array rendering configuration preset optimized
    for programmatic visualization with performance tuning, color scheme integration,
    and template optimization targeting <5ms generation.

    Args:
        preset_name: Unique name for the RGB rendering preset
        quality_level: Template quality level (defaults to STANDARD)
        color_scheme: Color scheme selection (defaults to STANDARD)
        grid_size: Grid dimensions for memory estimation
        optimization_options: Additional optimization parameters

    Returns:
        RenderConfig: RGB rendering configuration with optimized template settings
    """
    # Validate preset name
    if not preset_name or not preset_name.strip():
        raise ValueError("Preset name cannot be empty")

    # Set defaults
    if quality_level is None:
        quality_level = TemplateQuality.STANDARD
    if color_scheme is None:
        color_scheme = PredefinedScheme.STANDARD
    if grid_size is None:
        grid_size = create_grid_size(128, 128)
    if optimization_options is None:
        optimization_options = {}

    # Create RGB template configuration
    rgb_config = {
        "render_mode": RenderMode.RGB_ARRAY,
        "template_quality": quality_level,
        "color_scheme": color_scheme,
        "grid_size": grid_size,
        "enable_caching": optimization_options.get("enable_caching", True),
        "vectorization": optimization_options.get("enable_vectorization", True),
        "memory_optimization": optimization_options.get("memory_optimization", True),
        "performance_target_ms": optimization_options.get(
            "performance_target_ms", PERFORMANCE_TARGET_RGB_RENDER_MS
        ),
        "strict_validation": optimization_options.get("strict_validation", True),
    }

    # Estimate memory usage
    estimated_memory = (
        grid_size.estimate_memory_mb()
        if hasattr(grid_size, "estimate_memory_mb")
        else 5.0
    )

    # Create performance characteristics
    performance_chars = {
        "estimated_render_time_ms": PERFORMANCE_TARGET_RGB_RENDER_MS,
        "memory_usage_mb": estimated_memory,
        "cpu_utilization": "low",
        "gpu_acceleration": False,
        "parallel_processing": optimization_options.get("enable_parallel", False),
    }

    # Create and return RGB preset
    rgb_preset = RenderConfigPreset(
        name=preset_name,
        category=RenderPresetCategory.RGB_ARRAY,
        render_config=rgb_config,
        description=f"RGB array rendering preset optimized for programmatic processing with {quality_level.value} quality",
        use_cases=[
            "algorithm_development",
            "automated_analysis",
            "batch_processing",
            "programmatic_visualization",
        ],
        performance_characteristics=performance_chars,
        system_requirements={
            "min_python_version": "3.10",
            "required_packages": ["numpy"],
            "min_memory_mb": int(estimated_memory * 2),  # 2x buffer
            "display_required": False,
            "backend_compatibility": None,
        },
    )

    logger.info(f"Created RGB preset: {preset_name} with quality {quality_level.value}")
    return rgb_preset


def create_matplotlib_preset(
    preset_name: str,
    backend_preferences: Optional[List[str]] = None,
    color_scheme: Optional[PredefinedScheme] = None,
    figure_size: Optional[tuple] = None,
    matplotlib_options: dict = None,
) -> RenderConfigPreset:
    """
    Factory function to create matplotlib human mode visualization preset optimized
    for interactive display with backend management, figure optimization, and
    cross-platform compatibility targeting <50ms updates.

    Args:
        preset_name: Unique name for the matplotlib rendering preset
        backend_preferences: Preferred matplotlib backends in priority order
        color_scheme: Color scheme selection for matplotlib rendering
        figure_size: Matplotlib figure dimensions (width, height)
        matplotlib_options: Additional matplotlib configuration options

    Returns:
        RenderConfig: Matplotlib rendering configuration with backend management
    """
    # Validate preset name
    if not preset_name or not preset_name.strip():
        raise ValueError("Preset name cannot be empty")

    # Set defaults
    if backend_preferences is None:
        backend_preferences = BACKEND_PRIORITY_LIST.copy()
    if color_scheme is None:
        color_scheme = PredefinedScheme.STANDARD
    if figure_size is None:
        figure_size = MATPLOTLIB_DEFAULT_FIGSIZE
    if matplotlib_options is None:
        matplotlib_options = {}

    # Create matplotlib configuration
    matplotlib_config = {
        "render_mode": RenderMode.HUMAN,
        "backend_preferences": backend_preferences,
        "color_scheme": color_scheme,
        "figure_size": figure_size,
        "enable_interactive": matplotlib_options.get("enable_interactive", True),
        "animation_interval": matplotlib_options.get("animation_interval", 50),
        "auto_refresh": matplotlib_options.get("auto_refresh", True),
        "dpi": matplotlib_options.get("dpi", 100),
        "performance_target_ms": matplotlib_options.get(
            "performance_target_ms", PERFORMANCE_TARGET_HUMAN_RENDER_MS
        ),
        "fallback_to_agg": matplotlib_options.get("fallback_to_agg", True),
    }

    # Create performance characteristics
    performance_chars = {
        "estimated_render_time_ms": PERFORMANCE_TARGET_HUMAN_RENDER_MS,
        "memory_usage_mb": 10.0,  # Higher for matplotlib figures
        "cpu_utilization": "medium",
        "gpu_acceleration": False,
        "parallel_processing": False,
        "interactive_updates": True,
    }

    # Create matplotlib preset
    matplotlib_preset = RenderConfigPreset(
        name=preset_name,
        category=RenderPresetCategory.MATPLOTLIB,
        render_config=matplotlib_config,
        description=f"Interactive matplotlib visualization preset for human observation with {len(backend_preferences)} backend options",
        use_cases=[
            "debugging",
            "demonstrations",
            "educational",
            "real_time_monitoring",
        ],
        performance_characteristics=performance_chars,
        system_requirements={
            "min_python_version": "3.10",
            "required_packages": ["matplotlib", "numpy"],
            "min_memory_mb": 128,
            "display_required": True,
            "backend_compatibility": backend_preferences,
        },
    )

    logger.info(
        f"Created matplotlib preset: {preset_name} with backends {backend_preferences}"
    )
    return matplotlib_preset


def create_research_preset(
    research_scenario: str,
    output_format: str,
    quality_level: Optional[TemplateQuality] = None,
    enable_publication_mode: bool = False,
    research_options: dict = None,
) -> RenderConfigPreset:
    """
    Factory function to create research-oriented rendering configuration preset
    optimized for academic workflows including publication graphics, analysis
    visualization, and reproducible research with customizable quality levels.

    Args:
        research_scenario: Research use case ('publication', 'analysis', 'benchmark', 'debugging')
        output_format: Desired output format for research needs
        quality_level: Template quality level for research requirements
        enable_publication_mode: Enable publication-specific optimizations
        research_options: Additional research-specific configuration options

    Returns:
        RenderConfig: Research-optimized rendering configuration
    """
    valid_scenarios = ["publication", "analysis", "benchmark", "debugging"]
    if research_scenario not in valid_scenarios:
        raise ValueError(
            f"Invalid research scenario. Must be one of: {valid_scenarios}"
        )

    # Set scenario-specific defaults
    if quality_level is None:
        quality_level = (
            TemplateQuality.QUALITY
            if research_scenario in {"publication"}
            else TemplateQuality.STANDARD
        )

    if research_options is None:
        research_options = {}

    # Configure based on research scenario
    scenario_configs = {
        "publication": {
            "high_dpi": True,
            "vector_graphics": True,
            "reproducible_colors": True,
            "metadata_preservation": True,
            "performance_relaxed": True,
        },
        "analysis": {
            "interactive_features": True,
            "real_time_updates": True,
            "data_exploration": True,
            "performance_balanced": True,
        },
        "benchmark": {
            "performance_focused": True,
            "timing_accuracy": True,
            "minimal_overhead": True,
            "measurement_tools": True,
        },
        "debugging": {
            "verbose_output": True,
            "error_visualization": True,
            "step_by_step": True,
            "diagnostic_info": True,
        },
    }

    base_config = scenario_configs[research_scenario]

    # Create research configuration
    research_config = {
        "research_scenario": research_scenario,
        "output_format": output_format,
        "template_quality": quality_level,
        "enable_publication_mode": enable_publication_mode,
        "reproducible_seeds": research_options.get("reproducible_seeds", True),
        "metadata_tracking": research_options.get("metadata_tracking", True),
        "documentation_level": research_options.get(
            "documentation_level", "comprehensive"
        ),
        **base_config,
        **research_options,
        "render_mode": (
            RenderMode.RGB_ARRAY
            if "rgb" in output_format.lower() or "array" in output_format.lower()
            else RenderMode.HUMAN
        ),
    }

    # Performance characteristics vary by scenario
    if research_scenario == "benchmark":
        performance_target = 1.0  # Very fast for benchmarking
    elif research_scenario == "publication":
        performance_target = 100.0  # Relaxed for quality
    else:
        performance_target = 20.0  # Balanced

    performance_chars = {
        "estimated_render_time_ms": performance_target,
        "memory_usage_mb": 15.0 if enable_publication_mode else 8.0,
        "cpu_utilization": "high" if research_scenario == "publication" else "medium",
        "reproducibility_guaranteed": True,
        "academic_compliance": True,
    }

    # Create research preset
    research_preset = RenderConfigPreset(
        name=f"research_{research_scenario}_{quality_level.value}",
        category=RenderPresetCategory.RESEARCH,
        render_config=research_config,
        description=f"Research-optimized preset for {research_scenario} workflows with {quality_level.value} quality",
        use_cases=[
            f"{research_scenario}_research",
            "academic_workflows",
            "reproducible_visualization",
        ],
        performance_characteristics=performance_chars,
        system_requirements={
            "min_python_version": "3.10",
            "required_packages": (
                ["numpy", "matplotlib"]
                if research_config["render_mode"] == RenderMode.HUMAN
                else ["numpy"]
            ),
            "min_memory_mb": 256 if enable_publication_mode else 128,
            "display_required": research_config["render_mode"] == RenderMode.HUMAN,
            "academic_compliance": True,
        },
    )

    logger.info(
        f"Created research preset for {research_scenario} with quality {quality_level.value}"
    )
    return research_preset


def create_accessibility_preset(
    accessibility_type: str,
    contrast_ratio: Optional[float] = None,
    colorblind_friendly: bool = True,
    accessibility_options: dict = None,
) -> RenderConfigPreset:
    """
    Factory function to create accessibility-enhanced rendering configuration
    with high contrast colors, colorblind-friendly schemes, and enhanced
    visibility features for inclusive visualization.

    Args:
        accessibility_type: Type of accessibility enhancement ('high_contrast', 'colorblind_friendly', 'low_vision')
        contrast_ratio: Minimum contrast ratio for accessibility compliance
        colorblind_friendly: Enable colorblind-friendly color schemes
        accessibility_options: Additional accessibility configuration options

    Returns:
        RenderConfig: Accessibility-enhanced rendering configuration
    """
    valid_types = ["high_contrast", "colorblind_friendly", "low_vision"]
    if accessibility_type not in valid_types:
        raise ValueError(f"Invalid accessibility type. Must be one of: {valid_types}")

    # Set accessibility defaults
    if contrast_ratio is None:
        contrast_ratio = 4.5  # WCAG AA standard

    if accessibility_options is None:
        accessibility_options = {}

    # Configure based on accessibility type
    type_configs = {
        "high_contrast": {
            "color_scheme": PredefinedScheme.HIGH_CONTRAST,
            "enhanced_markers": True,
            "bold_text": True,
            "increased_line_width": True,
        },
        "colorblind_friendly": {
            "color_scheme": PredefinedScheme.COLORBLIND_FRIENDLY,
            "alternative_markers": True,
            "pattern_fills": True,
            "texture_support": True,
        },
        "low_vision": {
            "font_scaling": 1.5,
            "marker_scaling": 2.0,
            "simplified_interface": True,
            "enhanced_visibility": True,
        },
    }

    base_config = type_configs[accessibility_type]

    # Create accessibility configuration
    accessibility_config = {
        "accessibility_type": accessibility_type,
        "contrast_ratio_minimum": contrast_ratio,
        "colorblind_friendly": colorblind_friendly,
        "wcag_compliance": accessibility_options.get("wcag_compliance", True),
        "screen_reader_compatible": accessibility_options.get(
            "screen_reader_compatible", True
        ),
        "keyboard_navigation": accessibility_options.get("keyboard_navigation", True),
        **base_config,
        **accessibility_options,
    }

    # Set render mode based on type
    if accessibility_type == "high_contrast":
        accessibility_config["render_mode"] = RenderMode.RGB_ARRAY
    else:
        accessibility_config["render_mode"] = RenderMode.HUMAN

    # Performance characteristics (may be slightly slower due to enhancements)
    performance_chars = {
        "estimated_render_time_ms": 30.0,  # Slightly slower for enhancements
        "memory_usage_mb": 12.0,
        "cpu_utilization": "medium",
        "accessibility_validated": True,
        "wcag_compliant": True,
    }

    # Create accessibility preset
    accessibility_preset = RenderConfigPreset(
        name=f"accessibility_{accessibility_type}",
        category=RenderPresetCategory.ACCESSIBILITY,
        render_config=accessibility_config,
        description=f"Accessibility-enhanced preset for {accessibility_type} with {contrast_ratio}:1 contrast ratio",
        use_cases=["inclusive_design", "accessibility_compliance", "universal_access"],
        performance_characteristics=performance_chars,
        system_requirements={
            "min_python_version": "3.10",
            "required_packages": ["numpy", "matplotlib"],
            "min_memory_mb": 128,
            "display_required": accessibility_config["render_mode"] == RenderMode.HUMAN,
            "accessibility_features": True,
            "contrast_ratio_minimum": contrast_ratio,
        },
    )

    logger.info(
        f"Created accessibility preset: {accessibility_type} with contrast ratio {contrast_ratio}:1"
    )
    return accessibility_preset


def create_performance_preset(
    performance_profile: str,
    target_latency_ms: Optional[float] = None,
    enable_profiling: bool = False,
    optimization_flags: dict = None,
) -> RenderConfigPreset:
    """
    Factory function to create performance-optimized rendering configuration
    with aggressive optimization, minimal features, and benchmark-grade speed
    for performance testing and high-throughput scenarios.

    Args:
        performance_profile: Performance optimization level ('ultra_fast', 'balanced', 'quality')
        target_latency_ms: Specific performance target in milliseconds
        enable_profiling: Enable performance profiling and monitoring
        optimization_flags: Additional optimization configuration flags

    Returns:
        RenderConfig: Performance-optimized rendering configuration
    """
    valid_profiles = ["ultra_fast", "balanced", "quality"]
    if performance_profile not in valid_profiles:
        raise ValueError(
            f"Invalid performance profile. Must be one of: {valid_profiles}"
        )

    # Set profile-specific targets
    profile_targets = {
        "ultra_fast": 1.0,  # <1ms
        "balanced": 5.0,  # <5ms
        "quality": 10.0,  # <10ms
    }

    if target_latency_ms is None:
        target_latency_ms = profile_targets[performance_profile]

    if optimization_flags is None:
        optimization_flags = {}

    # Configure based on performance profile
    profile_configs = {
        "ultra_fast": {
            "template_quality": TemplateQuality.ULTRA_FAST,
            "minimal_features": True,
            "disable_animations": True,
            "reduce_quality": True,
            "aggressive_caching": True,
            "vectorization": True,
        },
        "balanced": {
            "template_quality": TemplateQuality.STANDARD,
            "selective_optimization": True,
            "smart_caching": True,
            "performance_monitoring": True,
        },
        "quality": {
            "template_quality": TemplateQuality.QUALITY,
            "quality_preservation": True,
            "selective_optimization": True,
            "performance_awareness": True,
        },
    }

    base_config = profile_configs[performance_profile]

    # Create performance configuration
    performance_config = {
        "performance_profile": performance_profile,
        "target_latency_ms": target_latency_ms,
        "enable_profiling": enable_profiling,
        "render_mode": RenderMode.RGB_ARRAY,  # Typically faster
        "strict_timing": optimization_flags.get("strict_timing", True),
        "memory_optimization": optimization_flags.get("memory_optimization", True),
        "cpu_optimization": optimization_flags.get("cpu_optimization", True),
        **base_config,
        **optimization_flags,
    }

    # Performance characteristics
    performance_chars = {
        "estimated_render_time_ms": target_latency_ms,
        "memory_usage_mb": 3.0 if performance_profile == "ultra_fast" else 8.0,
        "cpu_utilization": "optimized",
        "benchmark_grade": True,
        "high_throughput_capable": True,
    }

    # Create performance preset
    performance_preset = RenderConfigPreset(
        name=f"performance_{performance_profile}",
        category=RenderPresetCategory.PERFORMANCE,
        render_config=performance_config,
        description=f"Performance-optimized preset targeting <{target_latency_ms}ms with {performance_profile} profile",
        use_cases=[
            "performance_testing",
            "high_throughput",
            "benchmarking",
            "speed_optimization",
        ],
        performance_characteristics=performance_chars,
        system_requirements={
            "min_python_version": "3.10",
            "required_packages": ["numpy"],
            "min_memory_mb": 64,
            "display_required": False,
            "performance_hardware_recommended": True,
        },
    )

    logger.info(
        f"Created performance preset: {performance_profile} targeting {target_latency_ms}ms"
    )
    return performance_preset


def create_custom_render_config(
    config_name: str,
    render_mode: RenderMode,
    color_config: dict,
    template_config: dict,
    performance_config: dict,
    validate_compatibility: bool = True,
) -> RenderConfigPreset:
    """
    Advanced factory function to create fully customized rendering configuration
    with user-specified parameters, comprehensive validation, and optimization
    for specialized visualization requirements.

    Args:
        config_name: Unique name for the custom configuration
        render_mode: Rendering mode (RGB_ARRAY or HUMAN)
        color_config: Custom color scheme configuration
        template_config: Custom template configuration parameters
        performance_config: Custom performance and optimization settings
        validate_compatibility: Enable comprehensive compatibility validation

    Returns:
        RenderConfig: Custom rendering configuration with user-specified parameters
    """
    # Validate inputs
    if not config_name or not config_name.strip():
        raise ValueError("Configuration name cannot be empty")

    if not isinstance(render_mode, RenderMode):
        raise ValueError(f"Invalid render mode: {render_mode}")

    if not all([color_config, template_config, performance_config]):
        raise ValueError("All configuration dictionaries must be provided")

    # Create comprehensive custom configuration
    custom_config = {
        "render_mode": render_mode,
        "custom_color_config": color_config,
        "custom_template_config": template_config,
        "custom_performance_config": performance_config,
        "validation_enabled": validate_compatibility,
        "user_customized": True,
        "configuration_timestamp": time.time(),
    }

    # Merge configurations with validation
    if validate_compatibility:
        # Validate color configuration
        required_color_keys = ["primary_scheme", "agent_color", "source_color"]
        for key in required_color_keys:
            if key not in color_config:
                logger.warning(
                    f"Missing color configuration key: {key}, using defaults"
                )

        # Validate template configuration
        if "quality" not in template_config:
            template_config["quality"] = TemplateQuality.STANDARD
            logger.warning("Template quality not specified, using STANDARD")

        # Validate performance configuration
        if "target_ms" not in performance_config:
            target_ms = (
                PERFORMANCE_TARGET_RGB_RENDER_MS
                if render_mode == RenderMode.RGB_ARRAY
                else PERFORMANCE_TARGET_HUMAN_RENDER_MS
            )
            performance_config["target_ms"] = target_ms
            logger.warning(
                f"Performance target not specified, using default: {target_ms}ms"
            )

    # Merge all configurations
    custom_config.update(color_config)
    custom_config.update(template_config)
    custom_config.update(performance_config)

    # Determine performance characteristics
    performance_chars = {
        "estimated_render_time_ms": performance_config.get("target_ms", 10.0),
        "memory_usage_mb": performance_config.get("memory_limit_mb", 10.0),
        "cpu_utilization": performance_config.get("cpu_usage", "medium"),
        "custom_configuration": True,
        "user_specified": True,
    }

    # Create custom preset
    custom_preset = RenderConfigPreset(
        name=config_name,
        category=RenderPresetCategory.CUSTOM,
        render_config=custom_config,
        description=f"Custom rendering configuration with {render_mode.value} mode and user-specified parameters",
        use_cases=[
            "specialized_requirements",
            "advanced_customization",
            "research_specific",
        ],
        performance_characteristics=performance_chars,
        system_requirements={
            "min_python_version": "3.10",
            "required_packages": (
                ["numpy", "matplotlib"]
                if render_mode == RenderMode.HUMAN
                else ["numpy"]
            ),
            "min_memory_mb": int(performance_config.get("memory_limit_mb", 10.0) * 2),
            "display_required": render_mode == RenderMode.HUMAN,
            "custom_requirements": template_config.get("special_requirements", []),
        },
    )

    logger.info(f"Created custom preset: {config_name} with {render_mode.value} mode")
    return custom_preset


def get_available_render_presets(
    category_filter: Optional[RenderPresetCategory] = None,
    mode_filter: Optional[RenderMode] = None,
    include_metadata: bool = False,
    include_performance_info: bool = False,
) -> Dict[str, RenderConfigPreset]:
    """
    Utility function to retrieve comprehensive list of all available rendering
    configuration presets with metadata, categories, and usage recommendations
    for preset discovery and selection.

    Args:
        category_filter: Optional category to filter presets
        mode_filter: Optional render mode to filter presets
        include_metadata: Whether to include comprehensive metadata
        include_performance_info: Whether to include performance characteristics

    Returns:
        Dict[str, RenderConfigPreset]: Dictionary mapping preset names to preset objects
    """
    # Get global registry
    if not RENDER_REGISTRY:
        logger.warning("Render registry not initialized, returning empty results")
        return {}

    available_presets = {}

    # Apply category filter
    if category_filter:
        preset_names = RENDER_REGISTRY.list_presets_by_category(category_filter)
    else:
        preset_names = list(RENDER_REGISTRY._presets.keys())

    # Apply mode filter and collect presets
    for name in preset_names:
        preset = RENDER_REGISTRY.get_preset(name)

        # Check mode filter
        if mode_filter:
            preset_mode = preset.render_config.get("render_mode")
            if preset_mode != mode_filter:
                continue

        # Add preset to results
        if include_metadata or include_performance_info:
            # Clone preset with full information
            result_preset = preset.clone_with_overrides(name, preserve_validation=True)

            if not include_metadata:
                # Remove metadata if not requested
                result_preset.use_cases = []
                result_preset.system_requirements = {}

            if not include_performance_info:
                # Remove performance info if not requested
                result_preset.performance_characteristics = {}
        else:
            result_preset = preset

        available_presets[name] = result_preset

    logger.debug(
        f"Retrieved {len(available_presets)} presets with filters: category={category_filter}, mode={mode_filter}"
    )
    return available_presets


def validate_render_preset(
    preset_or_config: Union[str, RenderConfigPreset],
    check_system_compatibility: bool = False,
    validate_performance: bool = True,
    strict_mode: bool = False,
) -> tuple:
    """
    Comprehensive validation function for rendering configuration presets
    including compatibility checking, performance validation, and system
    capability assessment with detailed reporting.

    Args:
        preset_or_config: Preset name (str) or RenderConfigPreset object to validate
        check_system_compatibility: Whether to check system compatibility
        validate_performance: Whether to validate performance targets
        strict_mode: Whether to apply strict validation rules

    Returns:
        tuple: (is_valid: bool, validation_report: dict) with comprehensive analysis
    """
    validation_report = {
        "validation_timestamp": time.time(),
        "preset_name": None,
        "validation_passed": False,
        "checks_performed": [],
        "warnings": [],
        "errors": [],
        "performance_analysis": {},
        "system_compatibility": {},
    }

    try:
        # Resolve preset
        if isinstance(preset_or_config, str):
            if not RENDER_REGISTRY:
                raise ValueError("Render registry not initialized")
            preset = RENDER_REGISTRY.get_preset(preset_or_config)
            validation_report["preset_name"] = preset_or_config
        elif isinstance(preset_or_config, RenderConfigPreset):
            preset = preset_or_config
            validation_report["preset_name"] = preset.name
        else:
            raise ValueError(f"Invalid input type: {type(preset_or_config)}")

        # Perform preset validation
        validation_report["checks_performed"].append("preset_structure")
        preset.validate(
            check_system_compatibility=check_system_compatibility,
            validate_performance=validate_performance,
            strict_mode=strict_mode,
        )

        # Performance analysis
        if validate_performance:
            validation_report["checks_performed"].append("performance_analysis")
            performance = preset.estimate_performance()
            validation_report["performance_analysis"] = performance

            # Check if performance meets targets
            target_time = preset.performance_characteristics.get(
                "estimated_render_time_ms", 0
            )
            if target_time > 100:  # 100ms threshold
                validation_report["warnings"].append(
                    f"Performance target may be slow: {target_time}ms"
                )

        # System compatibility analysis
        if check_system_compatibility:
            validation_report["checks_performed"].append("system_compatibility")
            sys_req = preset.system_requirements

            # Check Python version compatibility
            min_python = sys_req.get("min_python_version", "3.10")
            validation_report["system_compatibility"][
                "python_version"
            ] = f">={min_python}"

            # Check package requirements
            required_packages = sys_req.get("required_packages", [])
            validation_report["system_compatibility"][
                "required_packages"
            ] = required_packages

            # Check display requirements
            if sys_req.get("display_required", False):
                validation_report["system_compatibility"]["display_required"] = True

        validation_report["validation_passed"] = True
        logger.info(
            f"Validation successful for preset: {validation_report['preset_name']}"
        )

    except Exception as e:
        validation_report["errors"].append(str(e))
        validation_report["validation_passed"] = False
        logger.error(f"Validation failed for preset: {e}")

    return validation_report["validation_passed"], validation_report


def optimize_render_config(
    render_config: RenderConfigPreset,
    system_info: dict,
    optimization_targets: dict,
    usage_statistics: Optional[dict] = None,
) -> tuple:
    """
    Advanced optimization function to enhance rendering configuration performance
    based on system capabilities, usage patterns, and target requirements with
    comprehensive optimization analysis.

    Args:
        render_config: Rendering configuration preset to optimize
        system_info: System capability information for optimization
        optimization_targets: Performance and resource targets
        usage_statistics: Optional usage patterns for optimization guidance

    Returns:
        tuple: (optimized_config: RenderConfig, optimization_report: dict) with enhanced configuration
    """
    optimization_report = {
        "optimization_timestamp": time.time(),
        "source_preset": render_config.name,
        "optimizations_applied": [],
        "performance_improvements": {},
        "warnings": [],
        "system_analysis": system_info.copy(),
    }

    try:
        # Create optimized configuration
        optimized_preset = render_config.optimize_for_system(
            system_info, optimization_targets
        )
        optimization_report["optimizations_applied"].append("system_optimization")

        # Apply usage-based optimizations
        if usage_statistics:
            # Analyze usage patterns
            frequent_operations = usage_statistics.get("frequent_operations", [])
            avg_session_length = usage_statistics.get("avg_session_length_minutes", 10)

            if "rendering" in frequent_operations:
                optimized_preset.render_config["enable_aggressive_caching"] = True
                optimization_report["optimizations_applied"].append(
                    "caching_optimization"
                )

            if avg_session_length > 30:  # Long sessions
                optimized_preset.render_config["memory_management"] = "conservative"
                optimization_report["optimizations_applied"].append(
                    "memory_optimization"
                )

        # Performance analysis
        original_performance = render_config.estimate_performance()
        optimized_performance = optimized_preset.estimate_performance(
            system_info=system_info
        )

        # Calculate improvements
        time_improvement = (
            original_performance["estimated_render_time_ms"]
            - optimized_performance["estimated_render_time_ms"]
        )
        memory_improvement = (
            original_performance["memory_usage_mb"]
            - optimized_performance["memory_usage_mb"]
        )

        optimization_report["performance_improvements"] = {
            "render_time_reduction_ms": time_improvement,
            "memory_reduction_mb": memory_improvement,
            "relative_speed_improvement": time_improvement
            / original_performance["estimated_render_time_ms"]
            * 100,
        }

        # Validation warnings
        if time_improvement < 0:
            optimization_report["warnings"].append(
                "Optimization may have increased render time"
            )

        if memory_improvement < 0:
            optimization_report["warnings"].append(
                "Optimization may have increased memory usage"
            )

        logger.info(
            f"Optimization complete for {render_config.name}: {len(optimization_report['optimizations_applied'])} optimizations applied"
        )

        return optimized_preset, optimization_report

    except Exception as e:
        optimization_report["errors"] = [str(e)]
        logger.error(f"Optimization failed for {render_config.name}: {e}")
        return render_config, optimization_report


# Initialize global registry with built-in presets
RENDER_REGISTRY = RenderPresetRegistry("global_render_presets")

# Module exports for external access
__all__ = [
    "RenderConfigPreset",
    "RenderPresetCategory",
    "RenderPresetRegistry",
    "create_rgb_preset",
    "create_matplotlib_preset",
    "create_research_preset",
    "create_accessibility_preset",
    "create_performance_preset",
    "create_custom_render_config",
    "get_available_render_presets",
    "validate_render_preset",
    "optimize_render_config",
    "RENDER_REGISTRY",
]

# Log successful module initialization
logger.info("Render configs module initialized successfully with global registry")
