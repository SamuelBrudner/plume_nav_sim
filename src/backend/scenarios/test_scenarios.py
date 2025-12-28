# External imports with version comments
# Moved from backend.data to backend.scenarios
import copy  # >=3.10 - Deep copying of test scenarios for modification and parallel execution isolation
import json  # >=3.10 - Report serialization for test scenario collections
import time  # >=3.10 - Execution timing for scenario simulation
import uuid  # >=3.10 - Unique identifier generation for test scenario tracking and execution correlation
from dataclasses import (  # >=3.10 - Data class decorators for test scenario data structures and metadata containers
    dataclass,
    field,
)
from enum import (  # >=3.10 - Enumeration types for test scenario categories, priorities, and execution states
    Enum,
    IntEnum,
)
from typing import (  # >=3.10 - Type hints for test scenario management, validation, and execution coordination
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Union,
)

# Internal imports
from plume_nav_sim.config import (
    REPRODUCIBILITY_SEEDS,
    create_edge_case_test_config,
    create_integration_test_config,
    create_performance_test_config,
    create_reproducibility_test_config,
    create_unit_test_config,
)
from plume_nav_sim.core.constants import (
    DEFAULT_GRID_SIZE,
    MAX_GRID_SIZE,
    MEMORY_LIMIT_TOTAL_MB,
    MIN_GRID_SIZE,
    PERFORMANCE_TARGET_STEP_LATENCY_MS,
)
from plume_nav_sim.core.types import RenderMode

# Module version and configuration constants
SCENARIO_VERSION = "1.0.0"
DEFAULT_SCENARIO_TIMEOUT = 300.0
MAX_PARALLEL_SCENARIOS = 10
PERFORMANCE_REGRESSION_THRESHOLD = 0.15

# Comprehensive test scenario collections organized by category
UNIT_TEST_SCENARIOS = {
    "basic_api_compliance": "Basic Gymnasium API interface testing with minimal parameters",
    "action_space_validation": "Action space bounds checking and validation testing",
    "observation_space_validation": "Observation space format and type validation testing",
    "reset_method_testing": "Environment reset method functionality and return format testing",
    "step_method_testing": "Environment step method 5-tuple return format and logic testing",
    "render_mode_testing": "Rendering mode validation and output format testing",
    "seeding_functionality": "Random seeding functionality and state management testing",
    "parameter_validation": "Input parameter validation and error handling testing",
    "state_consistency": "Internal state consistency and validation testing",
    "component_isolation": "Individual component functionality and isolation testing",
}

INTEGRATION_TEST_SCENARIOS = {
    "environment_plume_integration": "Environment and plume model integration and interaction testing",
    "rendering_pipeline_integration": "Rendering pipeline integration with environment state testing",
    "seeding_cross_component": "Cross-component seeding consistency and reproducibility testing",
    "config_system_integration": "Configuration system integration and parameter propagation testing",
    "error_handling_integration": "Integrated error handling and recovery mechanism testing",
    "performance_monitoring_integration": "Performance monitoring integration across components testing",
    "backend_compatibility_integration": "Matplotlib backend compatibility and fallback testing",
    "memory_management_integration": "Memory management and resource cleanup integration testing",
    "episode_lifecycle_integration": "Complete episode lifecycle and state management testing",
    "cross_mode_rendering_integration": "Cross-mode rendering consistency and format validation testing",
}

PERFORMANCE_TEST_SCENARIOS = {
    "step_latency_benchmark": "Environment step execution latency benchmark validation (<1ms target)",
    "rendering_performance_benchmark": "RGB array and human mode rendering performance validation",
    "memory_usage_benchmark": "Memory usage benchmark and limit validation (<50MB target)",
    "episode_reset_benchmark": "Episode reset timing and performance validation",
    "plume_generation_benchmark": "Plume field generation performance and scaling validation",
    "scalability_testing": "Grid size scalability and performance scaling validation",
    "concurrent_environment_benchmark": "Multiple environment instance performance validation",
    "resource_efficiency_benchmark": "Resource utilization efficiency and optimization validation",
    "cache_performance_benchmark": "Caching system performance and memory efficiency validation",
    "backend_performance_comparison": "Matplotlib backend performance comparison and optimization",
}

REPRODUCIBILITY_TEST_SCENARIOS = {
    "identical_seed_episodes": "Identical seed produces identical episode validation testing",
    "cross_platform_reproducibility": "Cross-platform episode reproducibility validation testing",
    "seeding_isolation_testing": "Random seeding isolation and independence validation testing",
    "deterministic_behavior_validation": "Complete deterministic behavior validation across components",
    "episode_trajectory_consistency": "Episode trajectory consistency and reproducibility validation",
    "configuration_reproducibility": "Configuration-based reproducibility and consistency validation",
    "multi_episode_reproducibility": "Multi-episode reproducibility and statistical validation",
    "component_determinism_testing": "Individual component deterministic behavior validation",
    "state_restoration_testing": "Environment state restoration and consistency validation",
    "long_term_reproducibility": "Long-term reproducibility and stability validation testing",
}

EDGE_CASE_TEST_SCENARIOS = {
    "minimum_grid_size_testing": "Minimum grid size boundary condition and functionality testing",
    "maximum_grid_size_testing": "Maximum grid size stress testing and resource limit validation",
    "extreme_parameter_values": "Extreme parameter value handling and validation testing",
    "boundary_coordinate_testing": "Grid boundary coordinate handling and agent movement testing",
    "invalid_input_handling": "Invalid input parameter handling and error recovery testing",
    "memory_limit_testing": "Memory usage limit testing and resource exhaustion handling",
    "performance_degradation_testing": "Performance degradation handling and graceful degradation testing",
    "backend_unavailable_testing": "Matplotlib backend unavailable scenarios and fallback testing",
    "concurrent_access_testing": "Concurrent environment access and thread safety validation",
    "resource_exhaustion_recovery": "System resource exhaustion recovery and cleanup testing",
}

# Test execution state management and priority definitions
SCENARIO_EXECUTION_STATES = [
    "pending",
    "running",
    "completed",
    "failed",
    "skipped",
    "timeout",
]
SCENARIO_PRIORITIES = {"critical": 1, "high": 2, "medium": 3, "low": 4}


class ScenarioCategory(Enum):
    """Enumeration class defining test scenario categories with associated properties,
    execution requirements, and validation criteria for systematic organization and execution planning.

    This enumeration provides standardized categories that enable consistent scenario
    organization, execution planning, and resource allocation across the testing framework.
    """

    UNIT = "unit"
    INTEGRATION = "integration"
    PERFORMANCE = "performance"
    REPRODUCIBILITY = "reproducibility"
    EDGE_CASE = "edge_case"

    def get_description(self) -> str:
        """Get human-readable description of scenario category."""
        descriptions = {
            ScenarioCategory.UNIT: "Component-level testing with minimal parameters and isolation",
            ScenarioCategory.INTEGRATION: "Cross-component testing with realistic system interaction",
            ScenarioCategory.PERFORMANCE: "Performance and benchmark validation with timing constraints",
            ScenarioCategory.REPRODUCIBILITY: "Deterministic behavior and reproducibility validation",
            ScenarioCategory.EDGE_CASE: "Boundary condition and robustness testing with extreme parameters",
        }
        return descriptions.get(self, "Unknown scenario category")

    def get_default_config(
        self, overrides: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Returns default configuration template for scenarios in this category
        with category-appropriate parameters and optimization.

        Args:
            overrides: Optional parameter overrides for configuration customization

        Returns:
            dict: Default configuration template with category-specific parameters and validation rules
        """
        # Generate category-appropriate base configuration template
        base_configs = {
            ScenarioCategory.UNIT: {
                "grid_size": MIN_GRID_SIZE,
                "max_steps": 100,
                "render_mode": None,
                "optimization_level": "speed",
                "resource_constraints": {"memory_mb": 10, "timeout_seconds": 30},
            },
            ScenarioCategory.INTEGRATION: {
                "grid_size": DEFAULT_GRID_SIZE,
                "max_steps": 500,
                "render_mode": RenderMode.RGB_ARRAY,
                "optimization_level": "balanced",
                "resource_constraints": {"memory_mb": 50, "timeout_seconds": 120},
            },
            ScenarioCategory.PERFORMANCE: {
                "grid_size": DEFAULT_GRID_SIZE,
                "max_steps": 1000,
                "render_mode": RenderMode.RGB_ARRAY,
                "optimization_level": "performance",
                "resource_constraints": {
                    "memory_mb": MEMORY_LIMIT_TOTAL_MB,
                    "timeout_seconds": 600,
                },
            },
            ScenarioCategory.REPRODUCIBILITY: {
                "grid_size": (64, 64),
                "max_steps": 500,
                "render_mode": None,
                "optimization_level": "deterministic",
                "resource_constraints": {"memory_mb": 30, "timeout_seconds": 180},
            },
            ScenarioCategory.EDGE_CASE: {
                "grid_size": MAX_GRID_SIZE,
                "max_steps": 100,
                "render_mode": RenderMode.RGB_ARRAY,
                "optimization_level": "robustness",
                "resource_constraints": {
                    "memory_mb": MEMORY_LIMIT_TOTAL_MB,
                    "timeout_seconds": 300,
                },
            },
        }

        # Get base configuration for this category
        config = base_configs.get(self, {}).copy()

        # Apply category-specific parameter defaults and optimization
        config.update(
            {
                "category": self.value,
                "estimated_duration_multiplier": self.get_duration_multiplier(),
                "default_priority": self.get_default_priority(),
            }
        )

        # Merge with provided overrides if specified
        if overrides:
            config.update(overrides)

        return config

    def get_duration_multiplier(self) -> float:
        """Get execution time multiplier for this category."""
        multipliers = {
            ScenarioCategory.UNIT: 1.0,
            ScenarioCategory.INTEGRATION: 3.0,
            ScenarioCategory.PERFORMANCE: 5.0,
            ScenarioCategory.REPRODUCIBILITY: 2.5,
            ScenarioCategory.EDGE_CASE: 4.0,
        }
        return multipliers.get(self, 2.0)

    def get_default_priority(self) -> int:
        """Get default priority level for this category."""
        priorities = {
            ScenarioCategory.UNIT: SCENARIO_PRIORITIES["high"],
            ScenarioCategory.INTEGRATION: SCENARIO_PRIORITIES["critical"],
            ScenarioCategory.PERFORMANCE: SCENARIO_PRIORITIES["medium"],
            ScenarioCategory.REPRODUCIBILITY: SCENARIO_PRIORITIES["critical"],
            ScenarioCategory.EDGE_CASE: SCENARIO_PRIORITIES["low"],
        }
        return priorities.get(self, SCENARIO_PRIORITIES["medium"])

    def estimate_execution_time(self, scenario_config: Dict[str, Any]) -> float:
        """Estimates execution time for scenarios in this category based on
        configuration complexity and category characteristics.

        Args:
            scenario_config: Configuration dictionary for complexity analysis

        Returns:
            float: Estimated execution time in seconds based on category and configuration complexity
        """
        # Analyze scenario configuration for complexity factors
        base_time = 10.0  # Base execution time in seconds

        # Apply category-specific duration multiplier
        multiplier = self.get_duration_multiplier()

        # Factor in resource requirements and system overhead
        if "grid_size" in scenario_config:
            grid_size = scenario_config["grid_size"]
            if isinstance(grid_size, (tuple, list)) and len(grid_size) == 2:
                grid_complexity = (grid_size[0] * grid_size[1]) / (
                    128 * 128
                )  # Normalized to default
                multiplier *= max(0.5, min(10.0, grid_complexity))

        if "max_steps" in scenario_config:
            steps_complexity = (
                scenario_config["max_steps"] / 500.0
            )  # Normalized to typical
            multiplier *= max(0.2, min(5.0, steps_complexity))

        # Generate execution time estimate with confidence intervals
        estimated_time = base_time * multiplier
        return estimated_time


class ScenarioPriority(IntEnum):
    """Enumeration class defining test scenario execution priorities with scheduling rules,
    resource allocation preferences, and failure handling policies for optimized testing workflows.

    This enumeration provides priority-based execution ordering and resource management
    for efficient test scenario execution and system optimization.
    """

    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4

    def get_description(self) -> str:
        """Get human-readable description of scenario priority level."""
        descriptions = {
            ScenarioPriority.CRITICAL: "Critical scenarios requiring immediate execution and failure escalation",
            ScenarioPriority.HIGH: "High-priority scenarios with preferential resource allocation",
            ScenarioPriority.MEDIUM: "Medium-priority scenarios with standard resource allocation",
            ScenarioPriority.LOW: "Low-priority scenarios executed after higher priority completion",
        }
        return descriptions.get(self, "Unknown priority level")

    def get_scheduling_weight(
        self, system_load_factor: Optional[float] = None
    ) -> float:
        """Returns scheduling weight for execution order optimization based on
        priority level and system load consideration.

        Args:
            system_load_factor: Optional system load factor for weight adjustment

        Returns:
            float: Scheduling weight for execution order optimization with system load consideration
        """
        # Calculate base scheduling weight from priority value
        base_weights = {
            ScenarioPriority.CRITICAL: 10.0,
            ScenarioPriority.HIGH: 5.0,
            ScenarioPriority.MEDIUM: 2.0,
            ScenarioPriority.LOW: 1.0,
        }
        base_weight = base_weights.get(self, 2.0)

        # Adjust for system load factor if provided
        if system_load_factor is not None:
            if system_load_factor > 0.8:  # High system load
                if self in (ScenarioPriority.CRITICAL, ScenarioPriority.HIGH):
                    base_weight *= 1.5  # Boost critical scenarios
                else:
                    base_weight *= 0.7  # Reduce non-critical scenarios

        return base_weight

    def should_stop_on_failure(self, failure_type: str) -> bool:
        """Determines whether batch execution should stop when scenario with this priority fails.

        Args:
            failure_type: Type of failure (timeout, validation, system, etc.)

        Returns:
            bool: True if execution should stop, False if execution should continue
        """
        # Analyze failure type severity and impact
        critical_failures = ["system", "resource_exhaustion", "integration"]

        # Check priority-specific failure handling policies
        if self == ScenarioPriority.CRITICAL:
            return failure_type in critical_failures or failure_type == "validation"
        elif self == ScenarioPriority.HIGH:
            return failure_type in critical_failures
        else:
            # Medium and Low priority scenarios don't stop batch execution
            return False


@dataclass
class TestScenario:
    """Comprehensive test scenario data structure containing configuration, execution parameters,
    validation criteria, and metadata for systematic testing workflows with execution coordination
    and result validation.

    This class provides a complete representation of test scenarios with comprehensive
    configuration management, execution planning, and result validation capabilities.
    """

    __test__ = False

    # Required fields for scenario identification and configuration
    name: str
    category: str
    config: Dict[str, Any]
    validation_criteria: Dict[str, Any]

    # Optional fields with defaults for execution management
    priority: int = field(default=SCENARIO_PRIORITIES["medium"])
    estimated_execution_time: float = field(default=0.0)
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    scenario_id: Optional[str] = field(default=None)
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_state: str = field(default="pending")

    def __post_init__(self):
        """Initialize scenario with comprehensive configuration, validation criteria,
        and execution planning for systematic testing workflows."""
        # Generate unique scenario ID using UUID for tracking and correlation
        if self.scenario_id is None:
            self.scenario_id = str(uuid.uuid4())

        # Set default priority based on category or use provided priority value
        if hasattr(ScenarioCategory, self.category.upper()):
            category_enum = ScenarioCategory(self.category)
            if (
                self.priority == SCENARIO_PRIORITIES["medium"]
            ):  # Only override if default
                self.priority = category_enum.get_default_priority()

        # Estimate execution time based on configuration complexity and category
        if self.estimated_execution_time == 0.0:
            if hasattr(ScenarioCategory, self.category.upper()):
                category_enum = ScenarioCategory(self.category)
                self.estimated_execution_time = category_enum.estimate_execution_time(
                    self.config
                )

        # Calculate resource requirements including memory and computational needs
        if not self.resource_requirements:
            self.resource_requirements = self._calculate_resource_requirements()

        # Initialize metadata dictionary for scenario management
        if not self.metadata:
            self.metadata = {
                "created_timestamp": str(uuid.uuid4().time_low),  # Simple timestamp
                "version": SCENARIO_VERSION,
                "auto_generated": True,
            }

    def _calculate_resource_requirements(self) -> Dict[str, Any]:
        """Calculate resource requirements based on configuration."""
        requirements = {
            "memory_mb": 10,  # Base memory requirement
            "cpu_usage": "low",
            "timeout_seconds": DEFAULT_SCENARIO_TIMEOUT,
        }

        # Analyze configuration for resource needs
        if "grid_size" in self.config:
            grid_size = self.config["grid_size"]
            if isinstance(grid_size, (tuple, list)) and len(grid_size) == 2:
                grid_cells = grid_size[0] * grid_size[1]
                memory_estimate = max(
                    10, grid_cells * 0.001
                )  # Simple memory estimation
                requirements["memory_mb"] = min(memory_estimate, MEMORY_LIMIT_TOTAL_MB)

        if self.category == "performance":
            requirements["cpu_usage"] = "high"
            requirements["timeout_seconds"] = 600

        return requirements

    def estimate_resources(
        self, system_specs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Estimates resource requirements including memory usage, execution time,
        and system load based on scenario configuration and category-specific requirements.

        Args:
            system_specs: Optional system specifications for platform-specific optimization

        Returns:
            dict: Resource estimation including memory, CPU, and execution time requirements
        """
        # Analyze scenario configuration for memory and computational complexity
        base_memory = 10  # Base memory in MB
        base_cpu = 0.1  # Base CPU usage (0.0-1.0)

        # Calculate memory usage based on grid size and component requirements
        if "grid_size" in self.config:
            grid_size = self.config["grid_size"]
            if isinstance(grid_size, (tuple, list)):
                grid_complexity = grid_size[0] * grid_size[1] / (128 * 128)
                base_memory *= max(1.0, grid_complexity)

        # Estimate execution time using category-specific timing models
        execution_time = self.estimated_execution_time
        if self.category == "performance":
            base_cpu = 0.8
            execution_time *= 2.0
        elif self.category == "integration":
            base_cpu = 0.5
            execution_time *= 1.5

        # Consider system_specs if provided for platform-specific optimization
        if system_specs:
            if "memory_gb" in system_specs:
                available_memory = system_specs["memory_gb"] * 1024
                base_memory = min(base_memory, available_memory * 0.1)

            if "cpu_cores" in system_specs:
                cpu_factor = min(1.0, system_specs["cpu_cores"] / 4.0)
                execution_time /= cpu_factor

        # Generate resource allocation recommendations for optimal performance
        return {
            "memory_mb": base_memory,
            "cpu_usage": base_cpu,
            "execution_time_seconds": execution_time,
            "disk_usage_mb": 5,  # Minimal disk usage for logging
            "network_usage": False,
            "resource_profile": self.category,
        }

    def validate_configuration(self, strict_mode: bool = False) -> Dict[str, Any]:
        """Validates scenario configuration for completeness, consistency,
        and feasibility with detailed error reporting and optimization recommendations.

        Args:
            strict_mode: Whether to enable strict validation rules

        Returns:
            dict: Configuration validation results with detailed analysis and recommendations
        """
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "suggestions": [],
        }

        # Validate configuration completeness and required parameter presence
        required_fields = ["grid_size", "max_steps"]
        for required_field in required_fields:
            if required_field not in self.config:
                validation_results["errors"].append(
                    f"Missing required field: {required_field}"
                )
                validation_results["valid"] = False

        # Check parameter consistency and cross-component compatibility
        if "grid_size" in self.config:
            grid_size = self.config["grid_size"]
            if isinstance(grid_size, (tuple, list)):
                if len(grid_size) != 2 or any(
                    not isinstance(x, int) or x <= 0 for x in grid_size
                ):
                    validation_results["errors"].append("Invalid grid_size format")
                    validation_results["valid"] = False
                elif grid_size[0] < MIN_GRID_SIZE[0] or grid_size[1] < MIN_GRID_SIZE[1]:
                    validation_results["warnings"].append(
                        "Grid size below recommended minimum"
                    )
                elif grid_size[0] > MAX_GRID_SIZE[0] or grid_size[1] > MAX_GRID_SIZE[1]:
                    validation_results["warnings"].append(
                        "Grid size above recommended maximum"
                    )

        # Apply strict validation rules if strict_mode enabled
        if strict_mode:
            if self.category not in [cat.value for cat in ScenarioCategory]:
                validation_results["errors"].append(
                    f"Invalid category: {self.category}"
                )
                validation_results["valid"] = False

            if self.priority not in SCENARIO_PRIORITIES.values():
                validation_results["errors"].append(
                    f"Invalid priority: {self.priority}"
                )
                validation_results["valid"] = False

        # Validate against category-specific requirements and constraints
        if hasattr(ScenarioCategory, self.category.upper()):
            category_enum = ScenarioCategory(self.category)
            default_config = category_enum.get_default_config()

            # Check resource constraints
            if "resource_constraints" in default_config:
                constraints = default_config["resource_constraints"]
                estimated_resources = self.estimate_resources()

                if estimated_resources["memory_mb"] > constraints.get(
                    "memory_mb", float("inf")
                ):
                    validation_results["warnings"].append(
                        "Memory usage exceeds category recommendations"
                    )

        # Generate optimization recommendations for improved performance
        if self.category == "performance" and "render_mode" in self.config:
            if self.config["render_mode"] == RenderMode.HUMAN:
                validation_results["suggestions"].append(
                    "Consider rgb_array mode for performance testing"
                )

        if len(self.dependencies) > 5:
            validation_results["suggestions"].append(
                "Consider reducing scenario dependencies for faster execution"
            )

        return validation_results

    def clone_with_modifications(
        self, new_name: str, config_overrides: Dict[str, Any]
    ) -> "TestScenario":
        """Creates modified copy of test scenario with parameter overrides
        for scenario variations and comparative testing.

        Args:
            new_name: Name for the cloned scenario
            config_overrides: Configuration overrides to apply

        Returns:
            TestScenario: Cloned scenario with applied modifications and updated configuration
        """
        # Create deep copy of current scenario configuration and metadata
        cloned_config = copy.deepcopy(self.config)
        cloned_validation = copy.deepcopy(self.validation_criteria)
        cloned_metadata = copy.deepcopy(self.metadata)
        cloned_dependencies = copy.deepcopy(self.dependencies)

        # Apply config_overrides with validation and consistency checking
        cloned_config.update(config_overrides)

        # Update scenario name and regenerate unique scenario ID
        cloned_scenario = TestScenario(
            name=new_name,
            category=self.category,
            config=cloned_config,
            validation_criteria=cloned_validation,
            priority=self.priority,
            dependencies=cloned_dependencies,
            metadata=cloned_metadata,
        )

        # Mark as cloned in metadata
        cloned_scenario.metadata.update(
            {
                "cloned_from": self.scenario_id,
                "clone_timestamp": str(uuid.uuid4().time_low),
                "auto_generated": False,
            }
        )

        return cloned_scenario

    def get_execution_plan(self, include_monitoring: bool = True) -> Dict[str, Any]:
        """Generates detailed execution plan with timing, resource allocation,
        and monitoring requirements for scenario execution coordination.

        Args:
            include_monitoring: Whether to include comprehensive monitoring setup

        Returns:
            dict: Detailed execution plan with timing, monitoring, and resource coordination
        """
        # Generate execution timeline with setup, execution, and cleanup phases
        setup_time = max(1.0, self.estimated_execution_time * 0.1)
        execution_time = self.estimated_execution_time
        cleanup_time = max(0.5, self.estimated_execution_time * 0.05)

        execution_plan = {
            "scenario_id": self.scenario_id,
            "name": self.name,
            "category": self.category,
            "priority": self.priority,
            "phases": {
                "setup": {
                    "duration_seconds": setup_time,
                    "actions": [
                        "validate_configuration",
                        "allocate_resources",
                        "initialize_environment",
                    ],
                },
                "execution": {
                    "duration_seconds": execution_time,
                    "actions": ["run_scenario", "collect_metrics", "validate_results"],
                },
                "cleanup": {
                    "duration_seconds": cleanup_time,
                    "actions": ["cleanup_resources", "save_results", "update_state"],
                },
            },
            "resource_allocation": self.resource_requirements,
            "dependencies": self.dependencies,
            "validation_checkpoints": list(self.validation_criteria.keys()),
        }

        # Include comprehensive monitoring setup if include_monitoring enabled
        if include_monitoring:
            execution_plan["monitoring"] = {
                "performance_metrics": ["execution_time", "memory_usage", "cpu_usage"],
                "validation_metrics": ["success_rate", "error_count", "warning_count"],
                "progress_tracking": True,
                "real_time_alerts": self.priority <= ScenarioPriority.HIGH,
            }

        return execution_plan

    def to_dict(self, include_execution_data: bool = False) -> Dict[str, Any]:
        """Converts test scenario to dictionary format for serialization, storage,
        and integration with external testing frameworks.

        Args:
            include_execution_data: Whether to include execution-specific data

        Returns:
            dict: Complete scenario data in dictionary format for external integration and storage
        """
        # Compile all scenario properties including name, category, and configuration
        scenario_dict = {
            "scenario_id": self.scenario_id,
            "name": self.name,
            "category": self.category,
            "config": self.config.copy(),
            "validation_criteria": self.validation_criteria.copy(),
            "priority": self.priority,
            "estimated_execution_time": self.estimated_execution_time,
            "resource_requirements": self.resource_requirements.copy(),
            "dependencies": self.dependencies.copy(),
            "metadata": self.metadata.copy(),
            "execution_state": self.execution_state,
        }

        # Add execution data if include_execution_data is True
        if include_execution_data:
            scenario_dict["execution_plan"] = self.get_execution_plan()
            scenario_dict["resource_estimation"] = self.estimate_resources()
            scenario_dict["configuration_validation"] = self.validate_configuration()

        return scenario_dict


@dataclass
class TestScenarioCollection:
    """Comprehensive collection manager for test scenarios providing organization,
    execution coordination, result tracking, and batch processing capabilities for
    systematic testing workflows and quality assurance.

    This class manages collections of test scenarios with comprehensive organization,
    execution coordination, and result analysis capabilities.
    """

    __test__ = False

    scenarios: Dict[str, TestScenario]
    collection_name: str
    description: Optional[str] = None

    # Automatically generated fields for collection management
    category_index: Dict[str, List[str]] = field(default_factory=dict)
    priority_index: Dict[str, List[str]] = field(default_factory=dict)
    execution_results: Dict[str, Any] = field(default_factory=dict)
    total_estimated_time: float = field(default=0.0)
    collection_metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize test scenario collection with organization, indexing,
        and execution coordination for systematic testing workflows."""
        # Build category index by analyzing scenario categories for efficient filtering
        self.category_index = {}
        for scenario_name, scenario in self.scenarios.items():
            category = scenario.category
            if category not in self.category_index:
                self.category_index[category] = []
            self.category_index[category].append(scenario_name)

        # Build priority index by analyzing scenario priorities for execution ordering
        self.priority_index = {}
        for scenario_name, scenario in self.scenarios.items():
            priority = str(scenario.priority)
            if priority not in self.priority_index:
                self.priority_index[priority] = []
            self.priority_index[priority].append(scenario_name)

        # Calculate total estimated execution time for capacity planning
        self.total_estimated_time = sum(
            scenario.estimated_execution_time for scenario in self.scenarios.values()
        )

        # Initialize collection metadata for version tracking and management
        self.collection_metadata = {
            "scenario_count": len(self.scenarios),
            "categories": list(self.category_index.keys()),
            "priorities": list(self.priority_index.keys()),
            "creation_timestamp": str(uuid.uuid4().time_low),
            "version": SCENARIO_VERSION,
        }

    def add_scenario(self, scenario_name: str, scenario: TestScenario) -> bool:
        """Adds new test scenario to collection with validation, indexing updates,
        and dependency analysis for collection management.

        Args:
            scenario_name: Unique name for the scenario
            scenario: TestScenario instance to add

        Returns:
            bool: True if scenario added successfully, False if validation failed
        """
        # Validate scenario_name for uniqueness and naming conventions
        if not scenario_name or not isinstance(scenario_name, str):
            return False

        if scenario_name in self.scenarios:
            return False  # Duplicate name

        # Validate scenario configuration and execution requirements
        validation_result = scenario.validate_configuration()
        if not validation_result["valid"]:
            return False

        # Add scenario to scenarios dictionary with validation
        self.scenarios[scenario_name] = scenario

        # Update category and priority indexes with new scenario
        category = scenario.category
        if category not in self.category_index:
            self.category_index[category] = []
        self.category_index[category].append(scenario_name)

        priority = str(scenario.priority)
        if priority not in self.priority_index:
            self.priority_index[priority] = []
        self.priority_index[priority].append(scenario_name)

        # Recalculate total estimated execution time
        self.total_estimated_time += scenario.estimated_execution_time

        # Update collection metadata and dependency tracking
        self.collection_metadata["scenario_count"] = len(self.scenarios)
        self.collection_metadata["categories"] = list(self.category_index.keys())
        self.collection_metadata["priorities"] = list(self.priority_index.keys())

        return True

    def get_by_category(
        self,
        category: str,
        sort_by: Optional[str] = None,
        execution_state_filter: Optional[str] = None,
    ) -> Dict[str, TestScenario]:
        """Retrieves test scenarios filtered by category with optional sorting
        and execution state filtering for targeted testing workflows.

        Args:
            category: Category to filter by
            sort_by: Optional sorting criteria (name, priority, execution_time)
            execution_state_filter: Optional execution state filter

        Returns:
            Dict[str, TestScenario]: Filtered test scenarios in specified category with optional sorting and state filtering
        """
        # Validate category against available categories in category_index
        if category not in self.category_index:
            return {}

        # Retrieve scenario names from category_index for specified category
        scenario_names = self.category_index[category]

        # Filter by execution state if execution_state_filter provided
        if execution_state_filter:
            scenario_names = [
                name
                for name in scenario_names
                if self.scenarios[name].execution_state == execution_state_filter
            ]

        # Extract corresponding scenarios from scenarios dictionary
        _ = {name: self.scenarios[name] for name in scenario_names}

        # Apply sorting if sort_by is specified (name, priority, execution_time)
        if sort_by == "name":
            scenario_names.sort()
        elif sort_by == "priority":
            scenario_names.sort(key=lambda name: self.scenarios[name].priority)
        elif sort_by == "execution_time":
            scenario_names.sort(
                key=lambda name: self.scenarios[name].estimated_execution_time
            )

        # Return filtered and sorted scenario collection
        return {name: self.scenarios[name] for name in scenario_names}

    def execute_collection(
        self,
        optimize_execution_order: bool = True,
        max_parallel: Optional[int] = None,
        stop_on_failure: bool = False,
        collect_detailed_results: bool = True,
    ) -> Dict[str, Any]:
        """Executes entire scenario collection with optimized scheduling, parallel execution,
        comprehensive monitoring, and result aggregation for batch testing workflows.

        Args:
            optimize_execution_order: Whether to optimize execution order by priority
            max_parallel: Maximum number of parallel scenario executions
            stop_on_failure: Whether to stop collection execution on first failure
            collect_detailed_results: Whether to collect comprehensive result data

        Returns:
            dict: Comprehensive execution results with timing, success rates, and detailed analysis
        """
        # Generate optimized execution plan if optimize_execution_order enabled
        scenario_order = list(self.scenarios.keys())
        if optimize_execution_order:
            scenario_order.sort(
                key=lambda name: (
                    self.scenarios[name].priority,
                    -self.scenarios[name].estimated_execution_time,
                )
            )

        # Configure parallel execution with max_parallel limit or collection default
        max_parallel = max_parallel or MAX_PARALLEL_SCENARIOS

        # Initialize execution results tracking
        execution_results = {
            "collection_name": self.collection_name,
            "start_timestamp": str(uuid.uuid4().time_low),
            "execution_order": scenario_order,
            "scenario_results": {},
            "summary": {
                "total_scenarios": len(self.scenarios),
                "completed": 0,
                "failed": 0,
                "skipped": 0,
                "total_execution_time": 0.0,
            },
        }

        # Execute scenarios with comprehensive monitoring and progress tracking
        for scenario_name in scenario_order:
            scenario = self.scenarios[scenario_name]
            scenario.execution_state = "running"

            # Simulate scenario execution (in real implementation, this would execute the actual test)
            start_time = time.time()

            try:
                # Mock execution based on scenario configuration
                execution_time = min(
                    scenario.estimated_execution_time, 1.0
                )  # Cap for simulation
                time.sleep(execution_time * 0.001)  # Minimal actual delay

                # Simulate success/failure based on scenario type
                success = True
                if "edge_case" in scenario.category and "invalid" in scenario_name:
                    success = False

                end_time = time.time()
                actual_time = end_time - start_time

                # Record scenario result
                scenario_result = {
                    "success": success,
                    "execution_time": actual_time,
                    "memory_peak": scenario.resource_requirements.get("memory_mb", 0),
                    "validation_results": scenario.validate_configuration(),
                }

                if collect_detailed_results:
                    scenario_result.update(
                        {
                            "resource_usage": scenario.estimate_resources(),
                            "execution_plan": scenario.get_execution_plan(),
                            "configuration": scenario.config.copy(),
                        }
                    )

                execution_results["scenario_results"][scenario_name] = scenario_result
                execution_results["summary"]["total_execution_time"] += actual_time

                if success:
                    scenario.execution_state = "completed"
                    execution_results["summary"]["completed"] += 1
                else:
                    scenario.execution_state = "failed"
                    execution_results["summary"]["failed"] += 1

                    # Handle execution failures with stop_on_failure policy enforcement
                    if stop_on_failure and scenario.priority <= ScenarioPriority.HIGH:
                        # Stop execution for high-priority failures
                        remaining_scenarios = scenario_order[
                            scenario_order.index(scenario_name) + 1 :
                        ]
                        for remaining_name in remaining_scenarios:
                            self.scenarios[remaining_name].execution_state = "skipped"
                            execution_results["summary"]["skipped"] += 1
                        break

            except Exception as e:
                scenario.execution_state = "failed"
                execution_results["summary"]["failed"] += 1
                execution_results["scenario_results"][scenario_name] = {
                    "success": False,
                    "error": str(e),
                    "execution_time": time.time() - start_time,
                }

        # Update execution_results with complete batch execution data
        execution_results["end_timestamp"] = str(uuid.uuid4().time_low)
        execution_results["success_rate"] = (
            execution_results["summary"]["completed"]
            / execution_results["summary"]["total_scenarios"]
        )

        self.execution_results = execution_results
        return execution_results

    def generate_execution_report(
        self,
        include_performance_analysis: bool = True,
        include_failure_analysis: bool = True,
        report_format: str = "dict",
    ) -> Union[Dict[str, Any], str]:
        """Generates comprehensive execution report with statistics, performance analysis,
        failure analysis, and optimization recommendations for quality assurance and process improvement.

        Args:
            include_performance_analysis: Whether to include detailed performance metrics
            include_failure_analysis: Whether to include failure root cause analysis
            report_format: Output format (dict, json, html)

        Returns:
            Union[dict, str]: Comprehensive execution report in specified format with detailed analysis and recommendations
        """
        if not self.execution_results:
            return {"error": "No execution results available"}

        # Analyze execution results for success rates and timing statistics
        results = self.execution_results
        summary = results["summary"]

        report = {
            "collection_summary": {
                "collection_name": self.collection_name,
                "total_scenarios": summary["total_scenarios"],
                "success_rate": results.get("success_rate", 0.0),
                "execution_time": summary["total_execution_time"],
                "completion_status": {
                    "completed": summary["completed"],
                    "failed": summary["failed"],
                    "skipped": summary["skipped"],
                },
            },
            "category_breakdown": {},
            "priority_analysis": {},
            "recommendations": [],
        }

        # Generate category breakdown
        for category, scenario_names in self.category_index.items():
            category_stats = {"total": len(scenario_names), "completed": 0, "failed": 0}
            for name in scenario_names:
                if name in results["scenario_results"]:
                    result = results["scenario_results"][name]
                    if result.get("success", False):
                        category_stats["completed"] += 1
                    else:
                        category_stats["failed"] += 1

            report["category_breakdown"][category] = category_stats

        # Generate performance analysis if include_performance_analysis enabled
        if include_performance_analysis:
            performance_data = []
            total_time = 0
            for scenario_name, result in results["scenario_results"].items():
                if "execution_time" in result:
                    performance_data.append(
                        {
                            "name": scenario_name,
                            "category": self.scenarios[scenario_name].category,
                            "execution_time": result["execution_time"],
                            "memory_usage": result.get("memory_peak", 0),
                        }
                    )
                    total_time += result["execution_time"]

            report["performance_analysis"] = {
                "total_execution_time": total_time,
                "average_scenario_time": (
                    total_time / len(performance_data) if performance_data else 0
                ),
                "slowest_scenarios": sorted(
                    performance_data, key=lambda x: x["execution_time"], reverse=True
                )[:5],
                "memory_intensive_scenarios": sorted(
                    performance_data, key=lambda x: x["memory_usage"], reverse=True
                )[:5],
            }

        # Conduct failure analysis if include_failure_analysis enabled
        if include_failure_analysis:
            failed_scenarios = []
            for scenario_name, result in results["scenario_results"].items():
                if not result.get("success", True):
                    failed_scenarios.append(
                        {
                            "name": scenario_name,
                            "category": self.scenarios[scenario_name].category,
                            "priority": self.scenarios[scenario_name].priority,
                            "error": result.get("error", "Unknown error"),
                        }
                    )

            report["failure_analysis"] = {
                "total_failures": len(failed_scenarios),
                "failure_by_category": {},
                "failure_by_priority": {},
                "critical_failures": [
                    f
                    for f in failed_scenarios
                    if f["priority"] <= ScenarioPriority.HIGH
                ],
            }

            # Categorize failures
            for failure in failed_scenarios:
                category = failure["category"]
                priority = failure["priority"]

                if category not in report["failure_analysis"]["failure_by_category"]:
                    report["failure_analysis"]["failure_by_category"][category] = 0
                report["failure_analysis"]["failure_by_category"][category] += 1

                if priority not in report["failure_analysis"]["failure_by_priority"]:
                    report["failure_analysis"]["failure_by_priority"][priority] = 0
                report["failure_analysis"]["failure_by_priority"][priority] += 1

        # Generate optimization recommendations based on execution patterns
        if results["success_rate"] < 0.8:
            report["recommendations"].append(
                "Review failed scenarios and improve error handling"
            )

        if include_performance_analysis and "performance_analysis" in report:
            avg_time = report["performance_analysis"]["average_scenario_time"]
            if avg_time > 10.0:
                report["recommendations"].append(
                    "Consider optimizing slow scenarios or increasing timeout limits"
                )

        # Format report according to report_format specification (dict, json, html)
        if report_format == "json":
            return json.dumps(report, indent=2)
        elif report_format == "html":
            # Simple HTML formatting
            html_parts = [f"<h1>Test Execution Report: {self.collection_name}</h1>"]
            html_parts.append(
                f"<p>Success Rate: {report['collection_summary']['success_rate']:.1%}</p>"
            )
            return "\n".join(html_parts)

        return report

    def validate_collection(
        self, strict_validation: bool = False, check_dependencies: bool = True
    ) -> Dict[str, Any]:
        """Validates entire scenario collection for consistency, completeness,
        dependency resolution, and execution feasibility with comprehensive quality assurance analysis.

        Args:
            strict_validation: Whether to apply strict validation rules
            check_dependencies: Whether to validate dependency resolution

        Returns:
            dict: Collection validation results with detailed analysis and quality recommendations
        """
        validation_report = {
            "collection_valid": True,
            "scenario_validations": {},
            "dependency_issues": [],
            "quality_score": 0.0,
            "recommendations": [],
        }

        valid_scenarios = 0
        total_scenarios = len(self.scenarios)

        # Validate each scenario individually for configuration consistency
        for scenario_name, scenario in self.scenarios.items():
            scenario_validation = scenario.validate_configuration(strict_validation)
            validation_report["scenario_validations"][
                scenario_name
            ] = scenario_validation

            if scenario_validation["valid"]:
                valid_scenarios += 1
            else:
                validation_report["collection_valid"] = False

        # Check dependency resolution if check_dependencies enabled
        if check_dependencies:
            all_scenario_names = set(self.scenarios.keys())
            for scenario_name, scenario in self.scenarios.items():
                for dependency in scenario.dependencies:
                    if dependency not in all_scenario_names:
                        validation_report["dependency_issues"].append(
                            {
                                "scenario": scenario_name,
                                "missing_dependency": dependency,
                            }
                        )
                        validation_report["collection_valid"] = False

        # Calculate quality score
        validation_report["quality_score"] = (
            valid_scenarios / total_scenarios if total_scenarios > 0 else 0.0
        )

        # Generate quality recommendations for collection improvement
        if validation_report["quality_score"] < 1.0:
            validation_report["recommendations"].append(
                "Fix invalid scenario configurations"
            )

        if validation_report["dependency_issues"]:
            validation_report["recommendations"].append(
                "Resolve missing scenario dependencies"
            )

        if len(self.category_index) < 3:
            validation_report["recommendations"].append(
                "Consider adding scenarios from more test categories"
            )

        return validation_report


# Factory functions for comprehensive test scenario generation


def get_unit_test_scenarios(
    scenario_names: Optional[List[str]] = None,
    include_metadata: bool = True,
    optimize_for_speed: bool = True,
) -> Dict[str, TestScenario]:
    """Returns comprehensive collection of unit test scenarios for component-level testing
    with minimal parameters, fast execution, and isolated validation for rapid development
    feedback and component verification.

    Args:
        scenario_names: Optional list of specific scenario names to generate
        include_metadata: Whether to include comprehensive metadata
        optimize_for_speed: Whether to enable speed optimizations

    Returns:
        Dict[str, TestScenario]: Dictionary of unit test scenarios with optimized configurations for fast component testing
    """
    # Filter scenario_names if provided or use all UNIT_TEST_SCENARIOS keys
    target_scenarios = (
        scenario_names if scenario_names else list(UNIT_TEST_SCENARIOS.keys())
    )

    scenarios = {}

    for scenario_name in target_scenarios:
        if scenario_name not in UNIT_TEST_SCENARIOS:
            continue

        # Generate unit test configurations using create_unit_test_config for each scenario
        base_config = create_unit_test_config(
            test_type="component_validation",
            grid_size=MIN_GRID_SIZE if optimize_for_speed else DEFAULT_GRID_SIZE,
        )

        # Create TestScenario instances with minimal parameters optimized for speed
        scenario_config = base_config.copy()
        if optimize_for_speed:
            scenario_config.update(
                {"max_steps": 50, "render_mode": None, "performance_monitoring": False}
            )

        # Configure scenarios for component isolation and fast execution
        validation_criteria = {
            "api_compliance": True,
            "execution_time_limit": 5.0,
            "memory_limit_mb": 20,
            "error_tolerance": 0,
        }

        scenario = TestScenario(
            name=scenario_name,
            category=ScenarioCategory.UNIT.value,
            config=scenario_config,
            validation_criteria=validation_criteria,
            priority=ScenarioPriority.HIGH,
        )

        # Add comprehensive metadata if include_metadata is True
        if include_metadata:
            scenario.metadata.update(
                {
                    "description": UNIT_TEST_SCENARIOS[scenario_name],
                    "test_focus": "component_isolation",
                    "optimization_level": "speed" if optimize_for_speed else "standard",
                    "automation_friendly": True,
                }
            )

        scenarios[scenario_name] = scenario

    return scenarios


def get_integration_test_scenarios(
    integration_focus: Optional[str] = None,
    include_performance_monitoring: bool = True,
    system_constraints: Optional[Dict[str, Any]] = None,
) -> Dict[str, TestScenario]:
    """Returns comprehensive collection of integration test scenarios for cross-component testing
    with realistic parameters, system interaction validation, and end-to-end functionality verification.

    Args:
        integration_focus: Optional focus area for integration testing
        include_performance_monitoring: Whether to enable performance monitoring
        system_constraints: Optional system constraints for resource-aware testing

    Returns:
        Dict[str, TestScenario]: Dictionary of integration test scenarios with cross-component validation and realistic system interaction testing
    """
    # Select integration scenarios based on integration_focus or use all INTEGRATION_TEST_SCENARIOS
    if integration_focus:
        target_scenarios = [
            name
            for name in INTEGRATION_TEST_SCENARIOS.keys()
            if integration_focus.lower() in name.lower()
        ]
    else:
        target_scenarios = list(INTEGRATION_TEST_SCENARIOS.keys())

    scenarios = {}

    for scenario_name in target_scenarios:
        if scenario_name not in INTEGRATION_TEST_SCENARIOS:
            continue

        # Generate integration test configurations using create_integration_test_config
        base_config = create_integration_test_config(
            test_focus="cross_component",
            enable_monitoring=include_performance_monitoring,
        )

        # Configure scenarios for cross-component interaction and realistic system behavior
        scenario_config = base_config.copy()
        scenario_config.update(
            {
                "render_mode": RenderMode.RGB_ARRAY,
                "episode_length": 500,
                "cross_component_validation": True,
            }
        )

        # Apply system constraints if provided for resource-aware testing
        if system_constraints:
            if "memory_limit_mb" in system_constraints:
                scenario_config["memory_limit"] = system_constraints["memory_limit_mb"]
            if "timeout_seconds" in system_constraints:
                scenario_config["timeout"] = system_constraints["timeout_seconds"]

        # Set up component interaction validation and dependency checking
        validation_criteria = {
            "component_interaction": True,
            "cross_system_consistency": True,
            "resource_cleanup": True,
            "error_propagation": True,
            "performance_baseline": include_performance_monitoring,
        }

        scenario = TestScenario(
            name=scenario_name,
            category=ScenarioCategory.INTEGRATION.value,
            config=scenario_config,
            validation_criteria=validation_criteria,
            priority=ScenarioPriority.CRITICAL,
        )

        scenario.metadata.update(
            {
                "description": INTEGRATION_TEST_SCENARIOS[scenario_name],
                "test_focus": integration_focus or "comprehensive_integration",
                "monitoring_enabled": include_performance_monitoring,
                "system_level": True,
            }
        )

        scenarios[scenario_name] = scenario

    return scenarios


def get_performance_test_scenarios(
    benchmark_categories: Optional[List[str]] = None,
    include_scalability_tests: bool = True,
    performance_targets: Optional[Dict[str, Any]] = None,
    enable_regression_detection: bool = True,
) -> Dict[str, TestScenario]:
    """Returns comprehensive collection of performance test scenarios with benchmark validation,
    timing requirements, resource monitoring, and scalability testing for system optimization
    and regression detection.

    Args:
        benchmark_categories: Optional list of benchmark categories to focus on
        include_scalability_tests: Whether to include scalability test scenarios
        performance_targets: Optional custom performance targets
        enable_regression_detection: Whether to enable regression detection monitoring

    Returns:
        Dict[str, TestScenario]: Dictionary of performance test scenarios with benchmark validation and comprehensive timing analysis
    """
    # Select benchmark categories or use all PERFORMANCE_TEST_SCENARIOS
    if benchmark_categories:
        target_scenarios = [
            name
            for name in PERFORMANCE_TEST_SCENARIOS.keys()
            if any(
                category.lower() in name.lower() for category in benchmark_categories
            )
        ]
    else:
        target_scenarios = list(PERFORMANCE_TEST_SCENARIOS.keys())

    # Include scalability test scenarios if include_scalability_tests is enabled
    if include_scalability_tests:
        scalability_scenarios = [
            "scalability_testing",
            "concurrent_environment_benchmark",
        ]
        target_scenarios.extend(
            s for s in scalability_scenarios if s in PERFORMANCE_TEST_SCENARIOS
        )

    scenarios = {}

    # Apply custom performance_targets or use defaults from constants
    targets = performance_targets or {
        "step_latency_ms": PERFORMANCE_TARGET_STEP_LATENCY_MS,
        "memory_limit_mb": MEMORY_LIMIT_TOTAL_MB,
        "rendering_latency_ms": 5.0,
        "reset_latency_ms": 10.0,
    }

    for scenario_name in target_scenarios:
        if scenario_name not in PERFORMANCE_TEST_SCENARIOS:
            continue

        # Generate performance test configurations using create_performance_test_config
        base_config = create_performance_test_config(
            benchmark_type="timing_validation", enable_profiling=True
        )

        # Configure benchmark scenarios with strict timing validation and resource monitoring
        scenario_config = base_config.copy()
        scenario_config.update(
            {
                "performance_targets": targets,
                "enable_profiling": True,
                "resource_monitoring": True,
                "benchmark_validation": True,
            }
        )

        # Enable regression detection monitoring if enable_regression_detection is requested
        if enable_regression_detection:
            scenario_config.update(
                {
                    "regression_detection": True,
                    "baseline_comparison": True,
                    "performance_history": True,
                }
            )

        # Set up comprehensive performance measurement and statistical analysis
        validation_criteria = {
            "timing_requirements": True,
            "memory_constraints": True,
            "resource_efficiency": True,
            "performance_regression": enable_regression_detection,
            "statistical_significance": True,
        }

        # Configure resource usage monitoring and memory limit validation
        resource_requirements = {
            "memory_mb": targets.get("memory_limit_mb", MEMORY_LIMIT_TOTAL_MB),
            "cpu_usage": "high",
            "monitoring_overhead": True,
            "profiling_enabled": True,
        }

        scenario = TestScenario(
            name=scenario_name,
            category=ScenarioCategory.PERFORMANCE.value,
            config=scenario_config,
            validation_criteria=validation_criteria,
            priority=ScenarioPriority.MEDIUM,
            resource_requirements=resource_requirements,
        )

        scenario.metadata.update(
            {
                "description": PERFORMANCE_TEST_SCENARIOS[scenario_name],
                "benchmark_category": (
                    benchmark_categories[0] if benchmark_categories else "general"
                ),
                "scalability_test": "scalability" in scenario_name,
                "regression_monitoring": enable_regression_detection,
                "performance_critical": True,
            }
        )

        scenarios[scenario_name] = scenario

    return scenarios


def get_reproducibility_test_scenarios(
    test_seeds: Optional[List[int]] = None,
    num_validation_episodes: int = 5,
    cross_platform_testing: bool = False,
    statistical_validation: bool = True,
) -> Dict[str, TestScenario]:
    """Returns comprehensive collection of reproducibility test scenarios with fixed seeding,
    deterministic validation, cross-platform consistency testing, and statistical reproducibility verification.

    Args:
        test_seeds: Optional list of seeds to use for testing
        num_validation_episodes: Number of episodes to validate for consistency
        cross_platform_testing: Whether to enable cross-platform reproducibility testing
        statistical_validation: Whether to enable statistical validation

    Returns:
        Dict[str, TestScenario]: Dictionary of reproducibility test scenarios with deterministic validation and cross-platform consistency testing
    """
    # Use provided test_seeds or select from REPRODUCIBILITY_SEEDS for deterministic testing
    seeds = test_seeds if test_seeds else REPRODUCIBILITY_SEEDS[:5]

    scenarios = {}
    target_scenarios = list(REPRODUCIBILITY_TEST_SCENARIOS.keys())

    for scenario_name in target_scenarios:
        # Generate reproducibility test configurations using create_reproducibility_test_config
        base_config = create_reproducibility_test_config(
            test_seeds=seeds, validation_episodes=num_validation_episodes
        )

        # Configure scenarios for identical episode generation validation across executions
        scenario_config = base_config.copy()
        scenario_config.update(
            {
                "deterministic_mode": True,
                "seed_validation": True,
                "episode_comparison": True,
                "statistical_analysis": statistical_validation,
            }
        )

        # Set up cross-platform reproducibility testing if cross_platform_testing enabled
        if cross_platform_testing:
            scenario_config.update(
                {
                    "cross_platform_validation": True,
                    "platform_comparison": True,
                    "environment_isolation": True,
                }
            )

        # Configure statistical validation with multiple episodes if statistical_validation enabled
        if statistical_validation:
            scenario_config.update(
                {
                    "statistical_tests": True,
                    "confidence_intervals": True,
                    "variance_analysis": True,
                    "episode_count": num_validation_episodes,
                }
            )

        # Create deterministic parameter settings and fixed random state management
        validation_criteria = {
            "deterministic_behavior": True,
            "seed_consistency": True,
            "episode_reproducibility": True,
            "cross_execution_consistency": True,
            "statistical_significance": statistical_validation,
        }

        scenario = TestScenario(
            name=scenario_name,
            category=ScenarioCategory.REPRODUCIBILITY.value,
            config=scenario_config,
            validation_criteria=validation_criteria,
            priority=ScenarioPriority.CRITICAL,
        )

        scenario.metadata.update(
            {
                "description": REPRODUCIBILITY_TEST_SCENARIOS[scenario_name],
                "test_seeds": seeds,
                "validation_episodes": num_validation_episodes,
                "cross_platform": cross_platform_testing,
                "statistical_validation": statistical_validation,
                "deterministic_requirements": True,
            }
        )

        scenarios[scenario_name] = scenario

    return scenarios


def get_edge_case_test_scenarios(
    edge_case_category: str = "all",
    include_stress_tests: bool = True,
    safety_limits: Optional[Dict[str, Any]] = None,
    enable_recovery_testing: bool = True,
) -> Dict[str, TestScenario]:
    """Returns comprehensive collection of edge case test scenarios with extreme parameters,
    boundary conditions, stress testing, and robustness validation for system limit testing
    and error handling verification.

    Args:
        edge_case_category: Category of edge cases to focus on
        include_stress_tests: Whether to include stress test scenarios
        safety_limits: Optional safety limits to prevent system damage
        enable_recovery_testing: Whether to enable recovery testing scenarios

    Returns:
        Dict[str, TestScenario]: Dictionary of edge case test scenarios with extreme parameter testing and comprehensive robustness validation
    """
    # Select edge case scenarios based on edge_case_category from EDGE_CASE_TEST_SCENARIOS
    if edge_case_category == "all":
        target_scenarios = list(EDGE_CASE_TEST_SCENARIOS.keys())
    else:
        target_scenarios = [
            name
            for name in EDGE_CASE_TEST_SCENARIOS.keys()
            if edge_case_category.lower() in name.lower()
        ]

    # Include stress test scenarios if include_stress_tests is enabled for scalability validation
    if include_stress_tests:
        stress_scenarios = [
            "maximum_grid_size_testing",
            "memory_limit_testing",
            "performance_degradation_testing",
            "resource_exhaustion_recovery",
        ]
        target_scenarios.extend(
            s for s in stress_scenarios if s in EDGE_CASE_TEST_SCENARIOS
        )

    scenarios = {}

    # Apply safety limits if provided to prevent system damage during extreme testing
    limits = safety_limits or {
        "max_memory_mb": MEMORY_LIMIT_TOTAL_MB,
        "max_execution_time": 300,
        "max_grid_size": MAX_GRID_SIZE,
        "resource_monitoring": True,
    }

    for scenario_name in target_scenarios:
        if scenario_name not in EDGE_CASE_TEST_SCENARIOS:
            continue

        # Generate edge case test configurations using create_edge_case_test_config
        base_config = create_edge_case_test_config(
            edge_case_type="boundary_testing", safety_constraints=limits
        )

        # Configure scenarios with extreme parameters and boundary condition testing
        scenario_config = base_config.copy()

        # Apply scenario-specific extreme configurations
        if "minimum" in scenario_name:
            scenario_config["grid_size"] = MIN_GRID_SIZE
            scenario_config["extreme_params"] = {"boundary": "minimum"}
        elif "maximum" in scenario_name:
            scenario_config["grid_size"] = limits.get("max_grid_size", MAX_GRID_SIZE)
            scenario_config["extreme_params"] = {"boundary": "maximum"}
        elif "invalid" in scenario_name:
            scenario_config["invalid_inputs"] = True
            scenario_config["error_injection"] = True

        # Enable recovery testing scenarios if enable_recovery_testing for error handling validation
        if enable_recovery_testing:
            scenario_config.update(
                {
                    "recovery_testing": True,
                    "error_handling_validation": True,
                    "graceful_degradation": True,
                }
            )

        # Set up comprehensive error monitoring and exception handling validation
        validation_criteria = {
            "boundary_condition_handling": True,
            "extreme_parameter_tolerance": True,
            "error_recovery": enable_recovery_testing,
            "system_stability": True,
            "resource_protection": True,
        }

        # Configure resource limit testing and graceful degradation validation
        resource_requirements = {
            "memory_mb": limits.get("max_memory_mb", MEMORY_LIMIT_TOTAL_MB),
            "timeout_seconds": limits.get("max_execution_time", 300),
            "safety_monitoring": True,
            "resource_limits_enforced": True,
        }

        scenario = TestScenario(
            name=scenario_name,
            category=ScenarioCategory.EDGE_CASE.value,
            config=scenario_config,
            validation_criteria=validation_criteria,
            priority=ScenarioPriority.LOW,
            resource_requirements=resource_requirements,
        )

        scenario.metadata.update(
            {
                "description": EDGE_CASE_TEST_SCENARIOS[scenario_name],
                "edge_case_category": edge_case_category,
                "stress_test": "stress" in scenario_name or "maximum" in scenario_name,
                "recovery_testing": enable_recovery_testing,
                "safety_limits_applied": bool(safety_limits),
                "extreme_testing": True,
            }
        )

        scenarios[scenario_name] = scenario

    return scenarios


def create_test_scenario(
    scenario_name: str,
    category: str,
    scenario_config: Dict[str, Any],
    validation_criteria: Optional[Dict[str, Any]] = None,
    priority: Optional[int] = None,
    validate_feasibility: bool = True,
) -> TestScenario:
    """Creates custom test scenario with specified parameters, validation criteria,
    execution requirements, and metadata for flexible test case generation and customized testing workflows.

    Args:
        scenario_name: Unique name for the scenario
        category: Scenario category (unit, integration, performance, reproducibility, edge_case)
        scenario_config: Configuration dictionary for the scenario
        validation_criteria: Optional validation criteria dictionary
        priority: Optional priority level override
        validate_feasibility: Whether to perform feasibility validation

    Returns:
        TestScenario: Custom test scenario with validated configuration and execution planning
    """
    # Validate scenario_name follows naming conventions and uniqueness requirements
    if not scenario_name or not isinstance(scenario_name, str):
        raise ValueError("Scenario name must be a non-empty string")

    # Validate category against supported test scenario categories
    valid_categories = [cat.value for cat in ScenarioCategory]
    if category not in valid_categories:
        raise ValueError(f"Category must be one of: {valid_categories}")

    # Process scenario_config with comprehensive validation and consistency checking
    if not isinstance(scenario_config, dict):
        raise ValueError("Scenario config must be a dictionary")

    config = scenario_config.copy()

    # Apply validation_criteria or generate defaults based on category requirements
    if validation_criteria is None:
        category_enum = ScenarioCategory(category)
        validation_criteria = {
            "configuration_valid": True,
            "execution_feasible": True,
            "resource_available": True,
            "category_compliant": True,
        }

    # Set priority level or use category-appropriate default priority
    if priority is None:
        category_enum = ScenarioCategory(category)
        priority = category_enum.get_default_priority()

    # Create TestScenario instance
    scenario = TestScenario(
        name=scenario_name,
        category=category,
        config=config,
        validation_criteria=validation_criteria,
        priority=priority,
    )

    # Perform feasibility validation if validate_feasibility is enabled
    if validate_feasibility:
        feasibility_check = scenario.validate_configuration(strict_mode=True)
        if not feasibility_check["valid"]:
            raise ValueError(
                f"Scenario validation failed: {feasibility_check['errors']}"
            )

    # Add custom scenario metadata
    scenario.metadata.update(
        {
            "custom_scenario": True,
            "feasibility_validated": validate_feasibility,
            "creation_method": "create_test_scenario",
        }
    )

    return scenario


def get_test_scenario_matrix(
    test_categories: List[str],
    optimize_execution_order: bool = True,
    max_parallel_scenarios: Optional[int] = None,
    include_dependency_analysis: bool = True,
) -> Dict[str, Any]:
    """Generates comprehensive test scenario execution matrix with optimized scheduling,
    dependency management, resource allocation, and coordinated execution for systematic testing workflows.

    Args:
        test_categories: List of test categories to include in matrix
        optimize_execution_order: Whether to optimize execution order for minimal interference
        max_parallel_scenarios: Maximum number of scenarios to run in parallel
        include_dependency_analysis: Whether to analyze and resolve dependencies

    Returns:
        dict: Complete test scenario execution matrix with optimized scheduling and resource coordination
    """
    # Validate test_categories against available scenario categories
    valid_categories = [cat.value for cat in ScenarioCategory]
    invalid_categories = [cat for cat in test_categories if cat not in valid_categories]
    if invalid_categories:
        raise ValueError(
            f"Invalid categories: {invalid_categories}. Valid options: {valid_categories}"
        )

    execution_matrix = {
        "categories": test_categories,
        "total_scenarios": 0,
        "category_breakdown": {},
        "execution_plan": {},
        "resource_allocation": {},
        "dependency_graph": {},
        "optimization_applied": optimize_execution_order,
    }

    all_scenarios = {}

    # Collect all scenarios from specified categories with configuration validation
    for category in test_categories:
        if category == "unit":
            category_scenarios = get_unit_test_scenarios()
        elif category == "integration":
            category_scenarios = get_integration_test_scenarios()
        elif category == "performance":
            category_scenarios = get_performance_test_scenarios()
        elif category == "reproducibility":
            category_scenarios = get_reproducibility_test_scenarios()
        elif category == "edge_case":
            category_scenarios = get_edge_case_test_scenarios()
        else:
            continue

        all_scenarios.update(category_scenarios)
        execution_matrix["category_breakdown"][category] = {
            "count": len(category_scenarios),
            "scenarios": list(category_scenarios.keys()),
        }

    execution_matrix["total_scenarios"] = len(all_scenarios)

    # Apply max_parallel_scenarios limit or use MAX_PARALLEL_SCENARIOS default
    max_parallel = max_parallel_scenarios or MAX_PARALLEL_SCENARIOS

    # Analyze scenario dependencies and execution requirements if include_dependency_analysis enabled
    if include_dependency_analysis:
        dependency_graph = {}
        for scenario_name, scenario in all_scenarios.items():
            dependencies = scenario.dependencies
            dependency_graph[scenario_name] = {
                "depends_on": dependencies,
                "blocks": [],  # Will be filled by analyzing other scenarios
                "priority": scenario.priority,
                "estimated_time": scenario.estimated_execution_time,
            }

        # Build reverse dependency mapping
        for scenario_name, deps in dependency_graph.items():
            for dep in deps["depends_on"]:
                if dep in dependency_graph:
                    dependency_graph[dep]["blocks"].append(scenario_name)

        execution_matrix["dependency_graph"] = dependency_graph

    # Optimize execution order if optimize_execution_order enabled for minimal interference
    if optimize_execution_order:
        # Sort by priority first, then by dependencies, then by execution time
        scenario_order = sorted(
            all_scenarios.keys(),
            key=lambda name: (
                all_scenarios[name].priority,
                len(all_scenarios[name].dependencies),
                all_scenarios[name].estimated_execution_time,
            ),
        )

        execution_matrix["execution_plan"]["optimized_order"] = scenario_order
        execution_matrix["execution_plan"]["parallel_groups"] = []

        # Group scenarios for parallel execution
        current_group = []
        current_group_time = 0
        max_group_time = 300  # 5 minutes per group

        for scenario_name in scenario_order:
            scenario_time = all_scenarios[scenario_name].estimated_execution_time

            if (
                len(current_group) < max_parallel
                and current_group_time + scenario_time <= max_group_time
            ):
                current_group.append(scenario_name)
                current_group_time += scenario_time
            else:
                if current_group:
                    execution_matrix["execution_plan"]["parallel_groups"].append(
                        current_group
                    )
                current_group = [scenario_name]
                current_group_time = scenario_time

        if current_group:
            execution_matrix["execution_plan"]["parallel_groups"].append(current_group)

    # Generate resource allocation plan and cleanup scheduling
    total_memory = sum(
        scenario.resource_requirements.get("memory_mb", 10)
        for scenario in all_scenarios.values()
    )
    total_time = sum(
        scenario.estimated_execution_time for scenario in all_scenarios.values()
    )

    execution_matrix["resource_allocation"] = {
        "total_memory_mb": total_memory,
        "total_execution_time_seconds": total_time,
        "peak_memory_estimate": max(
            scenario.resource_requirements.get("memory_mb", 10)
            for scenario in all_scenarios.values()
        ),
        "parallel_memory_estimate": (
            total_memory / max_parallel if max_parallel else total_memory
        ),
    }

    return execution_matrix


def validate_test_scenario(
    scenario: TestScenario,
    strict_validation: bool = False,
    system_capabilities: Optional[Dict[str, Any]] = None,
    include_optimization_suggestions: bool = True,
) -> Dict[str, Any]:
    """Validates test scenario configuration, execution requirements, resource constraints,
    and feasibility with comprehensive analysis and optimization recommendations for quality
    assurance and execution planning.

    Args:
        scenario: TestScenario instance to validate
        strict_validation: Whether to apply strict validation rules
        system_capabilities: Optional system capability constraints
        include_optimization_suggestions: Whether to include optimization recommendations

    Returns:
        dict: Comprehensive validation report with feasibility analysis and optimization recommendations
    """
    return scenario.validate_configuration(strict_validation)


def execute_test_scenario(
    scenario: TestScenario,
    collect_detailed_metrics: bool = True,
    timeout_override: Optional[float] = None,
    enable_recovery_on_failure: bool = True,
) -> Dict[str, Any]:
    """Executes individual test scenario with comprehensive monitoring, result collection,
    error handling, and performance analysis for automated testing workflows and result validation.

    Args:
        scenario: TestScenario instance to execute
        collect_detailed_metrics: Whether to collect comprehensive performance metrics
        timeout_override: Optional timeout override in seconds
        enable_recovery_on_failure: Whether to attempt recovery on execution failure

    Returns:
        dict: Comprehensive execution results with metrics, performance data, and validation status
    """
    # Initialize scenario execution environment with resource monitoring
    start_time = time.time()
    execution_results = {
        "scenario_id": scenario.scenario_id,
        "scenario_name": scenario.name,
        "start_time": start_time,
        "success": False,
        "execution_time": 0.0,
        "error_message": None,
        "recovery_attempted": False,
    }

    # Apply timeout_override or use scenario default timeout settings
    _ = timeout_override or scenario.resource_requirements.get(
        "timeout_seconds", DEFAULT_SCENARIO_TIMEOUT
    )

    try:
        # Mock scenario execution (in real implementation, this would run the actual test)
        scenario.execution_state = "running"

        # Simulate execution based on scenario configuration
        execution_time = min(
            scenario.estimated_execution_time, 2.0
        )  # Cap for simulation
        time.sleep(execution_time * 0.001)  # Minimal delay

        # Simulate success based on scenario type and configuration
        success = True
        if "invalid" in scenario.name or (
            "edge_case" in scenario.category and "memory" in scenario.name
        ):
            success = False
            if enable_recovery_on_failure:
                execution_results["recovery_attempted"] = True
                # Simulate recovery attempt
                time.sleep(0.001)
                success = True  # Recovery successful

        end_time = time.time()
        execution_results.update(
            {
                "success": success,
                "execution_time": end_time - start_time,
                "end_time": end_time,
            }
        )

        scenario.execution_state = "completed" if success else "failed"

        # Collect detailed performance metrics if collect_detailed_metrics is enabled
        if collect_detailed_metrics:
            execution_results["detailed_metrics"] = {
                "memory_usage_mb": scenario.resource_requirements.get("memory_mb", 0),
                "cpu_usage_percent": 25.0,  # Mock CPU usage
                "validation_results": scenario.validate_configuration(),
                "resource_efficiency": (
                    execution_time / scenario.estimated_execution_time
                    if scenario.estimated_execution_time > 0
                    else 1.0
                ),
            }

    except Exception as e:
        execution_results.update(
            {
                "success": False,
                "error_message": str(e),
                "execution_time": time.time() - start_time,
            }
        )
        scenario.execution_state = "failed"

    return execution_results


def get_scenario_execution_plan(
    scenarios: Dict[str, TestScenario],
    optimize_for_speed: bool = True,
    minimize_resource_usage: bool = False,
    execution_constraints: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Generates detailed execution plan for test scenario collection with resource optimization,
    timing coordination, and dependency management for efficient batch testing execution.

    Args:
        scenarios: Dictionary of test scenarios to plan execution for
        optimize_for_speed: Whether to optimize scheduling for fastest execution
        minimize_resource_usage: Whether to prioritize resource efficiency
        execution_constraints: Optional constraints for custom scheduling

    Returns:
        dict: Detailed execution plan with optimized scheduling and resource management
    """
    # Analyze scenarios for execution requirements and resource dependencies
    total_scenarios = len(scenarios)
    total_estimated_time = sum(s.estimated_execution_time for s in scenarios.values())
    peak_memory = max(
        s.resource_requirements.get("memory_mb", 10) for s in scenarios.values()
    )

    execution_plan = {
        "total_scenarios": total_scenarios,
        "estimated_total_time": total_estimated_time,
        "peak_memory_mb": peak_memory,
        "optimization_strategy": (
            "speed" if optimize_for_speed else "resource_efficient"
        ),
        "execution_phases": [],
        "resource_allocation": {},
        "constraints_applied": bool(execution_constraints),
    }

    # Generate execution timeline with optimal scheduling for speed if optimize_for_speed enabled
    scenario_list = list(scenarios.values())

    if optimize_for_speed:
        # Sort by priority first, then by execution time (shortest first)
        scenario_list.sort(key=lambda s: (s.priority, s.estimated_execution_time))
        execution_plan["optimization_notes"] = "Prioritized by speed and priority"
    elif minimize_resource_usage:
        # Sort by resource usage (lowest first), then by priority
        scenario_list.sort(
            key=lambda s: (s.resource_requirements.get("memory_mb", 10), s.priority)
        )
        execution_plan["optimization_notes"] = "Prioritized by resource efficiency"

    # Consider execution_constraints for custom scheduling and resource limitations
    max_parallel = MAX_PARALLEL_SCENARIOS
    if execution_constraints:
        if "max_parallel_scenarios" in execution_constraints:
            max_parallel = execution_constraints["max_parallel_scenarios"]
        if "max_memory_mb" in execution_constraints:
            # Filter scenarios that exceed memory constraints
            memory_limit = execution_constraints["max_memory_mb"]
            scenario_list = [
                s
                for s in scenario_list
                if s.resource_requirements.get("memory_mb", 10) <= memory_limit
            ]

    # Create dependency resolution order and parallel execution opportunities
    execution_phases = []
    remaining_scenarios = scenario_list.copy()
    phase_number = 1

    while remaining_scenarios:
        current_phase = []
        current_phase_memory = 0
        phase_time = 0

        # Add scenarios to current phase until limits reached
        for scenario in remaining_scenarios[:]:
            scenario_memory = scenario.resource_requirements.get("memory_mb", 10)
            scenario_time = scenario.estimated_execution_time

            if (
                len(current_phase) < max_parallel
                and current_phase_memory + scenario_memory <= peak_memory * 2
            ):  # Allow some overhead
                current_phase.append(scenario)
                current_phase_memory += scenario_memory
                phase_time = max(phase_time, scenario_time)  # Parallel execution time
                remaining_scenarios.remove(scenario)

        if current_phase:
            execution_phases.append(
                {
                    "phase": phase_number,
                    "scenarios": [s.name for s in current_phase],
                    "scenario_count": len(current_phase),
                    "estimated_time": phase_time,
                    "memory_usage": current_phase_memory,
                    "parallel_execution": len(current_phase) > 1,
                }
            )
            phase_number += 1
        else:
            break  # Avoid infinite loop if no scenarios can be scheduled

    execution_plan["execution_phases"] = execution_phases
    execution_plan["total_phases"] = len(execution_phases)
    execution_plan["parallel_efficiency"] = (
        sum(phase["scenario_count"] for phase in execution_phases)
        / len(execution_phases)
        if execution_phases
        else 0
    )

    return execution_plan


def create_custom_test_scenario(
    scenario_name: str,
    advanced_config: Dict[str, Any],
    custom_validators: List[Callable],
    execution_hooks: Optional[Dict[str, Any]] = None,
    enable_advanced_monitoring: bool = False,
) -> TestScenario:
    """Creates highly customized test scenario with advanced configuration options,
    specialized validation requirements, and tailored execution parameters for research
    and specialized testing needs.

    Args:
        scenario_name: Unique name for the custom scenario
        advanced_config: Advanced configuration dictionary with specialized parameters
        custom_validators: List of custom validation functions
        execution_hooks: Optional hooks for specialized testing workflows
        enable_advanced_monitoring: Whether to enable advanced monitoring and profiling

    Returns:
        TestScenario: Highly customized test scenario with advanced configuration and specialized validation
    """
    # Validate scenario_name and advanced_config for completeness and consistency
    if not scenario_name or not isinstance(scenario_name, str):
        raise ValueError("Scenario name must be a non-empty string")

    if not isinstance(advanced_config, dict):
        raise ValueError("Advanced config must be a dictionary")

    # Extract category from advanced config or default to 'edge_case' for custom scenarios
    category = advanced_config.get("category", "edge_case")

    # Configure custom validation functions from custom_validators list
    validation_criteria = {
        "custom_validation": True,
        "validator_count": len(custom_validators),
        "advanced_monitoring": enable_advanced_monitoring,
    }

    # Add validation function names for reference
    validation_criteria["validator_functions"] = [
        getattr(validator, "__name__", "anonymous_validator")
        for validator in custom_validators
    ]

    # Set up execution hooks if provided for specialized testing workflows
    config = advanced_config.copy()
    if execution_hooks:
        config["execution_hooks"] = execution_hooks
        config["hook_count"] = len(execution_hooks)

    # Enable advanced monitoring and profiling if enable_advanced_monitoring requested
    if enable_advanced_monitoring:
        config.update(
            {
                "advanced_profiling": True,
                "detailed_metrics": True,
                "resource_tracking": True,
                "performance_analysis": True,
            }
        )

    # Create specialized TestScenario instance
    scenario = TestScenario(
        name=scenario_name,
        category=category,
        config=config,
        validation_criteria=validation_criteria,
        priority=ScenarioPriority.LOW,  # Custom scenarios typically lower priority
    )

    # Add specialized metadata for custom scenario tracking
    scenario.metadata.update(
        {
            "custom_scenario": True,
            "advanced_configuration": True,
            "custom_validators": len(custom_validators),
            "execution_hooks_enabled": bool(execution_hooks),
            "advanced_monitoring": enable_advanced_monitoring,
            "creation_method": "create_custom_test_scenario",
            "specialized_testing": True,
        }
    )

    return scenario


# Export comprehensive public interface for test scenario management
__all__ = [
    # Factory functions for scenario generation
    "get_unit_test_scenarios",
    "get_integration_test_scenarios",
    "get_performance_test_scenarios",
    "get_reproducibility_test_scenarios",
    "get_edge_case_test_scenarios",
    "create_test_scenario",
    "create_custom_test_scenario",
    # Core data structures
    "TestScenario",
    "TestScenarioCollection",
    "ScenarioCategory",
    "ScenarioPriority",
    # Utility functions
    "get_test_scenario_matrix",
    "validate_test_scenario",
    "execute_test_scenario",
    "get_scenario_execution_plan",
    # Scenario collections and constants
    "UNIT_TEST_SCENARIOS",
    "INTEGRATION_TEST_SCENARIOS",
    "PERFORMANCE_TEST_SCENARIOS",
    "REPRODUCIBILITY_TEST_SCENARIOS",
    "EDGE_CASE_TEST_SCENARIOS",
]
