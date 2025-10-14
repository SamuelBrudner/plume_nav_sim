"""
Comprehensive benchmark data module providing performance baselines, scalability test scenarios,
statistical analysis data, and validation datasets for plume_nav_sim environment performance testing.

Supplies standardized benchmark scenarios, performance targets, baseline measurements, and statistical
reference data for automated performance validation, regression detection, and system optimization
across different configurations and hardware platforms.

This module establishes the foundational benchmark data infrastructure supporting automated performance
testing, continuous integration validation, and research-scale performance analysis with comprehensive
statistical analysis, trend detection, and optimization recommendations.
"""

import copy  # >=3.10 - Deep copying of benchmark configurations and data structures for scenario generation
import datetime  # >=3.10 - Timestamp management for benchmark data versioning, temporal analysis, and result tracking
import json  # >=3.10 - Benchmark data serialization, baseline storage, and configuration persistence for analysis
import statistics  # >=3.10 - Statistical functions for baseline calculation including mean, median, standard deviation, and confidence intervals
from dataclasses import (  # >=3.10 - Data structure definitions for benchmark data containers and performance baseline objects
    dataclass,
    field,
)
from typing import (  # >=3.10 - Type hints for benchmark data structures, analysis functions, and statistical calculations
    Any,
    Dict,
    List,
    Optional,
    Tuple,
)

# External imports with version comments
import numpy as np  # >=2.1.0 - Statistical analysis, array operations, and mathematical calculations for benchmark data generation and analysis

# Import example configurations for benchmark reference
from config.default_config import get_complete_default_config

# Internal imports from plume_nav_sim core modules
from plume_nav_sim.core.constants import (  # PERFORMANCE_TARGET_EPISODE_RESET_MS,  # Not available in constants; omit/reset metric
    DEFAULT_GRID_SIZE,
    MEMORY_LIMIT_TOTAL_MB,
    PERFORMANCE_TARGET_HUMAN_RENDER_MS,
    PERFORMANCE_TARGET_RGB_RENDER_MS,
    PERFORMANCE_TARGET_STEP_LATENCY_MS,
)

# Global benchmark data constants and configuration
BENCHMARK_DATA_VERSION = "1.0.0"
BASELINE_MEASUREMENT_CONFIDENCE = 0.95
DEFAULT_BENCHMARK_ITERATIONS = 1000
SCALABILITY_TEST_GRID_SIZES = [(32, 32), (64, 64), (128, 128), (256, 256), (512, 512)]
PERFORMANCE_TOLERANCE_PERCENT = 10.0
STATISTICAL_OUTLIER_THRESHOLD = 2.0

# Performance target definitions for all benchmark categories and validation
PERFORMANCE_TARGETS = {
    "step_latency_ms": PERFORMANCE_TARGET_STEP_LATENCY_MS,
    "episode_reset_ms": 10.0,
    "rgb_render_ms": PERFORMANCE_TARGET_RGB_RENDER_MS,
    "human_render_ms": PERFORMANCE_TARGET_HUMAN_RENDER_MS,
    "memory_usage_mb": MEMORY_LIMIT_TOTAL_MB,
    "plume_generation_ms": 10.0,  # Target for plume field generation
    "environment_creation_ms": 50.0,  # Target for environment instantiation
    "configuration_validation_ms": 5.0,  # Target for configuration validation
}


def get_quick_start_config() -> Any:
    """Return a default configuration suitable for quick benchmarking."""

    return get_complete_default_config()


@dataclass
class PerformanceBaseline:
    """
    Comprehensive performance baseline data structure containing statistical analysis, target validation,
    measurement metadata, and comparison utilities for benchmark validation and regression detection
    across different system configurations and performance metrics.

    Provides statistical analysis, confidence intervals, target compliance checking, and baseline
    comparison functionality for automated performance validation and continuous performance monitoring.
    """

    name: str
    measurements: Dict[str, List[float]]
    statistics: Dict[str, Dict[str, float]]
    targets: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    targets_met: bool = field(default=False)
    system_fingerprint: Optional[str] = field(default=None)
    created_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    version: str = field(default=BENCHMARK_DATA_VERSION)

    def __post_init__(self):
        """Initialize performance baseline with measurements, statistical analysis, target validation, and metadata for comprehensive benchmark comparison and regression detection."""
        # Calculate confidence intervals for all metrics
        for metric_name, values in self.measurements.items():
            if len(values) >= 2:  # Need at least 2 measurements for confidence interval
                try:
                    mean = statistics.mean(values)
                    if len(values) >= 3:
                        stdev = statistics.stdev(values)
                        # Calculate confidence interval using t-distribution approximation
                        margin_error = 1.96 * (
                            stdev / np.sqrt(len(values))
                        )  # 95% confidence
                        self.confidence_intervals[metric_name] = (
                            max(0.0, mean - margin_error),
                            mean + margin_error,
                        )
                    else:
                        # For small samples, use wider confidence interval
                        self.confidence_intervals[metric_name] = (
                            min(values),
                            max(values),
                        )
                except statistics.StatisticsError:
                    # Handle edge case with identical values
                    self.confidence_intervals[metric_name] = (values[0], values[0])

        # Validate all targets are met within tolerance
        self.targets_met = self._check_targets_compliance()

        # Generate system fingerprint for platform identification
        if self.system_fingerprint is None:
            self.system_fingerprint = self._generate_system_fingerprint()

    def _check_targets_compliance(self) -> bool:
        """Check if all performance measurements meet their targets within tolerance."""
        for metric_name in self.measurements:
            if not self.is_within_target(metric_name):
                return False
        return True

    def _generate_system_fingerprint(self) -> str:
        """Generate unique system fingerprint for platform-specific baseline identification."""
        import hashlib
        import platform

        # Create fingerprint from system information
        system_info = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "baseline_version": self.version,
            "measurement_count": sum(
                len(values) for values in self.measurements.values()
            ),
        }

        fingerprint_str = json.dumps(system_info, sort_keys=True)
        return hashlib.md5(fingerprint_str.encode()).hexdigest()[:16]

    def is_within_target(
        self, metric_name: str, tolerance_percent: Optional[float] = None
    ) -> bool:
        """
        Validates if baseline measurements meet performance targets within acceptable tolerance
        for regression detection and performance compliance checking.

        Args:
            metric_name: Name of performance metric to check
            tolerance_percent: Tolerance percentage, defaults to PERFORMANCE_TOLERANCE_PERCENT

        Returns:
            bool: True if metric meets target within tolerance, False otherwise
        """
        if metric_name not in self.measurements or metric_name not in self.targets:
            return False

        if tolerance_percent is None:
            tolerance_percent = PERFORMANCE_TOLERANCE_PERCENT

        # Get measurement statistics
        values = self.measurements[metric_name]
        if not values:
            return False

        mean_value = self.statistics.get(metric_name, {}).get(
            "mean", statistics.mean(values)
        )
        target_value = self.targets[metric_name]

        # Calculate tolerance band
        tolerance_margin = target_value * (tolerance_percent / 100.0)
        upper_limit = target_value + tolerance_margin

        # Check if measurement is within target + tolerance
        return mean_value <= upper_limit

    def get_performance_margin(
        self, metric_name: str, as_percentage: bool = False
    ) -> float:
        """
        Calculates performance margin between baseline measurements and targets with statistical
        confidence for optimization opportunity analysis and performance headroom assessment.

        Args:
            metric_name: Name of performance metric
            as_percentage: Return margin as percentage of target

        Returns:
            float: Performance margin as absolute value or percentage based on as_percentage parameter
        """
        if metric_name not in self.measurements or metric_name not in self.targets:
            return 0.0

        values = self.measurements[metric_name]
        if not values:
            return 0.0

        mean_value = self.statistics.get(metric_name, {}).get(
            "mean", statistics.mean(values)
        )
        target_value = self.targets[metric_name]

        # Calculate margin (positive means performance is better than target)
        margin = target_value - mean_value

        if as_percentage:
            return (margin / target_value) * 100.0 if target_value > 0 else 0.0
        else:
            return margin

    def compare_to_baseline(
        self,
        other_baseline: "PerformanceBaseline",
        include_significance_test: bool = False,
    ) -> Dict[str, Any]:
        """
        Compares current baseline to another baseline with statistical significance testing
        for regression detection and performance trend analysis.

        Args:
            other_baseline: Another PerformanceBaseline for comparison
            include_significance_test: Whether to perform statistical significance testing

        Returns:
            dict: Comprehensive comparison report with statistical analysis and significance testing results
        """
        comparison_report = {
            "comparison_timestamp": datetime.datetime.now().isoformat(),
            "baseline_names": (self.name, other_baseline.name),
            "metric_comparisons": {},
            "overall_performance_change": "unchanged",
            "significant_changes": [],
            "recommendations": [],
        }

        # Compare metrics that exist in both baselines
        common_metrics = set(self.measurements.keys()) & set(
            other_baseline.measurements.keys()
        )

        total_improvement = 0
        total_regression = 0

        for metric_name in common_metrics:
            self_values = self.measurements[metric_name]
            other_values = other_baseline.measurements[metric_name]

            if not self_values or not other_values:
                continue

            self_mean = statistics.mean(self_values)
            other_mean = statistics.mean(other_values)

            # Calculate percentage change (negative means improvement for latency metrics)
            if other_mean > 0:
                change_percent = ((self_mean - other_mean) / other_mean) * 100.0
            else:
                change_percent = 0.0

            # Determine if this is improvement or regression
            is_latency_metric = any(
                keyword in metric_name.lower()
                for keyword in ["latency", "time", "ms", "duration"]
            )

            if is_latency_metric:
                # For latency metrics, lower is better
                performance_change = (
                    "improvement"
                    if change_percent < -5
                    else ("regression" if change_percent > 5 else "unchanged")
                )
            else:
                # For other metrics, higher might be better (depends on context)
                performance_change = (
                    "improvement"
                    if change_percent > 5
                    else ("regression" if change_percent < -5 else "unchanged")
                )

            metric_comparison = {
                "current_mean": self_mean,
                "baseline_mean": other_mean,
                "change_percent": change_percent,
                "performance_change": performance_change,
                "current_target_compliance": self.is_within_target(metric_name),
                "baseline_target_compliance": other_baseline.is_within_target(
                    metric_name
                ),
            }

            # Add significance test if requested
            if (
                include_significance_test
                and len(self_values) >= 3
                and len(other_values) >= 3
            ):
                try:
                    # Simple t-test approximation
                    self_std = statistics.stdev(self_values)
                    other_std = statistics.stdev(other_values)
                    pooled_std = np.sqrt((self_std**2 + other_std**2) / 2)

                    if pooled_std > 0:
                        t_statistic = abs(self_mean - other_mean) / pooled_std
                        significant = t_statistic > 2.0  # Rough significance threshold
                        metric_comparison["statistically_significant"] = significant

                        if significant:
                            comparison_report["significant_changes"].append(
                                f"{metric_name}: {change_percent:.1f}% change (significant)"
                            )
                    else:
                        metric_comparison["statistically_significant"] = False

                except statistics.StatisticsError:
                    metric_comparison["statistically_significant"] = False

            comparison_report["metric_comparisons"][metric_name] = metric_comparison

            # Track overall performance trends
            if performance_change == "improvement":
                total_improvement += 1
            elif performance_change == "regression":
                total_regression += 1

        # Determine overall performance change
        if total_improvement > total_regression:
            comparison_report["overall_performance_change"] = "improvement"
            comparison_report["recommendations"].append(
                "Performance has generally improved compared to baseline"
            )
        elif total_regression > total_improvement:
            comparison_report["overall_performance_change"] = "regression"
            comparison_report["recommendations"].append(
                "Performance regression detected - investigate optimization opportunities"
            )
        else:
            comparison_report["overall_performance_change"] = "mixed"
            comparison_report["recommendations"].append(
                "Mixed performance changes - review individual metrics for optimization"
            )

        return comparison_report

    def to_dict(self, include_raw_measurements: bool = False) -> Dict[str, Any]:
        """
        Converts performance baseline to dictionary format for serialization, storage,
        and integration with external analysis tools.

        Args:
            include_raw_measurements: Whether to include raw measurement data

        Returns:
            dict: Complete baseline data in dictionary format for JSON serialization and external analysis
        """
        baseline_dict = {
            "name": self.name,
            "statistics": self.statistics,
            "targets": self.targets,
            "confidence_intervals": {
                metric: {"lower": ci[0], "upper": ci[1]}
                for metric, ci in self.confidence_intervals.items()
            },
            "targets_met": self.targets_met,
            "system_fingerprint": self.system_fingerprint,
            "created_at": self.created_at.isoformat(),
            "version": self.version,
            "measurement_summary": {
                metric: {
                    "count": len(values),
                    "mean": statistics.mean(values) if values else 0.0,
                    "min": min(values) if values else 0.0,
                    "max": max(values) if values else 0.0,
                }
                for metric, values in self.measurements.items()
            },
        }

        if include_raw_measurements:
            baseline_dict["raw_measurements"] = self.measurements

        return baseline_dict


@dataclass
class BenchmarkScenario:
    """
    Complete benchmark test scenario specification containing environment configuration, execution
    parameters, measurement protocols, and validation criteria for automated benchmark execution
    with statistical analysis and performance validation.

    Provides comprehensive scenario definition, execution planning, resource estimation, and
    validation configuration for systematic benchmark execution and performance analysis.
    """

    name: str
    config: Dict[str, Any]
    execution_plan: Dict[str, Any]
    validation_criteria: Dict[str, Any]
    performance_targets: Dict[str, float] = field(
        default_factory=lambda: PERFORMANCE_TARGETS.copy()
    )
    memory_profiling_enabled: bool = field(default=True)
    iterations: int = field(default=DEFAULT_BENCHMARK_ITERATIONS)
    estimated_execution_time: float = field(default=0.0)
    dependencies: Optional[Dict[str, Any]] = field(default=None)

    def __post_init__(self):
        """Initialize benchmark scenario with complete configuration, execution planning, and validation criteria for automated benchmark execution and performance analysis."""
        # Calculate estimated execution time
        self.estimated_execution_time = self._calculate_execution_time()

        # Initialize dependencies if not provided
        if self.dependencies is None:
            self.dependencies = {
                "system_requirements": {},
                "configuration_dependencies": [],
            }

        # Validate scenario configuration consistency
        self._validate_scenario_consistency()

    def _calculate_execution_time(self) -> float:
        """Calculate estimated execution time based on iterations and measurement complexity."""
        # Base time per iteration (rough estimate)
        base_time_per_iteration = 0.01  # 10ms per iteration baseline

        # Adjust for configuration complexity
        grid_size = self.config.get("grid_size", DEFAULT_GRID_SIZE)
        if isinstance(grid_size, (tuple, list)):
            grid_complexity = grid_size[0] * grid_size[1]
        else:
            grid_complexity = DEFAULT_GRID_SIZE[0] * DEFAULT_GRID_SIZE[1]

        complexity_multiplier = max(
            1.0, grid_complexity / (128 * 128)
        )  # Normalize to 128x128

        # Adjust for memory profiling overhead
        profiling_overhead = 1.5 if self.memory_profiling_enabled else 1.0

        # Calculate total estimated time in seconds
        estimated_seconds = (
            self.iterations
            * base_time_per_iteration
            * complexity_multiplier
            * profiling_overhead
        )

        return estimated_seconds

    def _validate_scenario_consistency(self) -> None:
        """Validate scenario configuration for consistency and feasibility."""
        required_config_keys = ["grid_size", "source_location", "plume_sigma"]
        for key in required_config_keys:
            if key not in self.config:
                raise ValueError(
                    f"Required configuration key '{key}' missing from scenario config"
                )

        # Validate iterations is reasonable
        if self.iterations <= 0 or self.iterations > 100000:
            raise ValueError(
                f"Iterations {self.iterations} outside reasonable range [1, 100000]"
            )

        # Ensure performance targets are present
        if not self.performance_targets:
            self.performance_targets = PERFORMANCE_TARGETS.copy()

    def get_execution_plan(self, optimize_for_accuracy: bool = False) -> Dict[str, Any]:
        """
        Generates detailed execution plan with timing protocols, measurement procedures,
        and resource requirements for automated benchmark execution and coordination.

        Args:
            optimize_for_accuracy: Enable optimization for statistical precision

        Returns:
            dict: Detailed execution plan with measurement protocols and resource optimization
        """
        # Base execution plan
        plan = {
            "scenario_name": self.name,
            "total_iterations": self.iterations,
            "estimated_duration_seconds": self.estimated_execution_time,
            "measurement_protocols": {
                "warmup_iterations": max(10, self.iterations // 100),
                "cooldown_iterations": 5,
                "measurement_interval": 1,  # Measure every iteration
                "statistical_analysis": True,
            },
            "resource_monitoring": {
                "memory_profiling": self.memory_profiling_enabled,
                "timing_precision": (
                    "microsecond" if optimize_for_accuracy else "millisecond"
                ),
                "system_monitoring": optimize_for_accuracy,
            },
            "data_collection": {
                "collect_raw_measurements": True,
                "calculate_statistics": True,
                "generate_confidence_intervals": True,
                "detect_outliers": True,
                "outlier_threshold": STATISTICAL_OUTLIER_THRESHOLD,
            },
        }

        # Optimize for accuracy if requested
        if optimize_for_accuracy:
            plan["measurement_protocols"].update(
                {
                    "warmup_iterations": max(50, self.iterations // 20),
                    "cooldown_iterations": 20,
                    "repeated_measurements": 3,  # Take multiple measurements per iteration
                    "statistical_validation": True,
                }
            )

            plan["resource_monitoring"].update(
                {
                    "cpu_affinity": True,
                    "process_priority": "high",
                    "disable_background_tasks": True,
                }
            )

        # Add scenario-specific configuration
        plan["scenario_config"] = self.config.copy()
        plan["validation_criteria"] = self.validation_criteria.copy()

        return plan

    def validate_scenario(
        self, check_resource_requirements: bool = False
    ) -> Dict[str, Any]:
        """
        Validates benchmark scenario configuration, feasibility, and resource requirements
        with comprehensive analysis and recommendation generation for execution optimization.

        Args:
            check_resource_requirements: Whether to check system resource requirements

        Returns:
            dict: Scenario validation results with feasibility analysis and optimization recommendations
        """
        validation_results = {
            "scenario_name": self.name,
            "validation_timestamp": datetime.datetime.now().isoformat(),
            "is_valid": True,
            "validation_errors": [],
            "validation_warnings": [],
            "resource_analysis": {},
            "optimization_recommendations": [],
        }

        # Validate basic configuration
        try:
            # Check grid size validity
            grid_size = self.config.get("grid_size", DEFAULT_GRID_SIZE)
            if isinstance(grid_size, (tuple, list)):
                if len(grid_size) != 2 or any(dim <= 0 for dim in grid_size):
                    validation_results["validation_errors"].append(
                        f"Invalid grid size: {grid_size}"
                    )
                    validation_results["is_valid"] = False

            # Check iterations
            if self.iterations <= 0:
                validation_results["validation_errors"].append(
                    f"Invalid iterations count: {self.iterations}"
                )
                validation_results["is_valid"] = False
            elif self.iterations < 10:
                validation_results["validation_warnings"].append(
                    f"Low iteration count {self.iterations} may reduce statistical confidence"
                )

            # Check performance targets
            for metric, target in self.performance_targets.items():
                if not isinstance(target, (int, float)) or target <= 0:
                    validation_results["validation_errors"].append(
                        f"Invalid performance target for {metric}: {target}"
                    )
                    validation_results["is_valid"] = False

        except Exception as e:
            validation_results["validation_errors"].append(
                f"Configuration validation error: {e}"
            )
            validation_results["is_valid"] = False

        # Resource requirements check
        if check_resource_requirements:
            try:
                resource_estimate = self.estimate_resources()
                validation_results["resource_analysis"] = resource_estimate

                # Check memory requirements
                estimated_memory = resource_estimate.get("estimated_memory_mb", 0)
                if estimated_memory > MEMORY_LIMIT_TOTAL_MB:
                    validation_results["validation_warnings"].append(
                        f"Estimated memory usage {estimated_memory:.1f}MB exceeds limit {MEMORY_LIMIT_TOTAL_MB}MB"
                    )

                # Check execution time
                if self.estimated_execution_time > 300:  # 5 minutes
                    validation_results["validation_warnings"].append(
                        f"Long execution time estimated: {self.estimated_execution_time:.1f}s"
                    )
                    validation_results["optimization_recommendations"].append(
                        "Consider reducing iterations for faster execution"
                    )

            except Exception as e:
                validation_results["validation_warnings"].append(
                    f"Resource analysis failed: {e}"
                )

        # Generate optimization recommendations
        if validation_results["is_valid"]:
            if self.iterations > 1000:
                validation_results["optimization_recommendations"].append(
                    "Consider parallel execution for large iteration counts"
                )

            if not self.memory_profiling_enabled:
                validation_results["optimization_recommendations"].append(
                    "Enable memory profiling for comprehensive performance analysis"
                )

        return validation_results

    def estimate_resources(
        self, system_specs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Estimates resource requirements including memory usage, execution time, and system load
        for benchmark scenario planning and resource allocation optimization.

        Args:
            system_specs: Optional system specifications for platform-specific estimates

        Returns:
            dict: Resource estimation including memory, time, and system requirements
        """
        # Base resource estimates
        grid_size = self.config.get("grid_size", DEFAULT_GRID_SIZE)
        if isinstance(grid_size, (tuple, list)):
            grid_cells = grid_size[0] * grid_size[1]
        else:
            grid_cells = DEFAULT_GRID_SIZE[0] * DEFAULT_GRID_SIZE[1]

        # Memory estimation
        base_memory_mb = 10.0  # Base environment memory
        field_memory_mb = (grid_cells * 4) / (1024 * 1024)  # float32 field
        profiling_overhead = 5.0 if self.memory_profiling_enabled else 0.0

        estimated_memory_mb = base_memory_mb + field_memory_mb + profiling_overhead

        # CPU estimation
        estimated_cpu_usage = min(
            100.0, 20.0 + (grid_cells / 10000)
        )  # Rough CPU usage %

        resource_estimate = {
            "estimated_memory_mb": estimated_memory_mb,
            "estimated_cpu_usage_percent": estimated_cpu_usage,
            "estimated_execution_time_seconds": self.estimated_execution_time,
            "grid_complexity": grid_cells,
            "resource_optimization": {
                "memory_efficient": estimated_memory_mb < 50,
                "cpu_intensive": estimated_cpu_usage > 50,
                "time_intensive": self.estimated_execution_time > 60,
            },
        }

        # Apply system-specific adjustments
        if system_specs:
            cpu_cores = system_specs.get("cpu_cores", 1)
            available_memory = system_specs.get("memory_gb", 4) * 1024

            # Adjust estimates based on system specs
            if cpu_cores > 1:
                resource_estimate["parallel_execution_possible"] = True
                resource_estimate["estimated_execution_time_seconds"] /= min(
                    cpu_cores, 4
                )

            resource_estimate["memory_utilization_percent"] = (
                estimated_memory_mb / available_memory
            ) * 100

        return resource_estimate

    def clone_with_modifications(
        self, new_name: str, config_overrides: Dict[str, Any]
    ) -> "BenchmarkScenario":
        """
        Creates modified copy of benchmark scenario with parameter overrides for scenario
        variations and comparative analysis.

        Args:
            new_name: New scenario name
            config_overrides: Configuration parameters to override

        Returns:
            BenchmarkScenario: Cloned scenario with applied modifications and updated configuration
        """
        # Deep copy current configuration and execution plan
        new_config = copy.deepcopy(self.config)
        new_execution_plan = copy.deepcopy(self.execution_plan)
        new_validation_criteria = copy.deepcopy(self.validation_criteria)

        # Apply configuration overrides
        for key, value in config_overrides.items():
            new_config[key] = value

        # Create new scenario with modifications
        new_scenario = BenchmarkScenario(
            name=new_name,
            config=new_config,
            execution_plan=new_execution_plan,
            validation_criteria=new_validation_criteria,
            performance_targets=self.performance_targets.copy(),
            memory_profiling_enabled=self.memory_profiling_enabled,
            iterations=self.iterations,
        )

        return new_scenario


@dataclass
class ScalabilityTestData:
    """
    Comprehensive scalability test data structure containing multi-dimensional scaling analysis,
    resource projections, and performance modeling for system capacity planning and optimization
    validation across different configurations and hardware platforms.

    Provides scaling model generation, performance projections, efficiency analysis, and test
    matrix generation for systematic scalability validation and capacity planning.
    """

    grid_sizes: List[Tuple[int, int]]
    performance_projections: Dict[str, Dict[Tuple[int, int], float]]
    resource_estimates: Dict[str, Dict[Tuple[int, int], float]]
    scaling_analysis: Dict[str, Any] = field(default_factory=dict)
    test_scenarios: List[Dict[str, Any]] = field(default_factory=list)
    optimization_recommendations: Dict[str, List[str]] = field(default_factory=dict)
    scaling_model_type: str = field(default="quadratic")

    def __post_init__(self):
        """Initialize scalability test data with grid size analysis, performance projections, and resource estimation for comprehensive scaling validation and capacity planning."""
        # Validate grid sizes are properly ordered
        self.grid_sizes = sorted(self.grid_sizes, key=lambda gs: gs[0] * gs[1])

        # Generate initial scaling analysis
        self._analyze_scaling_patterns()

        # Generate test scenarios for each grid size
        self._generate_test_scenarios()

        # Initialize optimization recommendations
        self._generate_optimization_recommendations()

    def _analyze_scaling_patterns(self) -> None:
        """Analyze scaling patterns and determine mathematical models."""
        if not self.performance_projections or not self.grid_sizes:
            return

        # Analyze scaling for each performance metric
        for metric_name, projections in self.performance_projections.items():
            if len(projections) < 3:  # Need at least 3 points for analysis
                continue

            # Extract grid complexities and performance values
            complexities = []
            values = []

            for grid_size in self.grid_sizes:
                if grid_size in projections:
                    complexity = grid_size[0] * grid_size[1]
                    complexities.append(complexity)
                    values.append(projections[grid_size])

            if len(complexities) >= 3 and len(values) >= 3:
                # Analyze scaling relationship
                scaling_analysis = self._determine_scaling_relationship(
                    complexities, values
                )
                self.scaling_analysis[metric_name] = scaling_analysis

    def _determine_scaling_relationship(
        self, complexities: List[int], values: List[float]
    ) -> Dict[str, Any]:
        """Determine mathematical relationship between complexity and performance values."""
        # Convert to numpy arrays for analysis
        x = np.array(complexities)
        y = np.array(values)

        # Test different scaling models
        models = {}

        # Linear model: y = ax + b
        try:
            coeffs_linear = np.polyfit(x, y, 1)
            y_pred_linear = np.polyval(coeffs_linear, x)
            r2_linear = 1 - (
                np.sum((y - y_pred_linear) ** 2) / np.sum((y - np.mean(y)) ** 2)
            )
            models["linear"] = {
                "coefficients": coeffs_linear.tolist(),
                "r_squared": r2_linear,
                "formula": f"y = {coeffs_linear[0]:.6f}x + {coeffs_linear[1]:.6f}",
            }
        except Exception:
            models["linear"] = {"r_squared": 0.0}

        # Quadratic model: y = ax² + bx + c
        try:
            coeffs_quad = np.polyfit(x, y, 2)
            y_pred_quad = np.polyval(coeffs_quad, x)
            r2_quad = 1 - (
                np.sum((y - y_pred_quad) ** 2) / np.sum((y - np.mean(y)) ** 2)
            )
            models["quadratic"] = {
                "coefficients": coeffs_quad.tolist(),
                "r_squared": r2_quad,
                "formula": f"y = {coeffs_quad[0]:.6f}x² + {coeffs_quad[1]:.6f}x + {coeffs_quad[2]:.6f}",
            }
        except Exception:
            models["quadratic"] = {"r_squared": 0.0}

        # Logarithmic model: y = a*log(x) + b (if applicable)
        if all(comp > 0 for comp in complexities):
            try:
                log_x = np.log(x)
                coeffs_log = np.polyfit(log_x, y, 1)
                y_pred_log = np.polyval(coeffs_log, log_x)
                r2_log = 1 - (
                    np.sum((y - y_pred_log) ** 2) / np.sum((y - np.mean(y)) ** 2)
                )
                models["logarithmic"] = {
                    "coefficients": coeffs_log.tolist(),
                    "r_squared": r2_log,
                    "formula": f"y = {coeffs_log[0]:.6f}*log(x) + {coeffs_log[1]:.6f}",
                }
            except Exception:
                models["logarithmic"] = {"r_squared": 0.0}

        # Determine best model
        best_model = max(models.keys(), key=lambda k: models[k].get("r_squared", 0))

        return {
            "models": models,
            "best_model": best_model,
            "best_fit_r_squared": models[best_model].get("r_squared", 0),
            "scaling_efficiency": (
                "good" if models[best_model].get("r_squared", 0) > 0.8 else "concerning"
            ),
        }

    def _generate_test_scenarios(self) -> None:
        """Generate test scenarios for each grid size with appropriate configurations."""
        for grid_size in self.grid_sizes:
            scenario = {
                "grid_size": grid_size,
                "complexity": grid_size[0] * grid_size[1],
                "source_location": (
                    grid_size[0] // 2,
                    grid_size[1] // 2,
                ),  # Center source
                "expected_memory_mb": (grid_size[0] * grid_size[1] * 4) / (1024 * 1024),
                "estimated_execution_time_s": self._estimate_execution_time(grid_size),
                "performance_category": self._categorize_grid_size(grid_size),
            }
            self.test_scenarios.append(scenario)

    def _estimate_execution_time(self, grid_size: Tuple[int, int]) -> float:
        """Estimate execution time for specific grid size."""
        complexity = grid_size[0] * grid_size[1]
        base_time = 0.001  # 1ms base time
        scaling_factor = complexity / (128 * 128)  # Normalize to 128x128
        return base_time * scaling_factor * DEFAULT_BENCHMARK_ITERATIONS

    def _categorize_grid_size(self, grid_size: Tuple[int, int]) -> str:
        """Categorize grid size into performance categories."""
        complexity = grid_size[0] * grid_size[1]

        if complexity <= 32 * 32:
            return "small"
        elif complexity <= 128 * 128:
            return "medium"
        elif complexity <= 256 * 256:
            return "large"
        else:
            return "extra_large"

    def _generate_optimization_recommendations(self) -> None:
        """Generate optimization recommendations based on scaling analysis."""
        recommendations = []

        # Analyze memory scaling
        if self.resource_estimates.get("memory_mb"):
            memory_projections = self.resource_estimates["memory_mb"]
            max_memory = max(memory_projections.values())

            if max_memory > MEMORY_LIMIT_TOTAL_MB:
                recommendations.append(
                    "Memory usage exceeds limits for large grids - consider memory optimization"
                )

        # Analyze performance scaling
        for metric_name, analysis in self.scaling_analysis.items():
            best_model = analysis.get("best_model", "unknown")
            r_squared = analysis.get("best_fit_r_squared", 0)

            if r_squared < 0.7:
                recommendations.append(
                    f"Poor scaling model fit for {metric_name} - investigate performance bottlenecks"
                )

            if best_model == "quadratic":
                recommendations.append(
                    f"Quadratic scaling detected for {metric_name} - consider algorithmic optimizations"
                )

        self.optimization_recommendations["general"] = recommendations

    def get_scaling_model(self, metric_name: str) -> Dict[str, Any]:
        """
        Generates mathematical scaling model with performance predictions and complexity
        analysis for capacity planning and optimization decision support.

        Args:
            metric_name: Name of performance metric to model

        Returns:
            dict: Scaling model with mathematical formula and performance predictions
        """
        if metric_name not in self.scaling_analysis:
            return {
                "model_available": False,
                "error": f"No scaling analysis available for metric: {metric_name}",
            }

        analysis = self.scaling_analysis[metric_name]
        best_model = analysis.get("best_model", "unknown")

        scaling_model = {
            "metric_name": metric_name,
            "model_available": True,
            "best_model_type": best_model,
            "model_accuracy": analysis.get("best_fit_r_squared", 0),
            "mathematical_formula": analysis.get("models", {})
            .get(best_model, {})
            .get("formula", "unknown"),
            "coefficients": analysis.get("models", {})
            .get(best_model, {})
            .get("coefficients", []),
            "scaling_efficiency": analysis.get("scaling_efficiency", "unknown"),
            "data_points": len(self.grid_sizes),
        }

        # Add predictions for extended grid sizes if model is good
        if scaling_model["model_accuracy"] > 0.7:
            extended_predictions = self._generate_extended_predictions(
                metric_name, best_model
            )
            scaling_model["extended_predictions"] = extended_predictions

        return scaling_model

    def _generate_extended_predictions(
        self, metric_name: str, model_type: str
    ) -> Dict[str, float]:
        """Generate performance predictions for extended grid sizes."""
        if metric_name not in self.scaling_analysis:
            return {}

        model_data = self.scaling_analysis[metric_name]["models"][model_type]
        coeffs = model_data.get("coefficients", [])

        # Generate predictions for larger grid sizes
        extended_sizes = [(512, 512), (1024, 1024), (2048, 2048)]
        predictions = {}

        for grid_size in extended_sizes:
            complexity = grid_size[0] * grid_size[1]

            try:
                if model_type == "linear" and len(coeffs) >= 2:
                    prediction = coeffs[0] * complexity + coeffs[1]
                elif model_type == "quadratic" and len(coeffs) >= 3:
                    prediction = (
                        coeffs[0] * complexity**2 + coeffs[1] * complexity + coeffs[2]
                    )
                elif model_type == "logarithmic" and len(coeffs) >= 2:
                    prediction = coeffs[0] * np.log(complexity) + coeffs[1]
                else:
                    continue

                predictions[f"{grid_size[0]}x{grid_size[1]}"] = max(0.0, prediction)

            except Exception:
                continue

        return predictions

    def project_performance(
        self,
        target_grid_size: Tuple[int, int],
        include_confidence_intervals: bool = False,
    ) -> Dict[str, Any]:
        """
        Projects performance metrics for specified grid size using scaling models with
        confidence intervals and optimization recommendations.

        Args:
            target_grid_size: Target grid size for performance projection
            include_confidence_intervals: Whether to include confidence intervals

        Returns:
            dict: Performance projections with confidence intervals and resource requirements
        """
        projections = {
            "target_grid_size": target_grid_size,
            "target_complexity": target_grid_size[0] * target_grid_size[1],
            "projections": {},
            "resource_estimates": {},
            "feasibility_analysis": {},
            "recommendations": [],
        }

        # Generate projections for each analyzed metric
        for metric_name in self.scaling_analysis:
            scaling_model = self.get_scaling_model(metric_name)

            if (
                scaling_model.get("model_available", False)
                and scaling_model["model_accuracy"] > 0.5
            ):
                model_type = scaling_model["best_model_type"]
                coeffs = scaling_model.get("coefficients", [])
                complexity = projections["target_complexity"]

                # Calculate projection
                try:
                    if model_type == "linear" and len(coeffs) >= 2:
                        projected_value = coeffs[0] * complexity + coeffs[1]
                    elif model_type == "quadratic" and len(coeffs) >= 3:
                        projected_value = (
                            coeffs[0] * complexity**2
                            + coeffs[1] * complexity
                            + coeffs[2]
                        )
                    elif model_type == "logarithmic" and len(coeffs) >= 2:
                        projected_value = coeffs[0] * np.log(complexity) + coeffs[1]
                    else:
                        projected_value = None

                    if projected_value is not None:
                        projection_data = {
                            "projected_value": max(0.0, projected_value),
                            "model_type": model_type,
                            "model_accuracy": scaling_model["model_accuracy"],
                            "extrapolation": target_grid_size not in self.grid_sizes,
                        }

                        # Add confidence intervals if requested
                        if (
                            include_confidence_intervals
                            and scaling_model["model_accuracy"] > 0.7
                        ):
                            # Simple confidence interval based on model accuracy
                            uncertainty = (
                                1 - scaling_model["model_accuracy"]
                            ) * projected_value
                            projection_data["confidence_interval"] = {
                                "lower": max(0.0, projected_value - uncertainty),
                                "upper": projected_value + uncertainty,
                                "confidence_level": 0.8,
                            }

                        projections["projections"][metric_name] = projection_data

                except Exception as e:
                    projections["projections"][metric_name] = {
                        "error": f"Projection calculation failed: {e}"
                    }

        # Resource estimates
        complexity = projections["target_complexity"]
        projected_memory_mb = (complexity * 4) / (
            1024 * 1024
        ) + 10  # Field + base memory
        projected_cpu_usage = min(100.0, 20.0 + (complexity / 10000))

        projections["resource_estimates"] = {
            "memory_mb": projected_memory_mb,
            "cpu_usage_percent": projected_cpu_usage,
            "execution_time_estimate_s": complexity
            * 0.000001
            * DEFAULT_BENCHMARK_ITERATIONS,
        }

        # Feasibility analysis
        projections["feasibility_analysis"] = {
            "memory_feasible": projected_memory_mb <= MEMORY_LIMIT_TOTAL_MB,
            "performance_feasible": all(
                proj.get("projected_value", 0)
                <= PERFORMANCE_TARGETS.get(metric, float("inf"))
                for metric, proj in projections["projections"].items()
                if "projected_value" in proj
            ),
            "recommended_for_testing": (
                projected_memory_mb <= MEMORY_LIMIT_TOTAL_MB
                and complexity <= 1024 * 1024
            ),
        }

        # Generate recommendations
        if not projections["feasibility_analysis"]["memory_feasible"]:
            projections["recommendations"].append(
                "Target grid size exceeds memory limits - consider memory optimization"
            )

        if not projections["feasibility_analysis"]["performance_feasible"]:
            projections["recommendations"].append(
                "Target grid size may not meet performance targets - optimize algorithms"
            )

        return projections

    def analyze_scaling_efficiency(
        self, metrics_to_analyze: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Analyzes scaling efficiency across different metrics with bottleneck identification
        and optimization opportunity assessment for system performance optimization.

        Args:
            metrics_to_analyze: Optional list of metrics to analyze, defaults to all available

        Returns:
            dict: Scaling efficiency analysis with bottleneck identification and optimization opportunities
        """
        if metrics_to_analyze is None:
            metrics_to_analyze = list(self.scaling_analysis.keys())

        efficiency_analysis = {
            "analysis_timestamp": datetime.datetime.now().isoformat(),
            "metrics_analyzed": metrics_to_analyze,
            "overall_efficiency": "unknown",
            "bottlenecks": [],
            "optimization_opportunities": [],
            "scaling_patterns": {},
            "efficiency_scores": {},
        }

        total_efficiency_score = 0.0
        analyzed_metrics = 0

        for metric_name in metrics_to_analyze:
            if metric_name not in self.scaling_analysis:
                continue

            analysis = self.scaling_analysis[metric_name]
            best_model = analysis.get("best_model", "unknown")
            r_squared = analysis.get("best_fit_r_squared", 0)

            # Calculate efficiency score (0-100)
            efficiency_score = r_squared * 100  # Model fit as efficiency indicator

            # Penalize poor scaling models
            if best_model == "quadratic":
                efficiency_score *= 0.7  # Quadratic scaling is less efficient
            elif best_model == "logarithmic":
                efficiency_score *= 1.2  # Logarithmic scaling is efficient

            efficiency_score = min(100.0, max(0.0, efficiency_score))

            efficiency_analysis["efficiency_scores"][metric_name] = efficiency_score
            efficiency_analysis["scaling_patterns"][metric_name] = {
                "model_type": best_model,
                "model_accuracy": r_squared,
                "efficiency_category": (
                    "excellent"
                    if efficiency_score >= 80
                    else (
                        "good"
                        if efficiency_score >= 60
                        else "moderate" if efficiency_score >= 40 else "poor"
                    )
                ),
            }

            # Identify bottlenecks
            if efficiency_score < 40:
                efficiency_analysis["bottlenecks"].append(
                    {
                        "metric": metric_name,
                        "issue": f"Poor scaling efficiency ({efficiency_score:.1f}%)",
                        "model_type": best_model,
                        "recommendation": "Investigate algorithmic improvements",
                    }
                )

            if best_model == "quadratic":
                efficiency_analysis["bottlenecks"].append(
                    {
                        "metric": metric_name,
                        "issue": "Quadratic scaling detected",
                        "model_type": best_model,
                        "recommendation": "Consider algorithm optimization to reduce complexity",
                    }
                )

            total_efficiency_score += efficiency_score
            analyzed_metrics += 1

        # Calculate overall efficiency
        if analyzed_metrics > 0:
            average_efficiency = total_efficiency_score / analyzed_metrics
            efficiency_analysis["overall_efficiency"] = (
                "excellent"
                if average_efficiency >= 80
                else (
                    "good"
                    if average_efficiency >= 60
                    else "moderate" if average_efficiency >= 40 else "poor"
                )
            )
            efficiency_analysis["average_efficiency_score"] = average_efficiency

        # Generate optimization opportunities
        efficiency_analysis["optimization_opportunities"] = []

        # Check for metrics with good potential
        for metric_name, score in efficiency_analysis["efficiency_scores"].items():
            if 40 <= score < 80:
                efficiency_analysis["optimization_opportunities"].append(
                    f"Improve {metric_name} scaling - current score {score:.1f}% has optimization potential"
                )

        # General recommendations based on overall efficiency
        if efficiency_analysis.get("average_efficiency_score", 0) < 60:
            efficiency_analysis["optimization_opportunities"].extend(
                [
                    "Consider parallel processing to improve scaling efficiency",
                    "Profile code to identify performance bottlenecks",
                    "Evaluate algorithmic complexity and optimization opportunities",
                ]
            )

        return efficiency_analysis

    def generate_test_matrix(
        self, optimize_execution_order: bool = False
    ) -> Dict[str, Any]:
        """
        Generates optimized test execution matrix with resource allocation and timing
        coordination for systematic scalability testing.

        Args:
            optimize_execution_order: Whether to optimize execution order for resource efficiency

        Returns:
            dict: Optimized test execution matrix with resource allocation and timing coordination
        """
        test_matrix = {
            "matrix_id": f"scalability_test_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "creation_timestamp": datetime.datetime.now().isoformat(),
            "grid_sizes": self.grid_sizes.copy(),
            "test_scenarios": [],
            "execution_plan": {},
            "resource_allocation": {},
            "estimated_total_time": 0.0,
        }

        # Generate test scenarios
        for i, scenario in enumerate(self.test_scenarios):
            test_scenario = {
                "test_id": f"scale_test_{i:02d}",
                "grid_size": scenario["grid_size"],
                "complexity": scenario["complexity"],
                "performance_category": scenario["performance_category"],
                "estimated_execution_time": scenario["estimated_execution_time_s"],
                "expected_memory_usage": scenario["expected_memory_mb"],
                "configuration": {
                    "grid_size": scenario["grid_size"],
                    "source_location": scenario["source_location"],
                    "iterations": DEFAULT_BENCHMARK_ITERATIONS,
                    "memory_profiling": True,
                    "statistical_analysis": True,
                },
                "success_criteria": {
                    "memory_limit_mb": MEMORY_LIMIT_TOTAL_MB,
                    "performance_targets": PERFORMANCE_TARGETS.copy(),
                },
            }
            test_matrix["test_scenarios"].append(test_scenario)

        # Optimize execution order if requested
        if optimize_execution_order:
            # Sort by complexity (memory usage) to optimize resource allocation
            test_matrix["test_scenarios"].sort(key=lambda x: x["complexity"])
            test_matrix["execution_plan"]["optimization"] = "ascending_complexity"
        else:
            test_matrix["execution_plan"]["optimization"] = "none"

        # Resource allocation planning
        max_memory_scenario = max(
            self.test_scenarios, key=lambda x: x["expected_memory_mb"]
        )
        total_execution_time = sum(
            s["estimated_execution_time"] for s in self.test_scenarios
        )

        test_matrix["resource_allocation"] = {
            "peak_memory_requirement_mb": max_memory_scenario["expected_memory_mb"],
            "memory_warning_threshold": MEMORY_LIMIT_TOTAL_MB * 0.8,
            "recommended_available_memory_mb": max_memory_scenario["expected_memory_mb"]
            * 1.2,
            "parallelization_possible": len(self.grid_sizes) > 1,
            "resource_monitoring_required": True,
        }

        # Execution timing and coordination
        test_matrix["execution_plan"].update(
            {
                "total_tests": len(test_matrix["test_scenarios"]),
                "estimated_serial_execution_time": total_execution_time,
                "recommended_execution_strategy": (
                    "parallel" if len(self.grid_sizes) <= 4 else "batched_serial"
                ),
                "cooldown_between_tests": 5.0,  # seconds
                "warmup_iterations": 10,
                "validation_after_each_test": True,
            }
        )

        test_matrix["estimated_total_time"] = total_execution_time + (
            len(test_matrix["test_scenarios"]) * 5.0
        )  # cooldown time

        return test_matrix


def get_performance_baseline(
    baseline_name: str,
    measurement_data: Dict[str, List[float]],
    system_info: Optional[Dict[str, Any]] = None,
    include_confidence_intervals: bool = True,
) -> PerformanceBaseline:
    """
    Creates comprehensive performance baseline data structure with statistical analysis, target validation,
    and measurement metadata for benchmark comparison and regression detection across different system configurations.

    Args:
        baseline_name: Descriptive name for the performance baseline
        measurement_data: Dictionary mapping metric names to lists of measurements
        system_info: Optional system information for platform-specific analysis
        include_confidence_intervals: Whether to calculate confidence intervals

    Returns:
        PerformanceBaseline: Complete performance baseline with statistical analysis and validation metadata
    """
    # Validate measurement data contains required metrics
    if not measurement_data:
        raise ValueError("Measurement data cannot be empty")

    # Calculate comprehensive statistical analysis
    statistics_data = {}
    for metric_name, values in measurement_data.items():
        if not values:
            continue

        metric_stats = {
            "count": len(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
            "range": max(values) - min(values),
        }

        # Add standard deviation and variance if multiple values
        if len(values) > 1:
            try:
                metric_stats["std_dev"] = statistics.stdev(values)
                metric_stats["variance"] = statistics.variance(values)
            except statistics.StatisticsError:
                metric_stats["std_dev"] = 0.0
                metric_stats["variance"] = 0.0
        else:
            metric_stats["std_dev"] = 0.0
            metric_stats["variance"] = 0.0

        # Calculate percentiles
        if len(values) >= 2:
            sorted_values = sorted(values)
            metric_stats["percentile_25"] = np.percentile(sorted_values, 25)
            metric_stats["percentile_75"] = np.percentile(sorted_values, 75)
            metric_stats["percentile_90"] = np.percentile(sorted_values, 90)
            metric_stats["percentile_95"] = np.percentile(sorted_values, 95)
            metric_stats["percentile_99"] = np.percentile(sorted_values, 99)

        statistics_data[metric_name] = metric_stats

    # Use appropriate performance targets
    targets = PERFORMANCE_TARGETS.copy()

    # Create performance baseline
    baseline = PerformanceBaseline(
        name=baseline_name,
        measurements=measurement_data.copy(),
        statistics=statistics_data,
        targets=targets,
    )

    # Include system information if provided
    if system_info:
        baseline.system_fingerprint = (
            f"{baseline.system_fingerprint}_{hash(str(system_info)) % 10000}"
        )

    return baseline


def create_benchmark_scenario(
    scenario_name: str,
    scenario_config: Dict[str, Any],
    iterations: int = DEFAULT_BENCHMARK_ITERATIONS,
    performance_targets: Optional[Dict[str, float]] = None,
    include_memory_profiling: bool = True,
) -> BenchmarkScenario:
    """
    Generates comprehensive benchmark test scenario with environment configuration, execution parameters,
    measurement criteria, and expected performance targets for automated benchmark execution and validation.

    Args:
        scenario_name: Descriptive name for benchmark scenario
        scenario_config: Environment configuration parameters
        iterations: Number of benchmark iterations to execute
        performance_targets: Optional custom performance targets
        include_memory_profiling: Whether to enable memory profiling

    Returns:
        BenchmarkScenario: Complete benchmark scenario with configuration, execution plan, and validation criteria
    """
    # Validate scenario configuration contains required parameters
    required_keys = ["grid_size", "source_location"]
    for key in required_keys:
        if key not in scenario_config:
            raise ValueError(f"Required scenario configuration key '{key}' missing")

    # Set default performance targets if not provided
    if performance_targets is None:
        performance_targets = PERFORMANCE_TARGETS.copy()

    # Generate execution plan with timing measurements and statistical analysis requirements
    execution_plan = {
        "measurement_protocol": {
            "warmup_iterations": max(10, iterations // 100),
            "measurement_iterations": iterations,
            "cooldown_iterations": 5,
            "statistical_analysis": True,
            "outlier_detection": True,
            "confidence_intervals": True,
        },
        "resource_monitoring": {
            "memory_profiling": include_memory_profiling,
            "timing_precision": "microsecond",
            "system_load_monitoring": True,
        },
        "data_collection": {
            "collect_raw_data": True,
            "calculate_statistics": True,
            "generate_reports": True,
        },
    }

    # Generate validation criteria based on performance targets and tolerance settings
    validation_criteria = {
        "performance_targets": performance_targets.copy(),
        "tolerance_percent": PERFORMANCE_TOLERANCE_PERCENT,
        "statistical_confidence": BASELINE_MEASUREMENT_CONFIDENCE,
        "outlier_threshold": STATISTICAL_OUTLIER_THRESHOLD,
        "minimum_success_rate": 0.95,
        "regression_detection": True,
    }

    # Create comprehensive benchmark scenario
    scenario = BenchmarkScenario(
        name=scenario_name,
        config=scenario_config.copy(),
        execution_plan=execution_plan,
        validation_criteria=validation_criteria,
        performance_targets=performance_targets,
        memory_profiling_enabled=include_memory_profiling,
        iterations=iterations,
    )

    return scenario


def get_scalability_test_data(
    grid_sizes: Optional[List[Tuple[int, int]]] = None,
    include_memory_estimates: bool = True,
    include_performance_projections: bool = True,
    scaling_analysis_type: str = "comprehensive",
) -> ScalabilityTestData:
    """
    Generates comprehensive scalability test dataset across multiple grid sizes with resource estimation,
    performance projection, and scaling analysis for system capacity planning and optimization validation.

    Args:
        grid_sizes: Optional list of grid sizes, defaults to SCALABILITY_TEST_GRID_SIZES
        include_memory_estimates: Whether to generate memory usage estimates
        include_performance_projections: Whether to generate performance projections
        scaling_analysis_type: Type of scaling analysis ('basic', 'comprehensive', 'detailed')

    Returns:
        ScalabilityTestData: Complete scalability test data with grid size analysis and performance projections
    """
    # Use default grid sizes if not provided
    if grid_sizes is None:
        grid_sizes = SCALABILITY_TEST_GRID_SIZES.copy()

    # Generate performance projections for different grid sizes
    performance_projections = {}
    if include_performance_projections:
        for metric_name, target_value in PERFORMANCE_TARGETS.items():
            metric_projections = {}

            for grid_size in grid_sizes:
                complexity = grid_size[0] * grid_size[1]
                base_complexity = 128 * 128  # Normalize to 128x128

                # Different scaling patterns for different metrics
                if "latency" in metric_name or "time" in metric_name:
                    # Time-based metrics typically scale with grid complexity
                    scaling_factor = complexity / base_complexity
                    if "step" in metric_name:
                        # Step latency scales approximately linearly
                        projected_value = target_value * scaling_factor
                    elif "render" in metric_name:
                        # Rendering scales more than linearly
                        projected_value = target_value * (scaling_factor**1.2)
                    elif "reset" in metric_name:
                        # Reset time scales with plume generation
                        projected_value = target_value * (scaling_factor**0.8)
                    else:
                        # Default linear scaling
                        projected_value = target_value * scaling_factor
                elif "memory" in metric_name:
                    # Memory usage scales approximately linearly with grid size
                    projected_value = (complexity * 4) / (
                        1024 * 1024
                    ) + 10  # Field + base
                else:
                    # Default scaling for other metrics
                    projected_value = target_value * (complexity / base_complexity)

                metric_projections[grid_size] = projected_value

            performance_projections[metric_name] = metric_projections

    # Generate resource estimates for each grid size
    resource_estimates = {}
    if include_memory_estimates:
        memory_estimates = {}
        cpu_estimates = {}

        for grid_size in grid_sizes:
            complexity = grid_size[0] * grid_size[1]

            # Memory estimate: field storage + base environment overhead
            field_memory_mb = (complexity * 4) / (1024 * 1024)  # float32 field
            base_memory_mb = 10.0
            total_memory_mb = field_memory_mb + base_memory_mb

            # CPU usage estimate (rough approximation)
            cpu_usage_percent = min(100.0, 20.0 + (complexity / 10000))

            memory_estimates[grid_size] = total_memory_mb
            cpu_estimates[grid_size] = cpu_usage_percent

        resource_estimates["memory_mb"] = memory_estimates
        resource_estimates["cpu_percent"] = cpu_estimates

    # Create scalability test data with comprehensive analysis
    scalability_data = ScalabilityTestData(
        grid_sizes=grid_sizes,
        performance_projections=performance_projections,
        resource_estimates=resource_estimates,
    )

    # Set scaling analysis type
    scalability_data.scaling_model_type = scaling_analysis_type

    return scalability_data


def get_benchmark_test_matrix(
    test_categories: List[str],
    optimize_execution_order: bool = True,
    matrix_constraints: Optional[Dict[str, Any]] = None,
    include_cross_validation: bool = False,
) -> Dict[str, Any]:
    """
    Creates comprehensive benchmark test matrix combining multiple scenarios, configurations, and measurement
    types for systematic performance validation with execution optimization and result correlation analysis.

    Args:
        test_categories: List of test categories to include in matrix
        optimize_execution_order: Whether to optimize execution order for minimal system interference
        matrix_constraints: Optional constraints to filter scenarios and optimize resource utilization
        include_cross_validation: Whether to include cross-validation scenarios

    Returns:
        dict: Complete benchmark test matrix with optimized execution plan and validation configuration
    """
    # Validate test categories
    available_categories = [
        "performance",
        "scalability",
        "memory",
        "accuracy",
        "stability",
    ]
    for category in test_categories:
        if category not in available_categories:
            raise ValueError(f"Unknown test category: {category}")

    test_matrix = {
        "matrix_id": f"benchmark_matrix_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "creation_timestamp": datetime.datetime.now().isoformat(),
        "test_categories": test_categories.copy(),
        "test_scenarios": [],
        "execution_plan": {},
        "validation_configuration": {},
        "resource_requirements": {},
        "estimated_execution_time": 0.0,
    }

    # Generate benchmark scenarios for each category
    scenario_id = 0

    if "performance" in test_categories:
        # Performance baseline scenarios
        for grid_size in [(64, 64), (128, 128), (256, 256)]:
            scenario = create_benchmark_scenario(
                scenario_name=f"performance_baseline_{grid_size[0]}x{grid_size[1]}",
                scenario_config={
                    "grid_size": grid_size,
                    "source_location": (grid_size[0] // 2, grid_size[1] // 2),
                    "plume_sigma": 12.0,
                },
                iterations=1000,
                include_memory_profiling=True,
            )

            test_matrix["test_scenarios"].append(
                {
                    "scenario_id": f"scenario_{scenario_id:03d}",
                    "category": "performance",
                    "scenario": scenario,
                    "priority": 1,
                    "estimated_duration": scenario.estimated_execution_time,
                }
            )
            scenario_id += 1

    if "scalability" in test_categories:
        # Scalability test scenarios
        scalability_data = get_scalability_test_data()
        test_matrix_scalability = scalability_data.generate_test_matrix(
            optimize_execution_order=optimize_execution_order
        )

        for test_scenario in test_matrix_scalability["test_scenarios"]:
            scenario_config = test_scenario["configuration"].copy()

            scenario = create_benchmark_scenario(
                scenario_name=f"scalability_{test_scenario['test_id']}",
                scenario_config=scenario_config,
                iterations=500,  # Fewer iterations for scalability tests
                include_memory_profiling=True,
            )

            test_matrix["test_scenarios"].append(
                {
                    "scenario_id": f"scenario_{scenario_id:03d}",
                    "category": "scalability",
                    "scenario": scenario,
                    "priority": 2,
                    "estimated_duration": test_scenario["estimated_execution_time"],
                }
            )
            scenario_id += 1

    if "memory" in test_categories:
        # Memory-focused test scenarios
        memory_test_configs = [
            {"grid_size": (256, 256), "focus": "large_grid"},
            {"grid_size": (512, 512), "focus": "memory_limit"},
            {"grid_size": (128, 128), "focus": "baseline_memory"},
        ]

        for config in memory_test_configs:
            grid_size = config["grid_size"]
            scenario = create_benchmark_scenario(
                scenario_name=f"memory_test_{config['focus']}",
                scenario_config={
                    "grid_size": grid_size,
                    "source_location": (grid_size[0] // 2, grid_size[1] // 2),
                    "plume_sigma": 15.0,
                },
                iterations=100,  # Fewer iterations for memory tests
                include_memory_profiling=True,
            )

            test_matrix["test_scenarios"].append(
                {
                    "scenario_id": f"scenario_{scenario_id:03d}",
                    "category": "memory",
                    "scenario": scenario,
                    "priority": 3,
                    "estimated_duration": scenario.estimated_execution_time,
                }
            )
            scenario_id += 1

    if "accuracy" in test_categories:
        # Accuracy validation scenarios
        quick_start_config = get_quick_start_config()

        scenario = create_benchmark_scenario(
            scenario_name="accuracy_validation_baseline",
            scenario_config=quick_start_config.to_dict(),
            iterations=2000,  # More iterations for accuracy testing
            include_memory_profiling=False,  # Focus on accuracy, not memory
        )

        test_matrix["test_scenarios"].append(
            {
                "scenario_id": f"scenario_{scenario_id:03d}",
                "category": "accuracy",
                "scenario": scenario,
                "priority": 1,
                "estimated_duration": scenario.estimated_execution_time,
            }
        )
        scenario_id += 1

    if "stability" in test_categories:
        # Stability and regression test scenarios
        stability_scenario = create_benchmark_scenario(
            scenario_name="stability_regression_test",
            scenario_config={
                "grid_size": DEFAULT_GRID_SIZE,
                "source_location": (
                    DEFAULT_GRID_SIZE[0] // 2,
                    DEFAULT_GRID_SIZE[1] // 2,
                ),
                "plume_sigma": 12.0,
            },
            iterations=5000,  # Long run for stability testing
            include_memory_profiling=True,
        )

        test_matrix["test_scenarios"].append(
            {
                "scenario_id": f"scenario_{scenario_id:03d}",
                "category": "stability",
                "scenario": stability_scenario,
                "priority": 2,
                "estimated_duration": stability_scenario.estimated_execution_time,
            }
        )
        scenario_id += 1

    # Apply matrix constraints if provided
    if matrix_constraints:
        original_count = len(test_matrix["test_scenarios"])

        # Filter by maximum execution time
        if "max_execution_time_seconds" in matrix_constraints:
            max_time = matrix_constraints["max_execution_time_seconds"]
            test_matrix["test_scenarios"] = [
                s
                for s in test_matrix["test_scenarios"]
                if s["estimated_duration"] <= max_time
            ]

        # Filter by maximum memory usage
        if "max_memory_mb" in matrix_constraints:
            max_memory = matrix_constraints["max_memory_mb"]
            # This would require estimating memory usage for each scenario
            # Implementation depends on detailed memory estimation

        # Limit total number of scenarios
        if "max_scenarios" in matrix_constraints:
            max_scenarios = matrix_constraints["max_scenarios"]
            # Sort by priority and take top scenarios
            test_matrix["test_scenarios"].sort(key=lambda x: x["priority"])
            test_matrix["test_scenarios"] = test_matrix["test_scenarios"][
                :max_scenarios
            ]

        filtered_count = len(test_matrix["test_scenarios"])
        test_matrix["matrix_constraints_applied"] = {
            "original_scenarios": original_count,
            "filtered_scenarios": filtered_count,
            "constraints": matrix_constraints.copy(),
        }

    # Optimize execution order if requested
    if optimize_execution_order:
        # Sort by priority first, then by estimated duration
        test_matrix["test_scenarios"].sort(
            key=lambda x: (x["priority"], x["estimated_duration"])
        )
        test_matrix["execution_plan"]["order_optimization"] = "priority_then_duration"
    else:
        test_matrix["execution_plan"]["order_optimization"] = "none"

    # Calculate resource requirements and execution planning
    total_duration = sum(s["estimated_duration"] for s in test_matrix["test_scenarios"])
    max_memory_scenario = max(
        test_matrix["test_scenarios"],
        key=lambda s: s["scenario"].estimate_resources().get("estimated_memory_mb", 0),
        default={"scenario": None},
    )

    if max_memory_scenario["scenario"]:
        peak_memory = max_memory_scenario["scenario"].estimate_resources()[
            "estimated_memory_mb"
        ]
    else:
        peak_memory = 0.0

    test_matrix["resource_requirements"] = {
        "total_scenarios": len(test_matrix["test_scenarios"]),
        "estimated_total_duration_seconds": total_duration,
        "estimated_peak_memory_mb": peak_memory,
        "parallel_execution_possible": len(test_matrix["test_scenarios"]) > 1,
        "resource_monitoring_required": True,
    }

    # Execution plan configuration
    test_matrix["execution_plan"].update(
        {
            "total_scenarios": len(test_matrix["test_scenarios"]),
            "sequential_execution": True,
            "cooldown_between_scenarios": 10.0,  # seconds
            "resource_monitoring": True,
            "failure_handling": "continue_with_warning",
            "result_aggregation": True,
        }
    )

    # Validation configuration
    test_matrix["validation_configuration"] = {
        "cross_validation_enabled": include_cross_validation,
        "statistical_analysis": True,
        "regression_detection": True,
        "performance_comparison": True,
        "result_correlation": include_cross_validation,
        "confidence_level": BASELINE_MEASUREMENT_CONFIDENCE,
        "outlier_detection": True,
        "outlier_threshold": STATISTICAL_OUTLIER_THRESHOLD,
    }

    test_matrix["estimated_execution_time"] = total_duration + (
        len(test_matrix["test_scenarios"]) * 10.0
    )

    return test_matrix


def generate_baseline_data(
    historical_measurements: List[Dict[str, Any]],
    confidence_level: float = BASELINE_MEASUREMENT_CONFIDENCE,
    detect_trends: bool = True,
    baseline_version: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generates comprehensive baseline performance data from historical measurements with statistical analysis,
    trend detection, and confidence interval calculation for benchmark validation and regression detection.

    Args:
        historical_measurements: List of historical measurement dictionaries
        confidence_level: Confidence level for statistical analysis
        detect_trends: Whether to perform trend analysis
        baseline_version: Optional version identifier for baseline

    Returns:
        dict: Generated baseline data with statistical analysis, trends, and validation thresholds
    """
    if not historical_measurements:
        raise ValueError("Historical measurements cannot be empty")

    # Aggregate measurements by metric name
    aggregated_measurements = {}
    timestamps = []

    for measurement in historical_measurements:
        timestamp = measurement.get("timestamp", datetime.datetime.now().isoformat())
        timestamps.append(timestamp)

        for metric_name, value in measurement.items():
            if metric_name == "timestamp":
                continue

            if isinstance(value, (int, float)):
                if metric_name not in aggregated_measurements:
                    aggregated_measurements[metric_name] = []
                aggregated_measurements[metric_name].append(value)

    # Calculate statistical baselines
    baseline_statistics = {}
    for metric_name, values in aggregated_measurements.items():
        if len(values) < 2:
            continue

        # Calculate comprehensive statistics
        metric_stats = {
            "count": len(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "mode": (
                statistics.mode(values) if len(set(values)) < len(values) else values[0]
            ),
            "min": min(values),
            "max": max(values),
            "range": max(values) - min(values),
            "std_dev": statistics.stdev(values) if len(values) > 1 else 0.0,
            "variance": statistics.variance(values) if len(values) > 1 else 0.0,
        }

        # Calculate confidence intervals
        mean_val = metric_stats["mean"]
        std_dev = metric_stats["std_dev"]
        n = len(values)

        if n >= 3 and std_dev > 0:
            # Use t-distribution approximation for confidence intervals
            t_value = 1.96  # Approximate for 95% confidence
            margin_error = t_value * (std_dev / np.sqrt(n))

            metric_stats["confidence_interval"] = {
                "lower": max(0.0, mean_val - margin_error),
                "upper": mean_val + margin_error,
                "confidence_level": confidence_level,
            }
        else:
            metric_stats["confidence_interval"] = {
                "lower": min(values),
                "upper": max(values),
                "confidence_level": confidence_level,
            }

        # Detect outliers
        if len(values) >= 10:
            q1 = np.percentile(values, 25)
            q3 = np.percentile(values, 75)
            iqr = q3 - q1

            lower_bound = q1 - (STATISTICAL_OUTLIER_THRESHOLD * iqr)
            upper_bound = q3 + (STATISTICAL_OUTLIER_THRESHOLD * iqr)

            outliers = [v for v in values if v < lower_bound or v > upper_bound]
            metric_stats["outliers"] = {
                "count": len(outliers),
                "percentage": (len(outliers) / len(values)) * 100,
                "values": outliers,
            }

        baseline_statistics[metric_name] = metric_stats

    # Trend detection
    trend_analysis = {}
    if detect_trends and len(historical_measurements) >= 5:
        # Perform simple trend analysis on each metric
        for metric_name, values in aggregated_measurements.items():
            if len(values) < 5:
                continue

            # Calculate moving averages and trend direction
            window_size = min(5, len(values) // 2)
            if len(values) >= window_size * 2:
                early_avg = statistics.mean(values[:window_size])
                late_avg = statistics.mean(values[-window_size:])

                trend_direction = (
                    "improving"
                    if late_avg < early_avg
                    else ("degrading" if late_avg > early_avg else "stable")
                )

                trend_magnitude = (
                    abs(late_avg - early_avg) / early_avg if early_avg > 0 else 0
                )

                trend_analysis[metric_name] = {
                    "direction": trend_direction,
                    "magnitude_percent": trend_magnitude * 100,
                    "early_average": early_avg,
                    "recent_average": late_avg,
                    "trend_significant": trend_magnitude > 0.1,  # 10% threshold
                }

    # Generate performance validation thresholds
    validation_thresholds = {}
    for metric_name, stats in baseline_statistics.items():
        if metric_name in PERFORMANCE_TARGETS:
            target = PERFORMANCE_TARGETS[metric_name]
            mean_val = stats["mean"]
            std_dev = stats["std_dev"]

            # Set thresholds based on historical performance and targets
            validation_thresholds[metric_name] = {
                "target": target,
                "warning_threshold": mean_val + (std_dev * 1.5),
                "error_threshold": mean_val + (std_dev * 3.0),
                "regression_threshold": target
                * (1 + PERFORMANCE_TOLERANCE_PERCENT / 100),
                "baseline_mean": mean_val,
                "baseline_std": std_dev,
            }

    # Compile comprehensive baseline data
    baseline_data = {
        "baseline_id": f"baseline_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "version": baseline_version or BENCHMARK_DATA_VERSION,
        "creation_timestamp": datetime.datetime.now().isoformat(),
        "confidence_level": confidence_level,
        "measurement_count": len(historical_measurements),
        "measurement_timespan": {
            "earliest": min(timestamps) if timestamps else None,
            "latest": max(timestamps) if timestamps else None,
        },
        "statistical_baselines": baseline_statistics,
        "trend_analysis": trend_analysis,
        "validation_thresholds": validation_thresholds,
        "data_quality": {
            "metrics_analyzed": len(baseline_statistics),
            "total_measurements": sum(
                len(values) for values in aggregated_measurements.values()
            ),
            "outliers_detected": sum(
                stats.get("outliers", {}).get("count", 0)
                for stats in baseline_statistics.values()
            ),
            "trends_detected": len(
                [
                    t
                    for t in trend_analysis.values()
                    if t.get("trend_significant", False)
                ]
            ),
        },
    }

    return baseline_data


def validate_benchmark_data(
    benchmark_data: Dict[str, Any],
    strict_validation: bool = False,
    validation_criteria: Optional[Dict[str, Any]] = None,
    include_recommendations: bool = True,
) -> Dict[str, Any]:
    """
    Validates benchmark data integrity, statistical consistency, and performance target compliance
    with comprehensive analysis reporting for quality assurance and data reliability verification.

    Args:
        benchmark_data: Benchmark data dictionary to validate
        strict_validation: Whether to apply strict validation rules
        validation_criteria: Optional custom validation criteria
        include_recommendations: Whether to generate quality recommendations

    Returns:
        dict: Comprehensive validation report with data integrity analysis and quality recommendations
    """
    validation_report = {
        "validation_timestamp": datetime.datetime.now().isoformat(),
        "data_id": benchmark_data.get("baseline_id", "unknown"),
        "validation_passed": True,
        "validation_errors": [],
        "validation_warnings": [],
        "data_quality_metrics": {},
        "statistical_consistency": {},
        "target_compliance": {},
        "recommendations": [] if include_recommendations else None,
    }

    # Validate benchmark data structure and completeness
    required_fields = [
        "statistical_baselines",
        "validation_thresholds",
        "creation_timestamp",
    ]
    for field in required_fields:
        if field not in benchmark_data:
            validation_report["validation_errors"].append(
                f"Missing required field: {field}"
            )
            validation_report["validation_passed"] = False

    # Validate statistical consistency
    if "statistical_baselines" in benchmark_data:
        baselines = benchmark_data["statistical_baselines"]

        for metric_name, stats in baselines.items():
            consistency_check = {
                "metric": metric_name,
                "consistent": True,
                "issues": [],
            }

            # Check for reasonable statistics
            if stats.get("count", 0) < 2:
                consistency_check["issues"].append(
                    "Insufficient measurements for statistics"
                )
                consistency_check["consistent"] = False

                if strict_validation:
                    validation_report["validation_errors"].append(
                        f"Metric {metric_name} has insufficient measurements"
                    )
                    validation_report["validation_passed"] = False
                else:
                    validation_report["validation_warnings"].append(
                        f"Metric {metric_name} has low measurement count"
                    )

            # Check for statistical anomalies
            mean_val = stats.get("mean", 0)
            std_dev = stats.get("std_dev", 0)
            min_val = stats.get("min", 0)
            max_val = stats.get("max", 0)

            if std_dev > mean_val * 2 and mean_val > 0:  # High variability
                consistency_check["issues"].append(
                    "High measurement variability detected"
                )
                validation_report["validation_warnings"].append(
                    f"High variability in {metric_name}: std_dev={std_dev:.3f}, mean={mean_val:.3f}"
                )

            if min_val < 0 and "latency" in metric_name.lower():  # Negative latency
                consistency_check["issues"].append("Negative latency values detected")
                consistency_check["consistent"] = False
                validation_report["validation_errors"].append(
                    f"Negative values in {metric_name} (latency metric)"
                )
                validation_report["validation_passed"] = False

            # Check confidence intervals
            ci = stats.get("confidence_interval", {})
            if ci:
                ci_lower = ci.get("lower", 0)
                ci_upper = ci.get("upper", 0)

                if ci_lower > ci_upper:
                    consistency_check["issues"].append(
                        "Invalid confidence interval bounds"
                    )
                    consistency_check["consistent"] = False
                    validation_report["validation_errors"].append(
                        f"Invalid confidence interval for {metric_name}"
                    )
                    validation_report["validation_passed"] = False

            validation_report["statistical_consistency"][
                metric_name
            ] = consistency_check

    # Validate performance target compliance
    if "validation_thresholds" in benchmark_data:
        thresholds = benchmark_data["validation_thresholds"]

        for metric_name, threshold_data in thresholds.items():
            target = threshold_data.get("target")
            baseline_mean = threshold_data.get("baseline_mean")

            if target is not None and baseline_mean is not None:
                # Check if baseline meets performance target
                target_met = baseline_mean <= target * (
                    1 + PERFORMANCE_TOLERANCE_PERCENT / 100
                )

                compliance_status = {
                    "metric": metric_name,
                    "target": target,
                    "baseline_mean": baseline_mean,
                    "target_met": target_met,
                    "performance_margin": target - baseline_mean,
                    "margin_percentage": (
                        ((target - baseline_mean) / target * 100) if target > 0 else 0
                    ),
                }

                if not target_met:
                    if strict_validation:
                        validation_report["validation_errors"].append(
                            f"Performance target not met for {metric_name}: {baseline_mean:.3f} > {target:.3f}"
                        )
                        validation_report["validation_passed"] = False
                    else:
                        validation_report["validation_warnings"].append(
                            f"Performance target exceeded for {metric_name}"
                        )

                validation_report["target_compliance"][metric_name] = compliance_status

    # Calculate data quality metrics
    if "statistical_baselines" in benchmark_data:
        baselines = benchmark_data["statistical_baselines"]

        total_measurements = sum(stats.get("count", 0) for stats in baselines.values())
        total_outliers = sum(
            stats.get("outliers", {}).get("count", 0) for stats in baselines.values()
        )

        validation_report["data_quality_metrics"] = {
            "metrics_count": len(baselines),
            "total_measurements": total_measurements,
            "total_outliers": total_outliers,
            "outlier_percentage": (
                (total_outliers / total_measurements * 100)
                if total_measurements > 0
                else 0
            ),
            "average_measurements_per_metric": (
                total_measurements / len(baselines) if baselines else 0
            ),
        }

        # Quality assessment
        quality_score = 100.0

        if validation_report["data_quality_metrics"]["outlier_percentage"] > 10:
            quality_score -= 20
            validation_report["validation_warnings"].append(
                "High outlier percentage detected - data quality may be compromised"
            )

        if (
            validation_report["data_quality_metrics"]["average_measurements_per_metric"]
            < 10
        ):
            quality_score -= 15
            validation_report["validation_warnings"].append(
                "Low measurement count per metric - statistical significance may be limited"
            )

        validation_report["data_quality_metrics"]["quality_score"] = max(
            0.0, quality_score
        )

    # Generate quality recommendations
    if include_recommendations:
        recommendations = validation_report["recommendations"]

        # Data collection recommendations
        if (
            validation_report["data_quality_metrics"].get(
                "average_measurements_per_metric", 0
            )
            < 30
        ):
            recommendations.append(
                "Increase measurement sample size to improve statistical confidence"
            )

        if validation_report["data_quality_metrics"].get("outlier_percentage", 0) > 5:
            recommendations.append(
                "Investigate and address outlier measurements to improve data quality"
            )

        # Performance recommendations
        non_compliant_metrics = [
            metric
            for metric, compliance in validation_report["target_compliance"].items()
            if not compliance.get("target_met", True)
        ]

        if non_compliant_metrics:
            recommendations.append(
                f"Optimize performance for metrics: {', '.join(non_compliant_metrics)}"
            )

        # Statistical consistency recommendations
        inconsistent_metrics = [
            metric
            for metric, consistency in validation_report[
                "statistical_consistency"
            ].items()
            if not consistency.get("consistent", True)
        ]

        if inconsistent_metrics:
            recommendations.append(
                f"Review measurement procedures for metrics: {', '.join(inconsistent_metrics)}"
            )

        # Validation mode recommendations
        if not strict_validation and validation_report["validation_warnings"]:
            recommendations.append(
                "Consider running strict validation mode for production deployment"
            )

    return validation_report


def calculate_performance_statistics(
    measurements: List[float],
    confidence_level: float = BASELINE_MEASUREMENT_CONFIDENCE,
    include_distribution_analysis: bool = True,
    baseline_comparison: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """
    Calculates comprehensive performance statistics from benchmark measurements including descriptive statistics,
    confidence intervals, performance distribution analysis, and trend detection for detailed performance analysis.

    Args:
        measurements: List of performance measurements
        confidence_level: Confidence level for interval calculations
        include_distribution_analysis: Whether to perform distribution analysis
        baseline_comparison: Optional baseline measurements for comparison

    Returns:
        dict: Comprehensive performance statistics with confidence intervals and distribution analysis
    """
    if not measurements:
        raise ValueError("Measurements list cannot be empty")

    if len(measurements) < 2:
        raise ValueError("Need at least 2 measurements for statistical analysis")

    # Calculate descriptive statistics
    stats = {
        "measurement_count": len(measurements),
        "descriptive_statistics": {
            "mean": statistics.mean(measurements),
            "median": statistics.median(measurements),
            "min": min(measurements),
            "max": max(measurements),
            "range": max(measurements) - min(measurements),
            "sum": sum(measurements),
        },
        "variability_statistics": {
            "std_dev": statistics.stdev(measurements) if len(measurements) > 1 else 0.0,
            "variance": (
                statistics.variance(measurements) if len(measurements) > 1 else 0.0
            ),
            "coefficient_of_variation": 0.0,
        },
        "percentiles": {},
        "confidence_intervals": {},
        "data_quality": {},
    }

    # Calculate coefficient of variation
    mean_val = stats["descriptive_statistics"]["mean"]
    std_dev = stats["variability_statistics"]["std_dev"]

    if mean_val > 0:
        stats["variability_statistics"]["coefficient_of_variation"] = (
            std_dev / mean_val
        ) * 100

    # Calculate percentiles
    sorted_measurements = sorted(measurements)
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]

    for p in percentiles:
        stats["percentiles"][f"p{p:02d}"] = np.percentile(sorted_measurements, p)

    # Calculate confidence intervals
    n = len(measurements)
    if n >= 3:
        # Use t-distribution for confidence intervals
        t_value = 1.96  # Approximate for 95% confidence
        if confidence_level == 0.99:
            t_value = 2.576
        elif confidence_level == 0.90:
            t_value = 1.645

        margin_error = t_value * (std_dev / np.sqrt(n))

        stats["confidence_intervals"] = {
            "confidence_level": confidence_level,
            "mean_ci_lower": max(0.0, mean_val - margin_error),
            "mean_ci_upper": mean_val + margin_error,
            "margin_of_error": margin_error,
        }
    else:
        stats["confidence_intervals"] = {
            "confidence_level": confidence_level,
            "mean_ci_lower": min(measurements),
            "mean_ci_upper": max(measurements),
            "margin_of_error": (max(measurements) - min(measurements)) / 2,
        }

    # Outlier detection
    if len(measurements) >= 10:
        q1 = stats["percentiles"]["p25"]
        q3 = stats["percentiles"]["p75"]
        iqr = q3 - q1

        lower_bound = q1 - (STATISTICAL_OUTLIER_THRESHOLD * iqr)
        upper_bound = q3 + (STATISTICAL_OUTLIER_THRESHOLD * iqr)

        outliers = [m for m in measurements if m < lower_bound or m > upper_bound]
        outlier_indices = [
            i for i, m in enumerate(measurements) if m < lower_bound or m > upper_bound
        ]

        stats["data_quality"]["outliers"] = {
            "count": len(outliers),
            "percentage": (len(outliers) / len(measurements)) * 100,
            "values": outliers,
            "indices": outlier_indices,
            "detection_method": "IQR",
            "threshold_multiplier": STATISTICAL_OUTLIER_THRESHOLD,
        }

    # Distribution analysis
    if include_distribution_analysis and len(measurements) >= 10:
        distribution_stats = {
            "skewness": None,
            "kurtosis": None,
            "normality_assessment": "unknown",
            "distribution_shape": "unknown",
        }

        try:
            # Basic distribution shape analysis
            mean_val = stats["descriptive_statistics"]["mean"]
            median_val = stats["descriptive_statistics"]["median"]

            if abs(mean_val - median_val) / mean_val < 0.1:  # Within 10%
                distribution_stats["distribution_shape"] = "approximately_symmetric"
                distribution_stats["normality_assessment"] = "likely_normal"
            elif mean_val > median_val:
                distribution_stats["distribution_shape"] = "right_skewed"
                distribution_stats["normality_assessment"] = "non_normal"
            else:
                distribution_stats["distribution_shape"] = "left_skewed"
                distribution_stats["normality_assessment"] = "non_normal"

            # Simple skewness approximation
            if len(measurements) >= 3:
                skewness_approx = (
                    (mean_val - median_val) / std_dev if std_dev > 0 else 0
                )
                distribution_stats["skewness"] = skewness_approx

        except Exception:
            distribution_stats["normality_assessment"] = "analysis_failed"

        stats["distribution_analysis"] = distribution_stats

    # Baseline comparison
    if baseline_comparison is not None and len(baseline_comparison) >= 2:
        try:
            baseline_mean = statistics.mean(baseline_comparison)
            baseline_std = (
                statistics.stdev(baseline_comparison)
                if len(baseline_comparison) > 1
                else 0.0
            )

            current_mean = stats["descriptive_statistics"]["mean"]

            # Calculate comparison metrics
            absolute_change = current_mean - baseline_mean
            percentage_change = (
                (absolute_change / baseline_mean * 100) if baseline_mean != 0 else 0
            )

            # Simple significance test approximation
            pooled_std = np.sqrt((std_dev**2 + baseline_std**2) / 2)
            effect_size = abs(absolute_change) / pooled_std if pooled_std > 0 else 0

            stats["baseline_comparison"] = {
                "baseline_mean": baseline_mean,
                "baseline_std": baseline_std,
                "current_mean": current_mean,
                "current_std": std_dev,
                "absolute_change": absolute_change,
                "percentage_change": percentage_change,
                "effect_size": effect_size,
                "change_direction": (
                    "improvement"
                    if absolute_change < 0
                    else "regression" if absolute_change > 0 else "unchanged"
                ),
                "statistical_significance": (
                    "likely"
                    if effect_size > 0.8
                    else "possible" if effect_size > 0.3 else "unlikely"
                ),
                "practical_significance": (
                    "yes" if abs(percentage_change) > 10 else "minimal"
                ),
            }

        except Exception as e:
            stats["baseline_comparison"] = {
                "error": f"Comparison calculation failed: {e}"
            }

    # Performance assessment
    stats["performance_assessment"] = {
        "consistency": (
            "excellent"
            if stats["variability_statistics"]["coefficient_of_variation"] < 5
            else (
                "good"
                if stats["variability_statistics"]["coefficient_of_variation"] < 15
                else (
                    "moderate"
                    if stats["variability_statistics"]["coefficient_of_variation"] < 30
                    else "poor"
                )
            )
        ),
        "data_quality": (
            "excellent"
            if stats["data_quality"].get("outliers", {}).get("percentage", 0) < 2
            else (
                "good"
                if stats["data_quality"].get("outliers", {}).get("percentage", 0) < 5
                else (
                    "moderate"
                    if stats["data_quality"].get("outliers", {}).get("percentage", 0)
                    < 10
                    else "poor"
                )
            )
        ),
        "sample_size": (
            "excellent"
            if n >= 100
            else (
                "good"
                if n >= 50
                else "adequate" if n >= 30 else "minimal" if n >= 10 else "insufficient"
            )
        ),
    }

    return stats


def get_regression_detection_data(
    metric_name: str,
    historical_data: Dict[str, Any],
    regression_threshold: float = 0.15,
    include_trend_analysis: bool = True,
) -> Dict[str, Any]:
    """
    Generates regression detection data structures with historical baselines, statistical thresholds,
    and change detection criteria for automated performance regression monitoring and alerting systems.

    Args:
        metric_name: Name of performance metric to monitor
        historical_data: Historical performance data for baseline establishment
        regression_threshold: Threshold for regression detection (15% = 0.15)
        include_trend_analysis: Whether to include trend analysis

    Returns:
        dict: Regression detection configuration with baselines, thresholds, and monitoring criteria
    """
    if not historical_data or not isinstance(historical_data, dict):
        raise ValueError("Historical data must be a non-empty dictionary")

    # Extract measurement data for the specific metric
    measurements = historical_data.get("measurements", {}).get(metric_name, [])
    if not measurements or len(measurements) < 5:
        raise ValueError(
            f"Insufficient historical data for metric '{metric_name}' - need at least 5 measurements"
        )

    # Calculate baseline performance metrics
    baseline_stats = calculate_performance_statistics(
        measurements=measurements, include_distribution_analysis=True
    )

    baseline_mean = baseline_stats["descriptive_statistics"]["mean"]
    baseline_std = baseline_stats["variability_statistics"]["std_dev"]
    baseline_median = baseline_stats["descriptive_statistics"]["median"]

    # Generate regression detection configuration
    regression_config = {
        "metric_name": metric_name,
        "configuration_timestamp": datetime.datetime.now().isoformat(),
        "baseline_statistics": {
            "mean": baseline_mean,
            "median": baseline_median,
            "std_dev": baseline_std,
            "measurement_count": len(measurements),
            "confidence_interval": baseline_stats["confidence_intervals"],
        },
        "detection_thresholds": {},
        "monitoring_criteria": {},
        "alerting_configuration": {},
        "statistical_control_limits": {},
    }

    # Calculate detection thresholds
    absolute_threshold = baseline_mean * regression_threshold
    statistical_threshold = baseline_std * 2  # 2 standard deviations

    # Use the more conservative threshold
    detection_threshold = max(absolute_threshold, statistical_threshold)

    regression_config["detection_thresholds"] = {
        "regression_threshold_percent": regression_threshold * 100,
        "absolute_threshold": detection_threshold,
        "statistical_threshold": statistical_threshold,
        "baseline_mean": baseline_mean,
        "upper_warning_limit": baseline_mean + detection_threshold,
        "upper_error_limit": baseline_mean + (detection_threshold * 2),
        "change_detection_sensitivity": (
            "high"
            if regression_threshold < 0.1
            else "medium" if regression_threshold < 0.2 else "low"
        ),
    }

    # Configure monitoring criteria
    regression_config["monitoring_criteria"] = {
        "single_measurement_threshold": detection_threshold,
        "consecutive_measurements_threshold": detection_threshold
        * 0.7,  # Lower threshold for multiple measurements
        "consecutive_count_trigger": 3,  # Trigger after 3 consecutive measurements above threshold
        "moving_average_window": 5,  # Calculate moving average over 5 measurements
        "moving_average_threshold": detection_threshold * 0.8,
        "trend_detection_window": 10,  # Analyze trend over 10 measurements
        "minimum_measurements_for_analysis": 3,
    }

    # Configure statistical control limits (SPC-style monitoring)
    control_limits = {
        "center_line": baseline_mean,
        "upper_control_limit": baseline_mean + (3 * baseline_std),  # UCL = mean + 3σ
        "lower_control_limit": max(
            0.0, baseline_mean - (3 * baseline_std)
        ),  # LCL = mean - 3σ
        "upper_warning_limit": baseline_mean + (2 * baseline_std),  # UWL = mean + 2σ
        "lower_warning_limit": max(
            0.0, baseline_mean - (2 * baseline_std)
        ),  # LWL = mean - 2σ
        "control_chart_type": "individuals",
        "control_rules": [
            "single_point_beyond_control_limits",
            "nine_consecutive_points_same_side",
            "six_consecutive_increasing_or_decreasing",
            "fourteen_consecutive_alternating",
            "two_of_three_beyond_2_sigma",
            "four_of_five_beyond_1_sigma",
        ],
    }

    regression_config["statistical_control_limits"] = control_limits

    # Configure alerting system
    regression_config["alerting_configuration"] = {
        "severity_levels": {
            "info": {
                "condition": "measurement_within_warning_limits",
                "threshold": baseline_mean + (baseline_std * 1),
                "action": "log_only",
            },
            "warning": {
                "condition": "measurement_beyond_warning_limits",
                "threshold": regression_config["detection_thresholds"][
                    "upper_warning_limit"
                ],
                "action": "notify_team",
            },
            "error": {
                "condition": "measurement_beyond_control_limits",
                "threshold": regression_config["detection_thresholds"][
                    "upper_error_limit"
                ],
                "action": "immediate_investigation",
            },
            "critical": {
                "condition": "consecutive_measurements_beyond_limits",
                "threshold": "multiple_threshold_violations",
                "action": "escalate_to_management",
            },
        },
        "notification_settings": {
            "enable_notifications": True,
            "notification_cooldown_minutes": 30,  # Prevent notification spam
            "escalation_delay_minutes": 60,
            "auto_recovery_detection": True,
        },
        "reporting_settings": {
            "generate_daily_reports": True,
            "generate_trend_reports": True,
            "include_statistical_analysis": True,
            "report_retention_days": 90,
        },
    }

    # Trend analysis
    if include_trend_analysis and len(measurements) >= 10:
        # Calculate trend over recent measurements
        recent_window = min(10, len(measurements))
        recent_measurements = measurements[-recent_window:]

        # Simple linear trend calculation
        x = list(range(len(recent_measurements)))
        y = recent_measurements

        # Calculate slope (trend direction and magnitude)
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_x2 = sum(xi * xi for xi in x)

        if n * sum_x2 - sum_x * sum_x != 0:
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            intercept = (sum_y - slope * sum_x) / n

            # Determine trend significance
            trend_magnitude = abs(
                slope * (recent_window - 1)
            )  # Projected change over window
            trend_significant = trend_magnitude > (baseline_std * 0.5)

            trend_analysis = {
                "trend_slope": slope,
                "trend_intercept": intercept,
                "trend_direction": (
                    "increasing"
                    if slope > 0
                    else "decreasing" if slope < 0 else "stable"
                ),
                "trend_magnitude": trend_magnitude,
                "trend_significant": trend_significant,
                "projected_change": slope * recent_window,
                "trend_window_size": recent_window,
                "trend_assessment": (
                    "concerning_upward"
                    if slope > 0 and trend_significant
                    else (
                        "improving_downward"
                        if slope < 0 and trend_significant
                        else "stable"
                    )
                ),
            }

            regression_config["trend_analysis"] = trend_analysis
        else:
            regression_config["trend_analysis"] = {
                "error": "Insufficient variation for trend analysis"
            }

    # Generate monitoring recommendations
    recommendations = []

    if baseline_stats["variability_statistics"]["coefficient_of_variation"] > 30:
        recommendations.append(
            "High measurement variability detected - consider additional data collection for stable baselines"
        )

    if len(measurements) < 30:
        recommendations.append(
            "Consider collecting more historical data for more robust regression detection"
        )

    if regression_threshold > 0.25:
        recommendations.append(
            "Large regression threshold may miss subtle performance degradations"
        )

    if baseline_std / baseline_mean > 0.2:  # CV > 20%
        recommendations.append(
            "Consider using median-based thresholds due to high measurement variability"
        )

    regression_config["monitoring_recommendations"] = recommendations

    return regression_config


def create_memory_usage_baseline(
    memory_measurements: Dict[str, List[float]],
    include_component_breakdown: bool = True,
    analyze_allocation_patterns: bool = True,
    memory_limits: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Creates comprehensive memory usage baseline data with component breakdown, allocation patterns,
    and resource consumption analysis for memory performance validation and leak detection.

    Args:
        memory_measurements: Dictionary mapping component names to memory usage measurements (in MB)
        include_component_breakdown: Whether to include detailed component analysis
        analyze_allocation_patterns: Whether to analyze memory allocation patterns for leak detection
        memory_limits: Optional memory limits for validation

    Returns:
        dict: Comprehensive memory usage baseline with component analysis and allocation pattern detection
    """
    if not memory_measurements:
        raise ValueError("Memory measurements cannot be empty")

    # Use default memory limits if not provided
    if memory_limits is None:
        memory_limits = {
            "total_system_mb": MEMORY_LIMIT_TOTAL_MB,
            "per_component_mb": MEMORY_LIMIT_TOTAL_MB
            * 0.8,  # 80% of total for any single component
            "warning_threshold_mb": MEMORY_LIMIT_TOTAL_MB
            * 0.7,  # 70% warning threshold
            "critical_threshold_mb": MEMORY_LIMIT_TOTAL_MB
            * 0.9,  # 90% critical threshold
        }

    memory_baseline = {
        "baseline_id": f"memory_baseline_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "creation_timestamp": datetime.datetime.now().isoformat(),
        "component_count": len(memory_measurements),
        "memory_limits": memory_limits.copy(),
        "component_statistics": {},
        "overall_statistics": {},
        "resource_utilization": {},
        "validation_results": {},
    }

    # Calculate statistics for each component
    total_measurements = []
    component_peaks = {}
    component_baselines = {}

    for component_name, measurements in memory_measurements.items():
        if not measurements:
            continue

        # Calculate comprehensive statistics for this component
        component_stats = calculate_performance_statistics(
            measurements=measurements, include_distribution_analysis=True
        )

        # Add memory-specific analysis
        peak_usage = max(measurements)
        baseline_usage = statistics.median(measurements)  # Use median as baseline
        average_usage = statistics.mean(measurements)

        component_analysis = {
            "measurement_count": len(measurements),
            "peak_usage_mb": peak_usage,
            "baseline_usage_mb": baseline_usage,
            "average_usage_mb": average_usage,
            "min_usage_mb": min(measurements),
            "usage_range_mb": max(measurements) - min(measurements),
            "standard_deviation_mb": component_stats["variability_statistics"][
                "std_dev"
            ],
            "coefficient_of_variation_percent": component_stats[
                "variability_statistics"
            ]["coefficient_of_variation"],
            "percentiles": component_stats["percentiles"],
            "memory_efficiency": {
                "stable": component_stats["variability_statistics"][
                    "coefficient_of_variation"
                ]
                < 10,
                "consistent": component_stats["variability_statistics"][
                    "coefficient_of_variation"
                ]
                < 20,
                "efficiency_rating": (
                    "excellent"
                    if component_stats["variability_statistics"][
                        "coefficient_of_variation"
                    ]
                    < 5
                    else (
                        "good"
                        if component_stats["variability_statistics"][
                            "coefficient_of_variation"
                        ]
                        < 15
                        else (
                            "moderate"
                            if component_stats["variability_statistics"][
                                "coefficient_of_variation"
                            ]
                            < 25
                            else "poor"
                        )
                    )
                ),
            },
        }

        # Check for memory limit violations
        limit_violations = {
            "peak_exceeds_component_limit": peak_usage
            > memory_limits.get("per_component_mb", float("inf")),
            "average_exceeds_warning": average_usage
            > memory_limits.get("warning_threshold_mb", float("inf")),
            "baseline_exceeds_warning": baseline_usage
            > memory_limits.get("warning_threshold_mb", float("inf")),
            "any_measurement_exceeds_critical": any(
                m > memory_limits.get("critical_threshold_mb", float("inf"))
                for m in measurements
            ),
        }

        component_analysis["limit_violations"] = limit_violations
        component_analysis["compliance_status"] = (
            "compliant" if not any(limit_violations.values()) else "non_compliant"
        )

        memory_baseline["component_statistics"][component_name] = component_analysis

        # Store for overall analysis
        total_measurements.extend(measurements)
        component_peaks[component_name] = peak_usage
        component_baselines[component_name] = baseline_usage

    # Calculate overall system statistics
    if total_measurements:
        overall_stats = calculate_performance_statistics(
            measurements=total_measurements, include_distribution_analysis=True
        )

        memory_baseline["overall_statistics"] = {
            "total_measurements": len(total_measurements),
            "system_peak_mb": max(total_measurements),
            "system_baseline_mb": statistics.median(total_measurements),
            "system_average_mb": statistics.mean(total_measurements),
            "system_min_mb": min(total_measurements),
            "system_range_mb": max(total_measurements) - min(total_measurements),
            "system_std_dev_mb": overall_stats["variability_statistics"]["std_dev"],
            "system_cv_percent": overall_stats["variability_statistics"][
                "coefficient_of_variation"
            ],
        }

        # System-level limit checking
        system_peak = memory_baseline["overall_statistics"]["system_peak_mb"]
        system_average = memory_baseline["overall_statistics"]["system_average_mb"]

        memory_baseline["overall_statistics"]["system_compliance"] = {
            "peak_within_limits": system_peak
            <= memory_limits.get("total_system_mb", float("inf")),
            "average_within_warning": system_average
            <= memory_limits.get("warning_threshold_mb", float("inf")),
            "system_utilization_percent": (
                system_peak / memory_limits.get("total_system_mb", 1)
            )
            * 100,
            "headroom_mb": memory_limits.get("total_system_mb", 0) - system_peak,
        }

    # Component breakdown analysis
    if include_component_breakdown:
        breakdown_analysis = {
            "component_contribution": {},
            "resource_distribution": {},
            "optimization_opportunities": [],
        }

        total_peak = sum(component_peaks.values()) if component_peaks else 1
        total_baseline = sum(component_baselines.values()) if component_baselines else 1

        for component_name in component_peaks:
            peak_contribution = (component_peaks[component_name] / total_peak) * 100
            baseline_contribution = (
                component_baselines[component_name] / total_baseline
            ) * 100

            breakdown_analysis["component_contribution"][component_name] = {
                "peak_contribution_percent": peak_contribution,
                "baseline_contribution_percent": baseline_contribution,
                "dominance": (
                    "high"
                    if peak_contribution > 50
                    else "medium" if peak_contribution > 25 else "low"
                ),
            }

            # Identify optimization opportunities
            if peak_contribution > 40:
                breakdown_analysis["optimization_opportunities"].append(
                    f"Component '{component_name}' dominates memory usage ({peak_contribution:.1f}%) - consider optimization"
                )

            component_stats = memory_baseline["component_statistics"][component_name]
            if component_stats["memory_efficiency"]["efficiency_rating"] == "poor":
                breakdown_analysis["optimization_opportunities"].append(
                    f"Component '{component_name}' shows high memory usage variability - investigate allocation patterns"
                )

        memory_baseline["component_breakdown"] = breakdown_analysis

    # Allocation pattern analysis
    if analyze_allocation_patterns:
        pattern_analysis = {
            "leak_detection": {},
            "allocation_trends": {},
            "pattern_classification": {},
        }

        for component_name, measurements in memory_measurements.items():
            if len(measurements) < 10:  # Need sufficient data for pattern analysis
                continue

            # Simple leak detection: look for consistent upward trend
            # Calculate trend over entire measurement period
            x = list(range(len(measurements)))
            y = measurements

            # Linear regression for trend detection
            n = len(x)
            sum_x = sum(x)
            sum_y = sum(y)
            sum_xy = sum(xi * yi for xi, yi in zip(x, y))
            sum_x2 = sum(xi * xi for xi in x)

            if n * sum_x2 - sum_x * sum_x != 0:
                slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)

                # Classify allocation pattern
                if slope > 0.1:  # Increasing by more than 0.1 MB per measurement
                    pattern_type = "increasing"
                    leak_risk = (
                        "high" if slope > 1.0 else "medium" if slope > 0.5 else "low"
                    )
                elif slope < -0.1:
                    pattern_type = "decreasing"
                    leak_risk = "none"
                else:
                    pattern_type = "stable"
                    leak_risk = "none"

                # Additional leak indicators
                recent_trend = (
                    measurements[-5:] if len(measurements) >= 5 else measurements
                )
                early_trend = (
                    measurements[:5]
                    if len(measurements) >= 10
                    else measurements[: len(measurements) // 2]
                )

                if len(recent_trend) >= 3 and len(early_trend) >= 3:
                    recent_avg = statistics.mean(recent_trend)
                    early_avg = statistics.mean(early_trend)
                    growth_rate = (
                        (recent_avg - early_avg) / early_avg if early_avg > 0 else 0
                    )

                    if growth_rate > 0.2:  # 20% growth
                        leak_risk = "high" if leak_risk == "medium" else leak_risk

                pattern_analysis["leak_detection"][component_name] = {
                    "trend_slope": slope,
                    "pattern_type": pattern_type,
                    "leak_risk": leak_risk,
                    "growth_rate_percent": (
                        growth_rate * 100 if "growth_rate" in locals() else 0
                    ),
                    "recommendation": (
                        "Investigate for memory leaks - consistent upward trend detected"
                        if leak_risk == "high"
                        else (
                            "Monitor for potential memory issues"
                            if leak_risk == "medium"
                            else "Memory usage appears stable"
                        )
                    ),
                }

        memory_baseline["allocation_patterns"] = pattern_analysis

    # Generate validation results and recommendations
    validation_issues = []
    recommendations = []

    # Check for system-level issues
    if (
        memory_baseline.get("overall_statistics", {})
        .get("system_compliance", {})
        .get("system_utilization_percent", 0)
        > 80
    ):
        validation_issues.append("High system memory utilization detected")
        recommendations.append(
            "Consider memory optimization or increasing system memory limits"
        )

    # Check component-level issues
    for component_name, stats in memory_baseline.get(
        "component_statistics", {}
    ).items():
        if stats.get("compliance_status") == "non_compliant":
            validation_issues.append(
                f"Component '{component_name}' exceeds memory limits"
            )
            recommendations.append(
                f"Optimize memory usage for component '{component_name}'"
            )

        if stats.get("memory_efficiency", {}).get("efficiency_rating") == "poor":
            validation_issues.append(
                f"Component '{component_name}' shows poor memory efficiency"
            )
            recommendations.append(
                f"Investigate memory allocation patterns for '{component_name}'"
            )

    # Check for potential memory leaks
    if "allocation_patterns" in memory_baseline:
        high_risk_components = [
            comp
            for comp, analysis in memory_baseline["allocation_patterns"][
                "leak_detection"
            ].items()
            if analysis.get("leak_risk") == "high"
        ]

        if high_risk_components:
            validation_issues.append(
                f"Potential memory leaks detected in: {', '.join(high_risk_components)}"
            )
            recommendations.append(
                "Immediate investigation required for potential memory leaks"
            )

    memory_baseline["validation_results"] = {
        "overall_compliance": len(validation_issues) == 0,
        "issues_found": validation_issues,
        "recommendations": recommendations,
        "validation_summary": (
            "Memory usage is compliant and efficient"
            if len(validation_issues) == 0
            else f"{len(validation_issues)} issues found requiring attention"
        ),
    }

    return memory_baseline


# Export comprehensive public interface for benchmark data module
__all__ = [
    # Factory functions for creating performance baseline data
    "get_performance_baseline",
    "create_benchmark_scenario",
    "get_scalability_test_data",
    "get_benchmark_test_matrix",
    # Data structure classes
    "PerformanceBaseline",
    "BenchmarkScenario",
    "ScalabilityTestData",
    # Global performance target definitions
    "PERFORMANCE_TARGETS",
    # Analysis and utility functions
    "generate_baseline_data",
    "validate_benchmark_data",
    "calculate_performance_statistics",
    "get_regression_detection_data",
    "create_memory_usage_baseline",
]
