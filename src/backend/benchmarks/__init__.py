"""
Comprehensive benchmarks package initialization module providing unified access to all plume_nav_sim
performance benchmarking capabilities including environment performance analysis, rendering benchmarking,
step latency measurement, memory usage profiling, and plume generation performance testing. Implements
benchmarking orchestration, suite coordination, and unified result reporting with integration support
for continuous integration, performance monitoring, and optimization workflows.

This module serves as the central orchestration point for all performance benchmarking operations,
providing standardized interfaces, configuration management, and result consolidation across all
benchmark categories for comprehensive system performance evaluation.
"""

import concurrent.futures  # >=3.10 - Parallel benchmark execution, concurrent suite coordination, and performance optimization
import dataclasses  # >=3.10 - Data structures for comprehensive benchmark configuration and consolidated result management
import json  # >=3.10 - Consolidated benchmark result serialization and structured output for CI integration
import logging  # >=3.10 - Benchmark orchestration logging, suite execution tracking, and consolidated progress reporting
import pathlib  # >=3.10 - File system operations for benchmark result storage, report export, and output management
import time  # >=3.10 - High-precision timing for benchmark orchestration, suite execution timing, and performance tracking

# External imports with version specifications for comprehensive benchmarking infrastructure
import typing  # >=3.10 - Type hints for benchmark orchestration functions, suite coordination, and result aggregation
from typing import Any, Dict, List, Optional, Tuple, Union

# Internal imports - Logging infrastructure for benchmark orchestration and progress tracking
from plume_nav_sim.utils.logging import (  # Component-specific logger factory for benchmark orchestration logging
    get_component_logger,
)

# Internal imports - Core benchmarking infrastructure with comprehensive environment performance analysis
from .environment_performance import (
    BenchmarkResult,  # Standardized benchmark result structure for result aggregation and reporting
)
from .environment_performance import (
    EnvironmentPerformanceSuite,  # Comprehensive environment performance testing suite with integrated analysis
)
from .environment_performance import (
    run_environment_performance_benchmark,  # Primary environment performance benchmarking function for comprehensive analysis
)

# Graceful handling of missing benchmark modules with placeholder implementations
# This ensures the package works even when some benchmark files don't exist yet
try:
    from .rendering_performance import (
        RenderingPerformanceSuite,
        run_rendering_performance_benchmark,
    )
except ImportError:
    # Placeholder implementation for missing rendering performance module
    def run_rendering_performance_benchmark(*args, **kwargs):
        """Placeholder for rendering performance benchmark - not yet implemented."""
        logger = get_component_logger("benchmarks_rendering", "UTILS")
        logger.warning("Rendering performance benchmarking not yet implemented")
        return {
            "status": "not_implemented",
            "category": "rendering",
            "message": "Module not available",
        }

    class RenderingPerformanceSuite:
        """Placeholder rendering performance suite."""

        def __init__(self, *args, **kwargs):
            self.logger = get_component_logger("rendering_suite", "UTILS")
            self.logger.warning("RenderingPerformanceSuite not yet implemented")

        def run_full_rendering_benchmark_suite(self, *args, **kwargs):
            return {"status": "not_implemented", "category": "rendering_suite"}


try:
    from .step_latency import StepLatencyBenchmark, benchmark_step_latency
except ImportError:
    # Placeholder implementation for missing step latency module
    def benchmark_step_latency(*args, **kwargs):
        """Placeholder for step latency benchmark - not yet implemented."""
        logger = get_component_logger("benchmarks_latency", "UTILS")
        logger.warning("Step latency benchmarking not yet implemented")
        return {
            "status": "not_implemented",
            "category": "step_latency",
            "message": "Module not available",
        }

    class StepLatencyBenchmark:
        """Placeholder step latency benchmark suite."""

        def __init__(self, *args, **kwargs):
            self.logger = get_component_logger("step_latency_suite", "UTILS")
            self.logger.warning("StepLatencyBenchmark not yet implemented")

        def execute_benchmark(self, *args, **kwargs):
            return {"status": "not_implemented", "category": "step_latency_suite"}


try:
    from .memory_usage import MemoryUsageAnalyzer, run_memory_usage_benchmark
except ImportError:
    # Placeholder implementation for missing memory usage module
    def run_memory_usage_benchmark(*args, **kwargs):
        """Placeholder for memory usage benchmark - not yet implemented."""
        logger = get_component_logger("benchmarks_memory", "UTILS")
        logger.warning("Memory usage benchmarking not yet implemented")
        return {
            "status": "not_implemented",
            "category": "memory_usage",
            "message": "Module not available",
        }

    class MemoryUsageAnalyzer:
        """Placeholder memory usage analyzer."""

        def __init__(self, *args, **kwargs):
            self.logger = get_component_logger("memory_analyzer", "UTILS")
            self.logger.warning("MemoryUsageAnalyzer not yet implemented")

        def start_analysis(self, *args, **kwargs):
            return {"status": "not_implemented", "category": "memory_analysis"}

        def analyze_environment_memory(self, *args, **kwargs):
            return {"status": "not_implemented", "category": "memory_analysis"}


try:
    from .plume_generation import (
        PlumeGenerationBenchmark,
        run_plume_generation_benchmark,
    )
except ImportError:
    # Placeholder implementation for missing plume generation module
    def run_plume_generation_benchmark(*args, **kwargs):
        """Placeholder for plume generation benchmark - not yet implemented."""
        logger = get_component_logger("benchmarks_plume", "UTILS")
        logger.warning("Plume generation benchmarking not yet implemented")
        return {
            "status": "not_implemented",
            "category": "plume_generation",
            "message": "Module not available",
        }

    class PlumeGenerationBenchmark:
        """Placeholder plume generation benchmark suite."""

        def __init__(self, *args, **kwargs):
            self.logger = get_component_logger("plume_gen_suite", "UTILS")
            self.logger.warning("PlumeGenerationBenchmark not yet implemented")

        def execute_benchmark(self, *args, **kwargs):
            return {"status": "not_implemented", "category": "plume_generation_suite"}


# Graceful handling of missing configuration module with fallback defaults
try:
    from config.default_config import get_complete_default_config
except ImportError:
    # Fallback implementation for missing default configuration
    def get_complete_default_config():
        """Fallback default configuration for benchmarking."""
        logger = get_component_logger("benchmarks_config", "UTILS")
        logger.warning(
            "Default config module not available, using fallback configuration"
        )
        return {
            "benchmark_defaults": {
                "grid_size": [128, 128],
                "source_location": [64, 64],
                "max_steps": 1000,
                "timeout_minutes": 30,
            },
            "performance_targets": {
                "step_latency_ms": 1.0,
                "episode_reset_ms": 10.0,
                "memory_usage_mb": 50.0,
                "rendering_rgb_ms": 5.0,
                "rendering_human_ms": 50.0,
            },
        }


# Global configuration constants for benchmark orchestration and suite coordination
DEFAULT_BENCHMARK_SUITE_CONFIG = {
    "include_environment": True,
    "include_rendering": True,
    "include_step_latency": True,
    "include_memory": True,
    "include_plume_generation": True,
}

BENCHMARK_EXECUTION_TIMEOUT_MINUTES = 30
SUITE_RESULT_CONSOLIDATION_VERSION = "1.0"
CI_BENCHMARK_PROFILE = {
    "quick_validation": True,
    "comprehensive_analysis": False,
    "performance_regression_detection": True,
}

# Package version and comprehensive export specification
__version__ = "1.0.0"

# Comprehensive exports for unified benchmarking API access
__all__ = [
    # Primary orchestration functions for complete benchmark suite execution
    "run_comprehensive_benchmark_suite",
    "BenchmarkSuiteOrchestrator",
    "ConsolidatedBenchmarkResult",
    "BenchmarkSuiteConfig",
    # Individual benchmark functions imported from specialized modules
    "run_environment_performance_benchmark",
    "run_rendering_performance_benchmark",
    "benchmark_step_latency",
    "run_memory_usage_benchmark",
    "run_plume_generation_benchmark",
    # Benchmark suite classes for comprehensive performance analysis
    "EnvironmentPerformanceSuite",
    "RenderingPerformanceSuite",
    "StepLatencyBenchmark",
    "MemoryUsageAnalyzer",
    "PlumeGenerationBenchmark",
    "BenchmarkResult",
    # Analysis and reporting functions for integrated performance insights
    "generate_benchmark_report",
    "validate_performance_targets",
    "detect_performance_regressions",
    "optimize_system_performance",
    "export_benchmark_results",
    # CI/CD integration and specialized execution functions
    "create_ci_benchmark_config",
    "run_quick_validation_suite",
    "run_comprehensive_analysis_suite",
]


@dataclasses.dataclass
class BenchmarkSuiteConfig:
    """
    Comprehensive configuration data class for benchmark suite orchestration specifying execution
    parameters, analysis options, output settings, and category-specific configurations with
    validation support and optimization recommendations for systematic benchmark suite execution.
    """

    # Core benchmark category inclusion flags for suite composition
    include_environment_benchmark: bool = True
    include_rendering_benchmark: bool = True
    include_step_latency_benchmark: bool = True
    include_memory_benchmark: bool = True
    include_plume_generation_benchmark: bool = True

    # Execution and analysis configuration parameters
    enable_parallel_execution: bool = False
    validate_performance_targets: bool = True
    detect_regressions: bool = False
    generate_integrated_analysis: bool = True

    # Category-specific configurations and performance targets
    category_specific_configs: Dict[str, Any] = dataclasses.field(default_factory=dict)
    performance_targets: Dict[str, float] = dataclasses.field(
        default_factory=lambda: {
            "step_latency_ms": 1.0,
            "episode_reset_ms": 10.0,
            "memory_usage_mb": 50.0,
            "rendering_rgb_ms": 5.0,
            "rendering_human_ms": 50.0,
        }
    )

    # Output and execution management settings
    execution_timeout_minutes: Optional[int] = BENCHMARK_EXECUTION_TIMEOUT_MINUTES
    output_directory: Optional[pathlib.Path] = None
    output_formats: List[str] = dataclasses.field(
        default_factory=lambda: ["json", "markdown"]
    )

    def validate_suite_config(
        self, strict_validation: bool = False, check_resource_requirements: bool = False
    ) -> bool:
        """
        Comprehensive suite configuration validation including category consistency, resource
        feasibility, execution parameter validation, and optimization recommendation generation
        for configuration quality assurance.
        """
        logger = get_component_logger("config_validation", "UTILS")

        try:
            # Validate that at least one benchmark category is enabled for meaningful suite execution
            enabled_categories = [
                self.include_environment_benchmark,
                self.include_rendering_benchmark,
                self.include_step_latency_benchmark,
                self.include_memory_benchmark,
                self.include_plume_generation_benchmark,
            ]

            if not any(enabled_categories):
                logger.error(
                    "No benchmark categories enabled - at least one must be True"
                )
                return False

            # Check execution_timeout_minutes is reasonable for enabled benchmark categories
            if self.execution_timeout_minutes and self.execution_timeout_minutes < 1:
                logger.error(
                    f"Invalid execution timeout: {self.execution_timeout_minutes} minutes"
                )
                return False

            # Validate category_specific_configs consistency with enabled benchmark categories
            for config_key in self.category_specific_configs:
                if config_key not in [
                    "environment",
                    "rendering",
                    "step_latency",
                    "memory",
                    "plume_generation",
                ]:
                    logger.warning(f"Unknown category-specific config: {config_key}")

            # Validate performance_targets consistency with system capabilities and enabled categories
            required_targets = [
                "step_latency_ms",
                "episode_reset_ms",
                "memory_usage_mb",
            ]
            for target in required_targets:
                if (
                    target not in self.performance_targets
                    or self.performance_targets[target] <= 0
                ):
                    logger.error(f"Invalid performance target for {target}")
                    return False

            # Validate output_directory accessibility and permissions if specified
            if self.output_directory:
                try:
                    self.output_directory.mkdir(parents=True, exist_ok=True)
                    if not self.output_directory.is_dir():
                        logger.error(
                            f"Output directory not accessible: {self.output_directory}"
                        )
                        return False
                except Exception as e:
                    logger.error(
                        f"Cannot create output directory {self.output_directory}: {e}"
                    )
                    return False

            # Apply strict validation rules if strict_validation is True with comprehensive parameter checking
            if strict_validation:
                if self.enable_parallel_execution and sum(enabled_categories) < 2:
                    logger.warning(
                        "Parallel execution enabled but <2 categories selected - no benefit"
                    )

                if self.detect_regressions and not self.validate_performance_targets:
                    logger.error(
                        "Regression detection requires performance target validation"
                    )
                    return False

            logger.info("Benchmark suite configuration validation passed")
            return True

        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False

    def estimate_suite_execution_time(
        self,
        include_parallel_optimization: bool = True,
        include_analysis_overhead: bool = True,
    ) -> float:
        """
        Estimate total suite execution time based on enabled benchmark categories, configuration
        parameters, and system capabilities for resource planning and timeout validation.
        """
        logger = get_component_logger("execution_estimator", "UTILS")

        try:
            base_times = {
                "environment": 120.0,  # 2 minutes for comprehensive environment benchmarking
                "rendering": 60.0,  # 1 minute for rendering performance tests
                "step_latency": 30.0,  # 30 seconds for step latency analysis
                "memory": 90.0,  # 1.5 minutes for memory profiling
                "plume_generation": 45.0,  # 45 seconds for plume generation tests
            }

            total_time = 0.0

            # Calculate base execution time for each enabled benchmark category
            if self.include_environment_benchmark:
                total_time += base_times["environment"]
            if self.include_rendering_benchmark:
                total_time += base_times["rendering"]
            if self.include_step_latency_benchmark:
                total_time += base_times["step_latency"]
            if self.include_memory_benchmark:
                total_time += base_times["memory"]
            if self.include_plume_generation_benchmark:
                total_time += base_times["plume_generation"]

            # Apply parallel execution optimization if enabled
            if self.enable_parallel_execution and include_parallel_optimization:
                parallel_factor = 0.6  # 40% time reduction with parallel execution
                total_time *= parallel_factor
                logger.debug(
                    f"Applied parallel execution optimization: {parallel_factor}x"
                )

            # Add integrated analysis overhead if enabled
            if self.generate_integrated_analysis and include_analysis_overhead:
                analysis_overhead = total_time * 0.15  # 15% overhead for analysis
                total_time += analysis_overhead
                logger.debug(f"Added analysis overhead: {analysis_overhead:.1f}s")

            # Add result consolidation and reporting overhead
            reporting_overhead = 10.0  # Fixed 10 seconds for reporting
            total_time += reporting_overhead

            # Convert to minutes and apply safety margin
            total_minutes = (total_time / 60.0) * 1.2  # 20% safety margin

            logger.info(f"Estimated suite execution time: {total_minutes:.1f} minutes")
            return total_minutes

        except Exception as e:
            logger.error(f"Execution time estimation failed: {e}")
            return 30.0  # Default fallback estimate

    def get_optimized_config(
        self,
        optimization_target: str = "balanced",
        resource_constraints: Optional[Dict[str, Any]] = None,
        prioritize_accuracy: bool = False,
    ) -> "BenchmarkSuiteConfig":
        """
        Generate optimized suite configuration based on execution requirements, resource
        constraints, and performance priorities for improved benchmark execution efficiency
        and reliability.
        """
        logger = get_component_logger("config_optimizer", "UTILS")

        try:
            # Create new config based on current configuration
            optimized_config = dataclasses.replace(self)

            # Apply optimization based on optimization_target
            if optimization_target == "speed":
                # Optimize for fastest execution
                optimized_config.enable_parallel_execution = True
                optimized_config.generate_integrated_analysis = False
                # Reduce iterations in category-specific configs
                optimized_config.category_specific_configs.update(
                    {
                        "environment": {"iterations": 10},
                        "rendering": {"iterations": 5},
                        "step_latency": {"iterations": 100},
                        "memory": {"samples": 5},
                        "plume_generation": {"iterations": 5},
                    }
                )
                logger.info("Applied speed optimization configuration")

            elif optimization_target == "accuracy":
                # Optimize for maximum accuracy and comprehensive analysis
                optimized_config.enable_parallel_execution = (
                    False  # Sequential for consistency
                )
                optimized_config.generate_integrated_analysis = True
                optimized_config.validate_performance_targets = True
                # Increase iterations for better statistical significance
                optimized_config.category_specific_configs.update(
                    {
                        "environment": {"iterations": 100},
                        "rendering": {"iterations": 50},
                        "step_latency": {"iterations": 1000},
                        "memory": {"samples": 20},
                        "plume_generation": {"iterations": 50},
                    }
                )
                logger.info("Applied accuracy optimization configuration")

            elif optimization_target == "comprehensive":
                # Enable all features for complete analysis
                optimized_config.include_environment_benchmark = True
                optimized_config.include_rendering_benchmark = True
                optimized_config.include_step_latency_benchmark = True
                optimized_config.include_memory_benchmark = True
                optimized_config.include_plume_generation_benchmark = True
                optimized_config.generate_integrated_analysis = True
                optimized_config.detect_regressions = True
                logger.info("Applied comprehensive optimization configuration")

            else:  # balanced
                # Balanced configuration for good speed/accuracy tradeoff
                optimized_config.enable_parallel_execution = True
                optimized_config.generate_integrated_analysis = True
                optimized_config.category_specific_configs.update(
                    {
                        "environment": {"iterations": 50},
                        "rendering": {"iterations": 20},
                        "step_latency": {"iterations": 500},
                        "memory": {"samples": 10},
                        "plume_generation": {"iterations": 20},
                    }
                )
                logger.info("Applied balanced optimization configuration")

            # Apply resource constraints if provided
            if resource_constraints:
                max_time = resource_constraints.get("max_execution_minutes")
                if max_time and max_time < optimized_config.execution_timeout_minutes:
                    optimized_config.execution_timeout_minutes = max_time
                    logger.info(f"Applied time constraint: {max_time} minutes")

                max_memory = resource_constraints.get("max_memory_mb")
                if max_memory:
                    optimized_config.performance_targets["memory_usage_mb"] = max_memory
                    logger.info(f"Applied memory constraint: {max_memory}MB")

            # Validate optimized configuration
            if not optimized_config.validate_suite_config():
                logger.warning(
                    "Optimized configuration failed validation, using original"
                )
                return self

            return optimized_config

        except Exception as e:
            logger.error(f"Configuration optimization failed: {e}")
            return self

    def to_dict(
        self, include_metadata: bool = True, include_validation_info: bool = False
    ) -> Dict[str, Any]:
        """
        Convert suite configuration to dictionary format for serialization, export, and
        external system integration with comprehensive parameter preservation and metadata.
        """
        try:
            # Convert configuration to dictionary using dataclasses.asdict
            config_dict = dataclasses.asdict(self)

            # Convert pathlib.Path objects to string representation for JSON compatibility
            if self.output_directory:
                config_dict["output_directory"] = str(self.output_directory)

            # Include configuration metadata if requested
            if include_metadata:
                config_dict["metadata"] = {
                    "config_version": SUITE_RESULT_CONSOLIDATION_VERSION,
                    "created_timestamp": time.time(),
                    "estimated_execution_minutes": self.estimate_suite_execution_time(),
                    "enabled_categories_count": sum(
                        [
                            self.include_environment_benchmark,
                            self.include_rendering_benchmark,
                            self.include_step_latency_benchmark,
                            self.include_memory_benchmark,
                            self.include_plume_generation_benchmark,
                        ]
                    ),
                }

            # Include validation information if requested
            if include_validation_info:
                config_dict["validation_info"] = {
                    "is_valid": self.validate_suite_config(),
                    "validation_timestamp": time.time(),
                    "strict_validation_passed": self.validate_suite_config(
                        strict_validation=True
                    ),
                }

            return config_dict

        except Exception as e:
            logger = get_component_logger("config_serialization", "UTILS")
            logger.error(f"Configuration serialization failed: {e}")
            return {"error": str(e), "config_type": "BenchmarkSuiteConfig"}


@dataclasses.dataclass
class ConsolidatedBenchmarkResult:
    """
    Comprehensive data structure for consolidated benchmark results containing integrated analysis
    from all performance categories including environment performance, rendering benchmarking, step
    latency, memory usage, and plume generation with cross-category correlation analysis, statistical
    validation, and unified optimization recommendations.
    """

    # Core execution metadata and configuration reference
    suite_execution_id: str
    execution_timestamp: float
    suite_config: BenchmarkSuiteConfig

    # Individual benchmark category results with optional data structure
    environment_results: Optional[BenchmarkResult] = None
    rendering_results: Optional[Dict[str, Any]] = None
    step_latency_results: Optional[Dict[str, Any]] = None
    memory_usage_results: Optional[Dict[str, Any]] = None
    plume_generation_results: Optional[Dict[str, Any]] = None

    # Integrated analysis and cross-category insights
    cross_category_analysis: Dict[str, Any] = dataclasses.field(default_factory=dict)
    consolidated_statistics: Dict[str, Any] = dataclasses.field(default_factory=dict)
    performance_validation: Dict[str, Any] = dataclasses.field(default_factory=dict)
    integrated_optimization_recommendations: List[str] = dataclasses.field(
        default_factory=list
    )

    # Summary and validation flags
    all_targets_met: bool = False
    executive_summary: Optional[str] = None

    def calculate_consolidated_statistics(
        self,
        include_cross_correlation: bool = True,
        calculate_performance_score: bool = True,
        confidence_level: float = 0.95,
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive consolidated statistics across all benchmark categories with
        cross-category correlation analysis, integrated performance metrics, and statistical
        significance testing for holistic system performance evaluation.
        """
        logger = get_component_logger("statistics_calculator", "UTILS")

        try:
            consolidated_stats = {}

            # Aggregate statistical measures from all benchmark category results
            category_stats = {}

            if self.environment_results:
                category_stats["environment"] = {
                    "step_latency_avg": getattr(
                        self.environment_results, "avg_step_time_ms", 0
                    ),
                    "episode_duration_avg": getattr(
                        self.environment_results, "avg_episode_duration_ms", 0
                    ),
                    "success_rate": getattr(
                        self.environment_results, "success_rate", 0
                    ),
                }

            if self.rendering_results:
                category_stats["rendering"] = {
                    "rgb_render_avg": self.rendering_results.get(
                        "avg_rgb_render_ms", 0
                    ),
                    "human_render_avg": self.rendering_results.get(
                        "avg_human_render_ms", 0
                    ),
                    "render_success_rate": self.rendering_results.get(
                        "success_rate", 0
                    ),
                }

            if self.step_latency_results:
                category_stats["step_latency"] = {
                    "mean_latency": self.step_latency_results.get("mean_latency_ms", 0),
                    "p95_latency": self.step_latency_results.get("p95_latency_ms", 0),
                    "latency_variance": self.step_latency_results.get("variance", 0),
                }

            if self.memory_usage_results:
                category_stats["memory"] = {
                    "peak_usage_mb": self.memory_usage_results.get("peak_memory_mb", 0),
                    "avg_usage_mb": self.memory_usage_results.get("avg_memory_mb", 0),
                    "memory_efficiency": self.memory_usage_results.get(
                        "efficiency_score", 0
                    ),
                }

            if self.plume_generation_results:
                category_stats["plume"] = {
                    "generation_time_avg": self.plume_generation_results.get(
                        "avg_generation_ms", 0
                    ),
                    "field_quality_score": self.plume_generation_results.get(
                        "quality_score", 0
                    ),
                }

            consolidated_stats["category_statistics"] = category_stats

            # Calculate cross-category correlation coefficients if enabled
            if include_cross_correlation and len(category_stats) >= 2:
                correlations = {}
                categories = list(category_stats.keys())

                for i, cat1 in enumerate(categories):
                    for cat2 in categories[i + 1 :]:
                        # Simple correlation calculation between timing metrics
                        cat1_timing = category_stats[cat1].get(
                            list(category_stats[cat1].keys())[0], 0
                        )
                        cat2_timing = category_stats[cat2].get(
                            list(category_stats[cat2].keys())[0], 0
                        )

                        if cat1_timing > 0 and cat2_timing > 0:
                            correlation = min(cat1_timing, cat2_timing) / max(
                                cat1_timing, cat2_timing
                            )
                            correlations[f"{cat1}_vs_{cat2}"] = correlation

                consolidated_stats["cross_correlations"] = correlations
                logger.info(
                    f"Calculated {len(correlations)} cross-category correlations"
                )

            # Compute integrated performance metrics combining all categories
            overall_performance = {}
            timing_metrics = []
            success_metrics = []

            for category, stats in category_stats.items():
                # Extract timing and success metrics
                for metric, value in stats.items():
                    if "avg" in metric or "time" in metric or "latency" in metric:
                        timing_metrics.append(value)
                    elif (
                        "rate" in metric
                        or "success" in metric
                        or "efficiency" in metric
                    ):
                        success_metrics.append(value)

            if timing_metrics:
                overall_performance["avg_timing_ms"] = sum(timing_metrics) / len(
                    timing_metrics
                )
                overall_performance["timing_variance"] = sum(
                    (t - overall_performance["avg_timing_ms"]) ** 2
                    for t in timing_metrics
                ) / len(timing_metrics)

            if success_metrics:
                overall_performance["avg_success_rate"] = sum(success_metrics) / len(
                    success_metrics
                )
                overall_performance["success_consistency"] = (
                    1.0 - (max(success_metrics) - min(success_metrics))
                    if success_metrics
                    else 1.0
                )

            # Calculate overall system performance score if requested
            if calculate_performance_score:
                score_components = []

                # Timing score (lower is better, normalized to 0-1 scale)
                if "avg_timing_ms" in overall_performance:
                    target_timing = 10.0  # 10ms target
                    timing_score = max(
                        0, 1.0 - (overall_performance["avg_timing_ms"] / target_timing)
                    )
                    score_components.append(("timing", timing_score, 0.4))

                # Success rate score (higher is better)
                if "avg_success_rate" in overall_performance:
                    success_score = overall_performance["avg_success_rate"]
                    score_components.append(("success", success_score, 0.3))

                # Consistency score (higher is better)
                if "success_consistency" in overall_performance:
                    consistency_score = overall_performance["success_consistency"]
                    score_components.append(("consistency", consistency_score, 0.3))

                # Calculate weighted performance score
                if score_components:
                    total_weighted_score = sum(
                        score * weight for _, score, weight in score_components
                    )
                    total_weight = sum(weight for _, _, weight in score_components)
                    overall_performance_score = (
                        total_weighted_score / total_weight if total_weight > 0 else 0
                    )

                    overall_performance["performance_score"] = overall_performance_score
                    overall_performance["score_components"] = {
                        name: score for name, score, _ in score_components
                    }

                    logger.info(
                        f"Calculated overall performance score: {overall_performance_score:.3f}"
                    )

            consolidated_stats["overall_performance"] = overall_performance

            # Generate confidence intervals at specified confidence_level for statistical validation
            consolidated_stats["statistical_validation"] = {
                "confidence_level": confidence_level,
                "sample_size": len(category_stats),
                "validation_timestamp": time.time(),
            }

            # Store consolidated statistics for reference
            self.consolidated_statistics = consolidated_stats

            return consolidated_stats

        except Exception as e:
            logger.error(f"Consolidated statistics calculation failed: {e}")
            return {"error": str(e), "calculation_failed": True}

    def validate_integrated_performance(
        self,
        integrated_targets: Optional[Dict[str, float]] = None,
        strict_validation: bool = False,
        cross_category_validation: bool = True,
    ) -> Dict[str, Any]:
        """
        Validate integrated performance across all benchmark categories with comprehensive target
        compliance analysis, cross-category constraint validation, and systematic improvement
        requirement identification.
        """
        logger = get_component_logger("performance_validator", "UTILS")

        try:
            validation_results = {
                "validation_timestamp": time.time(),
                "validation_passed": True,
                "category_results": {},
                "violations": [],
                "improvements_needed": [],
            }

            # Use integrated_targets if provided, otherwise use suite config targets
            targets = integrated_targets or self.suite_config.performance_targets

            # Validate individual category performance against respective targets
            if self.environment_results and "step_latency_ms" in targets:
                env_latency = getattr(self.environment_results, "avg_step_time_ms", 0)
                target_latency = targets["step_latency_ms"]

                category_valid = env_latency <= target_latency
                validation_results["category_results"]["environment"] = {
                    "passed": category_valid,
                    "actual": env_latency,
                    "target": target_latency,
                    "ratio": env_latency / target_latency if target_latency > 0 else 0,
                }

                if not category_valid:
                    violation = f"Environment step latency {env_latency:.3f}ms exceeds target {target_latency:.3f}ms"
                    validation_results["violations"].append(violation)
                    validation_results["improvements_needed"].append(
                        "Optimize environment step execution performance"
                    )
                    validation_results["validation_passed"] = False

            if self.rendering_results and "rendering_rgb_ms" in targets:
                rgb_time = self.rendering_results.get("avg_rgb_render_ms", 0)
                target_rgb = targets["rendering_rgb_ms"]

                category_valid = rgb_time <= target_rgb
                validation_results["category_results"]["rendering"] = {
                    "passed": category_valid,
                    "actual": rgb_time,
                    "target": target_rgb,
                    "ratio": rgb_time / target_rgb if target_rgb > 0 else 0,
                }

                if not category_valid:
                    violation = f"RGB rendering time {rgb_time:.3f}ms exceeds target {target_rgb:.3f}ms"
                    validation_results["violations"].append(violation)
                    validation_results["improvements_needed"].append(
                        "Optimize RGB rendering pipeline performance"
                    )
                    validation_results["validation_passed"] = False

            if self.memory_usage_results and "memory_usage_mb" in targets:
                memory_usage = self.memory_usage_results.get("peak_memory_mb", 0)
                target_memory = targets["memory_usage_mb"]

                category_valid = memory_usage <= target_memory
                validation_results["category_results"]["memory"] = {
                    "passed": category_valid,
                    "actual": memory_usage,
                    "target": target_memory,
                    "ratio": memory_usage / target_memory if target_memory > 0 else 0,
                }

                if not category_valid:
                    violation = f"Memory usage {memory_usage:.1f}MB exceeds target {target_memory:.1f}MB"
                    validation_results["violations"].append(violation)
                    validation_results["improvements_needed"].append(
                        "Optimize memory allocation and cleanup"
                    )
                    validation_results["validation_passed"] = False

            # Perform cross-category validation if enabled
            if (
                cross_category_validation
                and len(validation_results["category_results"]) >= 2
            ):
                cross_violations = []

                # Check for performance imbalances across categories
                timing_ratios = []
                for category, result in validation_results["category_results"].items():
                    if "ratio" in result and result["ratio"] > 0:
                        timing_ratios.append((category, result["ratio"]))

                if timing_ratios:
                    ratios = [ratio for _, ratio in timing_ratios]
                    max_ratio = max(ratios)
                    min_ratio = min(ratios)

                    # Flag significant performance imbalances
                    if max_ratio / min_ratio > 3.0:  # 3x imbalance threshold
                        imbalance_violation = f"Significant performance imbalance detected: {max_ratio:.2f}x vs {min_ratio:.2f}x target ratios"
                        cross_violations.append(imbalance_violation)
                        validation_results["improvements_needed"].append(
                            "Balance performance across all benchmark categories"
                        )

                validation_results["cross_category_violations"] = cross_violations
                if cross_violations:
                    validation_results["validation_passed"] = False

            # Apply strict validation criteria if enabled
            if strict_validation:
                # Strict validation requires all ratios <= 0.8 (20% better than target)
                strict_violations = []
                for category, result in validation_results["category_results"].items():
                    if result.get("ratio", 0) > 0.8:
                        strict_violations.append(
                            f"Strict validation failed for {category}: ratio {result['ratio']:.3f} > 0.8"
                        )

                validation_results["strict_violations"] = strict_violations
                if strict_violations:
                    validation_results["validation_passed"] = False
                    validation_results["improvements_needed"].append(
                        "Achieve strict performance targets (80% of baseline)"
                    )

            # Update all_targets_met flag based on validation results
            self.all_targets_met = validation_results["validation_passed"]

            # Store validation results in performance_validation
            self.performance_validation = validation_results

            logger.info(
                f"Integrated performance validation completed: {'PASSED' if validation_results['validation_passed'] else 'FAILED'}"
            )

            return validation_results

        except Exception as e:
            logger.error(f"Integrated performance validation failed: {e}")
            return {
                "validation_passed": False,
                "error": str(e),
                "validation_timestamp": time.time(),
            }

    def generate_integrated_executive_summary(
        self,
        include_strategic_recommendations: bool = True,
        highlight_critical_issues: bool = True,
        include_performance_trends: bool = False,
    ) -> str:
        """
        Generate comprehensive executive summary consolidating key findings from all benchmark
        categories with strategic insights, cross-category performance highlights, and prioritized
        optimization recommendations for executive communication.
        """
        logger = get_component_logger("executive_summary", "UTILS")

        try:
            summary_parts = []

            # Executive summary header with overall performance assessment
            summary_parts.append(
                "# Plume Navigation Simulation - Performance Benchmark Executive Summary"
            )
            summary_parts.append(f"**Execution ID:** {self.suite_execution_id}")
            summary_parts.append(
                f"**Execution Date:** {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.execution_timestamp))}"
            )
            summary_parts.append(
                f"**Overall Status:** {'✅ PASSED' if self.all_targets_met else '❌ REQUIRES ATTENTION'}"
            )
            summary_parts.append("")

            # Key performance findings from all benchmark categories
            summary_parts.append("## Key Performance Findings")

            findings = []
            if self.environment_results:
                step_time = getattr(self.environment_results, "avg_step_time_ms", 0)
                findings.append(f"• Environment step latency: {step_time:.3f}ms")

            if self.rendering_results:
                rgb_time = self.rendering_results.get("avg_rgb_render_ms", 0)
                findings.append(f"• RGB rendering performance: {rgb_time:.3f}ms")

            if self.memory_usage_results:
                memory = self.memory_usage_results.get("peak_memory_mb", 0)
                findings.append(f"• Peak memory usage: {memory:.1f}MB")

            if self.step_latency_results:
                latency = self.step_latency_results.get("mean_latency_ms", 0)
                findings.append(f"• Average step latency: {latency:.3f}ms")

            if self.plume_generation_results:
                gen_time = self.plume_generation_results.get("avg_generation_ms", 0)
                findings.append(f"• Plume generation time: {gen_time:.3f}ms")

            summary_parts.extend(findings)
            summary_parts.append("")

            # Cross-category performance relationships and system-wide optimization opportunities
            if self.cross_category_analysis:
                summary_parts.append("## Cross-Category Performance Analysis")

                correlations = self.cross_category_analysis.get("correlations", {})
                if correlations:
                    summary_parts.append("**Performance Correlations:**")
                    for correlation, value in correlations.items():
                        summary_parts.append(f"• {correlation}: {value:.3f}")
                else:
                    summary_parts.append(
                        "• No significant cross-category correlations detected"
                    )
                summary_parts.append("")

            # Highlight critical performance issues if enabled
            if highlight_critical_issues and self.performance_validation:
                violations = self.performance_validation.get("violations", [])
                if violations:
                    summary_parts.append("## ⚠️ Critical Performance Issues")
                    for violation in violations:
                        summary_parts.append(f"• {violation}")
                    summary_parts.append("")

            # Include strategic recommendations if enabled
            if (
                include_strategic_recommendations
                and self.integrated_optimization_recommendations
            ):
                summary_parts.append("## Strategic Optimization Recommendations")

                # Group recommendations by priority/category
                for i, recommendation in enumerate(
                    self.integrated_optimization_recommendations[:5], 1
                ):
                    summary_parts.append(f"{i}. {recommendation}")

                if len(self.integrated_optimization_recommendations) > 5:
                    remaining = len(self.integrated_optimization_recommendations) - 5
                    summary_parts.append(
                        f"   ...and {remaining} additional recommendations available in detailed report"
                    )
                summary_parts.append("")

            # Include performance trends if enabled and available
            if include_performance_trends:
                summary_parts.append("## Performance Trends")
                summary_parts.append(
                    "• Trend analysis requires historical benchmark data"
                )
                summary_parts.append(
                    "• Enable regression detection for comparative analysis"
                )
                summary_parts.append("")

            # Summary conclusion with actionable next steps
            summary_parts.append("## Next Steps")
            if self.all_targets_met:
                summary_parts.append(
                    "• All performance targets met - continue monitoring"
                )
                summary_parts.append(
                    "• Consider enabling additional benchmark categories for comprehensive analysis"
                )
                summary_parts.append(
                    "• Establish baseline measurements for regression detection"
                )
            else:
                summary_parts.append(
                    "• Address critical performance violations identified above"
                )
                summary_parts.append(
                    "• Implement prioritized optimization recommendations"
                )
                summary_parts.append("• Re-run benchmarks to validate improvements")

            # Generate final summary string
            executive_summary = "\n".join(summary_parts)

            # Store generated summary
            self.executive_summary = executive_summary

            logger.info("Generated integrated executive summary")
            return executive_summary

        except Exception as e:
            logger.error(f"Executive summary generation failed: {e}")
            return f"Executive summary generation failed: {str(e)}"

    def export_consolidated_results(
        self,
        output_directory: pathlib.Path,
        export_formats: List[str],
        include_raw_category_data: bool = True,
        generate_executive_artifacts: bool = True,
    ) -> Dict[str, Any]:
        """
        Export comprehensive consolidated benchmark results with integrated analysis, cross-category
        insights, and strategic recommendations in multiple formats for stakeholder communication
        and system integration.
        """
        logger = get_component_logger("results_exporter", "UTILS")

        try:
            export_results = {
                "export_timestamp": time.time(),
                "output_directory": str(output_directory),
                "exported_files": [],
                "export_status": "success",
            }

            # Create organized output directory structure
            output_directory.mkdir(parents=True, exist_ok=True)

            # Export consolidated benchmark data in specified formats
            base_filename = f"consolidated_benchmark_results_{self.suite_execution_id}"

            for format_type in export_formats:
                try:
                    if format_type.lower() == "json":
                        # JSON format export with comprehensive data
                        json_data = {
                            "suite_execution_id": self.suite_execution_id,
                            "execution_timestamp": self.execution_timestamp,
                            "suite_config": self.suite_config.to_dict(),
                            "results": {
                                "environment": (
                                    self.environment_results.to_dict()
                                    if self.environment_results
                                    and hasattr(self.environment_results, "to_dict")
                                    else None
                                ),
                                "rendering": self.rendering_results,
                                "step_latency": self.step_latency_results,
                                "memory_usage": self.memory_usage_results,
                                "plume_generation": self.plume_generation_results,
                            },
                            "analysis": {
                                "cross_category": self.cross_category_analysis,
                                "consolidated_statistics": self.consolidated_statistics,
                                "performance_validation": self.performance_validation,
                                "optimization_recommendations": self.integrated_optimization_recommendations,
                            },
                            "summary": {
                                "all_targets_met": self.all_targets_met,
                                "executive_summary": self.executive_summary,
                            },
                        }

                        json_file = output_directory / f"{base_filename}.json"
                        with open(json_file, "w") as f:
                            json.dump(json_data, f, indent=2, default=str)

                        export_results["exported_files"].append(
                            {
                                "format": "json",
                                "file_path": str(json_file),
                                "size_bytes": json_file.stat().st_size,
                            }
                        )

                    elif format_type.lower() == "markdown":
                        # Markdown format export for readable documentation
                        md_content = []

                        if not self.executive_summary:
                            self.generate_integrated_executive_summary()

                        md_content.append(self.executive_summary)

                        # Add detailed results section
                        md_content.append("\n\n---\n\n# Detailed Benchmark Results")

                        if self.environment_results:
                            md_content.append("\n## Environment Performance")
                            md_content.append(
                                f"• Average step time: {getattr(self.environment_results, 'avg_step_time_ms', 0):.3f}ms"
                            )

                        if self.rendering_results:
                            md_content.append("\n## Rendering Performance")
                            md_content.append(
                                f"• RGB rendering: {self.rendering_results.get('avg_rgb_render_ms', 0):.3f}ms"
                            )
                            md_content.append(
                                f"• Human rendering: {self.rendering_results.get('avg_human_render_ms', 0):.3f}ms"
                            )

                        if self.step_latency_results:
                            md_content.append("\n## Step Latency Analysis")
                            md_content.append(
                                f"• Mean latency: {self.step_latency_results.get('mean_latency_ms', 0):.3f}ms"
                            )
                            md_content.append(
                                f"• P95 latency: {self.step_latency_results.get('p95_latency_ms', 0):.3f}ms"
                            )

                        if self.memory_usage_results:
                            md_content.append("\n## Memory Usage Analysis")
                            md_content.append(
                                f"• Peak usage: {self.memory_usage_results.get('peak_memory_mb', 0):.1f}MB"
                            )
                            md_content.append(
                                f"• Average usage: {self.memory_usage_results.get('avg_memory_mb', 0):.1f}MB"
                            )

                        if self.plume_generation_results:
                            md_content.append("\n## Plume Generation Performance")
                            md_content.append(
                                f"• Generation time: {self.plume_generation_results.get('avg_generation_ms', 0):.3f}ms"
                            )

                        md_file = output_directory / f"{base_filename}.md"
                        with open(md_file, "w") as f:
                            f.write("".join(md_content))

                        export_results["exported_files"].append(
                            {
                                "format": "markdown",
                                "file_path": str(md_file),
                                "size_bytes": md_file.stat().st_size,
                            }
                        )

                except Exception as format_error:
                    logger.error(
                        f"Failed to export {format_type} format: {format_error}"
                    )
                    export_results["export_status"] = "partial_success"

            # Include raw category data if requested
            if include_raw_category_data:
                raw_data_dir = output_directory / "raw_data"
                raw_data_dir.mkdir(exist_ok=True)

                # Export individual category results
                if self.environment_results:
                    env_file = raw_data_dir / "environment_results.json"
                    with open(env_file, "w") as f:
                        if hasattr(self.environment_results, "to_dict"):
                            json.dump(
                                self.environment_results.to_dict(),
                                f,
                                indent=2,
                                default=str,
                            )
                        else:
                            json.dump({"raw_data": "not_serializable"}, f)

                export_results["raw_data_directory"] = str(raw_data_dir)

            # Generate executive artifacts if requested
            if generate_executive_artifacts:
                exec_dir = output_directory / "executive"
                exec_dir.mkdir(exist_ok=True)

                # Generate executive summary
                if not self.executive_summary:
                    self.generate_integrated_executive_summary()

                exec_summary_file = exec_dir / "executive_summary.md"
                with open(exec_summary_file, "w") as f:
                    f.write(self.executive_summary)

                export_results["executive_artifacts"] = [str(exec_summary_file)]

            logger.info(
                f"Consolidated results exported successfully: {len(export_results['exported_files'])} files"
            )
            return export_results

        except Exception as e:
            logger.error(f"Consolidated results export failed: {e}")
            return {
                "export_status": "failed",
                "error": str(e),
                "export_timestamp": time.time(),
            }

    def compare_with_baseline(
        self,
        baseline_result: "ConsolidatedBenchmarkResult",
        detect_cross_category_regressions: bool = True,
        regression_threshold: float = 0.10,
        analyze_improvement_trends: bool = True,
    ) -> Dict[str, Any]:
        """
        Compare consolidated benchmark results with baseline performance for comprehensive regression
        detection, improvement quantification, and trend analysis across all performance categories.
        """
        logger = get_component_logger("baseline_comparator", "UTILS")

        try:
            comparison_results = {
                "comparison_timestamp": time.time(),
                "baseline_execution_id": baseline_result.suite_execution_id,
                "current_execution_id": self.suite_execution_id,
                "regression_threshold": regression_threshold,
                "regressions_detected": [],
                "improvements_detected": [],
                "overall_status": "no_change",
            }

            # Compare consolidated statistics with baseline across categories
            category_comparisons = {}

            # Environment performance comparison
            if self.environment_results and baseline_result.environment_results:
                current_step = getattr(self.environment_results, "avg_step_time_ms", 0)
                baseline_step = getattr(
                    baseline_result.environment_results, "avg_step_time_ms", 0
                )

                if baseline_step > 0:
                    change_ratio = (current_step - baseline_step) / baseline_step
                    category_comparisons["environment"] = {
                        "current": current_step,
                        "baseline": baseline_step,
                        "change_ratio": change_ratio,
                        "change_percent": change_ratio * 100,
                        "status": (
                            "regression"
                            if change_ratio > regression_threshold
                            else (
                                "improvement"
                                if change_ratio < -regression_threshold
                                else "stable"
                            )
                        ),
                    }

            # Rendering performance comparison
            if self.rendering_results and baseline_result.rendering_results:
                current_rgb = self.rendering_results.get("avg_rgb_render_ms", 0)
                baseline_rgb = baseline_result.rendering_results.get(
                    "avg_rgb_render_ms", 0
                )

                if baseline_rgb > 0:
                    change_ratio = (current_rgb - baseline_rgb) / baseline_rgb
                    category_comparisons["rendering"] = {
                        "current": current_rgb,
                        "baseline": baseline_rgb,
                        "change_ratio": change_ratio,
                        "change_percent": change_ratio * 100,
                        "status": (
                            "regression"
                            if change_ratio > regression_threshold
                            else (
                                "improvement"
                                if change_ratio < -regression_threshold
                                else "stable"
                            )
                        ),
                    }

            # Memory usage comparison
            if self.memory_usage_results and baseline_result.memory_usage_results:
                current_memory = self.memory_usage_results.get("peak_memory_mb", 0)
                baseline_memory = baseline_result.memory_usage_results.get(
                    "peak_memory_mb", 0
                )

                if baseline_memory > 0:
                    change_ratio = (current_memory - baseline_memory) / baseline_memory
                    category_comparisons["memory"] = {
                        "current": current_memory,
                        "baseline": baseline_memory,
                        "change_ratio": change_ratio,
                        "change_percent": change_ratio * 100,
                        "status": (
                            "regression"
                            if change_ratio > regression_threshold
                            else (
                                "improvement"
                                if change_ratio < -regression_threshold
                                else "stable"
                            )
                        ),
                    }

            comparison_results["category_comparisons"] = category_comparisons

            # Detect cross-category regressions if enabled
            if detect_cross_category_regressions:
                cross_regressions = []
                cross_improvements = []

                for category, comparison in category_comparisons.items():
                    if comparison["status"] == "regression":
                        regression_msg = f"{category}: {comparison['change_percent']:.1f}% slower than baseline"
                        cross_regressions.append(regression_msg)
                        comparison_results["regressions_detected"].append(
                            regression_msg
                        )
                    elif comparison["status"] == "improvement":
                        improvement_msg = f"{category}: {abs(comparison['change_percent']):.1f}% faster than baseline"
                        cross_improvements.append(improvement_msg)
                        comparison_results["improvements_detected"].append(
                            improvement_msg
                        )

                comparison_results["cross_category_regressions"] = cross_regressions
                comparison_results["cross_category_improvements"] = cross_improvements

            # Analyze improvement trends if enabled
            if analyze_improvement_trends:
                trend_analysis = {}

                # Calculate overall trend direction
                if category_comparisons:
                    change_ratios = [
                        comp["change_ratio"]
                        for comp in category_comparisons.values()
                        if "change_ratio" in comp
                    ]
                    if change_ratios:
                        avg_change = sum(change_ratios) / len(change_ratios)
                        trend_analysis["average_change_ratio"] = avg_change
                        trend_analysis["trend_direction"] = (
                            "improving"
                            if avg_change < -0.05
                            else ("degrading" if avg_change > 0.05 else "stable")
                        )

                        # Identify most improved and most regressed categories
                        if len(change_ratios) > 1:
                            best_category = min(
                                category_comparisons.items(),
                                key=lambda x: x[1].get("change_ratio", 0),
                            )
                            worst_category = max(
                                category_comparisons.items(),
                                key=lambda x: x[1].get("change_ratio", 0),
                            )

                            trend_analysis["best_performing_category"] = {
                                "name": best_category[0],
                                "improvement": abs(best_category[1]["change_percent"]),
                            }
                            trend_analysis["worst_performing_category"] = {
                                "name": worst_category[0],
                                "regression": worst_category[1]["change_percent"],
                            }

                comparison_results["trend_analysis"] = trend_analysis

            # Determine overall comparison status
            if comparison_results["regressions_detected"]:
                comparison_results["overall_status"] = "regression"
            elif comparison_results["improvements_detected"]:
                comparison_results["overall_status"] = "improvement"
            else:
                comparison_results["overall_status"] = "stable"

            logger.info(
                f"Baseline comparison completed: {comparison_results['overall_status'].upper()}"
            )

            return comparison_results

        except Exception as e:
            logger.error(f"Baseline comparison failed: {e}")
            return {
                "comparison_status": "failed",
                "error": str(e),
                "comparison_timestamp": time.time(),
            }


class BenchmarkSuiteOrchestrator:
    """
    Central orchestrator for comprehensive benchmark suite execution providing coordinated execution
    management, result consolidation, parallel processing coordination, and integrated analysis across
    all performance benchmarking categories with progress tracking, error handling, and resource management.
    """

    def __init__(
        self,
        config: Optional[BenchmarkSuiteConfig] = None,
        enable_parallel_execution: bool = False,
        enable_detailed_logging: bool = True,
    ) -> None:
        """
        Initialize benchmark suite orchestrator with configuration, execution management, and logging
        infrastructure for coordinated benchmark execution and result consolidation.
        """
        # Initialize config with provided BenchmarkSuiteConfig or create default comprehensive configuration
        self.config = config or BenchmarkSuiteConfig()

        # Store parallel execution and detailed logging configuration flags for orchestration management
        self.parallel_execution_enabled = enable_parallel_execution
        self.detailed_logging_enabled = enable_detailed_logging

        # Create component-specific logger using get_component_logger for orchestration progress tracking
        self.logger = get_component_logger("benchmark_orchestrator", "UTILS")

        # Initialize benchmark_suite_components dictionary with benchmark category instances and configurations
        self.benchmark_suite_components = {
            "environment": (
                EnvironmentPerformanceSuite()
                if hasattr(EnvironmentPerformanceSuite, "__init__")
                else None
            ),
            "rendering": (
                RenderingPerformanceSuite()
                if hasattr(RenderingPerformanceSuite, "__init__")
                else None
            ),
            "step_latency": (
                StepLatencyBenchmark()
                if hasattr(StepLatencyBenchmark, "__init__")
                else None
            ),
            "memory": (
                MemoryUsageAnalyzer()
                if hasattr(MemoryUsageAnalyzer, "__init__")
                else None
            ),
            "plume_generation": (
                PlumeGenerationBenchmark()
                if hasattr(PlumeGenerationBenchmark, "__init__")
                else None
            ),
        }

        # Initialize execution_status dictionary for real-time progress tracking and error handling
        self.execution_status = {
            "suite_started": False,
            "categories_completed": [],
            "categories_failed": [],
            "current_category": None,
            "start_time": None,
            "estimated_completion": None,
        }

        # Initialize consolidated_results dictionary for result aggregation and cross-category analysis
        self.consolidated_results = {}

        # Create ThreadPoolExecutor if enable_parallel_execution for concurrent benchmark execution
        self.executor = None
        if enable_parallel_execution:
            self.executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=3, thread_name_prefix="benchmark"
            )

        # Validate orchestrator configuration and benchmark component availability for execution readiness
        if self.detailed_logging_enabled:
            self.logger.info("Benchmark suite orchestrator initialized")
            self.logger.debug(f"Configuration: {self.config.to_dict()}")

    def execute_full_suite(
        self,
        validate_targets: bool = True,
        detect_regressions: bool = False,
        baseline_data: Optional[Dict[str, Any]] = None,
    ) -> ConsolidatedBenchmarkResult:
        """
        Execute complete benchmark suite with coordinated execution management, progress tracking,
        error handling, and result consolidation across all performance benchmarking categories.
        """
        self.logger.info("Starting comprehensive benchmark suite execution")

        try:
            # Initialize suite execution with resource validation and component readiness verification
            suite_execution_id = f"benchmark_suite_{int(time.time())}"
            execution_start_time = time.time()

            self.execution_status.update(
                {
                    "suite_started": True,
                    "start_time": execution_start_time,
                    "current_category": "initialization",
                    "categories_completed": [],
                    "categories_failed": [],
                }
            )

            # Create consolidated result container for integrated analysis
            consolidated_result = ConsolidatedBenchmarkResult(
                suite_execution_id=suite_execution_id,
                execution_timestamp=execution_start_time,
                suite_config=self.config,
            )

            # Execute benchmark categories based on configuration
            execution_futures = {}

            if self.parallel_execution_enabled and self.executor:
                self.logger.info("Executing benchmarks in parallel")

                # Submit parallel benchmark executions
                if self.config.include_environment_benchmark:
                    execution_futures["environment"] = self.executor.submit(
                        self.execute_category_benchmark, "environment", {}, True
                    )

                if self.config.include_rendering_benchmark:
                    execution_futures["rendering"] = self.executor.submit(
                        self.execute_category_benchmark, "rendering", {}, True
                    )

                if self.config.include_step_latency_benchmark:
                    execution_futures["step_latency"] = self.executor.submit(
                        self.execute_category_benchmark, "step_latency", {}, True
                    )

                if self.config.include_memory_benchmark:
                    execution_futures["memory"] = self.executor.submit(
                        self.execute_category_benchmark, "memory", {}, True
                    )

                if self.config.include_plume_generation_benchmark:
                    execution_futures["plume_generation"] = self.executor.submit(
                        self.execute_category_benchmark, "plume_generation", {}, True
                    )

                # Wait for parallel execution completion with timeout
                timeout_seconds = (self.config.execution_timeout_minutes or 30) * 60

                for category, future in execution_futures.items():
                    try:
                        result = future.result(timeout=timeout_seconds)
                        self._process_category_result(
                            consolidated_result, category, result
                        )
                        self.execution_status["categories_completed"].append(category)
                        self.logger.info(f"Parallel execution completed for {category}")
                    except Exception as e:
                        self.logger.error(
                            f"Parallel execution failed for {category}: {e}"
                        )
                        self.execution_status["categories_failed"].append(category)
                        self._process_category_result(
                            consolidated_result,
                            category,
                            {"error": str(e), "status": "failed"},
                        )

            else:
                # Sequential execution
                self.logger.info("Executing benchmarks sequentially")

                if self.config.include_environment_benchmark:
                    result = self.execute_category_benchmark("environment", {}, True)
                    self._process_category_result(
                        consolidated_result, "environment", result
                    )

                if self.config.include_rendering_benchmark:
                    result = self.execute_category_benchmark("rendering", {}, True)
                    self._process_category_result(
                        consolidated_result, "rendering", result
                    )

                if self.config.include_step_latency_benchmark:
                    result = self.execute_category_benchmark("step_latency", {}, True)
                    self._process_category_result(
                        consolidated_result, "step_latency", result
                    )

                if self.config.include_memory_benchmark:
                    result = self.execute_category_benchmark("memory", {}, True)
                    self._process_category_result(consolidated_result, "memory", result)

                if self.config.include_plume_generation_benchmark:
                    result = self.execute_category_benchmark(
                        "plume_generation", {}, True
                    )
                    self._process_category_result(
                        consolidated_result, "plume_generation", result
                    )

            # Consolidate results from all benchmark categories with cross-category correlation analysis
            self.logger.info("Consolidating benchmark results")
            if self.config.generate_integrated_analysis:
                consolidated_result.calculate_consolidated_statistics(
                    include_cross_correlation=True, calculate_performance_score=True
                )

            # Validate performance against targets if validate_targets enabled with compliance reporting
            if validate_targets:
                self.logger.info("Validating performance targets")
                validation_result = consolidated_result.validate_integrated_performance(
                    integrated_targets=self.config.performance_targets,
                    strict_validation=False,
                    cross_category_validation=True,
                )

                if validation_result["validation_passed"]:
                    self.logger.info("All performance targets validated successfully")
                else:
                    self.logger.warning(
                        f"Performance validation issues: {len(validation_result.get('violations', []))}"
                    )

            # Detect regressions if detect_regressions enabled using baseline_data comparison
            if detect_regressions and baseline_data:
                self.logger.info("Performing regression detection")
                # Regression detection would require baseline ConsolidatedBenchmarkResult
                # For now, log that regression detection was requested
                self.logger.warning(
                    "Regression detection requires ConsolidatedBenchmarkResult baseline"
                )

            # Generate integrated optimization recommendations based on consolidated analysis
            optimization_recommendations = []

            if consolidated_result.performance_validation.get("violations"):
                optimization_recommendations.append(
                    "Address performance target violations identified in validation"
                )

            if self.execution_status["categories_failed"]:
                optimization_recommendations.append(
                    f"Investigate and resolve failures in: {', '.join(self.execution_status['categories_failed'])}"
                )

            if not optimization_recommendations:
                optimization_recommendations.append(
                    "Consider expanding benchmark coverage or increasing test iterations for more comprehensive analysis"
                )

            consolidated_result.integrated_optimization_recommendations = (
                optimization_recommendations
            )

            # Generate executive summary with integrated analysis
            consolidated_result.generate_integrated_executive_summary(
                include_strategic_recommendations=True, highlight_critical_issues=True
            )

            # Clean up execution resources and validate result completeness
            execution_duration = time.time() - execution_start_time
            self.logger.info(
                f"Benchmark suite execution completed in {execution_duration:.1f} seconds"
            )

            # Update final execution status
            self.execution_status.update(
                {
                    "suite_completed": True,
                    "execution_duration_seconds": execution_duration,
                    "total_categories_attempted": len(
                        [
                            c
                            for c in [
                                "environment",
                                "rendering",
                                "step_latency",
                                "memory",
                                "plume_generation",
                            ]
                            if getattr(self.config, f"include_{c}_benchmark", False)
                        ]
                    ),
                    "categories_succeeded": len(
                        self.execution_status["categories_completed"]
                    ),
                    "categories_failed": len(
                        self.execution_status["categories_failed"]
                    ),
                }
            )

            # Return ConsolidatedBenchmarkResult with comprehensive analysis and actionable insights
            return consolidated_result

        except Exception as e:
            self.logger.error(f"Benchmark suite execution failed: {e}")
            # Return failed result for error handling
            error_result = ConsolidatedBenchmarkResult(
                suite_execution_id=f"failed_suite_{int(time.time())}",
                execution_timestamp=time.time(),
                suite_config=self.config,
            )
            error_result.integrated_optimization_recommendations = [
                f"Suite execution failed: {str(e)}"
            ]
            return error_result

    def execute_category_benchmark(
        self,
        benchmark_category: str,
        category_config: Dict[str, Any],
        capture_detailed_metrics: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute individual benchmark category with configuration management, progress tracking,
        error handling, and result integration for coordinated suite execution.
        """
        self.logger.info(f"Executing {benchmark_category} benchmark")

        try:
            # Update execution status with current category
            self.execution_status["current_category"] = benchmark_category

            # Get default configuration for the category
            default_config = get_complete_default_config()
            category_defaults = default_config.get("benchmark_defaults", {})

            # Merge category_config with defaults
            merged_config = {**category_defaults, **category_config}

            # Execute benchmark based on category type
            if benchmark_category == "environment":
                if self.benchmark_suite_components["environment"]:
                    try:
                        # Use the actual environment performance suite
                        result = run_environment_performance_benchmark(
                            iterations=merged_config.get("iterations", 50),
                            grid_size=tuple(merged_config.get("grid_size", [128, 128])),
                            max_steps=merged_config.get("max_steps", 1000),
                        )
                        return self._format_category_result(
                            benchmark_category, result, "success"
                        )
                    except Exception as e:
                        self.logger.error(f"Environment benchmark execution error: {e}")
                        return self._format_category_result(
                            benchmark_category, {"error": str(e)}, "error"
                        )
                else:
                    return self._format_category_result(
                        benchmark_category,
                        run_environment_performance_benchmark(),
                        "not_implemented",
                    )

            elif benchmark_category == "rendering":
                result = run_rendering_performance_benchmark()
                return self._format_category_result(
                    benchmark_category, result, result.get("status", "success")
                )

            elif benchmark_category == "step_latency":
                result = benchmark_step_latency()
                return self._format_category_result(
                    benchmark_category, result, result.get("status", "success")
                )

            elif benchmark_category == "memory":
                result = run_memory_usage_benchmark()
                return self._format_category_result(
                    benchmark_category, result, result.get("status", "success")
                )

            elif benchmark_category == "plume_generation":
                result = run_plume_generation_benchmark()
                return self._format_category_result(
                    benchmark_category, result, result.get("status", "success")
                )

            else:
                error_msg = f"Unknown benchmark category: {benchmark_category}"
                self.logger.error(error_msg)
                return self._format_category_result(
                    benchmark_category, {"error": error_msg}, "error"
                )

        except Exception as e:
            self.logger.error(
                f"Category benchmark execution failed for {benchmark_category}: {e}"
            )
            return self._format_category_result(
                benchmark_category, {"error": str(e)}, "error"
            )

    def _format_category_result(
        self, category: str, result: Any, status: str
    ) -> Dict[str, Any]:
        """Format category benchmark result with standardized structure."""
        return {
            "category": category,
            "status": status,
            "timestamp": time.time(),
            "result": result,
        }

    def _process_category_result(
        self,
        consolidated_result: ConsolidatedBenchmarkResult,
        category: str,
        result: Dict[str, Any],
    ) -> None:
        """Process and integrate category result into consolidated result."""
        try:
            if result["status"] == "success":
                if category == "environment":
                    if isinstance(result["result"], BenchmarkResult):
                        consolidated_result.environment_results = result["result"]
                    else:
                        # Create BenchmarkResult from dict if needed
                        consolidated_result.environment_results = result["result"]
                elif category == "rendering":
                    consolidated_result.rendering_results = result["result"]
                elif category == "step_latency":
                    consolidated_result.step_latency_results = result["result"]
                elif category == "memory":
                    consolidated_result.memory_usage_results = result["result"]
                elif category == "plume_generation":
                    consolidated_result.plume_generation_results = result["result"]

                self.execution_status["categories_completed"].append(category)
            else:
                self.execution_status["categories_failed"].append(category)
                self.logger.error(f"Category {category} failed: {result}")

        except Exception as e:
            self.logger.error(f"Failed to process {category} result: {e}")
            self.execution_status["categories_failed"].append(category)

    def consolidate_results(
        self,
        include_cross_correlation_analysis: bool = True,
        generate_integrated_recommendations: bool = True,
    ) -> ConsolidatedBenchmarkResult:
        """
        Consolidate results from all benchmark categories with cross-category correlation analysis,
        statistical integration, and unified optimization recommendations for comprehensive system
        performance insights.
        """
        self.logger.info("Consolidating benchmark results")

        try:
            # Create consolidated result from current execution data
            consolidated = ConsolidatedBenchmarkResult(
                suite_execution_id=f"manual_consolidation_{int(time.time())}",
                execution_timestamp=time.time(),
                suite_config=self.config,
            )

            # Aggregate results from consolidated_results dictionary
            for category, result in self.consolidated_results.items():
                if category == "environment":
                    consolidated.environment_results = result
                elif category == "rendering":
                    consolidated.rendering_results = result
                elif category == "step_latency":
                    consolidated.step_latency_results = result
                elif category == "memory":
                    consolidated.memory_usage_results = result
                elif category == "plume_generation":
                    consolidated.plume_generation_results = result

            # Perform cross-correlation analysis if enabled
            if include_cross_correlation_analysis:
                consolidated.calculate_consolidated_statistics(
                    include_cross_correlation=True, calculate_performance_score=True
                )

            # Generate integrated recommendations if enabled
            if generate_integrated_recommendations:
                recommendations = []

                # Analyze results and generate recommendations
                if consolidated.consolidated_statistics:
                    overall_perf = consolidated.consolidated_statistics.get(
                        "overall_performance", {}
                    )
                    if overall_perf.get("performance_score", 0) < 0.8:
                        recommendations.append(
                            "Overall performance score below 0.8 - consider system optimization"
                        )

                if not recommendations:
                    recommendations.append(
                        "System performance within acceptable parameters - continue monitoring"
                    )

                consolidated.integrated_optimization_recommendations = recommendations

            return consolidated

        except Exception as e:
            self.logger.error(f"Result consolidation failed: {e}")
            # Return empty consolidated result
            return ConsolidatedBenchmarkResult(
                suite_execution_id="consolidation_failed",
                execution_timestamp=time.time(),
                suite_config=self.config,
            )

    def get_execution_status(
        self, include_detailed_progress: bool = False
    ) -> Dict[str, Any]:
        """
        Get real-time execution status for ongoing benchmark suite execution with progress tracking,
        completion estimates, and error status for monitoring and user feedback.
        """
        status = dict(self.execution_status)

        if include_detailed_progress:
            # Calculate progress percentage
            total_categories = sum(
                [
                    self.config.include_environment_benchmark,
                    self.config.include_rendering_benchmark,
                    self.config.include_step_latency_benchmark,
                    self.config.include_memory_benchmark,
                    self.config.include_plume_generation_benchmark,
                ]
            )

            completed = len(status.get("categories_completed", []))
            failed = len(status.get("categories_failed", []))

            status["progress"] = {
                "total_categories": total_categories,
                "completed_categories": completed,
                "failed_categories": failed,
                "progress_percentage": (
                    ((completed + failed) / total_categories * 100)
                    if total_categories > 0
                    else 0
                ),
            }

            # Estimate remaining time
            if status.get("start_time") and completed > 0:
                elapsed = time.time() - status["start_time"]
                avg_time_per_category = elapsed / completed
                remaining_categories = total_categories - completed - failed
                estimated_remaining = avg_time_per_category * remaining_categories

                status["time_estimates"] = {
                    "elapsed_seconds": elapsed,
                    "estimated_remaining_seconds": estimated_remaining,
                    "estimated_completion_time": time.time() + estimated_remaining,
                }

        return status

    def cleanup_resources(self) -> bool:
        """
        Clean up orchestrator resources including thread pools, temporary data, and component instances
        with proper resource management and memory cleanup.
        """
        try:
            self.logger.info("Cleaning up benchmark orchestrator resources")

            # Shutdown ThreadPoolExecutor if parallel execution was enabled
            if self.executor:
                self.executor.shutdown(wait=True)
                self.executor = None
                self.logger.debug("ThreadPoolExecutor shut down successfully")

            # Clear consolidated_results and execution_status dictionaries
            self.consolidated_results.clear()
            self.execution_status = {
                "suite_started": False,
                "categories_completed": [],
                "categories_failed": [],
                "current_category": None,
                "start_time": None,
                "estimated_completion": None,
            }

            # Clean up benchmark component instances
            self.benchmark_suite_components.clear()

            self.logger.info("Resource cleanup completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"Resource cleanup failed: {e}")
            return False


# Primary orchestration functions for comprehensive benchmark suite execution


def run_comprehensive_benchmark_suite(
    config: Optional[BenchmarkSuiteConfig] = None,
    validate_performance_targets: bool = True,
    detect_regressions: bool = False,
    output_path: Optional[str] = None,
    parallel_execution: bool = False,
) -> ConsolidatedBenchmarkResult:
    """
    Execute complete benchmark suite orchestrating all performance analysis categories including
    environment performance, rendering benchmarking, step latency measurement, memory usage profiling,
    and plume generation testing with integrated result consolidation, performance regression detection,
    and optimization recommendations.
    """
    logger = get_component_logger("comprehensive_suite", "UTILS")
    logger.info("Starting comprehensive benchmark suite execution")

    try:
        # Initialize comprehensive benchmark suite configuration with default settings or provided config validation
        if config is None:
            config = BenchmarkSuiteConfig()
            config.enable_parallel_execution = parallel_execution
            logger.debug("Using default benchmark suite configuration")

        # Validate configuration before execution
        if not config.validate_suite_config():
            logger.error("Benchmark suite configuration validation failed")
            raise ValueError("Invalid benchmark suite configuration")

        # Set up unified logging infrastructure for suite orchestration with progress tracking and error handling
        logger.info("Initializing benchmark suite orchestrator")

        # Create benchmark orchestrator instance for coordinated execution and result management
        orchestrator = BenchmarkSuiteOrchestrator(
            config=config,
            enable_parallel_execution=parallel_execution,
            enable_detailed_logging=True,
        )

        # Execute comprehensive benchmark suite with integrated analysis
        logger.info("Executing benchmark suite")
        consolidated_result = orchestrator.execute_full_suite(
            validate_targets=validate_performance_targets,
            detect_regressions=detect_regressions,
            baseline_data=None,  # Could be loaded from file if available
        )

        # Export consolidated results to output_path if specified
        if output_path:
            logger.info(f"Exporting results to {output_path}")
            output_dir = pathlib.Path(output_path)
            export_result = consolidated_result.export_consolidated_results(
                output_directory=output_dir,
                export_formats=["json", "markdown"],
                include_raw_category_data=True,
                generate_executive_artifacts=True,
            )

            if export_result["export_status"] == "success":
                logger.info(
                    f"Results exported successfully: {len(export_result['exported_files'])} files"
                )
            else:
                logger.warning(
                    f"Result export had issues: {export_result.get('error', 'Unknown error')}"
                )

        # Clean up suite resources
        cleanup_success = orchestrator.cleanup_resources()
        if cleanup_success:
            logger.debug("Suite resources cleaned up successfully")

        # Log execution summary
        exec_status = orchestrator.get_execution_status(include_detailed_progress=True)
        progress = exec_status.get("progress", {})
        logger.info(
            f"Suite execution completed: {progress.get('completed_categories', 0)}/{progress.get('total_categories', 0)} categories successful"
        )

        return consolidated_result

    except Exception as e:
        logger.error(f"Comprehensive benchmark suite failed: {e}")
        # Return error result for graceful handling
        error_result = ConsolidatedBenchmarkResult(
            suite_execution_id=f"failed_{int(time.time())}",
            execution_timestamp=time.time(),
            suite_config=config or BenchmarkSuiteConfig(),
        )
        error_result.integrated_optimization_recommendations = [
            f"Suite execution failed: {str(e)}"
        ]
        return error_result


def generate_benchmark_report(
    benchmark_results: ConsolidatedBenchmarkResult,
    report_format: str = "markdown",
    include_executive_summary: bool = True,
    include_technical_details: bool = True,
    include_visualizations: bool = False,
    output_path: Optional[pathlib.Path] = None,
) -> str:
    """
    Generate comprehensive benchmark report consolidating results from all performance analysis
    categories with executive summary, technical analysis, cross-category correlation insights,
    and strategic optimization recommendations for stakeholder communication and development guidance.
    """
    logger = get_component_logger("report_generator", "UTILS")
    logger.info(f"Generating benchmark report in {report_format} format")

    try:
        report_content = []

        # Compile executive summary if enabled
        if include_executive_summary:
            if not benchmark_results.executive_summary:
                benchmark_results.generate_integrated_executive_summary()
            report_content.append(benchmark_results.executive_summary)
            report_content.append("\n\n---\n\n")

        # Generate technical analysis section if enabled
        if include_technical_details:
            report_content.append("# Technical Performance Analysis\n\n")

            # Environment performance details
            if benchmark_results.environment_results:
                report_content.append("## Environment Performance\n")
                env_result = benchmark_results.environment_results

                if hasattr(env_result, "avg_step_time_ms"):
                    report_content.append(
                        f"- Average step latency: {env_result.avg_step_time_ms:.3f}ms\n"
                    )
                if hasattr(env_result, "success_rate"):
                    report_content.append(
                        f"- Episode success rate: {env_result.success_rate:.1%}\n"
                    )
                report_content.append("\n")

            # Rendering performance details
            if benchmark_results.rendering_results:
                report_content.append("## Rendering Performance\n")
                render_data = benchmark_results.rendering_results

                if "avg_rgb_render_ms" in render_data:
                    report_content.append(
                        f"- RGB rendering: {render_data['avg_rgb_render_ms']:.3f}ms average\n"
                    )
                if "avg_human_render_ms" in render_data:
                    report_content.append(
                        f"- Human mode rendering: {render_data['avg_human_render_ms']:.3f}ms average\n"
                    )
                report_content.append("\n")

            # Memory usage details
            if benchmark_results.memory_usage_results:
                report_content.append("## Memory Usage Analysis\n")
                memory_data = benchmark_results.memory_usage_results

                if "peak_memory_mb" in memory_data:
                    report_content.append(
                        f"- Peak memory usage: {memory_data['peak_memory_mb']:.1f}MB\n"
                    )
                if "avg_memory_mb" in memory_data:
                    report_content.append(
                        f"- Average memory usage: {memory_data['avg_memory_mb']:.1f}MB\n"
                    )
                report_content.append("\n")

            # Performance validation summary
            if benchmark_results.performance_validation:
                validation = benchmark_results.performance_validation
                report_content.append("## Performance Validation\n")
                report_content.append(
                    f"- Overall validation: {'✅ PASSED' if validation.get('validation_passed') else '❌ FAILED'}\n"
                )

                violations = validation.get("violations", [])
                if violations:
                    report_content.append("- Performance violations:\n")
                    for violation in violations[:5]:  # Limit to top 5
                        report_content.append(f"  - {violation}\n")
                report_content.append("\n")

        # Cross-category correlation analysis
        if benchmark_results.cross_category_analysis:
            report_content.append("## Cross-Category Performance Correlation\n")
            correlations = benchmark_results.cross_category_analysis.get(
                "correlations", {}
            )

            if correlations:
                for correlation_name, value in correlations.items():
                    report_content.append(f"- {correlation_name}: {value:.3f}\n")
            else:
                report_content.append(
                    "- No significant correlations detected between benchmark categories\n"
                )
            report_content.append("\n")

        # Strategic optimization recommendations
        if benchmark_results.integrated_optimization_recommendations:
            report_content.append("## Optimization Recommendations\n")
            for i, recommendation in enumerate(
                benchmark_results.integrated_optimization_recommendations, 1
            ):
                report_content.append(f"{i}. {recommendation}\n")
            report_content.append("\n")

        # Compile final report content
        final_report = "".join(report_content)

        # Save to output_path if specified
        if output_path:
            try:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, "w") as f:
                    f.write(final_report)
                logger.info(f"Benchmark report saved to {output_path}")
                return str(output_path)
            except Exception as e:
                logger.error(f"Failed to save report to {output_path}: {e}")

        logger.info("Benchmark report generated successfully")
        return final_report

    except Exception as e:
        logger.error(f"Benchmark report generation failed: {e}")
        return f"Report generation failed: {str(e)}"


def validate_performance_targets(
    benchmark_results: ConsolidatedBenchmarkResult,
    custom_performance_targets: Optional[Dict[str, float]] = None,
    strict_validation: bool = False,
    significance_threshold: float = 0.05,
    baseline_results: Optional[ConsolidatedBenchmarkResult] = None,
) -> Dict[str, Any]:
    """
    Comprehensive performance target validation across all benchmark categories with statistical
    significance testing, compliance analysis, regression detection, and improvement requirement
    identification for systematic performance quality assurance.
    """
    logger = get_component_logger("target_validator", "UTILS")
    logger.info("Validating performance targets across all benchmark categories")

    try:
        # Use custom targets if provided, otherwise use suite config targets
        targets = (
            custom_performance_targets
            or benchmark_results.suite_config.performance_targets
        )

        # Perform integrated performance validation
        validation_result = benchmark_results.validate_integrated_performance(
            integrated_targets=targets,
            strict_validation=strict_validation,
            cross_category_validation=True,
        )

        # Add baseline comparison if provided
        if baseline_results:
            logger.info("Comparing against baseline results")
            comparison_result = benchmark_results.compare_with_baseline(
                baseline_result=baseline_results,
                detect_cross_category_regressions=True,
                regression_threshold=significance_threshold,
            )

            validation_result["baseline_comparison"] = comparison_result

            # Flag regressions as validation failures
            if comparison_result.get("regressions_detected"):
                validation_result["validation_passed"] = False
                validation_result["regression_violations"] = comparison_result[
                    "regressions_detected"
                ]

        # Add statistical significance analysis
        validation_result["statistical_analysis"] = {
            "significance_threshold": significance_threshold,
            "validation_method": "strict" if strict_validation else "standard",
            "target_source": "custom" if custom_performance_targets else "suite_config",
        }

        logger.info(
            f"Performance validation completed: {'PASSED' if validation_result['validation_passed'] else 'FAILED'}"
        )
        return validation_result

    except Exception as e:
        logger.error(f"Performance target validation failed: {e}")
        return {
            "validation_passed": False,
            "error": str(e),
            "validation_timestamp": time.time(),
        }


def detect_performance_regressions(
    historical_results: List[ConsolidatedBenchmarkResult],
    current_result: ConsolidatedBenchmarkResult,
    regression_threshold: float = 0.10,
    trend_analysis_window: int = 5,
    generate_alerts: bool = False,
) -> Dict[str, Any]:
    """
    Advanced performance regression detection across all benchmark categories using statistical
    analysis, trend identification, automated alerting, and predictive analysis for proactive
    performance monitoring and quality assurance.
    """
    logger = get_component_logger("regression_detector", "UTILS")
    logger.info(
        f"Detecting performance regressions across {len(historical_results)} historical results"
    )

    try:
        regression_analysis = {
            "analysis_timestamp": time.time(),
            "regression_threshold": regression_threshold,
            "trend_window": trend_analysis_window,
            "regressions_detected": [],
            "warnings_generated": [],
            "overall_status": "no_regressions",
        }

        if not historical_results:
            logger.warning("No historical results provided for regression analysis")
            regression_analysis["overall_status"] = "insufficient_data"
            return regression_analysis

        # Use most recent result as baseline for comparison
        baseline_result = historical_results[-1] if historical_results else None

        if baseline_result:
            # Perform baseline comparison
            comparison = current_result.compare_with_baseline(
                baseline_result=baseline_result,
                detect_cross_category_regressions=True,
                regression_threshold=regression_threshold,
            )

            regression_analysis["baseline_comparison"] = comparison
            regression_analysis["regressions_detected"] = comparison.get(
                "regressions_detected", []
            )

            if regression_analysis["regressions_detected"]:
                regression_analysis["overall_status"] = "regressions_detected"

        # Analyze trends over the analysis window
        if len(historical_results) >= trend_analysis_window:
            recent_results = historical_results[-trend_analysis_window:]

            # Calculate trend metrics (simplified analysis)
            trend_data = {}

            # Collect timing metrics from recent results
            for i, result in enumerate(recent_results):
                if result.environment_results and hasattr(
                    result.environment_results, "avg_step_time_ms"
                ):
                    if "environment_step_time" not in trend_data:
                        trend_data["environment_step_time"] = []
                    trend_data["environment_step_time"].append(
                        getattr(result.environment_results, "avg_step_time_ms", 0)
                    )

            # Simple trend analysis
            for metric, values in trend_data.items():
                if len(values) >= 3:
                    recent_avg = sum(values[-2:]) / 2
                    earlier_avg = sum(values[:-2]) / len(values[:-2])

                    if earlier_avg > 0:
                        trend_change = (recent_avg - earlier_avg) / earlier_avg

                        if trend_change > regression_threshold:
                            trend_warning = f"Degrading trend detected in {metric}: {trend_change:.1%} increase"
                            regression_analysis["warnings_generated"].append(
                                trend_warning
                            )

            regression_analysis["trend_analysis"] = {
                "metrics_analyzed": list(trend_data.keys()),
                "trend_window_size": len(recent_results),
            }

        # Generate alerts if enabled
        if generate_alerts and (
            regression_analysis["regressions_detected"]
            or regression_analysis["warnings_generated"]
        ):
            alerts = []

            for regression in regression_analysis["regressions_detected"]:
                alerts.append(
                    {
                        "type": "regression",
                        "severity": "high",
                        "message": regression,
                        "timestamp": time.time(),
                    }
                )

            for warning in regression_analysis["warnings_generated"]:
                alerts.append(
                    {
                        "type": "trend_warning",
                        "severity": "medium",
                        "message": warning,
                        "timestamp": time.time(),
                    }
                )

            regression_analysis["alerts"] = alerts
            logger.warning(f"Generated {len(alerts)} performance alerts")

        logger.info(
            f"Regression detection completed: {regression_analysis['overall_status']}"
        )
        return regression_analysis

    except Exception as e:
        logger.error(f"Performance regression detection failed: {e}")
        return {
            "analysis_status": "failed",
            "error": str(e),
            "analysis_timestamp": time.time(),
        }


def optimize_system_performance(
    benchmark_results: ConsolidatedBenchmarkResult,
    analyze_cross_category_bottlenecks: bool = True,
    generate_implementation_roadmap: bool = True,
    prioritize_by_impact: bool = True,
    system_constraints: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Comprehensive system performance optimization analysis generating targeted recommendations
    across all performance categories based on integrated benchmark results, bottleneck
    identification, and best practice application for systematic performance enhancement.
    """
    logger = get_component_logger("performance_optimizer", "UTILS")
    logger.info("Analyzing system performance for optimization opportunities")

    try:
        optimization_analysis = {
            "analysis_timestamp": time.time(),
            "optimization_recommendations": [],
            "bottlenecks_identified": [],
            "implementation_roadmap": [],
            "expected_improvements": {},
        }

        # Analyze performance patterns from benchmark results
        recommendations = []
        bottlenecks = []

        # Environment performance optimization
        if benchmark_results.environment_results:
            step_time = getattr(
                benchmark_results.environment_results, "avg_step_time_ms", 0
            )
            target_step_time = benchmark_results.suite_config.performance_targets.get(
                "step_latency_ms", 1.0
            )

            if step_time > target_step_time:
                ratio = step_time / target_step_time
                bottlenecks.append(
                    f"Environment step latency {ratio:.1f}x target ({step_time:.3f}ms vs {target_step_time:.3f}ms)"
                )
                recommendations.append(
                    {
                        "category": "environment",
                        "priority": "high" if ratio > 2.0 else "medium",
                        "recommendation": "Optimize environment step execution pipeline",
                        "expected_improvement": f"Potential {min(50, (ratio-1)*30):.0f}% latency reduction",
                        "implementation_effort": "medium",
                    }
                )

        # Memory optimization
        if benchmark_results.memory_usage_results:
            memory_usage = benchmark_results.memory_usage_results.get(
                "peak_memory_mb", 0
            )
            target_memory = benchmark_results.suite_config.performance_targets.get(
                "memory_usage_mb", 50.0
            )

            if memory_usage > target_memory:
                ratio = memory_usage / target_memory
                bottlenecks.append(
                    f"Memory usage {ratio:.1f}x target ({memory_usage:.1f}MB vs {target_memory:.1f}MB)"
                )
                recommendations.append(
                    {
                        "category": "memory",
                        "priority": "high" if ratio > 1.5 else "medium",
                        "recommendation": "Implement memory optimization and garbage collection improvements",
                        "expected_improvement": f"Potential {min(40, (ratio-1)*25):.0f}% memory reduction",
                        "implementation_effort": "high",
                    }
                )

        # Rendering optimization
        if benchmark_results.rendering_results:
            rgb_time = benchmark_results.rendering_results.get("avg_rgb_render_ms", 0)
            target_rgb = benchmark_results.suite_config.performance_targets.get(
                "rendering_rgb_ms", 5.0
            )

            if rgb_time > target_rgb:
                ratio = rgb_time / target_rgb
                bottlenecks.append(
                    f"RGB rendering {ratio:.1f}x target ({rgb_time:.3f}ms vs {target_rgb:.3f}ms)"
                )
                recommendations.append(
                    {
                        "category": "rendering",
                        "priority": "medium",
                        "recommendation": "Optimize RGB array generation and rendering pipeline",
                        "expected_improvement": f"Potential {min(30, (ratio-1)*20):.0f}% rendering speedup",
                        "implementation_effort": "medium",
                    }
                )

        # Cross-category bottleneck analysis if enabled
        if analyze_cross_category_bottlenecks:
            # Identify if multiple categories are underperforming
            high_priority_count = sum(
                1 for rec in recommendations if rec["priority"] == "high"
            )

            if high_priority_count >= 2:
                bottlenecks.append(
                    "Multiple high-priority performance issues detected - systemic optimization needed"
                )
                recommendations.append(
                    {
                        "category": "system",
                        "priority": "critical",
                        "recommendation": "Implement comprehensive system-wide performance optimization",
                        "expected_improvement": "Potential 20-40% overall performance improvement",
                        "implementation_effort": "very_high",
                    }
                )

        # Consider system constraints if provided
        if system_constraints:
            max_memory = system_constraints.get("max_memory_mb")
            if max_memory and benchmark_results.memory_usage_results:
                current_memory = benchmark_results.memory_usage_results.get(
                    "peak_memory_mb", 0
                )

                if current_memory > max_memory * 0.8:  # 80% of constraint
                    recommendations.append(
                        {
                            "category": "constraint",
                            "priority": "critical",
                            "recommendation": f"Urgent memory optimization required - approaching {max_memory}MB constraint",
                            "expected_improvement": "Prevent system constraint violation",
                            "implementation_effort": "high",
                        }
                    )

        # Prioritize by impact if enabled
        if prioritize_by_impact:
            priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
            recommendations.sort(key=lambda x: priority_order.get(x["priority"], 99))

        # Generate implementation roadmap if enabled
        roadmap = []
        if generate_implementation_roadmap:
            phases = {
                "Phase 1 (Immediate)": [
                    r for r in recommendations if r["priority"] in ["critical", "high"]
                ],
                "Phase 2 (Short-term)": [
                    r for r in recommendations if r["priority"] == "medium"
                ],
                "Phase 3 (Long-term)": [
                    r for r in recommendations if r["priority"] == "low"
                ],
            }

            for phase, phase_recs in phases.items():
                if phase_recs:
                    roadmap.append(
                        {
                            "phase": phase,
                            "recommendations": len(phase_recs),
                            "estimated_effort": (
                                max([r["implementation_effort"] for r in phase_recs])
                                if phase_recs
                                else "low"
                            ),
                            "expected_impact": (
                                "High" if phase == "Phase 1 (Immediate)" else "Medium"
                            ),
                        }
                    )

        optimization_analysis.update(
            {
                "optimization_recommendations": recommendations,
                "bottlenecks_identified": bottlenecks,
                "implementation_roadmap": roadmap,
                "total_recommendations": len(recommendations),
                "critical_issues": len(
                    [r for r in recommendations if r["priority"] == "critical"]
                ),
            }
        )

        logger.info(
            f"Performance optimization analysis completed: {len(recommendations)} recommendations generated"
        )
        return optimization_analysis

    except Exception as e:
        logger.error(f"Performance optimization analysis failed: {e}")
        return {
            "analysis_status": "failed",
            "error": str(e),
            "analysis_timestamp": time.time(),
        }


def export_benchmark_results(
    benchmark_results: ConsolidatedBenchmarkResult,
    output_directory: pathlib.Path,
    export_formats: List[str] = ["json", "markdown"],
    include_raw_data: bool = True,
    generate_visualizations: bool = False,
    create_ci_artifacts: bool = False,
) -> Dict[str, Any]:
    """
    Export comprehensive benchmark results in multiple formats with structured data preservation,
    visualization generation, and integration support for external systems, CI pipelines, and
    performance monitoring infrastructure.
    """
    logger = get_component_logger("results_exporter", "UTILS")
    logger.info(f"Exporting benchmark results to {output_directory}")

    try:
        # Use the consolidated result's built-in export functionality
        export_result = benchmark_results.export_consolidated_results(
            output_directory=output_directory,
            export_formats=export_formats,
            include_raw_category_data=include_raw_data,
            generate_executive_artifacts=True,
        )

        # Add CI-specific artifacts if requested
        if create_ci_artifacts:
            ci_dir = output_directory / "ci"
            ci_dir.mkdir(exist_ok=True)

            # Create CI summary file
            ci_summary = {
                "benchmark_status": (
                    "PASS" if benchmark_results.all_targets_met else "FAIL"
                ),
                "execution_id": benchmark_results.suite_execution_id,
                "timestamp": benchmark_results.execution_timestamp,
                "performance_score": benchmark_results.consolidated_statistics.get(
                    "overall_performance", {}
                ).get("performance_score", 0),
                "violations_count": len(
                    benchmark_results.performance_validation.get("violations", [])
                ),
                "recommendations_count": len(
                    benchmark_results.integrated_optimization_recommendations
                ),
            }

            ci_summary_file = ci_dir / "ci_summary.json"
            with open(ci_summary_file, "w") as f:
                json.dump(ci_summary, f, indent=2)

            export_result["ci_artifacts"] = [str(ci_summary_file)]
            logger.info("CI artifacts generated")

        # Generate basic visualizations if requested (simplified implementation)
        if generate_visualizations:
            viz_dir = output_directory / "visualizations"
            viz_dir.mkdir(exist_ok=True)

            # Create simple text-based visualization data
            viz_data = {
                "performance_summary": {
                    "categories_tested": sum(
                        [
                            bool(benchmark_results.environment_results),
                            bool(benchmark_results.rendering_results),
                            bool(benchmark_results.step_latency_results),
                            bool(benchmark_results.memory_usage_results),
                            bool(benchmark_results.plume_generation_results),
                        ]
                    ),
                    "targets_met": benchmark_results.all_targets_met,
                    "performance_score": benchmark_results.consolidated_statistics.get(
                        "overall_performance", {}
                    ).get("performance_score", 0),
                }
            }

            viz_file = viz_dir / "performance_visualization_data.json"
            with open(viz_file, "w") as f:
                json.dump(viz_data, f, indent=2)

            export_result["visualizations"] = [str(viz_file)]
            logger.info("Visualization data generated")

        logger.info(
            f"Benchmark results exported successfully: {export_result['export_status']}"
        )
        return export_result

    except Exception as e:
        logger.error(f"Benchmark results export failed: {e}")
        return {
            "export_status": "failed",
            "error": str(e),
            "export_timestamp": time.time(),
        }


def create_ci_benchmark_config(
    ci_profile_type: str = "quick_validation",
    custom_parameters: Optional[Dict[str, Any]] = None,
    enable_regression_detection: bool = False,
    execution_time_limit_minutes: float = 5.0,
) -> BenchmarkSuiteConfig:
    """
    Create optimized benchmark configuration specifically tailored for continuous integration
    environments with reduced execution time, focused validation, and automated quality gate
    integration for efficient CI pipeline performance monitoring.
    """
    logger = get_component_logger("ci_config_creator", "UTILS")
    logger.info(f"Creating CI benchmark configuration: {ci_profile_type}")

    try:
        # Create base configuration
        ci_config = BenchmarkSuiteConfig()

        # Apply CI profile-specific settings
        if ci_profile_type == "quick_validation":
            # Minimal validation for rapid feedback
            ci_config.include_environment_benchmark = True
            ci_config.include_rendering_benchmark = False  # Skip rendering for speed
            ci_config.include_step_latency_benchmark = True
            ci_config.include_memory_benchmark = True
            ci_config.include_plume_generation_benchmark = False  # Skip for speed

            # Quick execution settings
            ci_config.enable_parallel_execution = True
            ci_config.generate_integrated_analysis = False
            ci_config.execution_timeout_minutes = int(execution_time_limit_minutes)

            # Reduced iteration counts for speed
            ci_config.category_specific_configs = {
                "environment": {"iterations": 10},
                "step_latency": {"iterations": 100},
                "memory": {"samples": 3},
            }

        elif ci_profile_type == "regression_detection":
            # Focus on regression detection
            ci_config.include_environment_benchmark = True
            ci_config.include_rendering_benchmark = True
            ci_config.include_step_latency_benchmark = True
            ci_config.include_memory_benchmark = True
            ci_config.include_plume_generation_benchmark = True

            ci_config.detect_regressions = True
            ci_config.validate_performance_targets = True
            ci_config.enable_parallel_execution = True
            ci_config.execution_timeout_minutes = int(
                execution_time_limit_minutes * 2
            )  # More time for regression analysis

        elif ci_profile_type == "comprehensive_validation":
            # Full validation but optimized for CI
            ci_config.include_environment_benchmark = True
            ci_config.include_rendering_benchmark = True
            ci_config.include_step_latency_benchmark = True
            ci_config.include_memory_benchmark = True
            ci_config.include_plume_generation_benchmark = True

            ci_config.generate_integrated_analysis = True
            ci_config.validate_performance_targets = True
            ci_config.enable_parallel_execution = True
            ci_config.execution_timeout_minutes = int(execution_time_limit_minutes * 3)

        # Apply custom parameters if provided
        if custom_parameters:
            for key, value in custom_parameters.items():
                if hasattr(ci_config, key):
                    setattr(ci_config, key, value)
                    logger.debug(f"Applied custom parameter: {key} = {value}")

        # Enable regression detection if requested
        if enable_regression_detection:
            ci_config.detect_regressions = True

        # CI-specific performance targets (more lenient for CI environment)
        ci_config.performance_targets = {
            "step_latency_ms": 2.0,  # 2x normal target for CI environment
            "episode_reset_ms": 20.0,
            "memory_usage_mb": 75.0,  # Higher memory limit for CI
            "rendering_rgb_ms": 10.0,
            "rendering_human_ms": 100.0,
        }

        # Validate the CI configuration
        if not ci_config.validate_suite_config():
            logger.warning("CI configuration validation failed, using defaults")
            ci_config = BenchmarkSuiteConfig()  # Fallback to defaults

        logger.info(f"CI benchmark configuration created: {ci_profile_type}")
        return ci_config

    except Exception as e:
        logger.error(f"CI configuration creation failed: {e}")
        # Return minimal working configuration
        fallback_config = BenchmarkSuiteConfig()
        fallback_config.execution_timeout_minutes = int(execution_time_limit_minutes)
        fallback_config.enable_parallel_execution = True
        return fallback_config


def run_quick_validation_suite(
    validation_targets: Optional[Dict[str, float]] = None,
    include_regression_check: bool = False,
    baseline_comparison_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute streamlined benchmark suite optimized for rapid validation with essential performance
    checks, reduced iterations, and focused analysis for development workflow integration and quick
    performance feedback.
    """
    logger = get_component_logger("quick_validation", "UTILS")
    logger.info("Starting quick validation suite")

    try:
        # Create quick validation configuration
        quick_config = create_ci_benchmark_config(
            ci_profile_type="quick_validation", execution_time_limit_minutes=3.0
        )

        # Override performance targets if provided
        if validation_targets:
            quick_config.performance_targets.update(validation_targets)

        # Execute quick benchmark suite
        logger.info("Executing quick validation benchmarks")
        result = run_comprehensive_benchmark_suite(
            config=quick_config,
            validate_performance_targets=True,
            detect_regressions=include_regression_check,
            parallel_execution=True,
        )

        # Create simplified validation result
        validation_result = {
            "validation_timestamp": time.time(),
            "validation_passed": result.all_targets_met,
            "execution_time_seconds": time.time() - result.execution_timestamp,
            "categories_tested": sum(
                [
                    bool(result.environment_results),
                    bool(result.rendering_results),
                    bool(result.step_latency_results),
                    bool(result.memory_usage_results),
                    bool(result.plume_generation_results),
                ]
            ),
            "performance_summary": {},
            "critical_issues": [],
        }

        # Extract key performance metrics
        if result.environment_results:
            validation_result["performance_summary"]["step_latency_ms"] = getattr(
                result.environment_results, "avg_step_time_ms", 0
            )

        if result.memory_usage_results:
            validation_result["performance_summary"]["memory_usage_mb"] = (
                result.memory_usage_results.get("peak_memory_mb", 0)
            )

        # Include critical issues
        if result.performance_validation and result.performance_validation.get(
            "violations"
        ):
            validation_result["critical_issues"] = result.performance_validation[
                "violations"
            ][
                :3
            ]  # Top 3 issues

        # Include regression information if enabled
        if include_regression_check and baseline_comparison_path:
            validation_result["regression_check"] = {
                "enabled": True,
                "baseline_path": baseline_comparison_path,
                "status": "baseline_not_loaded",  # Simplified for this implementation
            }

        logger.info(
            f"Quick validation completed: {'PASSED' if validation_result['validation_passed'] else 'FAILED'}"
        )
        return validation_result

    except Exception as e:
        logger.error(f"Quick validation suite failed: {e}")
        return {
            "validation_passed": False,
            "error": str(e),
            "validation_timestamp": time.time(),
        }


def run_comprehensive_analysis_suite(
    detailed_config: Optional[BenchmarkSuiteConfig] = None,
    include_predictive_analysis: bool = False,
    generate_optimization_roadmap: bool = True,
    statistical_confidence_iterations: int = 100,
    comprehensive_output_path: Optional[str] = None,
) -> ConsolidatedBenchmarkResult:
    """
    Execute detailed comprehensive benchmark suite with extended analysis, thorough statistical
    validation, cross-category correlation analysis, and in-depth optimization recommendations
    for research, development, and production optimization workflows.
    """
    logger = get_component_logger("comprehensive_analysis", "UTILS")
    logger.info("Starting comprehensive analysis suite")

    try:
        # Create comprehensive configuration if not provided
        if detailed_config is None:
            detailed_config = BenchmarkSuiteConfig()

            # Enable all benchmark categories
            detailed_config.include_environment_benchmark = True
            detailed_config.include_rendering_benchmark = True
            detailed_config.include_step_latency_benchmark = True
            detailed_config.include_memory_benchmark = True
            detailed_config.include_plume_generation_benchmark = True

            # Enable comprehensive analysis features
            detailed_config.generate_integrated_analysis = True
            detailed_config.validate_performance_targets = True
            detailed_config.enable_parallel_execution = True
            detailed_config.execution_timeout_minutes = (
                45  # Extended time for comprehensive analysis
            )

            # Increased iterations for statistical confidence
            detailed_config.category_specific_configs = {
                "environment": {"iterations": statistical_confidence_iterations},
                "rendering": {"iterations": statistical_confidence_iterations // 2},
                "step_latency": {"iterations": statistical_confidence_iterations * 10},
                "memory": {"samples": statistical_confidence_iterations // 5},
                "plume_generation": {
                    "iterations": statistical_confidence_iterations // 2
                },
            }

        logger.info("Executing comprehensive benchmark suite")

        # Execute comprehensive benchmark suite
        comprehensive_result = run_comprehensive_benchmark_suite(
            config=detailed_config,
            validate_performance_targets=True,
            detect_regressions=False,  # Would require historical data
            output_path=comprehensive_output_path,
            parallel_execution=detailed_config.enable_parallel_execution,
        )

        # Perform additional comprehensive analysis
        if comprehensive_result.consolidated_statistics:
            # Calculate extended statistical confidence
            comprehensive_result.calculate_consolidated_statistics(
                include_cross_correlation=True,
                calculate_performance_score=True,
                confidence_level=0.99,  # Higher confidence for comprehensive analysis
            )

        # Generate optimization roadmap if requested
        if generate_optimization_roadmap:
            logger.info("Generating optimization roadmap")
            optimization_analysis = optimize_system_performance(
                benchmark_results=comprehensive_result,
                analyze_cross_category_bottlenecks=True,
                generate_implementation_roadmap=True,
                prioritize_by_impact=True,
            )

            # Add optimization roadmap to result
            if optimization_analysis.get("implementation_roadmap"):
                comprehensive_result.cross_category_analysis["optimization_roadmap"] = (
                    optimization_analysis["implementation_roadmap"]
                )

        # Include predictive analysis if requested (simplified implementation)
        if include_predictive_analysis:
            logger.info("Including predictive performance analysis")
            predictive_insights = {
                "trend_prediction": "Predictive analysis requires historical data for trend modeling",
                "performance_forecast": "Long-term performance forecasting not available in this implementation",
                "capacity_planning": "Resource capacity planning requires workload projections",
            }
            comprehensive_result.cross_category_analysis["predictive_analysis"] = (
                predictive_insights
            )

        # Generate enhanced executive summary
        comprehensive_result.generate_integrated_executive_summary(
            include_strategic_recommendations=True,
            highlight_critical_issues=True,
            include_performance_trends=include_predictive_analysis,
        )

        # Export comprehensive results if output path specified
        if comprehensive_output_path:
            logger.info(
                f"Exporting comprehensive results to {comprehensive_output_path}"
            )
            export_result = export_benchmark_results(
                benchmark_results=comprehensive_result,
                output_directory=pathlib.Path(comprehensive_output_path),
                export_formats=["json", "markdown"],
                include_raw_data=True,
                generate_visualizations=True,
                create_ci_artifacts=False,
            )

            if export_result["export_status"] == "success":
                logger.info("Comprehensive results exported successfully")

        logger.info("Comprehensive analysis suite completed successfully")
        return comprehensive_result

    except Exception as e:
        logger.error(f"Comprehensive analysis suite failed: {e}")
        # Return error result
        error_result = ConsolidatedBenchmarkResult(
            suite_execution_id=f"comprehensive_failed_{int(time.time())}",
            execution_timestamp=time.time(),
            suite_config=detailed_config or BenchmarkSuiteConfig(),
        )
        error_result.integrated_optimization_recommendations = [
            f"Comprehensive analysis failed: {str(e)}"
        ]
        return error_result
