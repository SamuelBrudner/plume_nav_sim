"""
Comprehensive environment performance benchmarking module providing detailed analysis of PlumeSearchEnv
performance including step latency measurement, episode timing analysis, memory usage profiling, and
component-specific optimization validation. Implements statistical analysis, performance target validation,
and optimization recommendations for development performance tuning and continuous integration validation
with <1ms step latency verification and resource constraint compliance.

This module provides a complete performance analysis framework for the plume navigation environment,
offering detailed benchmarking capabilities including timing analysis, memory profiling, scalability
assessment, and comprehensive reporting for development teams and continuous integration workflows.

Key Features:
- Step latency benchmarking with <1ms target validation and statistical analysis
- Episode performance analysis including reset timing <10ms and completion metrics
- Memory usage profiling with <50MB total footprint validation and leak detection
- Component-specific performance analysis for PlumeSearchEnv optimization validation
- Scalability analysis across different grid configurations with resource constraint compliance
- Statistical validation with confidence intervals and performance regression detection
- Comprehensive reporting with executive summaries and actionable optimization recommendations

Performance Targets:
- Environment step execution: <1ms average latency with 0.1ms-2ms acceptable range
- Episode reset operations: <10ms target with 1ms-50ms acceptable range
- Memory usage: <50MB total system footprint for 128x128 default configuration
- RGB rendering: <5ms frame generation with performance optimization validation
"""

# External imports with version requirements for comprehensive performance analysis
import contextlib  # >=3.10 - Context managers for resource management and cleanup operations
import dataclasses  # >=3.10 - Data structure definitions for benchmark results and configuration
import gc  # >=3.10 - Garbage collection control for memory leak detection and optimization
import json  # >=3.10 - Benchmark result serialization and structured output generation
import logging  # >=3.10 - Benchmark execution logging and performance analysis documentation
import pathlib  # >=3.10 - File system operations for benchmark result storage and report generation
import statistics  # >=3.10 - Statistical analysis including mean, median, percentile calculations
import threading  # >=3.10 - Thread-safe performance monitoring and concurrent resource tracking
import time  # >=3.10 - High-precision timing measurements using perf_counter for step latency analysis
from typing import (  # >=3.10 - Type hints for comprehensive API
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

# Third-party imports with version comments for mathematical operations and system monitoring
import numpy as np  # >=2.1.0 - Statistical analysis, timing calculations, and performance metrics computation
import psutil  # >=5.8.0 - System resource monitoring including memory usage and process-level profiling

from plume_nav_sim.core.constants import (
    DEFAULT_GRID_SIZE,
    MEMORY_LIMIT_TOTAL_MB,
    PERFORMANCE_TARGET_STEP_LATENCY_MS,
)
from plume_nav_sim.core.enums import Action
from plume_nav_sim.core.geometry import GridSize

# Internal imports for environment benchmarking and validation framework integration
from plume_nav_sim.envs.plume_search_env import PlumeSearchEnv, create_plume_search_env
from plume_nav_sim.utils.validation import (
    ValidationContext,
    ValidationResult,
    validate_environment_config,
)

# Global configuration constants for benchmark execution and analysis
DEFAULT_BENCHMARK_ITERATIONS = 1000
DEFAULT_WARMUP_ITERATIONS = 100
MEMORY_SAMPLING_INTERVAL = 0.01  # 10ms sampling for continuous memory monitoring
BENCHMARK_TIMEOUT_SECONDS = 300  # 5 minutes maximum execution time
STATISTICAL_CONFIDENCE_LEVEL = 0.95  # 95% confidence intervals for statistical analysis
PERFORMANCE_REGRESSION_THRESHOLD = 0.1  # 10% performance degradation threshold
MEMORY_LEAK_THRESHOLD_MB = 5.0  # 5MB memory increase threshold for leak detection
GRID_SIZE_SCALING_FACTORS = [
    1,
    2,
    4,
    8,
]  # Scaling factors for grid size scalability analysis

# Module exports for comprehensive benchmarking functionality and external integration
__all__ = [
    "run_environment_performance_benchmark",
    "EnvironmentPerformanceSuite",
    "BenchmarkResult",
    "EnvironmentBenchmarkConfig",
    "PerformanceAnalysis",
    "TimingAnalyzer",
    "MemoryProfiler",
    "ScalabilityAnalyzer",
    "PerformanceReport",
    "benchmark_step_latency",
    "benchmark_episode_performance",
    "benchmark_memory_usage",
    "benchmark_rendering_performance",
    "analyze_scaling_performance",
    "validate_performance_targets",
    "generate_performance_report",
    "PerformanceAnalysis",
]


@dataclasses.dataclass
class EnvironmentBenchmarkConfig:
    """
    Configuration data class for environment performance benchmarking specifying benchmark parameters,
    performance targets, analysis options, and output settings with validation and serialization
    support for systematic benchmark execution.

    This configuration class provides comprehensive control over benchmark execution including
    iteration counts, memory profiling options, scaling analysis parameters, performance targets,
    and output formatting for systematic performance evaluation and optimization analysis.

    Attributes:
        iterations (int): Number of benchmark iterations for statistical significance
        warmup_iterations (int): Warmup iterations to stabilize system performance
        enable_memory_profiling (bool): Enable comprehensive memory usage analysis
        enable_scaling_analysis (bool): Enable grid size scaling performance analysis
        scaling_grid_sizes (List[tuple]): Grid sizes for scalability testing
        performance_targets (dict): Performance targets for validation and compliance
        timeout_seconds (Optional[int]): Maximum benchmark execution time
        validate_targets (bool): Enable performance target validation
        detect_memory_leaks (bool): Enable memory leak detection analysis
        memory_leak_threshold_mb (float): Threshold for memory leak detection
        include_action_analysis (bool): Enable action-specific performance analysis
        include_position_analysis (bool): Enable position impact analysis
        output_directory (Optional[pathlib.Path]): Directory for benchmark output files
        output_formats (List[str]): Output formats for benchmark results
    """

    iterations: int = DEFAULT_BENCHMARK_ITERATIONS
    warmup_iterations: int = DEFAULT_WARMUP_ITERATIONS
    enable_memory_profiling: bool = True
    enable_scaling_analysis: bool = False
    scaling_grid_sizes: List[tuple] = dataclasses.field(
        default_factory=lambda: [(32, 32), (64, 64), (128, 128), (256, 256)]
    )
    performance_targets: dict = dataclasses.field(
        default_factory=lambda: {
            "step_latency_ms": PERFORMANCE_TARGET_STEP_LATENCY_MS,
            "episode_reset_ms": 10.0,
            "memory_limit_mb": MEMORY_LIMIT_TOTAL_MB,
            "render_time_ms": 5.0,
        }
    )
    timeout_seconds: Optional[int] = BENCHMARK_TIMEOUT_SECONDS
    validate_targets: bool = True
    detect_memory_leaks: bool = True
    memory_leak_threshold_mb: float = MEMORY_LEAK_THRESHOLD_MB
    include_action_analysis: bool = False
    include_position_analysis: bool = False
    output_directory: Optional[pathlib.Path] = None
    output_formats: List[str] = dataclasses.field(default_factory=lambda: ["json"])

    def validate_config(self, strict_validation: bool = False) -> ValidationResult:
        """
        Comprehensive configuration validation including parameter consistency checking,
        resource feasibility analysis, and optimization recommendation generation.

        Args:
            strict_validation (bool): Enable strict type checking with additional validation

        Returns:
            ValidationResult: Configuration validation result with analysis and recommendations
        """
        errors = []
        warnings = []

        # Validate iteration parameters with reasonable bounds checking
        if self.iterations <= 0:
            errors.append("Iterations must be positive integer")
        elif self.iterations < 100:
            warnings.append("Low iteration count may reduce statistical significance")
        elif self.iterations > 10000:
            warnings.append(
                "High iteration count may increase execution time significantly"
            )

        if self.warmup_iterations < 0:
            errors.append("Warmup iterations cannot be negative")
        elif self.warmup_iterations > self.iterations:
            errors.append("Warmup iterations cannot exceed total iterations")

        # Validate timeout parameter with execution time estimation
        if self.timeout_seconds is not None:
            if self.timeout_seconds <= 0:
                errors.append("Timeout must be positive")
            else:
                estimated_time = (
                    self.iterations + self.warmup_iterations
                ) * 0.002  # 2ms per iteration estimate
                if estimated_time > self.timeout_seconds:
                    warnings.append(
                        f"Timeout ({self.timeout_seconds}s) may be insufficient for {self.iterations} iterations"
                    )

        # Validate scaling grid sizes for memory feasibility
        if self.enable_scaling_analysis:
            for grid_size in self.scaling_grid_sizes:
                if len(grid_size) != 2:
                    errors.append(f"Grid size must be 2-tuple: {grid_size}")
                else:
                    width, height = grid_size
                    if width <= 0 or height <= 0:
                        errors.append(f"Grid dimensions must be positive: {grid_size}")

                    # Memory feasibility check
                    estimated_memory = (width * height * 4) / (
                        1024 * 1024
                    )  # float32 to MB
                    if estimated_memory > 500:  # 500MB threshold
                        errors.append(
                            f"Grid size {grid_size} requires too much memory ({estimated_memory:.1f}MB)"
                        )

        # Validate performance targets consistency
        if self.performance_targets.get("step_latency_ms", 0) <= 0:
            errors.append("Step latency target must be positive")
        if self.performance_targets.get("memory_limit_mb", 0) <= 0:
            errors.append("Memory limit must be positive")

        # Validate memory leak detection parameters
        if self.memory_leak_threshold_mb <= 0:
            errors.append("Memory leak threshold must be positive")

        # Validate output configuration
        if self.output_directory is not None:
            try:
                self.output_directory.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                errors.append(f"Cannot create output directory: {e}")

        valid_formats = ["json", "csv", "txt", "html"]
        for fmt in self.output_formats:
            if fmt not in valid_formats:
                warnings.append(f"Unknown output format: {fmt}")

        context = ValidationContext(
            operation_name="EnvironmentBenchmarkConfig.validate_config",
            component_name="environment_performance",
            timestamp=time.time(),
        )
        context.merge_context({
            "iterations": self.iterations,
            "warmup_iterations": self.warmup_iterations,
            "scaling_grid_sizes": self.scaling_grid_sizes,
            "performance_targets": self.performance_targets,
            "timeout_seconds": self.timeout_seconds,
            "enable_memory_profiling": self.enable_memory_profiling,
            "enable_scaling_analysis": self.enable_scaling_analysis,
        })

        validation_result = ValidationResult(
            is_valid=len(errors) == 0,
            operation_name="EnvironmentBenchmarkConfig.validate_config",
            context=context,
        )

        for error in errors:
            validation_result.add_error(error)
        for warning in warnings:
            validation_result.add_warning(warning)

        validation_result.summary_message = (
            "Configuration valid"
            if validation_result.is_valid
            else "Configuration validation failed"
        )

        return validation_result

    def to_dict(self) -> dict:
        """
        Serialize configuration to dictionary format for JSON export and storage.

        Returns:
            dict: Complete configuration dictionary with all parameters and settings
        """
        return {
            "iterations": self.iterations,
            "warmup_iterations": self.warmup_iterations,
            "enable_memory_profiling": self.enable_memory_profiling,
            "enable_scaling_analysis": self.enable_scaling_analysis,
            "scaling_grid_sizes": self.scaling_grid_sizes,
            "performance_targets": self.performance_targets,
            "timeout_seconds": self.timeout_seconds,
            "validate_targets": self.validate_targets,
            "detect_memory_leaks": self.detect_memory_leaks,
            "memory_leak_threshold_mb": self.memory_leak_threshold_mb,
            "include_action_analysis": self.include_action_analysis,
            "include_position_analysis": self.include_position_analysis,
            "output_directory": (
                str(self.output_directory) if self.output_directory else None
            ),
            "output_formats": self.output_formats,
        }

    def estimate_execution_time(self, include_analysis_overhead: bool = True) -> float:
        """
        Estimate total benchmark execution time based on configuration parameters.

        Args:
            include_analysis_overhead (bool): Include analysis and reporting overhead

        Returns:
            float: Estimated total execution time in seconds
        """
        base_time = (
            self.iterations + self.warmup_iterations
        ) * 0.002  # 2ms per iteration

        if self.enable_memory_profiling:
            base_time *= 1.5  # 50% overhead for memory monitoring

        if self.enable_scaling_analysis:
            base_time *= len(self.scaling_grid_sizes)  # Multiple grid sizes

        if include_analysis_overhead:
            base_time *= 1.2  # 20% overhead for analysis and reporting

        return base_time


@dataclasses.dataclass
class BenchmarkResult:
    """
    Comprehensive benchmark result data structure containing performance metrics, statistical
    analysis, validation status, and optimization recommendations with serialization support
    and statistical analysis methods for benchmark data management and reporting.

    This class encapsulates all benchmark execution results including timing measurements,
    memory usage analysis, validation outcomes, and actionable optimization recommendations
    with comprehensive statistical analysis and serialization capabilities.

    Attributes:
        benchmark_name (str): Unique identifier for benchmark execution
        execution_timestamp (float): Unix timestamp of benchmark execution
        config (EnvironmentBenchmarkConfig): Configuration used for benchmark execution
        step_latency_metrics (dict): Step latency timing measurements and statistics
        episode_performance_metrics (dict): Episode-level performance analysis results
        memory_usage_metrics (dict): Memory usage profiling and leak detection results
        scalability_metrics (Optional[dict]): Grid size scaling analysis results
        validation_results (dict): Performance target validation outcomes
        optimization_recommendations (List[str]): Actionable optimization suggestions
        targets_met (bool): Overall performance target compliance status
        executive_summary (Optional[str]): High-level summary for stakeholder communication
    """

    benchmark_name: str
    execution_timestamp: float
    config: EnvironmentBenchmarkConfig
    step_latency_metrics: dict = dataclasses.field(default_factory=dict)
    episode_performance_metrics: dict = dataclasses.field(default_factory=dict)
    memory_usage_metrics: dict = dataclasses.field(default_factory=dict)
    scalability_metrics: Optional[dict] = None
    validation_results: dict = dataclasses.field(default_factory=dict)
    optimization_recommendations: List[str] = dataclasses.field(default_factory=list)
    targets_met: bool = False
    executive_summary: Optional[str] = None

    def calculate_statistics(
        self,
        include_distribution_analysis: bool = True,
        confidence_level: float = STATISTICAL_CONFIDENCE_LEVEL,
    ) -> dict:
        """
        Calculate comprehensive statistical analysis of benchmark metrics including
        descriptive statistics, distribution analysis, and confidence intervals.

        Args:
            include_distribution_analysis (bool): Include detailed distribution analysis
            confidence_level (float): Statistical confidence level for intervals

        Returns:
            dict: Comprehensive statistical analysis with descriptive statistics and intervals
        """
        stats_result = {
            "timestamp": time.time(),
            "confidence_level": confidence_level,
            "step_latency_statistics": {},
            "episode_performance_statistics": {},
            "memory_usage_statistics": {},
        }

        # Analyze step latency statistics with timing distribution
        if "timings" in self.step_latency_metrics:
            timings = self.step_latency_metrics["timings"]
            stats_result["step_latency_statistics"] = {
                "count": len(timings),
                "mean": statistics.mean(timings),
                "median": statistics.median(timings),
                "stdev": statistics.stdev(timings) if len(timings) > 1 else 0.0,
                "min": min(timings),
                "max": max(timings),
                "p95": np.percentile(timings, 95),
                "p99": np.percentile(timings, 99),
            }

            # Add confidence interval calculation
            if len(timings) > 30:  # Sufficient sample size
                margin_error = 1.96 * (
                    stats_result["step_latency_statistics"]["stdev"]
                    / np.sqrt(len(timings))
                )
                mean_val = stats_result["step_latency_statistics"]["mean"]
                stats_result["step_latency_statistics"]["confidence_interval"] = [
                    mean_val - margin_error,
                    mean_val + margin_error,
                ]

        # Analyze episode performance statistics
        if "episode_durations" in self.episode_performance_metrics:
            durations = self.episode_performance_metrics["episode_durations"]
            if durations:
                stats_result["episode_performance_statistics"] = {
                    "count": len(durations),
                    "mean_duration": statistics.mean(durations),
                    "median_duration": statistics.median(durations),
                    "total_episodes": len(durations),
                }

        # Analyze memory usage statistics with trend analysis
        if "memory_samples" in self.memory_usage_metrics:
            memory_samples = self.memory_usage_metrics["memory_samples"]
            if memory_samples:
                stats_result["memory_usage_statistics"] = {
                    "peak_usage_mb": max(memory_samples),
                    "mean_usage_mb": statistics.mean(memory_samples),
                    "memory_growth_mb": max(memory_samples) - min(memory_samples),
                    "sample_count": len(memory_samples),
                }

        return stats_result

    def validate_against_target(
        self,
        performance_targets: dict,
        strict_validation: bool = False,
        significance_threshold: float = 0.05,
    ) -> dict:
        """
        Validate benchmark results against performance targets with statistical
        significance testing and improvement recommendations.

        Args:
            performance_targets (dict): Target performance values for validation
            strict_validation (bool): Apply strict validation criteria
            significance_threshold (float): Statistical significance threshold

        Returns:
            dict: Target validation results with compliance status and recommendations
        """
        validation_result = {
            "validation_timestamp": time.time(),
            "targets_provided": performance_targets,
            "strict_mode": strict_validation,
            "compliance_results": {},
            "recommendations": [],
            "overall_compliance": True,
        }

        # Validate step latency against target
        step_target = performance_targets.get("step_latency_ms")
        if step_target and "timings" in self.step_latency_metrics:
            timings = self.step_latency_metrics["timings"]
            mean_latency = statistics.mean(timings)

            compliant = mean_latency <= step_target
            validation_result["compliance_results"]["step_latency"] = {
                "target": step_target,
                "actual": mean_latency,
                "compliant": compliant,
                "performance_ratio": mean_latency / step_target,
            }

            if not compliant:
                validation_result["overall_compliance"] = False
                validation_result["recommendations"].append(
                    f"Optimize step latency: {mean_latency:.3f}ms exceeds target {step_target}ms"
                )

        # Validate memory usage against target
        memory_target = performance_targets.get("memory_limit_mb")
        if memory_target and "peak_usage_mb" in self.memory_usage_metrics:
            peak_memory = self.memory_usage_metrics["peak_usage_mb"]

            compliant = peak_memory <= memory_target
            validation_result["compliance_results"]["memory_usage"] = {
                "target": memory_target,
                "actual": peak_memory,
                "compliant": compliant,
                "usage_ratio": peak_memory / memory_target,
            }

            if not compliant:
                validation_result["overall_compliance"] = False
                validation_result["recommendations"].append(
                    f"Reduce memory usage: {peak_memory:.1f}MB exceeds target {memory_target}MB"
                )

        # Update instance validation results and targets_met status
        self.validation_results = validation_result
        self.targets_met = validation_result["overall_compliance"]
        self.optimization_recommendations.extend(validation_result["recommendations"])

        return validation_result

    def generate_executive_summary(
        self, include_recommendations: bool = True, highlight_issues: bool = True
    ) -> str:
        """
        Generate executive summary of benchmark results with key findings and recommendations.

        Args:
            include_recommendations (bool): Include optimization recommendations
            highlight_issues (bool): Highlight performance issues and concerns

        Returns:
            str: Executive summary with key findings and strategic recommendations
        """
        summary_parts = [
            f"Performance Benchmark Summary - {self.benchmark_name}",
            f"Executed: {time.ctime(self.execution_timestamp)}",
            "",
        ]

        # Add step latency summary
        if "timings" in self.step_latency_metrics:
            timings = self.step_latency_metrics["timings"]
            mean_latency = statistics.mean(timings)
            target_met = mean_latency <= self.config.performance_targets.get(
                "step_latency_ms", 1.0
            )
            status = "PASS" if target_met else "FAIL"

            summary_parts.append(
                f"Step Latency: {mean_latency:.3f}ms (Target: {self.config.performance_targets.get('step_latency_ms', 1.0)}ms) [{status}]"
            )

        # Add memory usage summary
        if "peak_usage_mb" in self.memory_usage_metrics:
            peak_memory = self.memory_usage_metrics["peak_usage_mb"]
            target_met = peak_memory <= self.config.performance_targets.get(
                "memory_limit_mb", 50.0
            )
            status = "PASS" if target_met else "FAIL"

            summary_parts.append(
                f"Peak Memory: {peak_memory:.1f}MB (Target: {self.config.performance_targets.get('memory_limit_mb', 50.0)}MB) [{status}]"
            )

        # Add overall compliance status
        summary_parts.append("")
        overall_status = "COMPLIANT" if self.targets_met else "NON-COMPLIANT"
        summary_parts.append(f"Overall Performance: {overall_status}")

        # Include issues and recommendations if requested
        if highlight_issues and not self.targets_met:
            summary_parts.append("\nPerformance Issues Identified:")
            for issue in self.optimization_recommendations[:3]:  # Top 3 issues
                summary_parts.append(f"• {issue}")

        if include_recommendations and self.optimization_recommendations:
            summary_parts.append("\nOptimization Recommendations:")
            for rec in self.optimization_recommendations[:5]:  # Top 5 recommendations
                summary_parts.append(f"• {rec}")

        executive_summary = "\n".join(summary_parts)
        self.executive_summary = executive_summary

        return executive_summary

    def to_dict(
        self, include_raw_data: bool = False, include_analysis: bool = True
    ) -> dict:
        """
        Serialize benchmark result to dictionary format for JSON export and storage.

        Args:
            include_raw_data (bool): Include raw measurement data
            include_analysis (bool): Include statistical analysis results

        Returns:
            dict: Complete benchmark result dictionary with metadata and metrics
        """
        result_dict = {
            "benchmark_name": self.benchmark_name,
            "execution_timestamp": self.execution_timestamp,
            "execution_date": time.ctime(self.execution_timestamp),
            "config": self.config.to_dict(),
            "targets_met": self.targets_met,
            "executive_summary": self.executive_summary,
        }

        # Include performance metrics
        result_dict["step_latency_metrics"] = self.step_latency_metrics.copy()
        result_dict["episode_performance_metrics"] = (
            self.episode_performance_metrics.copy()
        )
        result_dict["memory_usage_metrics"] = self.memory_usage_metrics.copy()

        # Include scaling analysis if available
        if self.scalability_metrics:
            result_dict["scalability_metrics"] = self.scalability_metrics

        # Include validation results and recommendations
        result_dict["validation_results"] = self.validation_results
        result_dict["optimization_recommendations"] = self.optimization_recommendations

        # Optionally remove raw data for smaller output size
        if not include_raw_data:
            for metric_key in [
                "step_latency_metrics",
                "episode_performance_metrics",
                "memory_usage_metrics",
            ]:
                if metric_key in result_dict and "raw_data" in result_dict[metric_key]:
                    del result_dict[metric_key]["raw_data"]

        # Include statistical analysis if requested
        if include_analysis:
            result_dict["statistical_analysis"] = self.calculate_statistics()

        return result_dict


class PerformanceAnalysis:
    """High-level helper that interprets benchmark output for the test suite.

    The original performance tests expected an object capable of summarising raw
    measurements, identifying regressions, and synthesising actionable guidance.
    This lightweight implementation focuses on the behaviours exercised by the
    rewritten tests: trend analysis across the collected benchmark dictionaries
    and generation of concise optimisation recommendations.
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(f"{__name__}.PerformanceAnalysis")
        self._trend_history: List[dict] = []

    # ------------------------------------------------------------------
    # Trend analysis
    def analyze_performance_trends(self, performance_data: dict) -> dict:
        """Analyse benchmark dictionaries for stability and regressions.

        Args:
            performance_data: Mapping of metric names to measurement dictionaries.

        Returns:
            Dict with summary statistics, detected regressions and optimisation
            opportunities. The structure mirrors what the tests expect to store
            in their trackers while remaining tolerant of partial data.
        """

        analysis = {
            "analysis_timestamp": time.time(),
            "metrics_analyzed": list(performance_data.keys()),
            "trend_patterns": {},
            "regression_indicators": [],
            "optimization_opportunities": [],
            "statistical_significance": {},
            "performance_forecast": {},
            "optimization_recommendations": [],
        }

        for metric_name, metrics in performance_data.items():
            if not isinstance(metrics, dict) or not metrics:
                continue

            timings = metrics.get("timings")
            if timings:
                values = np.asarray(timings, dtype=float)
                mean_latency = float(np.mean(values))
                std_latency = float(np.std(values))
                analysis["trend_patterns"][metric_name] = {
                    "mean": mean_latency,
                    "stdev": std_latency,
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "samples": len(values),
                }
                analysis["statistical_significance"][metric_name] = {
                    "mean": mean_latency,
                    "stdev": std_latency,
                    "sample_count": len(values),
                }

                if mean_latency > PERFORMANCE_TARGET_STEP_LATENCY_MS:
                    analysis["regression_indicators"].append(
                        {
                            "metric": metric_name,
                            "observed_mean": mean_latency,
                            "target": PERFORMANCE_TARGET_STEP_LATENCY_MS,
                            "degradation_ms": mean_latency
                            - PERFORMANCE_TARGET_STEP_LATENCY_MS,
                        }
                    )
                else:
                    analysis["optimization_opportunities"].append(
                        {
                            "metric": metric_name,
                            "observed_mean": mean_latency,
                            "headroom_ms": PERFORMANCE_TARGET_STEP_LATENCY_MS
                            - mean_latency,
                        }
                    )

            memory_samples = metrics.get("memory_samples")
            if memory_samples:
                values = np.asarray(memory_samples, dtype=float)
                peak_memory = float(np.max(values))
                analysis["trend_patterns"][f"{metric_name}_memory"] = {
                    "mean": float(np.mean(values)),
                    "peak": peak_memory,
                    "samples": len(values),
                }
                if peak_memory > MEMORY_LIMIT_TOTAL_MB:
                    analysis["regression_indicators"].append(
                        {
                            "metric": f"{metric_name}_memory",
                            "observed_peak": peak_memory,
                            "target": MEMORY_LIMIT_TOTAL_MB,
                            "degradation_mb": peak_memory - MEMORY_LIMIT_TOTAL_MB,
                        }
                    )

        if analysis["regression_indicators"]:
            for regression in analysis["regression_indicators"]:
                if "degradation_ms" in regression:
                    analysis["optimization_recommendations"].append(
                        f"Reduce latency for {regression['metric']} by {regression['degradation_ms']:.3f}ms"
                    )
                elif "degradation_mb" in regression:
                    analysis["optimization_recommendations"].append(
                        f"Lower memory usage for {regression['metric']} by {regression['degradation_mb']:.1f}MB"
                    )
        elif analysis["optimization_opportunities"]:
            for opportunity in analysis["optimization_opportunities"]:
                analysis["optimization_recommendations"].append(
                    f"Maintain optimisation for {opportunity['metric']} (headroom {opportunity['headroom_ms']:.3f}ms)"
                )
        else:
            analysis["optimization_recommendations"].append(
                "Insufficient performance data to derive trends"
            )

        self._trend_history.append(analysis)
        return analysis

    # ------------------------------------------------------------------
    # Recommendation synthesis
    def generate_optimization_recommendations(
        self,
        benchmark_results: BenchmarkResult,
        trend_analysis: dict,
        include_scaling_guidance: bool = False,
    ) -> List[str]:
        """Create actionable recommendations from benchmark and trend data."""

        recommendations: List[str] = []

        # Start with any recommendations already produced by the benchmark suite.
        if benchmark_results.optimization_recommendations:
            recommendations.extend(benchmark_results.optimization_recommendations)

        # Interpret validation results.
        if benchmark_results.validation_results.get("compliance_results"):
            for metric, result in benchmark_results.validation_results[
                "compliance_results"
            ].items():
                if not result.get("compliant", True):
                    recommendations.append(
                        f"Improve {metric.replace('_', ' ')}: observed {result.get('actual')} vs target {result.get('target')}"
                    )

        # Incorporate any trend-level recommendations.
        recommendations.extend(trend_analysis.get("optimization_recommendations", []))

        # Provide generic guidance when all targets are currently met.
        if benchmark_results.targets_met and not recommendations:
            recommendations.append(
                "Performance targets satisfied; continue monitoring for regressions."
            )

        # Optionally include scalability hints when data is available.
        if include_scaling_guidance and benchmark_results.scalability_metrics:
            recommendations.append(
                "Review scalability metrics for larger grid sizes before deployment."
            )

        # Deduplicate while preserving order for deterministic tests.
        seen = set()
        deduped: List[str] = []
        for rec in recommendations:
            if rec not in seen:
                deduped.append(rec)
                seen.add(rec)

        if not deduped:
            deduped.append(
                "No optimisation changes required based on current measurements."
            )

        self.logger.debug("Generated %d performance recommendations", len(deduped))
        return deduped

    def compare_with_baseline(
        self,
        baseline_result: "BenchmarkResult",
        detect_regressions: bool = True,
        regression_threshold: float = PERFORMANCE_REGRESSION_THRESHOLD,
    ) -> dict:
        """
        Compare current benchmark results with baseline for regression detection.

        Args:
            baseline_result (BenchmarkResult): Baseline benchmark results
            detect_regressions (bool): Enable regression detection analysis
            regression_threshold (float): Performance regression threshold

        Returns:
            dict: Comparison analysis with regression detection and change quantification
        """
        comparison_result = {
            "comparison_timestamp": time.time(),
            "baseline_timestamp": baseline_result.execution_timestamp,
            "regression_threshold": regression_threshold,
            "performance_changes": {},
            "regressions_detected": [],
            "improvements_found": [],
        }

        # Compare step latency performance
        if (
            "timings" in self.step_latency_metrics
            and "timings" in baseline_result.step_latency_metrics
        ):

            current_mean = statistics.mean(self.step_latency_metrics["timings"])
            baseline_mean = statistics.mean(
                baseline_result.step_latency_metrics["timings"]
            )

            change_ratio = (current_mean - baseline_mean) / baseline_mean
            comparison_result["performance_changes"]["step_latency"] = {
                "current_ms": current_mean,
                "baseline_ms": baseline_mean,
                "change_percent": change_ratio * 100,
                "change_ratio": change_ratio,
            }

            if detect_regressions and change_ratio > regression_threshold:
                comparison_result["regressions_detected"].append(
                    f"Step latency regression: {change_ratio*100:.1f}% slower than baseline"
                )
            elif change_ratio < -0.05:  # 5% improvement threshold
                comparison_result["improvements_found"].append(
                    f"Step latency improvement: {abs(change_ratio)*100:.1f}% faster than baseline"
                )

        # Compare memory usage performance
        current_memory = self.memory_usage_metrics.get("peak_usage_mb", 0)
        baseline_memory = baseline_result.memory_usage_metrics.get("peak_usage_mb", 0)

        if current_memory > 0 and baseline_memory > 0:
            memory_change = (current_memory - baseline_memory) / baseline_memory
            comparison_result["performance_changes"]["memory_usage"] = {
                "current_mb": current_memory,
                "baseline_mb": baseline_memory,
                "change_percent": memory_change * 100,
                "change_mb": current_memory - baseline_memory,
            }

            if detect_regressions and memory_change > regression_threshold:
                comparison_result["regressions_detected"].append(
                    f"Memory usage regression: {memory_change*100:.1f}% increase from baseline"
                )

        return comparison_result


class TimingAnalyzer:
    """
    High-precision timing analysis utility for step latency measurement, episode timing
    analysis, and performance trend tracking with statistical validation and optimization
    recommendation generation for environment performance analysis.

    This class provides comprehensive timing measurement and analysis capabilities including
    high-precision step latency tracking, action-specific performance analysis, position
    impact assessment, and statistical validation with optimization recommendations.

    Attributes:
        detailed_tracking_enabled (bool): Enable detailed per-action and position tracking
        measurement_precision (int): Decimal precision for timing measurements
        timing_measurements (List[float]): High-precision timing measurement storage
        action_specific_timings (dict): Action-specific performance analysis data
        position_impact_analysis (dict): Position-based performance correlation data
        statistical_cache (dict): Cached statistical calculations for optimization
    """

    def __init__(
        self, enable_detailed_tracking: bool = False, measurement_precision: int = 6
    ):
        """
        Initialize timing analyzer with precision configuration and measurement infrastructure.

        Args:
            enable_detailed_tracking (bool): Enable detailed action and position tracking
            measurement_precision (int): Decimal precision for timing measurements
        """
        self.detailed_tracking_enabled = enable_detailed_tracking
        self.measurement_precision = measurement_precision
        self.timing_measurements = []
        self.action_specific_timings = {}
        self.position_impact_analysis = {}
        self.statistical_cache = {}

        # Initialize action-specific tracking if detailed tracking enabled
        if self.detailed_tracking_enabled:
            for action in Action:
                self.action_specific_timings[action.value] = []

    def record_step_timing(
        self,
        timing_value: float,
        action: Optional[int] = None,
        position: Optional[tuple] = None,
    ) -> None:
        """
        Record single step execution timing with context information.

        Args:
            timing_value (float): Step execution time in milliseconds
            action (Optional[int]): Action taken for action-specific analysis
            position (Optional[tuple]): Agent position for position impact analysis
        """
        # Round to specified precision and store measurement
        precise_timing = round(timing_value, self.measurement_precision)
        self.timing_measurements.append(precise_timing)

        # Record action-specific timing if detailed tracking enabled
        if (
            self.detailed_tracking_enabled
            and action is not None
            and action in self.action_specific_timings
        ):
            self.action_specific_timings[action].append(precise_timing)

        # Record position impact if detailed tracking enabled
        if self.detailed_tracking_enabled and position is not None:
            pos_key = f"{position[0]},{position[1]}"
            if pos_key not in self.position_impact_analysis:
                self.position_impact_analysis[pos_key] = []
            self.position_impact_analysis[pos_key].append(precise_timing)

        # Invalidate statistical cache for recalculation
        self.statistical_cache.clear()

    def analyze_timing_distribution(
        self,
        include_outlier_analysis: bool = True,
        confidence_level: float = STATISTICAL_CONFIDENCE_LEVEL,
    ) -> dict:
        """
        Comprehensive timing distribution analysis with statistical validation.

        Args:
            include_outlier_analysis (bool): Include outlier detection and analysis
            confidence_level (float): Statistical confidence level for intervals

        Returns:
            dict: Timing distribution analysis with statistical validation and recommendations
        """
        if not self.timing_measurements:
            return {"error": "No timing measurements available for analysis"}

        # Use cached results if available for performance optimization
        cache_key = f"distribution_{include_outlier_analysis}_{confidence_level}"
        if cache_key in self.statistical_cache:
            return self.statistical_cache[cache_key]

        timings = np.array(self.timing_measurements)

        analysis_result = {
            "analysis_timestamp": time.time(),
            "measurement_count": len(timings),
            "descriptive_statistics": {
                "mean": float(np.mean(timings)),
                "median": float(np.median(timings)),
                "std_deviation": float(np.std(timings)),
                "min_value": float(np.min(timings)),
                "max_value": float(np.max(timings)),
                "range": float(np.max(timings) - np.min(timings)),
            },
            "percentile_analysis": {
                "p25": float(np.percentile(timings, 25)),
                "p50": float(np.percentile(timings, 50)),
                "p75": float(np.percentile(timings, 75)),
                "p90": float(np.percentile(timings, 90)),
                "p95": float(np.percentile(timings, 95)),
                "p99": float(np.percentile(timings, 99)),
            },
        }

        # Add confidence interval calculation
        if len(timings) >= 30:  # Sufficient sample size for normal approximation
            std_error = np.std(timings) / np.sqrt(len(timings))
            z_score = 1.96 if confidence_level == 0.95 else 2.576  # 95% or 99%
            margin_error = z_score * std_error
            mean_val = analysis_result["descriptive_statistics"]["mean"]

            analysis_result["confidence_interval"] = {
                "confidence_level": confidence_level,
                "lower_bound": mean_val - margin_error,
                "upper_bound": mean_val + margin_error,
                "margin_of_error": margin_error,
            }

        # Outlier analysis using IQR method
        if include_outlier_analysis:
            q1 = analysis_result["percentile_analysis"]["p25"]
            q3 = analysis_result["percentile_analysis"]["p75"]
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            outliers = timings[(timings < lower_bound) | (timings > upper_bound)]

            analysis_result["outlier_analysis"] = {
                "outlier_count": len(outliers),
                "outlier_percentage": (len(outliers) / len(timings)) * 100,
                "outlier_threshold_lower": lower_bound,
                "outlier_threshold_upper": upper_bound,
                "outlier_values": outliers.tolist(),
            }

        # Performance target validation
        target_latency = PERFORMANCE_TARGET_STEP_LATENCY_MS
        mean_timing = analysis_result["descriptive_statistics"]["mean"]

        analysis_result["performance_validation"] = {
            "target_latency_ms": target_latency,
            "meets_target": mean_timing <= target_latency,
            "performance_ratio": mean_timing / target_latency,
            "target_exceedance_ms": max(0, mean_timing - target_latency),
        }

        # Generate optimization recommendations
        recommendations = []
        if mean_timing > target_latency:
            recommendations.append(
                f"Step latency {mean_timing:.3f}ms exceeds target {target_latency}ms"
            )

        if (
            analysis_result["descriptive_statistics"]["std_deviation"]
            > mean_timing * 0.5
        ):
            recommendations.append(
                "High timing variability detected - investigate system load"
            )

        if (
            include_outlier_analysis
            and analysis_result["outlier_analysis"]["outlier_percentage"] > 5
        ):
            recommendations.append(
                "Significant outliers detected - review system performance consistency"
            )

        analysis_result["optimization_recommendations"] = recommendations

        # Cache result for future use
        self.statistical_cache[cache_key] = analysis_result

        return analysis_result

    def get_action_performance_analysis(
        self, include_statistical_tests: bool = False
    ) -> dict:
        """
        Action-specific performance analysis identifying performance variations.

        Args:
            include_statistical_tests (bool): Include statistical significance tests

        Returns:
            dict: Action-specific performance analysis with optimization recommendations
        """
        if not self.detailed_tracking_enabled:
            return {"error": "Detailed tracking not enabled for action analysis"}

        action_analysis = {
            "analysis_timestamp": time.time(),
            "action_performance": {},
            "performance_ranking": [],
            "statistical_significance": {},
        }

        # Analyze performance for each action
        action_means = {}
        for action_value, timings in self.action_specific_timings.items():
            if timings:
                action_name = Action(action_value).name
                mean_timing = statistics.mean(timings)
                action_means[action_name] = mean_timing

                action_analysis["action_performance"][action_name] = {
                    "measurement_count": len(timings),
                    "mean_timing_ms": mean_timing,
                    "median_timing_ms": statistics.median(timings),
                    "std_deviation": (
                        statistics.stdev(timings) if len(timings) > 1 else 0.0
                    ),
                    "min_timing_ms": min(timings),
                    "max_timing_ms": max(timings),
                }

        # Rank actions by performance
        if action_means:
            sorted_actions = sorted(action_means.items(), key=lambda x: x[1])
            action_analysis["performance_ranking"] = [
                {"action": action, "mean_timing_ms": timing}
                for action, timing in sorted_actions
            ]

            # Identify performance differences
            if len(sorted_actions) > 1:
                fastest = sorted_actions[0][1]
                slowest = sorted_actions[-1][1]
                performance_gap = slowest - fastest

                action_analysis["performance_summary"] = {
                    "fastest_action": sorted_actions[0][0],
                    "slowest_action": sorted_actions[-1][0],
                    "performance_gap_ms": performance_gap,
                    "relative_difference_percent": (performance_gap / fastest) * 100,
                }

        return action_analysis


class MemoryProfiler:
    """
    Advanced memory usage profiler for environment resource monitoring, memory leak detection,
    component-specific memory analysis, and resource optimization with continuous monitoring
    capabilities for comprehensive memory management.

    This class provides detailed memory profiling capabilities including continuous monitoring,
    leak detection, component-specific analysis, and resource optimization recommendations
    for comprehensive environment memory management and optimization.

    Attributes:
        sampling_interval (float): Memory sampling interval in seconds
        leak_detection_enabled (bool): Enable memory leak detection analysis
        component_tracking_enabled (bool): Enable component-specific memory tracking
        memory_samples (List[float]): Continuous memory usage sample storage
        component_memory_usage (dict): Component-specific memory usage tracking
        baseline_memory (Optional[float]): Baseline memory usage for leak detection
        monitoring_thread (threading.Thread): Background monitoring thread
        monitoring_active (bool): Monitoring thread activity status
    """

    def __init__(
        self,
        sampling_interval: float = MEMORY_SAMPLING_INTERVAL,
        enable_leak_detection: bool = True,
        track_component_usage: bool = False,
    ):
        """
        Initialize memory profiler with sampling configuration and monitoring infrastructure.

        Args:
            sampling_interval (float): Memory sampling interval in seconds
            enable_leak_detection (bool): Enable memory leak detection analysis
            track_component_usage (bool): Enable component-specific memory tracking
        """
        self.sampling_interval = sampling_interval
        self.leak_detection_enabled = enable_leak_detection
        self.component_tracking_enabled = track_component_usage
        self.memory_samples = []
        self.component_memory_usage = {} if track_component_usage else None
        self.baseline_memory = None
        self.monitoring_thread = None
        self.monitoring_active = False

    def start_continuous_monitoring(
        self, duration_seconds: Optional[float] = None
    ) -> None:
        """
        Start continuous memory monitoring with background thread sampling.

        Args:
            duration_seconds (Optional[float]): Monitoring duration in seconds
        """
        if self.monitoring_active:
            return  # Already monitoring

        # Establish baseline memory usage
        if self.leak_detection_enabled and self.baseline_memory is None:
            process = psutil.Process()
            self.baseline_memory = process.memory_info().rss / (
                1024 * 1024
            )  # Convert to MB

        self.monitoring_active = True

        def monitoring_loop():
            process = psutil.Process()
            start_time = time.time()

            while self.monitoring_active:
                try:
                    # Sample current memory usage
                    memory_mb = process.memory_info().rss / (1024 * 1024)
                    self.memory_samples.append(memory_mb)

                    # Check duration limit
                    if (
                        duration_seconds
                        and (time.time() - start_time) >= duration_seconds
                    ):
                        break

                    time.sleep(self.sampling_interval)

                except Exception:
                    break  # Exit on error

            self.monitoring_active = False

        self.monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        self.monitoring_thread.start()

    def analyze_memory_patterns(
        self, detect_leaks: bool = True, analyze_trends: bool = True
    ) -> dict:
        """
        Comprehensive memory pattern analysis with leak detection and optimization recommendations.

        Args:
            detect_leaks (bool): Enable memory leak detection analysis
            analyze_trends (bool): Enable memory usage trend analysis

        Returns:
            dict: Memory pattern analysis with leak detection and optimization guidance
        """
        if not self.memory_samples:
            return {"error": "No memory samples available for analysis"}

        memory_array = np.array(self.memory_samples)

        analysis_result = {
            "analysis_timestamp": time.time(),
            "sample_count": len(memory_array),
            "sampling_duration_seconds": len(memory_array) * self.sampling_interval,
            "memory_statistics": {
                "peak_usage_mb": float(np.max(memory_array)),
                "minimum_usage_mb": float(np.min(memory_array)),
                "mean_usage_mb": float(np.mean(memory_array)),
                "median_usage_mb": float(np.median(memory_array)),
                "std_deviation_mb": float(np.std(memory_array)),
                "memory_range_mb": float(np.max(memory_array) - np.min(memory_array)),
            },
        }

        # Memory leak detection analysis
        if detect_leaks and self.leak_detection_enabled:
            initial_memory = memory_array[0]
            final_memory = memory_array[-1]
            memory_growth = final_memory - initial_memory

            # Calculate linear trend
            time_points = np.arange(len(memory_array))
            slope, intercept = np.polyfit(time_points, memory_array, 1)

            analysis_result["leak_detection"] = {
                "baseline_memory_mb": self.baseline_memory,
                "initial_sample_mb": initial_memory,
                "final_sample_mb": final_memory,
                "total_growth_mb": memory_growth,
                "growth_rate_mb_per_sample": slope,
                "leak_detected": memory_growth > MEMORY_LEAK_THRESHOLD_MB,
                "leak_severity": (
                    "high"
                    if memory_growth > MEMORY_LEAK_THRESHOLD_MB * 2
                    else (
                        "moderate"
                        if memory_growth > MEMORY_LEAK_THRESHOLD_MB
                        else "none"
                    )
                ),
            }

        # Memory usage trend analysis
        if analyze_trends:
            # Detect memory usage patterns
            memory_diff = np.diff(memory_array)
            increasing_trend = (
                np.sum(memory_diff > 0) / len(memory_diff)
                if len(memory_diff) > 0
                else 0
            )

            analysis_result["trend_analysis"] = {
                "increasing_trend_percentage": increasing_trend * 100,
                "average_change_per_sample": (
                    float(np.mean(memory_diff)) if len(memory_diff) > 0 else 0
                ),
                "max_increase_mb": (
                    float(np.max(memory_diff)) if len(memory_diff) > 0 else 0
                ),
                "max_decrease_mb": (
                    float(np.min(memory_diff)) if len(memory_diff) > 0 else 0
                ),
                "trend_classification": (
                    "increasing"
                    if increasing_trend > 0.6
                    else "stable" if 0.4 <= increasing_trend <= 0.6 else "decreasing"
                ),
            }

        # Generate optimization recommendations
        recommendations = []
        peak_memory = analysis_result["memory_statistics"]["peak_usage_mb"]

        if peak_memory > MEMORY_LIMIT_TOTAL_MB:
            recommendations.append(
                f"Peak memory usage ({peak_memory:.1f}MB) exceeds target ({MEMORY_LIMIT_TOTAL_MB}MB)"
            )

        if (
            detect_leaks
            and "leak_detection" in analysis_result
            and analysis_result["leak_detection"]["leak_detected"]
        ):
            recommendations.append(
                "Memory leak detected - investigate resource cleanup"
            )

        if analysis_result["memory_statistics"]["std_deviation_mb"] > 5.0:
            recommendations.append(
                "High memory usage variability - optimize memory allocation patterns"
            )

        analysis_result["optimization_recommendations"] = recommendations

        return analysis_result

    def stop_monitoring(self) -> dict:
        """
        Stop continuous memory monitoring and finalize analysis.

        Returns:
            dict: Final memory monitoring results with analysis and recommendations
        """
        self.monitoring_active = False

        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=2.0)  # Wait up to 2 seconds

        # Perform final analysis
        final_analysis = self.analyze_memory_patterns(
            detect_leaks=self.leak_detection_enabled, analyze_trends=True
        )

        final_analysis["monitoring_completed"] = time.time()
        final_analysis["total_samples_collected"] = len(self.memory_samples)

        return final_analysis


class ScalabilityAnalyzer:
    """
    Environment scalability analysis utility for performance scaling assessment across
    different grid sizes, memory scaling validation, and configuration optimization
    recommendations with mathematical modeling for large-scale deployment planning.

    This class provides comprehensive scalability analysis including grid size performance
    scaling, memory usage patterns, mathematical modeling, and configuration optimization
    recommendations for deployment planning and system sizing decisions.

    Attributes:
        grid_size_range (List[tuple]): Grid sizes for systematic scaling analysis
        memory_modeling_enabled (bool): Enable mathematical memory modeling
        performance_prediction_enabled (bool): Enable performance prediction modeling
        scaling_measurements (dict): Grid-size-specific performance measurements
        performance_models (dict): Mathematical performance models and coefficients
        optimization_recommendations (dict): Configuration optimization guidance
    """

    def __init__(
        self,
        grid_size_range: List[tuple],
        enable_memory_modeling: bool = True,
        enable_performance_prediction: bool = False,
    ):
        """
        Initialize scalability analyzer with grid size range and modeling configuration.

        Args:
            grid_size_range (List[tuple]): Grid sizes for scaling analysis
            enable_memory_modeling (bool): Enable memory scaling mathematical modeling
            enable_performance_prediction (bool): Enable performance prediction capabilities
        """
        self.grid_size_range = grid_size_range
        self.memory_modeling_enabled = enable_memory_modeling
        self.performance_prediction_enabled = enable_performance_prediction
        self.scaling_measurements = {}
        self.performance_models = {}
        self.optimization_recommendations = {}

    def execute_scaling_analysis(
        self, iterations_per_size: int = 100, validate_resource_limits: bool = True
    ) -> dict:
        """
        Execute comprehensive scaling analysis across grid size range.

        Args:
            iterations_per_size (int): Benchmark iterations for each grid size
            validate_resource_limits (bool): Validate resource limits for each configuration

        Returns:
            dict: Scaling analysis results with performance trends and recommendations
        """
        scaling_results = {
            "analysis_timestamp": time.time(),
            "grid_sizes_tested": self.grid_size_range,
            "iterations_per_size": iterations_per_size,
            "scaling_measurements": {},
            "performance_trends": {},
            "resource_validation": {},
            "optimization_recommendations": [],
        }

        for grid_size in self.grid_size_range:
            width, height = grid_size
            grid_key = f"{width}x{height}"

            try:
                # Create environment with specific grid size
                env = create_plume_search_env(
                    grid_size=grid_size,
                    max_steps=100,  # Short episodes for scaling analysis
                )

                # Measure step latency for this grid size
                step_timings = []
                memory_usage = []

                # Initialize memory monitoring
                process = psutil.Process()
                initial_memory = process.memory_info().rss / (1024 * 1024)

                # Run benchmark iterations
                obs, info = env.reset()

                for i in range(iterations_per_size):
                    action = env.action_space.sample()

                    start_time = time.perf_counter()
                    obs, reward, terminated, truncated, info = env.step(action)
                    step_time = (
                        time.perf_counter() - start_time
                    ) * 1000  # Convert to ms

                    step_timings.append(step_time)

                    # Sample memory usage periodically
                    if i % 10 == 0:
                        current_memory = process.memory_info().rss / (1024 * 1024)
                        memory_usage.append(current_memory - initial_memory)

                    if terminated or truncated:
                        obs, info = env.reset()

                # Calculate scaling metrics
                mean_step_time = statistics.mean(step_timings)
                peak_memory = max(memory_usage) if memory_usage else 0
                total_cells = width * height

                scaling_results["scaling_measurements"][grid_key] = {
                    "grid_dimensions": grid_size,
                    "total_cells": total_cells,
                    "mean_step_time_ms": mean_step_time,
                    "peak_memory_mb": peak_memory,
                    "memory_per_cell_bytes": (
                        (peak_memory * 1024 * 1024) / total_cells
                        if total_cells > 0
                        else 0
                    ),
                    "step_time_per_cell_ns": (
                        (mean_step_time * 1000) / total_cells if total_cells > 0 else 0
                    ),
                }

                # Validate resource limits if requested
                if validate_resource_limits:
                    memory_limit_exceeded = peak_memory > MEMORY_LIMIT_TOTAL_MB
                    latency_limit_exceeded = (
                        mean_step_time > PERFORMANCE_TARGET_STEP_LATENCY_MS
                    )

                    scaling_results["resource_validation"][grid_key] = {
                        "memory_compliant": not memory_limit_exceeded,
                        "latency_compliant": not latency_limit_exceeded,
                        "overall_compliant": not (
                            memory_limit_exceeded or latency_limit_exceeded
                        ),
                    }

                env.close()

            except Exception as e:
                scaling_results["scaling_measurements"][grid_key] = {
                    "error": str(e),
                    "analysis_failed": True,
                }

        # Analyze scaling trends
        if (
            len(
                [
                    m
                    for m in scaling_results["scaling_measurements"].values()
                    if "error" not in m
                ]
            )
            >= 2
        ):
            scaling_results["performance_trends"] = self._analyze_scaling_trends(
                scaling_results["scaling_measurements"]
            )

        # Generate optimization recommendations
        scaling_results["optimization_recommendations"] = (
            self._generate_scaling_recommendations(scaling_results)
        )

        return scaling_results

    def _analyze_scaling_trends(self, measurements: dict) -> dict:
        """
        Analyze performance scaling trends across grid sizes.

        Args:
            measurements (dict): Scaling measurements across grid sizes

        Returns:
            dict: Scaling trend analysis with mathematical modeling
        """
        # Extract data for trend analysis
        cell_counts = []
        step_times = []
        memory_usage = []

        for grid_key, data in measurements.items():
            if "error" not in data:
                cell_counts.append(data["total_cells"])
                step_times.append(data["mean_step_time_ms"])
                memory_usage.append(data["peak_memory_mb"])

        if len(cell_counts) < 2:
            return {"error": "Insufficient data for trend analysis"}

        # Convert to numpy arrays for analysis
        cells = np.array(cell_counts)
        times = np.array(step_times)
        memory = np.array(memory_usage)

        trend_analysis = {
            "latency_scaling": {},
            "memory_scaling": {},
            "efficiency_metrics": {},
        }

        # Analyze step latency scaling (expect roughly O(n) for grid operations)
        if len(cells) >= 2:
            latency_slope, latency_intercept = np.polyfit(cells, times, 1)
            latency_r_squared = np.corrcoef(cells, times)[0, 1] ** 2

            trend_analysis["latency_scaling"] = {
                "slope_ms_per_cell": float(latency_slope),
                "intercept_ms": float(latency_intercept),
                "r_squared": float(latency_r_squared),
                "scaling_classification": self._classify_scaling_behavior(
                    latency_slope, cells, times
                ),
            }

        # Analyze memory scaling (expect roughly O(n) for grid storage)
        if len(cells) >= 2:
            memory_slope, memory_intercept = np.polyfit(cells, memory, 1)
            memory_r_squared = np.corrcoef(cells, memory)[0, 1] ** 2

            trend_analysis["memory_scaling"] = {
                "slope_mb_per_cell": float(memory_slope),
                "intercept_mb": float(memory_intercept),
                "r_squared": float(memory_r_squared),
                "scaling_classification": self._classify_scaling_behavior(
                    memory_slope, cells, memory
                ),
            }

        # Calculate efficiency metrics
        if len(cells) >= 2:
            min_cells, max_cells = min(cells), max(cells)
            min_idx, max_idx = np.argmin(cells), np.argmax(cells)

            scaling_factor = max_cells / min_cells
            latency_factor = times[max_idx] / times[min_idx]
            memory_factor = memory[max_idx] / memory[min_idx]

            trend_analysis["efficiency_metrics"] = {
                "grid_size_scaling_factor": float(scaling_factor),
                "latency_scaling_factor": float(latency_factor),
                "memory_scaling_factor": float(memory_factor),
                "latency_efficiency": float(scaling_factor / latency_factor),
                "memory_efficiency": float(scaling_factor / memory_factor),
            }

        return trend_analysis

    def _classify_scaling_behavior(
        self, slope: float, x_values: np.ndarray, y_values: np.ndarray
    ) -> str:
        """
        Classify scaling behavior based on slope and correlation.

        Args:
            slope (float): Linear regression slope
            x_values (np.ndarray): Independent variable values
            y_values (np.ndarray): Dependent variable values

        Returns:
            str: Scaling behavior classification
        """
        r_squared = np.corrcoef(x_values, y_values)[0, 1] ** 2

        if r_squared < 0.5:
            return "irregular"
        elif abs(slope) < 1e-6:
            return "constant"
        elif slope > 0:
            if slope < 1e-5:
                return "sub_linear"
            elif slope < 1e-3:
                return "linear"
            else:
                return "super_linear"
        else:
            return "decreasing"

    def _generate_scaling_recommendations(self, scaling_results: dict) -> List[str]:
        """
        Generate optimization recommendations based on scaling analysis.

        Args:
            scaling_results (dict): Complete scaling analysis results

        Returns:
            List[str]: Actionable optimization recommendations
        """
        recommendations = []
        measurements = scaling_results["scaling_measurements"]

        # Find optimal grid size based on resource constraints
        compliant_sizes = []
        for grid_key, validation in scaling_results.get(
            "resource_validation", {}
        ).items():
            if validation.get("overall_compliant", False):
                compliant_sizes.append(grid_key)

        if compliant_sizes:
            recommendations.append(
                f"Recommended compliant grid sizes: {', '.join(compliant_sizes)}"
            )
        else:
            recommendations.append(
                "No grid sizes meet all performance targets - consider optimization"
            )

        # Analyze performance trends for recommendations
        trends = scaling_results.get("performance_trends", {})

        if "latency_scaling" in trends:
            latency_classification = trends["latency_scaling"].get(
                "scaling_classification", ""
            )
            if latency_classification == "super_linear":
                recommendations.append(
                    "Step latency scales poorly - optimize core algorithms"
                )
            elif latency_classification == "linear":
                recommendations.append(
                    "Step latency scales linearly - acceptable for moderate grid sizes"
                )

        if "memory_scaling" in trends:
            memory_classification = trends["memory_scaling"].get(
                "scaling_classification", ""
            )
            if memory_classification == "super_linear":
                recommendations.append(
                    "Memory usage scales poorly - implement memory optimization"
                )

        # Specific grid size recommendations
        for grid_key, data in measurements.items():
            if "error" not in data:
                if data["mean_step_time_ms"] > PERFORMANCE_TARGET_STEP_LATENCY_MS * 2:
                    recommendations.append(
                        f"Grid size {grid_key} has poor latency performance"
                    )
                if data["peak_memory_mb"] > MEMORY_LIMIT_TOTAL_MB:
                    recommendations.append(
                        f"Grid size {grid_key} exceeds memory limits"
                    )

        return recommendations


class EnvironmentPerformanceSuite:
    """
    Comprehensive environment performance testing suite orchestrating execution of all
    performance benchmarks including step latency, episode timing, memory profiling, and
    scalability analysis with integrated validation, statistical analysis, and optimization
    recommendations for systematic performance evaluation.

    This class provides a complete performance testing framework coordinating all benchmark
    components including timing analysis, memory profiling, scalability assessment, and
    comprehensive reporting with optimization recommendations for development teams.

    Attributes:
        config (EnvironmentBenchmarkConfig): Benchmark execution configuration
        detailed_logging_enabled (bool): Enable detailed execution logging
        output_directory (Optional[pathlib.Path]): Output directory for benchmark results
        logger (logging.Logger): Suite execution logger
        timing_analyzer (TimingAnalyzer): High-precision timing analysis component
        memory_profiler (MemoryProfiler): Memory usage profiling component
        scalability_analyzer (ScalabilityAnalyzer): Grid size scaling analysis component
        benchmark_history (dict): Historical benchmark execution data
        performance_baselines (dict): Performance baseline data for regression detection
    """

    def __init__(
        self,
        config: Optional[EnvironmentBenchmarkConfig] = None,
        enable_detailed_logging: bool = False,
        output_directory: Optional[str] = None,
    ):
        """
        Initialize environment performance suite with configuration and component setup.

        Args:
            config (Optional[EnvironmentBenchmarkConfig]): Benchmark configuration
            enable_detailed_logging (bool): Enable detailed execution logging
            output_directory (Optional[str]): Directory for benchmark output files
        """
        self.config = config if config is not None else EnvironmentBenchmarkConfig()
        self.detailed_logging_enabled = enable_detailed_logging
        self.output_directory = (
            pathlib.Path(output_directory) if output_directory else None
        )

        # Configure logging
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        if enable_detailed_logging:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
            )
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.DEBUG)

        # Initialize analysis components
        self.timing_analyzer = TimingAnalyzer(
            enable_detailed_tracking=self.config.include_action_analysis
            or self.config.include_position_analysis
        )
        self.memory_profiler = MemoryProfiler(
            enable_leak_detection=self.config.detect_memory_leaks,
            track_component_usage=True,
        )

        if self.config.enable_scaling_analysis:
            self.scalability_analyzer = ScalabilityAnalyzer(
                grid_size_range=self.config.scaling_grid_sizes,
                enable_memory_modeling=True,
                enable_performance_prediction=True,
            )
        else:
            self.scalability_analyzer = None

        # Initialize performance tracking
        self.benchmark_history = {}
        self.performance_baselines = {}

        # Create output directory if specified
        if self.output_directory:
            try:
                self.output_directory.mkdir(parents=True, exist_ok=True)
                self.logger.info(f"Benchmark output directory: {self.output_directory}")
            except Exception as e:
                self.logger.warning(f"Failed to create output directory: {e}")

    def run_full_benchmark_suite(
        self,
        validate_targets: bool = True,
        include_scaling_analysis: bool = None,
        custom_targets: Optional[dict] = None,
    ) -> BenchmarkResult:
        """
        Execute comprehensive benchmark suite including all performance categories.

        Args:
            validate_targets (bool): Enable performance target validation
            include_scaling_analysis (bool): Override scaling analysis configuration
            custom_targets (Optional[dict]): Custom performance targets for validation

        Returns:
            BenchmarkResult: Comprehensive suite execution results with integrated analysis
        """
        suite_start_time = time.time()

        # Override scaling analysis setting if specified
        if include_scaling_analysis is not None:
            original_scaling = self.config.enable_scaling_analysis
            self.config.enable_scaling_analysis = include_scaling_analysis

        # Create benchmark result container
        benchmark_result = BenchmarkResult(
            benchmark_name=f"EnvironmentPerformanceSuite_{int(suite_start_time)}",
            execution_timestamp=suite_start_time,
            config=self.config,
        )

        self.logger.info("Starting comprehensive performance benchmark suite")

        try:
            # Execute step latency benchmarking
            self.logger.info("Executing step latency benchmark")
            step_latency_results = self._execute_step_latency_benchmark()
            benchmark_result.step_latency_metrics = step_latency_results

            # Execute episode performance benchmarking
            self.logger.info("Executing episode performance benchmark")
            episode_results = self._execute_episode_benchmark()
            benchmark_result.episode_performance_metrics = episode_results

            # Execute memory usage benchmarking
            if self.config.enable_memory_profiling:
                self.logger.info("Executing memory usage benchmark")
                memory_results = self._execute_memory_benchmark()
                benchmark_result.memory_usage_metrics = memory_results

            # Execute scalability analysis
            if self.config.enable_scaling_analysis and self.scalability_analyzer:
                self.logger.info("Executing scalability analysis")
                scaling_results = self.scalability_analyzer.execute_scaling_analysis(
                    iterations_per_size=min(100, self.config.iterations // 10),
                    validate_resource_limits=True,
                )
                benchmark_result.scalability_metrics = scaling_results

            # Validate performance targets if requested
            if validate_targets:
                targets = (
                    custom_targets
                    if custom_targets
                    else self.config.performance_targets
                )
                validation_results = benchmark_result.validate_against_target(
                    performance_targets=targets, strict_validation=True
                )
                self.logger.info(
                    f"Target validation completed: {validation_results.get('overall_compliance', False)}"
                )

            # Generate optimization recommendations
            recommendations = self._generate_suite_recommendations(benchmark_result)
            benchmark_result.optimization_recommendations.extend(recommendations)

            # Generate executive summary
            benchmark_result.generate_executive_summary(
                include_recommendations=True,
                highlight_issues=not benchmark_result.targets_met,
            )

            # Update benchmark history
            self.benchmark_history[benchmark_result.benchmark_name] = benchmark_result

            # Save results if output directory specified
            if self.output_directory:
                self._save_benchmark_results(benchmark_result)

            execution_time = time.time() - suite_start_time
            self.logger.info(
                f"Benchmark suite completed in {execution_time:.2f} seconds"
            )

            return benchmark_result

        except Exception as e:
            self.logger.error(f"Benchmark suite execution failed: {e}")
            benchmark_result.optimization_recommendations.append(
                f"Suite execution failed: {e}"
            )
            return benchmark_result

        finally:
            # Restore original scaling analysis setting
            if include_scaling_analysis is not None:
                self.config.enable_scaling_analysis = original_scaling

    def _execute_step_latency_benchmark(self) -> dict:
        """
        Execute comprehensive step latency benchmarking with statistical analysis.

        Returns:
            dict: Step latency benchmark results with statistical analysis
        """
        # Create environment for benchmarking
        env = create_plume_search_env()

        try:
            # Run benchmark with warmup
            obs, info = env.reset(seed=42)  # Deterministic seed for reproducibility

            # Warmup iterations
            for _ in range(self.config.warmup_iterations):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    obs, info = env.reset()

            # Actual benchmark iterations
            for i in range(self.config.iterations):
                action = env.action_space.sample()

                # Measure step latency with high precision
                start_time = time.perf_counter()
                obs, reward, terminated, truncated, info = env.step(action)
                step_time = (time.perf_counter() - start_time) * 1000  # Convert to ms

                # Record timing measurement
                self.timing_analyzer.record_step_timing(
                    timing_value=step_time,
                    action=action if self.config.include_action_analysis else None,
                    position=None,  # Would need agent position from environment state
                )

                if terminated or truncated:
                    obs, info = env.reset()

            # Analyze timing distribution
            timing_analysis = self.timing_analyzer.analyze_timing_distribution(
                include_outlier_analysis=True,
                confidence_level=STATISTICAL_CONFIDENCE_LEVEL,
            )

            # Add action-specific analysis if enabled
            if self.config.include_action_analysis:
                action_analysis = self.timing_analyzer.get_action_performance_analysis(
                    include_statistical_tests=True
                )
                timing_analysis["action_analysis"] = action_analysis

            timing_analysis["timings"] = self.timing_analyzer.timing_measurements
            return timing_analysis

        finally:
            env.close()

    def _execute_episode_benchmark(self) -> dict:
        """
        Execute episode-level performance benchmarking.

        Returns:
            dict: Episode performance benchmark results
        """
        env = create_plume_search_env(
            max_steps=100
        )  # Shorter episodes for benchmarking

        try:
            episode_durations = []
            reset_times = []

            num_episodes = min(
                50, self.config.iterations // 20
            )  # Reasonable episode count

            for episode in range(num_episodes):
                # Measure reset time
                reset_start = time.perf_counter()
                obs, info = env.reset(seed=episode + 42)
                reset_time = (time.perf_counter() - reset_start) * 1000
                reset_times.append(reset_time)

                # Run episode
                episode_start = time.time()
                step_count = 0

                while True:
                    action = env.action_space.sample()
                    obs, reward, terminated, truncated, info = env.step(action)
                    step_count += 1

                    if terminated or truncated:
                        break

                episode_duration = time.time() - episode_start
                episode_durations.append(episode_duration)

            return {
                "episode_durations": episode_durations,
                "reset_times": reset_times,
                "mean_episode_duration": statistics.mean(episode_durations),
                "mean_reset_time": statistics.mean(reset_times),
                "total_episodes": len(episode_durations),
            }

        finally:
            env.close()

    def _execute_memory_benchmark(self) -> dict:
        """
        Execute memory usage benchmarking with leak detection.

        Returns:
            dict: Memory usage benchmark results
        """
        # Start continuous memory monitoring
        self.memory_profiler.start_continuous_monitoring(duration_seconds=30.0)

        env = create_plume_search_env()

        try:
            # Run environment operations while monitoring memory
            obs, info = env.reset()

            for i in range(min(1000, self.config.iterations)):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)

                if terminated or truncated:
                    obs, info = env.reset()

                # Trigger occasional garbage collection for leak detection
                if i % 100 == 0:
                    gc.collect()

        finally:
            env.close()

        # Stop monitoring and analyze results
        memory_analysis = self.memory_profiler.stop_monitoring()
        return memory_analysis

    def _generate_suite_recommendations(
        self, benchmark_result: BenchmarkResult
    ) -> List[str]:
        """
        Generate comprehensive optimization recommendations based on suite results.

        Args:
            benchmark_result (BenchmarkResult): Complete benchmark results

        Returns:
            List[str]: Comprehensive optimization recommendations
        """
        recommendations = []

        # Step latency recommendations
        if "performance_validation" in benchmark_result.step_latency_metrics:
            validation = benchmark_result.step_latency_metrics["performance_validation"]
            if not validation.get("meets_target", True):
                recommendations.append(
                    "Optimize step execution for <1ms latency target"
                )

        # Memory usage recommendations
        if benchmark_result.memory_usage_metrics:
            peak_memory = benchmark_result.memory_usage_metrics.get(
                "memory_statistics", {}
            ).get("peak_usage_mb", 0)
            if peak_memory > MEMORY_LIMIT_TOTAL_MB:
                recommendations.append(
                    f"Reduce memory usage: {peak_memory:.1f}MB exceeds {MEMORY_LIMIT_TOTAL_MB}MB limit"
                )

        # Scaling recommendations
        if benchmark_result.scalability_metrics:
            scaling_recs = benchmark_result.scalability_metrics.get(
                "optimization_recommendations", []
            )
            recommendations.extend(scaling_recs)

        return recommendations

    def _save_benchmark_results(self, benchmark_result: BenchmarkResult) -> None:
        """
        Save benchmark results to output directory in specified formats.

        Args:
            benchmark_result (BenchmarkResult): Benchmark results to save
        """
        if not self.output_directory:
            return

        base_filename = f"benchmark_{benchmark_result.benchmark_name}"

        for output_format in self.config.output_formats:
            try:
                if output_format == "json":
                    filepath = self.output_directory / f"{base_filename}.json"
                    with open(filepath, "w") as f:
                        json.dump(benchmark_result.to_dict(), f, indent=2)

                elif output_format == "txt":
                    filepath = self.output_directory / f"{base_filename}.txt"
                    with open(filepath, "w") as f:
                        if benchmark_result.executive_summary:
                            f.write(benchmark_result.executive_summary)

                self.logger.info(f"Saved benchmark results: {filepath}")

            except Exception as e:
                self.logger.warning(
                    f"Failed to save results in {output_format} format: {e}"
                )


def run_environment_performance_benchmark(
    config: Optional[EnvironmentBenchmarkConfig] = None,
    validate_targets: bool = True,
    include_scaling_analysis: bool = False,
    output_path: Optional[str] = None,
) -> BenchmarkResult:
    """
    Primary benchmark function executing comprehensive environment performance analysis
    including step latency measurement, episode timing analysis, memory profiling, and
    scalability assessment with statistical validation and optimization recommendations.

    Args:
        config (Optional[EnvironmentBenchmarkConfig]): Benchmark configuration parameters
        validate_targets (bool): Enable performance target validation
        include_scaling_analysis (bool): Enable grid size scaling analysis
        output_path (Optional[str]): Output directory for benchmark results

    Returns:
        BenchmarkResult: Comprehensive benchmark result with performance analysis and recommendations
    """
    # Use default configuration if none provided
    if config is None:
        config = EnvironmentBenchmarkConfig()
        config.enable_scaling_analysis = include_scaling_analysis

    # Create performance suite and execute benchmark
    suite = EnvironmentPerformanceSuite(
        config=config, enable_detailed_logging=True, output_directory=output_path
    )

    return suite.run_full_benchmark_suite(
        validate_targets=validate_targets,
        include_scaling_analysis=include_scaling_analysis,
    )


def benchmark_step_latency(
    env: PlumeSearchEnv,
    iterations: int = 1000,
    analyze_by_action: bool = False,
    analyze_by_position: bool = False,
) -> dict:
    """
    Detailed step latency benchmarking function measuring environment step execution time.

    Args:
        env (PlumeSearchEnv): Environment instance for benchmarking
        iterations (int): Number of step iterations for statistical analysis
        analyze_by_action (bool): Enable action-specific performance analysis
        analyze_by_position (bool): Enable position impact assessment

    Returns:
        dict: Step latency analysis with timing statistics and target compliance validation
    """
    timing_analyzer = TimingAnalyzer(
        enable_detailed_tracking=analyze_by_action or analyze_by_position
    )

    obs, info = env.reset(seed=42)

    for i in range(iterations):
        action = env.action_space.sample()

        start_time = time.perf_counter()
        obs, reward, terminated, truncated, info = env.step(action)
        step_time = (time.perf_counter() - start_time) * 1000

        timing_analyzer.record_step_timing(
            timing_value=step_time,
            action=action if analyze_by_action else None,
            position=None,  # Would need agent position from environment
        )

        if terminated or truncated:
            obs, info = env.reset()

    return timing_analyzer.analyze_timing_distribution(
        include_outlier_analysis=True, confidence_level=STATISTICAL_CONFIDENCE_LEVEL
    )


def benchmark_episode_performance(
    env: PlumeSearchEnv,
    num_episodes: int = 20,
    include_reset_analysis: bool = True,
    max_episode_steps: Optional[int] = None,
) -> dict:
    """
    Episode-level performance benchmarking analyzing episode reset timing and completion performance.

    Args:
        env (PlumeSearchEnv): Environment instance for benchmarking
        num_episodes (int): Number of episodes for analysis
        include_reset_analysis (bool): Enable episode reset timing analysis
        max_episode_steps (Optional[int]): Maximum steps per episode

    Returns:
        dict: Episode performance analysis with timing statistics and resource utilization
    """
    episode_durations = []
    reset_times = []
    step_counts = []

    for episode in range(num_episodes):
        # Measure reset time if requested
        if include_reset_analysis:
            reset_start = time.perf_counter()
            obs, info = env.reset(seed=episode + 42)
            reset_time = (time.perf_counter() - reset_start) * 1000
            reset_times.append(reset_time)
        else:
            obs, info = env.reset(seed=episode + 42)

        # Run episode
        episode_start = time.time()
        step_count = 0

        while True:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            step_count += 1

            if terminated or truncated:
                break
            if max_episode_steps and step_count >= max_episode_steps:
                break

        episode_duration = time.time() - episode_start
        episode_durations.append(episode_duration)
        step_counts.append(step_count)

    results = {
        "episode_durations": episode_durations,
        "step_counts": step_counts,
        "mean_episode_duration": statistics.mean(episode_durations),
        "mean_steps_per_episode": statistics.mean(step_counts),
        "total_episodes": len(episode_durations),
    }

    if include_reset_analysis and reset_times:
        results["reset_times"] = reset_times
        results["mean_reset_time"] = statistics.mean(reset_times)

    return results


def benchmark_memory_usage(
    env: PlumeSearchEnv,
    monitoring_duration: int = 30,
    detect_leaks: bool = True,
    analyze_gc_patterns: bool = False,
) -> dict:
    """
    Comprehensive memory usage benchmarking with component-specific profiling and leak detection.

    Args:
        env (PlumeSearchEnv): Environment instance for benchmarking
        monitoring_duration (int): Memory monitoring duration in seconds
        detect_leaks (bool): Enable memory leak detection analysis
        analyze_gc_patterns (bool): Enable garbage collection pattern analysis

    Returns:
        dict: Memory usage analysis with component profiling and optimization recommendations
    """
    profiler = MemoryProfiler(
        enable_leak_detection=detect_leaks, track_component_usage=True
    )

    # Start continuous monitoring
    profiler.start_continuous_monitoring(duration_seconds=monitoring_duration)

    # Run environment operations during monitoring
    obs, info = env.reset()

    operation_count = 0
    start_time = time.time()

    while (time.time() - start_time) < monitoring_duration:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        operation_count += 1

        if terminated or truncated:
            obs, info = env.reset()

        # Trigger garbage collection periodically if analyzing GC patterns
        if analyze_gc_patterns and operation_count % 100 == 0:
            gc.collect()

    # Stop monitoring and get results
    results = profiler.stop_monitoring()
    results["total_operations"] = operation_count

    return results


def benchmark_rendering_performance(
    env: PlumeSearchEnv, iterations: int = 25, mode: str = "rgb_array"
) -> Dict[str, Any]:
    """Lightweight rendering benchmark used by the trimmed test harness."""

    if mode != "rgb_array":
        raise ValueError(
            "Only rgb_array rendering is supported in the educational benchmark"
        )

    render_times_ms: List[float] = []
    output_shape: Optional[Tuple[int, int, int]] = None

    for _ in range(max(1, iterations)):
        start_time = time.perf_counter()
        frame = env.render(mode="rgb_array")
        render_times_ms.append((time.perf_counter() - start_time) * 1000)
        if hasattr(frame, "shape"):
            output_shape = tuple(int(v) for v in frame.shape)

    return {
        "mode": mode,
        "frames_rendered": len(render_times_ms),
        "average_render_time_ms": statistics.mean(render_times_ms),
        "max_render_time_ms": max(render_times_ms),
        "min_render_time_ms": min(render_times_ms),
        "output_shape": output_shape,
    }


def analyze_scaling_performance(
    grid_sizes: List[tuple],
    iterations_per_size: int = 100,
    measure_memory_scaling: bool = True,
    validate_resource_limits: bool = True,
) -> dict:
    """
    Scalability performance analysis testing environment performance across different grid sizes.

    Args:
        grid_sizes (List[tuple]): Grid sizes for scalability testing
        iterations_per_size (int): Benchmark iterations for each grid size
        measure_memory_scaling (bool): Enable memory usage scaling analysis
        validate_resource_limits (bool): Validate resource limits for each configuration

    Returns:
        dict: Scaling analysis with performance trends and configuration recommendations
    """
    analyzer = ScalabilityAnalyzer(
        grid_size_range=grid_sizes,
        enable_memory_modeling=measure_memory_scaling,
        enable_performance_prediction=True,
    )

    return analyzer.execute_scaling_analysis(
        iterations_per_size=iterations_per_size,
        validate_resource_limits=validate_resource_limits,
    )


def validate_performance_targets(
    benchmark_results: BenchmarkResult,
    performance_targets: dict,
    strict_validation: bool = False,
    baseline_results: Optional[BenchmarkResult] = None,
) -> dict:
    """
    Performance target validation function comparing benchmark results against requirements.

    Args:
        benchmark_results (BenchmarkResult): Benchmark execution results
        performance_targets (dict): Performance targets for validation
        strict_validation (bool): Apply strict validation criteria
        baseline_results (Optional[BenchmarkResult]): Baseline results for regression detection

    Returns:
        dict: Performance validation results with compliance status and improvement recommendations
    """
    validation_result = benchmark_results.validate_against_target(
        performance_targets=performance_targets, strict_validation=strict_validation
    )

    # Add baseline comparison if provided
    if baseline_results:
        comparison = benchmark_results.compare_with_baseline(
            baseline_result=baseline_results,
            detect_regressions=True,
            regression_threshold=PERFORMANCE_REGRESSION_THRESHOLD,
        )
        validation_result["baseline_comparison"] = comparison

    return validation_result


def generate_performance_report(
    benchmark_results: BenchmarkResult,
    report_format: str = "text",
    include_technical_details: bool = True,
    output_path: Optional[pathlib.Path] = None,
) -> str:
    """
    Comprehensive performance report generation function creating detailed analysis documentation.

    Args:
        benchmark_results (BenchmarkResult): Benchmark execution results
        report_format (str): Report format specification (text, json, html)
        include_technical_details (bool): Include detailed technical analysis
        output_path (Optional[pathlib.Path]): Output file path for report

    Returns:
        str: Generated performance report content or file path
    """
    if report_format == "json":
        report_content = json.dumps(
            benchmark_results.to_dict(include_raw_data=False, include_analysis=True),
            indent=2,
        )
    else:
        # Generate text report
        report_parts = [
            "=" * 80,
            "ENVIRONMENT PERFORMANCE BENCHMARK REPORT",
            "=" * 80,
            "",
            benchmark_results.generate_executive_summary(
                include_recommendations=True, highlight_issues=True
            ),
        ]

        if include_technical_details:
            # Add detailed technical analysis
            if benchmark_results.step_latency_metrics:
                report_parts.extend(
                    [
                        "",
                        "-" * 40,
                        "STEP LATENCY ANALYSIS",
                        "-" * 40,
                        json.dumps(benchmark_results.step_latency_metrics, indent=2),
                    ]
                )

            if benchmark_results.memory_usage_metrics:
                report_parts.extend(
                    [
                        "",
                        "-" * 40,
                        "MEMORY USAGE ANALYSIS",
                        "-" * 40,
                        json.dumps(benchmark_results.memory_usage_metrics, indent=2),
                    ]
                )

        report_content = "\n".join(report_parts)

    # Save to file if output path specified
    if output_path:
        try:
            with open(output_path, "w") as f:
                f.write(report_content)
            return str(output_path)
        except Exception as e:
            return f"Error saving report: {e}"

    return report_content


class PerformanceReport:
    """
    Comprehensive performance report generation utility creating detailed documentation with
    executive summaries, technical analysis, and actionable recommendations for stakeholder
    communication and development guidance.

    This class provides comprehensive report generation capabilities including executive
    summaries, detailed technical analysis, optimization recommendations, and multiple
    output formats for stakeholder communication and development team guidance.

    Attributes:
        benchmark_result (BenchmarkResult): Benchmark execution results
        report_format (str): Report format specification (text, json, html)
        visualizations_enabled (bool): Enable chart and graph generation
        report_sections (dict): Organized report content structure
        optimization_priorities (List[str]): Prioritized optimization recommendations
        executive_summary (Optional[str]): Generated executive summary content
    """

    def __init__(
        self,
        benchmark_result: BenchmarkResult,
        report_format: str = "text",
        include_visualizations: bool = False,
    ):
        """
        Initialize performance report generator with benchmark results and format configuration.

        Args:
            benchmark_result (BenchmarkResult): Benchmark execution results
            report_format (str): Report format specification (text, json, html)
            include_visualizations (bool): Enable visualization generation
        """
        self.benchmark_result = benchmark_result
        self.report_format = report_format.lower()
        self.visualizations_enabled = include_visualizations
        self.report_sections = {}
        self.optimization_priorities = []
        self.executive_summary = None

    def generate_comprehensive_report(
        self,
        include_technical_details: bool = True,
        include_raw_data: bool = False,
        output_path: Optional[pathlib.Path] = None,
    ) -> str:
        """
        Generate complete performance report with comprehensive analysis and recommendations.

        Args:
            include_technical_details (bool): Include detailed technical analysis
            include_raw_data (bool): Include raw measurement data
            output_path (Optional[pathlib.Path]): Output file path for report

        Returns:
            str: Comprehensive formatted report content or file path
        """
        # Generate executive summary
        self.executive_summary = self.benchmark_result.generate_executive_summary(
            include_recommendations=True, highlight_issues=True
        )

        # Compile report sections
        self.report_sections = {
            "executive_summary": self.executive_summary,
            "benchmark_metadata": {
                "benchmark_name": self.benchmark_result.benchmark_name,
                "execution_timestamp": self.benchmark_result.execution_timestamp,
                "execution_date": time.ctime(self.benchmark_result.execution_timestamp),
                "configuration": self.benchmark_result.config.to_dict(),
            },
        }

        # Add performance analysis sections
        if include_technical_details:
            if self.benchmark_result.step_latency_metrics:
                self.report_sections["step_latency_analysis"] = (
                    self.benchmark_result.step_latency_metrics
                )

            if self.benchmark_result.episode_performance_metrics:
                self.report_sections["episode_performance_analysis"] = (
                    self.benchmark_result.episode_performance_metrics
                )

            if self.benchmark_result.memory_usage_metrics:
                self.report_sections["memory_usage_analysis"] = (
                    self.benchmark_result.memory_usage_metrics
                )

            if self.benchmark_result.scalability_metrics:
                self.report_sections["scalability_analysis"] = (
                    self.benchmark_result.scalability_metrics
                )

        # Add validation and recommendations
        self.report_sections["validation_results"] = (
            self.benchmark_result.validation_results
        )
        self.report_sections["optimization_recommendations"] = (
            self.benchmark_result.optimization_recommendations
        )

        # Format report according to specified format
        if self.report_format == "json":
            formatted_report = json.dumps(self.report_sections, indent=2)
        elif self.report_format == "html":
            formatted_report = self._generate_html_report()
        else:
            formatted_report = self._generate_text_report(include_raw_data)

        # Save to file if output path specified
        if output_path:
            try:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, "w") as f:
                    f.write(formatted_report)
                return str(output_path)
            except Exception as e:
                return f"Error saving report to {output_path}: {e}"

        return formatted_report

    def _generate_text_report(self, include_raw_data: bool = False) -> str:
        """
        Generate detailed text format report with comprehensive analysis.

        Args:
            include_raw_data (bool): Include raw measurement data

        Returns:
            str: Formatted text report
        """
        report_lines = [
            "=" * 100,
            "PLUME NAVIGATION ENVIRONMENT PERFORMANCE BENCHMARK REPORT",
            "=" * 100,
            "",
            f"Benchmark: {self.benchmark_result.benchmark_name}",
            f"Executed: {time.ctime(self.benchmark_result.execution_timestamp)}",
            f"Targets Met: {'YES' if self.benchmark_result.targets_met else 'NO'}",
            "",
            "EXECUTIVE SUMMARY",
            "-" * 50,
            self.executive_summary or "No executive summary available",
            "",
        ]

        # Add step latency analysis
        if "step_latency_analysis" in self.report_sections:
            latency_data = self.report_sections["step_latency_analysis"]

            report_lines.extend(["STEP LATENCY PERFORMANCE", "-" * 50])

            if "descriptive_statistics" in latency_data:
                stats = latency_data["descriptive_statistics"]
                report_lines.extend(
                    [
                        f"Mean Latency: {stats['mean']:.3f}ms",
                        f"Median Latency: {stats['median']:.3f}ms",
                        f"95th Percentile: {stats.get('p95', 'N/A')}ms",
                        f"Standard Deviation: {stats['std_deviation']:.3f}ms",
                        f"Min/Max: {stats['min_value']:.3f}ms / {stats['max_value']:.3f}ms",
                    ]
                )

            if "performance_validation" in latency_data:
                validation = latency_data["performance_validation"]
                status = "PASS" if validation.get("meets_target", False) else "FAIL"
                report_lines.append(f"Target Compliance: {status}")

            report_lines.append("")

        # Add memory usage analysis
        if "memory_usage_analysis" in self.report_sections:
            memory_data = self.report_sections["memory_usage_analysis"]

            report_lines.extend(["MEMORY USAGE ANALYSIS", "-" * 50])

            if "memory_statistics" in memory_data:
                stats = memory_data["memory_statistics"]
                report_lines.extend(
                    [
                        f"Peak Usage: {stats['peak_usage_mb']:.1f}MB",
                        f"Mean Usage: {stats['mean_usage_mb']:.1f}MB",
                        f"Memory Growth: {stats['memory_growth_mb']:.1f}MB",
                    ]
                )

            if "leak_detection" in memory_data:
                leak_info = memory_data["leak_detection"]
                leak_status = (
                    "DETECTED" if leak_info.get("leak_detected", False) else "NONE"
                )
                report_lines.append(f"Memory Leaks: {leak_status}")

            report_lines.append("")

        # Add optimization recommendations
        if self.benchmark_result.optimization_recommendations:
            report_lines.extend(["OPTIMIZATION RECOMMENDATIONS", "-" * 50])

            for i, rec in enumerate(
                self.benchmark_result.optimization_recommendations, 1
            ):
                report_lines.append(f"{i}. {rec}")

            report_lines.append("")

        # Add configuration details
        report_lines.extend(
            [
                "BENCHMARK CONFIGURATION",
                "-" * 50,
                f"Iterations: {self.benchmark_result.config.iterations}",
                f"Warmup Iterations: {self.benchmark_result.config.warmup_iterations}",
                f"Memory Profiling: {self.benchmark_result.config.enable_memory_profiling}",
                f"Scaling Analysis: {self.benchmark_result.config.enable_scaling_analysis}",
                "",
            ]
        )

        report_lines.extend(
            ["=" * 100, f"Report generated at: {time.ctime()}", "=" * 100]
        )

        return "\n".join(report_lines)

    def _generate_html_report(self) -> str:
        """
        Generate HTML format report with structured presentation.

        Returns:
            str: HTML formatted report
        """
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Performance Benchmark Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; }}
                h2 {{ color: #34495e; border-bottom: 1px solid #bdc3c7; }}
                .summary {{ background: #ecf0f1; padding: 20px; border-radius: 5px; }}
                .metrics {{ display: flex; flex-wrap: wrap; gap: 20px; }}
                .metric-box {{ background: white; padding: 15px; border: 1px solid #bdc3c7; border-radius: 5px; flex: 1; min-width: 200px; }}
                .pass {{ color: #27ae60; font-weight: bold; }}
                .fail {{ color: #e74c3c; font-weight: bold; }}
                .recommendations {{ background: #fff3cd; padding: 15px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <h1>Performance Benchmark Report</h1>

            <div class="summary">
                <h2>Executive Summary</h2>
                <p>{self.executive_summary or 'No summary available'}</p>
                <p><strong>Overall Status:</strong>
                   <span class="{'pass' if self.benchmark_result.targets_met else 'fail'}">
                   {'COMPLIANT' if self.benchmark_result.targets_met else 'NON-COMPLIANT'}
                   </span>
                </p>
            </div>

            <h2>Performance Metrics</h2>
            <div class="metrics">
        """

        # Add step latency metrics
        if "step_latency_analysis" in self.report_sections:
            latency_data = self.report_sections["step_latency_analysis"]
            if "descriptive_statistics" in latency_data:
                stats = latency_data["descriptive_statistics"]
                html_content += f"""
                <div class="metric-box">
                    <h3>Step Latency</h3>
                    <p><strong>Mean:</strong> {stats['mean']:.3f}ms</p>
                    <p><strong>Median:</strong> {stats['median']:.3f}ms</p>
                    <p><strong>95th Percentile:</strong> {stats.get('p95', 'N/A')}ms</p>
                    <p><strong>Range:</strong> {stats['min_value']:.3f} - {stats['max_value']:.3f}ms</p>
                </div>
                """

        # Add memory metrics
        if "memory_usage_analysis" in self.report_sections:
            memory_data = self.report_sections["memory_usage_analysis"]
            if "memory_statistics" in memory_data:
                stats = memory_data["memory_statistics"]
                html_content += f"""
                <div class="metric-box">
                    <h3>Memory Usage</h3>
                    <p><strong>Peak:</strong> {stats['peak_usage_mb']:.1f}MB</p>
                    <p><strong>Mean:</strong> {stats['mean_usage_mb']:.1f}MB</p>
                    <p><strong>Growth:</strong> {stats['memory_growth_mb']:.1f}MB</p>
                </div>
                """

        html_content += """
            </div>

            <h2>Recommendations</h2>
            <div class="recommendations">
                <ul>
        """

        for rec in self.benchmark_result.optimization_recommendations:
            html_content += f"<li>{rec}</li>"

        html_content += f"""
                </ul>
            </div>

            <footer>
                <p><small>Report generated: {time.ctime()}</small></p>
            </footer>
        </body>
        </html>
        """

        return html_content


def benchmark_rendering_performance(env: PlumeSearchEnv, num_renders: int = 50) -> dict:
    """Benchmark RGB-array rendering performance for a given environment.

    Measures multiple rgb_array renders and returns timing statistics and basic
    output-format validations. This utility is intentionally lightweight so it
    can be imported by tests without requiring optional rendering backends.

    Args:
        env (PlumeSearchEnv): Initialized environment instance.
        num_renders (int): Number of render calls to time.

    Returns:
        dict: Results containing timing stats and validation flags, e.g.::

            {
              'count': 50,
              'mean_ms': 1.2,
              'p95_ms': 2.5,
              'std_ms': 0.3,
              'format_compliant_rate': 1.0
            }
    """
    import statistics

    timings_ms = []
    compliant = 0

    try:
        env.reset(seed=42)
    except Exception:
        pass

    for i in range(max(1, num_renders)):
        try:
            if i > 0 and hasattr(env, "action_space"):
                env.step(env.action_space.sample())
        except Exception:
            pass

        start = time.perf_counter()
        try:
            arr = env.render(mode="rgb_array")
        finally:
            dt = (time.perf_counter() - start) * 1000.0
            timings_ms.append(dt)

        try:
            import numpy as _np

            ok = (
                arr is not None
                and hasattr(arr, "shape")
                and len(arr.shape) == 3
                and arr.shape[2] == 3
                and getattr(arr, "dtype", None) == _np.uint8
                and _np.all((arr >= 0) & (arr <= 255))
            )
            compliant += 1 if ok else 0
        except Exception:
            pass

    mean_ms = statistics.mean(timings_ms)
    std_ms = statistics.pstdev(timings_ms) if len(timings_ms) > 1 else 0.0
    try:
        import numpy as _np

        p95 = float(_np.percentile(timings_ms, 95))
    except Exception:
        p95 = max(timings_ms) if timings_ms else 0.0

    return {
        "count": len(timings_ms),
        "mean_ms": float(mean_ms),
        "p95_ms": float(p95),
        "std_ms": float(std_ms),
        "format_compliant_rate": (compliant / len(timings_ms)) if timings_ms else 0.0,
        "timings_ms": timings_ms,
    }


class PerformanceAnalysis:
    """High-level helper for post-benchmark analysis used by tests.

    Provides simple wrappers to analyze performance trends and generate
    optimization recommendations using the data structures produced by this
    module. The implementation intentionally reuses existing analyzers where
    possible and degrades gracefully if some metrics are missing.
    """

    def analyze_performance_trends(self, performance_data: dict) -> dict:
        """Analyze trends across step latency, memory, and scalability data.

        Args:
            performance_data (dict): Expected keys include 'step_latency',
                'memory_usage', and optionally 'scalability'.

        Returns:
            dict: Consolidated trend analysis with best-effort contents.
        """
        trends: Dict[str, Any] = {}

        scalability = performance_data.get("scalability") or {}
        measurements = scalability.get("scaling_measurements") or {}
        if measurements:
            grid_size_range = [
                tuple(value["grid_dimensions"])
                for value in measurements.values()
                if isinstance(value, dict) and "grid_dimensions" in value
            ]
            if not grid_size_range:
                grid_size_range = []
                for key in measurements.keys():
                    if isinstance(key, str) and "x" in key:
                        try:
                            width_str, height_str = key.split("x", 1)
                            grid_size_range.append((int(width_str), int(height_str)))
                        except (ValueError, TypeError):
                            continue

            analyzer = ScalabilityAnalyzer(
                grid_size_range=grid_size_range or [(32, 32)]
            )
            trends["scaling"] = analyzer._analyze_scaling_trends(measurements)

        step = performance_data.get("step_latency") or {}
        mem = performance_data.get("memory_usage") or {}

        def _summary(stats: dict, keys: list[str]) -> dict:
            return {k: stats.get(k) for k in keys if k in stats}

        if step:
            trends["step_latency_summary"] = _summary(
                step,
                [
                    "mean_step_time_ms",
                    "p95_step_time_ms",
                    "std_dev_step_time_ms",
                ],
            )
        if mem:
            trends["memory_summary"] = _summary(
                mem,
                [
                    "mean_memory_delta_mb",
                    "peak_memory_mb",
                    "memory_leak_suspected",
                ],
            )

        return trends

    def generate_optimization_recommendations(
        self,
        benchmark_results: "BenchmarkResult",
        trend_analysis: dict,
        include_scaling_guidance: bool = True,
    ) -> list[str]:
        """Generate human-readable optimization suggestions from results and trends.

        Args:
            benchmark_results: Completed BenchmarkResult from a suite run.
            trend_analysis: Output of analyze_performance_trends.
            include_scaling_guidance: Whether to include grid-scaling hints.

        Returns:
            list[str]: Recommendation strings.
        """
        recs: List[str] = []

        step = benchmark_results.step_latency_metrics or {}
        if step and step.get("mean_step_time_ms") is not None:
            mean = step["mean_step_time_ms"]
            target = benchmark_results.config.performance_targets.get(
                "step_latency_ms", PERFORMANCE_TARGET_STEP_LATENCY_MS
            )
            if mean > target:
                recs.append(
                    "Optimize action processing and state update loops to reduce step latency"
                )

        mem = benchmark_results.memory_usage_metrics or {}
        if mem and mem.get("peak_memory_mb") is not None:
            mem_limit = benchmark_results.config.performance_targets.get(
                "memory_limit_mb", MEMORY_LIMIT_TOTAL_MB
            )
            if mem.get("peak_memory_mb", 0) > mem_limit:
                recs.append(
                    "Reduce intermediate allocations or enable in-place operations to lower peak memory"
                )

        if include_scaling_guidance and "scaling" in trend_analysis:
            scaling = trend_analysis["scaling"]
            cls = scaling.get("latency_scaling", {}).get("scaling_classification")
            if cls == "super_linear":
                recs.append(
                    "Super-linear latency scaling detected; profile hotspots and improve algorithmic complexity"
                )

        return recs
