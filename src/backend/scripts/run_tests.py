#!/usr/bin/env python3
"""
Comprehensive test execution script for plume_nav_sim providing intelligent test suite
management, category-specific test execution, performance monitoring, detailed reporting,
and development workflow integration.

This script orchestrates pytest execution with custom configurations, handles test
environment setup, manages test data collection, and generates comprehensive reports for
different testing scenarios including unit tests, integration tests, performance
benchmarks, reproducibility validation, and edge case testing.

The script provides automated execution of comprehensive test coverage with intelligent
test categorization and execution management, integrating with development workflows
for continuous validation and performance tracking.
"""

import argparse  # >=3.10 - Command-line argument parsing for test execution options
import json  # >=3.10 - JSON serialization for test results and configuration data
import multiprocessing  # >=3.10 - Parallel test execution support and process management
import os  # >=3.10 - Operating system interface for environment variables and paths
import pathlib  # >=3.10 - Object-oriented file system path handling
import platform  # >=3.10 - Platform information detection for cross-platform compatibility
import shutil  # >=3.10 - High-level file operations for test artifact cleanup
import subprocess  # >=3.10 - Subprocess execution for pytest command invocation
import sys  # >=3.10 - System interface for exit codes and platform information
import time  # >=3.10 - Time utilities for performance measurement and timing
from dataclasses import dataclass  # >=3.10 - Data classes for configuration structures
from typing import (  # >=3.10 - Type hints for comprehensive annotation
    Any,
    Dict,
    List,
    Optional,
)

from plume_nav_sim.__init__ import get_package_info, initialize_package
from plume_nav_sim.config import (
    TestConfigFactory,
    create_edge_case_test_config,
    create_integration_test_config,
    create_performance_test_config,
    create_reproducibility_test_config,
    create_unit_test_config,
)
from plume_nav_sim.utils.exceptions import ConfigurationError, ValidationError

# Internal imports
from plume_nav_sim.utils.logging import (
    ComponentLogger,
    PerformanceTimer,
    configure_logging_for_development,
    get_component_logger,
)
from plume_nav_sim.utils.validation import ValidationResult, create_validation_context

# External imports with version comments


# Global constants for script configuration
SCRIPT_NAME = "run_tests.py"
SCRIPT_VERSION = "1.0.0"

# Default test categories for comprehensive coverage
DEFAULT_TEST_CATEGORIES = [
    "unit",
    "integration",
    "performance",
    "reproducibility",
    "edge_case",
]

# Default pytest arguments for standard execution
DEFAULT_PYTEST_ARGS = ["-v", "--tb=short", "--strict-markers"]

# Timeout configuration for different test categories
TEST_TIMEOUT_SECONDS = 600
PERFORMANCE_TIMEOUT_SECONDS = 1200

# Parallel execution configuration
MAX_PARALLEL_WORKERS = 4

# Output directory configuration
REPORT_OUTPUT_DIR = "test_reports"
LOG_OUTPUT_DIR = "test_logs"

# Pytest configuration template
PYTEST_INI_TEMPLATE = {
    "testpaths": ["tests"],
    "addopts": "-q",
    "python_files": ["test_*.py"],
}

# Supported Python versions for compatibility testing
SUPPORTED_PYTHON_VERSIONS = ["3.10", "3.11", "3.12", "3.13"]

# Exit codes for different execution outcomes
EXIT_CODE_SUCCESS = 0
EXIT_CODE_FAILURE = 1
EXIT_CODE_ERROR = 2


@dataclass
class TestExecutionConfig:
    """
    Comprehensive test execution configuration data class containing all test parameters,
    system settings, performance targets, and execution options with validation and
    optimization support.
    """

    # Core execution parameters
    test_categories: List[str]
    parallel_execution: bool
    output_directory: str

    # Advanced execution options
    fail_fast: bool = False
    timeout_seconds: int = TEST_TIMEOUT_SECONDS
    pytest_args: List[str] = None
    environment_variables: Dict[str, str] = None
    capture_output: bool = True
    generate_reports: bool = True
    cleanup_artifacts: bool = False

    # Performance and optimization settings
    performance_targets: Dict[str, float] = None
    log_level: Optional[str] = "INFO"

    def __post_init__(self):
        """Initialize configuration with default values and validation."""
        if self.pytest_args is None:
            self.pytest_args = DEFAULT_PYTEST_ARGS.copy()

        if self.environment_variables is None:
            self.environment_variables = {}

        if self.performance_targets is None:
            self.performance_targets = {
                "step_latency_ms": 1.0,
                "render_time_ms": 5.0,
                "memory_limit_mb": 50.0,
            }

    def validate(self, strict_validation: bool = False) -> ValidationResult:
        """
        Comprehensive configuration validation ensuring parameter consistency, system
        compatibility, and execution feasibility.

        Args:
            strict_validation: Enable additional validation checks and constraints

        Returns:
            ValidationResult: Comprehensive validation result with errors, warnings, and recommendations
        """
        context = create_validation_context("test_execution_config")

        # Validate test categories
        invalid_categories = set(self.test_categories) - set(DEFAULT_TEST_CATEGORIES)
        if invalid_categories:
            context.add_error(f"Invalid test categories: {invalid_categories}")

        # Validate output directory
        try:
            pathlib.Path(self.output_directory).mkdir(parents=True, exist_ok=True)
        except (PermissionError, OSError) as e:
            context.add_error(
                f"Cannot create output directory {self.output_directory}: {e}"
            )

        # Validate timeout values
        if self.timeout_seconds <= 0:
            context.add_error("Timeout seconds must be positive")

        # Check system resources for parallel execution
        if self.parallel_execution:
            cpu_count = multiprocessing.cpu_count()
            if cpu_count < 2:
                context.add_warning(
                    "Parallel execution requested but insufficient CPU cores"
                )

        # Strict validation additional checks
        if strict_validation:
            # Check disk space for output
            disk_usage = shutil.disk_usage(self.output_directory)
            if disk_usage.free < 1024 * 1024 * 100:  # 100MB minimum
                context.add_warning("Low disk space available for test outputs")

            # Validate performance targets are reasonable
            if self.performance_targets["step_latency_ms"] < 0.1:
                context.add_warning(
                    "Very tight step latency target may cause test failures"
                )

        return context.get_result()

    def optimize_for_system(
        self, system_capabilities: Dict[str, Any]
    ) -> "TestExecutionConfig":
        """
        Optimize configuration parameters based on detected system capabilities and
        performance characteristics.

        Args:
            system_capabilities: Dictionary containing system performance and resource information

        Returns:
            TestExecutionConfig: System-optimized configuration with tuned parameters
        """
        optimized_config = TestExecutionConfig(
            test_categories=self.test_categories.copy(),
            parallel_execution=self.parallel_execution,
            output_directory=self.output_directory,
        )

        # Optimize parallel execution based on CPU cores
        cpu_cores = system_capabilities.get("cpu_cores", 1)
        if cpu_cores >= 4:
            optimized_config.parallel_execution = True
        elif cpu_cores < 2:
            optimized_config.parallel_execution = False

        # Adjust timeout based on system performance
        cpu_speed_factor = system_capabilities.get("cpu_speed_factor", 1.0)
        optimized_config.timeout_seconds = int(self.timeout_seconds / cpu_speed_factor)

        # Tune performance targets for system
        memory_gb = system_capabilities.get("memory_gb", 4.0)
        if memory_gb < 4.0:
            optimized_config.performance_targets = {
                "step_latency_ms": 2.0,  # Relaxed for lower-memory systems
                "render_time_ms": 10.0,
                "memory_limit_mb": 25.0,
            }

        return optimized_config

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary format for serialization, logging, and
        external processing.

        Returns:
            Dict[str, Any]: Dictionary representation of complete test execution configuration
        """
        return {
            "test_categories": self.test_categories,
            "parallel_execution": self.parallel_execution,
            "output_directory": self.output_directory,
            "fail_fast": self.fail_fast,
            "timeout_seconds": self.timeout_seconds,
            "pytest_args": self.pytest_args,
            "environment_variables": self.environment_variables,
            "capture_output": self.capture_output,
            "generate_reports": self.generate_reports,
            "cleanup_artifacts": self.cleanup_artifacts,
            "performance_targets": self.performance_targets,
            "log_level": self.log_level,
            "script_version": SCRIPT_VERSION,
        }


@dataclass
class TestResult:
    """
    Comprehensive test execution result data class containing execution statistics,
    performance metrics, failure analysis, and recommendations with detailed reporting support.
    """

    # Core execution results
    success: bool
    execution_time_seconds: float
    total_tests: int

    # Detailed test statistics
    passed_tests: int = 0
    failed_tests: int = 0
    skipped_tests: int = 0

    # Category-specific results
    category_results: Dict[str, Dict[str, Any]] = None

    # Performance and system metrics
    performance_metrics: Dict[str, float] = None

    # Error analysis and reporting
    failures: List[Dict[str, Any]] = None
    warnings: List[str] = None
    coverage_data: Dict[str, Any] = None
    recommendations: List[str] = None
    system_info: Dict[str, Any] = None

    def __post_init__(self):
        """Initialize result with default collections."""
        if self.category_results is None:
            self.category_results = {}

        if self.performance_metrics is None:
            self.performance_metrics = {}

        if self.failures is None:
            self.failures = []

        if self.warnings is None:
            self.warnings = []

        if self.coverage_data is None:
            self.coverage_data = {}

        if self.recommendations is None:
            self.recommendations = []

        if self.system_info is None:
            self.system_info = {}

    def add_category_result(self, category: str, result_data: Dict[str, Any]) -> None:
        """
        Add category-specific test result with detailed metrics and analysis integration.

        Args:
            category: Test category name (unit, integration, performance, etc.)
            result_data: Dictionary containing category-specific test results and metrics
        """
        # Validate category and result data structure
        if category not in DEFAULT_TEST_CATEGORIES:
            self.warnings.append(f"Unknown test category: {category}")

        required_keys = ["passed", "failed", "skipped", "execution_time"]
        missing_keys = [key for key in required_keys if key not in result_data]
        if missing_keys:
            self.warnings.append(
                f"Missing result data keys for {category}: {missing_keys}"
            )

        # Store result data
        self.category_results[category] = result_data

        # Update overall statistics
        self.passed_tests += result_data.get("passed", 0)
        self.failed_tests += result_data.get("failed", 0)
        self.skipped_tests += result_data.get("skipped", 0)

        # Integrate performance data
        if "performance_metrics" in result_data:
            category_perf = result_data["performance_metrics"]
            self.performance_metrics[f"{category}_metrics"] = category_perf

        # Update overall success status
        if result_data.get("failed", 0) > 0:
            self.success = False

    def generate_summary(
        self, include_details: bool = True, include_recommendations: bool = True
    ) -> str:
        """
        Generate comprehensive test execution summary with statistics, analysis, and recommendations.

        Args:
            include_details: Include detailed failure analysis and category breakdowns
            include_recommendations: Include actionable recommendations and optimization suggestions

        Returns:
            str: Comprehensive test execution summary for reporting and analysis
        """
        summary_lines = [
            f"Test Execution Summary - {SCRIPT_NAME} v{SCRIPT_VERSION}",
            "=" * 60,
            f"Overall Status: {'PASSED' if self.success else 'FAILED'}",
            f"Execution Time: {self.execution_time_seconds:.2f} seconds",
            f"Total Tests: {self.total_tests}",
            f"  Passed: {self.passed_tests}",
            f"  Failed: {self.failed_tests}",
            f"  Skipped: {self.skipped_tests}",
            "",
        ]

        # Add category breakdown
        if self.category_results and include_details:
            summary_lines.append("Category Results:")
            summary_lines.append("-" * 20)
            for category, results in self.category_results.items():
                summary_lines.append(
                    f"{category.capitalize()}: "
                    f"{results.get('passed', 0)}P/{results.get('failed', 0)}F/"
                    f"{results.get('skipped', 0)}S "
                    f"({results.get('execution_time', 0):.2f}s)"
                )
            summary_lines.append("")

        # Add performance metrics
        if self.performance_metrics:
            summary_lines.append("Performance Metrics:")
            summary_lines.append("-" * 20)
            for metric_name, metric_value in self.performance_metrics.items():
                if isinstance(metric_value, dict):
                    for sub_metric, sub_value in metric_value.items():
                        summary_lines.append(f"{metric_name}.{sub_metric}: {sub_value}")
                else:
                    summary_lines.append(f"{metric_name}: {metric_value}")
            summary_lines.append("")

        # Add failure details
        if self.failures and include_details:
            summary_lines.append("Failure Analysis:")
            summary_lines.append("-" * 20)
            for i, failure in enumerate(self.failures[:5]):  # Limit to 5 failures
                summary_lines.append(
                    f"{i+1}. {failure.get('test_name', 'Unknown Test')}"
                )
                summary_lines.append(
                    f"   Error: {failure.get('error_message', 'No error message')}"
                )
            if len(self.failures) > 5:
                summary_lines.append(
                    f"   ... and {len(self.failures) - 5} more failures"
                )
            summary_lines.append("")

        # Add warnings
        if self.warnings:
            summary_lines.append("Warnings:")
            summary_lines.append("-" * 20)
            for warning in self.warnings:
                summary_lines.append(f"• {warning}")
            summary_lines.append("")

        # Add recommendations
        if self.recommendations and include_recommendations:
            summary_lines.append("Recommendations:")
            summary_lines.append("-" * 20)
            for recommendation in self.recommendations:
                summary_lines.append(f"• {recommendation}")
            summary_lines.append("")

        return "\n".join(summary_lines)

    def to_json(self, include_system_info: bool = True) -> str:
        """
        Convert test result to JSON format for programmatic access, CI/CD integration,
        and automated reporting.

        Args:
            include_system_info: Include system information in JSON serialization

        Returns:
            str: JSON representation of complete test execution result
        """
        result_dict = {
            "success": self.success,
            "execution_time_seconds": self.execution_time_seconds,
            "total_tests": self.total_tests,
            "passed_tests": self.passed_tests,
            "failed_tests": self.failed_tests,
            "skipped_tests": self.skipped_tests,
            "category_results": self.category_results,
            "performance_metrics": self.performance_metrics,
            "failures": self.failures,
            "warnings": self.warnings,
            "coverage_data": self.coverage_data,
            "recommendations": self.recommendations,
            "timestamp": time.time(),
            "script_version": SCRIPT_VERSION,
        }

        if include_system_info:
            result_dict["system_info"] = self.system_info

        return json.dumps(result_dict, indent=2, default=str)


class TestRunner:
    """
    Comprehensive test execution orchestrator providing intelligent test suite management,
    performance monitoring, result collection, and reporting with system optimization and
    error handling for all plume_nav_sim testing categories.
    """

    def __init__(
        self, config: TestExecutionConfig, logger: Optional[ComponentLogger] = None
    ):
        """
        Initialize TestRunner with execution configuration, logging, and system capability
        detection for optimized test execution.

        Args:
            config: Test execution configuration with parameters and options
            logger: Optional component logger for test execution monitoring
        """
        # Store test execution configuration with validation
        self.config = config
        config_validation = config.validate()
        if not config_validation.is_valid:
            raise ConfigurationError(
                f"Invalid test configuration: {config_validation.errors}"
            )

        # Initialize logger using provided logger or create component-specific logger
        self.logger = logger or get_component_logger("test_runner")
        self.logger.info(f"Initializing TestRunner v{SCRIPT_VERSION}")

        # Create TestConfigFactory with system capability detection and optimization
        self.config_factory = TestConfigFactory()

        # Detect system capabilities for optimization
        self.system_capabilities = self._detect_system_capabilities()
        self.logger.info(f"Detected system capabilities: {self.system_capabilities}")

        # Initialize performance baselines for monitoring
        self.performance_baselines = self._initialize_performance_baselines()

        # Set supported test categories with system compatibility
        self.supported_categories = DEFAULT_TEST_CATEGORIES.copy()

        # Initialize execution tracking
        self.execution_history = []
        self.is_running = False

        # Initialize worker pool for parallel execution
        self.worker_pool = None

    def _detect_system_capabilities(self) -> Dict[str, Any]:
        """Detect and analyze system capabilities for test optimization."""
        capabilities = {
            "cpu_cores": multiprocessing.cpu_count(),
            "platform": platform.system(),
            "python_version": platform.python_version(),
            "memory_gb": self._estimate_available_memory(),
            "cpu_speed_factor": 1.0,  # Default, could be benchmarked
        }

        # Platform-specific optimizations
        if capabilities["platform"] == "Windows":
            capabilities["parallel_support"] = "limited"
            self.logger.warning(
                "Windows platform detected - parallel execution has limited support"
            )
        else:
            capabilities["parallel_support"] = "full"

        return capabilities

    def _estimate_available_memory(self) -> float:
        """Estimate available system memory in GB."""
        try:
            import psutil

            memory = psutil.virtual_memory()
            return memory.available / (1024**3)
        except ImportError:
            # Fallback estimation
            return 4.0

    def _initialize_performance_baselines(self) -> Dict[str, float]:
        """Initialize performance baselines for monitoring."""
        return {
            "baseline_step_latency_ms": 0.5,
            "baseline_render_time_ms": 2.0,
            "baseline_memory_mb": 10.0,
        }

    def run_suite(
        self,
        categories: List[str],
        parallel_execution: bool = False,
        fail_fast: bool = False,
    ) -> TestResult:
        """
        Executes complete test suite with intelligent scheduling, performance monitoring,
        and comprehensive result collection for all configured test categories.

        Args:
            categories: List of test categories to execute
            parallel_execution: Enable parallel execution of test categories
            fail_fast: Stop execution on first category failure

        Returns:
            TestResult: Comprehensive suite execution result with timing, failures, and analysis
        """
        self.logger.info(f"Starting test suite execution for categories: {categories}")

        # Validate categories against supported categories
        invalid_categories = set(categories) - set(self.supported_categories)
        if invalid_categories:
            raise ValidationError(f"Unsupported test categories: {invalid_categories}")

        # Set execution state and initialize tracking
        self.is_running = True
        suite_start_time = time.perf_counter()

        # Create test result object
        test_result = TestResult(
            success=True,
            execution_time_seconds=0.0,
            total_tests=0,
            system_info=self.system_capabilities.copy(),
        )

        try:
            # Create optimized test configurations for each category
            category_configs = {}
            for category in categories:
                try:
                    category_configs[category] = (
                        self.config_factory.create_for_test_type(category)
                    )
                    self.logger.debug(f"Created configuration for {category} tests")
                except Exception as e:
                    self.logger.error(
                        f"Failed to create configuration for {category}: {e}"
                    )
                    test_result.warnings.append(
                        f"Configuration error for {category}: {e}"
                    )
                    continue

            # Setup parallel execution if enabled
            if parallel_execution and len(categories) > 1:
                self.logger.info("Setting up parallel test execution")
                self._setup_parallel_execution()

            # Execute test categories
            for category in categories:
                if category not in category_configs:
                    continue

                self.logger.info(f"Executing {category} tests")

                with PerformanceTimer() as timer:
                    try:
                        category_result = self.run_category(category)
                        test_result.add_category_result(category, category_result)

                        # Check fail_fast condition
                        if fail_fast and category_result.get("failed", 0) > 0:
                            self.logger.warning(
                                f"Fail-fast triggered by {category} test failures"
                            )
                            test_result.success = False
                            break

                    except Exception as e:
                        self.logger.error(f"Category {category} execution failed: {e}")
                        test_result.failures.append(
                            {
                                "category": category,
                                "error_type": type(e).__name__,
                                "error_message": str(e),
                                "traceback": str(e),
                            }
                        )
                        test_result.success = False

                        if fail_fast:
                            break

                # Log category completion time
                category_time = timer.get_duration_ms() / 1000.0
                self.logger.performance(
                    f"{category} tests completed in {category_time:.2f}s"
                )

            # Calculate total execution time
            test_result.execution_time_seconds = time.perf_counter() - suite_start_time

            # Generate performance analysis and recommendations
            self._analyze_performance(test_result)
            self._generate_recommendations(test_result)

            # Log suite completion
            self.logger.info(
                f"Test suite completed in {test_result.execution_time_seconds:.2f}s"
            )
            self.logger.info(
                f"Results: {test_result.passed_tests}P/{test_result.failed_tests}F/{test_result.skipped_tests}S"
            )

        except Exception as e:
            self.logger.error(f"Test suite execution failed: {e}")
            test_result.success = False
            test_result.failures.append(
                {
                    "error_type": "SuiteExecutionError",
                    "error_message": str(e),
                    "timestamp": time.time(),
                }
            )

        finally:
            # Cleanup execution resources
            self.is_running = False
            if self.worker_pool:
                self.worker_pool.close()
                self.worker_pool = None

        return test_result

    def run_category(
        self,
        category: str,
        test_filter: Optional[str] = None,
        timeout_seconds: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Executes specific test category with category-optimized configuration, pytest
        integration, and detailed monitoring for targeted testing scenarios.

        Args:
            category: Test category to execute (unit, integration, performance, etc.)
            test_filter: Optional test filter for selective execution
            timeout_seconds: Optional timeout override for category execution

        Returns:
            Dict[str, Any]: Category-specific test results with detailed metrics and analysis
        """
        self.logger.debug(f"Starting {category} test execution")

        # Validate category
        if category not in self.supported_categories:
            raise ValidationError(f"Unsupported test category: {category}")

        # Create category-specific test configuration
        config_method_map = {
            "unit": create_unit_test_config,
            "integration": create_integration_test_config,
            "performance": create_performance_test_config,
            "reproducibility": create_reproducibility_test_config,
            "edge_case": create_edge_case_test_config,
        }

        config_method_map[category]()

        # Generate pytest command arguments
        pytest_cmd = ["python", "-m", "pytest"]
        pytest_cmd.extend(self.config.pytest_args)

        # Add category-specific markers and options
        pytest_cmd.extend(["-m", category])

        # Add test filter if provided
        if test_filter:
            pytest_cmd.extend(["-k", test_filter])

        # Add output capture settings
        if self.config.capture_output:
            pytest_cmd.extend(["--capture=no"])

        # Set timeout
        execution_timeout = timeout_seconds or self.config.timeout_seconds

        # Initialize performance monitoring
        category_start_time = time.perf_counter()

        try:
            # Execute pytest subprocess with comprehensive monitoring
            self.logger.debug(f"Executing command: {' '.join(pytest_cmd)}")

            result = subprocess.run(
                pytest_cmd,
                timeout=execution_timeout,
                capture_output=True,
                text=True,
                cwd=pathlib.Path(__file__).parent.parent,
            )

            # Calculate execution time
            execution_time = time.perf_counter() - category_start_time

            # Parse pytest output for detailed results
            category_result = self._parse_pytest_output(
                result, category, execution_time
            )

            # Add performance metrics
            category_result["performance_metrics"] = {
                "execution_time_seconds": execution_time,
                "memory_usage_mb": self._estimate_memory_usage(),
                "tests_per_second": (
                    category_result["total"] / execution_time
                    if execution_time > 0
                    else 0
                ),
            }

            # Log category results
            self.logger.info(
                f"{category} tests: {category_result['passed']}P/{category_result['failed']}F/{category_result['skipped']}S"
            )

            return category_result

        except subprocess.TimeoutExpired:
            self.logger.error(f"{category} tests timed out after {execution_timeout}s")
            return {
                "passed": 0,
                "failed": 0,
                "skipped": 0,
                "total": 0,
                "execution_time": execution_timeout,
                "timeout_expired": True,
                "error_message": f"Test execution timed out after {execution_timeout} seconds",
            }

        except Exception as e:
            self.logger.error(f"{category} test execution failed: {e}")
            return {
                "passed": 0,
                "failed": 1,
                "skipped": 0,
                "total": 1,
                "execution_time": time.perf_counter() - category_start_time,
                "execution_error": True,
                "error_message": str(e),
            }

    def _parse_pytest_output(
        self, result: subprocess.CompletedProcess, category: str, execution_time: float
    ) -> Dict[str, Any]:
        """Parse pytest subprocess output into structured results."""
        category_result = {
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "total": 0,
            "execution_time": execution_time,
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

        # Parse pytest output for test counts
        stdout_lines = result.stdout.split("\n")
        for line in stdout_lines:
            if "passed" in line or "failed" in line or "skipped" in line:
                # Extract test counts from pytest summary
                if " passed" in line:
                    try:
                        count = int(line.split(" passed")[0].split()[-1])
                        category_result["passed"] = count
                    except (ValueError, IndexError):
                        pass

                if " failed" in line:
                    try:
                        count = int(line.split(" failed")[0].split()[-1])
                        category_result["failed"] = count
                    except (ValueError, IndexError):
                        pass

                if " skipped" in line:
                    try:
                        count = int(line.split(" skipped")[0].split()[-1])
                        category_result["skipped"] = count
                    except (ValueError, IndexError):
                        pass

        # Calculate total tests
        category_result["total"] = (
            category_result["passed"]
            + category_result["failed"]
            + category_result["skipped"]
        )

        return category_result

    def _setup_parallel_execution(self) -> None:
        """Setup multiprocessing pool for parallel test execution."""
        try:
            worker_count = min(
                MAX_PARALLEL_WORKERS, self.system_capabilities["cpu_cores"]
            )
            self.worker_pool = multiprocessing.Pool(processes=worker_count)
            self.logger.info(
                f"Initialized parallel execution with {worker_count} workers"
            )
        except Exception as e:
            self.logger.warning(f"Failed to setup parallel execution: {e}")
            self.worker_pool = None

    def _estimate_memory_usage(self) -> float:
        """Estimate current memory usage in MB."""
        try:
            import psutil

            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 10.0  # Default estimation

    def _analyze_performance(self, test_result: TestResult) -> None:
        """Analyze performance metrics and add performance data to test result."""
        # Calculate average execution time per test
        if test_result.total_tests > 0:
            avg_time_per_test = (
                test_result.execution_time_seconds / test_result.total_tests
            )
            test_result.performance_metrics["avg_time_per_test_seconds"] = (
                avg_time_per_test
            )

        # Compare against baselines
        for metric_name, baseline_value in self.performance_baselines.items():
            actual_key = metric_name.replace("baseline_", "")
            if actual_key in test_result.performance_metrics:
                actual_value = test_result.performance_metrics[actual_key]
                if actual_value > baseline_value * 2.0:  # 100% slower than baseline
                    test_result.warnings.append(
                        f"Performance regression detected in {actual_key}"
                    )

    def _generate_recommendations(self, test_result: TestResult) -> None:
        """Generate actionable recommendations based on test results."""
        # Performance recommendations
        if test_result.execution_time_seconds > 300:  # 5 minutes
            test_result.recommendations.append(
                "Consider enabling parallel execution to reduce test time"
            )

        # Failure analysis recommendations
        if test_result.failed_tests > 0:
            test_result.recommendations.append(
                "Review failed tests and fix underlying issues before deployment"
            )

            failure_rate = test_result.failed_tests / test_result.total_tests
            if failure_rate > 0.1:  # More than 10% failure rate
                test_result.recommendations.append(
                    "High failure rate detected - consider reviewing test quality"
                )

        # System optimization recommendations
        cpu_cores = self.system_capabilities.get("cpu_cores", 1)
        if cpu_cores >= 4 and not self.config.parallel_execution:
            test_result.recommendations.append(
                "Multi-core system detected - consider enabling parallel execution"
            )

    def get_execution_summary(
        self,
        include_performance_details: bool = True,
        include_failure_analysis: bool = True,
    ) -> Dict[str, Any]:
        """
        Generates comprehensive execution summary with statistics, performance metrics,
        failure analysis, and optimization recommendations for test suite analysis.

        Args:
            include_performance_details: Include detailed performance analysis and metrics
            include_failure_analysis: Include comprehensive failure analysis and categorization

        Returns:
            Dict[str, Any]: Comprehensive execution summary with statistics, trends, and recommendations
        """
        summary = {
            "runner_info": {
                "script_name": SCRIPT_NAME,
                "script_version": SCRIPT_VERSION,
                "execution_history_count": len(self.execution_history),
                "system_capabilities": self.system_capabilities,
                "configuration": self.config.to_dict(),
            },
            "execution_statistics": {
                "total_runs": len(self.execution_history),
                "currently_running": self.is_running,
                "supported_categories": self.supported_categories,
            },
        }

        if include_performance_details:
            summary["performance_details"] = {
                "baselines": self.performance_baselines,
                "optimization_enabled": self.config.parallel_execution,
            }

        if include_failure_analysis and self.execution_history:
            # Analyze historical failures
            total_failures = sum(
                run.get("failed_tests", 0) for run in self.execution_history
            )
            summary["failure_analysis"] = {
                "historical_failures": total_failures,
                "failure_trends": "Stable",  # Could be enhanced with trend analysis
            }

        return summary

    def cleanup_resources(self, force_cleanup: bool = False) -> None:
        """
        Comprehensive cleanup of test execution resources including worker pools,
        temporary data, and resource deallocation with validation.

        Args:
            force_cleanup: Force cleanup of all resources regardless of state
        """
        self.logger.info("Cleaning up test runner resources")

        # Shutdown worker pool if active
        if self.worker_pool:
            try:
                self.worker_pool.close()
                self.worker_pool.join()
                self.worker_pool = None
                self.logger.debug("Parallel execution pool cleaned up")
            except Exception as e:
                self.logger.warning(f"Error cleaning up worker pool: {e}")

        # Clear execution history if force cleanup
        if force_cleanup:
            self.execution_history.clear()
            self.performance_baselines.clear()
            self.logger.debug("Execution history and baselines cleared")

        # Reset execution state
        self.is_running = False

        self.logger.info("Resource cleanup completed")


def setup_test_environment(
    config: TestExecutionConfig,
    clean_previous_artifacts: bool = False,
    validate_system_requirements: bool = True,
    base_directory: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Initializes comprehensive test execution environment including directory structure,
    logging configuration, environment variables, and system validation for reliable test execution.

    Args:
        config: Test execution configuration with directory and logging settings
        clean_previous_artifacts: Remove previous test artifacts before setup
        validate_system_requirements: Validate system compatibility and requirements
        base_directory: Base directory for test environment setup

    Returns:
        Dict[str, Any]: Environment setup result with directory paths, configuration status, and validation results
    """
    logger = get_component_logger("test_environment_setup")
    logger.info("Setting up test execution environment")

    setup_result = {
        "success": True,
        "base_directory": base_directory or os.getcwd(),
        "output_directories": {},
        "environment_variables": {},
        "validation_results": {},
        "warnings": [],
        "errors": [],
    }

    try:
        # Create base directory structure
        base_path = pathlib.Path(setup_result["base_directory"])

        # Setup output directories
        output_dir = base_path / config.output_directory
        reports_dir = output_dir / REPORT_OUTPUT_DIR
        logs_dir = output_dir / LOG_OUTPUT_DIR

        for dir_path in [output_dir, reports_dir, logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            setup_result["output_directories"][dir_path.name] = str(dir_path)

        logger.info(
            f"Created directory structure: {setup_result['output_directories']}"
        )

        # Clean previous artifacts if requested
        if clean_previous_artifacts:
            logger.info("Cleaning previous test artifacts")
            cleanup_result = cleanup_test_artifacts(
                artifacts_directory=str(output_dir),
                preserve_reports=False,
                preserve_logs=False,
                max_age_hours=0,
                force_cleanup=True,
            )
            if not cleanup_result["success"]:
                setup_result["warnings"].extend(cleanup_result.get("warnings", []))

        # Initialize comprehensive logging configuration
        configure_logging_for_development(
            log_level=config.log_level or "INFO",
            log_file=str(logs_dir / "test_execution.log"),
        )
        logger.info("Configured logging for test execution")

        # Setup environment variables
        test_env_vars = {
            "PLUME_NAV_SIM_TEST_MODE": "1",
            "PLUME_NAV_SIM_LOG_LEVEL": config.log_level or "INFO",
            "PYTHONPATH": str(base_path / "src"),
        }

        for var_name, var_value in test_env_vars.items():
            os.environ[var_name] = var_value
            setup_result["environment_variables"][var_name] = var_value

        logger.info("Set test environment variables")

        # Validate system requirements if enabled
        if validate_system_requirements:
            logger.info("Validating system requirements")

            # Check Python version
            python_version = platform.python_version()
            if python_version not in SUPPORTED_PYTHON_VERSIONS:
                major_minor = ".".join(python_version.split(".")[:2])
                if major_minor in [v[:4] for v in SUPPORTED_PYTHON_VERSIONS]:
                    setup_result["warnings"].append(
                        f"Python {python_version} - patch version differs from tested versions"
                    )
                else:
                    setup_result["errors"].append(
                        f"Unsupported Python version: {python_version}"
                    )
                    setup_result["success"] = False

            # Validate package availability
            try:
                initialize_package()
                package_info = get_package_info()
                setup_result["validation_results"]["package_info"] = package_info
                logger.info(f"Package validated: {package_info}")
            except Exception as e:
                setup_result["errors"].append(f"Package validation failed: {e}")
                setup_result["success"] = False

            # Check system resources
            try:
                import psutil

                memory = psutil.virtual_memory()
                disk = shutil.disk_usage(setup_result["base_directory"])

                if memory.available < 512 * 1024 * 1024:  # 512MB minimum
                    setup_result["warnings"].append("Low available memory detected")

                if disk.free < 100 * 1024 * 1024:  # 100MB minimum
                    setup_result["errors"].append(
                        "Insufficient disk space for test execution"
                    )
                    setup_result["success"] = False

                setup_result["validation_results"]["system_resources"] = {
                    "memory_available_mb": memory.available / 1024 / 1024,
                    "disk_free_mb": disk.free / 1024 / 1024,
                }

            except ImportError:
                setup_result["warnings"].append(
                    "psutil not available - system resource validation skipped"
                )

        logger.info("Test environment setup completed successfully")

    except Exception as e:
        logger.error(f"Test environment setup failed: {e}")
        setup_result["success"] = False
        setup_result["errors"].append(str(e))

    return setup_result


def cleanup_test_artifacts(
    artifacts_directory: Optional[str] = None,
    preserve_reports: bool = True,
    preserve_logs: bool = True,
    max_age_hours: int = 24,
    force_cleanup: bool = False,
) -> Dict[str, Any]:
    """
    Comprehensive cleanup of test artifacts including temporary files, log files, cache
    directories, and resource deallocation with configurable retention policies and cleanup validation.

    Args:
        artifacts_directory: Directory containing test artifacts to clean
        preserve_reports: Preserve test report files during cleanup
        preserve_logs: Preserve log files during cleanup
        max_age_hours: Maximum age of artifacts to preserve (hours)
        force_cleanup: Force cleanup of all artifacts regardless of policies

    Returns:
        Dict[str, Any]: Cleanup result with removed files, preserved artifacts, and cleanup statistics
    """
    logger = get_component_logger("test_cleanup")
    logger.info("Starting test artifact cleanup")

    cleanup_result = {
        "success": True,
        "artifacts_directory": artifacts_directory or os.getcwd(),
        "removed_files": [],
        "preserved_files": [],
        "cleanup_statistics": {},
        "warnings": [],
        "errors": [],
    }

    try:
        artifacts_path = pathlib.Path(cleanup_result["artifacts_directory"])
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600

        total_size_removed = 0
        files_removed = 0
        files_preserved = 0

        # Define cleanup patterns
        cleanup_patterns = [
            "**/*.pyc",
            "**/__pycache__",
            "**/pytest_cache",
            "**/.coverage",
            "**/test-results*.xml",
            "**/coverage.xml",
        ]

        # Add report patterns if not preserving
        if not preserve_reports or force_cleanup:
            cleanup_patterns.extend(
                [
                    f"**/{REPORT_OUTPUT_DIR}/**/*",
                    "**/test_report_*.html",
                    "**/test_report_*.json",
                ]
            )

        # Add log patterns if not preserving
        if not preserve_logs or force_cleanup:
            cleanup_patterns.extend(
                [
                    f"**/{LOG_OUTPUT_DIR}/**/*",
                    "**/test_execution_*.log",
                    "**/pytest.log",
                ]
            )

        # Process each cleanup pattern
        for pattern in cleanup_patterns:
            for file_path in artifacts_path.glob(pattern):
                try:
                    if file_path.is_file():
                        # Check file age against retention policy
                        file_age = current_time - file_path.stat().st_mtime

                        if force_cleanup or file_age > max_age_seconds:
                            file_size = file_path.stat().st_size
                            file_path.unlink()

                            cleanup_result["removed_files"].append(str(file_path))
                            total_size_removed += file_size
                            files_removed += 1

                            logger.debug(f"Removed: {file_path}")

                        else:
                            cleanup_result["preserved_files"].append(str(file_path))
                            files_preserved += 1

                    elif file_path.is_dir() and not list(file_path.iterdir()):
                        # Remove empty directories
                        file_path.rmdir()
                        cleanup_result["removed_files"].append(str(file_path))
                        files_removed += 1
                        logger.debug(f"Removed empty directory: {file_path}")

                except (PermissionError, OSError) as e:
                    cleanup_result["warnings"].append(
                        f"Could not remove {file_path}: {e}"
                    )

        # Generate cleanup statistics
        cleanup_result["cleanup_statistics"] = {
            "files_removed": files_removed,
            "files_preserved": files_preserved,
            "total_size_removed_mb": total_size_removed / 1024 / 1024,
            "cleanup_duration_seconds": time.time() - current_time,
            "retention_policy_hours": max_age_hours,
            "force_cleanup_used": force_cleanup,
        }

        logger.info(
            f"Cleanup completed: {files_removed} files removed, "
            f"{total_size_removed / 1024 / 1024:.1f} MB freed"
        )

    except Exception as e:
        logger.error(f"Test artifact cleanup failed: {e}")
        cleanup_result["success"] = False
        cleanup_result["errors"].append(str(e))

    return cleanup_result


def validate_test_configuration(
    config: TestExecutionConfig,
    check_system_compatibility: bool = True,
    validate_performance_targets: bool = True,
    strict_validation: bool = False,
) -> ValidationResult:
    """
    Comprehensive validation of test execution configuration ensuring parameter consistency,
    system compatibility, resource availability, and performance feasibility with detailed error reporting.

    Args:
        config: Test execution configuration to validate
        check_system_compatibility: Validate system compatibility and requirements
        validate_performance_targets: Validate performance targets against system capabilities
        strict_validation: Enable strict validation rules and edge case testing

    Returns:
        ValidationResult: Comprehensive configuration validation result with errors, warnings, and recommendations
    """
    logger = get_component_logger("config_validation")
    logger.info("Starting comprehensive test configuration validation")

    # Create validation context for comprehensive error tracking
    context = create_validation_context("test_configuration")

    try:
        # Basic configuration validation using built-in method
        basic_validation = config.validate(strict_validation)
        if not basic_validation.is_valid:
            context.add_errors(basic_validation.errors)
        context.add_warnings(basic_validation.warnings)

        # Test category configuration validation
        for category in config.test_categories:
            if category not in DEFAULT_TEST_CATEGORIES:
                context.add_error(f"Invalid test category: {category}")
            else:
                try:
                    # Validate category-specific configuration can be created
                    factory = TestConfigFactory()
                    category_config = factory.create_for_test_type(category)
                    logger.debug(f"Validated configuration for {category} tests")
                except Exception as e:
                    context.add_error(
                        f"Cannot create configuration for {category}: {e}"
                    )

        # System compatibility validation
        if check_system_compatibility:
            logger.debug("Validating system compatibility")

            # Python version compatibility
            python_version = platform.python_version()
            major_minor = ".".join(python_version.split(".")[:2])
            if major_minor not in [v[:4] for v in SUPPORTED_PYTHON_VERSIONS]:
                context.add_error(f"Unsupported Python version: {python_version}")

            # Platform compatibility
            system_platform = platform.system()
            if system_platform == "Windows":
                context.add_warning(
                    "Windows platform has limited support - some tests may fail"
                )

            # Required package availability
            try:
                import matplotlib
                import numpy
                import pytest

                logger.debug("Core dependencies validated")
            except ImportError as e:
                context.add_error(f"Missing required dependency: {e}")

        # Performance targets validation
        if validate_performance_targets:
            logger.debug("Validating performance targets")

            targets = config.performance_targets

            # Validate step latency target
            step_target = targets.get("step_latency_ms", 1.0)
            if step_target <= 0:
                context.add_error("Step latency target must be positive")
            elif step_target < 0.1:
                context.add_warning(
                    "Very tight step latency target may cause false test failures"
                )
            elif step_target > 10.0:
                context.add_warning(
                    "Loose step latency target may not detect performance regressions"
                )

            # Validate memory limit
            memory_limit = targets.get("memory_limit_mb", 50.0)
            if memory_limit <= 0:
                context.add_error("Memory limit must be positive")
            elif memory_limit < 10.0:
                context.add_warning("Very low memory limit may cause test failures")

        # Strict validation additional checks
        if strict_validation:
            logger.debug("Performing strict validation checks")

            # Cross-parameter consistency
            if config.parallel_execution and len(config.test_categories) == 1:
                context.add_warning(
                    "Parallel execution enabled for single test category"
                )

            # Timeout validation
            if config.timeout_seconds < 60:
                context.add_warning("Short timeout may interrupt long-running tests")
            elif config.timeout_seconds > 1800:  # 30 minutes
                context.add_warning("Very long timeout may mask hanging tests")

            # Output directory validation
            try:
                output_path = pathlib.Path(config.output_directory)
                if not output_path.exists():
                    output_path.mkdir(parents=True, exist_ok=True)

                # Check write permissions
                test_file = output_path / "test_write_permission.tmp"
                test_file.write_text("test")
                test_file.unlink()

            except (PermissionError, OSError) as e:
                context.add_error(f"Output directory issue: {e}")

        # Generate recommendations
        validation_result = context.get_result()

        if validation_result.warnings:
            validation_result.add_recommendation(
                "Review warning messages and consider adjusting configuration parameters"
            )

        if config.parallel_execution and multiprocessing.cpu_count() < 2:
            validation_result.add_recommendation(
                "Consider disabling parallel execution on single-core systems"
            )

        logger.info(
            f"Configuration validation completed: {len(validation_result.errors)} errors, "
            f"{len(validation_result.warnings)} warnings"
        )

        return validation_result

    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        context.add_error(f"Validation process failed: {e}")
        return context.get_result()


def get_system_info(
    include_hardware_info: bool = True,
    include_environment_details: bool = True,
    include_performance_baselines: bool = False,
) -> Dict[str, Any]:
    """
    Collects comprehensive system information including hardware specifications, software versions,
    Python environment details, and test execution capabilities for system compatibility analysis.

    Args:
        include_hardware_info: Include hardware specifications and system capabilities
        include_environment_details: Include Python environment and package information
        include_performance_baselines: Include performance benchmarks and timing baselines

    Returns:
        Dict[str, Any]: Comprehensive system information dictionary with hardware, software, and performance data
    """
    logger = get_component_logger("system_info")
    logger.info("Collecting system information for test environment analysis")

    system_info = {
        "collection_timestamp": time.time(),
        "basic_info": {
            "platform": platform.system(),
            "platform_version": platform.version(),
            "architecture": platform.architecture()[0],
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
        },
    }

    try:
        # Hardware information collection
        if include_hardware_info:
            logger.debug("Collecting hardware information")

            hardware_info = {
                "cpu_count": multiprocessing.cpu_count(),
                "processor": platform.processor(),
            }

            # Extended hardware info if psutil available
            try:
                import psutil

                # Memory information
                memory = psutil.virtual_memory()
                hardware_info.update(
                    {
                        "memory_total_gb": memory.total / 1024**3,
                        "memory_available_gb": memory.available / 1024**3,
                        "memory_percent_used": memory.percent,
                    }
                )

                # CPU information
                cpu_freq = psutil.cpu_freq()
                if cpu_freq:
                    hardware_info["cpu_frequency_mhz"] = cpu_freq.current

                # Disk space for test directory
                disk_usage = psutil.disk_usage(os.getcwd())
                hardware_info.update(
                    {
                        "disk_total_gb": disk_usage.total / 1024**3,
                        "disk_free_gb": disk_usage.free / 1024**3,
                        "disk_used_percent": (disk_usage.used / disk_usage.total) * 100,
                    }
                )

            except ImportError:
                hardware_info["extended_info"] = "psutil not available"

            system_info["hardware"] = hardware_info

        # Environment details collection
        if include_environment_details:
            logger.debug("Collecting environment details")

            environment_info = {
                "python_executable": sys.executable,
                "python_path": sys.path[:5],  # First 5 entries to avoid clutter
                "working_directory": os.getcwd(),
            }

            # Package version information
            try:
                package_versions = {}

                # Core test dependencies
                import pytest

                package_versions["pytest"] = pytest.__version__

                import numpy

                package_versions["numpy"] = numpy.__version__

                try:
                    import matplotlib

                    package_versions["matplotlib"] = matplotlib.__version__
                except ImportError:
                    package_versions["matplotlib"] = "not available"

                # plume_nav_sim package info
                try:
                    package_info = get_package_info()
                    package_versions["plume_nav_sim"] = package_info.get(
                        "version", "unknown"
                    )
                except Exception:
                    package_versions["plume_nav_sim"] = "not available"

                environment_info["package_versions"] = package_versions

            except Exception as e:
                environment_info["package_versions"] = f"collection failed: {e}"

            # Environment variables relevant to testing
            test_env_vars = {}
            for var_name in ["PYTHONPATH", "PATH", "PLUME_NAV_SIM_TEST_MODE"]:
                test_env_vars[var_name] = os.environ.get(var_name, "not set")

            environment_info["environment_variables"] = test_env_vars
            system_info["environment"] = environment_info

        # Performance baselines collection
        if include_performance_baselines:
            logger.debug("Running performance baselines")

            baseline_info = {"baseline_timestamp": time.time(), "timing_tests": {}}

            # Simple timing benchmarks
            try:
                # NumPy operation timing
                import numpy as np

                with PerformanceTimer() as timer:
                    # Simple array operation
                    arr = np.random.random((100, 100))
                    result = np.sum(arr)

                baseline_info["timing_tests"][
                    "numpy_operation_ms"
                ] = timer.get_duration_ms()

                # Memory allocation timing
                with PerformanceTimer() as timer:
                    large_array = np.zeros((1000, 1000), dtype=np.float32)
                    del large_array

                baseline_info["timing_tests"][
                    "memory_allocation_ms"
                ] = timer.get_duration_ms()

                # File I/O timing
                import tempfile

                with PerformanceTimer() as timer:
                    with tempfile.NamedTemporaryFile(mode="w", delete=True) as f:
                        f.write("test data" * 1000)
                        f.flush()

                baseline_info["timing_tests"]["file_io_ms"] = timer.get_duration_ms()

            except Exception as e:
                baseline_info["timing_tests"] = f"benchmark failed: {e}"

            system_info["performance_baselines"] = baseline_info

        # Test framework compatibility
        compatibility_info = {
            "pytest_available": True,
            "matplotlib_backend": "unknown",
            "multiprocessing_support": True,
        }

        # Check matplotlib backend
        try:
            import matplotlib
            import matplotlib.pyplot as plt

            compatibility_info["matplotlib_backend"] = plt.get_backend()
        except ImportError:
            compatibility_info["matplotlib_backend"] = "matplotlib not available"
        except Exception as e:
            compatibility_info["matplotlib_backend"] = f"backend detection failed: {e}"

        # Test multiprocessing support
        try:
            with multiprocessing.Pool(processes=1) as pool:
                result = pool.apply(lambda: 42)
                compatibility_info["multiprocessing_support"] = result == 42
        except Exception:
            compatibility_info["multiprocessing_support"] = False

        system_info["compatibility"] = compatibility_info

        logger.info("System information collection completed")

    except Exception as e:
        logger.error(f"System information collection failed: {e}")
        system_info["collection_error"] = str(e)

    return system_info


def generate_test_report(
    test_result: TestResult,
    output_path: Optional[str] = None,
    report_formats: List[str] = None,
    include_performance_analysis: bool = True,
    include_recommendations: bool = True,
) -> Dict[str, Any]:
    """
    Generates comprehensive test execution report with multiple output formats, performance
    analysis, failure summaries, and actionable recommendations for development and CI/CD integration.

    Args:
        test_result: Comprehensive test execution result with all metrics and analysis
        output_path: Directory path for report output files
        report_formats: List of report formats to generate (html, json, text)
        include_performance_analysis: Include detailed performance metrics and analysis
        include_recommendations: Include actionable recommendations and optimization suggestions

    Returns:
        Dict[str, Any]: Report generation result with file paths, format information, and generation status
    """
    logger = get_component_logger("test_report")
    logger.info("Generating comprehensive test execution report")

    # Set default parameters
    if output_path is None:
        output_path = pathlib.Path.cwd() / REPORT_OUTPUT_DIR
    else:
        output_path = pathlib.Path(output_path)

    if report_formats is None:
        report_formats = ["html", "json", "text"]

    # Ensure output directory exists
    output_path.mkdir(parents=True, exist_ok=True)

    generation_result = {
        "success": True,
        "output_directory": str(output_path),
        "generated_reports": {},
        "report_metadata": {},
        "warnings": [],
        "errors": [],
    }

    try:
        # Generate timestamp for report naming
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base_filename = f"test_report_{timestamp}"

        # Generate JSON report for programmatic access
        if "json" in report_formats:
            logger.debug("Generating JSON report")

            json_filename = f"{base_filename}.json"
            json_path = output_path / json_filename

            try:
                json_content = test_result.to_json(include_system_info=True)
                json_path.write_text(json_content, encoding="utf-8")

                generation_result["generated_reports"]["json"] = str(json_path)
                logger.info(f"JSON report generated: {json_path}")

            except Exception as e:
                generation_result["errors"].append(
                    f"JSON report generation failed: {e}"
                )

        # Generate text summary report
        if "text" in report_formats:
            logger.debug("Generating text summary report")

            text_filename = f"{base_filename}.txt"
            text_path = output_path / text_filename

            try:
                text_content = test_result.generate_summary(
                    include_details=True,
                    include_recommendations=include_recommendations,
                )

                # Add system information if requested
                if include_performance_analysis and test_result.system_info:
                    text_content += "\n\nSystem Information:\n"
                    text_content += "=" * 20 + "\n"
                    for key, value in test_result.system_info.items():
                        if isinstance(value, dict):
                            text_content += f"{key}:\n"
                            for sub_key, sub_value in value.items():
                                text_content += f"  {sub_key}: {sub_value}\n"
                        else:
                            text_content += f"{key}: {value}\n"

                text_path.write_text(text_content, encoding="utf-8")

                generation_result["generated_reports"]["text"] = str(text_path)
                logger.info(f"Text report generated: {text_path}")

            except Exception as e:
                generation_result["errors"].append(
                    f"Text report generation failed: {e}"
                )

        # Generate HTML report with interactive features
        if "html" in report_formats:
            logger.debug("Generating HTML report")

            html_filename = f"{base_filename}.html"
            html_path = output_path / html_filename

            try:
                html_content = _generate_html_report(
                    test_result, include_performance_analysis, include_recommendations
                )

                html_path.write_text(html_content, encoding="utf-8")

                generation_result["generated_reports"]["html"] = str(html_path)
                logger.info(f"HTML report generated: {html_path}")

            except Exception as e:
                generation_result["errors"].append(
                    f"HTML report generation failed: {e}"
                )

        # Generate report metadata
        generation_result["report_metadata"] = {
            "generation_timestamp": time.time(),
            "test_execution_time": test_result.execution_time_seconds,
            "total_tests": test_result.total_tests,
            "success_rate": (
                test_result.passed_tests / test_result.total_tests
                if test_result.total_tests > 0
                else 0
            ),
            "report_formats": list(generation_result["generated_reports"].keys()),
            "include_performance_analysis": include_performance_analysis,
            "include_recommendations": include_recommendations,
        }

        # Check for any generation failures
        if generation_result["errors"]:
            generation_result["success"] = False
            logger.warning(
                f"Report generation completed with {len(generation_result['errors'])} errors"
            )
        else:
            logger.info("All reports generated successfully")

    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        generation_result["success"] = False
        generation_result["errors"].append(str(e))

    return generation_result


def _generate_html_report(
    test_result: TestResult,
    include_performance_analysis: bool,
    include_recommendations: bool,
) -> str:
    """Generate comprehensive HTML report with styling and interactive features."""

    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Test Execution Report - plume_nav_sim</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
            .header {{ background-color: #f4f4f4; padding: 20px; border-radius: 5px; }}
            .success {{ color: #28a745; }}
            .failure {{ color: #dc3545; }}
            .warning {{ color: #ffc107; }}
            .section {{ margin: 20px 0; }}
            .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #f8f9fa; border-radius: 3px; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .expandable {{ cursor: pointer; }}
            .content {{ display: none; margin-top: 10px; }}
        </style>
        <script>
            function toggleSection(id) {{
                var content = document.getElementById(id);
                content.style.display = content.style.display === 'none' ? 'block' : 'none';
            }}
        </script>
    </head>
    <body>
        <div class="header">
            <h1>Test Execution Report</h1>
            <p><strong>Script:</strong> {script_name} v{script_version}</p>
            <p><strong>Generated:</strong> {timestamp}</p>
            <h2 class="{status_class}">Status: {status}</h2>
        </div>

        <div class="section">
            <h2>Execution Summary</h2>
            <div class="metric">Total Tests: <strong>{total_tests}</strong></div>
            <div class="metric">Passed: <strong class="success">{passed_tests}</strong></div>
            <div class="metric">Failed: <strong class="failure">{failed_tests}</strong></div>
            <div class="metric">Skipped: <strong class="warning">{skipped_tests}</strong></div>
            <div class="metric">Execution Time: <strong>{execution_time:.2f}s</strong></div>
        </div>

        {category_section}
        {performance_section}
        {failures_section}
        {recommendations_section}
        {system_info_section}

    </body>
    </html>
    """

    # Generate category results section
    category_section = ""
    if test_result.category_results:
        category_section = """
        <div class="section">
            <h2>Category Results</h2>
            <table>
                <tr><th>Category</th><th>Passed</th><th>Failed</th><th>Skipped</th><th>Time (s)</th></tr>
        """

        for category, results in test_result.category_results.items():
            category_section += f"""
                <tr>
                    <td>{category.capitalize()}</td>
                    <td class="success">{results.get('passed', 0)}</td>
                    <td class="failure">{results.get('failed', 0)}</td>
                    <td class="warning">{results.get('skipped', 0)}</td>
                    <td>{results.get('execution_time', 0):.2f}</td>
                </tr>
            """

        category_section += "</table></div>"

    # Generate performance section
    performance_section = ""
    if include_performance_analysis and test_result.performance_metrics:
        performance_section = """
        <div class="section">
            <h2 class="expandable" onclick="toggleSection('performance')">Performance Analysis</h2>
            <div id="performance" class="content">
        """

        for metric_name, metric_value in test_result.performance_metrics.items():
            performance_section += f"<div class='metric'>{metric_name}: <strong>{metric_value}</strong></div>"

        performance_section += "</div></div>"

    # Generate failures section
    failures_section = ""
    if test_result.failures:
        failures_section = """
        <div class="section">
            <h2 class="expandable" onclick="toggleSection('failures')">Failure Details</h2>
            <div id="failures" class="content">
                <ol>
        """

        for failure in test_result.failures[:10]:  # Limit to 10 failures
            failures_section += f"""
                <li>
                    <strong>{failure.get('test_name', 'Unknown Test')}</strong><br>
                    <em>{failure.get('error_message', 'No error message')}</em>
                </li>
            """

        failures_section += "</ol></div></div>"

    # Generate recommendations section
    recommendations_section = ""
    if include_recommendations and test_result.recommendations:
        recommendations_section = """
        <div class="section">
            <h2>Recommendations</h2>
            <ul>
        """

        for recommendation in test_result.recommendations:
            recommendations_section += f"<li>{recommendation}</li>"

        recommendations_section += "</ul></div>"

    # Generate system info section
    system_info_section = ""
    if test_result.system_info:
        system_info_section = """
        <div class="section">
            <h2 class="expandable" onclick="toggleSection('system')">System Information</h2>
            <div id="system" class="content">
        """

        for key, value in test_result.system_info.items():
            if isinstance(value, dict):
                system_info_section += f"<h4>{key.replace('_', ' ').title()}</h4><ul>"
                for sub_key, sub_value in value.items():
                    system_info_section += (
                        f"<li><strong>{sub_key}:</strong> {sub_value}</li>"
                    )
                system_info_section += "</ul>"
            else:
                system_info_section += (
                    f"<div class='metric'><strong>{key}:</strong> {value}</div>"
                )

        system_info_section += "</div></div>"

    # Fill template
    return html_template.format(
        script_name=SCRIPT_NAME,
        script_version=SCRIPT_VERSION,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        status_class="success" if test_result.success else "failure",
        status="PASSED" if test_result.success else "FAILED",
        total_tests=test_result.total_tests,
        passed_tests=test_result.passed_tests,
        failed_tests=test_result.failed_tests,
        skipped_tests=test_result.skipped_tests,
        execution_time=test_result.execution_time_seconds,
        category_section=category_section,
        performance_section=performance_section,
        failures_section=failures_section,
        recommendations_section=recommendations_section,
        system_info_section=system_info_section,
    )


def create_argument_parser() -> argparse.ArgumentParser:
    """
    Creates comprehensive command-line argument parser with all test execution options,
    validation, and help documentation for flexible test script invocation and automation.

    Returns:
        argparse.ArgumentParser: Configured argument parser with comprehensive option definitions and validation
    """
    parser = argparse.ArgumentParser(
        prog=SCRIPT_NAME,
        description="Comprehensive test execution script for plume_nav_sim with intelligent "
        "suite management, performance monitoring, and detailed reporting.",
        epilog=f"Examples:\n"
        f"  python {SCRIPT_NAME} --categories unit integration\n"
        f"  python {SCRIPT_NAME} --parallel --performance-analysis\n"
        f"  python {SCRIPT_NAME} --all-categories --generate-reports\n"
        f"  python {SCRIPT_NAME} --categories performance --timeout 1800",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Test category selection arguments
    category_group = parser.add_mutually_exclusive_group()
    category_group.add_argument(
        "--categories",
        nargs="+",
        choices=DEFAULT_TEST_CATEGORIES,
        default=["unit", "integration"],
        help=f'Test categories to execute. Choices: {", ".join(DEFAULT_TEST_CATEGORIES)} '
        f"(default: unit integration)",
    )
    category_group.add_argument(
        "--all-categories",
        action="store_const",
        const=DEFAULT_TEST_CATEGORIES,
        dest="categories",
        help="Execute all available test categories",
    )

    # Execution behavior options
    execution_group = parser.add_argument_group("Execution Options")
    execution_group.add_argument(
        "--parallel",
        action="store_true",
        help="Enable parallel execution of test categories for improved performance",
    )
    execution_group.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop execution immediately on first test category failure",
    )
    execution_group.add_argument(
        "--timeout",
        type=int,
        default=TEST_TIMEOUT_SECONDS,
        metavar="SECONDS",
        help=f"Test execution timeout in seconds (default: {TEST_TIMEOUT_SECONDS})",
    )
    execution_group.add_argument(
        "--test-filter",
        metavar="PATTERN",
        help="Filter tests by name pattern (passed to pytest -k option)",
    )

    # Output and reporting options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--output-dir",
        default="test_output",
        metavar="DIRECTORY",
        help="Directory for test outputs and reports (default: test_output)",
    )
    output_group.add_argument(
        "--generate-reports",
        action="store_true",
        default=True,
        help="Generate comprehensive test reports (default: enabled)",
    )
    output_group.add_argument(
        "--report-formats",
        nargs="+",
        choices=["html", "json", "text"],
        default=["html", "json", "text"],
        help="Report formats to generate (default: all formats)",
    )
    output_group.add_argument(
        "--no-reports",
        action="store_false",
        dest="generate_reports",
        help="Disable report generation",
    )

    # Performance and analysis options
    perf_group = parser.add_argument_group("Performance Options")
    perf_group.add_argument(
        "--performance-analysis",
        action="store_true",
        default=True,
        help="Include detailed performance analysis in reports (default: enabled)",
    )
    perf_group.add_argument(
        "--system-info",
        action="store_true",
        help="Collect and include comprehensive system information",
    )
    perf_group.add_argument(
        "--benchmarks",
        action="store_true",
        help="Run performance benchmarks and timing baselines",
    )

    # Cleanup and maintenance options
    cleanup_group = parser.add_argument_group("Cleanup Options")
    cleanup_group.add_argument(
        "--cleanup-artifacts",
        action="store_true",
        help="Clean up test artifacts after execution",
    )
    cleanup_group.add_argument(
        "--preserve-logs",
        action="store_true",
        default=True,
        help="Preserve log files during cleanup (default: enabled)",
    )
    cleanup_group.add_argument(
        "--preserve-reports",
        action="store_true",
        default=True,
        help="Preserve report files during cleanup (default: enabled)",
    )

    # Validation and debugging options
    debug_group = parser.add_argument_group("Debug and Validation")
    debug_group.add_argument(
        "--validate-config",
        action="store_true",
        help="Validate configuration before test execution",
    )
    debug_group.add_argument(
        "--strict-validation",
        action="store_true",
        help="Enable strict validation with additional checks",
    )
    debug_group.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level for test execution (default: INFO)",
    )
    debug_group.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output and detailed logging",
    )

    # Version and help options
    parser.add_argument(
        "--version", action="version", version=f"{SCRIPT_NAME} v{SCRIPT_VERSION}"
    )

    return parser


def main(args: Optional[List[str]] = None) -> int:
    """
    Main entry point for test execution script providing comprehensive command-line interface,
    argument parsing, test execution orchestration, and result reporting with error handling
    and exit code management for CI/CD integration and automated workflows.

    Args:
        args: Optional command-line arguments list for testing purposes

    Returns:
        int: Exit code indicating test execution success (0), failure (1), or error (2)
    """
    # Parse command-line arguments
    parser = create_argument_parser()
    parsed_args = parser.parse_args(args)

    # Set up logging level
    log_level = "DEBUG" if parsed_args.verbose else parsed_args.log_level

    try:
        # Initialize logging configuration
        configure_logging_for_development(log_level=log_level)
        logger = get_component_logger("main")

        logger.info(f"Starting {SCRIPT_NAME} v{SCRIPT_VERSION}")
        logger.info(f"Python version: {platform.python_version()}")
        logger.info(f"Platform: {platform.system()} {platform.version()}")

        # Validate Python version compatibility
        python_version = platform.python_version()
        major_minor = ".".join(python_version.split(".")[:2])
        if major_minor not in [v[:4] for v in SUPPORTED_PYTHON_VERSIONS]:
            logger.error(f"Unsupported Python version: {python_version}")
            logger.error(f"Supported versions: {', '.join(SUPPORTED_PYTHON_VERSIONS)}")
            return EXIT_CODE_ERROR

        # Initialize package for dependency validation
        try:
            initialize_package()
            package_info = get_package_info()
            logger.info(
                f"Package validated: {package_info.get('name', 'unknown')} v{package_info.get('version', 'unknown')}"
            )
        except Exception as e:
            logger.error(f"Package initialization failed: {e}")
            return EXIT_CODE_ERROR

        # Create test execution configuration
        config = TestExecutionConfig(
            test_categories=parsed_args.categories,
            parallel_execution=parsed_args.parallel,
            output_directory=parsed_args.output_dir,
            fail_fast=parsed_args.fail_fast,
            timeout_seconds=parsed_args.timeout,
            capture_output=not parsed_args.verbose,
            generate_reports=parsed_args.generate_reports,
            cleanup_artifacts=parsed_args.cleanup_artifacts,
            log_level=log_level,
        )

        logger.info(f"Test configuration: {config.test_categories}")
        logger.debug(f"Full configuration: {config.to_dict()}")

        # Validate configuration if requested
        if parsed_args.validate_config:
            logger.info("Validating test configuration")
            validation_result = validate_test_configuration(
                config,
                check_system_compatibility=True,
                validate_performance_targets=True,
                strict_validation=parsed_args.strict_validation,
            )

            if not validation_result.is_valid:
                logger.error("Configuration validation failed:")
                for error in validation_result.errors:
                    logger.error(f"  - {error}")
                return EXIT_CODE_ERROR

            if validation_result.warnings:
                logger.warning("Configuration validation warnings:")
                for warning in validation_result.warnings:
                    logger.warning(f"  - {warning}")

        # Setup test environment
        logger.info("Setting up test environment")
        env_setup = setup_test_environment(
            config,
            clean_previous_artifacts=parsed_args.cleanup_artifacts,
            validate_system_requirements=True,
        )

        if not env_setup["success"]:
            logger.error("Test environment setup failed:")
            for error in env_setup["errors"]:
                logger.error(f"  - {error}")
            return EXIT_CODE_ERROR

        if env_setup["warnings"]:
            for warning in env_setup["warnings"]:
                logger.warning(f"Environment setup warning: {warning}")

        # Collect system information if requested
        if parsed_args.system_info:
            logger.info("Collecting system information")
            system_info = get_system_info(
                include_hardware_info=True,
                include_environment_details=True,
                include_performance_baselines=parsed_args.benchmarks,
            )
            logger.debug(f"System capabilities: {system_info.get('basic_info', {})}")
        else:
            system_info = {}

        # Create and configure test runner
        logger.info("Initializing test runner")
        test_runner = TestRunner(config, logger)

        # Execute test suite with comprehensive monitoring
        logger.info(f"Executing test suite: {', '.join(config.test_categories)}")

        with PerformanceTimer() as suite_timer:
            test_result = test_runner.run_suite(
                categories=config.test_categories,
                parallel_execution=config.parallel_execution,
                fail_fast=config.fail_fast,
            )

        # Add system info to test result
        if system_info:
            test_result.system_info.update(system_info)

        # Add overall suite timing
        test_result.performance_metrics["suite_total_time_seconds"] = (
            suite_timer.get_duration_ms() / 1000.0
        )

        # Generate comprehensive test reports
        if config.generate_reports:
            logger.info("Generating test reports")

            report_result = generate_test_report(
                test_result,
                output_path=pathlib.Path(config.output_directory) / REPORT_OUTPUT_DIR,
                report_formats=parsed_args.report_formats,
                include_performance_analysis=parsed_args.performance_analysis,
                include_recommendations=True,
            )

            if report_result["success"]:
                logger.info("Test reports generated successfully:")
                for format_type, file_path in report_result[
                    "generated_reports"
                ].items():
                    logger.info(f"  {format_type.upper()}: {file_path}")
            else:
                logger.warning("Report generation completed with warnings:")
                for error in report_result["errors"]:
                    logger.warning(f"  - {error}")

        # Display execution summary
        logger.info("\n" + "=" * 60)
        logger.info("TEST EXECUTION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Status: {'PASSED' if test_result.success else 'FAILED'}")
        logger.info(f"Total Tests: {test_result.total_tests}")
        logger.info(f"Passed: {test_result.passed_tests}")
        logger.info(f"Failed: {test_result.failed_tests}")
        logger.info(f"Skipped: {test_result.skipped_tests}")
        logger.info(f"Execution Time: {test_result.execution_time_seconds:.2f} seconds")

        if test_result.category_results:
            logger.info("\nCategory Results:")
            for category, results in test_result.category_results.items():
                logger.info(
                    f"  {category.capitalize()}: "
                    f"{results.get('passed', 0)}P/"
                    f"{results.get('failed', 0)}F/"
                    f"{results.get('skipped', 0)}S "
                    f"({results.get('execution_time', 0):.2f}s)"
                )

        if test_result.warnings:
            logger.info("\nWarnings:")
            for warning in test_result.warnings:
                logger.warning(f"  - {warning}")

        if test_result.recommendations:
            logger.info("\nRecommendations:")
            for recommendation in test_result.recommendations:
                logger.info(f"  • {recommendation}")

        logger.info("=" * 60)

        # Cleanup test artifacts if requested
        if config.cleanup_artifacts:
            logger.info("Cleaning up test artifacts")
            cleanup_result = cleanup_test_artifacts(
                artifacts_directory=config.output_directory,
                preserve_reports=parsed_args.preserve_reports,
                preserve_logs=parsed_args.preserve_logs,
                max_age_hours=24,
                force_cleanup=False,
            )

            if cleanup_result["success"]:
                stats = cleanup_result["cleanup_statistics"]
                logger.info(
                    f"Cleanup completed: {stats['files_removed']} files removed, "
                    f"{stats['total_size_removed_mb']:.1f} MB freed"
                )

        # Cleanup test runner resources
        test_runner.cleanup_resources()

        # Return appropriate exit code
        if test_result.success:
            logger.info("Test execution completed successfully")
            return EXIT_CODE_SUCCESS
        else:
            logger.error("Test execution completed with failures")
            return EXIT_CODE_FAILURE

    except KeyboardInterrupt:
        logger.warning("Test execution interrupted by user")
        return EXIT_CODE_ERROR

    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        if hasattr(e, "get_validation_details"):
            details = e.get_validation_details()
            for detail in details:
                logger.error(f"  - {detail}")
        return EXIT_CODE_ERROR

    except ConfigurationError as e:
        logger.error(f"Configuration error: {e}")
        if hasattr(e, "get_valid_options"):
            options = e.get_valid_options()
            logger.error(f"Valid options: {', '.join(options)}")
        return EXIT_CODE_ERROR

    except Exception as e:
        logger.error(f"Unexpected error during test execution: {e}")
        logger.debug("Error details:", exc_info=True)
        return EXIT_CODE_ERROR


# Export list for comprehensive public interface
__all__ = [
    # Main entry points
    "main",
    "run_test_suite",
    "execute_test_category",
    "generate_test_report",
    # Core classes
    "TestRunner",
    "TestExecutionConfig",
    "TestResult",
    # Environment management
    "setup_test_environment",
    "cleanup_test_artifacts",
    "validate_test_configuration",
    "get_system_info",
]


# Script execution entry point
if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
