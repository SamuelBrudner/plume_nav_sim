"""
Scripts package initialization module for plume_nav_sim providing centralized access to development,
testing, validation, and maintenance scripts. Serves as the primary interface for accessing script
functionality including installation validation, development environment setup, test execution,
performance benchmarking, and system maintenance utilities with consistent API and error handling.

This module implements a unified interface to all plume_nav_sim development and maintenance scripts,
enabling seamless integration with development workflows, automated testing, environment setup,
and maintenance procedures through standardized wrapper functions and error handling.
"""

import functools  # >=3.10 - Function utilities for decorators and wrapper functions for script integration
import logging

# Standard library imports for system interface and type annotations
import sys  # >=3.10 - System interface for exit codes, command line arguments, and platform information access
import typing  # >=3.10 - Type hints for function signatures, return types, and comprehensive type annotation
from typing import Any, Callable, Dict, List, Optional, Union

# Internal script imports for cache cleanup utilities
from .clean_cache import main as clean_cache_main

# Internal script imports for documentation generation
from .generate_docs import main as generate_docs_main

# Internal script imports for test execution infrastructure
from .run_tests import TestExecutionConfig, TestResult, TestRunner
from .run_tests import main as run_tests_main
from .run_tests import run_test_suite

# Internal script imports for development environment setup
from .setup_dev_env import create_virtual_environment
from .setup_dev_env import main as setup_dev_env_main
from .setup_dev_env import setup_pre_commit_hooks

# Internal script imports for validation functionality
from .validate_installation import check_python_version
from .validate_installation import main as validate_installation_main
from .validate_installation import validate_environment_functionality

# Package version and metadata
__version__ = "1.0.0"

# Comprehensive export list for all script functionality and utilities
__all__ = [
    "validate_installation",
    "setup_development_environment",
    "run_tests",
    "run_performance_profiling",
    "clean_cache",
    "generate_documentation",
    "check_python_version",
    "validate_environment_functionality",
    "create_virtual_environment",
    "setup_pre_commit_hooks",
    "run_test_suite",
    "TestRunner",
    "TestExecutionConfig",
    "TestResult",
]

# Configure logging for script operations
logger = logging.getLogger("plume_nav_sim.scripts")

# Script registry mapping script names to entry points and metadata
SCRIPT_REGISTRY = {
    "validate_installation": validate_installation_main,
    "setup_dev_env": setup_dev_env_main,
    "run_tests": run_tests_main,
    "performance_profiling": None,  # Missing implementation - graceful handling
    "clean_cache": clean_cache_main,
    "generate_docs": generate_docs_main,
}

# Default configuration for script execution with consistent behavior
DEFAULT_SCRIPT_CONFIG = {
    "verbose": False,
    "quiet": False,
    "timeout_seconds": 300,
    "exit_on_error": True,
}


def validate_installation(
    args: Optional[List[str]] = None,
    verbose: bool = False,
    performance_tests: bool = False,
) -> int:
    """
    Wrapper function providing simplified interface to installation validation with consistent API
    and error handling for package verification and system compatibility checking.

    This function provides a simplified interface to the comprehensive installation validation
    functionality, enabling easy integration with development workflows and automated testing
    through standardized parameter handling and error reporting.

    Args:
        args (Optional[List[str]]): Optional command line arguments for validation configuration
        verbose (bool): Enable verbose output for detailed validation reporting and diagnostics
        performance_tests (bool): Include performance testing in validation for comprehensive checks

    Returns:
        int: Exit code indicating validation success (0), failure (1), or warnings (2)
    """
    logger.info("Starting installation validation")

    try:
        # Parse optional args parameter or use default validation arguments
        validation_args = args or []

        # Configure verbosity level based on verbose parameter and args
        if verbose and "--verbose" not in validation_args:
            validation_args.append("--verbose")

        # Enable performance testing if performance_tests parameter is True
        if performance_tests and "--performance-tests" not in validation_args:
            validation_args.append("--performance-tests")

        # Call validate_installation.main with processed arguments and configuration
        import sys

        original_argv = sys.argv.copy()
        try:
            sys.argv = ["validate_installation"] + validation_args
            exit_code = validate_installation_main()

            # Handle exceptions with graceful error reporting and exit code management
            logger.info(
                f"Installation validation completed with exit code: {exit_code}"
            )
            return exit_code or 0

        finally:
            sys.argv = original_argv

    except KeyboardInterrupt:
        logger.warning("Installation validation interrupted by user")
        return 2
    except Exception as e:
        logger.error(f"Installation validation failed with exception: {e}")
        return 1


def setup_development_environment(
    venv_name: Optional[str] = None,
    skip_validation: bool = False,
    force_recreate: bool = False,
) -> int:
    """
    Wrapper function providing simplified interface to development environment setup with consistent
    configuration and comprehensive error handling for automated development workflows.

    This function provides streamlined access to the development environment setup functionality,
    enabling automated development configuration with customizable options for virtual environment
    management and validation procedures.

    Args:
        venv_name (Optional[str]): Virtual environment name or use default environment configuration
        skip_validation (bool): Skip post-setup validation for faster setup in trusted environments
        force_recreate (bool): Force recreation of existing environment for clean setup

    Returns:
        int: Exit code indicating setup success (0), failure (1), or warnings (2)
    """
    logger.info("Starting development environment setup")

    try:
        # Prepare setup arguments using venv_name or default virtual environment name
        setup_args = []

        if venv_name:
            setup_args.extend(["--venv-name", venv_name])

        # Configure validation skip flag based on skip_validation parameter
        if skip_validation:
            setup_args.append("--skip-validation")

        # Set force recreation option for existing environment handling
        if force_recreate:
            setup_args.append("--force-recreate")

        # Call setup_dev_env.main with processed configuration and options
        import sys

        original_argv = sys.argv.copy()
        try:
            sys.argv = ["setup_dev_env"] + setup_args
            exit_code = setup_dev_env_main()

            # Handle setup exceptions with detailed error reporting and recovery suggestions
            logger.info(
                f"Development environment setup completed with exit code: {exit_code}"
            )
            return exit_code or 0

        finally:
            sys.argv = original_argv

    except KeyboardInterrupt:
        logger.warning("Development environment setup interrupted by user")
        return 2
    except Exception as e:
        logger.error(f"Development environment setup failed with exception: {e}")
        return 1


def run_tests(
    test_categories: Optional[List[str]] = None,
    parallel_execution: bool = False,
    performance_benchmarks: bool = False,
    output_directory: Optional[str] = None,
) -> int:
    """
    Wrapper function providing simplified interface to test execution with configuration options
    and comprehensive result reporting for development and CI/CD workflows.

    This function provides streamlined access to the comprehensive test execution infrastructure,
    enabling automated testing with configurable options for parallel execution, performance
    benchmarking, and result reporting.

    Args:
        test_categories (Optional[List[str]]): Test categories to execute or use default comprehensive categories
        parallel_execution (bool): Enable parallel test execution for improved performance
        performance_benchmarks (bool): Include performance benchmarking in test execution
        output_directory (Optional[str]): Output directory for test reports and artifacts

    Returns:
        int: Exit code indicating test success (0), failure (1), or errors (2)
    """
    logger.info("Starting test execution")

    try:
        # Process test_categories parameter or use default comprehensive test categories
        test_args = []

        if test_categories:
            test_args.extend(["--categories"] + test_categories)

        # Configure parallel execution based on system capabilities and parameter settings
        if parallel_execution:
            test_args.append("--parallel")

        # Enable performance benchmarking if performance_benchmarks parameter is True
        if performance_benchmarks:
            test_args.append("--performance")

        # Setup output directory for test reports and artifacts using provided or default path
        if output_directory:
            test_args.extend(["--output-dir", output_directory])

        # Call run_tests.main with configured test execution parameters
        import sys

        original_argv = sys.argv.copy()
        try:
            sys.argv = ["run_tests"] + test_args
            exit_code = run_tests_main()

            # Handle test execution exceptions with comprehensive error analysis and reporting
            logger.info(f"Test execution completed with exit code: {exit_code}")
            return exit_code or 0

        finally:
            sys.argv = original_argv

    except KeyboardInterrupt:
        logger.warning("Test execution interrupted by user")
        return 2
    except Exception as e:
        logger.error(f"Test execution failed with exception: {e}")
        return 1


def run_performance_profiling(
    benchmark_categories: Optional[List[str]] = None,
    generate_reports: bool = True,
    output_directory: Optional[str] = None,
) -> int:
    """
    Wrapper function providing simplified interface to performance profiling and benchmarking with
    configuration management and result analysis for system optimization.

    This function provides access to performance profiling functionality for system optimization
    analysis. Currently returns a graceful error as the performance profiling script is not
    implemented in the proof-of-life version.

    Args:
        benchmark_categories (Optional[List[str]]): Benchmark categories to execute or use defaults
        generate_reports (bool): Generate performance analysis reports and visualizations
        output_directory (Optional[str]): Output directory for profiling results and reports

    Returns:
        int: Exit code indicating profiling success (0), failure (1), or errors (2)
    """
    logger.warning(
        "Performance profiling script is not available in proof-of-life implementation"
    )

    # Process benchmark_categories or use default performance benchmarking categories
    categories = benchmark_categories or [
        "environment_performance",
        "rendering_benchmarks",
    ]
    logger.info(f"Requested benchmark categories: {categories}")

    # Configure report generation based on generate_reports parameter and output capabilities
    if generate_reports:
        logger.info(
            f"Report generation requested, output directory: {output_directory or 'default'}"
        )

    # Return appropriate exit code indicating performance profiling is not available
    logger.error(
        "Performance profiling functionality not implemented in proof-of-life version"
    )
    return 2  # Warning exit code for missing functionality


def clean_cache(
    preserve_reports: bool = False, force_cleanup: bool = False, max_age_hours: int = 24
) -> int:
    """
    Wrapper function providing simplified interface to cache and temporary file cleanup with
    configurable retention policies and comprehensive cleanup reporting.

    This function provides streamlined access to the cache and temporary file cleanup functionality,
    enabling automated development environment maintenance with customizable retention policies
    and comprehensive cleanup reporting.

    Args:
        preserve_reports (bool): Preserve test reports and analysis files during cleanup
        force_cleanup (bool): Force cleanup of all cache files regardless of age or usage
        max_age_hours (int): Maximum age in hours for files to be retained during cleanup

    Returns:
        int: Exit code indicating cleanup success (0), failure (1), or warnings (2)
    """
    logger.info("Starting cache cleanup")

    try:
        # Configure cleanup parameters including file preservation and retention policies
        cleanup_args = []

        if preserve_reports:
            cleanup_args.append("--preserve-reports")

        # Set force cleanup mode based on force_cleanup parameter for complete cleanup
        if force_cleanup:
            cleanup_args.append("--force")

        # Apply age-based cleanup using max_age_hours for selective file removal
        cleanup_args.extend(["--max-age", str(max_age_hours)])

        # Call clean_cache.main with configured cleanup parameters and retention settings
        import sys

        original_argv = sys.argv.copy()
        try:
            sys.argv = ["clean_cache"] + cleanup_args
            exit_code = clean_cache_main()

            # Handle cleanup exceptions with detailed error reporting and resource status
            logger.info(f"Cache cleanup completed with exit code: {exit_code}")
            return exit_code or 0

        finally:
            sys.argv = original_argv

    except KeyboardInterrupt:
        logger.warning("Cache cleanup interrupted by user")
        return 2
    except Exception as e:
        logger.error(f"Cache cleanup failed with exception: {e}")
        return 1


def generate_documentation(
    doc_formats: Optional[List[str]] = None,
    include_api_docs: bool = True,
    output_directory: Optional[str] = None,
) -> int:
    """
    Wrapper function providing simplified interface to documentation generation with format options
    and comprehensive processing for automated documentation workflows.

    This function provides streamlined access to the comprehensive documentation generation
    functionality, enabling automated documentation workflows with configurable output formats
    and processing options.

    Args:
        doc_formats (Optional[List[str]]): Documentation formats to generate or use defaults
        include_api_docs (bool): Include comprehensive API documentation in generation
        output_directory (Optional[str]): Output directory for generated documentation files

    Returns:
        int: Exit code indicating documentation generation success (0), failure (1), or warnings (2)
    """
    logger.info("Starting documentation generation")

    try:
        # Process doc_formats parameter or use default documentation formats
        doc_args = []

        if doc_formats:
            doc_args.extend(["--formats"] + doc_formats)

        # Configure API documentation inclusion based on include_api_docs parameter
        if include_api_docs:
            doc_args.append("--include-api")

        # Setup output directory for generated documentation and asset files
        if output_directory:
            doc_args.extend(["--output-dir", output_directory])

        # Call generate_docs.main with configured documentation generation parameters
        import sys

        original_argv = sys.argv.copy()
        try:
            sys.argv = ["generate_docs"] + doc_args
            exit_code = generate_docs_main()

            # Handle documentation generation exceptions with detailed error analysis
            logger.info(
                f"Documentation generation completed with exit code: {exit_code}"
            )
            return exit_code or 0

        finally:
            sys.argv = original_argv

    except KeyboardInterrupt:
        logger.warning("Documentation generation interrupted by user")
        return 2
    except Exception as e:
        logger.error(f"Documentation generation failed with exception: {e}")
        return 1


def get_script_registry() -> dict:
    """
    Returns registry of available scripts with their entry points and metadata for dynamic
    script discovery and execution management.

    This function provides access to the complete script registry, enabling dynamic script
    discovery, metadata retrieval, and programmatic script execution management for
    development tools and automation systems.

    Returns:
        dict: Dictionary mapping script names to entry points and metadata with execution information
    """
    # Return copy of SCRIPT_REGISTRY dictionary with script name to function mappings
    registry_copy = SCRIPT_REGISTRY.copy()

    # Include script metadata and descriptions for documentation and help systems
    script_metadata = {
        "validate_installation": {
            "function": registry_copy["validate_installation"],
            "description": "Comprehensive installation validation and system compatibility checking",
            "available": registry_copy["validate_installation"] is not None,
        },
        "setup_dev_env": {
            "function": registry_copy["setup_dev_env"],
            "description": "Automated development environment setup and configuration",
            "available": registry_copy["setup_dev_env"] is not None,
        },
        "run_tests": {
            "function": registry_copy["run_tests"],
            "description": "Comprehensive test execution with parallel processing and reporting",
            "available": registry_copy["run_tests"] is not None,
        },
        "performance_profiling": {
            "function": registry_copy["performance_profiling"],
            "description": "Performance profiling and benchmarking (not available in proof-of-life)",
            "available": registry_copy["performance_profiling"] is not None,
        },
        "clean_cache": {
            "function": registry_copy["clean_cache"],
            "description": "Cache and temporary file cleanup with retention policies",
            "available": registry_copy["clean_cache"] is not None,
        },
        "generate_docs": {
            "function": registry_copy["generate_docs"],
            "description": "Comprehensive documentation generation with multiple output formats",
            "available": registry_copy["generate_docs"] is not None,
        },
    }

    # Provide consistent interface for script discovery and dynamic execution
    return script_metadata


def run_script_by_name(
    script_name: str, args: Optional[List[str]] = None, config: Optional[dict] = None
) -> int:
    """
    Dynamic script execution by name with argument passing and consistent error handling for
    programmatic script invocation and automation workflows.

    This function enables dynamic script execution through name-based lookup with standardized
    argument passing and error handling, supporting automation workflows and programmatic
    script invocation with consistent behavior and reporting.

    Args:
        script_name (str): Name of script to execute from the available script registry
        args (Optional[List[str]]): Command line arguments to pass to the script
        config (Optional[dict]): Configuration dictionary to merge with default settings

    Returns:
        int: Exit code from executed script or error code for invalid script name
    """
    logger.info(f"Executing script by name: {script_name}")

    try:
        # Validate script_name exists in SCRIPT_REGISTRY with available script checking
        if script_name not in SCRIPT_REGISTRY:
            logger.error(f"Script '{script_name}' not found in registry")
            logger.info(f"Available scripts: {list(SCRIPT_REGISTRY.keys())}")
            return 1

        # Retrieve script function from registry using script name lookup
        script_function = SCRIPT_REGISTRY[script_name]

        if script_function is None:
            logger.error(
                f"Script '{script_name}' is not available (implementation missing)"
            )
            return 2

        # Merge provided config with DEFAULT_SCRIPT_CONFIG for complete configuration
        merged_config = DEFAULT_SCRIPT_CONFIG.copy()
        if config:
            merged_config.update(config)

        # Process args parameter and prepare for script execution with proper formatting
        script_args = args or []

        # Apply configuration options to arguments if supported
        if merged_config.get("verbose", False) and "--verbose" not in script_args:
            script_args.append("--verbose")
        if merged_config.get("quiet", False) and "--quiet" not in script_args:
            script_args.append("--quiet")

        # Execute script function with processed arguments and merged configuration
        import sys

        original_argv = sys.argv.copy()
        try:
            sys.argv = [script_name] + script_args
            exit_code = script_function()

            logger.info(f"Script '{script_name}' completed with exit code: {exit_code}")
            return exit_code or 0

        finally:
            sys.argv = original_argv

    except KeyboardInterrupt:
        logger.warning(f"Script '{script_name}' interrupted by user")
        return 2
    except Exception as e:
        # Handle script execution exceptions with consistent error reporting
        logger.error(f"Script '{script_name}' execution failed: {e}")
        return 1


def get_available_scripts(
    include_descriptions: bool = False,
) -> Union[List[str], List[dict]]:
    """
    Returns list of available script names with descriptions and usage information for help
    systems and script discovery interfaces.

    This function provides comprehensive script discovery functionality, enabling help systems,
    documentation tools, and user interfaces to present available scripts with appropriate
    metadata and usage information.

    Args:
        include_descriptions (bool): Include detailed descriptions and metadata for each script

    Returns:
        Union[List[str], List[dict]]: List of script names or list of dictionaries with script information
    """
    # Extract script names from SCRIPT_REGISTRY keys for basic script listing
    script_names = list(SCRIPT_REGISTRY.keys())

    if not include_descriptions:
        return script_names

    # Include script descriptions and metadata if include_descriptions is True
    script_info = []
    script_registry = get_script_registry()

    for script_name in script_names:
        metadata = script_registry.get(script_name, {})

        # Format script information as list of dictionaries with name, description, and usage
        script_info.append(
            {
                "name": script_name,
                "description": metadata.get("description", "No description available"),
                "available": metadata.get("available", False),
                "function": metadata.get("function") is not None,
            }
        )

    # Return simple list of names or comprehensive information based on parameter
    return script_info
