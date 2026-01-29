#!/usr/bin/env python3
"""
Comprehensive installation validation script for plume_nav_sim package providing automated
verification of package installation, dependency availability, API compliance, basic
functionality testing, performance benchmarking, and system requirements checking with
detailed error reporting and troubleshooting guidance for users and developers.

This validation script serves as a comprehensive diagnostic tool for verifying proper
installation and functionality of the plume_nav_sim package. It performs systematic
validation of all system components, dependencies, and performance characteristics
while providing detailed troubleshooting guidance for common issues.

Usage:
    python validate_installation.py [OPTIONS]

Options:
    --verbose       Enable detailed validation output and diagnostic information
    --quiet         Minimal output with errors and warnings only
    --quick         Basic validation without extended testing and performance benchmarking
    --performance   Enable comprehensive performance testing and benchmarking
    --format        Output format selection (text, json, markdown)
    --timeout       Validation timeout in seconds (default: 300)
    --help          Display usage information and examples

Exit Codes:
    0 - All validation checks passed successfully
    1 - Critical validation failures detected
    2 - Warnings present but basic functionality available
"""

import argparse  # >=3.10 - Command-line argument parsing for validation configuration and execution parameters
import importlib  # >=3.10 - Dynamic module importing for dependency availability checking and version detection
import os  # >=3.10 - Environment variable access, path checking, and system resource detection for installation validation
import platform  # >=3.10 - System platform detection for cross-platform compatibility validation and system information reporting

# Standard library imports with version comments
import sys  # >=3.10 - Python version checking, exit codes, and system information access for compatibility validation
import time  # >=3.10 - Performance timing measurements for basic latency validation and benchmarking operations
import traceback  # >=3.10 - Error traceback capture for comprehensive debugging and error reporting
from typing import (  # >=3.10 - Type hints for function parameters, return types, and comprehensive validation result structures
    Any,
    Dict,
    List,
    Optional,
    Tuple,
)

# Third-party imports with version comments and error handling
try:
    import psutil  # >=5.9.0 - Memory usage monitoring and system resource validation for performance and resource constraint checking

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

try:
    import gymnasium  # >=0.29.0 - Gymnasium framework validation including gym.make() functionality and environment registry access for API compliance testing

    GYMNASIUM_AVAILABLE = True
except ImportError:
    GYMNASIUM_AVAILABLE = False
    gymnasium = None

try:
    import numpy as np  # >=2.1.0 - NumPy functionality validation including array operations, data types, and mathematical functions for core dependency verification

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    import matplotlib  # >=3.9.0 - Matplotlib backend detection and basic plotting functionality validation for rendering capability verification
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    matplotlib = None
    plt = None

# Internal imports with comprehensive error handling
try:
    # Main package interface imports for version checking, package information, and initialization validation
    from plume_nav_sim import (
        ENV_ID,
        create_plume_search_env,
        get_package_info,
        get_version,
        is_registered,
        register_env,
        unregister_env,
    )

    PLUME_NAV_SIM_AVAILABLE = True
except ImportError as e:
    PLUME_NAV_SIM_AVAILABLE = False
    plume_nav_sim_import_error = str(e)

try:
    # Core constants for validation targets and constraints
    from plume_nav_sim.core.constants import (
        DEFAULT_GRID_SIZE,
        MEMORY_LIMIT_TOTAL_MB,
        PACKAGE_VERSION,
        PERFORMANCE_TARGET_RGB_RENDER_MS,
        PERFORMANCE_TARGET_STEP_LATENCY_MS,
    )

    CONSTANTS_AVAILABLE = True
except ImportError:
    CONSTANTS_AVAILABLE = False

# Global validation configuration constants
VALIDATION_SUCCESS_EXIT_CODE = 0
VALIDATION_FAILURE_EXIT_CODE = 1
VALIDATION_WARNING_EXIT_CODE = 2
REQUIRED_PYTHON_VERSION = (3, 10)
VALIDATION_TIMEOUT_SECONDS = 300.0
BASIC_PERFORMANCE_TEST_STEPS = 100
RENDERING_TEST_FRAMES = 5
MEMORY_USAGE_SAMPLE_INTERVAL = 0.1
VERBOSE_OUTPUT = False
QUIET_MODE = False

# Global validation results and system information storage
_validation_results = {}
_system_info = {}


def main(args: Optional[List[str]] = None) -> int:
    """
    Main validation script entry point providing comprehensive installation validation with
    command-line argument processing, systematic validation execution, detailed reporting,
    and appropriate exit codes for automated testing and user feedback.

    This function orchestrates the complete validation workflow, processing command-line
    arguments, executing systematic validation checks, compiling comprehensive reports,
    and providing appropriate exit codes for integration with CI/CD systems and user workflows.

    Args:
        args: Optional command-line arguments list for testing and programmatic usage

    Returns:
        int: Exit code - 0 for success, 1 for failure, 2 for warnings, with comprehensive validation status
    """
    global VERBOSE_OUTPUT, QUIET_MODE

    try:
        # Parse command-line arguments including --verbose, --quiet, --quick, --performance flags
        parsed_args = parse_command_line_arguments(args or sys.argv[1:])

        # Set global flags based on parsed arguments for consistent behavior throughout validation
        VERBOSE_OUTPUT = parsed_args.get("verbose", False)
        QUIET_MODE = parsed_args.get("quiet", False)

        # Initialize global validation results dictionary and system information collection
        global _validation_results, _system_info
        _validation_results = {
            "start_time": time.time(),
            "python_version": None,
            "system_resources": None,
            "package_installation": None,
            "dependencies": None,
            "environment_functionality": None,
            "registration_system": None,
            "performance_metrics": None,
            "overall_status": "running",
        }
        _system_info = {
            "platform": platform.platform(),
            "python_executable": sys.executable,
            "python_path": sys.path[:3],  # First 3 paths for brevity
            "environment_variables": {
                "PATH": os.environ.get("PATH", ""),
                "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
                "DISPLAY": os.environ.get("DISPLAY", "not_set"),
            },
        }

        # Display validation banner with package version and system information
        if not QUIET_MODE:
            display_validation_banner()

        # Execute systematic validation workflow with progress reporting and error tracking
        validation_success = True
        validation_warnings = 0

        try:
            # Run check_python_version() validation with compatibility verification
            if VERBOSE_OUTPUT:
                print("🔍 Checking Python version compatibility...")

            python_result = check_python_version(strict_checking=True)
            _validation_results["python_version"] = python_result

            if not python_result:
                validation_success = False
                if not QUIET_MODE:
                    print("❌ Python version check failed")
            elif VERBOSE_OUTPUT:
                print("✅ Python version compatible")

            # Execute check_system_resources() for memory and platform validation
            if VERBOSE_OUTPUT:
                print("🔍 Checking system resources and platform compatibility...")

            system_adequate, system_info = check_system_resources(
                check_performance_capability=True, estimate_resource_usage=True
            )
            _validation_results["system_resources"] = {
                "adequate": system_adequate,
                "details": system_info,
            }

            if not system_adequate:
                validation_success = False
                if not QUIET_MODE:
                    print("❌ System resources inadequate")
            elif VERBOSE_OUTPUT:
                print("✅ System resources adequate")

            # Run check_package_installation() for package availability and version verification
            if VERBOSE_OUTPUT:
                print("🔍 Validating plume_nav_sim package installation...")

            package_result = check_package_installation(
                check_installation_path=True, verify_package_integrity=True
            )
            _validation_results["package_installation"] = package_result

            if not package_result:
                validation_success = False
                if not QUIET_MODE:
                    print("❌ Package installation validation failed")
            elif VERBOSE_OUTPUT:
                print("✅ Package installation valid")

            # Execute check_dependencies() for external library validation and version checking
            if VERBOSE_OUTPUT:
                print("🔍 Checking external dependencies...")

            dependency_results = check_dependencies(
                check_optional_dependencies=True, test_integration_compatibility=True
            )
            _validation_results["dependencies"] = dependency_results

            # Check if all required dependencies are available
            required_deps = ["gymnasium", "numpy", "matplotlib"]
            missing_required = [
                dep for dep in required_deps if not dependency_results.get(dep, False)
            ]

            if missing_required:
                validation_success = False
                if not QUIET_MODE:
                    print(f"❌ Required dependencies missing: {missing_required}")
            elif VERBOSE_OUTPUT:
                print("✅ All dependencies available")

            # Run validate_environment_functionality() for basic API and functionality testing
            if validation_success:  # Only if previous checks passed
                if VERBOSE_OUTPUT:
                    print("🔍 Testing environment functionality...")

                env_result = validate_environment_functionality(
                    test_extended_functionality=not parsed_args.get("quick", False),
                    num_test_steps=BASIC_PERFORMANCE_TEST_STEPS,
                )
                _validation_results["environment_functionality"] = env_result

                if not env_result:
                    validation_success = False
                    if not QUIET_MODE:
                        print("❌ Environment functionality test failed")
                elif VERBOSE_OUTPUT:
                    print("✅ Environment functionality validated")

            # Execute validate_registration_system() for Gymnasium integration validation
            if validation_success:  # Only if previous checks passed
                if VERBOSE_OUTPUT:
                    print("🔍 Validating registration system...")

                reg_result = validate_registration_system(
                    test_custom_parameters=True, cleanup_registration=True
                )
                _validation_results["registration_system"] = reg_result

                if not reg_result:
                    validation_success = False
                    if not QUIET_MODE:
                        print("❌ Registration system validation failed")
                elif VERBOSE_OUTPUT:
                    print("✅ Registration system validated")

            # Run basic performance validation if --performance flag enabled
            if parsed_args.get("performance", False) and validation_success:
                if VERBOSE_OUTPUT:
                    print("🔍 Running performance benchmarks...")

                perf_metrics = run_basic_performance_validation(
                    test_duration_steps=BASIC_PERFORMANCE_TEST_STEPS * 2,
                    test_rendering_performance=True,
                    monitor_memory_usage=PSUTIL_AVAILABLE,
                )
                _validation_results["performance_metrics"] = perf_metrics

                # Check if performance meets targets
                step_time = perf_metrics.get("average_step_time_ms", 0)
                if step_time > PERFORMANCE_TARGET_STEP_LATENCY_MS * 2:
                    validation_warnings += 1
                    if not QUIET_MODE:
                        print(
                            f"⚠️ Performance warning: Step time {step_time:.2f}ms exceeds target"
                        )
                elif VERBOSE_OUTPUT:
                    print(
                        f"✅ Performance acceptable: {step_time:.2f}ms average step time"
                    )

        except KeyboardInterrupt:
            if not QUIET_MODE:
                print("\n⚠️ Validation interrupted by user")
            _validation_results["overall_status"] = "interrupted"
            return VALIDATION_WARNING_EXIT_CODE

        except Exception as e:
            if not QUIET_MODE:
                print(f"\n❌ Unexpected error during validation: {e}")
                if VERBOSE_OUTPUT:
                    traceback.print_exc()
            _validation_results["overall_status"] = "error"
            return VALIDATION_FAILURE_EXIT_CODE

        # Compile comprehensive validation report with detailed findings and recommendations
        _validation_results["end_time"] = time.time()
        _validation_results["duration_seconds"] = (
            _validation_results["end_time"] - _validation_results["start_time"]
        )

        if validation_success:
            _validation_results["overall_status"] = (
                "success" if validation_warnings == 0 else "success_with_warnings"
            )
        else:
            _validation_results["overall_status"] = "failed"

        # Display validation summary with pass/fail status, warnings, and troubleshooting guidance
        if not QUIET_MODE:
            report = generate_validation_report(
                include_system_details=VERBOSE_OUTPUT,
                include_troubleshooting_guide=not validation_success,
                report_format=parsed_args.get("format", "text"),
            )
            print("\n" + "=" * 80)
            print(report)

        # Display troubleshooting guidance if validation failed
        if not validation_success:
            failed_components = []
            if not _validation_results.get("python_version"):
                failed_components.append("python_version")
            if not _validation_results.get("system_resources", {}).get("adequate"):
                failed_components.append("system_resources")
            if not _validation_results.get("package_installation"):
                failed_components.append("package_installation")
            if missing_required:
                failed_components.extend(missing_required)

            display_troubleshooting_guide(
                failed_validations=failed_components,
                show_platform_specific_guidance=True,
                include_advanced_troubleshooting=VERBOSE_OUTPUT,
            )

        # Return appropriate exit code based on validation results and error severity
        if validation_success:
            return (
                VALIDATION_SUCCESS_EXIT_CODE
                if validation_warnings == 0
                else VALIDATION_WARNING_EXIT_CODE
            )
        else:
            return VALIDATION_FAILURE_EXIT_CODE

    except Exception as e:
        if not QUIET_MODE:
            print(f"💥 Critical error in validation script: {e}")
            if VERBOSE_OUTPUT:
                traceback.print_exc()
        return VALIDATION_FAILURE_EXIT_CODE


def check_python_version(strict_checking: bool = False) -> bool:
    """
    Validates Python version compatibility ensuring minimum Python 3.10 requirements with
    comprehensive version checking, feature availability validation, and compatibility
    warnings for optimal package functionality.

    This function performs comprehensive Python version validation, checking not only the
    version number but also the availability of required Python features and standard
    library modules essential for package operation.

    Args:
        strict_checking: Enable enhanced validation rigor and comprehensive error checking

    Returns:
        bool: True if Python version meets requirements, False with detailed error information
    """
    try:
        # Get current Python version using sys.version_info with major and minor version extraction
        current_version = sys.version_info
        major, minor = current_version.major, current_version.minor

        if VERBOSE_OUTPUT:
            print(f"   Python version: {major}.{minor}.{current_version.micro}")
            print(f"   Python implementation: {platform.python_implementation()}")
            print(f"   Python compiler: {platform.python_compiler()}")

        # Compare against REQUIRED_PYTHON_VERSION (3, 10) with strict or relaxed checking modes
        required_major, required_minor = REQUIRED_PYTHON_VERSION

        if major < required_major or (
            major == required_major and minor < required_minor
        ):
            if not QUIET_MODE:
                print(
                    f"❌ Python {major}.{minor} is below minimum requirement {required_major}.{required_minor}"
                )
            return False

        # Validate Python implementation (CPython vs PyPy) for compatibility verification
        implementation = platform.python_implementation()
        if implementation != "CPython" and strict_checking:
            if not QUIET_MODE:
                print(f"⚠️ Non-CPython implementation detected: {implementation}")
            # Don't fail for non-CPython, just warn

        # Check for required Python features including typing module, dataclasses, and asyncio
        required_modules = [
            "typing",
            "dataclasses",
            "asyncio",
            "functools",
            "itertools",
            "collections",
        ]
        missing_modules = []

        for module_name in required_modules:
            try:
                importlib.import_module(module_name)
            except ImportError:
                missing_modules.append(module_name)

        if missing_modules:
            if not QUIET_MODE:
                print(
                    f"❌ Missing required standard library modules: {missing_modules}"
                )
            return False

        # Test Python installation integrity with basic import and execution validation
        try:
            # Test basic operations that the package will use
            import json
            import os
            import time

            test_data = {"test": True, "timestamp": time.time()}
            json_str = json.dumps(test_data)
            parsed_data = json.loads(json_str)
            assert parsed_data["test"] is True

            # Test file system operations
            current_dir = os.getcwd()
            assert os.path.exists(current_dir)

        except Exception as e:
            if not QUIET_MODE:
                print(f"❌ Python installation integrity test failed: {e}")
            return False

        # Add Python version information to _system_info global for reporting
        _system_info.update(
            {
                "python_version": f"{major}.{minor}.{current_version.micro}",
                "python_implementation": implementation,
                "python_compiler": platform.python_compiler(),
                "python_build": platform.python_build(),
                "python_executable": sys.executable,
            }
        )

        if VERBOSE_OUTPUT:
            print(f"   ✅ Python version {major}.{minor} meets requirements")

        return True

    except Exception as e:
        if not QUIET_MODE:
            print(f"❌ Python version check failed with error: {e}")
        return False


def check_system_resources(
    check_performance_capability: bool = False, estimate_resource_usage: bool = False
) -> Tuple[bool, Dict[str, Any]]:
    """
    Validates system resource availability including memory capacity, CPU capabilities,
    disk space, and platform compatibility ensuring adequate resources for plume_nav_sim
    operation with performance estimation and resource recommendations.

    This function provides comprehensive system resource validation, checking memory
    availability, CPU capabilities, platform compatibility, and optional performance
    characteristics to ensure optimal package operation.

    Args:
        check_performance_capability: Flag for performance capability assessment
        estimate_resource_usage: Flag to estimate memory usage for default configuration

    Returns:
        Tuple[bool, Dict[str, Any]]: Tuple of (resource_adequate, resource_info) with detailed system analysis
    """
    resource_info = {
        "platform": platform.system(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "memory_available_mb": None,
        "memory_total_mb": None,
        "cpu_count": os.cpu_count(),
        "display_available": None,
        "matplotlib_backend": None,
        "estimated_memory_usage_mb": None,
        "resource_warnings": [],
    }

    resource_adequate = True

    try:
        # Detect system platform using platform.system() for compatibility validation
        system_platform = platform.system()

        if VERBOSE_OUTPUT:
            print(f"   Platform: {system_platform} {platform.release()}")
            print(f"   Architecture: {platform.machine()}")
            print(f"   CPU count: {resource_info['cpu_count']}")

        # Check available system memory using psutil.virtual_memory() against MEMORY_LIMIT_TOTAL_MB
        if PSUTIL_AVAILABLE and psutil:
            try:
                memory = psutil.virtual_memory()
                available_mb = memory.available / (1024 * 1024)
                total_mb = memory.total / (1024 * 1024)

                resource_info.update(
                    {
                        "memory_available_mb": available_mb,
                        "memory_total_mb": total_mb,
                        "memory_percent_used": memory.percent,
                    }
                )

                if VERBOSE_OUTPUT:
                    print(
                        f"   Memory: {available_mb:.0f}MB available / {total_mb:.0f}MB total ({memory.percent:.1f}% used)"
                    )

                # Check against memory limits
                if available_mb < MEMORY_LIMIT_TOTAL_MB:
                    resource_info["resource_warnings"].append(
                        f"Available memory ({available_mb:.0f}MB) below recommended ({MEMORY_LIMIT_TOTAL_MB}MB)"
                    )
                    resource_adequate = False

            except Exception as e:
                resource_info["resource_warnings"].append(f"Memory check failed: {e}")
        else:
            resource_info["resource_warnings"].append(
                "psutil not available - cannot check memory usage"
            )

        # Validate CPU information and core count for performance capability assessment
        cpu_count = os.cpu_count()
        if cpu_count is None or cpu_count < 2:
            resource_info["resource_warnings"].append(
                "CPU core count appears low or unavailable"
            )

        # Check available disk space for package installation and data storage requirements
        try:
            if system_platform != "Windows":
                # Unix-like systems
                statvfs = os.statvfs("/")
                free_bytes = statvfs.f_bavail * statvfs.f_frsize
                free_mb = free_bytes / (1024 * 1024)
                resource_info["disk_free_mb"] = free_mb

                if VERBOSE_OUTPUT:
                    print(f"   Disk space: {free_mb:.0f}MB available")

                if free_mb < 100:  # Need at least 100MB
                    resource_info["resource_warnings"].append("Low disk space")
        except Exception:
            # Disk space check not critical
            pass

        # Test matplotlib backend availability for rendering capability validation
        if MATPLOTLIB_AVAILABLE and matplotlib:
            try:
                import matplotlib.pyplot as plt

                backend = matplotlib.get_backend()
                resource_info["matplotlib_backend"] = backend

                if VERBOSE_OUTPUT:
                    print(f"   Matplotlib backend: {backend}")

                # Test basic plotting capability
                fig, ax = plt.subplots(figsize=(1, 1))
                ax.plot([0, 1], [0, 1])
                plt.close(fig)

                resource_info["matplotlib_functional"] = True

            except Exception as e:
                resource_info["matplotlib_functional"] = False
                resource_info["resource_warnings"].append(
                    f"Matplotlib backend issue: {e}"
                )

        # Validate display capability for interactive rendering with DISPLAY environment variable
        display_env = os.environ.get("DISPLAY")
        resource_info["display_available"] = display_env is not None

        if system_platform != "Windows" and not display_env:
            resource_info["resource_warnings"].append(
                "No DISPLAY environment variable - GUI rendering may not work"
            )

        # Estimate memory usage for default grid size configuration using resource calculation
        if estimate_resource_usage:
            try:
                if CONSTANTS_AVAILABLE:
                    width, height = DEFAULT_GRID_SIZE
                    grid_cells = width * height
                    # Estimate memory for float32 arrays
                    estimated_mb = (grid_cells * 4 + 1024) / (
                        1024 * 1024
                    )  # Add 1KB overhead
                    resource_info["estimated_memory_usage_mb"] = estimated_mb

                    if VERBOSE_OUTPUT:
                        print(
                            f"   Estimated memory usage: {estimated_mb:.1f}MB for {width}x{height} grid"
                        )

                    if estimated_mb > MEMORY_LIMIT_TOTAL_MB:
                        resource_info["resource_warnings"].append(
                            f"Estimated usage ({estimated_mb:.1f}MB) exceeds limit ({MEMORY_LIMIT_TOTAL_MB}MB)"
                        )
                        resource_adequate = False
            except Exception:
                pass

        # Perform basic performance capability testing if check_performance_capability enabled
        if check_performance_capability:
            try:
                # Simple CPU benchmark
                start_time = time.perf_counter()
                sum(i * i for i in range(10000))
                cpu_time = (time.perf_counter() - start_time) * 1000

                resource_info["cpu_benchmark_ms"] = cpu_time

                if VERBOSE_OUTPUT:
                    print(f"   CPU benchmark: {cpu_time:.2f}ms")

                if cpu_time > 100:  # Very slow CPU
                    resource_info["resource_warnings"].append("CPU appears to be slow")

            except Exception:
                pass

        # Add system architecture (x86_64, ARM) for compatibility verification
        architecture = platform.machine().lower()
        supported_archs = ["x86_64", "amd64", "i386", "i686", "aarch64", "arm64"]

        if not any(arch in architecture for arch in supported_archs):
            resource_info["resource_warnings"].append(
                f"Unknown architecture: {architecture}"
            )

        # Add system resource information to _system_info for comprehensive reporting
        _system_info.update({"system_resources": resource_info})

        # Check for critical warnings that should fail validation
        critical_issues = [
            w
            for w in resource_info["resource_warnings"]
            if "memory" in w.lower() and "below" in w.lower()
        ]

        if critical_issues and resource_adequate:
            resource_adequate = False

        if VERBOSE_OUTPUT and resource_info["resource_warnings"]:
            print("   Resource warnings:")
            for warning in resource_info["resource_warnings"]:
                print(f"     ⚠️ {warning}")

        return resource_adequate, resource_info

    except Exception as e:
        resource_info["resource_warnings"].append(f"System resource check failed: {e}")
        return False, resource_info


def check_package_installation(
    check_installation_path: bool = False, verify_package_integrity: bool = False
) -> bool:
    """
    Validates plume_nav_sim package installation including module accessibility, version
    verification, package integrity, and component availability with comprehensive import
    testing and installation diagnostics.

    This function performs comprehensive validation of the package installation, checking
    not only basic import capability but also version consistency, component availability,
    and installation integrity.

    Args:
        check_installation_path: Flag to check package installation path and accessibility
        verify_package_integrity: Flag to verify all core modules are accessible

    Returns:
        bool: True if package installed correctly, False with detailed installation diagnostic information
    """
    try:
        # Attempt to import plume_nav_sim package with comprehensive error handling and diagnostic reporting
        if not PLUME_NAV_SIM_AVAILABLE:
            if not QUIET_MODE:
                print(f"❌ Cannot import plume_nav_sim: {plume_nav_sim_import_error}")
            return False

        if VERBOSE_OUTPUT:
            print("   ✅ plume_nav_sim package import successful")

        # Validate package version using get_version() function against expected PACKAGE_VERSION
        try:
            version = get_version()
            if CONSTANTS_AVAILABLE and version != PACKAGE_VERSION:
                if not QUIET_MODE:
                    print(
                        f"⚠️ Version mismatch: imported {version}, expected {PACKAGE_VERSION}"
                    )

            if VERBOSE_OUTPUT:
                print(f"   Package version: {version}")

        except Exception as e:
            if not QUIET_MODE:
                print(f"❌ Version check failed: {e}")
            return False

        # Check package installation path and accessibility using importlib and package introspection
        if check_installation_path:
            try:
                import plume_nav_sim

                package_path = plume_nav_sim.__file__
                package_dir = os.path.dirname(package_path)

                if VERBOSE_OUTPUT:
                    print(f"   Installation path: {package_dir}")

                # Check if path is accessible and readable
                if not os.path.exists(package_dir):
                    if not QUIET_MODE:
                        print(f"❌ Package directory not found: {package_dir}")
                    return False

                # Check for editable installation
                if (
                    "site-packages" not in package_dir
                    and "dist-packages" not in package_dir
                ):
                    if VERBOSE_OUTPUT:
                        print("   ✅ Editable installation detected")
                else:
                    if VERBOSE_OUTPUT:
                        print("   ✅ Standard installation detected")

            except Exception as e:
                if not QUIET_MODE:
                    print(f"⚠️ Installation path check failed: {e}")

        # Verify all core modules are accessible including envs, registration, utils, core, and render
        if verify_package_integrity:
            core_modules = [
                "plume_nav_sim.envs",
                "plume_nav_sim.envs.plume_search_env",
                "plume_nav_sim.registration",
                "plume_nav_sim.registration.register",
                "plume_nav_sim.core",
                "plume_nav_sim.core.constants",
                "plume_nav_sim.core.types",
                "plume_nav_sim.utils",
                "plume_nav_sim.utils.validation",
            ]

            missing_modules = []
            for module_name in core_modules:
                try:
                    importlib.import_module(module_name)
                except ImportError:
                    missing_modules.append(module_name)

            if missing_modules:
                if not QUIET_MODE:
                    print(f"❌ Missing core modules: {missing_modules}")
                return False
            elif VERBOSE_OUTPUT:
                print(f"   ✅ All {len(core_modules)} core modules accessible")

        # Test package initialization using initialize_package() function with error tracking
        try:
            # Test basic package functionality
            package_info = get_package_info(include_environment_info=False)

            if VERBOSE_OUTPUT:
                print(f"   Package name: {package_info.get('package_name', 'unknown')}")
                print(
                    f"   Environment ID: {package_info.get('environment_id', 'unknown')}"
                )

            # Test environment constants
            if CONSTANTS_AVAILABLE:
                from plume_nav_sim.core.constants import validate_constant_consistency

                is_valid, report = validate_constant_consistency(strict_mode=False)

                if not is_valid:
                    if not QUIET_MODE:
                        print("⚠️ Package constants consistency issues detected")
                elif VERBOSE_OUTPUT:
                    print("   ✅ Package constants validated")

        except Exception as e:
            if not QUIET_MODE:
                print(f"❌ Package initialization test failed: {e}")
            return False

        # Test package constants and configuration loading with validation and consistency checking
        try:
            # Import and validate critical constants
            if CONSTANTS_AVAILABLE:
                required_constants = [
                    "PACKAGE_VERSION",
                    "DEFAULT_GRID_SIZE",
                    "DEFAULT_SOURCE_LOCATION",
                    "PERFORMANCE_TARGET_STEP_LATENCY_MS",
                    "MEMORY_LIMIT_TOTAL_MB",
                ]

                from plume_nav_sim.core import constants

                missing_constants = []

                for const_name in required_constants:
                    if not hasattr(constants, const_name):
                        missing_constants.append(const_name)

                if missing_constants:
                    if not QUIET_MODE:
                        print(f"❌ Missing required constants: {missing_constants}")
                    return False
                elif VERBOSE_OUTPUT:
                    print("   ✅ All required constants available")

        except Exception as e:
            if not QUIET_MODE:
                print(f"⚠️ Constants validation failed: {e}")

        if VERBOSE_OUTPUT:
            print("   ✅ Package installation validation completed successfully")

        return True

    except Exception as e:
        if not QUIET_MODE:
            print(f"❌ Package installation check failed: {e}")
        return False


def check_dependencies(
    check_optional_dependencies: bool = False,
    test_integration_compatibility: bool = False,
) -> Dict[str, bool]:
    """
    Comprehensive dependency validation checking availability, version compatibility, and
    functionality of gymnasium, numpy, and matplotlib with feature testing, version
    comparison, and integration compatibility verification.

    This function provides thorough validation of all package dependencies, testing not
    only their availability but also version compatibility and functional integration
    with the package requirements.

    Args:
        check_optional_dependencies: Flag to check optional dependencies like psutil
        test_integration_compatibility: Flag to test cross-library operations and compatibility

    Returns:
        Dict[str, bool]: Dictionary mapping dependency names to availability status with detailed version and compatibility information
    """
    dependency_status = {"gymnasium": False, "numpy": False, "matplotlib": False}

    try:
        # Check gymnasium availability and version using importlib with >=0.29.0 requirement validation
        if GYMNASIUM_AVAILABLE:
            try:
                import gymnasium

                gym_version = gymnasium.__version__

                if VERBOSE_OUTPUT:
                    print(f"   Gymnasium version: {gym_version}")

                # Test basic gymnasium functionality
                from gymnasium import spaces

                # Test basic space creation
                test_space = spaces.Discrete(4)
                test_action = test_space.sample()
                assert test_space.contains(test_action)

                dependency_status["gymnasium"] = True

                if VERBOSE_OUTPUT:
                    print("   ✅ Gymnasium functional")

            except Exception as e:
                if not QUIET_MODE:
                    print(f"❌ Gymnasium test failed: {e}")
                dependency_status["gymnasium"] = False
        else:
            if not QUIET_MODE:
                print("❌ Gymnasium not available")

        # Validate gymnasium.Env class availability and basic functionality for environment inheritance
        if dependency_status["gymnasium"]:
            try:
                from gymnasium.spaces import Box, Discrete

                # Test space creation that the package uses
                Discrete(4)
                Box(
                    low=0.0,
                    high=1.0,
                    shape=(1,),
                    dtype=np.float32 if NUMPY_AVAILABLE else float,
                )

                if VERBOSE_OUTPUT:
                    print("   ✅ Gymnasium spaces functional")

            except Exception as e:
                if not QUIET_MODE:
                    print(f"❌ Gymnasium Env class test failed: {e}")
                dependency_status["gymnasium"] = False

        # Check numpy availability and version compatibility with >=2.1.0 requirement and Python 3.10+ support
        if NUMPY_AVAILABLE:
            try:
                numpy_version = np.__version__

                if VERBOSE_OUTPUT:
                    print(f"   NumPy version: {numpy_version}")

                # Test numpy array operations, data types (float32, uint8), and mathematical functions
                test_array = np.array([1.0, 2.0, 3.0], dtype=np.float32)
                test_result = np.sqrt(test_array)
                assert test_result.dtype == np.float32

                # Test uint8 operations for rendering
                rgb_array = np.zeros((10, 10, 3), dtype=np.uint8)
                rgb_array[5, 5] = [255, 0, 0]  # Red pixel
                assert rgb_array[5, 5, 0] == 255

                # Test mathematical functions
                test_gaussian = np.exp(-((np.arange(10) - 5) ** 2) / (2 * 2**2))
                assert len(test_gaussian) == 10
                assert test_gaussian[5] == 1.0  # Peak at center

                dependency_status["numpy"] = True

                if VERBOSE_OUTPUT:
                    print("   ✅ NumPy functional")

            except Exception as e:
                if not QUIET_MODE:
                    print(f"❌ NumPy test failed: {e}")
                dependency_status["numpy"] = False
        else:
            if not QUIET_MODE:
                print("❌ NumPy not available")

        # Validate matplotlib availability and version with >=3.9.0 requirement for rendering capabilities
        if MATPLOTLIB_AVAILABLE:
            try:
                mpl_version = matplotlib.__version__

                if VERBOSE_OUTPUT:
                    print(f"   Matplotlib version: {mpl_version}")

                # Test matplotlib backend availability and basic plotting functionality for rendering validation
                import matplotlib.pyplot as plt

                backend = plt.get_backend()

                if VERBOSE_OUTPUT:
                    print(f"   Matplotlib backend: {backend}")

                # Test basic plotting functionality
                fig, ax = plt.subplots(figsize=(2, 2))

                if NUMPY_AVAILABLE:
                    x = np.linspace(0, 1, 10)
                    y = x**2
                    ax.plot(x, y)

                    # Test imshow functionality for concentration field rendering
                    test_field = np.random.random((5, 5))
                    ax.imshow(test_field, cmap="gray")

                plt.close(fig)

                dependency_status["matplotlib"] = True

                if VERBOSE_OUTPUT:
                    print("   ✅ Matplotlib functional")

            except Exception as e:
                if not QUIET_MODE:
                    print(f"❌ Matplotlib test failed: {e}")
                dependency_status["matplotlib"] = False
        else:
            if not QUIET_MODE:
                print("❌ Matplotlib not available")

        # Check optional dependencies including psutil for resource monitoring if check_optional_dependencies enabled
        if check_optional_dependencies:
            if PSUTIL_AVAILABLE:
                try:
                    memory_info = psutil.virtual_memory()
                    dependency_status["psutil"] = True

                    if VERBOSE_OUTPUT:
                        print(
                            f"   ✅ psutil functional (Memory: {memory_info.percent:.1f}% used)"
                        )

                except Exception as e:
                    if VERBOSE_OUTPUT:
                        print(f"⚠️ psutil test failed: {e}")
                    dependency_status["psutil"] = False
            else:
                if VERBOSE_OUTPUT:
                    print("   ⚠️ psutil not available (optional)")
                dependency_status["psutil"] = False

        # Test integration compatibility between dependencies with cross-library operations if enabled
        if test_integration_compatibility and all(
            [dependency_status["gymnasium"], dependency_status["numpy"]]
        ):
            try:
                # Test gymnasium + numpy integration
                from gymnasium.spaces import Box

                # Create observation space that uses numpy
                obs_space = Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
                sample_obs = obs_space.sample()

                assert isinstance(sample_obs, np.ndarray)
                assert sample_obs.dtype == np.float32
                assert obs_space.contains(sample_obs)

                if VERBOSE_OUTPUT:
                    print("   ✅ Gymnasium + NumPy integration functional")

                # Test matplotlib + numpy integration if matplotlib available
                if dependency_status["matplotlib"]:
                    fig, ax = plt.subplots(figsize=(1, 1))
                    data = np.random.random((3, 3))
                    ax.imshow(data, cmap="gray")
                    plt.close(fig)

                    if VERBOSE_OUTPUT:
                        print("   ✅ Matplotlib + NumPy integration functional")

            except Exception as e:
                if VERBOSE_OUTPUT:
                    print(f"⚠️ Integration compatibility test failed: {e}")

        # Generate dependency compatibility matrix with version information and feature availability
        if VERBOSE_OUTPUT:
            total_deps = len([k for k in dependency_status.keys() if not k == "psutil"])
            available_deps = len(
                [k for k, v in dependency_status.items() if v and k != "psutil"]
            )
            print(f"   Dependencies available: {available_deps}/{total_deps}")

        return dependency_status

    except Exception as e:
        if not QUIET_MODE:
            print(f"❌ Dependency check failed: {e}")
        return dependency_status


def validate_environment_functionality(
    test_extended_functionality: bool = False,
    num_test_steps: int = BASIC_PERFORMANCE_TEST_STEPS,
) -> bool:
    """
    Comprehensive environment functionality validation testing PlumeSearchEnv basic operations,
    API compliance, state management, and core functionality with error handling and detailed
    validation reporting.

    This function performs thorough testing of the environment implementation, validating
    Gymnasium API compliance, state management, rendering capabilities, and mathematical
    correctness of the plume model.

    Args:
        test_extended_functionality: Flag to enable extended testing including multiple episodes
        num_test_steps: Number of steps to test for basic functionality validation

    Returns:
        bool: True if environment functionality validated successfully, False with detailed error analysis
    """
    try:
        if not PLUME_NAV_SIM_AVAILABLE:
            if not QUIET_MODE:
                print("❌ Cannot test environment - package not available")
            return False

        if VERBOSE_OUTPUT:
            print("   Testing environment creation...")

        # Create PlumeSearchEnv instance using default configuration with comprehensive error handling
        try:
            # Test both direct instantiation and factory function
            env = create_plume_search_env()

            if VERBOSE_OUTPUT:
                print("   ✅ Environment created successfully")

        except Exception as e:
            if not QUIET_MODE:
                print(f"❌ Environment creation failed: {e}")
            return False

        # Validate environment initialization including action_space, observation_space, and internal state
        try:
            # Check that spaces are properly defined
            action_space = env.action_space
            observation_space = env.observation_space

            if not hasattr(action_space, "n") or action_space.n != 4:
                if not QUIET_MODE:
                    print(
                        f"❌ Invalid action space: expected Discrete(4), got {action_space}"
                    )
                return False

            if not hasattr(observation_space, "shape") or observation_space.shape != (
                1,
            ):
                if not QUIET_MODE:
                    print(
                        f"❌ Invalid observation space: expected Box(shape=(1,)), got {observation_space}"
                    )
                return False

            if VERBOSE_OUTPUT:
                print(f"   Action space: {action_space}")
                print(f"   Observation space: {observation_space}")

        except Exception as e:
            if not QUIET_MODE:
                print(f"❌ Environment space validation failed: {e}")
            return False

        # Test environment reset() method with seed parameter returning proper (observation, info) tuple format
        try:
            # Test reset without seed
            obs, info = env.reset()

            # Validate observation format, data type (float32), and value range compliance with Box space
            if not isinstance(obs, np.ndarray):
                if not QUIET_MODE:
                    print(f"❌ Reset observation is not numpy array: {type(obs)}")
                return False

            if obs.shape != (1,):
                if not QUIET_MODE:
                    print(f"❌ Reset observation shape incorrect: {obs.shape}")
                return False

            if obs.dtype != np.float32:
                if not QUIET_MODE:
                    print(f"❌ Reset observation dtype incorrect: {obs.dtype}")
                return False

            if not (0.0 <= obs[0] <= 1.0):
                if not QUIET_MODE:
                    print(f"❌ Reset observation value out of range: {obs[0]}")
                return False

            if not isinstance(info, dict):
                if not QUIET_MODE:
                    print(f"❌ Reset info is not dict: {type(info)}")
                return False

            if VERBOSE_OUTPUT:
                print(
                    f"   ✅ Reset successful: obs={obs[0]:.4f}, info keys={list(info.keys())}"
                )

            # Test reset with seed for reproducibility
            obs1, info1 = env.reset(seed=42)
            obs2, info2 = env.reset(seed=42)

            if not np.allclose(obs1, obs2, atol=1e-6):
                if not QUIET_MODE:
                    print("❌ Reset with same seed not reproducible")
                return False

            if VERBOSE_OUTPUT:
                print("   ✅ Seeded reset reproducible")

        except Exception as e:
            if not QUIET_MODE:
                print(f"❌ Environment reset test failed: {e}")
            return False

        # Test environment step() method with valid actions returning proper 5-tuple format
        try:
            step_count = 0
            max_steps = min(num_test_steps, 50)  # Limit for basic test

            while step_count < max_steps:
                # Test all four actions
                for action in range(4):
                    # Test step() return values including observation, reward, terminated, truncated, and info dictionary
                    result = env.step(action)

                    if len(result) != 5:
                        if not QUIET_MODE:
                            print(f"❌ Step returned {len(result)} values, expected 5")
                        return False

                    obs, reward, terminated, truncated, info = result

                    # Validate observation
                    if (
                        not isinstance(obs, np.ndarray)
                        or obs.shape != (1,)
                        or obs.dtype != np.float32
                    ):
                        if not QUIET_MODE:
                            print(
                                f"❌ Step observation invalid: {type(obs)}, {obs.shape if hasattr(obs, 'shape') else 'no shape'}"
                            )
                        return False

                    if not (0.0 <= obs[0] <= 1.0):
                        if not QUIET_MODE:
                            print(f"❌ Step observation out of range: {obs[0]}")
                        return False

                    # Validate reward
                    if not isinstance(reward, (int, float)):
                        if not QUIET_MODE:
                            print(f"❌ Step reward not numeric: {type(reward)}")
                        return False

                    # Validate termination flags
                    if not isinstance(terminated, bool) or not isinstance(
                        truncated, bool
                    ):
                        if not QUIET_MODE:
                            print(
                                f"❌ Step termination flags not boolean: {type(terminated)}, {type(truncated)}"
                            )
                        return False

                    # Validate info dict
                    if not isinstance(info, dict):
                        if not QUIET_MODE:
                            print(f"❌ Step info not dict: {type(info)}")
                        return False

                    step_count += 1

                    # Check if episode ended
                    if terminated or truncated:
                        # Reset for next test
                        env.reset()
                        break

                if step_count >= max_steps:
                    break

            if VERBOSE_OUTPUT:
                print(f"   ✅ Step method validated over {step_count} steps")

        except Exception as e:
            if not QUIET_MODE:
                print(f"❌ Environment step test failed: {e}")
            return False

        # Test action space validation with valid actions (0-3) and invalid action error handling
        try:
            # Test valid actions
            for action in range(4):
                try:
                    env.step(action)
                except Exception as e:
                    if not QUIET_MODE:
                        print(f"❌ Valid action {action} failed: {e}")
                    return False

            # Test invalid action handling
            try:
                env.step(-1)  # Should handle gracefully or raise appropriate error
            except Exception:
                pass  # Expected to fail

            try:
                env.step(4)  # Should handle gracefully or raise appropriate error
            except Exception:
                pass  # Expected to fail

            if VERBOSE_OUTPUT:
                print("   ✅ Action validation functional")

        except Exception as e:
            if VERBOSE_OUTPUT:
                print(f"   ⚠️ Action validation test issues: {e}")

        # Test rendering functionality with rgb_array mode returning proper numpy array format
        try:
            # Test RGB array rendering
            rgb_array = env.render(mode="rgb_array")

            if rgb_array is not None:
                if not isinstance(rgb_array, np.ndarray):
                    if not QUIET_MODE:
                        print(f"❌ RGB render returned non-array: {type(rgb_array)}")
                    return False

                if len(rgb_array.shape) != 3 or rgb_array.shape[2] != 3:
                    if not QUIET_MODE:
                        print(f"❌ RGB render shape incorrect: {rgb_array.shape}")
                    return False

                if rgb_array.dtype != np.uint8:
                    if not QUIET_MODE:
                        print(f"❌ RGB render dtype incorrect: {rgb_array.dtype}")
                    return False

                if VERBOSE_OUTPUT:
                    print(f"   ✅ RGB rendering functional: {rgb_array.shape}")
            else:
                if VERBOSE_OUTPUT:
                    print("   ⚠️ RGB rendering returned None")

            # Test human rendering (may fail in headless environments)
            try:
                env.render(mode="human")
                if VERBOSE_OUTPUT:
                    print("   ✅ Human rendering functional")
            except Exception as e:
                if VERBOSE_OUTPUT:
                    print(
                        f"   ⚠️ Human rendering failed (may be expected in headless): {e}"
                    )

        except Exception as e:
            if not QUIET_MODE:
                print(f"❌ Rendering test failed: {e}")
            return False

        # Run extended functionality tests including multiple episodes and edge cases if enabled
        if test_extended_functionality:
            try:
                if VERBOSE_OUTPUT:
                    print("   Running extended functionality tests...")

                # Test multiple episodes
                for episode in range(3):
                    obs, info = env.reset(seed=episode)
                    episode_steps = 0
                    max_episode_steps = 100

                    while episode_steps < max_episode_steps:
                        action = episode_steps % 4  # Cycle through actions
                        obs, reward, terminated, truncated, info = env.step(action)
                        episode_steps += 1

                        if terminated or truncated:
                            break

                    if VERBOSE_OUTPUT and episode == 0:
                        print(
                            f"     Episode {episode}: {episode_steps} steps, final reward: {reward}"
                        )

                if VERBOSE_OUTPUT:
                    print("   ✅ Extended functionality tests passed")

            except Exception as e:
                if not QUIET_MODE:
                    print(f"❌ Extended functionality test failed: {e}")
                return False

        # Validate environment cleanup using close() method with resource deallocation
        try:
            env.close()
            if VERBOSE_OUTPUT:
                print("   ✅ Environment cleanup successful")
        except Exception as e:
            if VERBOSE_OUTPUT:
                print(f"   ⚠️ Environment cleanup warning: {e}")

        return True

    except Exception as e:
        if not QUIET_MODE:
            print(f"❌ Environment functionality validation failed: {e}")
        return False


def validate_registration_system(
    test_custom_parameters: bool = False, cleanup_registration: bool = True
) -> bool:
    """
    Validates Gymnasium environment registration system including registration functionality,
    gym.make() integration, environment discovery, and registry management with comprehensive
    integration testing.

    This function performs comprehensive validation of the environment registration system,
    testing not only basic registration but also parameter passing, registry management,
    and integration with the Gymnasium ecosystem.

    Args:
        test_custom_parameters: Flag to test custom parameter passing through registration
        cleanup_registration: Flag to clean up test registrations after validation

    Returns:
        bool: True if registration system functional, False with detailed registration diagnostic information
    """
    try:
        if not PLUME_NAV_SIM_AVAILABLE or not GYMNASIUM_AVAILABLE:
            if not QUIET_MODE:
                print("❌ Cannot test registration - required packages not available")
            return False

        if VERBOSE_OUTPUT:
            print("   Testing environment registration...")

        # Test environment registration using register_env() function with comprehensive error handling
        try:
            # First, clean up any existing registration
            if is_registered(ENV_ID):
                unregister_env(ENV_ID)

            # Register the environment
            registered_id = register_env()

            if registered_id != ENV_ID:
                if not QUIET_MODE:
                    print(f"❌ Registration returned unexpected ID: {registered_id}")
                return False

            if VERBOSE_OUTPUT:
                print(f"   ✅ Environment registered: {registered_id}")

        except Exception as e:
            if not QUIET_MODE:
                print(f"❌ Environment registration failed: {e}")
            return False

        # Validate registration status using is_registered() function for ENV_ID availability checking
        try:
            if not is_registered(ENV_ID):
                if not QUIET_MODE:
                    print(
                        f"❌ Environment not found in registry after registration: {ENV_ID}"
                    )
                return False

            if VERBOSE_OUTPUT:
                print(f"   ✅ Registration status confirmed: {ENV_ID}")

        except Exception as e:
            if not QUIET_MODE:
                print(f"❌ Registration status check failed: {e}")
            return False

        # Test gym.make() functionality with ENV_ID creating environment instance successfully
        try:
            import gymnasium as gym

            env = gym.make(ENV_ID)

            if not hasattr(env, "reset") or not hasattr(env, "step"):
                if not QUIET_MODE:
                    print(
                        "❌ gym.make() created invalid environment: missing reset/step methods"
                    )
                return False

            if VERBOSE_OUTPUT:
                print(f"   ✅ gym.make() successful: {type(env).__name__}")

            # Test basic functionality through gym.make
            obs, info = env.reset()
            obs, reward, terminated, truncated, info = env.step(0)

            env.close()

            if VERBOSE_OUTPUT:
                print("   ✅ gym.make() environment functional")

        except Exception as e:
            if not QUIET_MODE:
                print(f"❌ gym.make() test failed: {e}")
            return False

        # Test custom parameter passing through registration system if test_custom_parameters enabled
        if test_custom_parameters:
            try:
                if VERBOSE_OUTPUT:
                    print("   Testing custom parameter registration...")

                # Register with custom parameters
                custom_env_id = "TestPlume-Custom-v0"

                # Clean up any existing custom registration
                if is_registered(custom_env_id):
                    unregister_env(custom_env_id)

                # Test registration with custom grid size
                if CONSTANTS_AVAILABLE:
                    custom_params = {
                        "grid_size": (64, 64),
                        "source_location": (32, 32),
                        "goal_radius": 2.0,
                    }
                else:
                    custom_params = {}

                custom_id = register_env(env_id=custom_env_id, kwargs=custom_params)

                if custom_id != custom_env_id:
                    if not QUIET_MODE:
                        print(
                            f"❌ Custom registration failed: expected {custom_env_id}, got {custom_id}"
                        )
                    return False

                # Test custom environment creation
                custom_env = gym.make(custom_env_id)
                custom_obs, custom_info = custom_env.reset()
                custom_env.close()

                if VERBOSE_OUTPUT:
                    print(
                        f"   ✅ Custom parameter registration successful: {custom_env_id}"
                    )

                # Clean up custom registration
                if cleanup_registration:
                    unregister_env(custom_env_id)

            except Exception as e:
                if not QUIET_MODE:
                    print(f"❌ Custom parameter test failed: {e}")
                return False

        # Validate integration between plume_nav_sim registration and Gymnasium framework
        try:
            # Check that the environment is properly listed in the Gymnasium registry
            import gymnasium as gym

            # Get registry information
            if hasattr(gym.envs, "registry") and hasattr(
                gym.envs.registry, "env_specs"
            ):
                env_spec = gym.envs.registry.env_specs.get(ENV_ID)

                if env_spec is None:
                    if not QUIET_MODE:
                        print(
                            f"❌ Environment not found in Gymnasium registry: {ENV_ID}"
                        )
                    return False

                # Check entry point
                entry_point = getattr(env_spec, "entry_point", None)
                if not entry_point or "plume_nav_sim" not in entry_point:
                    if not QUIET_MODE:
                        print(f"❌ Invalid entry point in registry: {entry_point}")
                    return False

                if VERBOSE_OUTPUT:
                    print(f"   ✅ Registry integration verified: {entry_point}")
            else:
                if VERBOSE_OUTPUT:
                    print("   ⚠️ Cannot access Gymnasium registry for verification")

        except Exception as e:
            if VERBOSE_OUTPUT:
                print(f"   ⚠️ Registry integration check failed: {e}")

        # Test environment unregistration and cleanup if cleanup_registration enabled
        if cleanup_registration:
            try:
                success = unregister_env(ENV_ID)

                if not success:
                    if VERBOSE_OUTPUT:
                        print("   ⚠️ Unregistration reported failure")
                else:
                    if VERBOSE_OUTPUT:
                        print("   ✅ Environment unregistered successfully")

                # Verify unregistration
                if is_registered(ENV_ID):
                    if VERBOSE_OUTPUT:
                        print("   ⚠️ Environment still registered after unregistration")
                else:
                    if VERBOSE_OUTPUT:
                        print("   ✅ Unregistration verified")

            except Exception as e:
                if VERBOSE_OUTPUT:
                    print(f"   ⚠️ Cleanup failed: {e}")

        return True

    except Exception as e:
        if not QUIET_MODE:
            print(f"❌ Registration system validation failed: {e}")
        return False


def run_basic_performance_validation(
    test_duration_steps: int = BASIC_PERFORMANCE_TEST_STEPS,
    test_rendering_performance: bool = False,
    monitor_memory_usage: bool = False,
) -> Dict[str, float]:
    """
    Basic performance validation testing environment step latency, rendering performance,
    memory usage, and resource efficiency against established performance targets with
    benchmarking and optimization recommendations.

    This function provides comprehensive performance validation, measuring key metrics
    like step latency, rendering performance, and memory usage against established
    targets to ensure acceptable performance characteristics.

    Args:
        test_duration_steps: Number of steps to run for performance measurement
        test_rendering_performance: Flag to test RGB array rendering performance
        monitor_memory_usage: Flag to monitor memory usage during operations

    Returns:
        Dict[str, float]: Performance metrics dictionary with timing results, memory usage, and efficiency analysis
    """
    performance_metrics = {
        "test_duration_steps": test_duration_steps,
        "average_step_time_ms": 0.0,
        "min_step_time_ms": float("inf"),
        "max_step_time_ms": 0.0,
        "reset_time_ms": 0.0,
        "total_test_time_ms": 0.0,
        "steps_per_second": 0.0,
        "memory_usage_mb": 0.0,
        "memory_growth_mb": 0.0,
        "render_time_ms": 0.0,
        "performance_warnings": [],
    }

    try:
        if not PLUME_NAV_SIM_AVAILABLE:
            performance_metrics["performance_warnings"].append(
                "Package not available for testing"
            )
            return performance_metrics

        if VERBOSE_OUTPUT:
            print(f"   Running performance test: {test_duration_steps} steps")

        # Initialize performance metrics dictionary and create environment for testing
        env = create_plume_search_env()

        # Monitor initial memory usage
        initial_memory = 0.0
        if monitor_memory_usage and PSUTIL_AVAILABLE:
            try:
                process = psutil.Process()
                initial_memory = process.memory_info().rss / (1024 * 1024)  # MB
                performance_metrics["initial_memory_mb"] = initial_memory
            except Exception:
                monitor_memory_usage = False

        # Measure environment reset performance with timing measurement and target validation
        reset_start = time.perf_counter()
        obs, info = env.reset(seed=42)
        reset_time = (
            time.perf_counter() - reset_start
        ) * 1000  # Convert to milliseconds
        performance_metrics["reset_time_ms"] = reset_time

        if VERBOSE_OUTPUT:
            print(f"     Reset time: {reset_time:.3f}ms")

        # Run environment step performance test measuring latency over test_duration_steps
        step_times = []
        total_start = time.perf_counter()

        for step in range(test_duration_steps):
            action = step % 4  # Cycle through all actions

            step_start = time.perf_counter()
            obs, reward, terminated, truncated, info = env.step(action)
            step_end = time.perf_counter()

            step_time_ms = (step_end - step_start) * 1000
            step_times.append(step_time_ms)

            # Reset environment if episode ends
            if terminated or truncated:
                env.reset(seed=step)

        total_time = (time.perf_counter() - total_start) * 1000
        performance_metrics["total_test_time_ms"] = total_time

        # Calculate average, minimum, and maximum step execution times with statistical analysis
        if step_times:
            performance_metrics["average_step_time_ms"] = sum(step_times) / len(
                step_times
            )
            performance_metrics["min_step_time_ms"] = min(step_times)
            performance_metrics["max_step_time_ms"] = max(step_times)
            performance_metrics["steps_per_second"] = (
                test_duration_steps / total_time
            ) * 1000

            if VERBOSE_OUTPUT:
                print(
                    f"     Average step time: {performance_metrics['average_step_time_ms']:.3f}ms"
                )
                print(
                    f"     Step time range: {performance_metrics['min_step_time_ms']:.3f} - {performance_metrics['max_step_time_ms']:.3f}ms"
                )
                print(
                    f"     Steps per second: {performance_metrics['steps_per_second']:.1f}"
                )

        # Validate step latency against PERFORMANCE_TARGET_STEP_LATENCY_MS (1ms) target
        if CONSTANTS_AVAILABLE:
            avg_time = performance_metrics["average_step_time_ms"]
            target_time = PERFORMANCE_TARGET_STEP_LATENCY_MS

            if avg_time > target_time * 2:  # More than 2x target
                performance_metrics["performance_warnings"].append(
                    f"Step latency ({avg_time:.2f}ms) significantly exceeds target ({target_time}ms)"
                )
            elif avg_time > target_time:
                performance_metrics["performance_warnings"].append(
                    f"Step latency ({avg_time:.2f}ms) exceeds target ({target_time}ms)"
                )

        # Measure rendering performance for rgb_array mode if test_rendering_performance enabled
        if test_rendering_performance:
            try:
                render_times = []
                render_frames = min(RENDERING_TEST_FRAMES, 10)

                for frame in range(render_frames):
                    render_start = time.perf_counter()
                    env.render(mode="rgb_array")
                    render_end = time.perf_counter()

                    render_time_ms = (render_end - render_start) * 1000
                    render_times.append(render_time_ms)

                if render_times:
                    avg_render_time = sum(render_times) / len(render_times)
                    performance_metrics["render_time_ms"] = avg_render_time
                    performance_metrics["render_frames_tested"] = len(render_times)

                    if VERBOSE_OUTPUT:
                        print(
                            f"     Average render time: {avg_render_time:.3f}ms ({render_frames} frames)"
                        )

                    # Check against rendering performance target
                    render_target = (
                        PERFORMANCE_TARGET_RGB_RENDER_MS if CONSTANTS_AVAILABLE else 5.0
                    )

                    if avg_render_time > render_target * 2:
                        performance_metrics["performance_warnings"].append(
                            f"Render time ({avg_render_time:.2f}ms) significantly exceeds target ({render_target}ms)"
                        )

            except Exception as e:
                performance_metrics["performance_warnings"].append(
                    f"Rendering test failed: {e}"
                )

        # Monitor memory usage during operations using psutil if monitor_memory_usage enabled
        if monitor_memory_usage and PSUTIL_AVAILABLE:
            try:
                process = psutil.Process()
                final_memory = process.memory_info().rss / (1024 * 1024)  # MB
                performance_metrics["final_memory_mb"] = final_memory
                performance_metrics["memory_usage_mb"] = final_memory
                performance_metrics["memory_growth_mb"] = final_memory - initial_memory

                if VERBOSE_OUTPUT:
                    print(
                        f"     Memory usage: {final_memory:.1f}MB (growth: {performance_metrics['memory_growth_mb']:.1f}MB)"
                    )

                # Check against memory limits
                if CONSTANTS_AVAILABLE and final_memory > MEMORY_LIMIT_TOTAL_MB:
                    performance_metrics["performance_warnings"].append(
                        f"Memory usage ({final_memory:.1f}MB) exceeds limit ({MEMORY_LIMIT_TOTAL_MB}MB)"
                    )

            except Exception as e:
                performance_metrics["performance_warnings"].append(
                    f"Memory monitoring failed: {e}"
                )

        # Test memory efficiency and resource cleanup with garbage collection validation
        try:
            env.close()

            # Force garbage collection
            import gc

            gc.collect()

            if monitor_memory_usage and PSUTIL_AVAILABLE:
                try:
                    process = psutil.Process()
                    cleanup_memory = process.memory_info().rss / (1024 * 1024)  # MB
                    performance_metrics["cleanup_memory_mb"] = cleanup_memory
                    performance_metrics["memory_freed_mb"] = (
                        final_memory - cleanup_memory
                    )

                    if VERBOSE_OUTPUT:
                        print(
                            f"     Memory after cleanup: {cleanup_memory:.1f}MB (freed: {performance_metrics['memory_freed_mb']:.1f}MB)"
                        )
                except Exception:
                    pass

        except Exception as e:
            performance_metrics["performance_warnings"].append(
                f"Cleanup test failed: {e}"
            )

        # Generate performance summary
        if VERBOSE_OUTPUT:
            warning_count = len(performance_metrics["performance_warnings"])
            if warning_count == 0:
                print("   ✅ Performance validation completed - no warnings")
            else:
                print(
                    f"   ⚠️ Performance validation completed with {warning_count} warnings"
                )
                for warning in performance_metrics["performance_warnings"]:
                    print(f"     ⚠️ {warning}")

        return performance_metrics

    except Exception as e:
        performance_metrics["performance_warnings"].append(
            f"Performance validation failed: {e}"
        )
        if not QUIET_MODE:
            print(f"❌ Performance validation error: {e}")
        return performance_metrics


def generate_validation_report(
    include_system_details: bool = False,
    include_troubleshooting_guide: bool = False,
    report_format: str = "text",
) -> str:
    """
    Generates comprehensive validation report summarizing all validation results, system
    information, performance metrics, and troubleshooting guidance with formatted output
    for user feedback and diagnostic purposes.

    This function compiles all validation results into a comprehensive report suitable
    for display, logging, or external analysis, with multiple output formats and
    configurable detail levels.

    Args:
        include_system_details: Flag to include detailed system information
        include_troubleshooting_guide: Flag to include troubleshooting recommendations
        report_format: Format specification (text, markdown, json)

    Returns:
        str: Formatted validation report with comprehensive findings, status, and recommendations
    """
    try:
        if report_format.lower() == "json":
            # Return JSON format report
            import json

            report_data = {
                "validation_results": _validation_results,
                "system_info": _system_info if include_system_details else {},
                "timestamp": time.time(),
                "report_format": "json",
            }
            return json.dumps(report_data, indent=2, default=str)

        # Generate text/markdown format report
        report_lines = []

        # Add validation header and package information
        if report_format.lower() == "markdown":
            report_lines.append("# Plume Navigation Simulation Validation Report")
            report_lines.append("=" * 50)
        else:
            report_lines.append("PLUME NAVIGATION SIMULATION VALIDATION REPORT")
            report_lines.append("=" * 50)

        # Add timestamp and duration
        start_time = _validation_results.get("start_time", time.time())
        end_time = _validation_results.get("end_time", time.time())
        duration = _validation_results.get("duration_seconds", end_time - start_time)

        report_lines.append(
            f"Report generated: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}"
        )
        report_lines.append(f"Validation duration: {duration:.2f} seconds")
        report_lines.append("")

        # Add overall status
        overall_status = _validation_results.get("overall_status", "unknown")
        if overall_status == "success":
            status_icon = "✅"
            status_text = "SUCCESS - All validation checks passed"
        elif overall_status == "success_with_warnings":
            status_icon = "⚠️"
            status_text = "SUCCESS WITH WARNINGS - Basic functionality available"
        elif overall_status == "failed":
            status_icon = "❌"
            status_text = "FAILED - Critical issues detected"
        else:
            status_icon = "❓"
            status_text = f"UNKNOWN STATUS - {overall_status}"

        if report_format.lower() == "markdown":
            report_lines.append("## Overall Status")
            report_lines.append(f"{status_icon} **{status_text}**")
        else:
            report_lines.append(f"Overall Status: {status_icon} {status_text}")

        report_lines.append("")

        # Add detailed validation results
        if report_format.lower() == "markdown":
            report_lines.append("## Validation Results")
        else:
            report_lines.append("VALIDATION RESULTS:")
            report_lines.append("-" * 20)

        # Python version check
        python_result = _validation_results.get("python_version", False)
        python_icon = "✅" if python_result else "❌"
        report_lines.append(
            f"{python_icon} Python Version Compatibility: {'PASS' if python_result else 'FAIL'}"
        )

        # System resources check
        system_result = _validation_results.get("system_resources", {})
        system_adequate = system_result.get("adequate", False)
        system_icon = "✅" if system_adequate else "❌"
        report_lines.append(
            f"{system_icon} System Resources: {'ADEQUATE' if system_adequate else 'INADEQUATE'}"
        )

        # Package installation check
        package_result = _validation_results.get("package_installation", False)
        package_icon = "✅" if package_result else "❌"
        report_lines.append(
            f"{package_icon} Package Installation: {'VALID' if package_result else 'INVALID'}"
        )

        # Dependencies check
        deps_result = _validation_results.get("dependencies", {})
        required_deps = ["gymnasium", "numpy", "matplotlib"]
        deps_available = sum(1 for dep in required_deps if deps_result.get(dep, False))
        deps_icon = (
            "✅"
            if deps_available == len(required_deps)
            else "❌" if deps_available == 0 else "⚠️"
        )
        report_lines.append(
            f"{deps_icon} Dependencies: {deps_available}/{len(required_deps)} available"
        )

        # Environment functionality check
        env_result = _validation_results.get("environment_functionality", False)
        env_icon = "✅" if env_result else "❌"
        report_lines.append(
            f"{env_icon} Environment Functionality: {'VALIDATED' if env_result else 'FAILED'}"
        )

        # Registration system check
        reg_result = _validation_results.get("registration_system", False)
        reg_icon = "✅" if reg_result else "❌"
        report_lines.append(
            f"{reg_icon} Registration System: {'FUNCTIONAL' if reg_result else 'FAILED'}"
        )

        # Performance metrics if available
        perf_metrics = _validation_results.get("performance_metrics")
        if perf_metrics and isinstance(perf_metrics, dict):
            warnings_count = len(perf_metrics.get("performance_warnings", []))
            perf_icon = "✅" if warnings_count == 0 else "⚠️"

            avg_step_time = perf_metrics.get("average_step_time_ms", 0)
            if avg_step_time > 0:
                report_lines.append(
                    f"{perf_icon} Performance: {avg_step_time:.2f}ms avg step time"
                )
                if warnings_count > 0:
                    report_lines.append(f"  └─ {warnings_count} performance warnings")

        report_lines.append("")

        # Add system information summary if include_system_details enabled
        if include_system_details:
            if report_format.lower() == "markdown":
                report_lines.append("## System Information")
            else:
                report_lines.append("SYSTEM INFORMATION:")
                report_lines.append("-" * 18)

            system_info = _system_info

            # Platform information
            platform_info = system_info.get("platform", "unknown")
            report_lines.append(f"Platform: {platform_info}")

            # Python information
            python_version = system_info.get("python_version", "unknown")
            python_impl = system_info.get("python_implementation", "unknown")
            report_lines.append(f"Python: {python_version} ({python_impl})")

            # System resources
            sys_resources = system_info.get("system_resources", {})
            if sys_resources:
                memory_info = sys_resources.get("memory_total_mb")
                cpu_count = sys_resources.get("cpu_count")
                if memory_info:
                    report_lines.append(f"Memory: {memory_info:.0f}MB total")
                if cpu_count:
                    report_lines.append(f"CPU cores: {cpu_count}")

            # Package information
            if PLUME_NAV_SIM_AVAILABLE:
                try:
                    package_info = get_package_info()
                    package_version = package_info.get("version", "unknown")
                    report_lines.append(f"Package version: {package_version}")
                except Exception:
                    report_lines.append("Package version: unable to determine")

            report_lines.append("")

        # Add troubleshooting guidance if include_troubleshooting_guide enabled
        if include_troubleshooting_guide:
            if report_format.lower() == "markdown":
                report_lines.append("## Troubleshooting Recommendations")
            else:
                report_lines.append("TROUBLESHOOTING RECOMMENDATIONS:")
                report_lines.append("-" * 34)

            # Identify failed components
            failed_components = []
            if not _validation_results.get("python_version"):
                failed_components.append(
                    "Python version incompatible - upgrade to Python 3.10+"
                )

            if not _validation_results.get("system_resources", {}).get("adequate"):
                failed_components.append(
                    "Insufficient system resources - check memory availability"
                )

            if not _validation_results.get("package_installation"):
                failed_components.append(
                    "Package installation issues - reinstall with 'pip install -e .'"
                )

            deps_result = _validation_results.get("dependencies", {})
            missing_deps = [
                dep
                for dep in ["gymnasium", "numpy", "matplotlib"]
                if not deps_result.get(dep)
            ]
            if missing_deps:
                failed_components.append(
                    f"Missing dependencies: {', '.join(missing_deps)} - install with pip"
                )

            if not _validation_results.get("environment_functionality"):
                failed_components.append(
                    "Environment functionality issues - check package integrity"
                )

            if not _validation_results.get("registration_system"):
                failed_components.append(
                    "Registration system problems - verify Gymnasium installation"
                )

            # Add specific recommendations
            if failed_components:
                report_lines.append("Issues identified:")
                for i, issue in enumerate(failed_components, 1):
                    report_lines.append(f"{i}. {issue}")

                report_lines.append("")
                report_lines.append("Next steps:")
                report_lines.append(
                    "• Review error messages above for specific guidance"
                )
                report_lines.append(
                    "• Ensure all dependencies are installed: pip install gymnasium>=0.29.0 numpy>=2.1.0 matplotlib>=3.9.0"
                )
                report_lines.append(
                    "• Reinstall package: pip uninstall plume-nav-sim && pip install -e ."
                )
                report_lines.append(
                    "• Check Python version: python --version (requires 3.10+)"
                )
                report_lines.append(
                    "• For headless systems, set appropriate matplotlib backend"
                )
            else:
                report_lines.append("No critical issues identified.")
                report_lines.append("Check performance warnings if present.")

            report_lines.append("")

        # Add validation summary
        if report_format.lower() == "markdown":
            report_lines.append("## Summary")
        else:
            report_lines.append("SUMMARY:")
            report_lines.append("-" * 8)

        if overall_status == "success":
            report_lines.append(
                "✅ Validation completed successfully. Package is ready for use."
            )
        elif overall_status == "success_with_warnings":
            report_lines.append(
                "⚠️ Validation completed with warnings. Basic functionality is available."
            )
        else:
            report_lines.append(
                "❌ Validation failed. Please address the issues above before using the package."
            )

        return "\n".join(report_lines)

    except Exception as e:
        return f"Error generating validation report: {e}"


def display_troubleshooting_guide(
    failed_validations: List[str],
    show_platform_specific_guidance: bool = False,
    include_advanced_troubleshooting: bool = False,
) -> None:
    """
    Displays comprehensive troubleshooting guidance for common installation and validation
    issues with specific error solutions, platform-specific guidance, and step-by-step
    resolution instructions.

    This function provides detailed troubleshooting assistance tailored to the specific
    validation failures detected, offering platform-specific solutions and advanced
    debugging techniques.

    Args:
        failed_validations: List of failed validation components for targeted troubleshooting
        show_platform_specific_guidance: Flag to include platform-specific guidance
        include_advanced_troubleshooting: Flag to include advanced debugging information
    """
    if QUIET_MODE:
        return

    print("\n" + "🔧" * 20 + " TROUBLESHOOTING GUIDE " + "🔧" * 20)
    print()

    # Display troubleshooting header with guidance overview and common issue categories
    print("Common issues and solutions:")
    print()

    # Iterate through failed_validations list providing specific solutions for each failure type
    if "python_version" in failed_validations:
        print("❌ PYTHON VERSION ISSUES:")
        print("   Problem: Python version below minimum requirement (3.10)")
        print("   Solutions:")
        print("   • Install Python 3.10 or later from https://python.org")
        print("   • Use pyenv to manage Python versions: pyenv install 3.10.0")
        print("   • Check current version: python --version")
        print("   • Use virtual environment with correct Python version")
        print()

    if "system_resources" in failed_validations:
        print("❌ SYSTEM RESOURCE ISSUES:")
        print("   Problem: Insufficient memory or system resources")
        print("   Solutions:")
        print("   • Close unnecessary applications to free memory")
        print(
            "   • Check available memory: free -h (Linux) or Activity Monitor (macOS)"
        )
        print("   • Use smaller grid sizes for testing: grid_size=(64, 64)")
        print("   • Consider upgrading system memory")
        print()

    if "package_installation" in failed_validations:
        print("❌ PACKAGE INSTALLATION ISSUES:")
        print("   Problem: plume_nav_sim package not properly installed")
        print("   Solutions:")
        print("   • Install in development mode: pip install -e .")
        print(
            "   • Uninstall and reinstall: pip uninstall plume-nav-sim && pip install -e ."
        )
        print("   • Check installation path and permissions")
        print("   • Use virtual environment to avoid conflicts")
        print("   • Verify Python path includes package location")
        print()

    # Show dependency installation issues including pip install commands and virtual environment setup
    dependency_issues = [
        item
        for item in failed_validations
        if item in ["gymnasium", "numpy", "matplotlib"]
    ]
    if dependency_issues:
        print("❌ DEPENDENCY INSTALLATION ISSUES:")
        print(
            f"   Problem: Missing required dependencies: {', '.join(dependency_issues)}"
        )
        print("   Solutions:")
        print(
            "   • Install dependencies: pip install gymnasium>=0.29.0 numpy>=2.1.0 matplotlib>=3.9.0"
        )
        print("   • Upgrade pip: python -m pip install --upgrade pip")
        print("   • Use conda instead: conda install gymnasium numpy matplotlib")
        print("   • Check for conflicting versions: pip list | grep gymnasium")
        print("   • Create fresh virtual environment:")
        print("     python -m venv plume_env")
        print("     source plume_env/bin/activate  # Linux/macOS")
        print("     plume_env\\Scripts\\activate     # Windows")
        print("     pip install -r requirements.txt")
        print()

    # Display environment functionality issues including API errors and configuration problems
    if "environment_functionality" in failed_validations:
        print("❌ ENVIRONMENT FUNCTIONALITY ISSUES:")
        print("   Problem: Environment creation or operation failed")
        print("   Solutions:")
        print("   • Check all dependencies are properly installed")
        print(
            "   • Verify package integrity: python -c \"import plume_nav_sim; print('OK')\""
        )
        print("   • Try with default parameters first")
        print("   • Check for import errors in package modules")
        print("   • Enable debug logging for detailed error information")
        print()

    # Show registration system problems including Gymnasium integration issues and environment discovery
    if "registration_system" in failed_validations:
        print("❌ REGISTRATION SYSTEM ISSUES:")
        print("   Problem: Gymnasium environment registration failed")
        print("   Solutions:")
        print("   • Verify Gymnasium installation: pip install gymnasium>=0.29.0")
        print("   • Clear environment registry cache")
        print("   • Check for naming conflicts with existing environments")
        print("   • Try manual registration: register_env() before gym.make()")
        print("   • Restart Python interpreter to clear registry")
        print()

    # Show platform-specific guidance for Linux, macOS, Windows if show_platform_specific_guidance enabled
    if show_platform_specific_guidance:
        current_platform = platform.system()
        print(f"🖥️ PLATFORM-SPECIFIC GUIDANCE ({current_platform}):")
        print()

        if current_platform == "Linux":
            print("   Linux-specific solutions:")
            print(
                "   • Install system dependencies: sudo apt-get install python3-dev python3-tk"
            )
            print("   • For headless systems: export MPLBACKEND=Agg")
            print("   • Check DISPLAY variable: echo $DISPLAY")
            print("   • Use virtual display: Xvfb :99 -screen 0 1024x768x24 &")
            print(
                "   • Install additional packages: sudo apt-get install python3-matplotlib"
            )

        elif current_platform == "Darwin":  # macOS
            print("   macOS-specific solutions:")
            print("   • Install Xcode command line tools: xcode-select --install")
            print("   • Use Homebrew Python: brew install python@3.10")
            print("   • Set matplotlib backend: export MPLBACKEND=TkAgg")
            print("   • Install tkinter: brew install python-tk")
            print("   • Check PATH: echo $PATH")

        elif current_platform == "Windows":
            print("   Windows-specific solutions:")
            print("   • Use Python from python.org (not Microsoft Store)")
            print("   • Install Visual C++ redistributables")
            print("   • Use Command Prompt or PowerShell (not Git Bash)")
            print("   • Set matplotlib backend: set MPLBACKEND=TkAgg")
            print("   • Check Windows long path support")

        print()

    # Display advanced troubleshooting including debug mode activation and detailed logging if enabled
    if include_advanced_troubleshooting:
        print("🔬 ADVANCED TROUBLESHOOTING:")
        print()
        print("   Debug information collection:")
        print("   • Run with verbose output: python validate_installation.py --verbose")
        print("   • Enable debug logging:")
        print("     import logging; logging.basicConfig(level=logging.DEBUG)")
        print("   • Check import paths:")
        print("     python -c \"import sys; print('\\n'.join(sys.path))\"")
        print("   • Validate environment variables:")
        print("     python -c \"import os; print(os.environ.get('PYTHONPATH'))\"")
        print("   • Memory profiling:")
        print("     pip install memory_profiler")
        print("     python -m memory_profiler validate_installation.py")
        print()

        print("   Package debugging:")
        print(
            '   • Check package structure: python -c "import plume_nav_sim; print(plume_nav_sim.__file__)"'
        )
        print(
            '   • Validate constants: python -c "from plume_nav_sim.core.constants import *"'
        )
        print("   • Test minimal environment:")
        print(
            "     python -c \"from plume_nav_sim import create_plume_search_env; env = create_plume_search_env(); print('OK')\""
        )
        print()

    # Show community resources including documentation links, issue reporting, and support channels
    print("📚 ADDITIONAL RESOURCES:")
    print("   • Package documentation: Check README.md and docs/ directory")
    print("   • Gymnasium documentation: https://gymnasium.farama.org/")
    print("   • NumPy installation guide: https://numpy.org/install/")
    print(
        "   • Matplotlib installation: https://matplotlib.org/stable/users/installing/"
    )
    print(
        "   • Python environment management: https://docs.python.org/3/tutorial/venv.html"
    )
    print()

    # Display validation retry instructions and next steps for issue resolution
    print("🔄 NEXT STEPS:")
    print("   1. Address the specific issues listed above")
    print("   2. Restart your Python interpreter/terminal")
    print("   3. Re-run validation: python validate_installation.py --verbose")
    print("   4. If issues persist, create minimal reproduction case")
    print("   5. Report issues with full error output and system information")
    print()


def parse_command_line_arguments(args: List[str]) -> Dict[str, Any]:
    """
    Parses command-line arguments for validation script configuration including verbosity
    control, performance testing options, output formatting, and validation scope selection
    with comprehensive argument validation.

    This function processes command-line arguments to configure the validation script
    execution, providing flexible control over output verbosity, testing scope, and
    report formatting.

    Args:
        args: Command-line arguments list for parsing and validation

    Returns:
        Dict[str, Any]: Parsed arguments dictionary with validation configuration and execution parameters
    """
    try:
        # Initialize argument parser with script description and usage information
        parser = argparse.ArgumentParser(
            description="Comprehensive installation validation for plume_nav_sim package",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  python validate_installation.py --verbose
  python validate_installation.py --performance --format json
  python validate_installation.py --quick --quiet
  python validate_installation.py --help
            """,
        )

        # Add --verbose flag for detailed validation output and diagnostic information
        parser.add_argument(
            "--verbose",
            "-v",
            action="store_true",
            help="Enable detailed validation output and diagnostic information",
        )

        # Add --quiet flag for minimal output with errors and warnings only
        parser.add_argument(
            "--quiet",
            "-q",
            action="store_true",
            help="Minimal output with errors and warnings only",
        )

        # Add --quick flag for basic validation without extended testing and performance benchmarking
        parser.add_argument(
            "--quick",
            action="store_true",
            help="Basic validation without extended testing and performance benchmarking",
        )

        # Add --performance flag for comprehensive performance testing and benchmarking
        parser.add_argument(
            "--performance",
            "-p",
            action="store_true",
            help="Enable comprehensive performance testing and benchmarking",
        )

        # Add --format option for output format selection (text, json, markdown)
        parser.add_argument(
            "--format",
            "-f",
            choices=["text", "json", "markdown"],
            default="text",
            help="Output format selection (default: text)",
        )

        # Add --timeout option for validation timeout configuration with default values
        parser.add_argument(
            "--timeout",
            "-t",
            type=float,
            default=VALIDATION_TIMEOUT_SECONDS,
            help=f"Validation timeout in seconds (default: {VALIDATION_TIMEOUT_SECONDS})",
        )

        # Parse provided args list with error handling for invalid arguments and malformed input
        try:
            parsed_args = parser.parse_args(args)
        except SystemExit as e:
            # argparse calls sys.exit on error or --help
            if e.code == 0:  # --help was called
                sys.exit(0)
            else:  # parsing error
                print("❌ Invalid command line arguments")
                sys.exit(VALIDATION_FAILURE_EXIT_CODE)

        # Convert to dictionary for easier access
        args_dict = vars(parsed_args)

        # Validate argument combinations for consistency and compatibility checking
        if args_dict["verbose"] and args_dict["quiet"]:
            print(
                "⚠️ Warning: --verbose and --quiet are mutually exclusive, using --verbose"
            )
            args_dict["quiet"] = False

        if args_dict["quick"] and args_dict["performance"]:
            print(
                "⚠️ Warning: --quick and --performance are contradictory, using --performance"
            )
            args_dict["quick"] = False

        # Validate timeout value
        if args_dict["timeout"] <= 0:
            print("⚠️ Warning: Invalid timeout value, using default")
            args_dict["timeout"] = VALIDATION_TIMEOUT_SECONDS

        return args_dict

    except Exception as e:
        print(f"❌ Argument parsing failed: {e}")
        return {
            "verbose": False,
            "quiet": False,
            "quick": False,
            "performance": False,
            "format": "text",
            "timeout": VALIDATION_TIMEOUT_SECONDS,
        }


def display_validation_banner() -> None:
    """
    Displays validation banner with package information and system details.

    This function creates an informative banner that introduces the validation
    process and provides key package and system information.
    """
    if QUIET_MODE:
        return

    print()
    print("🚀" * 25)
    print("    PLUME NAVIGATION SIMULATION")
    print("      INSTALLATION VALIDATOR")
    print("🚀" * 25)
    print()

    # Display package version if available
    if PLUME_NAV_SIM_AVAILABLE:
        try:
            version = get_version()
            print(f"📦 Package Version: {version}")
        except Exception:
            print("📦 Package Version: unable to determine")
    else:
        print("📦 Package Version: not installed")

    # Display Python version
    python_version = (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
    print(f"🐍 Python Version: {python_version}")

    # Display platform
    system_platform = f"{platform.system()} {platform.release()}"
    print(f"🖥️ Platform: {system_platform}")

    print()
    print("Starting comprehensive validation...")
    print()


if __name__ == "__main__":
    """
    Main entry point for the validation script.

    This section handles direct script execution, processing command-line arguments
    and invoking the main validation function with appropriate error handling.
    """
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⚠️ Validation interrupted by user")
        sys.exit(VALIDATION_WARNING_EXIT_CODE)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(VALIDATION_FAILURE_EXIT_CODE)
