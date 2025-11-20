#!/usr/bin/env python3
"""
Cache cleaning utility script for plume_nav_sim development environment providing comprehensive cache cleanup
including Python bytecode cache, logging system cache, pytest cache, matplotlib rendering cache, performance
baselines, and temporary development files with safety checks and verbose reporting capabilities.

This script implements comprehensive cache management for development environments with backup capabilities,
rollback support, safety validation, and detailed reporting for maintaining clean development state.
"""

import argparse  # >=3.10 - Command-line argument parsing for cache cleanup options and verbosity control
import json  # >=3.10 - JSON serialization for backup manifests and statistics reporting
import os  # >=3.10 - Operating system interface for file system operations and environment variables

# External imports with version comments
import shutil  # >=3.10 - High-level file operations for directory tree removal and cache cleanup
import sys  # >=3.10 - System-specific parameters and functions for Python runtime cache management
import tempfile  # >=3.10 - Temporary file and directory management for cleanup operations
import time  # >=3.10 - Timing operations for cleanup performance measurement and reporting
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..plume_nav_sim.core.constants import (
    PACKAGE_NAME,  # Package identifier for cache directory identification and cleanup scope determination
)
from ..plume_nav_sim.logging.config import (
    DEFAULT_LOG_DIR,  # Default log directory path for log file cleanup and cache management
)

# Internal imports for logging and system integration
from ..plume_nav_sim.utils.logging import (
    ComponentType,  # Component type enumeration for logger configuration and script classification
)
from ..plume_nav_sim.utils.logging import (
    clear_logger_cache,  # Utility function for clearing cached logger instances with proper resource management
)
from ..plume_nav_sim.utils.logging import (
    get_component_logger,  # Create component-specific logger for cache cleanup operations
)

# Script identification constants
SCRIPT_NAME = "clean_cache"

# Cache type definitions with patterns and descriptions
CACHE_TYPES = {
    "python": {
        "patterns": ["**/__pycache__", "**/*.pyc", "**/*.pyo"],
        "description": "Python bytecode cache files",
    },
    "pytest": {"patterns": ["**/.pytest_cache"], "description": "pytest testing cache"},
    "logging": {
        "patterns": ["**/logs/*.log*", "**/plume_nav_sim*.log*"],
        "description": "Log files and logging cache",
    },
    "matplotlib": {
        "patterns": ["**/.matplotlib"],
        "description": "Matplotlib rendering cache",
    },
    "performance": {
        "patterns": ["**/performance_*.json", "**/benchmarks/*.tmp"],
        "description": "Performance measurement baselines and temporary files",
    },
    "temp": {
        "patterns": ["**/tmp_*", "**/.tmp", "**/temp_*"],
        "description": "Temporary development files",
    },
}

# Default safe patterns for cleanup operations
DEFAULT_SAFE_PATTERNS = [
    "**/__pycache__",
    "**/.pytest_cache",
    "**/logs/*.log*",
    "**/.matplotlib",
]

# Patterns to exclude from cleanup to protect important files
EXCLUDED_PATTERNS = [
    "**/src/backend/plume_nav_sim/**/*.py",
    "**/src/backend/tests/**/*.py",
    "**/src/backend/examples/**/*.py",
    "**/pyproject.toml",
    "**/README.md",
    "**/LICENSE",
]

# Backup and safety configuration constants
BACKUP_SUFFIX = ".cache_backup"
MAX_BACKUP_AGE_DAYS = 7


def main() -> int:
    """
    Main entry point for cache cleanup script providing command-line interface, argument parsing,
    and orchestration of cleanup operations with comprehensive reporting and error handling.

    Returns:
        int: Exit code (0 for success, 1 for errors, 2 for warnings)
    """
    # Initialize component logger using get_component_logger with ComponentType.UTILS
    logger = get_component_logger(
        component_name="clean_cache_script",
        component_type=ComponentType.UTILS,
        enable_performance_tracking=True,
    )

    try:
        logger.info(f"Starting {SCRIPT_NAME} - plume_nav_sim cache cleanup utility")

        # Parse command-line arguments using setup_argument_parser for cleanup options
        parser = setup_argument_parser()
        args = parser.parse_args()

        # Validate cleanup parameters and safety options from command line
        if args.cache_types:
            invalid_types = set(args.cache_types) - set(CACHE_TYPES.keys())
            if invalid_types:
                logger.error(
                    f"Invalid cache types: {invalid_types}. Valid types: {list(CACHE_TYPES.keys())}"
                )
                return 1

        # Initialize cleanup statistics tracking dictionary for reporting
        cleanup_stats = {
            "start_time": time.time(),
            "total_files_removed": 0,
            "total_directories_removed": 0,
            "total_space_reclaimed_mb": 0.0,
            "cache_type_results": {},
            "warnings": [],
            "errors": [],
            "backup_files": [],
            "dry_run_mode": args.dry_run,
        }

        # Set up root directory for cleanup operations
        root_dir = Path(args.target_directory) if args.target_directory else Path.cwd()

        # Create CacheCleanupManager instance for coordinated cleanup operations
        cleanup_manager = CacheCleanupManager(
            root_directory=root_dir,
            cleanup_config={
                "dry_run": args.dry_run,
                "verbose": args.verbose,
                "force": args.force,
                "backup_enabled": args.backup,
                "older_than_days": getattr(args, "older_than", None),
            },
            enable_backup=args.backup,
        )

        # Execute cleanup operations based on selected cache types and options
        selected_types = (
            args.cache_types if args.cache_types else list(CACHE_TYPES.keys())
        )

        if args.verbose:
            logger.info(f"Cache cleanup targeting types: {selected_types}")
            logger.info(f"Root directory: {root_dir}")
            logger.info(f"Dry run mode: {args.dry_run}")

        # Execute coordinated cleanup through manager
        cleanup_results = cleanup_manager.execute_cleanup(
            cache_types=selected_types, force_cleanup=args.force
        )

        # Update cleanup statistics with results
        cleanup_stats.update(cleanup_results)

        # Generate comprehensive cleanup report with statistics and warnings
        report = generate_cleanup_report(cleanup_stats, args.verbose, args.dry_run)

        if not args.quiet:
            print(report)

        # Log comprehensive summary
        logger.info(
            f"Cache cleanup completed: {cleanup_stats['total_files_removed']} files, "
            f"{cleanup_stats['total_directories_removed']} directories, "
            f"{cleanup_stats['total_space_reclaimed_mb']:.2f} MB reclaimed"
        )

        # Handle any errors or exceptions with proper logging and user feedback
        if cleanup_stats["errors"]:
            logger.error(
                f"Cleanup completed with {len(cleanup_stats['errors'])} errors"
            )
            if not args.quiet:
                print("\nERRORS ENCOUNTERED:")
                for error in cleanup_stats["errors"]:
                    print(f"  - {error}")
            return 1

        # Return appropriate exit code based on cleanup success and warning status
        if cleanup_stats["warnings"]:
            logger.warning(
                f"Cleanup completed with {len(cleanup_stats['warnings'])} warnings"
            )
            return 2

        return 0

    except KeyboardInterrupt:
        logger.warning("Cache cleanup interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error during cache cleanup: {e}", exception=e)
        return 1


def setup_argument_parser() -> argparse.ArgumentParser:
    """
    Configures command-line argument parser with comprehensive options for cache cleanup including
    selective cleanup, safety options, verbosity control, and dry-run capabilities.

    Returns:
        argparse.ArgumentParser: Configured argument parser with all cache cleanup options and help documentation
    """
    # Create ArgumentParser with description and usage information
    parser = argparse.ArgumentParser(
        prog=SCRIPT_NAME,
        description="Comprehensive cache cleanup utility for plume_nav_sim development environment",
        epilog=f"""
Examples:
  {SCRIPT_NAME}                                    # Clean all cache types
  {SCRIPT_NAME} --cache-types python pytest      # Clean only Python and pytest cache
  {SCRIPT_NAME} --dry-run --verbose               # Preview cleanup with detailed output
  {SCRIPT_NAME} --backup --force                  # Backup before aggressive cleanup
  {SCRIPT_NAME} --older-than 7                    # Clean files older than 7 days
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Add cache type selection arguments with choices from CACHE_TYPES keys
    parser.add_argument(
        "--cache-types",
        "-t",
        nargs="*",
        choices=list(CACHE_TYPES.keys()),
        help=f"Specific cache types to clean. Options: {list(CACHE_TYPES.keys())}. Default: all types",
    )

    # Add safety options including --dry-run and --backup flags
    parser.add_argument(
        "--dry-run",
        "-n",
        action="store_true",
        help="Preview cleanup operations without making changes",
    )

    parser.add_argument(
        "--backup",
        "-b",
        action="store_true",
        help="Create backup before cleanup for rollback capability",
    )

    # Add verbosity control with --verbose and --quiet options
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable detailed output and progress reporting",
    )

    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress output except for errors"
    )

    # Add force option for aggressive cleanup with --force flag
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Perform aggressive cleanup including locked files and active caches",
    )

    # Add target directory option for specifying cleanup root directory
    parser.add_argument(
        "--target-directory",
        "-d",
        type=str,
        help="Target directory for cleanup operations (default: current directory)",
    )

    # Add age-based cleanup options with --older-than parameter
    parser.add_argument(
        "--older-than",
        type=int,
        metavar="DAYS",
        help="Only clean files older than specified number of days",
    )

    # Add utility options
    parser.add_argument(
        "--list-cache-types",
        action="store_true",
        help="List available cache types and their descriptions",
    )

    return parser


def clean_python_cache(
    root_dir: Path, dry_run: bool = False, verbose: bool = False
) -> Dict[str, Any]:
    """
    Removes Python bytecode cache files including __pycache__ directories, .pyc files, and .pyo files
    with safe pattern matching and comprehensive reporting of cleanup operations.

    Args:
        root_dir: Root directory for cache cleanup operations
        dry_run: Whether to perform dry run without actual deletion
        verbose: Whether to enable verbose progress reporting

    Returns:
        dict: Dictionary containing cleanup statistics including files removed, directories cleaned, and space reclaimed
    """
    logger = get_component_logger("python_cache_cleaner", ComponentType.UTILS)

    # Initialize cleanup statistics tracking for Python cache files
    stats = {
        "files_removed": 0,
        "directories_removed": 0,
        "space_reclaimed_mb": 0.0,
        "cache_directories_found": 0,
        "errors": [],
        "warnings": [],
    }

    try:
        logger.debug(f"Starting Python cache cleanup in {root_dir}")

        # Find all __pycache__ directories using recursive glob pattern matching
        pycache_pattern = root_dir.rglob("__pycache__")
        pycache_dirs = list(pycache_pattern)
        stats["cache_directories_found"] = len(pycache_dirs)

        if verbose:
            logger.info(f"Found {len(pycache_dirs)} __pycache__ directories")

        # Calculate total size of files to be removed for space reporting
        total_size = 0
        for pycache_dir in pycache_dirs:
            try:
                size, file_count = calculate_directory_size(
                    pycache_dir, follow_symlinks=False
                )
                total_size += size
            except Exception as e:
                stats["errors"].append(f"Error calculating size for {pycache_dir}: {e}")

        # Remove __pycache__ directories using shutil.rmtree with error handling
        for pycache_dir in pycache_dirs:
            try:
                if verbose:
                    logger.debug(f"Removing __pycache__ directory: {pycache_dir}")

                if not dry_run:
                    # Validate directory is actually __pycache__ for safety
                    if pycache_dir.name == "__pycache__":
                        shutil.rmtree(pycache_dir, ignore_errors=False)
                        stats["directories_removed"] += 1
                    else:
                        stats["warnings"].append(
                            f"Skipped non-__pycache__ directory: {pycache_dir}"
                        )
                else:
                    stats["directories_removed"] += 1

            except PermissionError as e:
                stats["errors"].append(f"Permission denied removing {pycache_dir}: {e}")
            except Exception as e:
                stats["errors"].append(f"Error removing {pycache_dir}: {e}")

        # Find individual .pyc and .pyo files not in __pycache__ directories
        individual_pyc_files = []
        for pattern in ["**/*.pyc", "**/*.pyo"]:
            for pyc_file in root_dir.rglob(pattern[3:]):  # Remove **/ prefix
                # Skip files that are in __pycache__ directories (already handled)
                if "__pycache__" not in pyc_file.parts:
                    individual_pyc_files.append(pyc_file)

        if verbose and individual_pyc_files:
            logger.info(f"Found {len(individual_pyc_files)} individual bytecode files")

        # Remove individual bytecode files with proper exception handling
        for pyc_file in individual_pyc_files:
            try:
                if verbose:
                    logger.debug(f"Removing bytecode file: {pyc_file}")

                if not dry_run:
                    pyc_file.unlink()

                stats["files_removed"] += 1

            except PermissionError as e:
                stats["errors"].append(f"Permission denied removing {pyc_file}: {e}")
            except FileNotFoundError:
                # File may have been removed by parent directory cleanup
                pass
            except Exception as e:
                stats["errors"].append(f"Error removing {pyc_file}: {e}")

        # Update cleanup statistics with files removed and space reclaimed
        stats["space_reclaimed_mb"] = total_size / (1024 * 1024)

        # Log cleanup progress with file counts and directory paths if verbose
        logger.info(
            f"Python cache cleanup completed: {stats['directories_removed']} directories, "
            f"{stats['files_removed']} files, {stats['space_reclaimed_mb']:.2f} MB"
        )

        if stats["errors"] and verbose:
            logger.warning(
                f"Encountered {len(stats['errors'])} errors during Python cache cleanup"
            )

        return stats

    except Exception as e:
        logger.error(f"Python cache cleanup failed: {e}", exception=e)
        stats["errors"].append(f"Python cache cleanup failed: {e}")
        return stats


def clean_pytest_cache(
    root_dir: Path,
    preserve_config: bool = True,
    dry_run: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Removes pytest cache directories and related testing artifacts with safety checks to preserve
    important test configurations and comprehensive cleanup reporting.

    Args:
        root_dir: Root directory for cache cleanup operations
        preserve_config: Whether to preserve pytest configuration files
        dry_run: Whether to perform dry run without actual deletion
        verbose: Whether to enable verbose progress reporting

    Returns:
        dict: Dictionary containing pytest cache cleanup statistics and preserved configuration information
    """
    logger = get_component_logger("pytest_cache_cleaner", ComponentType.UTILS)

    # Initialize pytest cache cleanup statistics tracking
    stats = {
        "cache_directories_removed": 0,
        "config_files_preserved": 0,
        "space_reclaimed_mb": 0.0,
        "coverage_files_removed": 0,
        "errors": [],
        "warnings": [],
    }

    try:
        logger.debug(f"Starting pytest cache cleanup in {root_dir}")

        # Locate .pytest_cache directories using pattern matching
        pytest_cache_dirs = list(root_dir.rglob(".pytest_cache"))

        if verbose:
            logger.info(f"Found {len(pytest_cache_dirs)} .pytest_cache directories")

        # Check for important pytest configuration files if preserve_config is True
        config_files = []
        if preserve_config:
            config_patterns = ["pytest.ini", "pyproject.toml", "tox.ini", "setup.cfg"]
            for pattern in config_patterns:
                config_files.extend(root_dir.rglob(pattern))

            if verbose and config_files:
                logger.info(
                    f"Found {len(config_files)} pytest configuration files to preserve"
                )

        # Calculate size of cache directories for space reporting
        total_size = 0
        for cache_dir in pytest_cache_dirs:
            try:
                size, file_count = calculate_directory_size(
                    cache_dir, follow_symlinks=False
                )
                total_size += size
            except Exception as e:
                stats["errors"].append(f"Error calculating size for {cache_dir}: {e}")

        # Remove .pytest_cache directories with proper error handling
        for cache_dir in pytest_cache_dirs:
            try:
                if verbose:
                    logger.debug(f"Removing pytest cache directory: {cache_dir}")

                if not dry_run:
                    # Validate directory is actually .pytest_cache for safety
                    if cache_dir.name == ".pytest_cache":
                        shutil.rmtree(cache_dir, ignore_errors=False)
                        stats["cache_directories_removed"] += 1
                    else:
                        stats["warnings"].append(
                            f"Skipped non-pytest cache directory: {cache_dir}"
                        )
                else:
                    stats["cache_directories_removed"] += 1

            except PermissionError as e:
                stats["errors"].append(f"Permission denied removing {cache_dir}: {e}")
            except Exception as e:
                stats["errors"].append(f"Error removing {cache_dir}: {e}")

        # Remove temporary pytest files and coverage data
        temp_patterns = [".coverage", "coverage.xml", ".coverage.*", "htmlcov/**"]
        for pattern in temp_patterns:
            for temp_file in root_dir.rglob(pattern):
                try:
                    if verbose:
                        logger.debug(f"Removing pytest temp file: {temp_file}")

                    if not dry_run:
                        if temp_file.is_file():
                            temp_file.unlink()
                            stats["coverage_files_removed"] += 1
                        elif temp_file.is_dir():
                            shutil.rmtree(temp_file)
                            stats["coverage_files_removed"] += 1
                    else:
                        stats["coverage_files_removed"] += 1

                except PermissionError as e:
                    stats["errors"].append(
                        f"Permission denied removing {temp_file}: {e}"
                    )
                except FileNotFoundError:
                    pass  # File already removed
                except Exception as e:
                    stats["errors"].append(f"Error removing {temp_file}: {e}")

        # Update cleanup statistics with directories and files removed
        stats["space_reclaimed_mb"] = total_size / (1024 * 1024)
        stats["config_files_preserved"] = len(config_files)

        # Log pytest cache cleanup progress with directory paths if verbose
        logger.info(
            f"Pytest cache cleanup completed: {stats['cache_directories_removed']} directories, "
            f"{stats['coverage_files_removed']} temp files, {stats['space_reclaimed_mb']:.2f} MB"
        )

        return stats

    except Exception as e:
        logger.error(f"Pytest cache cleanup failed: {e}", exception=e)
        stats["errors"].append(f"Pytest cache cleanup failed: {e}")
        return stats


def clean_logging_cache(
    root_dir: Path,
    preserve_recent_logs: bool = True,
    dry_run: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Clears logging system cache including logger instances, performance baselines, and log files
    with integrated cleanup of logging infrastructure and proper resource management.

    Args:
        root_dir: Root directory for cache cleanup operations
        preserve_recent_logs: Whether to preserve recent log files (files newer than 24 hours)
        dry_run: Whether to perform dry run without actual deletion
        verbose: Whether to enable verbose progress reporting

    Returns:
        dict: Dictionary containing logging cache cleanup statistics including logger cache, log files, and performance data
    """
    logger = get_component_logger("logging_cache_cleaner", ComponentType.UTILS)

    # Initialize logging cache cleanup statistics tracking
    stats = {
        "logger_instances_cleared": 0,
        "log_files_removed": 0,
        "recent_logs_preserved": 0,
        "space_reclaimed_mb": 0.0,
        "performance_baselines_cleared": 0,
        "errors": [],
        "warnings": [],
    }

    try:
        logger.debug(f"Starting logging cache cleanup in {root_dir}")

        # Clear logger instance cache using clear_logger_cache function
        if not dry_run:
            cleared_count = clear_logger_cache(force_cleanup=True)
            stats["logger_instances_cleared"] = cleared_count

            if verbose:
                logger.info(f"Cleared {cleared_count} cached logger instances")
        else:
            # Estimate logger cache size for dry run
            stats["logger_instances_cleared"] = 10  # Estimated

        # Clear LoggerFactory cache using LoggerFactory.clear_cache method
        try:
            # Create temporary factory to access cache clearing functionality
            from ..plume_nav_sim.logging.config import LoggerFactory, LoggingConfig

            temp_factory = LoggerFactory(LoggingConfig())

            if not dry_run:
                factory_cleared = temp_factory.clear_cache(close_loggers=True)
                stats["logger_instances_cleared"] += factory_cleared

                if verbose:
                    logger.info(
                        f"Cleared {factory_cleared} LoggerFactory cached instances"
                    )
        except Exception as e:
            stats["warnings"].append(f"Could not clear LoggerFactory cache: {e}")

        # Identify log files in DEFAULT_LOG_DIR and other log locations
        log_directories = [DEFAULT_LOG_DIR, root_dir / "logs"]
        log_files = []
        total_log_size = 0

        for log_dir in log_directories:
            if log_dir.exists():
                # Find log files with various patterns
                log_patterns = [
                    "*.log",
                    "*.log.*",
                    f"{PACKAGE_NAME}*.log*",
                    "performance*.log*",
                ]
                for pattern in log_patterns:
                    for log_file in log_dir.glob(pattern):
                        if log_file.is_file():
                            log_files.append(log_file)
                            try:
                                total_log_size += log_file.stat().st_size
                            except Exception:
                                pass  # Ignore size calculation errors

        if verbose:
            logger.info(
                f"Found {len(log_files)} log files totaling {total_log_size / (1024 * 1024):.2f} MB"
            )

        # Preserve recent log files if preserve_recent_logs is True (files newer than 24 hours)
        current_time = time.time()
        recent_threshold = current_time - (24 * 3600)  # 24 hours ago

        files_to_remove = []
        for log_file in log_files:
            try:
                file_mtime = log_file.stat().st_mtime
                if preserve_recent_logs and file_mtime > recent_threshold:
                    stats["recent_logs_preserved"] += 1
                    if verbose:
                        logger.debug(f"Preserving recent log file: {log_file}")
                else:
                    files_to_remove.append(log_file)
            except Exception as e:
                stats["errors"].append(f"Error checking log file {log_file}: {e}")

        # Remove old log files and rotate log directories with proper handling
        removed_size = 0
        for log_file in files_to_remove:
            try:
                if verbose:
                    logger.debug(f"Removing log file: {log_file}")

                if not dry_run:
                    file_size = log_file.stat().st_size
                    log_file.unlink()
                    removed_size += file_size
                    stats["log_files_removed"] += 1
                else:
                    stats["log_files_removed"] += 1

            except PermissionError as e:
                stats["errors"].append(f"Permission denied removing {log_file}: {e}")
            except FileNotFoundError:
                pass  # File already removed
            except Exception as e:
                stats["errors"].append(f"Error removing {log_file}: {e}")

        # Clear performance baselines and timing measurement cache
        performance_patterns = [
            "**/performance_*.json",
            "**/timing_*.json",
            "**/benchmarks/*.tmp",
        ]
        for pattern in performance_patterns:
            for perf_file in root_dir.rglob(pattern[3:]):  # Remove **/ prefix
                try:
                    if verbose:
                        logger.debug(f"Removing performance file: {perf_file}")

                    if not dry_run:
                        perf_file.unlink()

                    stats["performance_baselines_cleared"] += 1

                except PermissionError as e:
                    stats["errors"].append(
                        f"Permission denied removing {perf_file}: {e}"
                    )
                except FileNotFoundError:
                    pass
                except Exception as e:
                    stats["errors"].append(f"Error removing {perf_file}: {e}")

        # Update cleanup statistics with logger cache and log file removal
        stats["space_reclaimed_mb"] = removed_size / (1024 * 1024)

        # Log logging cache cleanup progress with cache statistics if verbose
        logger.info(
            f"Logging cache cleanup completed: {stats['logger_instances_cleared']} logger instances, "
            f"{stats['log_files_removed']} log files, {stats['space_reclaimed_mb']:.2f} MB"
        )

        return stats

    except Exception as e:
        logger.error(f"Logging cache cleanup failed: {e}", exception=e)
        stats["errors"].append(f"Logging cache cleanup failed: {e}")
        return stats


def clean_matplotlib_cache(
    root_dir: Path,
    preserve_font_cache: bool = True,
    dry_run: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Removes matplotlib rendering cache and font cache directories with careful handling of
    platform-specific cache locations and user-specific matplotlib configurations.

    Args:
        root_dir: Root directory for cache cleanup operations
        preserve_font_cache: Whether to preserve matplotlib font cache files
        dry_run: Whether to perform dry run without actual deletion
        verbose: Whether to enable verbose progress reporting

    Returns:
        dict: Dictionary containing matplotlib cache cleanup statistics including rendering cache and font cache information
    """
    logger = get_component_logger("matplotlib_cache_cleaner", ComponentType.UTILS)

    # Initialize matplotlib cache cleanup statistics tracking
    stats = {
        "cache_directories_removed": 0,
        "font_cache_files_removed": 0,
        "temp_files_removed": 0,
        "space_reclaimed_mb": 0.0,
        "font_cache_preserved": 0,
        "errors": [],
        "warnings": [],
    }

    try:
        logger.debug(f"Starting matplotlib cache cleanup in {root_dir}")

        # Locate matplotlib cache directories using standard cache locations
        matplotlib_cache_dirs = []

        # Find project-local matplotlib cache
        local_mpl_cache = list(root_dir.rglob(".matplotlib"))
        matplotlib_cache_dirs.extend(local_mpl_cache)

        # Find platform-specific matplotlib cache paths
        home_dir = Path.home()

        # Linux: ~/.cache/matplotlib
        linux_cache = home_dir / ".cache" / "matplotlib"
        if linux_cache.exists():
            matplotlib_cache_dirs.append(linux_cache)

        # macOS: ~/Library/Caches/matplotlib
        macos_cache = home_dir / "Library" / "Caches" / "matplotlib"
        if macos_cache.exists():
            matplotlib_cache_dirs.append(macos_cache)

        # Windows: AppData cache locations
        if os.name == "nt":
            appdata = os.environ.get("APPDATA")
            if appdata:
                windows_cache = Path(appdata) / "matplotlib"
                if windows_cache.exists():
                    matplotlib_cache_dirs.append(windows_cache)

        if verbose:
            logger.info(
                f"Found {len(matplotlib_cache_dirs)} matplotlib cache directories"
            )

        # Calculate size of matplotlib cache directories and files
        total_size = 0
        font_cache_files = []

        for cache_dir in matplotlib_cache_dirs:
            try:
                size, file_count = calculate_directory_size(
                    cache_dir, follow_symlinks=False
                )
                total_size += size

                # Identify font cache files if preserve_font_cache is False
                if not preserve_font_cache:
                    font_patterns = [
                        "*.ttf",
                        "*.otf",
                        "fontList.json",
                        "fontlist-*.json",
                    ]
                    for pattern in font_patterns:
                        font_cache_files.extend(cache_dir.rglob(pattern))

            except Exception as e:
                stats["errors"].append(
                    f"Error processing matplotlib cache {cache_dir}: {e}"
                )

        # Remove matplotlib cache directories with proper error handling
        for cache_dir in matplotlib_cache_dirs:
            try:
                if verbose:
                    logger.debug(f"Processing matplotlib cache directory: {cache_dir}")

                # If preserving font cache, remove selectively
                if preserve_font_cache:
                    # Remove non-font cache files
                    for item in cache_dir.rglob("*"):
                        if (
                            item.is_file()
                            and not any(
                                item.name.endswith(ext) for ext in [".ttf", ".otf"]
                            )
                            and "font" not in item.name.lower()
                        ):
                            try:
                                if not dry_run:
                                    item.unlink()
                                stats["temp_files_removed"] += 1
                            except Exception as e:
                                stats["errors"].append(f"Error removing {item}: {e}")

                    stats["font_cache_preserved"] += len(
                        list(cache_dir.rglob("*font*"))
                    )
                else:
                    # Remove entire cache directory
                    if not dry_run:
                        shutil.rmtree(cache_dir, ignore_errors=False)

                    stats["cache_directories_removed"] += 1

            except PermissionError as e:
                stats["errors"].append(f"Permission denied removing {cache_dir}: {e}")
            except Exception as e:
                stats["errors"].append(f"Error removing {cache_dir}: {e}")

        # Remove temporary matplotlib rendering files and backends cache
        temp_patterns = ["matplotlib-*", "mpl_*", "*.mplstyle.bak"]
        temp_locations = [Path(tempfile.gettempdir()), root_dir]

        for temp_dir in temp_locations:
            if temp_dir.exists():
                for pattern in temp_patterns:
                    for temp_file in temp_dir.glob(pattern):
                        try:
                            if verbose:
                                logger.debug(
                                    f"Removing matplotlib temp file: {temp_file}"
                                )

                            if not dry_run:
                                if temp_file.is_file():
                                    temp_file.unlink()
                                elif temp_file.is_dir():
                                    shutil.rmtree(temp_file)

                            stats["temp_files_removed"] += 1

                        except PermissionError as e:
                            stats["errors"].append(
                                f"Permission denied removing {temp_file}: {e}"
                            )
                        except FileNotFoundError:
                            pass
                        except Exception as e:
                            stats["errors"].append(f"Error removing {temp_file}: {e}")

        # Update cleanup statistics with matplotlib cache removal data
        stats["space_reclaimed_mb"] = total_size / (1024 * 1024)

        # Log matplotlib cache cleanup progress with cache locations if verbose
        logger.info(
            f"Matplotlib cache cleanup completed: {stats['cache_directories_removed']} directories, "
            f"{stats['temp_files_removed']} temp files, {stats['space_reclaimed_mb']:.2f} MB"
        )

        if preserve_font_cache and stats["font_cache_preserved"]:
            logger.info(f"Preserved {stats['font_cache_preserved']} font cache files")

        return stats

    except Exception as e:
        logger.error(f"Matplotlib cache cleanup failed: {e}", exception=e)
        stats["errors"].append(f"Matplotlib cache cleanup failed: {e}")
        return stats


def clean_performance_cache(
    root_dir: Path,
    preserve_baselines: bool = False,
    dry_run: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Removes performance measurement baselines, benchmark temporary files, and timing data with
    optional preservation of recent performance data for continuous monitoring.

    Args:
        root_dir: Root directory for cache cleanup operations
        preserve_baselines: Whether to preserve recent baseline files
        dry_run: Whether to perform dry run without actual deletion
        verbose: Whether to enable verbose progress reporting

    Returns:
        dict: Dictionary containing performance cache cleanup statistics including baseline data and benchmark files
    """
    logger = get_component_logger("performance_cache_cleaner", ComponentType.UTILS)

    # Initialize performance cache cleanup statistics tracking
    stats = {
        "baseline_files_removed": 0,
        "benchmark_files_removed": 0,
        "timing_files_removed": 0,
        "recent_baselines_preserved": 0,
        "space_reclaimed_mb": 0.0,
        "errors": [],
        "warnings": [],
    }

    try:
        logger.debug(f"Starting performance cache cleanup in {root_dir}")

        # Locate performance baseline files using glob pattern matching
        performance_patterns = {
            "baselines": [
                "**/performance_*.json",
                "**/baseline_*.json",
                "**/perf_*.json",
            ],
            "benchmarks": [
                "**/benchmarks/**/*.tmp",
                "**/benchmarks/**/*.temp",
                "**/benchmark_*.json",
            ],
            "timing": ["**/timing_*.log", "**/timing_*.json", "**/.timing_*"],
        }

        performance_files = {"baselines": [], "benchmarks": [], "timing": []}
        total_size = 0

        # Find all performance-related files
        for category, patterns in performance_patterns.items():
            for pattern in patterns:
                for perf_file in root_dir.rglob(pattern[3:]):  # Remove **/ prefix
                    if perf_file.is_file():
                        performance_files[category].append(perf_file)
                        try:
                            total_size += perf_file.stat().st_size
                        except Exception:
                            pass

        if verbose:
            total_files = sum(len(files) for files in performance_files.values())
            logger.info(
                f"Found {total_files} performance files totaling {total_size / (1024 * 1024):.2f} MB"
            )

        # Preserve recent baseline files if preserve_baselines is True
        current_time = time.time()
        baseline_preserve_threshold = current_time - (7 * 24 * 3600)  # 7 days ago

        files_to_remove = {"baselines": [], "benchmarks": [], "timing": []}

        for category, files in performance_files.items():
            for perf_file in files:
                try:
                    if category == "baselines" and preserve_baselines:
                        file_mtime = perf_file.stat().st_mtime
                        if file_mtime > baseline_preserve_threshold:
                            stats["recent_baselines_preserved"] += 1
                            if verbose:
                                logger.debug(f"Preserving recent baseline: {perf_file}")
                            continue

                    files_to_remove[category].append(perf_file)

                except Exception as e:
                    stats["errors"].append(
                        f"Error checking performance file {perf_file}: {e}"
                    )

        # Remove performance baseline files with timestamp checking
        removed_size = 0
        for baseline_file in files_to_remove["baselines"]:
            try:
                if verbose:
                    logger.debug(f"Removing baseline file: {baseline_file}")

                if not dry_run:
                    file_size = baseline_file.stat().st_size
                    baseline_file.unlink()
                    removed_size += file_size

                stats["baseline_files_removed"] += 1

            except PermissionError as e:
                stats["errors"].append(
                    f"Permission denied removing {baseline_file}: {e}"
                )
            except FileNotFoundError:
                pass
            except Exception as e:
                stats["errors"].append(f"Error removing {baseline_file}: {e}")

        # Clear benchmark temporary files and measurement cache
        for benchmark_file in files_to_remove["benchmarks"]:
            try:
                if verbose:
                    logger.debug(f"Removing benchmark file: {benchmark_file}")

                if not dry_run:
                    try:
                        file_size = benchmark_file.stat().st_size
                        removed_size += file_size
                    except Exception:
                        pass

                    benchmark_file.unlink()

                stats["benchmark_files_removed"] += 1

            except PermissionError as e:
                stats["errors"].append(
                    f"Permission denied removing {benchmark_file}: {e}"
                )
            except FileNotFoundError:
                pass
            except Exception as e:
                stats["errors"].append(f"Error removing {benchmark_file}: {e}")

        # Remove timing data artifacts and performance logs
        for timing_file in files_to_remove["timing"]:
            try:
                if verbose:
                    logger.debug(f"Removing timing file: {timing_file}")

                if not dry_run:
                    try:
                        file_size = timing_file.stat().st_size
                        removed_size += file_size
                    except Exception:
                        pass

                    timing_file.unlink()

                stats["timing_files_removed"] += 1

            except PermissionError as e:
                stats["errors"].append(f"Permission denied removing {timing_file}: {e}")
            except FileNotFoundError:
                pass
            except Exception as e:
                stats["errors"].append(f"Error removing {timing_file}: {e}")

        # Update cleanup statistics with performance data removal
        stats["space_reclaimed_mb"] = removed_size / (1024 * 1024)

        # Log performance cache cleanup progress with file counts if verbose
        logger.info(
            f"Performance cache cleanup completed: {stats['baseline_files_removed']} baselines, "
            f"{stats['benchmark_files_removed']} benchmarks, {stats['timing_files_removed']} timing files, "
            f"{stats['space_reclaimed_mb']:.2f} MB"
        )

        return stats

    except Exception as e:
        logger.error(f"Performance cache cleanup failed: {e}", exception=e)
        stats["errors"].append(f"Performance cache cleanup failed: {e}")
        return stats


def clean_temporary_files(
    root_dir: Path, max_age_days: int = 7, dry_run: bool = False, verbose: bool = False
) -> Dict[str, Any]:
    """
    Removes temporary development files and directories created during development and testing
    with pattern-based identification and age-based cleanup criteria.

    Args:
        root_dir: Root directory for cache cleanup operations
        max_age_days: Maximum age in days for temporary files to be kept
        dry_run: Whether to perform dry run without actual deletion
        verbose: Whether to enable verbose progress reporting

    Returns:
        dict: Dictionary containing temporary file cleanup statistics including files removed and age criteria applied
    """
    logger = get_component_logger("temp_files_cleaner", ComponentType.UTILS)

    # Initialize temporary file cleanup statistics tracking
    stats = {
        "temp_files_removed": 0,
        "temp_directories_removed": 0,
        "space_reclaimed_mb": 0.0,
        "age_threshold_days": max_age_days,
        "files_preserved_by_age": 0,
        "errors": [],
        "warnings": [],
    }

    try:
        logger.debug(
            f"Starting temporary file cleanup in {root_dir} (max age: {max_age_days} days)"
        )

        # Find temporary files and directories using pattern matching from CACHE_TYPES
        temp_patterns = CACHE_TYPES["temp"]["patterns"]
        temp_items = []

        for pattern in temp_patterns:
            for temp_item in root_dir.rglob(pattern[3:]):  # Remove **/ prefix
                temp_items.append(temp_item)

        # Add common temporary file patterns
        additional_patterns = ["*.tmp", "*.temp", "*.bak", "*.swp", "*~", "*.pid"]
        for pattern in additional_patterns:
            for temp_item in root_dir.rglob(pattern):
                temp_items.append(temp_item)

        # Remove duplicates
        temp_items = list(set(temp_items))

        if verbose:
            logger.info(f"Found {len(temp_items)} temporary items")

        # Check file modification times for age-based cleanup criteria
        current_time = time.time()
        age_threshold = current_time - (max_age_days * 24 * 3600)

        items_to_remove = []
        total_size = 0

        for temp_item in temp_items:
            try:
                # Filter files older than max_age_days for removal
                item_mtime = temp_item.stat().st_mtime

                if item_mtime < age_threshold:
                    items_to_remove.append(temp_item)

                    # Calculate size for space reporting
                    if temp_item.is_file():
                        total_size += temp_item.stat().st_size
                    elif temp_item.is_dir():
                        size, _ = calculate_directory_size(
                            temp_item, follow_symlinks=False
                        )
                        total_size += size
                else:
                    stats["files_preserved_by_age"] += 1
                    if verbose:
                        age_days = (current_time - item_mtime) / (24 * 3600)
                        logger.debug(
                            f"Preserving recent temp item: {temp_item} (age: {age_days:.1f} days)"
                        )

            except Exception as e:
                stats["errors"].append(f"Error checking temp item {temp_item}: {e}")

        if verbose:
            logger.info(
                f"Will remove {len(items_to_remove)} items older than {max_age_days} days, "
                f"preserving {stats['files_preserved_by_age']} recent items"
            )

        # Remove temporary files with age validation and safety checks
        removed_size = 0
        for temp_item in items_to_remove:
            try:
                # Validate item is safe to remove (not in excluded patterns)
                is_excluded = any(
                    temp_item.match(pattern) for pattern in EXCLUDED_PATTERNS
                )

                if is_excluded:
                    stats["warnings"].append(f"Skipped excluded file: {temp_item}")
                    continue

                if verbose:
                    logger.debug(f"Removing temporary item: {temp_item}")

                if not dry_run:
                    if temp_item.is_file():
                        try:
                            file_size = temp_item.stat().st_size
                            temp_item.unlink()
                            removed_size += file_size
                            stats["temp_files_removed"] += 1
                        except Exception as e:
                            stats["errors"].append(
                                f"Error removing file {temp_item}: {e}"
                            )

                    elif temp_item.is_dir():
                        try:
                            dir_size, _ = calculate_directory_size(
                                temp_item, follow_symlinks=False
                            )
                            shutil.rmtree(temp_item)
                            removed_size += dir_size
                            stats["temp_directories_removed"] += 1
                        except Exception as e:
                            stats["errors"].append(
                                f"Error removing directory {temp_item}: {e}"
                            )
                else:
                    # Dry run counting
                    if temp_item.is_file():
                        stats["temp_files_removed"] += 1
                    elif temp_item.is_dir():
                        stats["temp_directories_removed"] += 1

            except PermissionError as e:
                stats["errors"].append(f"Permission denied removing {temp_item}: {e}")
            except FileNotFoundError:
                pass  # File already removed
            except Exception as e:
                stats["errors"].append(f"Error processing {temp_item}: {e}")

        # Remove empty temporary directories after file cleanup
        if not dry_run:
            for temp_item in reversed(
                items_to_remove
            ):  # Process in reverse to handle nested directories
                if temp_item.exists() and temp_item.is_dir():
                    try:
                        if not any(temp_item.iterdir()):  # Directory is empty
                            temp_item.rmdir()
                            if verbose:
                                logger.debug(f"Removed empty directory: {temp_item}")
                    except Exception:
                        pass  # Ignore errors when removing empty directories

        # Update cleanup statistics with temporary file removal data
        stats["space_reclaimed_mb"] = removed_size / (1024 * 1024)

        # Log temporary file cleanup progress with age information if verbose
        logger.info(
            f"Temporary file cleanup completed: {stats['temp_files_removed']} files, "
            f"{stats['temp_directories_removed']} directories, {stats['space_reclaimed_mb']:.2f} MB "
            f"(age threshold: {max_age_days} days)"
        )

        return stats

    except Exception as e:
        logger.error(f"Temporary file cleanup failed: {e}", exception=e)
        stats["errors"].append(f"Temporary file cleanup failed: {e}")
        return stats


def create_backup(
    source_path: Path, backup_type: str, backup_dir: Path
) -> Optional[Path]:
    """
    Creates backup of cache directories and files before cleanup with compressed archive creation,
    safety checks, and restoration capabilities for critical cache data.

    Args:
        source_path: Path to source directory or file to backup
        backup_type: Type identifier for backup organization
        backup_dir: Directory to store backup files

    Returns:
        Path: Path to created backup file or None if backup creation failed
    """
    logger = get_component_logger("backup_manager", ComponentType.UTILS)

    try:
        # Validate source path exists and is accessible for backup creation
        if not source_path.exists():
            logger.warning(f"Source path does not exist: {source_path}")
            return None

        # Create backup directory structure if it doesn't exist
        backup_dir.mkdir(parents=True, exist_ok=True)

        # Generate timestamped backup filename with backup_type identifier
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"{backup_type}_backup_{timestamp}"
        backup_path = backup_dir / backup_filename

        logger.debug(f"Creating backup of {source_path} to {backup_path}")

        # Create compressed archive of source path using shutil.make_archive
        try:
            if source_path.is_file():
                # For single files, copy directly
                backup_file_path = backup_path.with_suffix(".bak")
                shutil.copy2(source_path, backup_file_path)
                final_backup_path = backup_file_path
            else:
                # For directories, create compressed archive
                archive_path = str(backup_path)
                final_backup_path = Path(
                    shutil.make_archive(archive_path, "zip", source_path)
                )

            # Verify backup archive integrity and completeness
            if final_backup_path.exists():
                backup_size = final_backup_path.stat().st_size
                if backup_size > 0:
                    logger.info(
                        f"Backup created successfully: {final_backup_path} ({format_size(backup_size)})"
                    )

                    # Set appropriate backup file permissions for security
                    final_backup_path.chmod(0o600)  # Read/write for owner only

                    return final_backup_path
                else:
                    logger.error(f"Backup file is empty: {final_backup_path}")
                    final_backup_path.unlink()  # Remove empty backup
                    return None
            else:
                logger.error(f"Backup file was not created: {final_backup_path}")
                return None

        except Exception as e:
            logger.error(f"Failed to create backup archive: {e}")
            return None

    except Exception as e:
        logger.error(f"Backup creation failed: {e}")
        return None


def restore_backup(
    backup_path: Path, restore_location: Path, verify_integrity: bool = True
) -> bool:
    """
    Restores cache data from backup archive with validation, integrity checking, and selective
    restoration capabilities for recovery from cleanup errors.

    Args:
        backup_path: Path to backup archive file
        restore_location: Target location for restoration
        verify_integrity: Whether to verify backup integrity before restoration

    Returns:
        bool: True if restoration successful, False otherwise
    """
    logger = get_component_logger("backup_restore", ComponentType.UTILS)

    try:
        # Validate backup file exists and is readable for restoration
        if not backup_path.exists():
            logger.error(f"Backup file does not exist: {backup_path}")
            return False

        if not backup_path.is_file():
            logger.error(f"Backup path is not a file: {backup_path}")
            return False

        # Verify backup archive integrity if verify_integrity is True
        if verify_integrity:
            try:
                backup_size = backup_path.stat().st_size
                if backup_size == 0:
                    logger.error(f"Backup file is empty: {backup_path}")
                    return False

                # For ZIP archives, attempt to read the archive header
                if backup_path.suffix.lower() == ".zip":
                    import zipfile

                    with zipfile.ZipFile(backup_path, "r") as zf:
                        # Test the archive integrity
                        bad_file = zf.testzip()
                        if bad_file:
                            logger.error(f"Backup archive is corrupted: {bad_file}")
                            return False

                logger.info(f"Backup integrity verified: {backup_path}")

            except Exception as e:
                logger.error(f"Backup integrity verification failed: {e}")
                if verify_integrity:
                    return False

        # Create restoration target directory if it doesn't exist
        restore_location.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Restoring backup from {backup_path} to {restore_location}")

        # Extract backup archive to restoration location using shutil.unpack_archive
        try:
            if backup_path.suffix.lower() == ".bak":
                # Simple file backup
                shutil.copy2(backup_path, restore_location)
            else:
                # Compressed archive
                shutil.unpack_archive(str(backup_path), str(restore_location.parent))

            # Verify restored files match backup manifest and checksums
            if restore_location.exists():
                logger.info(f"Backup restored successfully to {restore_location}")
                return True
            else:
                logger.error(
                    f"Restoration failed - target not found: {restore_location}"
                )
                return False

        except Exception as e:
            logger.error(f"Failed to extract backup archive: {e}")
            return False

    except Exception as e:
        logger.error(f"Backup restoration failed: {e}")
        return False


def calculate_directory_size(
    directory_path: Path, follow_symlinks: bool = False
) -> Tuple[int, int]:
    """
    Calculates total size of directory tree including all files and subdirectories with
    human-readable size formatting and performance optimization for large directory structures.

    Args:
        directory_path: Path to directory for size calculation
        follow_symlinks: Whether to follow symbolic links during traversal

    Returns:
        tuple: Tuple of (total_size_bytes: int, file_count: int) with directory statistics
    """
    # Initialize size and file count accumulators for directory traversal
    total_size = 0
    file_count = 0

    try:
        # Walk directory tree using os.walk or pathlib.Path.rglob
        if follow_symlinks:
            # Use os.walk for symlink following capability
            for dirpath, dirnames, filenames in os.walk(
                directory_path, followlinks=True
            ):
                for filename in filenames:
                    file_path = Path(dirpath) / filename
                    try:
                        # Sum file sizes using os.path.getsize with error handling for inaccessible files
                        file_size = file_path.stat().st_size
                        total_size += file_size
                        file_count += 1
                    except (OSError, FileNotFoundError, PermissionError):
                        # Handle permission errors and missing files gracefully
                        continue
        else:
            # Use pathlib for better performance when not following symlinks
            for item in directory_path.rglob("*"):
                if item.is_file():
                    try:
                        file_size = item.stat().st_size
                        total_size += file_size
                        file_count += 1
                    except (OSError, FileNotFoundError, PermissionError):
                        continue

        # Return tuple with total size in bytes and file count
        return total_size, file_count

    except Exception:
        # Handle any unexpected errors during directory traversal
        return 0, 0


def format_size(size_bytes: int, decimal_places: int = 2) -> str:
    """
    Formats byte size values into human-readable strings with appropriate units (B, KB, MB, GB)
    and decimal precision for user-friendly size reporting.

    Args:
        size_bytes: Size value in bytes to format
        decimal_places: Number of decimal places for formatting precision

    Returns:
        str: Human-readable size string with appropriate units and precision
    """
    # Define size unit constants (bytes, KB, MB, GB, TB)
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    unit_size = 1024.0

    # Determine appropriate unit based on size magnitude
    size = float(size_bytes)
    unit_index = 0

    while size >= unit_size and unit_index < len(units) - 1:
        size /= unit_size
        unit_index += 1

    # Format size with specified decimal places for readability
    if unit_index == 0:  # Bytes - no decimal places needed
        return f"{int(size)} {units[unit_index]}"
    else:
        return f"{size:.{decimal_places}f} {units[unit_index]}"


def generate_cleanup_report(
    cleanup_stats: Dict[str, Any], verbose: bool = False, dry_run: bool = False
) -> str:
    """
    Generates comprehensive cleanup report with statistics, warnings, recommendations, and summary
    of cleanup operations for user information and audit purposes.

    Args:
        cleanup_stats: Dictionary containing cleanup statistics from all operations
        verbose: Whether to include verbose details in the report
        dry_run: Whether this was a dry run operation

    Returns:
        str: Formatted cleanup report with statistics, warnings, and recommendations
    """
    # Initialize report formatting with cleanup statistics summary
    report_lines = []

    # Add header with operation type (dry run vs actual)
    if dry_run:
        report_lines.append("=== CACHE CLEANUP PREVIEW (DRY RUN) ===")
    else:
        report_lines.append("=== CACHE CLEANUP RESULTS ===")

    report_lines.append("")

    # Calculate total files removed and space reclaimed across all cache types
    total_files = cleanup_stats.get("total_files_removed", 0)
    total_dirs = cleanup_stats.get("total_directories_removed", 0)
    total_space_mb = cleanup_stats.get("total_space_reclaimed_mb", 0.0)

    # Add execution summary
    start_time = cleanup_stats.get("start_time", time.time())
    duration = time.time() - start_time

    report_lines.append(f"Execution Time: {duration:.2f} seconds")
    report_lines.append(
        f"Total Files {'to be ' if dry_run else ''}Removed: {total_files:,}"
    )
    report_lines.append(
        f"Total Directories {'to be ' if dry_run else ''}Removed: {total_dirs:,}"
    )
    report_lines.append(
        f"Total Space {'to be ' if dry_run else ''}Reclaimed: {format_size(int(total_space_mb * 1024 * 1024))}"
    )
    report_lines.append("")

    # Format cache type specific statistics with file counts and sizes
    if "cache_type_results" in cleanup_stats:
        report_lines.append("=== CACHE TYPE BREAKDOWN ===")

        for cache_type, results in cleanup_stats["cache_type_results"].items():
            if results:
                report_lines.append(f"\n{cache_type.upper()} Cache:")
                report_lines.append(
                    f"  Description: {CACHE_TYPES.get(cache_type, {}).get('description', 'N/A')}"
                )

                # Extract relevant statistics based on cache type
                if "files_removed" in results:
                    report_lines.append(
                        f"  Files {'to be ' if dry_run else ''}removed: {results['files_removed']:,}"
                    )
                if "directories_removed" in results:
                    report_lines.append(
                        f"  Directories {'to be ' if dry_run else ''}removed: {results['directories_removed']:,}"
                    )
                if "cache_directories_removed" in results:
                    report_lines.append(
                        f"  Cache directories {'to be ' if dry_run else ''}removed: {results['cache_directories_removed']:,}"
                    )
                if "space_reclaimed_mb" in results:
                    space_size = int(results["space_reclaimed_mb"] * 1024 * 1024)
                    report_lines.append(
                        f"  Space {'to be ' if dry_run else ''}reclaimed: {format_size(space_size)}"
                    )

                # Add type-specific details if verbose
                if verbose:
                    for key, value in results.items():
                        if key not in [
                            "files_removed",
                            "directories_removed",
                            "cache_directories_removed",
                            "space_reclaimed_mb",
                            "errors",
                            "warnings",
                        ]:
                            report_lines.append(
                                f"  {key.replace('_', ' ').title()}: {value}"
                            )

        report_lines.append("")

    # Include warnings for any errors or issues encountered during cleanup
    warnings = cleanup_stats.get("warnings", [])
    if warnings:
        report_lines.append("=== WARNINGS ===")
        for warning in warnings:
            report_lines.append(f"  ⚠️  {warning}")
        report_lines.append("")

    # Include errors if any occurred
    errors = cleanup_stats.get("errors", [])
    if errors:
        report_lines.append("=== ERRORS ===")
        for error in errors:
            report_lines.append(f"  ❌ {error}")
        report_lines.append("")

    # Add recommendations for cache management and cleanup frequency
    report_lines.append("=== RECOMMENDATIONS ===")

    if total_space_mb > 100:
        report_lines.append(
            "  💡 Large amount of cache cleared - consider more frequent cleanups"
        )
    elif total_space_mb < 1:
        report_lines.append(
            "  💡 Minimal cache accumulation - current cleanup frequency is appropriate"
        )

    if errors:
        report_lines.append("  🔧 Review errors above and check file permissions")

    if "backup_files" in cleanup_stats and cleanup_stats["backup_files"]:
        report_lines.append(
            f"  💾 {len(cleanup_stats['backup_files'])} backup files created for rollback capability"
        )

    report_lines.append(
        "  📅 Regular cache cleanup helps maintain optimal development performance"
    )

    # Include dry-run simulation results if dry_run was enabled
    if dry_run:
        report_lines.append("")
        report_lines.append("=== DRY RUN NOTICE ===")
        report_lines.append("  This was a preview - no files were actually removed")
        report_lines.append("  Run without --dry-run flag to perform actual cleanup")

    # Generate summary section with overall cleanup effectiveness
    report_lines.append("")
    report_lines.append("=== SUMMARY ===")

    if total_files > 0 or total_dirs > 0:
        efficiency = (
            "High" if total_space_mb > 10 else "Medium" if total_space_mb > 1 else "Low"
        )
        report_lines.append(f"  Cleanup Efficiency: {efficiency}")
        report_lines.append(
            f"  Cache Accumulation: {'Heavy' if total_space_mb > 50 else 'Moderate' if total_space_mb > 10 else 'Light'}"
        )
    else:
        report_lines.append("  No cache files found for cleanup")

    if not errors:
        report_lines.append("  ✅ Cleanup completed successfully")
    else:
        report_lines.append(f"  ⚠️  Cleanup completed with {len(errors)} errors")

    # Return comprehensive formatted cleanup report string
    return "\n".join(report_lines)


def validate_cleanup_safety(
    target_path: Path, patterns: List[str], strict_mode: bool = False
) -> Tuple[bool, List[str]]:
    """
    Validates cleanup operations for safety by checking for important files, active processes,
    and potential data loss risks with comprehensive safety assessment and warning generation.

    Args:
        target_path: Path to be cleaned up
        patterns: List of cleanup patterns to validate
        strict_mode: Whether to apply strict safety validation

    Returns:
        tuple: Tuple of (is_safe: bool, warnings: list) with safety assessment and warning messages
    """
    logger = get_component_logger("safety_validator", ComponentType.UTILS)

    # Initialize safety validation with target path and pattern checking
    warnings = []
    is_safe = True

    try:
        # Check target path against EXCLUDED_PATTERNS to prevent important file removal
        for excluded_pattern in EXCLUDED_PATTERNS:
            if target_path.match(excluded_pattern):
                warnings.append(
                    f"Target path matches excluded pattern: {excluded_pattern}"
                )
                is_safe = False

        # Validate patterns don't match critical system or project files
        critical_files = [
            "pyproject.toml",
            "Pipfile",
            "setup.cfg",
            "tox.ini",
            ".gitignore",
            "README.md",
            "LICENSE",
        ]

        for pattern in patterns:
            for critical_file in critical_files:
                if target_path.glob(critical_file):
                    # Check if pattern would match critical files
                    for potential_match in target_path.rglob(pattern):
                        if potential_match.name in critical_files:
                            warnings.append(
                                f"Pattern '{pattern}' may match critical file: {potential_match}"
                            )
                            if strict_mode:
                                is_safe = False

        # Check for active processes that might be using cache files
        if target_path.exists():
            # Simple check for .lock files or .pid files
            for lock_file in target_path.rglob("*.lock"):
                warnings.append(
                    f"Found lock file indicating active process: {lock_file}"
                )
                if strict_mode:
                    is_safe = False

            for pid_file in target_path.rglob("*.pid"):
                warnings.append(
                    f"Found PID file indicating running process: {pid_file}"
                )
                if strict_mode:
                    is_safe = False

        # Detect important configuration files that should be preserved
        config_patterns = ["*.ini", "*.conf", "*.yaml", "*.yml", "*.toml"]
        config_files_found = []

        for config_pattern in config_patterns:
            config_files_found.extend(target_path.rglob(config_pattern))

        if config_files_found:
            config_count = len(config_files_found)
            warnings.append(f"Found {config_count} configuration files in cleanup path")

            if (
                config_count > 10
            ):  # Many config files might indicate important directory
                warnings.append(
                    "Large number of config files suggests important directory"
                )
                if strict_mode:
                    is_safe = False

        # Apply strict safety checks if strict_mode is enabled
        if strict_mode:
            # Additional checks for strict mode
            if target_path == Path.home():
                warnings.append(
                    "Target path is user home directory - extremely dangerous"
                )
                is_safe = False

            if target_path == Path("/"):
                warnings.append(
                    "Target path is root directory - system destroying operation"
                )
                is_safe = False

            # Check for version control directories
            vcs_dirs = [".git", ".svn", ".hg", ".bzr"]
            for vcs_dir in vcs_dirs:
                if (target_path / vcs_dir).exists():
                    warnings.append(f"Found version control directory: {vcs_dir}")
                    # VCS directories are generally safe to preserve

        # Generate warning messages for potential safety issues
        if not warnings:
            logger.debug(f"Safety validation passed for {target_path}")
        else:
            logger.info(
                f"Safety validation found {len(warnings)} potential issues for {target_path}"
            )

        # Return safety status with comprehensive warning list
        return is_safe, warnings

    except Exception as e:
        logger.error(f"Safety validation failed: {e}")
        warnings.append(f"Safety validation error: {e}")
        return False, warnings


class CacheCleanupManager:
    """
    Central manager class for coordinating cache cleanup operations with comprehensive statistics tracking,
    error handling, rollback capabilities, and detailed reporting for safe and efficient cache management.
    """

    def __init__(
        self,
        root_directory: Path,
        cleanup_config: Dict[str, Any],
        enable_backup: bool = False,
    ):
        """
        Initialize CacheCleanupManager with root directory, configuration settings, and backup
        capabilities for comprehensive cache cleanup coordination.

        Args:
            root_directory: Root directory path for cleanup operations
            cleanup_config: Dictionary containing cleanup configuration settings
            enable_backup: Whether to enable backup creation before cleanup
        """
        # Store root directory path and validate accessibility
        self.root_directory = root_directory.resolve()
        if not self.root_directory.exists():
            raise ValueError(f"Root directory does not exist: {self.root_directory}")

        # Configure cleanup settings from cleanup_config dictionary
        self.cleanup_configuration = cleanup_config.copy()

        # Set backup enablement flag and initialize backup registry
        self.backup_enabled = enable_backup
        self.backup_registry = {}

        # Initialize cleanup statistics tracking dictionary
        self.cleanup_statistics = {
            "total_files_removed": 0,
            "total_directories_removed": 0,
            "total_space_reclaimed_mb": 0.0,
            "cache_type_results": {},
            "operations_completed": 0,
            "operations_failed": 0,
        }

        # Create component logger using get_component_logger with ComponentType.UTILS
        self.logger = get_component_logger(
            component_name="cache_cleanup_manager",
            component_type=ComponentType.UTILS,
            enable_performance_tracking=True,
        )

        # Initialize cleanup warnings list for issue tracking
        self.cleanup_warnings = []

        # Set dry_run_mode based on configuration settings
        self.dry_run_mode = cleanup_config.get("dry_run", False)

        # Validate cleanup configuration and safety parameters
        is_safe, safety_warnings = validate_cleanup_safety(
            self.root_directory,
            list(CACHE_TYPES.keys()),
            strict_mode=cleanup_config.get("strict_mode", False),
        )

        if safety_warnings:
            self.cleanup_warnings.extend(safety_warnings)

        if not is_safe and not cleanup_config.get("force", False):
            raise ValueError(
                f"Cleanup operation not safe for {self.root_directory}. Use --force to override."
            )

        self.logger.info(f"CacheCleanupManager initialized for {self.root_directory}")

    def execute_cleanup(
        self, cache_types: List[str], force_cleanup: bool = False
    ) -> Dict[str, Any]:
        """
        Executes comprehensive cache cleanup operations with coordinated cleanup of all cache types,
        backup creation, error handling, and detailed progress reporting.

        Args:
            cache_types: List of cache types to clean up
            force_cleanup: Whether to force cleanup despite warnings

        Returns:
            dict: Dictionary containing comprehensive cleanup results, statistics, and operation status
        """
        self.logger.info(f"Starting cleanup execution for cache types: {cache_types}")

        # Validate cleanup safety and generate warnings for potential issues
        if not force_cleanup:
            for cache_type in cache_types:
                if cache_type not in CACHE_TYPES:
                    raise ValueError(f"Invalid cache type: {cache_type}")

        # Initialize cleanup progress tracking and statistics collection
        cleanup_results = {
            "start_time": time.time(),
            "cache_type_results": {},
            "backup_files": [],
            "warnings": list(self.cleanup_warnings),
            "errors": [],
            "total_files_removed": 0,
            "total_directories_removed": 0,
            "total_space_reclaimed_mb": 0.0,
        }

        try:
            # Create backups of important cache data if backup_enabled is True
            if self.backup_enabled:
                backup_results = self.create_cleanup_backup(
                    cache_paths=[self.root_directory],
                    backup_name=f"cleanup_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                )
                if backup_results:
                    cleanup_results["backup_files"].append(str(backup_results))
                    self.logger.info(f"Backup created: {backup_results}")

            # Execute cleanup operations for each specified cache type in cache_types
            for cache_type in cache_types:
                try:
                    self.logger.debug(f"Processing cache type: {cache_type}")

                    # Execute appropriate cleanup function based on cache_type
                    cache_result = self.cleanup_cache_type(
                        cache_type=cache_type, type_config=self.cleanup_configuration
                    )

                    # Track cleanup progress and statistics for cache type
                    cleanup_results["cache_type_results"][cache_type] = cache_result

                    # Accumulate totals
                    if "files_removed" in cache_result:
                        cleanup_results["total_files_removed"] += cache_result[
                            "files_removed"
                        ]
                    if "directories_removed" in cache_result:
                        cleanup_results["total_directories_removed"] += cache_result[
                            "directories_removed"
                        ]
                    if "cache_directories_removed" in cache_result:
                        cleanup_results["total_directories_removed"] += cache_result[
                            "cache_directories_removed"
                        ]
                    if "space_reclaimed_mb" in cache_result:
                        cleanup_results["total_space_reclaimed_mb"] += cache_result[
                            "space_reclaimed_mb"
                        ]

                    # Handle cleanup errors with proper logging and rollback capabilities
                    if "errors" in cache_result and cache_result["errors"]:
                        cleanup_results["errors"].extend(cache_result["errors"])
                        self.cleanup_statistics["operations_failed"] += 1
                    else:
                        self.cleanup_statistics["operations_completed"] += 1

                    if "warnings" in cache_result and cache_result["warnings"]:
                        cleanup_results["warnings"].extend(cache_result["warnings"])

                    self.logger.debug(
                        f"Completed cache type {cache_type}: "
                        f"{cache_result.get('files_removed', 0)} files, "
                        f"{cache_result.get('space_reclaimed_mb', 0):.2f} MB"
                    )

                except Exception as e:
                    error_msg = f"Failed to clean {cache_type} cache: {e}"
                    cleanup_results["errors"].append(error_msg)
                    self.logger.error(error_msg, exception=e)
                    self.cleanup_statistics["operations_failed"] += 1

            # Update cleanup statistics with results from each cache type cleanup
            self.cleanup_statistics.update(
                {
                    "total_files_removed": cleanup_results["total_files_removed"],
                    "total_directories_removed": cleanup_results[
                        "total_directories_removed"
                    ],
                    "total_space_reclaimed_mb": cleanup_results[
                        "total_space_reclaimed_mb"
                    ],
                }
            )

            # Clean up any temporary files created during cleanup process
            self._cleanup_temp_files()

            cleanup_results["end_time"] = time.time()
            cleanup_results["duration_seconds"] = (
                cleanup_results["end_time"] - cleanup_results["start_time"]
            )

            self.logger.info(
                f"Cleanup execution completed in {cleanup_results['duration_seconds']:.2f} seconds"
            )

            # Return comprehensive cleanup results dictionary
            return cleanup_results

        except Exception as e:
            error_msg = f"Cleanup execution failed: {e}"
            cleanup_results["errors"].append(error_msg)
            self.logger.error(error_msg, exception=e)
            return cleanup_results

    def cleanup_cache_type(
        self, cache_type: str, type_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Executes cleanup for specific cache type with specialized handling, progress tracking,
        and error recovery for targeted cache management operations.

        Args:
            cache_type: Type of cache to clean (python, pytest, logging, etc.)
            type_config: Configuration dictionary for the cache type cleanup

        Returns:
            dict: Dictionary containing cache type specific cleanup statistics and operation results
        """
        # Validate cache_type against supported CACHE_TYPES
        if cache_type not in CACHE_TYPES:
            raise ValueError(f"Unsupported cache type: {cache_type}")

        self.logger.debug(f"Cleaning {cache_type} cache in {self.root_directory}")

        # Execute appropriate cleanup function based on cache_type
        try:
            if cache_type == "python":
                return clean_python_cache(
                    root_dir=self.root_directory,
                    dry_run=type_config.get("dry_run", False),
                    verbose=type_config.get("verbose", False),
                )

            elif cache_type == "pytest":
                return clean_pytest_cache(
                    root_dir=self.root_directory,
                    preserve_config=not type_config.get("force", False),
                    dry_run=type_config.get("dry_run", False),
                    verbose=type_config.get("verbose", False),
                )

            elif cache_type == "logging":
                return clean_logging_cache(
                    root_dir=self.root_directory,
                    preserve_recent_logs=not type_config.get("force", False),
                    dry_run=type_config.get("dry_run", False),
                    verbose=type_config.get("verbose", False),
                )

            elif cache_type == "matplotlib":
                return clean_matplotlib_cache(
                    root_dir=self.root_directory,
                    preserve_font_cache=not type_config.get("force", False),
                    dry_run=type_config.get("dry_run", False),
                    verbose=type_config.get("verbose", False),
                )

            elif cache_type == "performance":
                return clean_performance_cache(
                    root_dir=self.root_directory,
                    preserve_baselines=not type_config.get("force", False),
                    dry_run=type_config.get("dry_run", False),
                    verbose=type_config.get("verbose", False),
                )

            elif cache_type == "temp":
                max_age = type_config.get("older_than_days", 7)
                return clean_temporary_files(
                    root_dir=self.root_directory,
                    max_age_days=max_age,
                    dry_run=type_config.get("dry_run", False),
                    verbose=type_config.get("verbose", False),
                )

            else:
                raise ValueError(
                    f"No cleanup function defined for cache type: {cache_type}"
                )

        except Exception as e:
            self.logger.error(f"Cache type {cache_type} cleanup failed: {e}")
            return {
                "files_removed": 0,
                "directories_removed": 0,
                "space_reclaimed_mb": 0.0,
                "errors": [f"Cache cleanup failed: {e}"],
                "warnings": [],
            }

    def create_cleanup_backup(
        self, cache_paths: List[Path], backup_name: str
    ) -> Optional[Path]:
        """
        Creates comprehensive backup of cache directories before cleanup with archive creation,
        manifest generation, and integrity verification for safe cleanup operations.

        Args:
            cache_paths: List of paths to backup before cleanup
            backup_name: Name identifier for the backup archive

        Returns:
            Path: Path to created backup archive or None if backup creation failed
        """
        if not self.backup_enabled:
            return None

        try:
            # Create backup directory structure with timestamp organization
            backup_dir = self.root_directory / ".cache_backups"
            backup_dir.mkdir(exist_ok=True)

            # Generate backup manifest with cache paths and metadata
            manifest = {
                "backup_name": backup_name,
                "created_at": datetime.now().isoformat(),
                "root_directory": str(self.root_directory),
                "cache_paths": [str(p) for p in cache_paths],
                "backup_version": "1.0",
            }

            # Create manifest file
            manifest_path = backup_dir / f"{backup_name}_manifest.json"
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)

            # Create compressed backup archive using create_backup function
            if len(cache_paths) == 1:
                backup_path = create_backup(
                    source_path=cache_paths[0],
                    backup_type=backup_name,
                    backup_dir=backup_dir,
                )
            else:
                # For multiple paths, create a consolidated backup
                temp_backup_dir = backup_dir / f"{backup_name}_temp"
                temp_backup_dir.mkdir(exist_ok=True)

                try:
                    for i, cache_path in enumerate(cache_paths):
                        if cache_path.exists():
                            dest_path = temp_backup_dir / f"cache_{i}_{cache_path.name}"
                            if cache_path.is_file():
                                shutil.copy2(cache_path, dest_path)
                            else:
                                shutil.copytree(
                                    cache_path, dest_path, ignore_errors=True
                                )

                    backup_path = create_backup(
                        source_path=temp_backup_dir,
                        backup_type=backup_name,
                        backup_dir=backup_dir,
                    )
                finally:
                    # Clean up temporary directory
                    if temp_backup_dir.exists():
                        shutil.rmtree(temp_backup_dir, ignore_errors=True)

            # Register backup in backup_registry for tracking and cleanup
            if backup_path:
                backup_id = f"{backup_name}_{int(time.time())}"
                self.backup_registry[backup_id] = {
                    "backup_path": backup_path,
                    "manifest_path": manifest_path,
                    "created_at": time.time(),
                    "backup_name": backup_name,
                }

                self.logger.info(f"Backup created and registered: {backup_path}")
                return backup_path
            else:
                self.logger.error(f"Failed to create backup for {backup_name}")
                return None

        except Exception as e:
            self.logger.error(f"Backup creation failed: {e}")
            return None

    def rollback_cleanup(self, backup_id: str, verify_restoration: bool = True) -> bool:
        """
        Rolls back cleanup operations using created backups with selective restoration, integrity
        verification, and comprehensive error handling for cleanup recovery.

        Args:
            backup_id: Identifier for the backup to restore
            verify_restoration: Whether to verify restoration integrity

        Returns:
            bool: True if rollback successful, False otherwise
        """
        try:
            # Locate backup archive using backup_id from backup_registry
            if backup_id not in self.backup_registry:
                self.logger.error(f"Backup ID not found: {backup_id}")
                return False

            backup_info = self.backup_registry[backup_id]
            backup_path = backup_info["backup_path"]

            # Validate backup archive existence and integrity
            if not isinstance(backup_path, Path):
                backup_path = Path(backup_path)

            if not backup_path.exists():
                self.logger.error(f"Backup file not found: {backup_path}")
                return False

            # Create restoration plan with target directories and file mappings
            restore_location = self.root_directory / "restored_cache"

            self.logger.info(f"Rolling back cleanup using backup: {backup_path}")

            # Execute restoration using restore_backup function
            restoration_success = restore_backup(
                backup_path=backup_path,
                restore_location=restore_location,
                verify_integrity=verify_restoration,
            )

            if restoration_success:
                self.logger.info(
                    f"Rollback completed successfully to {restore_location}"
                )

                # Update cleanup statistics to reflect rollback operation
                self.cleanup_statistics["rollback_performed"] = True
                self.cleanup_statistics["rollback_backup_id"] = backup_id
                self.cleanup_statistics["rollback_timestamp"] = time.time()

                return True
            else:
                self.logger.error("Backup restoration failed")
                return False

        except Exception as e:
            self.logger.error(f"Rollback failed: {e}")
            return False

    def get_cleanup_summary(self, include_details: bool = False) -> Dict[str, Any]:
        """
        Generates comprehensive cleanup summary with statistics, warnings, recommendations,
        and operation details for reporting and audit purposes.

        Args:
            include_details: Whether to include detailed operation information

        Returns:
            dict: Dictionary containing comprehensive cleanup summary with statistics and recommendations
        """
        # Compile cleanup statistics from all cache type operations
        summary = {
            "overall_statistics": self.cleanup_statistics.copy(),
            "configuration_used": self.cleanup_configuration.copy(),
            "warnings_count": len(self.cleanup_warnings),
            "backup_enabled": self.backup_enabled,
            "dry_run_mode": self.dry_run_mode,
            "root_directory": str(self.root_directory),
        }

        # Generate warnings summary from cleanup_warnings list
        if self.cleanup_warnings:
            summary["warnings"] = self.cleanup_warnings[
                :10
            ]  # Limit to first 10 warnings
            if len(self.cleanup_warnings) > 10:
                summary["additional_warnings_count"] = len(self.cleanup_warnings) - 10

        # Create recommendations for future cache management
        recommendations = []

        total_space = self.cleanup_statistics.get("total_space_reclaimed_mb", 0)
        if total_space > 100:
            recommendations.append(
                "Consider more frequent cache cleanup to prevent large accumulations"
            )
        elif total_space > 50:
            recommendations.append("Current cleanup frequency appears appropriate")
        else:
            recommendations.append(
                "Cache accumulation is minimal - cleanup frequency can be reduced"
            )

        if self.cleanup_statistics.get("operations_failed", 0) > 0:
            recommendations.append(
                "Review failed operations and check file permissions"
            )

        if not self.backup_enabled:
            recommendations.append(
                "Consider enabling backups for safer cleanup operations"
            )

        summary["recommendations"] = recommendations

        # Include detailed operation information if include_details is True
        if include_details:
            summary["detailed_operations"] = {
                "backup_registry": {
                    k: {
                        "backup_path": str(v["backup_path"]),
                        "created_at": v["created_at"],
                        "backup_name": v["backup_name"],
                    }
                    for k, v in self.backup_registry.items()
                },
                "cleanup_warnings_detailed": self.cleanup_warnings,
                "performance_metrics": {
                    "operations_completed": self.cleanup_statistics.get(
                        "operations_completed", 0
                    ),
                    "operations_failed": self.cleanup_statistics.get(
                        "operations_failed", 0
                    ),
                    "success_rate": (
                        self.cleanup_statistics.get("operations_completed", 0)
                        / max(
                            1,
                            self.cleanup_statistics.get("operations_completed", 0)
                            + self.cleanup_statistics.get("operations_failed", 0),
                        )
                    )
                    * 100,
                },
            }

        # Generate backup and rollback information summary
        summary["backup_summary"] = {
            "backups_created": len(self.backup_registry),
            "backup_enabled": self.backup_enabled,
            "rollback_capability": len(self.backup_registry) > 0,
        }

        # Return comprehensive cleanup summary dictionary
        return summary

    def cleanup_old_backups(
        self, max_age_days: int = MAX_BACKUP_AGE_DAYS
    ) -> Dict[str, Any]:
        """
        Removes old backup files exceeding age limits with date-based filtering, space management,
        and proper cleanup of backup registry for backup maintenance.

        Args:
            max_age_days: Maximum age in days for backup files to be retained

        Returns:
            dict: Dictionary containing backup cleanup statistics and space reclaimed information
        """
        backup_cleanup_stats = {
            "old_backups_removed": 0,
            "space_reclaimed_mb": 0.0,
            "backups_preserved": 0,
            "registry_entries_cleaned": 0,
            "errors": [],
        }

        try:
            current_time = time.time()
            age_threshold = current_time - (max_age_days * 24 * 3600)

            # Scan backup directory for backup files with timestamp extraction
            backup_dir = self.root_directory / ".cache_backups"
            if not backup_dir.exists():
                return backup_cleanup_stats

            backups_to_remove = []
            registry_keys_to_remove = []

            # Filter backup files older than max_age_days using date comparison
            for backup_id, backup_info in self.backup_registry.items():
                backup_created = backup_info.get("created_at", current_time)

                if backup_created < age_threshold:
                    backups_to_remove.append(backup_info)
                    registry_keys_to_remove.append(backup_id)
                else:
                    backup_cleanup_stats["backups_preserved"] += 1

            # Also check for orphaned backup files
            for backup_file in backup_dir.glob("*.zip"):
                try:
                    file_mtime = backup_file.stat().st_mtime
                    if file_mtime < age_threshold:
                        # Check if this file is in registry
                        is_registered = any(
                            str(info.get("backup_path", "")) == str(backup_file)
                            for info in self.backup_registry.values()
                        )
                        if not is_registered:
                            backups_to_remove.append({"backup_path": backup_file})
                except Exception as e:
                    backup_cleanup_stats["errors"].append(
                        f"Error checking backup file {backup_file}: {e}"
                    )

            # Remove old backup files with proper error handling
            total_size_removed = 0
            for backup_info in backups_to_remove:
                try:
                    backup_path = backup_info["backup_path"]
                    if isinstance(backup_path, str):
                        backup_path = Path(backup_path)

                    if backup_path.exists():
                        file_size = backup_path.stat().st_size
                        backup_path.unlink()
                        total_size_removed += file_size
                        backup_cleanup_stats["old_backups_removed"] += 1

                        self.logger.debug(f"Removed old backup: {backup_path}")

                    # Remove associated manifest files
                    manifest_path = backup_path.with_name(
                        backup_path.stem + "_manifest.json"
                    )
                    if manifest_path.exists():
                        manifest_path.unlink()

                except Exception as e:
                    backup_cleanup_stats["errors"].append(f"Error removing backup: {e}")

            # Update backup_registry to remove references to deleted backups
            for key in registry_keys_to_remove:
                if key in self.backup_registry:
                    del self.backup_registry[key]
                    backup_cleanup_stats["registry_entries_cleaned"] += 1

            backup_cleanup_stats["space_reclaimed_mb"] = total_size_removed / (
                1024 * 1024
            )

            # Log backup cleanup progress with file counts and space reclaimed
            self.logger.info(
                f"Old backup cleanup completed: {backup_cleanup_stats['old_backups_removed']} backups removed, "
                f"{backup_cleanup_stats['space_reclaimed_mb']:.2f} MB reclaimed"
            )

            return backup_cleanup_stats

        except Exception as e:
            self.logger.error(f"Old backup cleanup failed: {e}")
            backup_cleanup_stats["errors"].append(f"Old backup cleanup failed: {e}")
            return backup_cleanup_stats

    def _cleanup_temp_files(self):
        """Internal method to clean up temporary files created during cleanup operations."""
        try:
            temp_patterns = [".cleanup_temp_*", "*.cleanup_tmp"]
            for pattern in temp_patterns:
                for temp_file in self.root_directory.glob(pattern):
                    try:
                        if temp_file.is_file():
                            temp_file.unlink()
                        elif temp_file.is_dir():
                            shutil.rmtree(temp_file)
                    except Exception:
                        pass  # Ignore cleanup errors for temporary files
        except Exception:
            pass  # Ignore all errors in temp cleanup


# Script execution guard for direct invocation
if __name__ == "__main__":
    sys.exit(main())
