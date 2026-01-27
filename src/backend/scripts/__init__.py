"""Entry points for plume_nav_sim maintenance scripts."""

from __future__ import annotations

import logging
from typing import List, Optional

from .clean_cache import main as clean_cache_main
from .run_tests import TestExecutionConfig, TestResult, TestRunner
from .run_tests import main as run_tests_main
from .run_tests import run_test_suite
from .validate_installation import check_python_version
from .validate_installation import main as validate_installation_main
from .validate_installation import validate_environment_functionality

logger = logging.getLogger("plume_nav_sim.scripts")

__all__ = [
    "validate_installation",
    "run_tests",
    "run_benchmarks",
    "clean_cache",
    "check_python_version",
    "validate_environment_functionality",
    "run_test_suite",
    "TestRunner",
    "TestExecutionConfig",
    "TestResult",
]


def validate_installation(
    args: Optional[List[str]] = None,
    *,
    verbose: bool = False,
    performance_tests: bool = False,
) -> int:
    args = list(args or [])
    if verbose and "--verbose" not in args:
        args.append("--verbose")
    if performance_tests and "--performance-tests" not in args:
        args.append("--performance-tests")
    return validate_installation_main(args)


def run_tests(args: Optional[List[str]] = None) -> int:
    return run_tests_main(args)


def clean_cache() -> int:
    return clean_cache_main()


def run_benchmarks(_args: Optional[List[str]] = None) -> int:
    raise RuntimeError("Benchmark runner has been removed; use profiling tooling instead.")
