"""Entry points for plume_nav_sim maintenance scripts."""

from __future__ import annotations

import logging
from typing import List, Optional

from .clean_cache import main as clean_cache_main
from .run_tests import TestExecutionConfig, TestResult, TestRunner
from .run_tests import main as run_tests_main
from .run_tests import run_test_suite

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
    del args, verbose, performance_tests
    raise RuntimeError(
        "validate_installation.py has been archived. "
        "Use `pytest`, `python -m plume_nav_sim.scripts.run_tests`, and "
        "`python scripts/validate_constants_drift.py` for checks."
    )


def check_python_version(*_args, **_kwargs) -> bool:
    raise RuntimeError(
        "check_python_version helper was archived with validate_installation.py"
    )


def validate_environment_functionality(*_args, **_kwargs) -> bool:
    raise RuntimeError(
        "validate_environment_functionality helper was archived "
        "with validate_installation.py"
    )


def run_tests(args: Optional[List[str]] = None) -> int:
    return run_tests_main(args)


def clean_cache() -> int:
    return clean_cache_main()


def run_benchmarks(_args: Optional[List[str]] = None) -> int:
    raise RuntimeError("Benchmark runner has been removed; use profiling tooling instead.")
