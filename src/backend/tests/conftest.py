"""Minimal pytest configuration for the kata environment."""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register coverage and custom options defined in pytest.ini."""

    parser.addoption("--cov", action="append", default=[], help="Stub option.")
    parser.addoption("--cov-report", action="append", default=[], help="Stub option.")
    parser.addoption("--cov-branch", action="store_true", default=False, help="Stub option.")
    parser.addoption("--cov-fail-under", action="store", default=None, help="Stub option.")
    parser.addini("addopts_faulthandler", "Stub ini option for kata environment.", default="")
    parser.addini(
        "collect_ignore",
        "Stub ini option for kata environment.",
        type="linelist",
        default=[],
    )
    parser.addini(
        "collect_ignore_glob",
        "Stub ini option for kata environment.",
        type="linelist",
        default=[],
    )
    parser.addini(
        "random-order",
        "Stub ini option for kata environment.",
        default="false",
    )
    parser.addini(
        "random-order-bucket",
        "Stub ini option for kata environment.",
        default="none",
    )
