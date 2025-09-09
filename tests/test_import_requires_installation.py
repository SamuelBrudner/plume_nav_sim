"""Validate import behaviour when package metadata is missing."""

import importlib.metadata
import sys

import pytest


def test_import_warns_when_not_installed(monkeypatch):
    from importlib.metadata import PackageNotFoundError

    def fake_distribution(name: str):
        raise PackageNotFoundError

    monkeypatch.setattr(importlib.metadata, "distribution", fake_distribution)
    sys.modules.pop("plume_nav_sim", None)

    with pytest.warns(UserWarning, match="pip install -e"):
        __import__("plume_nav_sim")
