import importlib.metadata
import sys

import pytest


def test_import_requires_installation(monkeypatch):
    """Importing without installed distribution should fail."""
    from importlib.metadata import PackageNotFoundError

    def fake_distribution(name: str):
        raise PackageNotFoundError

    monkeypatch.setattr(importlib.metadata, "distribution", fake_distribution)
    sys.modules.pop("plume_nav_sim", None)

    with pytest.raises(ImportError, match="pip install -e"):
        __import__("plume_nav_sim")
