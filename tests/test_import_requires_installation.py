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


def test_import_requires_installation_logs(monkeypatch):
    """Importing without installation logs clear instructions."""
    from importlib.metadata import PackageNotFoundError
    from loguru import logger

    def fake_distribution(name: str):
        raise PackageNotFoundError

    logs = []

    monkeypatch.setattr(importlib.metadata, "distribution", fake_distribution)
    monkeypatch.setattr(logger, "error", lambda msg: logs.append(msg))
    sys.modules.pop("plume_nav_sim", None)

    with pytest.raises(
        ImportError,
        match=r"setup_env\.sh --dev' or 'pip install -e \.",
    ):
        __import__("plume_nav_sim")
    assert any("setup_env.sh --dev" in m for m in logs)
