import importlib.util
import pathlib
import sys
import types

import pytest


def test_import_error_when_db_missing(monkeypatch):
    """Importing CLI main should fail if the database session module is absent."""
    # Simulate missing database session layer
    monkeypatch.setitem(sys.modules, "plume_nav_sim.db.session", None)

    path = (
        pathlib.Path(__file__).resolve().parents[2]
        / "src"
        / "plume_nav_sim"
        / "cli"
        / "main.py"
    )
    spec = importlib.util.spec_from_file_location("plume_nav_sim.cli.main", path)
    module = importlib.util.module_from_spec(spec)

    with pytest.raises(ImportError) as exc:
        assert spec.loader is not None
        spec.loader.exec_module(module)

    assert "database layer" in str(exc.value).lower()
