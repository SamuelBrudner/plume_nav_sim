import importlib.util
import pathlib
import sys

import pytest


def test_import_error_when_hydra_missing(monkeypatch):
    """Importing CLI main should fail if Hydra is absent."""
    monkeypatch.setitem(sys.modules, "hydra", None)

    path = pathlib.Path(__file__).resolve().parents[2] / "src" / "plume_nav_sim" / "cli" / "main.py"
    spec = importlib.util.spec_from_file_location("plume_nav_sim.cli.main", path)
    module = importlib.util.module_from_spec(spec)

    with pytest.raises(ImportError) as exc:
        assert spec.loader is not None
        spec.loader.exec_module(module)

    assert "hydra" in str(exc.value).lower()
