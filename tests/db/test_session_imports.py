import builtins
import importlib.util
import pathlib
import sys

import pytest


def test_session_import_raises_without_sqlalchemy(monkeypatch):
    monkeypatch.delitem(sys.modules, 'sqlalchemy', raising=False)

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith('sqlalchemy'):
            raise ImportError('No module named sqlalchemy')
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', fake_import)

    session_path = pathlib.Path(__file__).resolve().parents[2] / 'src' / 'plume_nav_sim' / 'db' / 'session.py'
    spec = importlib.util.spec_from_file_location('session', session_path)
    with pytest.raises(ImportError, match="sqlalchemy"):
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
