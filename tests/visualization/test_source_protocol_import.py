import importlib
import sys
import types

import pytest
import io
from loguru import logger


def test_source_protocol_import_logs_success(monkeypatch):
    """Visualization module should log successful SourceProtocol import."""
    stream = io.StringIO()

    def fake_get_module_logger(name):
        handler = logger.StreamHandler(stream)
        logger.handlers = [handler]
        logger.setLevel(logger.INFO)
        return logger

    import plume_nav_sim.utils.logging_setup as logging_setup
    monkeypatch.setattr(logging_setup, 'get_module_logger', fake_get_module_logger)

    monkeypatch.delitem(sys.modules, 'plume_nav_sim.utils.visualization', raising=False)
    importlib.import_module('plume_nav_sim.utils.visualization')
    assert "Successfully imported SourceProtocol" in stream.getvalue()


def test_source_protocol_import_error_propagated(monkeypatch):
    """Visualization module should propagate ImportError when SourceProtocol is missing."""
    monkeypatch.delitem(sys.modules, 'plume_nav_sim.utils.visualization', raising=False)
    dummy_module = types.ModuleType('plume_nav_sim.core.protocols')
    monkeypatch.setitem(sys.modules, 'plume_nav_sim.core.protocols', dummy_module)
    with pytest.raises(ImportError):
        importlib.import_module('plume_nav_sim.utils.visualization')
