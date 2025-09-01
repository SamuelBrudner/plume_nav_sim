import importlib
import sys
import builtins
import logging
import pytest

AGENT_MODULES = [
    "plume_nav_sim.examples.agents.infotaxis_agent",
    "plume_nav_sim.examples.agents.casting_agent",
    "plume_nav_sim.examples.agents.reactive_agent",
]


@pytest.mark.parametrize("module_path", AGENT_MODULES)
def test_protocol_import_logs_success(module_path, caplog):
    import plume_nav_sim.examples.agents  # ensure package is loaded

    caplog.set_level(logging.DEBUG)
    if module_path in sys.modules:
        del sys.modules[module_path]
    importlib.import_module(module_path)
    assert any(
        "NavigatorProtocol" in message and "import" in message
        for message in caplog.messages
    )


@pytest.mark.parametrize("module_path", AGENT_MODULES)
def test_protocol_import_missing_dependency_raises(module_path, monkeypatch):
    import plume_nav_sim.examples.agents  # ensure package is loaded

    if module_path in sys.modules:
        del sys.modules[module_path]
    original_import = builtins.__import__

    def mock_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "plume_nav_sim.protocols.navigator":
            raise ImportError("mocked missing dependency")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    with pytest.raises(ImportError, match="NavigatorProtocol"):
        importlib.import_module(module_path)
