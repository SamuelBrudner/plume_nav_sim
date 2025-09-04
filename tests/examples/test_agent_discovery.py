import importlib
import builtins
import logging
import types
import sys
import pytest

# Stub optional video plume dependency to ensure examples package imports
# Stub optional video plume dependency to ensure examples package imports
video_plume_stub = types.ModuleType("video_plume")

class VideoPlume:  # minimal stub
    pass

video_plume_stub.VideoPlume = VideoPlume
sys.modules.setdefault("plume_nav_sim.envs.video_plume", video_plume_stub)

# Stub spaces factory dependency
spaces_stub = types.ModuleType("spaces")

class SpacesFactory:  # minimal stub
    pass

spaces_stub.SpacesFactory = SpacesFactory
sys.modules.setdefault("plume_nav_sim.envs.spaces", spaces_stub)

import plume_nav_sim.examples as examples


def test_missing_agent_not_exposed(monkeypatch, caplog):
    """Agents failing to import should not be exposed and should log warning."""
    original_import = builtins.__import__

    def mock_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "plume_nav_sim.examples.agents.reactive_agent":
            raise ImportError("mocked missing module")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    with caplog.at_level(logging.WARNING):
        importlib.reload(examples)

    assert "ReactiveAgent" not in examples.list_available_example_agents()
    assert any(
        "ReactiveAgent" in record.getMessage() and "unavailable" in record.getMessage()
        for record in caplog.records
    )
    with pytest.raises(KeyError):
        examples.get_agent_info("ReactiveAgent")

    monkeypatch.setattr(builtins, "__import__", original_import, raising=False)
    importlib.reload(examples)
