from __future__ import annotations

from plume_nav_debugger.odc import discovery
from plume_nav_debugger.odc.provider import DebuggerProvider


def test_find_provider_prefers_env_reflection(monkeypatch):
    monkeypatch.setattr(discovery, "_load_entry_point_provider", lambda: None)

    class DummyProvider(DebuggerProvider):
        pass

    class DummyEnv:
        def get_debugger_provider(self):
            return DummyProvider()

    provider = discovery.find_provider(DummyEnv(), object())
    assert isinstance(provider, DummyProvider)
