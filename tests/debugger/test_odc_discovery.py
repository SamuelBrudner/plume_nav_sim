import types


def test_entry_point_provider_precedence(monkeypatch):
    # Fake entry point structure
    class _EP:
        def __init__(self, loader):
            self._loader = loader

        def load(self):
            return self._loader

    class _Selection:
        def __init__(self, items):
            self._items = items

        def select(self, group):
            if group == "plume_nav_sim.debugger_plugins":
                return self._items
            return []

    # Provide a DebuggerProvider factory
    from plume_nav_debugger.odc.provider import DebuggerProvider

    class _MyProvider(DebuggerProvider):
        pass

    def _factory(env, policy):
        return _MyProvider()

    fake_eps = _Selection([_EP(_factory)])

    import importlib.metadata as im

    monkeypatch.setattr(im, "entry_points", lambda: fake_eps)

    # Reflection fallback that should NOT be used
    class Env:
        def get_debugger_provider(self):
            return None

    class Pol:
        def get_debugger_provider(self):
            return None

    from plume_nav_debugger.odc.discovery import find_provider

    p = find_provider(Env(), Pol())
    assert isinstance(p, DebuggerProvider)
