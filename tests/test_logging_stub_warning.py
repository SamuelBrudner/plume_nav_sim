import sys
import types
import importlib.metadata


def _install_loguru_stub(monkeypatch):
    messages = []
    loguru_stub = types.ModuleType("loguru")

    class StubLogger:
        def warning(self, msg, *args, **kwargs):
            messages.append(msg)

        def info(self, *args, **kwargs):
            pass

        def debug(self, *args, **kwargs):
            pass

        def error(self, *args, **kwargs):
            pass

        def remove(self, *args, **kwargs):
            pass

        def add(self, *args, **kwargs):
            return 0

        def bind(self, *args, **kwargs):
            return self

    loguru_stub.logger = StubLogger()
    monkeypatch.setitem(sys.modules, "loguru", loguru_stub)
    return messages


def _patch_distribution(monkeypatch):
    real_dist = importlib.metadata.distribution

    class DummyDist:
        def locate_file(self, path):
            return path

    def fake_distribution(name):
        if name == "plume_nav_sim":
            return DummyDist()
        return real_dist(name)

    monkeypatch.setattr(importlib.metadata, "distribution", fake_distribution)


def test_root_warns_on_stub(monkeypatch):
    messages = _install_loguru_stub(monkeypatch)
    _patch_distribution(monkeypatch)
    sys.modules.pop("plume_nav_sim", None)

    import plume_nav_sim  # noqa: F401

    assert any("limited" in m for m in messages)


def test_wind_warns_on_stub(monkeypatch):
    messages = _install_loguru_stub(monkeypatch)
    _patch_distribution(monkeypatch)
    sys.modules.pop("plume_nav_sim", None)

    import plume_nav_sim  # noqa: F401
    messages.clear()

    import types
    numba_stub = types.ModuleType("numba")
    numba_stub.jit = lambda *a, **k: (lambda f: f)
    numba_stub.prange = range
    monkeypatch.setitem(sys.modules, "numba", numba_stub)

    pandas_stub = types.ModuleType("pandas")
    monkeypatch.setitem(sys.modules, "pandas", pandas_stub)

    import plume_nav_sim.models.wind  # noqa: F401

    assert any("limited" in m for m in messages)
