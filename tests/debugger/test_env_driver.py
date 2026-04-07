from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtWidgets  # noqa: E402

from plume_nav_debugger.env_driver import DebuggerConfig, EnvDriver  # noqa: E402


@pytest.fixture(scope="session")
def qapp() -> QtWidgets.QApplication:
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


class _DummyActionSpace:
    def sample(self) -> int:
        return 0


class _ResetFailEnv:
    action_space = _DummyActionSpace()

    def reset(self, *, seed=None):  # noqa: ARG002
        raise RuntimeError("reset boom")

    def render(self, mode="rgb_array"):  # noqa: ARG002
        return np.zeros((2, 2, 3), dtype=np.uint8)

    def close(self) -> None:
        return None


class _WorkingEnv:
    action_space = _DummyActionSpace()

    def reset(self, *, seed=None):  # noqa: ARG002
        return np.zeros((1,), dtype=np.float32), {"agent_xy": (1, 2)}

    def render(self, mode="rgb_array"):  # noqa: ARG002
        return np.zeros((2, 2, 3), dtype=np.uint8)

    def close(self) -> None:
        return None


class _DummyController:
    def reset(self, *, seed=None) -> None:  # noqa: ARG002
        return None


def test_initialize_aborts_when_preview_reset_fails(
    monkeypatch,
    qapp: QtWidgets.QApplication,  # noqa: ARG001
) -> None:
    stream_calls: list[tuple[tuple, dict]] = []

    monkeypatch.setattr("plume_nav_sim.make_env", lambda **kwargs: _ResetFailEnv())
    monkeypatch.setattr(
        "plume_nav_sim.runner.runner.stream",
        lambda *args, **kwargs: stream_calls.append((args, kwargs)) or iter(()),
    )
    monkeypatch.setattr(
        "plume_nav_debugger.frame_overlays.OverlayInfoWrapper",
        lambda env: env,
        raising=False,
    )

    driver = EnvDriver(DebuggerConfig(action_type="discrete"))
    messages: list[str] = []
    driver.error_occurred.connect(lambda msg: messages.append(str(msg)))

    driver.initialize()

    assert messages
    assert messages[-1] == "Env reset failed: reset boom"
    assert driver._iter is None
    assert stream_calls == []


def test_initialize_emits_error_when_stream_creation_fails(
    monkeypatch,
    qapp: QtWidgets.QApplication,  # noqa: ARG001
) -> None:
    monkeypatch.setattr("plume_nav_sim.make_env", lambda **kwargs: _WorkingEnv())
    monkeypatch.setattr(
        "plume_nav_sim.runner.runner.stream",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("stream boom")),
    )
    monkeypatch.setattr(
        "plume_nav_debugger.frame_overlays.OverlayInfoWrapper",
        lambda env: env,
        raising=False,
    )

    driver = EnvDriver(DebuggerConfig(action_type="discrete"))
    messages: list[str] = []
    driver.error_occurred.connect(lambda msg: messages.append(str(msg)))

    driver.initialize()

    assert messages
    assert messages[-1] == "Runner stream init failed: stream boom"
    assert driver._iter is None


def test_reset_aborts_when_preview_reset_fails(
    monkeypatch,
    qapp: QtWidgets.QApplication,  # noqa: ARG001
) -> None:
    stream_calls: list[tuple[tuple, dict]] = []

    monkeypatch.setattr(
        "plume_nav_sim.runner.runner.stream",
        lambda *args, **kwargs: stream_calls.append((args, kwargs)) or iter(()),
    )

    driver = EnvDriver(DebuggerConfig(action_type="discrete"))
    driver._env = _ResetFailEnv()
    driver._controller = _DummyController()
    driver._iter = iter(())

    messages: list[str] = []
    driver.error_occurred.connect(lambda msg: messages.append(str(msg)))

    driver.reset(seed=42)

    assert messages
    assert messages[-1] == "Reset failed: reset boom"
    assert driver._iter is None
    assert stream_calls == []


def test_recreate_env_aborts_when_preview_reset_fails(
    monkeypatch,
    qapp: QtWidgets.QApplication,  # noqa: ARG001
) -> None:
    stream_calls: list[tuple[tuple, dict]] = []

    monkeypatch.setattr("plume_nav_sim.make_env", lambda **kwargs: _ResetFailEnv())
    monkeypatch.setattr(
        "plume_nav_sim.runner.runner.stream",
        lambda *args, **kwargs: stream_calls.append((args, kwargs)) or iter(()),
    )
    monkeypatch.setattr(
        "plume_nav_debugger.frame_overlays.OverlayInfoWrapper",
        lambda env: env,
        raising=False,
    )

    driver = EnvDriver(DebuggerConfig(action_type="discrete"))
    driver._controller = _DummyController()

    messages: list[str] = []
    driver.error_occurred.connect(lambda msg: messages.append(str(msg)))

    driver._recreate_env((3, 4))

    assert messages
    assert messages[-1] == "Reset failed after env recreation: reset boom"
    assert driver._iter is None
    assert stream_calls == []
