from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtCore, QtWidgets  # noqa: E402

import plume_nav_debugger.main_window as main_window  # noqa: E402
from plume_nav_debugger.replay_driver import ReplayLoadError  # noqa: E402


@pytest.fixture(scope="session")
def qapp() -> QtWidgets.QApplication:
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


class _DummyDriver(QtCore.QObject):
    frame_ready = QtCore.Signal(object)
    step_done = QtCore.Signal(object)
    episode_finished = QtCore.Signal()
    action_space_changed = QtCore.Signal(list)
    policy_changed = QtCore.Signal(object)
    provider_mux_changed = QtCore.Signal(object)
    run_meta_changed = QtCore.Signal(int, object)
    error_occurred = QtCore.Signal(str)
    replay_validation_failed = QtCore.Signal(object)
    timeline_changed = QtCore.Signal(int, int)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self._run_dir: Path | None = None

    def initialize(self) -> None:
        return None

    def set_policy_explore(self, enabled: bool) -> None:  # noqa: ARG002
        return None

    def get_action_names(self) -> list[str]:
        return []

    def get_grid_size(self) -> tuple[int, int]:
        return (8, 8)

    def is_running(self) -> bool:
        return False

    def pause(self) -> None:
        return None

    def start(self, interval_ms: int = 50) -> None:  # noqa: ARG002
        return None

    def step_once(self) -> None:
        return None

    def step_back(self) -> None:
        return None

    def reset(self, seed=None) -> None:  # noqa: ARG002
        return None

    def close(self) -> None:
        return None


class _ReplayLoadFailDriver(_DummyDriver):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._loaded = False

    def load_run(self, path: Path) -> None:  # noqa: ARG002
        raise ReplayLoadError("boom")

    def is_loaded(self) -> bool:
        return self._loaded

    def total_steps(self) -> int:
        return 0

    def current_index(self) -> int:
        return 0

    def total_episodes(self) -> int:
        return 0

    def current_episode_index(self) -> int:
        return 0

    def get_timeline_markers(self) -> dict[str, list[int]]:
        return {}


def _make_window(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,  # noqa: ARG001
) -> main_window.MainWindow:
    monkeypatch.setattr(main_window, "EnvDriver", _DummyDriver)
    monkeypatch.setattr(main_window, "ReplayDriver", _ReplayLoadFailDriver)
    monkeypatch.setattr(main_window.QtCore.QTimer, "singleShot", lambda *args, **kwargs: None)
    window = main_window.MainWindow()
    return window


def test_replay_load_failure_reverts_to_live_driver(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,  # noqa: ARG001
) -> None:
    window = _make_window(monkeypatch, qapp)
    try:
        window._load_replay_from_path(Path("/tmp/missing-run"))

        assert window._active_driver is window.live_driver
        assert window.statusBar().currentMessage() == "Replay load failed: boom"
    finally:
        window.close()


def test_driver_switch_pause_failure_surfaces_status(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,  # noqa: ARG001
) -> None:
    window = _make_window(monkeypatch, qapp)
    try:
        monkeypatch.setattr(
            window.live_driver,
            "pause",
            lambda: (_ for _ in ()).throw(RuntimeError("pause boom")),
        )
        window._switch_driver(window.replay_driver)

        assert (
            window.statusBar().currentMessage()
            == "Driver pause failed during switch: pause boom"
        )
    finally:
        window.close()


def test_timeline_refresh_failure_surfaces_status(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,  # noqa: ARG001
) -> None:
    window = _make_window(monkeypatch, qapp)
    try:
        monkeypatch.setattr(
            window.controls,
            "set_timeline",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("timeline boom")),
        )

        window._refresh_timeline_controls()

        assert window.statusBar().currentMessage() == "Timeline refresh failed: timeline boom"
    finally:
        window.close()


def test_replay_marker_refresh_failure_surfaces_status(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QtWidgets.QApplication,  # noqa: ARG001
) -> None:
    window = _make_window(monkeypatch, qapp)
    try:
        monkeypatch.setattr(
            window.controls,
            "set_replay_markers",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("markers boom")),
        )

        window._refresh_replay_markers()

        assert window.statusBar().currentMessage() == "Replay marker refresh failed: markers boom"
    finally:
        window.close()
