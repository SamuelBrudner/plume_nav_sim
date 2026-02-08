from __future__ import annotations

import os

import pytest

# Ensure Qt can run in headless CI (must be set before importing QtWidgets).
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtWidgets  # noqa: E402

from plume_nav_debugger.replay_driver import ReplayDriver  # noqa: E402
from plume_nav_debugger.widgets.control_bar import ControlBar  # noqa: E402


@pytest.fixture(scope="session")
def qapp() -> QtWidgets.QApplication:
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


class _Ev:
    def __init__(
        self,
        *,
        terminated: bool = False,
        truncated: bool = False,
        info: dict | None = None,
    ) -> None:
        self.terminated = bool(terminated)
        self.truncated = bool(truncated)
        self.info = info or {}


def test_replay_driver_markers_and_semantic_jumps(
    qapp: QtWidgets.QApplication,  # noqa: ARG001
) -> None:
    d = ReplayDriver()
    d._engine = object()  # type: ignore[assignment]
    d._events = [
        _Ev(),
        _Ev(terminated=True, info={"goal_reached": True}),
        _Ev(),
        _Ev(truncated=True),
        _Ev(),
    ]
    d._episode_starts = [0, 3]
    d._index = 0

    markers = d.get_timeline_markers()
    assert markers["episode"] == [0, 3]
    assert markers["terminated"] == [1]
    assert markers["truncated"] == [3]
    assert markers["goal"] == [1]

    assert d.jump_next_done() is True
    assert d.current_index() == 1

    # No additional goal event after index=1.
    assert d.jump_next_goal() is False
    assert d.current_index() == 1

    assert d.jump_next_episode() is True
    assert d.current_index() == 3

    assert d.jump_prev_episode() is True
    assert d.current_index() == 0


def test_control_bar_replay_jump_signals_and_markers(
    qapp: QtWidgets.QApplication,  # noqa: ARG001
) -> None:
    c = ControlBar()
    c.set_timeline(5, 0, total_episodes=2, current_episode=0)

    got: list[str] = []
    c.jump_prev_episode_requested.connect(lambda: got.append("prev_ep"))
    c.jump_next_episode_requested.connect(lambda: got.append("next_ep"))
    c.jump_next_done_requested.connect(lambda: got.append("next_done"))
    c.jump_next_goal_requested.connect(lambda: got.append("next_goal"))

    c.prev_episode_btn.click()
    c.next_episode_btn.click()
    c.next_done_btn.click()
    c.next_goal_btn.click()

    assert got == ["prev_ep", "next_ep", "next_done", "next_goal"]

    c.set_replay_markers({"episode": [0, 2], "truncated": [2]})
    assert c.timeline_slider.get_markers()["episode"] == [0, 2]
