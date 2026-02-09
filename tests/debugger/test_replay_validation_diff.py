from __future__ import annotations

import os

import pytest

# Ensure Qt can run in headless CI (must be set before importing QtWidgets).
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtWidgets  # noqa: E402

from plume_nav_debugger.replay_driver import ReplayDriver  # noqa: E402
from plume_nav_debugger.widgets.replay_diff_dialog import ReplayDiffDialog  # noqa: E402
from plume_nav_sim.data_capture.replay import (  # noqa: E402
    ReplayConsistencyError,
    ReplayFieldMismatch,
    ReplayValidationDiff,
)
from plume_nav_sim.data_capture.schemas import RunMeta  # noqa: E402


@pytest.fixture(scope="session")
def qapp() -> QtWidgets.QApplication:
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


def test_replay_diff_dialog_formats_and_copies(
    qapp: QtWidgets.QApplication,  # noqa: ARG001
) -> None:
    dlg = ReplayDiffDialog()
    payload = {
        "run_dir": "/tmp/run-1",
        "global_step_index": 3,
        "episode_id": "ep-000001",
        "episode_step": 3,
        "action": 2,
        "mismatches": [{"field": "reward", "expected": 0.0, "actual": 1.0}],
    }
    dlg.set_payload(payload)

    assert "global_step_index" in dlg.text.toPlainText()
    dlg.copy_btn.click()
    assert "global_step_index" in QtWidgets.QApplication.clipboard().text()


def test_replay_driver_emits_validation_diff(monkeypatch, tmp_path, qapp):  # noqa: ARG001
    meta = RunMeta(
        run_id="run-1",
        experiment="exp",
        start_time="2026-02-08T00:00:00Z",
        env_config={"grid_size": [8, 8], "source_location": [4, 4]},
        extra=None,
    )
    diff = ReplayValidationDiff(
        run_dir=str(tmp_path),
        global_step_index=2,
        episode_id="ep-000001",
        episode_step=2,
        action=1,
        mismatches=(ReplayFieldMismatch("reward", 0.0, 1.0),),
    )

    class _Artifacts:
        steps: list = []

    class DummyEngine:
        run_meta = meta
        _artifacts = _Artifacts()

        def validate(self, *, env_factory, render=False, distance_tolerance=None, obs_tolerance=None):  # noqa: ARG002
            raise ReplayConsistencyError("boom", diff=diff)

        def iter_events(self, *, start_index=0, render=False):  # noqa: ARG002
            return iter([])

    monkeypatch.setattr(
        "plume_nav_debugger.replay_driver.load_replay_engine",
        lambda _run_dir: DummyEngine(),
    )

    d = ReplayDriver()
    msgs: list[str] = []
    diffs: list[object] = []
    d.error_occurred.connect(lambda msg: msgs.append(str(msg)))
    d.replay_validation_failed.connect(lambda payload: diffs.append(payload))

    d.load_run(tmp_path)

    assert msgs and "boom" in msgs[-1]
    assert diffs
    payload = diffs[-1]
    assert isinstance(payload, dict)
    assert payload["global_step_index"] == 2
    assert payload["mismatches"][0]["field"] == "reward"

