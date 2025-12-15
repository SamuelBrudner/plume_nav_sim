from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

try:
    from PySide6 import QtCore, QtGui, QtWidgets
except Exception as e:  # pragma: no cover - GUI import guard
    raise RuntimeError(
        "PySide6 is required for the debugger app. Install it in your env (e.g., conda run -n plume-nav-sim pip install PySide6)."
    ) from e

from plume_nav_debugger.config import (
    LEGACY_QSETTINGS_ORG,
    QSETTINGS_APP,
    QSETTINGS_ORG,
    DebuggerPreferences,
)
from plume_nav_debugger.env_driver import DebuggerConfig, EnvDriver
from plume_nav_debugger.inspector.introspection import format_pipeline
from plume_nav_debugger.inspector.models import ActionPanelModel, ObservationPanelModel
from plume_nav_debugger.inspector.plots import normalize_series_to_polyline
from plume_nav_debugger.odc.mux import ProviderMux
from plume_nav_debugger.replay_driver import ReplayDriver, ReplayLoadError


class FrameView(QtWidgets.QLabel):
    def __init__(self) -> None:
        super().__init__()
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setMinimumSize(320, 320)
        self._last_image = None

    @QtCore.Slot(np.ndarray)
    def update_frame(self, frame: np.ndarray) -> None:
        # Expect HxWx3 uint8
        if not isinstance(frame, np.ndarray) or frame.ndim != 3:
            return
        h, w, c = frame.shape
        if c != 3:
            return
        img = QtGui.QImage(frame.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(img.copy())
        self.setPixmap(pix.scaled(self.size(), QtCore.Qt.KeepAspectRatio))
        self._last_image = pix

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self._last_image is not None:
            self.setPixmap(
                self._last_image.scaled(self.size(), QtCore.Qt.KeepAspectRatio)
            )


class ControlBar(QtWidgets.QWidget):
    start = QtCore.Signal()
    pause = QtCore.Signal()
    step = QtCore.Signal()
    reset = QtCore.Signal(int)
    mode_changed = QtCore.Signal(str)
    load_replay = QtCore.Signal()
    seek_requested = QtCore.Signal(int)
    episode_seek_requested = QtCore.Signal(int)

    def __init__(self) -> None:
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        main_row = QtWidgets.QHBoxLayout()

        self.start_btn = QtWidgets.QPushButton("Start")
        self.pause_btn = QtWidgets.QPushButton("Pause")
        self.step_btn = QtWidgets.QPushButton("Step")
        # Mode and replay loader
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["Live", "Replay"])
        self.load_replay_btn = QtWidgets.QPushButton("Load Replay…")
        # Speed control
        self.interval_spin = QtWidgets.QSpinBox()
        self.interval_spin.setRange(5, 2000)
        self.interval_spin.setSingleStep(5)
        self.interval_spin.setValue(50)
        # Policy selection
        self.policy_combo = QtWidgets.QComboBox()
        self.policy_combo.addItems(
            [
                "Greedy TD (bacterial)",
                "Stochastic TD",
                "Deterministic TD",
                "Random Sampler",
            ]
        )
        self.custom_policy_edit = QtWidgets.QLineEdit()
        self.custom_policy_edit.setPlaceholderText("custom.module:ClassOrCallable")
        self.custom_load_btn = QtWidgets.QPushButton("Load")
        self.seed_edit = QtWidgets.QLineEdit()
        self.seed_edit.setPlaceholderText("Seed")
        self.reset_btn = QtWidgets.QPushButton("Reset")

        main_row.addWidget(self.start_btn)
        main_row.addWidget(self.pause_btn)
        main_row.addWidget(self.step_btn)
        main_row.addSpacing(8)
        main_row.addWidget(QtWidgets.QLabel("Mode:"))
        main_row.addWidget(self.mode_combo)
        main_row.addWidget(self.load_replay_btn)
        main_row.addSpacing(8)
        main_row.addWidget(QtWidgets.QLabel("Policy:"))
        main_row.addWidget(self.policy_combo)
        main_row.addWidget(self.custom_policy_edit)
        main_row.addWidget(self.custom_load_btn)
        main_row.addSpacing(8)
        main_row.addWidget(QtWidgets.QLabel("Interval (ms):"))
        main_row.addWidget(self.interval_spin)
        main_row.addStretch(1)
        main_row.addWidget(QtWidgets.QLabel("Seed:"))
        main_row.addWidget(self.seed_edit)
        main_row.addWidget(self.reset_btn)

        # Replay timeline row (hidden in live mode)
        self._timeline_row = QtWidgets.QWidget()
        replay_row = QtWidgets.QHBoxLayout(self._timeline_row)
        replay_row.setContentsMargins(0, 0, 0, 0)
        self.replay_label = QtWidgets.QLabel("Replay: none loaded")
        self.timeline_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.timeline_slider.setRange(0, 0)
        self.timeline_slider.setEnabled(False)
        self.timeline_spin = QtWidgets.QSpinBox()
        self.timeline_spin.setRange(0, 0)
        self.timeline_spin.setEnabled(False)
        self.episode_spin = QtWidgets.QSpinBox()
        self.episode_spin.setRange(1, 1)
        self.episode_spin.setEnabled(False)
        self.timeline_status = QtWidgets.QLabel("Step: -/-")
        self.episode_status = QtWidgets.QLabel("Episode: -/-")

        replay_row.addWidget(self.replay_label)
        replay_row.addSpacing(8)
        replay_row.addWidget(self.timeline_slider, stretch=1)
        replay_row.addWidget(self.timeline_spin)
        replay_row.addSpacing(6)
        replay_row.addWidget(QtWidgets.QLabel("Episode:"))
        replay_row.addWidget(self.episode_spin)
        replay_row.addWidget(self.timeline_status)
        replay_row.addWidget(self.episode_status)
        self._timeline_row.setVisible(False)

        layout.addLayout(main_row)
        layout.addWidget(self._timeline_row)

        self.start_btn.clicked.connect(self.start)
        self.pause_btn.clicked.connect(self.pause)
        self.step_btn.clicked.connect(self.step)
        self.reset_btn.clicked.connect(self._emit_reset)
        self.mode_combo.currentTextChanged.connect(self.mode_changed)
        self.load_replay_btn.clicked.connect(self.load_replay)
        self.timeline_slider.sliderReleased.connect(self._emit_seek_from_slider)
        self.timeline_slider.sliderMoved.connect(self._on_slider_moved)
        self.timeline_spin.editingFinished.connect(self._emit_seek_from_spin)
        self.timeline_slider.valueChanged.connect(self._sync_slider_spin)
        self.episode_spin.editingFinished.connect(self._emit_seek_from_episode)
        # Expose additional signals
        # Note: MainWindow will connect signals directly to handlers

        self._updating_timeline = False

    @QtCore.Slot()
    def _emit_reset(self) -> None:
        text = self.seed_edit.text().strip()
        # If seed field is empty or not an int, emit a sentinel -1 and let driver reuse last seed
        if text.isdigit():
            self.reset.emit(int(text))
        else:
            # Use -1 to indicate "reuse last episode seed"
            self.reset.emit(-1)

    def set_mode(self, mode: str) -> None:
        is_replay = str(mode).lower().startswith("replay")
        self.policy_combo.setEnabled(not is_replay)
        self.custom_policy_edit.setEnabled(not is_replay)
        self.custom_load_btn.setEnabled(not is_replay)
        self.seed_edit.setEnabled(not is_replay)
        self._timeline_row.setVisible(is_replay)

    def set_replay_label(self, text: str) -> None:
        self.replay_label.setText(f"Replay: {text}")

    def set_timeline(
        self,
        total_steps: int,
        current_step: int,
        *,
        total_episodes: int | None = None,
        current_episode: int | None = None,
    ) -> None:
        self._updating_timeline = True
        total = max(0, int(total_steps))
        cur = max(-1, int(current_step))
        ep_total = None if total_episodes is None else max(0, int(total_episodes))
        ep_cur = None if current_episode is None else max(-1, int(current_episode))
        if total <= 0:
            self.timeline_slider.setEnabled(False)
            self.timeline_spin.setEnabled(False)
            self.timeline_slider.setRange(0, 0)
            self.timeline_spin.setRange(0, 0)
            self.timeline_spin.setValue(0)
            self.timeline_status.setText("Step: -/-")
        else:
            max_idx = max(0, total - 1)
            cur = min(max_idx, max(cur, 0))
            self.timeline_slider.setEnabled(True)
            self.timeline_spin.setEnabled(True)
            self.timeline_slider.setRange(0, max_idx)
            self.timeline_spin.setRange(0, max_idx)
            self.timeline_slider.setValue(cur)
            self.timeline_spin.setValue(cur)
            self.timeline_status.setText(f"Step: {cur}/{max_idx}")
        if ep_total is None or ep_total <= 0:
            self.episode_spin.setEnabled(False)
            self.episode_spin.setRange(1, 1)
            self.episode_spin.setValue(1)
            self.episode_status.setText("Episode: -/-")
        else:
            max_ep = max(0, ep_total - 1)
            ep_cur = min(max_ep, max(ep_cur or 0, 0))
            # Display episodes as 1-based for UI
            self.episode_spin.setEnabled(True)
            self.episode_spin.setRange(1, ep_total)
            self.episode_spin.setValue(ep_cur + 1)
            self.episode_status.setText(f"Episode: {ep_cur + 1}/{ep_total}")
        self._updating_timeline = False

    @QtCore.Slot()
    def _emit_seek_from_slider(self) -> None:
        if self.timeline_slider.isEnabled():
            self.seek_requested.emit(int(self.timeline_slider.value()))

    @QtCore.Slot()
    def _emit_seek_from_spin(self) -> None:
        if self.timeline_spin.isEnabled():
            self.seek_requested.emit(int(self.timeline_spin.value()))

    @QtCore.Slot()
    def _emit_seek_from_episode(self) -> None:
        if self.episode_spin.isEnabled():
            self.episode_seek_requested.emit(int(self.episode_spin.value()) - 1)

    @QtCore.Slot(int)
    def _on_slider_moved(self, value: int) -> None:
        if not self._updating_timeline:
            self.timeline_spin.setValue(int(value))

    @QtCore.Slot(int)
    def _sync_slider_spin(self, value: int) -> None:
        if not self._updating_timeline:
            self.timeline_spin.setValue(int(value))


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Plume Nav Debugger (MVP)")
        self.resize(800, 700)

        self.frame_view = FrameView()
        self.controls = ControlBar()
        self.inspector = InspectorWidget()
        self._total_reward: float = 0.0

        central = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout(central)
        vbox.addWidget(self.frame_view, stretch=1)
        vbox.addWidget(self.controls, stretch=0)
        self.setCentralWidget(central)

        # Dockable inspector window (can float or dock)
        self.inspector_dock = QtWidgets.QDockWidget("Inspector", self)
        self.inspector_dock.setObjectName("InspectorDock")
        self.inspector_dock.setWidget(self.inspector)
        self.inspector_dock.setFeatures(
            QtWidgets.QDockWidget.DockWidgetMovable
            | QtWidgets.QDockWidget.DockWidgetFloatable
            | QtWidgets.QDockWidget.DockWidgetClosable
        )
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, self.inspector_dock)

        # Menu to toggle inspector visibility
        view_menu = self.menuBar().addMenu("View")
        self.action_toggle_inspector = QtGui.QAction("Inspector", self, checkable=True)
        self.action_toggle_inspector.setChecked(True)
        self.action_toggle_inspector.toggled.connect(self.inspector_dock.setVisible)
        self.inspector_dock.visibilityChanged.connect(
            self.action_toggle_inspector.setChecked
        )
        view_menu.addAction(self.action_toggle_inspector)
        edit_menu = self.menuBar().addMenu("Edit")
        prefs_action = QtGui.QAction("Preferences…", self)
        prefs_action.triggered.connect(self._open_preferences)
        edit_menu.addAction(prefs_action)

        # Restore UI layout/state
        self._restore_ui_state()

        # Load preferences and configure driver
        self.prefs = DebuggerPreferences.initial_load()
        cfg = DebuggerConfig.from_env()
        # Enforce strict provider-only mode via Inspector (always on; not configurable)
        self.live_driver = EnvDriver(cfg)
        self.replay_driver = ReplayDriver()
        self._active_driver = None
        self._active_mode = "live"
        self.controls.start.connect(self._on_start_clicked)
        self.controls.pause.connect(self._on_pause_clicked)
        self.controls.step.connect(self._on_step_clicked)
        self.controls.reset.connect(self._on_reset_clicked)
        self.controls.interval_spin.valueChanged.connect(self._on_interval_changed)
        self.controls.policy_combo.currentIndexChanged.connect(self._on_policy_selected)
        self.controls.custom_load_btn.clicked.connect(self._on_custom_policy_load)
        self.controls.mode_changed.connect(self._on_mode_changed)
        self.controls.load_replay.connect(self._on_load_replay)
        self.controls.seek_requested.connect(self._on_seek_requested)
        self.controls.episode_seek_requested.connect(self._on_episode_seek_requested)

        # Status bar showing step/total reward/flags and run meta
        self._status = QtWidgets.QLabel("ready")
        self._run_meta = QtWidgets.QLabel("")
        self.statusBar().addPermanentWidget(self._status, 1)
        self.statusBar().addPermanentWidget(self._run_meta, 0)

        # Keyboard shortcuts
        self._setup_shortcuts()

        # Start in live mode and initialize once to show first frame
        self._switch_driver(self.live_driver)
        QtCore.QTimer.singleShot(0, self.live_driver.initialize)
        # Also refresh action names shortly after init
        QtCore.QTimer.singleShot(
            50,
            lambda: self.inspector.on_action_names(self.live_driver.get_action_names()),
        )
        # Initialize start override ranges based on grid size
        QtCore.QTimer.singleShot(
            60, lambda: self.inspector.set_grid_size(*self.live_driver.get_grid_size())
        )
        QtCore.QTimer.singleShot(
            70,
            lambda: self.inspector.set_observation_pipeline_from_env(
                getattr(self.live_driver, "_env", None)
            ),
        )
        # Enforce strict provider-only mode unconditionally
        QtCore.QTimer.singleShot(
            80, lambda: self.inspector.set_strict_provider_only(True)
        )
        # Apply inspector display prefs
        QtCore.QTimer.singleShot(90, lambda: self._apply_inspector_prefs())
        # Apply theme
        QtCore.QTimer.singleShot(100, lambda: self._apply_theme(self.prefs.theme))

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802
        try:
            from PySide6 import QtCore as _QtCore

            settings = _QtCore.QSettings(QSETTINGS_ORG, QSETTINGS_APP)
            settings.setValue("geometry", self.saveGeometry())
            settings.setValue("windowState", self.saveState())
        except Exception:
            pass
        super().closeEvent(event)

    def _restore_ui_state(self) -> None:
        try:
            from PySide6 import QtCore as _QtCore

            settings = _QtCore.QSettings(QSETTINGS_ORG, QSETTINGS_APP)
            geom = settings.value("geometry")
            if geom is None:
                legacy = _QtCore.QSettings(LEGACY_QSETTINGS_ORG, QSETTINGS_APP)
                geom = legacy.value("geometry")
            if geom is not None:
                self.restoreGeometry(geom)
            st = settings.value("windowState")
            if st is None:
                legacy = _QtCore.QSettings(LEGACY_QSETTINGS_ORG, QSETTINGS_APP)
                st = legacy.value("windowState")
            if st is not None:
                self.restoreState(st)
        except Exception:
            pass

    def _connect_driver(self, driver: object) -> None:
        try:
            driver.frame_ready.connect(self.frame_view.update_frame)  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            driver.step_done.connect(self._on_step_event)  # type: ignore[attr-defined]
            driver.step_done.connect(self.inspector.on_step_event)  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            driver.episode_finished.connect(self._on_episode_finished)  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            driver.action_space_changed.connect(self.inspector.on_action_names)  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            driver.policy_changed.connect(self.inspector.on_policy_changed)  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            driver.provider_mux_changed.connect(self.inspector.on_mux_changed)  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            driver.run_meta_changed.connect(self._on_run_meta)  # type: ignore[attr-defined]
        except Exception:
            pass
        if hasattr(driver, "error_occurred"):
            try:
                driver.error_occurred.connect(self._on_driver_error)  # type: ignore[attr-defined]
            except Exception:
                pass
        if hasattr(driver, "timeline_changed"):
            try:
                driver.timeline_changed.connect(self._on_timeline_changed)  # type: ignore[attr-defined]
            except Exception:
                pass

    def _disconnect_driver(self, driver: object) -> None:
        try:
            driver.frame_ready.disconnect(self.frame_view.update_frame)  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            driver.step_done.disconnect(self._on_step_event)  # type: ignore[attr-defined]
            driver.step_done.disconnect(self.inspector.on_step_event)  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            driver.episode_finished.disconnect(self._on_episode_finished)  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            driver.action_space_changed.disconnect(self.inspector.on_action_names)  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            driver.policy_changed.disconnect(self.inspector.on_policy_changed)  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            driver.provider_mux_changed.disconnect(self.inspector.on_mux_changed)  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            driver.run_meta_changed.disconnect(self._on_run_meta)  # type: ignore[attr-defined]
        except Exception:
            pass
        if hasattr(driver, "error_occurred"):
            try:
                driver.error_occurred.disconnect(self._on_driver_error)  # type: ignore[attr-defined]
            except Exception:
                pass
        if hasattr(driver, "timeline_changed"):
            try:
                driver.timeline_changed.disconnect(self._on_timeline_changed)  # type: ignore[attr-defined]
            except Exception:
                pass

    def _switch_driver(self, driver: object) -> None:
        if driver is self._active_driver:
            return
        if self._active_driver is not None:
            try:
                if hasattr(self._active_driver, "pause"):
                    self._active_driver.pause()  # type: ignore[attr-defined]
            except Exception:
                pass
            self._disconnect_driver(self._active_driver)
        self._active_driver = driver
        self._connect_driver(driver)
        self._active_mode = "replay" if driver is self.replay_driver else "live"
        self.controls.set_mode("Replay" if self._active_mode == "replay" else "Live")
        self._total_reward = 0.0
        self._status.setText("t=0  reward_total=0.00  term=False trunc=False")
        if self._active_mode == "live":
            try:
                self.inspector.on_action_names(self.live_driver.get_action_names())
            except Exception:
                pass
        else:
            try:
                if self.replay_driver.is_loaded():
                    run_dir = getattr(self.replay_driver, "_run_dir", None)
                    label = run_dir.name if isinstance(run_dir, Path) else "loaded run"
                    self.controls.set_replay_label(label)
            except Exception:
                pass
        self._refresh_timeline_controls()

    # UI wiring --------------------------------------------------------------
    @QtCore.Slot()
    def _on_start_clicked(self) -> None:
        if self._active_driver is None:
            return
        if self._active_mode == "replay" and not self.replay_driver.is_loaded():
            self.statusBar().showMessage("Load a replay run before starting", 2500)
            return
        interval = int(self.controls.interval_spin.value())
        try:
            self._active_driver.start(interval)  # type: ignore[attr-defined]
        except Exception as exc:
            self.statusBar().showMessage(f"Start failed: {exc}", 3000)

    @QtCore.Slot(int)
    def _on_interval_changed(self, value: int) -> None:
        if self._active_driver is not None:
            try:
                if self._active_driver.is_running():  # type: ignore[attr-defined]
                    self._active_driver.start(int(value))  # type: ignore[attr-defined]
            except Exception:
                pass

    @QtCore.Slot()
    def _on_pause_clicked(self) -> None:
        try:
            if self._active_driver is not None:
                self._active_driver.pause()  # type: ignore[attr-defined]
        except Exception:
            pass

    @QtCore.Slot()
    def _on_step_clicked(self) -> None:
        if self._active_mode == "replay" and not self.replay_driver.is_loaded():
            self.statusBar().showMessage("Load a replay run before stepping", 2500)
            return
        try:
            if self._active_driver is not None:
                self._active_driver.step_once()  # type: ignore[attr-defined]
        except Exception as exc:
            self.statusBar().showMessage(f"Step failed: {exc}", 3000)

    def _setup_shortcuts(self) -> None:
        # Space: toggle start/pause
        space = QtGui.QShortcut(QtGui.QKeySequence("Space"), self)
        space.activated.connect(self._toggle_run)

        # N: step once
        step = QtGui.QShortcut(QtGui.QKeySequence("N"), self)
        step.activated.connect(self._on_step_clicked)

        # R: reset with seed from edit (delegates to existing handler)
        reset = QtGui.QShortcut(QtGui.QKeySequence("R"), self)
        reset.activated.connect(self.controls._emit_reset)  # type: ignore[attr-defined]

    @QtCore.Slot()
    def _toggle_run(self) -> None:
        try:
            if self._active_driver is not None and self._active_driver.is_running():  # type: ignore[attr-defined]
                self._active_driver.pause()  # type: ignore[attr-defined]
            else:
                self._on_start_clicked()
        except Exception:
            self._on_start_clicked()

    def _refresh_timeline_controls(self) -> None:
        try:
            if self._active_mode == "replay" and self.replay_driver.is_loaded():
                self.controls.set_timeline(
                    self.replay_driver.total_steps(),
                    self.replay_driver.current_index(),
                    total_episodes=self.replay_driver.total_episodes(),
                    current_episode=self.replay_driver.current_episode_index(),
                )
            else:
                self.controls.set_timeline(0, 0, total_episodes=0, current_episode=0)
        except Exception:
            pass

    @QtCore.Slot(object)
    def _on_step_event(self, ev) -> None:
        try:
            self._total_reward += float(getattr(ev, "reward", 0.0))
        except Exception:
            pass
        t = getattr(ev, "t", "?")
        term = bool(getattr(ev, "terminated", False))
        trunc = bool(getattr(ev, "truncated", False))
        self._status.setText(
            f"t={t}  reward_total={self._total_reward:.2f}  term={term} trunc={trunc}"
        )

    @QtCore.Slot()
    def _on_episode_finished(self) -> None:
        self.statusBar().showMessage("Episode finished", 3000)
        # keep last metrics displayed; allow restart

    @QtCore.Slot(int)
    def _on_reset_clicked(self, seed: int) -> None:
        self._total_reward = 0.0
        self._status.setText("t=0  reward_total=0.00  term=False trunc=False")
        if self._active_mode == "replay":
            if self.replay_driver.is_loaded():
                self.replay_driver.seek_to(0)
            self._refresh_timeline_controls()
            return
        # Interpret -1 as "reuse last episode seed" (live mode only)
        self.live_driver.reset(None if seed == -1 else seed)

    @QtCore.Slot(int, object)
    def _on_run_meta(self, seed_val: int, start_xy: object) -> None:
        try:
            if isinstance(start_xy, tuple) and len(start_xy) == 2:
                x, y = int(start_xy[0]), int(start_xy[1])
                self._run_meta.setText(f"seed={seed_val} start=({x},{y})")
            else:
                self._run_meta.setText(f"seed={seed_val}")
        except Exception:
            self._run_meta.setText("")

    @QtCore.Slot(str)
    def _on_driver_error(self, message: str) -> None:
        try:
            msg = str(message).strip()
            if not msg:
                msg = "Error"
            self.statusBar().showMessage(msg, 6000)
        except Exception:
            pass

    @QtCore.Slot(str)
    def _on_mode_changed(self, mode: str) -> None:
        if str(mode).lower().startswith("replay"):
            self._switch_driver(self.replay_driver)
            if not self.replay_driver.is_loaded():
                self.statusBar().showMessage(
                    "Replay mode: load artifacts to begin playback", 2500
                )
        else:
            self._switch_driver(self.live_driver)
        self._refresh_timeline_controls()

    @QtCore.Slot()
    def _on_load_replay(self) -> None:
        dlg_dir = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select replay run directory", str(Path.home())
        )
        if dlg_dir:
            self._load_replay_from_path(Path(dlg_dir))

    @QtCore.Slot(int)
    def _on_seek_requested(self, step: int) -> None:
        if self._active_mode != "replay":
            self.statusBar().showMessage("Seek is available in replay mode only", 2500)
            return
        if not self.replay_driver.is_loaded():
            self.statusBar().showMessage("Load a replay run first", 2500)
            return
        self._total_reward = 0.0
        self._status.setText("t=0  reward_total=0.00  term=False trunc=False")
        self.replay_driver.seek_to(int(step))
        self._refresh_timeline_controls()

    @QtCore.Slot(int)
    def _on_episode_seek_requested(self, episode_index: int) -> None:
        if self._active_mode != "replay":
            self.statusBar().showMessage(
                "Episode seek is available in replay mode only", 2500
            )
            return
        if not self.replay_driver.is_loaded():
            self.statusBar().showMessage("Load a replay run first", 2500)
            return
        self._total_reward = 0.0
        self._status.setText("t=0  reward_total=0.00  term=False trunc=False")
        self.replay_driver.seek_to_episode(int(episode_index))
        self._refresh_timeline_controls()

    @QtCore.Slot(int, int)
    def _on_timeline_changed(self, current: int, total: int) -> None:
        try:
            total_eps = 0
            cur_ep = 0
            if self._active_mode == "replay" and self.replay_driver.is_loaded():
                total_eps = self.replay_driver.total_episodes()
                cur_ep = self.replay_driver.current_episode_index()
            self.controls.set_timeline(
                total,
                current,
                total_episodes=total_eps,
                current_episode=cur_ep,
            )
        except Exception:
            pass

    def _load_replay_from_path(self, path: Path) -> None:
        self._total_reward = 0.0
        # Switch early so emitted signals wire to UI; revert on failure
        self._switch_driver(self.replay_driver)
        try:
            self.replay_driver.load_run(path)
        except ReplayLoadError as exc:  # pragma: no cover - UI safety
            self._switch_driver(self.live_driver)
            self.statusBar().showMessage(f"Replay load failed: {exc}", 5000)
            return
        except Exception as exc:  # pragma: no cover - UI safety
            self._switch_driver(self.live_driver)
            self.statusBar().showMessage(f"Replay load error: {exc}", 5000)
            return
        # Switch into replay mode and surface metadata
        try:
            self.controls.mode_combo.setCurrentText("Replay")
        except Exception:
            self.controls.set_mode("Replay")
        self._switch_driver(self.replay_driver)
        run_dir = Path(path)
        self.controls.set_replay_label(run_dir.name)
        self.statusBar().showMessage(f"Loaded replay from {run_dir}", 3000)
        try:
            self.inspector.set_grid_size(*self.replay_driver.get_grid_size())
        except Exception:
            pass
        self._refresh_timeline_controls()

    # Preferences ------------------------------------------------------------
    def _open_preferences(self) -> None:
        dlg = PreferencesDialog(self.prefs, self)
        if dlg.exec() == QtWidgets.QDialog.Accepted:
            self.prefs = dlg.get_prefs()
            self.prefs.to_qsettings()
            try:
                self.prefs.save_json_file()
            except Exception:
                pass
            # Apply display-related prefs
            self._apply_inspector_prefs()
            self._apply_theme(self.prefs.theme)
            try:
                self.controls.interval_spin.setValue(
                    int(self.prefs.default_interval_ms)
                )
            except Exception:
                pass

    def _apply_inspector_prefs(self) -> None:
        try:
            self.inspector.set_show_pipeline(self.prefs.show_pipeline)
            self.inspector.set_show_preview(self.prefs.show_preview)
            self.inspector.set_show_sparkline(self.prefs.show_sparkline)
        except Exception:
            pass

    def _apply_theme(self, theme: str) -> None:
        try:
            if str(theme).lower() == "dark":
                self._enable_dark_palette()
            else:
                QtWidgets.QApplication.setPalette(
                    QtWidgets.QApplication.style().standardPalette()
                )
        except Exception:
            pass

    def _enable_dark_palette(self) -> None:
        pal = QtGui.QPalette()
        pal.setColor(QtGui.QPalette.Window, QtGui.QColor(53, 53, 53))
        pal.setColor(QtGui.QPalette.WindowText, QtCore.Qt.white)
        pal.setColor(QtGui.QPalette.Base, QtGui.QColor(35, 35, 35))
        pal.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(53, 53, 53))
        pal.setColor(QtGui.QPalette.ToolTipBase, QtCore.Qt.white)
        pal.setColor(QtGui.QPalette.ToolTipText, QtCore.Qt.white)
        pal.setColor(QtGui.QPalette.Text, QtCore.Qt.white)
        pal.setColor(QtGui.QPalette.Button, QtGui.QColor(53, 53, 53))
        pal.setColor(QtGui.QPalette.ButtonText, QtCore.Qt.white)
        pal.setColor(QtGui.QPalette.BrightText, QtCore.Qt.red)
        pal.setColor(QtGui.QPalette.Highlight, QtGui.QColor(42, 130, 218))
        pal.setColor(QtGui.QPalette.HighlightedText, QtCore.Qt.black)
        QtWidgets.QApplication.setPalette(pal)

    @QtCore.Slot(int)
    def _on_policy_selected(self, idx: int) -> None:
        if self._active_mode != "live":
            self.statusBar().showMessage("Policy selection is live-mode only", 2000)
            return
        # Map combo index to built-in policies
        label = self.controls.policy_combo.itemText(idx)
        try:
            if label == "Greedy TD (bacterial)":
                from plume_nav_sim.policies import TemporalDerivativePolicy

                policy = TemporalDerivativePolicy(
                    eps=0.0, eps_after_turn=0.0, uniform_random_on_non_increase=True
                )
                self.live_driver.set_policy(policy, seed=self._current_seed_value())
                self.statusBar().showMessage(
                    "Loaded Greedy TD (bacterial) policy", 1500
                )
            elif label == "Deterministic TD":
                from plume_nav_sim.policies import TemporalDerivativeDeterministicPolicy

                policy = TemporalDerivativeDeterministicPolicy()
                self.live_driver.set_policy(policy, seed=self._current_seed_value())
                self.statusBar().showMessage("Loaded Deterministic TD policy", 1500)
            elif label == "Stochastic TD":
                from plume_nav_sim.policies import TemporalDerivativePolicy

                policy = TemporalDerivativePolicy(eps=0.05, eps_after_turn=0.0)
                self.live_driver.set_policy(policy, seed=self._current_seed_value())
                self.statusBar().showMessage("Loaded Stochastic TD policy", 1500)
            elif label == "Random Sampler":
                # Use the driver's default random sampler
                policy = (
                    self.live_driver._make_default_policy()
                )  # noqa: SLF001 - internal is fine for UI wiring
                self.live_driver.set_policy(policy, seed=self._current_seed_value())
                self.statusBar().showMessage("Loaded Random Sampler policy", 1500)
        except Exception as e:  # pragma: no cover - UI safety
            self.statusBar().showMessage(f"Policy load failed: {e}", 3000)

    @QtCore.Slot()
    def _on_custom_policy_load(self) -> None:
        if self._active_mode != "live":
            self.statusBar().showMessage("Custom policies are live-mode only", 2500)
            return
        from plume_nav_sim.compose.policy_loader import load_policy

        spec = self.controls.custom_policy_edit.text().strip()
        if not spec:
            self.statusBar().showMessage("Enter custom policy as module:Attr", 2500)
            return
        try:
            loaded = load_policy(spec)
            self.live_driver.set_policy(loaded.obj, seed=self._current_seed_value())
            self.statusBar().showMessage(f"Loaded custom policy: {loaded.spec}", 2000)
            # Names may change with new policy/env combination
            self.inspector.on_action_names(self.live_driver.get_action_names())
        except Exception as e:  # pragma: no cover - UI safety
            self.statusBar().showMessage(f"Custom policy load failed: {e}", 4000)

    def _current_seed_value(self) -> Optional[int]:
        text = self.controls.seed_edit.text().strip()
        return int(text) if text.isdigit() else self.live_driver.config.seed

    # No inspector-originating control handlers; inspector is read-only


class PreferencesDialog(QtWidgets.QDialog):
    def __init__(
        self, prefs: DebuggerPreferences, parent: Optional[QtWidgets.QWidget] = None
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Preferences")
        self._prefs = DebuggerPreferences(**vars(prefs))
        layout = QtWidgets.QFormLayout(self)
        # Inspector toggles
        self.chk_pipeline = QtWidgets.QCheckBox("Show pipeline")
        self.chk_pipeline.setChecked(self._prefs.show_pipeline)
        self.chk_preview = QtWidgets.QCheckBox("Show observation preview")
        self.chk_preview.setChecked(self._prefs.show_preview)
        self.chk_spark = QtWidgets.QCheckBox("Show sparkline for vectors")
        self.chk_spark.setChecked(self._prefs.show_sparkline)
        # Interval
        self.spin_interval = QtWidgets.QSpinBox()
        self.spin_interval.setRange(1, 5000)
        self.spin_interval.setValue(int(self._prefs.default_interval_ms))
        # Theme
        self.combo_theme = QtWidgets.QComboBox()
        self.combo_theme.addItems(["light", "dark"])
        idx = 0 if str(self._prefs.theme).lower() != "dark" else 1
        self.combo_theme.setCurrentIndex(idx)
        # Buttons
        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)

        layout.addRow(self.chk_pipeline)
        layout.addRow(self.chk_preview)
        layout.addRow(self.chk_spark)
        layout.addRow("Default interval (ms)", self.spin_interval)
        layout.addRow("Theme", self.combo_theme)
        layout.addRow(btns)

    def get_prefs(self) -> DebuggerPreferences:
        self._prefs.show_pipeline = bool(self.chk_pipeline.isChecked())
        self._prefs.show_preview = bool(self.chk_preview.isChecked())
        self._prefs.show_sparkline = bool(self.chk_spark.isChecked())
        self._prefs.default_interval_ms = int(self.spin_interval.value())
        self._prefs.theme = str(self.combo_theme.currentText())
        return self._prefs


class ActionPanelWidget(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__()
        layout = QtWidgets.QGridLayout(self)
        # Policy insight
        self.expected_action_label = QtWidgets.QLabel("expected action: -")
        self.distribution_label = QtWidgets.QLabel("distribution: N/A")
        self.source_label = QtWidgets.QLabel("source: none")
        # Layout
        layout.addWidget(self.expected_action_label, 0, 0)
        layout.addWidget(self.distribution_label, 0, 1)
        layout.addWidget(self.source_label, 1, 0, 1, 2)

        self._action_names: list[str] = []

    @QtCore.Slot(object)
    def on_step_event(self, ev: object) -> None:
        # Deprecated direct update; InspectorWidget now orchestrates via models
        pass

    @QtCore.Slot(list)
    def on_action_names(self, names: list[str]) -> None:
        try:
            self._action_names = list(names)
            self.action_combo.clear()
            if self._action_names:
                self.action_combo.addItems(self._action_names)
        except Exception:
            pass

    def set_grid_size(self, w: int, h: int) -> None:
        pass


class ObservationPanelWidget(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__()
        layout = QtWidgets.QGridLayout(self)
        self.obs_shape = QtWidgets.QLabel("shape: -")
        self.obs_stats = QtWidgets.QLabel("min/mean/max: -/-/-")
        self.pipeline_label = QtWidgets.QLabel("")
        self.preview_label = QtWidgets.QLabel("")
        self.sparkline = _SparklineWidget()
        self._show_pipeline = True
        self._show_preview = True
        self._show_sparkline = True
        layout.addWidget(QtWidgets.QLabel("Policy Observation"), 0, 0)
        layout.addWidget(self.obs_shape, 0, 1)
        layout.addWidget(self.obs_stats, 0, 2)
        layout.addWidget(self.pipeline_label, 1, 0, 1, 3)
        layout.addWidget(self.preview_label, 2, 0, 1, 1)
        layout.addWidget(self.sparkline, 2, 1, 1, 2)

    @QtCore.Slot(object)
    def on_step_event(self, ev: object) -> None:
        # Deprecated direct update; InspectorWidget now orchestrates via models
        pass

    # Display toggles
    def set_show_pipeline(self, flag: bool) -> None:
        self._show_pipeline = bool(flag)
        self.pipeline_label.setVisible(self._show_pipeline)

    def set_show_preview(self, flag: bool) -> None:
        self._show_preview = bool(flag)
        self.preview_label.setVisible(self._show_preview)

    def set_show_sparkline(self, flag: bool) -> None:
        self._show_sparkline = bool(flag)
        self.sparkline.setVisible(self._show_sparkline)


class _SparklineWidget(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._values: Optional[np.ndarray] = None
        self.setMinimumSize(160, 48)

    def set_values(self, values: Optional[np.ndarray]) -> None:
        if values is None:
            self._values = None
        else:
            try:
                arr = np.asarray(values, dtype=float).ravel()
                self._values = arr
            except Exception:
                self._values = None
        self.update()

    def sizeHint(self) -> QtCore.QSize:  # noqa: N802
        return QtCore.QSize(200, 60)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        painter = QtGui.QPainter(self)
        rect = self.rect()
        painter.fillRect(rect, self.palette().window())

        if self._values is None or self._values.size == 0:
            painter.end()
            return

        pts = normalize_series_to_polyline(
            self._values, rect.width(), rect.height(), pad=3
        )
        if not pts:
            painter.end()
            return

        pen = QtGui.QPen(self.palette().highlight().color(), 2)
        painter.setPen(pen)
        path = QtGui.QPainterPath()
        path.moveTo(pts[0][0], pts[0][1])
        for x, y in pts[1:]:
            path.lineTo(x, y)
        painter.drawPath(path)
        painter.end()


class InspectorWidget(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__()
        vbox = QtWidgets.QVBoxLayout(self)
        # Strict-mode banner (hidden by default)
        self.info_label = QtWidgets.QLabel("")
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet("color: #a60; padding: 4px;")
        self.info_label.setTextFormat(QtCore.Qt.RichText)
        self.info_label.setOpenExternalLinks(True)
        self.info_label.setVisible(False)
        self.tabs = QtWidgets.QTabWidget()
        self.action_panel = ActionPanelWidget()
        self.obs_panel = ObservationPanelWidget()
        self.tabs.addTab(self.action_panel, "Action")
        self.tabs.addTab(self.obs_panel, "Observation")
        vbox.addWidget(self.info_label)
        vbox.addWidget(self.tabs)

        # No control signals: Inspector is information-only

        # Models for TDD-friendly logic
        self._obs_model = ObservationPanelModel()
        self._act_model = ActionPanelModel()
        self._policy_for_probe = None
        self._mux: Optional[ProviderMux] = None
        self._pipeline_text: str = ""
        self._strict_provider_only: bool = True

    @QtCore.Slot(object)
    def on_step_event(self, ev: object) -> None:
        try:
            obs = getattr(ev, "obs", None)
            action = getattr(ev, "action", None)

            # Update models
            if isinstance(obs, np.ndarray):
                self._obs_model.update(obs)
            else:
                self._obs_model.summary = None

            self._act_model.update_event(
                action if isinstance(action, (int, np.integer)) else None
            )

            # Probe distribution (provider-only)
            if isinstance(obs, np.ndarray):
                dist = None
                if self._mux is not None:
                    try:
                        dist = self._mux.get_policy_distribution(obs)
                    except Exception:
                        dist = None
                if dist is not None:
                    self._act_model.state.distribution = [float(x) for x in dist]
                    self._act_model.state.distribution_source = "provider"
                else:
                    self._act_model.state.distribution = None
                    self._act_model.state.distribution_source = None

            # Update UI: observation
            if self._obs_model.summary is not None:
                s = self._obs_model.summary
                self.obs_panel.obs_shape.setText(
                    f"shape: {'x'.join(str(d) for d in s.shape)}"
                )
                self.obs_panel.obs_stats.setText(
                    f"min/mean/max: {s.vmin:.3f}/{s.vmean:.3f}/{s.vmax:.3f}"
                )
                # Provider observation metadata (preferred in preview area)
                meta_shown = False
                try:
                    if self._mux is not None and hasattr(
                        self._mux, "describe_observation"
                    ):
                        info = self._mux.describe_observation(obs)  # type: ignore[attr-defined]
                        if info is not None:
                            kind = getattr(info, "kind", None)
                            label = getattr(info, "label", None)
                            parts = []
                            if isinstance(kind, str) and kind:
                                parts.append(f"kind={kind}")
                            if isinstance(label, str) and label:
                                parts.append(f"label={label}")
                            if parts:
                                self.obs_panel.preview_label.setText("; ".join(parts))
                                meta_shown = True
                except Exception:
                    meta_shown = False

                # Small preview for tiny vectors (only if no provider meta)
                try:
                    if not meta_shown and isinstance(obs, np.ndarray) and obs.size <= 8:
                        vals = ", ".join(f"{float(v):.3f}" for v in obs.ravel())
                        self.obs_panel.preview_label.setText(f"values: [{vals}]")
                    else:
                        self.obs_panel.preview_label.setText("")
                    # Sparkline for vector observations
                    if isinstance(obs, np.ndarray) and obs.size > 1:
                        self.obs_panel.sparkline.set_values(obs.astype(float))
                    else:
                        self.obs_panel.sparkline.set_values(None)
                except Exception:
                    self.obs_panel.preview_label.setText("")
                    self.obs_panel.sparkline.set_values(None)
            else:
                self.obs_panel.obs_shape.setText("shape: -")
                self.obs_panel.obs_stats.setText("min/mean/max: -/-/-")
                self.obs_panel.preview_label.setText("")
                self.obs_panel.sparkline.set_values(None)
            # Pipeline label if known
            if self._pipeline_text:
                if hasattr(self.obs_panel, "pipeline_label"):
                    self.obs_panel.pipeline_label.setText(self._pipeline_text)

            # Update UI: action
            self.action_panel.expected_action_label.setText(
                f"expected action: {self._act_model.state.action_label}"
            )
            if self._act_model.state.distribution is not None:
                src = self._act_model.state.distribution_source or "probs"
                p = self._act_model.state.distribution
                preview = ", ".join(f"{v:.2f}" for v in p[:6])
                self.action_panel.distribution_label.setText(
                    f"distribution ({src}): [{preview}]"
                )
            else:
                self.action_panel.distribution_label.setText("distribution: N/A")
        except Exception:
            pass

    @QtCore.Slot(list)
    def on_action_names(self, names: list[str]) -> None:
        self._act_model.set_action_names(names)
        self.action_panel.on_action_names(names)

    def set_grid_size(self, w: int, h: int) -> None:
        self.action_panel.set_grid_size(w, h)

    @QtCore.Slot(object)
    def on_policy_changed(self, policy: object) -> None:
        self._policy_for_probe = policy

    def set_observation_pipeline_from_env(self, env: object) -> None:
        # Provider-only UI: pipeline derived only via ProviderMux
        self._pipeline_text = ""

    @QtCore.Slot(object)
    def on_mux_changed(self, mux: object) -> None:
        try:
            self._mux = mux  # type: ignore[assignment]
            # Update pipeline with provider if available
            if hasattr(self._mux, "get_pipeline"):
                try:
                    names = self._mux.get_pipeline()  # type: ignore[attr-defined]
                    self._pipeline_text = f"Pipeline: {format_pipeline(list(names))}"
                except Exception:
                    pass
            # Update source label to indicate provider
            try:
                self.action_panel.source_label.setText("source: provider")
            except Exception:
                pass
            self._update_strict_banner()
        except Exception:
            self._mux = None
            try:
                self.action_panel.source_label.setText("source: none")
            except Exception:
                pass
            self._update_strict_banner()

    def set_strict_provider_only(self, flag: bool) -> None:
        self._strict_provider_only = bool(flag)
        self._update_strict_banner()

    # Inspector display preferences passthrough
    def set_show_pipeline(self, flag: bool) -> None:
        try:
            self.obs_panel.set_show_pipeline(flag)
        except Exception:
            pass

    def set_show_preview(self, flag: bool) -> None:
        try:
            self.obs_panel.set_show_preview(flag)
        except Exception:
            pass

    def set_show_sparkline(self, flag: bool) -> None:
        try:
            self.obs_panel.set_show_sparkline(flag)
        except Exception:
            pass

    def _update_strict_banner(self) -> None:
        try:
            show = bool(self._strict_provider_only) and (self._mux is None)
            if show:
                self.info_label.setText(
                    "Strict mode: no DebuggerProvider detected. Inspector shows limited information. "
                    "Implement a provider (ODC) to enable action labels, distributions, and pipeline details. "
                    '<a href="https://plume-nav-sim.dev/odc" style="color:#06c;">Read ODC docs</a>.'
                )
            self.info_label.setVisible(show)
        except Exception:
            self.info_label.setVisible(False)
